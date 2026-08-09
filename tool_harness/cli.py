#!/usr/bin/env python3
"""isaacli's raw CLI.

Usage:
    isaacli
    isaacli "run git status and tell me what is pending"
    isaacli --workspace /path/to/project

The process runs in the foreground. Closing the terminal takes this Python down;
commands run by Isaac are born in their own process group and with
bwrap --die-with-parent.
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import shlex
from pathlib import Path

try:
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.shortcuts import CompleteStyle, PromptSession
except ImportError:  # pragma: no cover - optional dependency
    PromptSession = None
    CompleteStyle = None
    FormattedText = None
    KeyBindings = None

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import agent
import config
import terminal_ui
import tools
from cli_i18n import set_language, t
from cli_permissions import (
    DESTRUCTIVE_COMMANDS, DESTRUCTIVE_GIT, READ_ONLY_COMMANDS, READ_ONLY_GH,
    READ_ONLY_GIT, _command_parts, _command_rule, _destructive_command,
    _safe_read_command,
)
from cli_presentation import (
    ANSI, APP_VERSION, WORDMARK_ISAAC, _color, _colored_prompt,
    _format_markdown_terminal, _friendly_path, _markdown_inline, _pad_visual,
    _print_welcome, _short_context, _shorten, _terminal_safe_text,
    _uses_color, _visual_width, _welcome_lines,
)
from cli_sessions import (
    CLI_KNOWLEDGE, FEEDBACK_DIR, SESSION_ID_LEGACY, SESSION_ID_UUID,
    SESSIONS_DIR, SessionsMixin, _build_history, _load_session, _new_session_id,
    _now, _resume_command, _valid_session_id,
)
from cli_commands import (
    COMMAND_ALIASES, COMMANDS, SLASH_COMMANDS, CommandsMixin, _CommandCompleter,
    _filter_commands, _install_autocomplete, _score_command,
)
from cli_ollama import (
    OllamaMixin, _close_without_interruption, _install_signals, _ollama_ok,
    _pid_identity, _same_process, _shared_ollama_state,
)
from cli_providers import ProvidersMixin

MAX_PREVIEW_CHARS = 1800
MAX_PREVIEW_LINES = 28

def _announce_rate_limit(seconds, attempt):
    """Show the provider's own wait so a paused turn does not look like a freeze."""
    # \r\033[2K: the progress line is redrawn in place and has no newline of its
    # own, so writing over it is what keeps the notice on a line of its own.
    print("\r\033[2K" + _color(
        t("cli.api.rate_limit_wait", seconds=f"{seconds:.0f}",
          attempt=attempt, total=agent.RATE_LIMIT_RETRIES), "warn"), flush=True)


agent.RATE_LIMIT_NOTICE = _announce_rate_limit


def _exit_code(result):
    m = re.search(r"\(exit code: (-?\d+)\)\s*$", result.strip())
    return int(m.group(1)) if m else None


def _preview(text):
    lines = text.splitlines()
    cut_lines = len(lines) > MAX_PREVIEW_LINES
    if cut_lines:
        lines = lines[:MAX_PREVIEW_LINES]
    out = "\n".join(lines)
    cut_chars = len(out) > MAX_PREVIEW_CHARS
    if cut_chars:
        out = out[:MAX_PREVIEW_CHARS].rstrip()
    return out, cut_lines or cut_chars


_PROMPT_SESSION = None


def _prompt_session():
    global _PROMPT_SESSION
    if _PROMPT_SESSION is None:
        keys = KeyBindings()

        @keys.add("s-tab")
        def _toggle_permissions(event):
            event.app.exit(result="/mode")

        @keys.add("escape", "enter")
        def _line_break(event):
            event.current_buffer.insert_text("\n")

        _PROMPT_SESSION = PromptSession(
            completer=_CommandCompleter(),
            complete_while_typing=False,
            complete_style=CompleteStyle.COLUMN,
            reserve_space_for_menu=12,
            key_bindings=keys,
            enable_history_search=True,
        )

        def _update_palette(buffer):
            text = buffer.document.text_before_cursor
            if text.startswith("/") and not any(c.isspace() for c in text):
                buffer.start_completion(select_first=False)
            elif buffer.complete_state is not None:
                buffer.cancel_completion()

        _PROMPT_SESSION.default_buffer.on_text_changed += _update_palette
    return _PROMPT_SESSION


def _read_input():
    if PromptSession is not None and terminal_ui.interactive():
        prompt = FormattedText([("ansibrightcyan bold", "❯ ")])
        return _prompt_session().prompt(prompt)
    return input(_colored_prompt("❯ "))


class IsaacCLI(SessionsMixin, CommandsMixin, OllamaMixin, ProvidersMixin):
    def __init__(self, model, workspace, max_steps, autostart_ollama=True,
                 thinking=None, num_ctx=None, config_file=None, provider=None):
        self.model = model
        self.thinking = thinking
        self.num_ctx = num_ctx
        self.config_file = config_file
        try:
            language = config.load(config_file).get("language")
        except ValueError:
            language = None
        self.tr = set_language(language)
        self.provider = provider or {"provider": "ollama"}
        self.workspace = Path(workspace).expanduser().resolve()
        self.max_steps = max_steps
        self.autostart_ollama = autostart_ollama
        self.ollama_proc = None
        self._ollama_registered = False
        self._runtime_pid = os.getpid()
        self._runtime_start = _pid_identity(self._runtime_pid)
        self.history = []
        self.session_id = _new_session_id()
        self.session_path = SESSIONS_DIR / f"{self.session_id}.jsonl"
        self.feedback_path = FEEDBACK_DIR / f"{self.session_id}.jsonl"
        self.turns = 0
        self.failures = 0
        self.commands = []
        self.total_usage = {"prompt_eval_count": 0, "eval_count": 0,
                            "total_duration": 0, "eval_duration": 0}
        self.last_answer = ""
        self.ratings = 0
        self.permission_mode = "safe"
        self._working_visible = False
        self._assistant_label_pending = True
        self._generation_start = None
        self._turn_start = None
        self._generation_chunks = 0
        self._generation_status_at = 0.0
        self._token_buffer = []
        self._first_token_at = None
        self._stream_started = False
        self._output_block = False
        self.resume_transcript = []
        self.set_workspace(self.workspace, reset=True)
        self._log("meta", event="start", pid=os.getpid(), model=self.model,
                  workspace=str(self.workspace))

    def set_workspace(self, path, reset=False):
        new = Path(path).expanduser().resolve()
        if not new.exists():
            raise FileNotFoundError(t("cli.workspace.missing", path=new))
        if not new.is_dir():
            raise NotADirectoryError(t("cli.workspace.not_dir", path=new))
        self.workspace = new
        tools.SANDBOX_ROOT = new
        if reset or not self.history:
            self.history = _build_history(new)
        else:
            # Model-facing, so it stays English regardless of the interface language.
            self.history.append({
                "role": "system",
                "content": f"The working directory is now: {new}",
            })
            self._log("meta", event="workspace", workspace=str(new))

    def _show_working(self):
        self._assistant_label_pending = True
        self._generation_start = time.monotonic()
        self._generation_chunks = 0
        self._generation_status_at = 0.0
        self._token_buffer = []
        self._first_token_at = None
        self._stream_started = False
        print()
        self._output_block = False
        print(_color(t("cli.working.waiting"), "dim"), end="", flush=True)
        self._working_visible = True

    def _flush_tokens(self):
        if not self._token_buffer:
            return False
        text = "".join(self._token_buffer)
        self._token_buffer = []
        if self._working_visible:
            print("\r\033[2K", end="", flush=True)
            self._working_visible = False
        if self._assistant_label_pending:
            print(_color("isaac:", "assistant"), end=" ", flush=True)
            self._assistant_label_pending = False
        print(_format_markdown_terminal(text), end="", flush=True)
        self._stream_started = True
        self._output_block = True
        return True

    def _clear_working(self):
        if self._flush_tokens():
            return
        if self._working_visible:
            print("\r\033[2K", end="", flush=True)
            self._working_visible = False

    def _print_rate(self, elapsed):
        rate = max(self._generation_chunks - 1, 1) / elapsed
        print(
            "\r\033[2K" + _color(t("cli.working.rate", rate=f"{rate:.1f}"), "dim"),
            end="", flush=True,
        )

    def _token(self, chunk):
        if not chunk:
            return
        now = time.monotonic()
        if self._first_token_at is None:
            self._first_token_at = now
        self._token_buffer.append(chunk)
        self._generation_chunks += 1
        elapsed = now - self._first_token_at
        if elapsed >= 0.05 and now - self._generation_status_at >= 0.05:
            self._print_rate(elapsed)
            self._generation_status_at = now
        # We keep the whole step buffered so we do not break Markdown markers
        # (for example ** and ```) that can arrive split across several tokens.

    def _thinking_token(self, chunk):
        """Count the reasoning stream without revealing its content in the terminal."""
        if not chunk or not self._working_visible:
            return
        now = time.monotonic()
        if self._first_token_at is None:
            self._first_token_at = now
        self._generation_chunks += 1
        elapsed = now - self._first_token_at
        if elapsed >= 0.05 and now - self._generation_status_at >= 0.05:
            self._print_rate(elapsed)
            self._generation_status_at = now

    def _tool_before(self, name, args):
        self._clear_working()
        if self._output_block:
            print()
        try:
            data = json.loads(args) if isinstance(args, str) else (args or {})
        except json.JSONDecodeError:
            data = {}
        if name == "run_command":
            cmd = data.get("cmd", args)
            print(_color(f"$ {cmd}", "tool"), flush=True)
            self._output_block = True
            self._log("tool_start", name=name, cmd=cmd)
            return self._approve_and_run(cmd)
        summary = json.dumps(data, ensure_ascii=False) if data else str(args)
        print(_color(f"[{name}] → {summary[:180]}", "tool"), flush=True)
        self._output_block = True
        self._log("tool_start", name=name, args=data or args)
        return None

    def _approve_and_run(self, cmd):
        """Apply the human policy before handing the command to bwrap."""
        import execution

        rule = _command_rule(cmd)
        # Validate before offering a choice: asking about a command that stays
        # refused whatever the answer would be theatre. After the network fix and
        # the removal of the post-approval vetoes, what is left here is what
        # approval genuinely cannot change: shell operators, which have no shell
        # to interpret them, and unreadable quoting.
        try:
            execution.review(cmd, authorized=True)
        except execution.Denied as e:
            return f"$ {cmd}\nDENIED: {e}\n(exit code: 126)"
        try:
            data = config.load(self.config_file)
        except ValueError:
            data = config.empty_config()
        saved = rule and rule in config.permission_rules(data, self.workspace)
        automatic = self.permission_mode == "safe" and _safe_read_command(cmd)
        if saved or automatic:
            return execution.run_command(cmd, authorized=saved)

        print(_color(t("cli.permission.required"), "warn"))
        print(t("cli.permission.scope_note"))
        if _destructive_command(cmd):
            print(_color(t("cli.permission.dangerous"), "bad"))
        try:
            index = terminal_ui.select_inline(
                [
                    t("cli.permission.once"),
                    t("cli.permission.always_workspace", rule=rule),
                    t("cli.permission.always_global", rule=rule),
                    t("cli.permission.deny"),
                ],
                shortcuts={"w": 1, "g": 2, "n": 3}, input_fn=input, initial=0,
                prompt=t("cli.permission.prompt"),
                chosen_label=t("cli.permission.chosen", option="{option}"),
            )
        except (EOFError, KeyboardInterrupt):
            index = 3
            print()
        if index == 3:
            self._log("permission", cmd=cmd, rule=rule, decision="denied")
            return (f"$ {cmd}\nDENIED BY USER: the command was not authorized.\n"
                    "(exit code: 126)")
        if index in (1, 2) and rule:
            config.add_permission(
                data, rule, workspace=self.workspace if index == 1 else None,
            )
            config.save(data, self.config_file)
        decision = {0: "once", 1: "workspace", 2: "global"}[index]
        self._log("permission", cmd=cmd, rule=rule, decision=decision)
        return execution.run_command(cmd, authorized=True)

    def _tool_after(self, name, args, result, _via):
        if name == "run_command":
            try:
                data = json.loads(args) if isinstance(args, str) else (args or {})
            except json.JSONDecodeError:
                data = {}
            cmd = data.get("cmd", args)
            code = _exit_code(result)
            item = {
                "id": len(self.commands) + 1,
                "cmd": cmd,
                "code": code,
                "result": result,
                "denied": "DENIED BY USER:" in result,
            }
            self.commands.append(item)
            if code is not None and code != 0 and not item["denied"]:
                self.failures += 1
            if item["denied"]:
                status, color = t("cli.command.denied"), "warn"
            elif code == 0:
                status, color = t("cli.command.ok"), "tool"
            else:
                status = (t("cli.command.failed", code=code) if code is not None
                          else t("cli.command.no_code"))
                color = "bad"
            print(_color(t("cli.command.status", id=item["id"], status=status), color),
                  flush=True)
            text, truncated = _preview(result)
            print(text, flush=True)
            if truncated:
                print(_color(t("cli.command.truncated", id=item["id"]), "dim"), flush=True)
            self._log("tool_result", name=name, cmd=cmd, code=code, result=result)
        else:
            text, truncated = _preview(result)
            print(_color(f"[{name}] ← {text}", "tool"), flush=True)
            if truncated:
                print(_color(t("cli.command.truncated_log"), "dim"), flush=True)
            self._log("tool_result", name=name, result=result)
        self._output_block = True
        self._assistant_label_pending = True

    def ask(self, request):
        if not self.ensure_ollama(warn=True):
            if self.provider.get("provider") == "ollama":
                print(t("cli.error.ollama_unavailable"))
                error = "ollama_unavailable"
            else:
                print(t("cli.error.api_unavailable"))
                error = "api_unavailable"
            self._log("error", error=error)
            return 1
        self._log("user", content=request)
        commands_before = len(self.commands)
        self._turn_start = time.monotonic()
        try:
            with terminal_ui.busy_input():
                r = agent.run(
                    request,
                    self.model,
                    max_steps=self.max_steps,
                    verbose=False,
                    on_token=self._token,
                    on_thinking=self._thinking_token,
                    on_tool_before=self._tool_before,
                    on_tool=self._tool_after,
                    on_working=self._show_working,
                    history=self.history,
                    thinking=self.thinking,
                    num_ctx=self.num_ctx,
                    provider=self.provider,
                )
        except RuntimeError as e:
            self._clear_working()
            print(t("cli.error.generic", error=e))
            self._log("error", error=str(e))
            return 1
        except urllib.error.URLError as e:
            self._clear_working()
            if self.provider.get("provider") != "ollama":
                print(t("cli.error.api_no_response", error=e))
            elif self.ensure_ollama(warn=True):
                print("\n" + t("cli.error.ollama_started"))
            else:
                print("\n" + t("cli.error.ollama_no_response", error=e))
            self._log("error", error=str(e))
            return 1
        except KeyboardInterrupt:
            self._clear_working()
            print("\n" + t("cli.error.interrupted"))
            self._log("error", error="interrupted")
            return 130

        self._clear_working()
        final = (r or {}).get("final") or ""
        calls = (r or {}).get("calls") or []
        empty_answer = not final and not calls
        self.last_answer = final
        usage = (r or {}).get("usage") or {}
        for key in self.total_usage:
            self.total_usage[key] += int(usage.get(key) or 0)
        self.turns += 1
        if final and not final.endswith("\n"):
            print()
        if (r or {}).get("thinking_adjusted") and self.thinking not in (None, False):
            self.thinking = None
            persisted = self._persist_adjusted_thinking()
            key = ("thinking.rejected.persisted" if persisted
                   else "thinking.rejected.session_only")
            print(_color(t(key), "warn"))
        if empty_answer:
            print(_color(t("cli.error.empty_answer"), "bad"))
        eval_count = int(usage.get("eval_count") or 0)
        eval_duration = int(usage.get("eval_duration") or 0) / 1_000_000_000
        measured_time = eval_duration or max(
            time.monotonic() - (self._turn_start or time.monotonic()), 0.001,
        )
        if eval_count:
            approx = "" if eval_duration else "≈ "
            print()
            print(_color(t("cli.generation.rate", approx=approx,
                           rate=f"{eval_count / measured_time:.1f}",
                           count=eval_count), "dim"))
        asked_for_mutation = bool(re.search(
            r"\b(apag(?:ue|ar)|delet(?:e|ar)|remov(?:a|er)|exclu(?:a|ir)|"
            r"cri(?:e|ar)|edit(?:e|ar)|alter(?:e|ar)|modifi(?:que|car)|"
            r"delete|remove|create|edit|modify)\b", request, re.IGNORECASE,
        ))
        if asked_for_mutation and not calls:
            print(_color(t("cli.note.no_mutation"), "warn"))
        new_commands = self.commands[commands_before:]
        if (new_commands and not new_commands[-1].get("denied")
                and new_commands[-1].get("code") not in (None, 0)):
            print(_color(t("cli.note.command_failed",
                           code=new_commands[-1]["code"]), "warn"))
        self._log("assistant_final", content=final, usage=usage,
                  calls=len(calls))
        self.feedback_reminder(bool(new_commands))
        print()
        return 1 if empty_answer else 0

    def repl(self):
        # No alternate buffer: the conversation belongs to the terminal's own
        # scrollback, so the wheel scrolls it from start to end. In the
        # alternate buffer there is no scrollback and terminals translate the
        # wheel into ↑/↓, which stole the prompt's message history.
        code = self._repl_screen()
        print()
        print(_color(t("cli.resume.hint"), "dim"))
        print(_resume_command(self.session_id))
        print()
        return code

    def _initialize_repl(self):
        _install_autocomplete()
        # The session opens on a screen of its own. Scrolling up from here has to
        # reach the first message of the conversation, not the shell that ran the
        # launcher.
        terminal_ui.clear()
        _print_welcome(self.model, self._engine_label(), self.workspace)
        print()

        if self.resume_transcript:
            limit = 20
            items = self.resume_transcript[-limit:]
            omitted = len(self.resume_transcript) - len(items)
            print(_color(t("cli.history.resumed"), "dim"))
            if omitted:
                print(_color(t("cli.history.resumed_omitted", count=omitted), "dim"))
            for role, content in items:
                if role in ("user", "assistant"):
                    text = content if len(content) <= 2000 else content[:2000] + "…"
                    label = "❯" if role == "user" else "isaac:"
                    color = "prompt" if role == "user" else "assistant"
                    visible = (_format_markdown_terminal(text)
                               if role == "assistant" else _terminal_safe_text(text))
                    print("\n" + _color(label, color), visible)
                elif role == "tool_start":
                    name = content.get("name") or "unknown"
                    if name == "run_command":
                        cmd = content.get("cmd") or (content.get("args") or {}).get("cmd", "")
                        print("\n" + _color(f"$ {cmd}", "tool"))
                    else:
                        args = json.dumps(content.get("args") or {}, ensure_ascii=False)
                        print("\n" + _color(f"[{name}] → {args[:500]}", "tool"))
                elif role == "permission":
                    decisions = {
                        "once": t("cli.permission.decision.once"),
                        "workspace": t("cli.permission.decision.workspace"),
                        "global": t("cli.permission.decision.global"),
                        "denied": t("cli.permission.decision.denied"),
                        # values recorded before the rename
                        "uma_vez": t("cli.permission.decision.once"),
                        "recusado": t("cli.permission.decision.denied"),
                    }
                    decision = decisions.get(content.get("decision"),
                                             content.get("decision")
                                             or t("cli.history.unknown"))
                    print(_color(t("cli.permission.label", decision=decision), "dim"))
                elif role == "tool_result":
                    name = content.get("name") or "unknown"
                    result = content.get("result") or ""
                    text = result if len(result) <= 2000 else result[:2000] + "…"
                    if name == "run_command":
                        code = content.get("code")
                        status = (t("cli.command.ok") if code == 0
                                  else t("cli.command.exit_code", code=code))
                        print(_color(t("cli.history.command_status", status=status), "tool"))
                        print(text)
                    else:
                        print(_color(f"[{name}] ← {text}", "tool"))
            print("\n" + _color(t("cli.history.resumed_end"), "dim") + "\n")

    def _repl_screen(self):
        # From before the Ollama autostart until the final draw, stdin stays with
        # no echo and is flushed when the first prompt opens. That closes the race
        # with keys typed while the application is still loading.
        with terminal_ui.busy_input():
            self._initialize_repl()
        while True:
            try:
                text = _read_input().strip()
            except EOFError:
                print()
                self._log("meta", event="eof")
                return 0
            except KeyboardInterrupt:
                print()
                self._log("meta", event="ctrl_c_exit")
                return 130
            if not text:
                continue
            try:
                if self.internal_command(text):
                    continue
            except EOFError:
                self._log("meta", event="exit")
                return 0
            self.ask(text)


def main(argv=None):
    _install_signals()
    arguments = list(sys.argv[1:] if argv is None else argv)
    setup_requested = bool(arguments and arguments[0] == "setup")
    if setup_requested:
        if len(arguments) > 1:
            print(t("cli.setup.usage"))
            return 2
        arguments = []

    ap = argparse.ArgumentParser(
        prog="isaacli",
        epilog=t("cli.args.epilog"),
    )
    ap.add_argument("--version", action="version", version=f"Isaac CLI v{APP_VERSION}")
    ap.add_argument("request", nargs="*", help=t("cli.args.request"))
    ap.add_argument("--model", dest="model", default=None)
    ap.add_argument("--resume", metavar="ID", help=t("cli.args.resume"))
    ap.add_argument("--workspace", "--dir", default=os.getcwd())
    ap.add_argument("--max-steps", type=int, default=12, help=argparse.SUPPRESS)
    args = ap.parse_args(arguments)

    resumed = None
    if args.resume:
        if args.request:
            print(t("cli.resume.no_request"))
            return 2
        try:
            resumed = _load_session(args.resume)
        except ValueError as e:
            print(t("cli.error.generic", error=e))
            return 2

    try:
        config_data = config.load()
    except ValueError as e:
        print(t("cli.config.warning", error=e))
        config_data = config.empty_config()
    set_language(config_data.get("language"))
    _profile_name, default_profile = config.profile(config_data)
    needs_setup = (
        setup_requested
        or (
            args.model is None
            and not os.environ.get("ISAACLI_MODEL")
            and default_profile is None
            and resumed is None
            and sys.stdin.isatty()
        )
    )
    if needs_setup:
        import setup_ollama
        code = setup_ollama.run_setup()
        if code == 0:
            config_data = config.load()
            set_language(config_data.get("language"))
            _profile_name, default_profile = config.profile(config_data)
        elif code == 130:
            return 130
        elif setup_requested:
            # Never hide a failure/cancellation by opening another model.
            return code
        elif default_profile is None:
            # Without a profile there is no safe engine to open; the message
            # below asks for another try instead of picking a model for the user.
            config_data = config.empty_config()
            default_profile = None
    model = (
        args.model
        or os.environ.get("ISAACLI_MODEL")
        or (resumed or {}).get("model")
        or (default_profile or {}).get("model")
    )
    if not model:
        print(t("cli.model.none_configured"))
        return 2
    if (args.model is None and not os.environ.get("ISAACLI_MODEL")
            and resumed is None and default_profile
            and default_profile.get("model") == model):
        model_profile = default_profile
    else:
        _model_name, model_profile = config.profile_for_model(config_data, model)
    thinking = (model_profile or {}).get("thinking")

    workspace = resumed["workspace"] if resumed else args.workspace
    cli = IsaacCLI(
        model, workspace, args.max_steps, thinking=thinking,
        num_ctx=(model_profile or {}).get("num_ctx"),
    )
    cli.provider = cli._provider_from_profile(model_profile)
    if resumed:
        cli.history = resumed["history"]
        cli.resume_transcript = resumed["transcript"]
        cli._log("meta", event="resume", source=resumed["id"],
                 source_log=str(resumed["path"]))
    try:
        if args.request:
            return cli.ask(" ".join(args.request))
        return cli.repl()
    finally:
        _close_without_interruption(cli)


if __name__ == "__main__":
    sys.exit(main())
