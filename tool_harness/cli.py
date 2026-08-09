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
import urllib.request
import uuid
import shlex
from contextlib import contextmanager
from pathlib import Path

try:
    import readline
except ImportError:  # pragma: no cover - Windows/minimal environment
    readline = None

try:
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.shortcuts import CompleteStyle, PromptSession
except ImportError:  # pragma: no cover - optional dependency
    PromptSession = None
    Completer = object
    Completion = None
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
from cli_presentation import (
    ANSI, APP_VERSION, WORDMARK_ISAAC, _color, _colored_prompt,
    _format_markdown_terminal, _friendly_path, _markdown_inline, _pad_visual,
    _print_welcome, _short_context, _shorten, _terminal_safe_text,
    _uses_color, _visual_width, _welcome_lines,
)

SESSIONS_DIR = HERE / "cli_sessions"
FEEDBACK_DIR = HERE / "feedback"

# The catalogue key for each slash command's description. The text itself lives
# in locales/*.json so the palette and /help speak the chosen language.
COMMANDS = (
    ("/help", "cli.cmd.help"),
    ("/setup", "cli.cmd.setup"),
    ("/status", "cli.cmd.status"),
    ("/tools", "cli.cmd.tools"),
    ("/sessions", "cli.cmd.sessions"),
    ("/history", "cli.cmd.history"),
    ("/show", "cli.cmd.show"),
    ("/log", "cli.cmd.log"),
    ("/feedback", "cli.cmd.feedback"),
    ("/good", "cli.cmd.good"),
    ("/bad", "cli.cmd.bad"),
    ("/score", "cli.cmd.score"),
    ("/workspace", "cli.cmd.workspace"),
    ("/model", "cli.cmd.model"),
    ("/permissions", "cli.cmd.permissions"),
    ("/mode", "cli.cmd.mode"),
    ("/language", "cli.cmd.language"),
    ("/clear", "cli.cmd.clear"),
    ("/new", "cli.cmd.new"),
    ("/exit", "cli.cmd.exit"),
)
SLASH_COMMANDS = [command for command, _key in COMMANDS]

# Portuguese names kept as hidden aliases: they were the original spelling and
# breaking them would break muscle memory for no gain.
COMMAND_ALIASES = {"/bom": "/good", "/ruim": "/bad", "/nota": "/score"}

MAX_PREVIEW_CHARS = 1800
MAX_PREVIEW_LINES = 28
SESSION_ID_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
SESSION_ID_LEGACY = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{6}-[a-f0-9]{6}"
)

READ_ONLY_COMMANDS = {"ls", "cat", "head", "tail", "wc", "grep", "find"}
READ_ONLY_GIT = {"status", "diff", "log", "show"}
READ_ONLY_GH = {
    ("issue", "view"), ("pr", "view"), ("repo", "view"),
    ("release", "view"), ("run", "view"), ("auth", "status"),
    ("search", "issues"), ("search", "prs"), ("search", "repos"),
    ("search", "commits"),
}

# Commands whose effect is destructive or hard to undo. The sandbox and the
# approval prompt already gate them; naming them out loud is what keeps approval
# from becoming a reflex.
DESTRUCTIVE_COMMANDS = {"rm", "rmdir", "mv", "dd", "truncate", "shred", "chmod", "chown"}
DESTRUCTIVE_GIT = {"push", "reset", "clean", "checkout", "restore", "rebase", "revert"}


def _runtime_ollama_dir():
    base = os.environ.get("ISAACLI_RUNTIME_DIR") or os.environ.get("XDG_RUNTIME_DIR")
    if base:
        return Path(base) / "isaacli"
    return Path("/tmp") / f"isaacli-{os.getuid()}"


def _pid_identity(pid):
    """Stable identity so we never signal a PID that has been recycled."""
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().split()[21]
    except (OSError, ValueError, IndexError):
        return None


def _same_process(pid, identity):
    current = _pid_identity(pid)
    return bool(current and identity and current == str(identity))


@contextmanager
def _shared_ollama_state():
    """Serialise autostart/autostop across several Isaac sessions."""
    import fcntl

    folder = _runtime_ollama_dir()
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = folder / "ollama.lock"
    state_path = folder / "ollama.json"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            try:
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
            except (OSError, json.JSONDecodeError):
                state = {}
            yield state
            tmp = state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            os.replace(tmp, state_path)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


CLI_KNOWLEDGE = """You are isaacli running as a local CLI in the user's terminal.

OPERATING CONTEXT:
- Always answer in the same language as the user's latest message. If the user
  writes in Portuguese, answer in Brazilian Portuguese; do not switch to English.
- The current working directory is: {workspace}
- File and terminal tools are confined to that directory.
- To read public web content (pages, documentation, links or HTTP APIs), use
  fetch_url. It is a general web-reading tool, not a GitHub-only workaround.
- For structured, read-only GitHub queries, you may use `gh issue view`,
  `gh pr view`, `gh repo view`, `gh release view`, `gh run view` or `gh search`.
  Prefer fetch_url for public links; use gh when its GitHub-specific structure or
  authenticated access is useful. If gh reports missing/invalid authentication
  for a public link, use fetch_url immediately; do not inspect tokens, environment
  variables or credential files, and do not clone a repository just to read it.
- Before asking the user to clarify a local file, directory or project target,
  try to resolve it with list_dir, find, grep or read_file. If the user says
  "the txt file", "the config" or similar and the workspace can identify it,
  inspect the workspace instead of asking for an exact name.
- To inspect the project, use run_command with short commands: git status,
  git diff, ls, find, wc, pytest, python3.
- run_command executes exactly one program without a shell. Never use pipes,
  redirections, `&&`, `||`, `;`, `cd`, `$VARIABLE` or `2>/dev/null`; make separate
  tool calls instead.
- If `graphify-out/graph.json` exists and the user asks where a flow, resource,
  module, test or architectural relation lives, look it up first with
  `graphify query "question" --graph graphify-out/graph.json --budget 700`.
  Graphify is for locating context; after that read the files and verify before
  declaring success. If there is no graph, fall back to local search with
  find/rg, and do not edit before locating.
- To delete a file or perform another operation not covered by a specialized
  file tool, call run_command with the exact terminal command (for example,
  `rm hello-world.txt`). The CLI, not you, handles user approval.
- Never claim that you created, edited, deleted, committed, tested or otherwise
  changed something unless at least one tool actually performed that action in
  this turn and its result confirms success.
- For git: run git status and git diff before proposing a commit.
- You may use git add, git commit and git push when the user asks for it; the
  CLI's own approval step is what actually gates execution, not this prompt.
- Before proposing a command that is destructive or hard to reverse (delete,
  overwrite, force flags, push, reset, and the like), say so plainly in your
  message so the user is approving something they understood, not reflexively
  hitting enter.
- If any tool returns a non-zero exit code, that is a failure. NEVER say a
  commit, push or test worked when the output showed an error.
- If git commit fails, stop and explain the error before trying to push.
- Keep decisions and results short. The terminal shows a summary of the commands
  and keeps the full output in the session log.
"""


def _install_signals():
    def leave(_signum, _frame):
        raise SystemExit(130)

    # SIGINT has to become KeyboardInterrupt so the REPL can restore the screen
    # and print the session summary. HUP/TERM keep terminating immediately.
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (AttributeError, ValueError):
        pass
    for sig in (signal.SIGHUP, signal.SIGTERM):
        try:
            signal.signal(sig, leave)
        except (AttributeError, ValueError):
            pass


def _close_without_interruption(cli):
    """Finish the cleanup even when the user hits Ctrl+C again on the way out."""
    previous = {}
    for sig in (signal.SIGINT, signal.SIGHUP, signal.SIGTERM):
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, signal.SIG_IGN)
        except (AttributeError, ValueError):
            pass
    try:
        while True:
            try:
                cli.close()
                return
            except KeyboardInterrupt:
                # A SIGINT may already have been delivered at the instant the
                # finally block started. From here on new ones are ignored.
                continue
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (AttributeError, ValueError):
                pass


def _ollama_ok(timeout=2):
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=timeout) as r:
            return json.load(r).get("version") or "ok"
    except Exception:
        return None


def _now():
    return dt.datetime.now().isoformat(timespec="seconds")


def _new_session_id():
    return str(uuid.uuid4())


def _valid_session_id(session_id):
    return bool(SESSION_ID_UUID.fullmatch(session_id)
                or SESSION_ID_LEGACY.fullmatch(session_id))


def _resume_command(session_id):
    launcher = HERE.parent / "isaacli"
    global_on_path = shutil.which("isaacli")
    if global_on_path:
        try:
            if Path(global_on_path).resolve() == launcher.resolve():
                return f"isaacli --resume {session_id}"
        except OSError:
            pass
    return f"{shlex.quote(str(launcher))} --resume {session_id}"


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


def _command_parts(cmd):
    try:
        return shlex.split(cmd)
    except ValueError:
        return []


def _command_rule(cmd):
    parts = _command_parts(cmd)
    if not parts:
        return ""
    if parts[0] == "git" and len(parts) > 1:
        sub = next((p for p in parts[1:] if not p.startswith("-")), "*")
        return f"git {sub}"
    if parts[0] == "gh" and len(parts) > 2:
        route = [p for p in parts[1:] if not p.startswith("-")][:2]
        return "gh " + " ".join(route)
    return parts[0]


def _safe_read_command(cmd):
    parts = _command_parts(cmd)
    if not parts:
        return False
    if parts[0] in READ_ONLY_COMMANDS:
        return True
    if parts[0] == "gh":
        route = tuple(p for p in parts[1:] if not p.startswith("-"))[:2]
        return route in READ_ONLY_GH
    return (parts[0] == "git" and len(parts) > 1
            and next((p for p in parts[1:] if not p.startswith("-")), None)
            in READ_ONLY_GIT)


def _destructive_command(cmd):
    """Whether the command deserves an explicit warning above the approval prompt.

    Being wrong here is cheap in one direction and expensive in the other: an
    extra warning costs a line of text, a missing one costs the habit of reading
    before pressing enter.
    """
    parts = _command_parts(cmd)
    if not parts:
        return False
    if parts[0] in DESTRUCTIVE_COMMANDS:
        return True
    if parts[0] == "git":
        sub = next((p for p in parts[1:] if not p.startswith("-")), None)
        return sub in DESTRUCTIVE_GIT
    return any(flag in parts for flag in ("--force", "-f", "--hard", "--delete"))


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


def _install_autocomplete():
    if readline is None:
        return

    def complete(text, state):
        options = [c for c in SLASH_COMMANDS if c.startswith(text)]
        try:
            return options[state] + " "
        except IndexError:
            return None

    readline.set_completer(complete)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")
    # Shift+Tab sends /mode as if the user had typed it. Works in GNU readline
    # without replacing the line editor (and its correct wrapping).
    readline.parse_and_bind('"\\e[Z": "/mode\\n"')


def _score_command(query, command, description):
    """Rank prefixes first and accept fuzzy matching on the rest."""
    query = query.casefold().strip()
    command_normal = command.casefold()
    description_normal = description.casefold()
    if not query or query == "/":
        return (0, SLASH_COMMANDS.index(command))
    if command_normal.startswith(query):
        return (0, len(command))
    term = query.removeprefix("/")
    name = command_normal.removeprefix("/")
    if term in name:
        return (1, name.index(term), len(name))
    if term in description_normal:
        return (2, description_normal.index(term), len(description_normal))
    position = 0
    distance = 0
    for character in term:
        found = name.find(character, position)
        if found < 0:
            return None
        distance += found - position
        position = found + 1
    return (3, distance, len(name))


def _filter_commands(query):
    found = []
    for command, key in COMMANDS:
        description = t(key)
        points = _score_command(query, command, description)
        if points is not None:
            found.append((points, command, description))
    found.sort(key=lambda item: item[0])
    return [(command, description) for _points, command, description in found]


class _CommandCompleter(Completer):
    def get_completions(self, document, _event):
        text = document.text_before_cursor
        if not text.startswith("/") or any(c.isspace() for c in text):
            return
        for command, description in _filter_commands(text):
            yield Completion(
                command,
                start_position=-len(text),
                display=command,
                display_meta=description,
            )


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


def _build_history(workspace):
    return [{"role": "system", "content": (
        agent.TOOLS_KNOWLEDGE + "\n\n" +
        CLI_KNOWLEDGE.format(workspace=str(workspace))
    )}]


def _load_session(session_id):
    """Rebuild the conversation and tool calls from a local JSONL by exact ID."""
    if not _valid_session_id(session_id):
        raise ValueError(t("cli.session.invalid_id"))
    path = SESSIONS_DIR / f"{session_id}.jsonl"
    if not path.is_file():
        raise ValueError(t("cli.session.not_found", id=session_id))
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError(t("cli.session.too_large"))

    events = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(t("cli.session.invalid_log", line=number)) from e
        if isinstance(event, dict):
            events.append(event)
    if not events:
        raise ValueError(t("cli.session.empty"))

    workspace = Path(events[-1].get("workspace") or os.getcwd()).expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(t("cli.session.workspace_gone", path=workspace))

    # Logs written before the identifiers were translated use Portuguese field
    # names. Reading both keeps `--resume` working on sessions already on disk;
    # dropping it would silently rebuild those sessions as empty.
    def field(event, name, legacy):
        value = event.get(name)
        return event.get(legacy) if value is None else value

    model = next((field(e, "model", "modelo") for e in reversed(events)
                  if field(e, "model", "modelo")), None)
    history = _build_history(workspace)
    transcript = []
    pending_tool = None
    tool_number = 0
    for event in events:
        kind = field(event, "type", "tipo")
        if kind == "meta" and field(event, "event", "evento") == "clear":
            history = _build_history(workspace)
            transcript = []
        elif kind == "user" and isinstance(event.get("content"), str):
            history.append({"role": "user", "content": event["content"]})
            transcript.append(("user", event["content"]))
        elif kind == "tool_start":
            tool_number += 1
            name = field(event, "name", "nome") or "unknown"
            args = event.get("args")
            if args is None and name == "run_command":
                args = {"cmd": event.get("cmd", "")}
            pending_tool = f"resume-tool-{tool_number}"
            history.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": pending_tool, "type": "function",
                                "function": {"name": name, "arguments": args or {}}}],
            })
            transcript.append(("tool_start", {
                "name": name, "args": args or {}, "cmd": event.get("cmd"),
            }))
        elif kind == "permission":
            transcript.append(("permission", {
                "cmd": event.get("cmd"),
                "decision": field(event, "decision", "decisao"),
            }))
        elif kind == "tool_result" and isinstance(field(event, "result", "resultado"), str):
            result = field(event, "result", "resultado")
            history.append({"role": "tool", "tool_call_id": pending_tool or "resume-tool",
                            "content": result})
            transcript.append(("tool_result", {
                "name": field(event, "name", "nome") or "unknown",
                "code": field(event, "code", "codigo"),
                "result": result,
            }))
            pending_tool = None
        elif kind == "assistant_final" and isinstance(event.get("content"), str):
            history.append({"role": "assistant", "content": event["content"]})
            if event["content"]:
                transcript.append(("assistant", event["content"]))
    return {"id": session_id, "path": path, "workspace": workspace,
            "model": model, "history": history, "transcript": transcript}


class IsaacCLI:
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

    def _provider_from_profile(self, item):
        if not item or item.get("provider", "ollama") == "ollama":
            return {"provider": "ollama"}
        secret_path = (Path(self.config_file).with_name("secrets.json")
                       if self.config_file else None)
        return {
            "provider": "openai_compatible",
            "provider_name": item.get("provider_name") or "API",
            "base_url": item.get("base_url"),
            "api_key": config.load_secret(item.get("credential"), secret_path),
        }

    def _persist_adjusted_thinking(self):
        """Write thinking=None to the active profile after the provider rejected
        the configured reasoning effort, so the error (and the extra round trip
        it costs) does not repeat in every future conversation. Returns False
        when there is nowhere to persist it. The caller has to tell the user
        that, not swallow the failure."""
        try:
            data = config.load(self.config_file)
        except ValueError as e:
            self._log("error", error=f"thinking_adjusted: unreadable configuration ({e})")
            return False
        name, item = config.profile(data)
        if not item or item.get("model") != self.model:
            self._log("error", error="thinking_adjusted: no saved profile matches "
                      f"the active model ({self.model})")
            return False
        item["thinking"] = None
        data["profiles"][name] = item
        config.save(data, self.config_file)
        return True

    def ensure_ollama(self, warn=False):
        if self.provider.get("provider") != "ollama":
            return ((self.provider.get("provider_name") or "API")
                    if self.provider.get("api_key") and self.provider.get("base_url") else None)
        with _shared_ollama_state() as state:
            version = _ollama_ok()
            server_valid = (
                state.get("managed")
                and _same_process(state.get("server_pid"), state.get("server_start"))
            )
            clients = [
                item for item in state.get("clients", [])
                if _same_process(item.get("pid"), item.get("start"))
            ]
            if not server_valid:
                state.clear()
                clients = []
            if version:
                if server_valid:
                    current = {"pid": self._runtime_pid, "start": self._runtime_start}
                    clients = [c for c in clients if c.get("pid") != self._runtime_pid]
                    clients.append(current)
                    state["clients"] = clients
                    self._ollama_registered = True
                # Without valid state, the server belongs to the user/system.
                return version
            if not self.autostart_ollama:
                return None
            exe = shutil.which("ollama")
            if not exe:
                if warn:
                    print(_color(t("cli.ollama.not_found"), "bad"))
                return None

            if warn:
                print(_color(t("cli.ollama.starting"), "warn"))
            try:
                self.ollama_proc = subprocess.Popen(
                    [exe, "serve"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception as e:
                self._log("error", error=f"ollama_autostart: {e}")
                if warn:
                    print(_color(t("cli.ollama.start_failed", error=e), "bad"))
                return None

            self._log("meta", event="ollama_autostart", pid=self.ollama_proc.pid)
            for _ in range(40):
                time.sleep(0.25)
                version = _ollama_ok(timeout=1)
                if version:
                    state.update({
                        "managed": True,
                        "server_pid": self.ollama_proc.pid,
                        "server_start": _pid_identity(self.ollama_proc.pid),
                        "clients": [{"pid": self._runtime_pid,
                                     "start": self._runtime_start}],
                    })
                    self._ollama_registered = True
                    return version
                if self.ollama_proc.poll() is not None:
                    self._log("error", error=(
                        f"ollama serve exited with code {self.ollama_proc.returncode}"
                    ))
                    return None
            version = _ollama_ok(timeout=1)
            if version:
                state.update({
                    "managed": True,
                    "server_pid": self.ollama_proc.pid,
                    "server_start": _pid_identity(self.ollama_proc.pid),
                    "clients": [{"pid": self._runtime_pid,
                                 "start": self._runtime_start}],
                })
                self._ollama_registered = True
                return version
            if self.ollama_proc.poll() is None:
                self.ollama_proc.terminate()
                self.ollama_proc.wait(timeout=3)
            return None

    def close(self):
        if not self._ollama_registered:
            return
        with _shared_ollama_state() as state:
            clients = [
                item for item in state.get("clients", [])
                if item.get("pid") != self._runtime_pid
                and _same_process(item.get("pid"), item.get("start"))
            ]
            state["clients"] = clients
            server_pid = state.get("server_pid")
            server_valid = (
                state.get("managed")
                and _same_process(server_pid, state.get("server_start"))
            )
            if clients or not server_valid:
                if not server_valid:
                    state.clear()
                self._ollama_registered = False
                return

            # The lock stays held until the process exits: a new session must not
            # see the server and register itself during the shutdown window.
            try:
                os.kill(int(server_pid), signal.SIGTERM)
                deadline = time.monotonic() + 3
                while _same_process(server_pid, state.get("server_start")):
                    if time.monotonic() >= deadline:
                        os.kill(int(server_pid), signal.SIGKILL)
                        break
                    time.sleep(0.05)
            except ProcessLookupError:
                pass
            state.clear()
            self._ollama_registered = False
            self._log("meta", event="ollama_autostop", pid=server_pid)

    def _log(self, kind, **data):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": _now(),
            "type": kind,
            "session_id": self.session_id,
            "model": self.model,
            "workspace": str(self.workspace),
            **data,
        }
        with self.session_path.open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

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

    def help_screen(self):
        print(t("cli.help.body"))

    def internal_command(self, text):
        if not text.startswith("/"):
            return False
        if text == "/":
            self.help_screen()
            return True
        parts = text.split(maxsplit=1)
        cmd = COMMAND_ALIASES.get(parts[0], parts[0])
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            raise EOFError
        if cmd in ("/help", "/?"):
            self.help_screen()
            return True
        if cmd == "/setup":
            import setup_ollama
            code = setup_ollama.run_setup(config_file=self.config_file)
            if code == 0:
                try:
                    data = config.load(self.config_file)
                    name, item = config.profile(data)
                except ValueError as e:
                    self.redraw_session(t("cli.setup.reread_failed", error=e))
                    return True
                if item:
                    self.model = item["model"]
                    self.thinking = item.get("thinking")
                    self.num_ctx = item.get("num_ctx")
                    self.provider = self._provider_from_profile(item)
                    self._log("meta", event="setup", profile=name,
                              model=self.model, thinking=self.thinking)
                    self.redraw_session(
                        t("cli.setup.profile_loaded", name=name, model=self.model))
            else:
                self.redraw_session(
                    t("cli.setup.cancelled") if code == 130
                    else t("cli.setup.incomplete")
                )
            return True
        if cmd == "/status":
            self.status()
            return True
        if cmd == "/tools":
            self.list_tools()
            return True
        if cmd == "/sessions":
            self.list_sessions()
            return True
        if cmd == "/history":
            self.show_history(arg)
            return True
        if cmd == "/show":
            self.show_command(arg or "last")
            return True
        if cmd == "/log":
            print(self.session_path)
            return True
        if cmd == "/feedback":
            self.feedback_help()
            return True
        if cmd == "/good":
            self.save_feedback("good", 10, arg)
            return True
        if cmd == "/bad":
            self.save_feedback("bad", 0, arg)
            return True
        if cmd == "/score":
            self.score_command(arg)
            return True
        if cmd == "/workspace":
            if not arg:
                print(self.workspace)
            else:
                self.set_workspace(arg)
                print(t("cli.workspace.now", path=self.workspace))
            return True
        if cmd == "/model":
            if not arg:
                self.select_model()
            else:
                try:
                    data = config.load(self.config_file)
                except ValueError:
                    data = config.empty_config()
                item = (data.get("profiles") or {}).get(arg)
                if item:
                    self.model = item["model"]
                    self.thinking = item.get("thinking")
                    self.num_ctx = item.get("num_ctx")
                    self.provider = self._provider_from_profile(item)
                    source = t("cli.model.source.profile", name=arg)
                else:
                    self.model = arg
                    self.thinking = None
                    self.num_ctx = None
                    self.provider = {"provider": "ollama"}
                    source = t("cli.model.source.ollama")
                self._log("meta", event="model", model=self.model,
                          thinking=self.thinking)
                print(t("cli.model.set", model=self.model, source=source))
            return True
        if cmd == "/mode":
            self.permission_mode = (
                "authorized_only" if self.permission_mode == "safe" else "safe"
            )
            print(t("cli.mode.changed", mode=self._permission_mode_label()))
            self._log("meta", event="permission_mode", mode=self.permission_mode)
            return True
        if cmd == "/permissions":
            try:
                data = config.load(self.config_file)
            except ValueError as e:
                print(t("cli.config.error", error=e))
                return True
            permissions = data.get("permissions") or {}
            if arg in ("clear workspace", "clear global"):
                if arg == "clear global":
                    permissions["global"] = []
                else:
                    (permissions.get("workspaces") or {}).pop(str(self.workspace), None)
                config.save(data, self.config_file)
                print(t("cli.permissions.cleared", scope=arg.removeprefix("clear ")))
                return True
            global_rules = permissions.get("global") or []
            local_rules = (permissions.get("workspaces") or {}).get(str(self.workspace), [])
            none = t("cli.permissions.none")
            print(t("cli.permissions.mode", mode=self._permission_mode_label()))
            print(t("cli.permissions.global",
                    rules=", ".join(global_rules) if global_rules else none))
            print(t("cli.permissions.workspace",
                    rules=", ".join(local_rules) if local_rules else none))
            print(t("cli.permissions.clear_hint"))
            return True
        if cmd == "/language":
            from i18n import SUPPORTED_LANGUAGES
            try:
                data = config.load(self.config_file)
            except ValueError:
                data = config.empty_config()
            codes = list(SUPPORTED_LANGUAGES)
            current = data.get("language") or "en"
            initial = codes.index(current) if current in codes else 0
            index = terminal_ui.select(
                t("cli.language.title"),
                [SUPPORTED_LANGUAGES[code] for code in codes],
                initial=initial,
                more_above=t("ui.more_above", count="{count}"),
                more_below=t("ui.more_below", count="{count}"),
            )
            data["language"] = codes[index]
            config.save(data, self.config_file)
            self.tr = set_language(codes[index])
            self._log("meta", event="language", language=codes[index])
            # Full-screen selector: come back through redraw_session like /setup
            # and /model, otherwise the leftover menu stays on screen and the
            # conversation is gone.
            self.redraw_session(
                t("cli.language.set", language=SUPPORTED_LANGUAGES[codes[index]]))
            return True
        if cmd == "/clear":
            self.history = _build_history(self.workspace)
            self._log("meta", event="clear")
            print(t("cli.clear.done"))
            return True
        if cmd == "/new":
            self.new_session()
            return True

        print(t("cli.unknown_command", cmd=cmd))
        return True

    def select_model(self):
        import setup_ollama

        code = setup_ollama.run_model_selector(config_file=self.config_file)
        if code != 0:
            message = (t("cli.model.selection_cancelled") if code == 130
                       else t("cli.model.unchanged"))
            self.redraw_session(message)
            return
        try:
            data = config.load(self.config_file)
        except ValueError as e:
            print(t("cli.config.error", error=e))
            return
        name, item = config.profile(data)
        if not item:
            print(t("cli.model.profile_missing"))
            return
        self.model = item["model"]
        self.thinking = item.get("thinking")
        self.num_ctx = item.get("num_ctx")
        self.provider = self._provider_from_profile(item)
        self._log("meta", event="model", profile=name, model=self.model,
                  thinking=self.thinking, num_ctx=self.num_ctx)
        context = (t("cli.model.context_suffix", context=_short_context(self.num_ctx))
                   if self.num_ctx else "")
        effort = (t("cli.model.effort_suffix", effort=self.thinking)
                  if self.thinking not in (None, False) else t("cli.model.no_reasoning"))
        self.redraw_session(
            t("cli.model.summary", name=name, context=context, effort=effort))

    def new_session(self):
        previous_id = self.session_id
        previous_path = self.session_path
        new_id = _new_session_id()
        self._log("meta", event="new_session", next_session=new_id)

        self.session_id = new_id
        self.session_path = SESSIONS_DIR / f"{new_id}.jsonl"
        self.feedback_path = FEEDBACK_DIR / f"{new_id}.jsonl"
        self.turns = 0
        self.failures = 0
        self.commands = []
        self.total_usage = {"prompt_eval_count": 0, "eval_count": 0,
                            "total_duration": 0, "eval_duration": 0}
        self.last_answer = ""
        self.ratings = 0
        self.resume_transcript = []
        self._working_visible = False
        self._assistant_label_pending = True
        self._token_buffer = []
        self._output_block = False
        self.set_workspace(self.workspace, reset=True)
        self._log("meta", event="start", pid=os.getpid(), model=self.model,
                  workspace=str(self.workspace), previous_session=previous_id)

        terminal_ui.clear()
        print(_color(t("cli.new.session", id=new_id), "assistant"))
        print(_color(t("cli.new.previous", path=previous_path), "dim"))
        print(_color(t("cli.new.resume", command=_resume_command(previous_id)), "dim"))

    def _permission_mode_label(self):
        return (t("cli.mode.saved_only") if self.permission_mode == "authorized_only"
                else t("cli.mode.safe_auto"))

    def _engine_label(self):
        if self.provider.get("provider") == "ollama":
            version = self.ensure_ollama(warn=False)
            return f"Ollama {version}" if version else t("cli.engine.unavailable")
        return self.provider.get("provider_name") or t("cli.engine.openai_compatible")

    def status(self):
        if self.provider.get("provider") == "ollama":
            engine = _ollama_ok() or t("cli.engine.no_response")
        else:
            engine = self.provider.get("provider_name") or t("cli.engine.openai_compatible")
        duration_s = self.total_usage.get("total_duration", 0) / 1_000_000_000
        default = t("cli.status.model_default")
        print(t("cli.status.session", id=self.session_id))
        print(t("cli.status.log", path=self.session_path))
        print(t("cli.status.pid", pid=os.getpid()))
        print(t("cli.status.model", model=self.model))
        print(t("cli.status.reasoning",
                value=self.thinking if self.thinking is not None else default))
        print(t("cli.status.context",
                value=_short_context(self.num_ctx) if self.num_ctx else default))
        print(t("cli.status.workspace", path=self.workspace))
        print(t("cli.status.engine", engine=engine))
        print(t("cli.status.turns", turns=self.turns))
        print(t("cli.status.commands", commands=len(self.commands), failures=self.failures))
        print(t("cli.status.permissions", mode=self._permission_mode_label()))
        print(t("cli.status.feedback", count=self.ratings, path=self.feedback_path))
        print(t("cli.status.tokens",
                prompt=self.total_usage.get("prompt_eval_count", 0),
                response=self.total_usage.get("eval_count", 0),
                seconds=f"{duration_s:.2f}"))
        print(t("cli.status.slash", commands=" ".join(SLASH_COMMANDS)))
        self._log("status", turns=self.turns, commands=len(self.commands),
                  failures=self.failures, usage=self.total_usage)

    def list_tools(self):
        import execution

        names = [s["function"]["name"] for s in tools.SCHEMA]
        print(t("cli.tools.list", names=", ".join(names)))
        print(t("cli.tools.terminal", names=", ".join(sorted(execution.ALLOWED))))
        print(t("cli.tools.git", names=", ".join(sorted(execution.GIT_ALLOWED))))

    def list_sessions(self):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(SESSIONS_DIR.glob("*.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            print(t("cli.sessions.none"))
            return
        for p in files[:12]:
            stat = p.stat()
            modified = dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            current = t("cli.sessions.current") if p == self.session_path else ""
            print(f"{p.stem}  {modified}  {stat.st_size} bytes{current}")

    def _history_text(self):
        events = list(self.resume_transcript)
        try:
            events.extend(_load_session(self.session_id)["transcript"])
        except ValueError:
            pass
        lines = []
        for role, content in events:
            if role == "user":
                lines.extend(["", f"❯ {content}"])
            elif role == "assistant":
                lines.extend(["", f"isaac: {content}"])
            elif role == "tool_start":
                name = content.get("name") or "unknown"
                if name == "run_command":
                    cmd = content.get("cmd") or (content.get("args") or {}).get("cmd", "")
                    lines.extend(["", f"$ {cmd}"])
                else:
                    args = json.dumps(content.get("args") or {}, ensure_ascii=False)
                    lines.extend(["", f"[{name}] → {args}"])
            elif role == "permission":
                lines.append(t("cli.history.permission",
                               decision=content.get("decision") or t("cli.history.unknown")))
            elif role == "tool_result":
                lines.append(content.get("result") or "")
        return "\n".join(lines).strip() or t("cli.history.empty")

    def redraw_session(self, message=None):
        """Report back to the conversation after a full-screen menu closed.

        The menu is the only thing that uses the alternate buffer, so leaving it
        already restores the conversation exactly as it was. Reprinting the
        transcript here would duplicate it in the scrollback; only the outcome
        of the menu is announced.
        """
        if not terminal_ui.interactive():
            if message:
                print(message)
            return
        if message:
            print(_color(message, "dim"))
        print()

    def show_history(self, _movement=""):
        # Printed normally, no full screen and no mouse capture: it stays in the
        # terminal's native scrollback, with formatted, copyable markdown.
        print(_format_markdown_terminal(self._history_text()))

    def show_command(self, which):
        if not self.commands:
            print(t("cli.show.none"))
            return
        if which == "last":
            item = self.commands[-1]
        else:
            try:
                number = int(which)
            except ValueError:
                print(t("cli.show.usage"))
                return
            item = next((c for c in self.commands if c["id"] == number), None)
            if item is None:
                print(t("cli.show.missing", number=number))
                return
        print(_color(t("cli.show.full", id=item["id"], cmd=item["cmd"]), "tool"))
        print(item["result"])

    def feedback_help(self):
        print(t("cli.feedback.body",
                feedback=self.feedback_path, session=self.session_path))

    def score_command(self, arg):
        if not arg:
            print(t("cli.score.usage"))
            return
        parts = arg.split(maxsplit=1)
        try:
            score = int(parts[0])
        except ValueError:
            print(t("cli.score.not_integer"))
            return
        if score < 0 or score > 10:
            print(t("cli.score.out_of_range"))
            return
        comment = parts[1] if len(parts) > 1 else ""
        self.save_feedback("score", score, comment)

    def save_feedback(self, kind, score, comment):
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": _now(),
            "feedback_kind": kind,
            "score": score,
            "comment": comment,
            "session_id": self.session_id,
            "session_path": str(self.session_path),
            "model": self.model,
            "workspace": str(self.workspace),
            "turns": self.turns,
            "commands": len(self.commands),
            "failures": self.failures,
            "usage": self.total_usage,
            "last_answer": self.last_answer,
        }
        with self.feedback_path.open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.ratings += 1
        self._log("feedback", **event)
        print(t("cli.feedback.saved", score=score, path=self.feedback_path))

    def feedback_reminder(self, had_command):
        if self.turns == 0:
            return
        if not had_command and self.turns % 3 != 0:
            return
        print()
        print(_color(t("cli.feedback.reminder"), "dim"))

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
