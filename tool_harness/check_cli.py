#!/usr/bin/env python3
"""Cheap tests for Isaac's CLI, without calling Ollama."""
import io
import builtins
import json
import os
import pty
import select
import sys
import tempfile
import termios
import time
from contextlib import redirect_stdout, nullcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Point the configuration at a throwaway directory BEFORE importing the CLI.
# Without this the suite reads (and /permissions would write to) the real
# ~/.config/isaacli/config.json, so the assertions would depend on whatever
# language the person running the tests happens to have configured.
root = Path(tempfile.mkdtemp())
os.environ["XDG_CONFIG_HOME"] = str(root / "config-home")

import cli as app
import config
import setup_ollama
import terminal_ui
import tools
import execution
from i18n import Translator

failures = []

EN = Translator("en")
PT = Translator("pt-BR")


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


check(not hasattr(app, "FALLBACK_MODEL"),
      "the CLI has no hardcoded fallback model")

original_interactive = terminal_ui.interactive
terminal_ui.interactive = lambda _input_fn=input: True
try:
    screen_sequences = io.StringIO()
    with redirect_stdout(screen_sequences), terminal_ui.alternate_screen():
        pass
finally:
    terminal_ui.interactive = original_interactive
check("\033[?1049h" in screen_sequences.getvalue()
      and "\033[?1049l" in screen_sequences.getvalue()
      and "\033[?1007h" not in screen_sequences.getvalue()
      and "\033[?1000h" not in screen_sequences.getvalue(),
      "the screen does not turn the wheel into arrows nor block native selection")

original_launcher_which = app.shutil.which
try:
    app.shutil.which = (
        lambda name: str(HERE.parent / "isaacli") if name == "isaacli" else None)
    check(app._resume_command("session") == "isaacli --resume session",
          "resume uses the short command when the global install points at this app")
    app.shutil.which = lambda _name: None
    check(str(HERE.parent / "isaacli") in app._resume_command("session"),
          "resume uses the full path when isaacli is not on PATH yet")
finally:
    app.shutil.which = original_launcher_which


class FakeInterruptedClose:
    def __init__(self):
        self.attempts = 0

    def close(self):
        self.attempts += 1
        if self.attempts == 1:
            raise KeyboardInterrupt


fake_close = FakeInterruptedClose()
app._close_without_interruption(fake_close)
check(fake_close.attempts == 2,
      "a repeated Ctrl+C neither interrupts nor shows a traceback during cleanup")

filtered = app._filter_commands("/sta")
check(filtered and filtered[0][0] == "/status",
      "incremental search prioritises the command prefix")
check(len(app._filter_commands("/")) == len(app.SLASH_COMMANDS),
      "a lone slash offers every command")
check(any(command == "/sessions"
          for command, _description in app._filter_commands("saved")),
      "command search also matches the description text")

if app.PromptSession is not None:
    from prompt_toolkit.document import Document
    completer = app._CommandCompleter()
    completions = list(completer.get_completions(Document("/sta"), None))
    check(completions and completions[0].text == "/status"
          and completions[0].start_position == -4,
          "the menu replaces the query with the selected command when completing")


sub = root / "project"
sub.mkdir()

cli = app.IsaacCLI("isaac-granite", sub, 4, autostart_ollama=False)
check(bool(app.SESSION_ID_UUID.fullmatch(cli.session_id)),
      "new sessions use a full UUIDv4")
check(tools.SANDBOX_ROOT == sub.resolve(), "the initial workspace becomes SANDBOX_ROOT")
check(str(sub.resolve()) in cli.history[0]["content"],
      "the system prompt states the workspace")
check("same language" in cli.history[0]["content"],
      "the system prompt requires answering in the user's language")
check(cli.session_path.exists(), "the CLI creates the session's JSONL log")

out = io.StringIO()
with redirect_stdout(out):
    cli.internal_command("/workspace")
check(str(sub.resolve()) in out.getvalue(), "/workspace with no argument shows the folder")

out = io.StringIO()
with redirect_stdout(out):
    cli.internal_command(f"/workspace {root}")
check(tools.SANDBOX_ROOT == root.resolve(), "/workspace swaps SANDBOX_ROOT")
check(str(root.resolve()) in out.getvalue(), "/workspace echoes the new folder")

out = io.StringIO()
with redirect_stdout(out):
    cli.internal_command("/model other")
check(cli.model == "other", "/model swaps the model")

setup_config = root / "config-setup.json"
cli_setup = app.IsaacCLI(
    "old-model", sub, 4, autostart_ollama=False, config_file=setup_config,
)
original_setup = setup_ollama.run_setup


def fake_setup(config_file=None):
    data = config.empty_config()
    data["profiles"]["new"] = {
        "provider": "ollama", "model": "new-model", "num_ctx": 16384,
        "thinking": "medium",
    }
    data["default_profile"] = "new"
    config.save(data, config_file)
    return 0


setup_ollama.run_setup = fake_setup
try:
    with redirect_stdout(io.StringIO()):
        cli_setup.internal_command("/setup")
finally:
    setup_ollama.run_setup = original_setup
check(cli_setup.model == "new-model" and cli_setup.thinking == "medium",
      "/setup reloads the engine in the session without closing the CLI")

picker_config = root / "config-picker.json"
picker_data = config.empty_config()
picker_data["profiles"]["qwen"] = {
    "provider": "ollama", "model": "qwen-model", "num_ctx": 16384,
    "context_limit": 32768, "thinking": "medium",
}
picker_data["default_profile"] = "qwen"
config.save(picker_data, picker_config)
cli_picker = app.IsaacCLI(
    "qwen-model", sub, 4, autostart_ollama=False, thinking="medium",
    num_ctx=16384, config_file=picker_config,
)
model_redraws = []
cli_picker.redraw_session = lambda message=None: model_redraws.append(message)
original_model_selector = setup_ollama.run_model_selector


def fake_model_selector(config_file=None):
    data = config.load(config_file)
    item = data["profiles"]["qwen"]
    item["thinking"] = "high"
    item["num_ctx"] = 32768
    data["default_profile"] = "qwen"
    config.save(data, config_file)
    return 0


setup_ollama.run_model_selector = fake_model_selector
try:
    with redirect_stdout(io.StringIO()):
        cli_picker.internal_command("/model")
finally:
    setup_ollama.run_model_selector = original_model_selector
picker_data = config.load(picker_config)
check(cli_picker.model == "qwen-model" and cli_picker.thinking == "high"
      and cli_picker.num_ctx == 32768,
      "/model selects the profile, effort and context through menus")
check(picker_data["default_profile"] == "qwen"
      and picker_data["profiles"]["qwen"]["num_ctx"] == 32768,
      "/model persists the quick selection without repeating /setup")
check(model_redraws and "qwen" in model_redraws[-1],
      "/model redraws the session after the full-screen menu closes")

# Every command that opens a full-screen selector has to come back through
# redraw_session. Without it the closed menu stays on screen and the
# conversation is gone -- which is exactly what /language used to do.
language_config = root / "config-language.json"
cli_language = app.IsaacCLI(
    "any-model", sub, 4, autostart_ollama=False, config_file=language_config,
)
language_redraws = []
cli_language.redraw_session = lambda message=None: language_redraws.append(message)
original_select = terminal_ui.select
terminal_ui.select = lambda *_a, **_kw: 1   # second entry: English
try:
    with redirect_stdout(io.StringIO()):
        cli_language.internal_command("/language")
finally:
    terminal_ui.select = original_select
check(config.load(language_config)["language"] == "en",
      "/language persists the chosen language")
check(language_redraws and "English" in (language_redraws[-1] or ""),
      "/language redraws the session after the full-screen menu closes")
app.set_language("en")

cli_redraw = app.IsaacCLI(
    "redraw-model", sub, 4, autostart_ollama=False,
    config_file=root / "config-redraw.json",
)
cli_redraw._log("user", content="previous question")
cli_redraw._log("assistant_final", content="previous answer\n" * 30)
cli_redraw.ensure_ollama = lambda warn=False: "test"
original_interactive_redraw = terminal_ui.interactive
original_clear_redraw = terminal_ui.clear
terminal_ui.interactive = lambda _input_fn=input: True
terminal_ui.clear = lambda _input_fn=input: None
redraw_out = io.StringIO()
try:
    with redirect_stdout(redraw_out):
        cli_redraw.redraw_session("model changed")
finally:
    terminal_ui.interactive = original_interactive_redraw
    terminal_ui.clear = original_clear_redraw
check("previous question" in redraw_out.getvalue()
      and "previous answer" in redraw_out.getvalue()
      and "model changed" in redraw_out.getvalue(),
      "the redraw restores the current conversation like a resume")

# /history no longer opens another full-screen layer: it is a plain print, so it
# stays in the terminal's native scrollback, with formatted, copyable markdown.
history_out = io.StringIO()
with redirect_stdout(history_out):
    cli_redraw.show_history()
check("previous question" in history_out.getvalue()
      and "previous answer" in history_out.getvalue(),
      "/history prints the full conversation without hijacking the screen")

cli_new = app.IsaacCLI(
    "new-model", sub, 4, autostart_ollama=False, config_file=root / "config-new.json",
)
previous_session = cli_new.session_id
previous_path = cli_new.session_path
cli_new.history.append({"role": "user", "content": "old context"})
cli_new.turns = 3
cli_new.commands.append({"id": 1})
with redirect_stdout(io.StringIO()):
    cli_new.internal_command("/new")
check(cli_new.session_id != previous_session and cli_new.session_path != previous_path,
      "/new creates another ID and another session file")
check(len(cli_new.history) == 1 and cli_new.turns == 0 and not cli_new.commands,
      "/new resets the context and counters without closing the CLI")
check(f'"next_session": "{cli_new.session_id}"' in previous_path.read_text()
      and cli_new.session_path.exists(),
      "/new closes the previous log and starts the new one with traceability")

cli_api = app.IsaacCLI(
    "api-model", sub, 4, autostart_ollama=False,
    config_file=root / "config-provider.json",
)
# The helper looks for secrets.json next to the config; use the expected path.
config.save_secret("api:test", "key",
                   (root / "config-provider.json").with_name("secrets.json"))
api_provider = cli_api._provider_from_profile({
    "provider": "openai_compatible", "provider_name": "Free server",
    "base_url": "https://api.example.test/v1", "credential": "api:test",
})
check(api_provider["provider"] == "openai_compatible"
      and api_provider["api_key"] == "key",
      "the CLI loads a generic API profile and its secret")

cli._log("user", content="message for the internal history")
cli._log("assistant_final", content="answer for the internal history")
internal_history = cli._history_text()
check("message for the internal history" in internal_history
      and "answer for the internal history" in internal_history,
      "/history rebuilds the session's messages without using the shell scrollback")

out = io.StringIO()
with redirect_stdout(out):
    cli.internal_command("/status")
status = out.getvalue()
check(cli.session_id in status and "Ollama tokens:" in status and "ratings:" in status,
      "/status shows the session, usage and ratings")

out = io.StringIO()
with redirect_stdout(out):
    cli.internal_command("/tools")
tools_out = out.getvalue()
check("run_command" in tools_out and "git:" in tools_out, "/tools lists the tools and git")
check("fetch_url" in tools_out, "/tools shows the dedicated web reader")
check(app._safe_read_command(
    "gh issue view 246 --repo aws-cloudformation/cloudformation-validate"),
    "read-only gh queries do not ask for needless approval")
check(not app._safe_read_command("gh issue close 246"),
      "mutating gh operations are never classified as reads")

check(app._destructive_command("rm notes.txt")
      and app._destructive_command("git push origin main")
      and app._destructive_command("git reset --hard")
      and not app._destructive_command("git status")
      and not app._destructive_command("ls"),
      "destructive commands are recognised so approval is not a reflex")

issue_url = "https://github.com/aws-cloudformation/cloudformation-validate/issues/246"
check(tools._normalize_web_url(issue_url) ==
      "https://api.github.com/repos/aws-cloudformation/cloudformation-validate/issues/246",
      "fetch_url converts a GitHub issue link into the public API")
original_getaddrinfo = tools.socket.getaddrinfo
try:
    tools.socket.getaddrinfo = lambda *_a, **_kw: [
        (tools.socket.AF_INET, tools.socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80)),
    ]
    private_web = tools.execute("fetch_url", {"url": "http://localhost/secret"})
finally:
    tools.socket.getaddrinfo = original_getaddrinfo
check("does not reach localhost" in private_web,
      "fetch_url blocks localhost and private networks before connecting")

cli._tool_after("run_command", {"cmd": "git status"},
                "$ git status\nok\n(exit code: 0)", "test")
out = io.StringIO()
with redirect_stdout(out):
    cli.internal_command("/show 1")
check("$ git status" in out.getvalue(), "/show expands a saved command")

failures_before_denial = cli.failures
out = io.StringIO()
with redirect_stdout(out):
    cli._tool_after(
        "run_command", {"cmd": "rm x"},
        "$ rm x\nDENIED BY USER: the command was not authorized.\n(exit code: 126)",
        "test",
    )
check(EN.t("cli.command.denied") in out.getvalue()
      and cli.failures == failures_before_denial,
      "a human denial is neither classified nor counted as a failure")

out = io.StringIO()
with redirect_stdout(out):
    cli.internal_command("/")
check("/setup" in out.getvalue() and "/status" in out.getvalue()
      and "/good" in out.getvalue(),
      "a lone slash shows the help and the engine repair")

out = io.StringIO()
with redirect_stdout(out):
    cli.internal_command("/feedback")
check(str(cli.feedback_path) in out.getvalue(), "/feedback shows where ratings go")

out = io.StringIO()
with redirect_stdout(out):
    cli.internal_command("/good it was useful")
check(cli.feedback_path.exists() and cli.ratings == 1, "/good saves the rating")
check('"score": 10' in cli.feedback_path.read_text(), "/good records a score of 10")

out = io.StringIO()
with redirect_stdout(out):
    cli.internal_command("/score 7 testing was missing")
check(cli.ratings == 2 and '"score": 7' in cli.feedback_path.read_text(),
      "/score saves a numeric score")

out = io.StringIO()
with redirect_stdout(out):
    cli.internal_command("/nota 5 alias antigo")
check(cli.ratings == 3 and '"score": 5' in cli.feedback_path.read_text(),
      "the Portuguese aliases (/nota, /bom, /ruim) still work")

out = io.StringIO()
with redirect_stdout(out):
    try:
        app.main(["--help"])
    except SystemExit:
        pass
check("--max-steps" not in out.getvalue(), "--max-steps stays hidden in the normal help")

out = io.StringIO()
with redirect_stdout(out):
    try:
        app.main(["--version"])
    except SystemExit as e:
        version_code = e.code
check(version_code == 0 and f"Isaac CLI v{app.APP_VERSION}" in out.getvalue(),
      "--version reports the application version")

original_setup_main = setup_ollama.run_setup
setup_ollama.run_setup = lambda: 1
try:
    with redirect_stdout(io.StringIO()):
        failed_setup_code = app.main(["setup"])
finally:
    setup_ollama.run_setup = original_setup_main
check(failed_setup_code == 1,
      "an explicit setup that fails does not silently open a fallback model")

cli._assistant_label_pending = True
cli._working_visible = False
out = io.StringIO()
with redirect_stdout(out):
    cli._show_working()
    cli._first_token_at = time.monotonic() - 1
    cli._generation_status_at = float("inf")
    cli._token("Hello")
    cli._clear_working()
check(out.getvalue().startswith("\n" + EN.t("cli.working.waiting")[:8])
      and "\r\033[2Kisaac: Hello" in out.getvalue(),
      "the working indicator is transient and leaves a break before the answer")

panel = app._welcome_lines(
    "long-model", "Ollama 0.30.10", sub, width=100, user="Weverton",
)
check(all(app._visual_width(line) == 100 for line in panel)
      and f"Isaac CLI v{app.APP_VERSION}" in panel[0]
      and "Welcome back, Weverton!" in "\n".join(panel)
      and "┬" in panel[0] and panel[1].count("│") == 3
      and all(line in "\n".join(panel) for line in app.WORDMARK_ISAAC)
      and EN.t("cli.welcome.shift_tab") in "\n".join(panel),
      "the welcome panel has the version, identity and stable alignment")
compact_panel = app._welcome_lines(
    "a-very-long-model-name", "engine", sub, width=40,
    user="A very long user name indeed",
)
check(all(app._visual_width(line) == 40 for line in compact_panel),
      "the welcome panel also fits a narrow terminal")

markdown = app._format_markdown_terminal(
    "# Title\n**strong** and `code`\n- [x] done\n```python\nprint(1)\n```\n"
    "[site](https://example.test)\x1b[2J",
    colors=True,
)
check("**" not in markdown and "```" not in markdown
      and "\033[1mstrong\033[0m" in markdown
      and "•" in markdown and "☑" in markdown
      and "\x1b[2J" not in markdown and "https://example.test" in markdown,
      "common Markdown gets styled and the model's control codes are stripped")

out = io.StringIO()
with redirect_stdout(out):
    cli._show_working()
    cli._first_token_at = time.monotonic() - 1
    cli._thinking_token("do not show this reasoning")
check("tok/s" in out.getvalue()
      and "do not show this reasoning" not in out.getvalue(),
      "thinking updates tok/s without revealing the reasoning")

# While the agent works there is no prompt reading stdin. Arrows and scrolling
# must not be echoed, but Ctrl+C has to keep being a signal.
master_fd, slave_fd = pty.openpty()
try:
    before = termios.tcgetattr(slave_fd)
    with terminal_ui.busy_input(fd=slave_fd):
        during = termios.tcgetattr(slave_fd)
        check(not (during[3] & termios.ECHO), "busy input disables echo")
        check(not (during[3] & termios.ICANON), "busy input disables line mode")
        check(bool(during[3] & termios.ISIG), "busy input preserves Ctrl+C")
        os.write(master_fd, b"\x1b[B\x1b[A")
    after = termios.tcgetattr(slave_fd)
    check(after == before, "busy input restores the terminal")
    ready, _, _ = select.select([slave_fd], [], [], 0)
    check(not ready, "keys typed during startup do not leak into the first prompt")
finally:
    os.close(master_fd)
    os.close(slave_fd)

# Command policy: reads run automatically, mutations only after a human decision.
original_exec = execution.run_command
original_permission_input = builtins.input
exec_calls = []


def fake_exec(cmd, authorized=False):
    exec_calls.append((cmd, authorized))
    return f"$ {cmd}\n(exit code: 0)"


execution.run_command = fake_exec
try:
    cli.config_file = root / "config-permissions.json"
    cli._approve_and_run("ls")
    check(exec_calls[-1] == ("ls", False),
          "safe mode runs a read without interrupting")
    builtins.input = lambda _prompt="": "w"
    cli._approve_and_run("rm file.txt")
    permission_data = config.load(cli.config_file)
    check("rm" in config.permission_rules(permission_data, cli.workspace),
          "approval can persist a rule for this workspace only")
    builtins.input = lambda _prompt="": (
        _ for _ in ()).throw(AssertionError("it must not ask"))
    cli._approve_and_run("rm other.txt")
    check(exec_calls[-1] == ("rm other.txt", True),
          "a persisted rule authorizes an equivalent later call")
    cli.permission_mode = "authorized_only"
    builtins.input = lambda _prompt="": "n"
    denied = cli._approve_and_run("git status")
    check("DENIED BY USER" in denied,
          "authorized-only mode asks even for a read")
finally:
    execution.run_command = original_exec
    builtins.input = original_permission_input

original_agent_run = app.agent.run
original_ensure = cli.ensure_ollama
try:
    cli.ensure_ollama = lambda warn=False: "test"
    app.agent.run = lambda *_a, **_kw: {
        "final": "Deleted successfully.", "calls": [], "usage": {},
    }
    out = io.StringIO()
    with redirect_stdout(out):
        cli.ask("delete the file")
finally:
    app.agent.run = original_agent_run
    cli.ensure_ollama = original_ensure
check("no changing tool was executed" in out.getvalue(),
      "the CLI contradicts a hallucinated success when no tool changed anything")

try:
    cli.ensure_ollama = lambda warn=False: "test"
    app.agent.run = lambda *_a, **_kw: {
        "final": "", "calls": [], "usage": {"eval_count": 3},
    }
    out = io.StringIO()
    with redirect_stdout(out):
        empty_answer_code = cli.ask("empty test")
finally:
    app.agent.run = original_agent_run
    cli.ensure_ollama = original_ensure
check(empty_answer_code == 1 and "no visible answer" in out.getvalue(),
      "the CLI reports a truly empty answer instead of showing only metrics")


def denied_agent(*_a, on_tool=None, **_kw):
    result = ("$ rm x\nDENIED BY USER: the command was not authorized.\n"
              "(exit code: 126)")
    on_tool("run_command", {"cmd": "rm x"}, result, "test")
    return {"final": "The deletion was denied.",
            "calls": [("run_command", {"cmd": "rm x"}, result, "test")],
            "usage": {}}


try:
    cli.ensure_ollama = lambda warn=False: "test"
    app.agent.run = denied_agent
    out = io.StringIO()
    with redirect_stdout(out):
        cli.ask("delete x")
finally:
    app.agent.run = original_agent_run
    cli.ensure_ollama = original_ensure
check(EN.t("cli.command.denied") in out.getvalue()
      and "Isaac CLI note" not in out.getvalue(),
      "a human denial does not produce a failure note at the end of the turn")

# When the provider rejects reasoning_effort (agent.run signals
# thinking_adjusted), the warning has to come from the language catalog (not
# hardcoded) and distinguish whether the correction was written to the profile
# or only applies to this session.
persist_config = root / "thinking-persist.json"
config.save({
    "version": 1, "language": "pt-BR", "default_profile": "profile-x",
    "profiles": {"profile-x": {
        "provider": "openai_compatible", "provider_name": "Test",
        "base_url": "https://api.example.test/v1", "model": "reasoning-model",
        "thinking": "medium", "credential": "api:profile-x", "temperature": 0,
    }},
    "permissions": {"global": [], "workspaces": {}},
}, persist_config)
cli_persist = app.IsaacCLI(
    "reasoning-model", sub, 4, autostart_ollama=False, thinking="medium",
    config_file=persist_config,
    provider={"provider": "openai_compatible", "api_key": "k", "base_url": "https://x"},
)
cli_persist.ensure_ollama = lambda warn=False: "test"
app.agent.run = lambda *_a, **_kw: {
    "final": "ok", "calls": [], "usage": {}, "thinking_adjusted": True,
}
persist_out = io.StringIO()
try:
    with redirect_stdout(persist_out):
        cli_persist.ask("hi")
finally:
    app.agent.run = original_agent_run
_, profile_after_adjust = config.profile(config.load(persist_config))
check(cli_persist.thinking is None and profile_after_adjust["thinking"] is None,
      "rejected reasoning is disabled in the session and written to the profile")
check(PT.t("thinking.rejected.persisted") in persist_out.getvalue(),
      "the rejected-reasoning warning comes from the language catalog (pt-BR), not hardcoded")

no_profile_config = root / "thinking-no-profile.json"
config.save({
    "version": 1, "language": "pt-BR", "default_profile": None, "profiles": {},
    "permissions": {"global": [], "workspaces": {}},
}, no_profile_config)
cli_no_profile = app.IsaacCLI(
    "no-profile-model", sub, 4, autostart_ollama=False, thinking="medium",
    config_file=no_profile_config,
    provider={"provider": "openai_compatible", "api_key": "k", "base_url": "https://x"},
)
cli_no_profile.ensure_ollama = lambda warn=False: "test"
app.agent.run = lambda *_a, **_kw: {
    "final": "ok", "calls": [], "usage": {}, "thinking_adjusted": True,
}
no_profile_out = io.StringIO()
try:
    with redirect_stdout(no_profile_out):
        cli_no_profile.ask("hi")
finally:
    app.agent.run = original_agent_run
check(cli_no_profile.thinking is None
      and PT.t("thinking.rejected.session_only") in no_profile_out.getvalue(),
      "with no matching saved profile, the warning says the fix is session-only")

# The two CLIs above switched the process language to pt-BR; put it back so the
# remaining assertions read the English catalog.
app.set_language("en")

cli.history.append({"role": "user", "content": "junk"})
cli.internal_command("/clear")
check(len(cli.history) == 1, "/clear clears the history")
check(str(root.resolve()) in cli.history[0]["content"], "/clear keeps the current workspace")

try:
    cli.internal_command("/exit")
    exited = False
except EOFError:
    exited = True
check(exited, "/exit ends the REPL")

original_input = builtins.input
original_ollama = cli.ensure_ollama
builtins.input = lambda _prompt="": (_ for _ in ()).throw(KeyboardInterrupt())
cli.ensure_ollama = lambda warn=False: "test"
try:
    with redirect_stdout(io.StringIO()):
        ctrl_c_code = cli._repl_screen()
finally:
    builtins.input = original_input
    cli.ensure_ollama = original_ollama
check(ctrl_c_code == 130, "Ctrl+C at the prompt ends the interface without a traceback")

cli.resume_transcript = [("user", "old message"), ("assistant", "old answer")]
builtins.input = lambda _prompt="": (_ for _ in ()).throw(KeyboardInterrupt())
cli.ensure_ollama = lambda warn=False: "test"
try:
    out = io.StringIO()
    with redirect_stdout(out):
        cli._repl_screen()
finally:
    builtins.input = original_input
    cli.ensure_ollama = original_ollama
check("old message" in out.getvalue() and "old answer" in out.getvalue(),
      "the REPL redraws the recent conversation when resuming")
cli.resume_transcript = []

original_alt_screen = app.terminal_ui.alternate_screen
original_repl_screen = cli._repl_screen
app.terminal_ui.alternate_screen = nullcontext
cli._repl_screen = lambda: 0
try:
    out = io.StringIO()
    with redirect_stdout(out):
        cli.repl()
finally:
    app.terminal_ui.alternate_screen = original_alt_screen
    cli._repl_screen = original_repl_screen
output_lines = out.getvalue().splitlines()
check(app._resume_command(cli.session_id) in output_lines,
      "the resume command sits alone on a copyable line")

resume_id = "2026-08-07-123456-abcdef"
resume_path = app.SESSIONS_DIR / f"{resume_id}.jsonl"
resume_events = [
    {"type": "meta", "workspace": str(sub), "model": "resume-model"},
    {"type": "user", "workspace": str(sub), "model": "resume-model",
     "content": "read the file"},
    {"type": "tool_start", "workspace": str(sub), "model": "resume-model",
     "name": "read_file", "args": {"path": "a.txt"}},
    {"type": "permission", "workspace": str(sub), "model": "resume-model",
     "cmd": "cat a.txt", "decision": "once"},
    {"type": "tool_result", "workspace": str(sub), "model": "resume-model",
     "name": "read_file", "result": "content"},
    {"type": "assistant_final", "workspace": str(sub), "model": "resume-model",
     "content": "The file contains content."},
]
resume_path.write_text("\n".join(json.dumps(e, ensure_ascii=False)
                                 for e in resume_events) + "\n")
resumed = app._load_session(resume_id)
check(resumed["model"] == "resume-model" and resumed["workspace"] == sub,
      "--resume recovers the model and the workspace")
check(any(m.get("role") == "tool" and m.get("content") == "content"
          for m in resumed["history"]), "--resume rebuilds messages and tools")
check([role for role, _ in resumed["transcript"]] == [
    "user", "tool_start", "permission", "tool_result", "assistant"
], "--resume prepares the visible conversation including tools and permissions")

# Logs recorded before the identifiers were translated must still resume.
legacy_id = "2026-08-06-101010-abcdef"
legacy_path = app.SESSIONS_DIR / f"{legacy_id}.jsonl"
legacy_events = [
    {"tipo": "user", "workspace": str(sub), "modelo": "legacy-model",
     "content": "pergunta antiga"},
    {"tipo": "tool_result", "workspace": str(sub), "modelo": "legacy-model",
     "nome": "read_file", "codigo": 0, "resultado": "conteudo antigo"},
]
legacy_path.write_text("\n".join(json.dumps(e, ensure_ascii=False)
                                 for e in legacy_events) + "\n")
legacy = app._load_session(legacy_id)
check(legacy["model"] == "legacy-model"
      and any(m.get("content") == "conteudo antigo" for m in legacy["history"]),
      "--resume still reads session logs written with the old field names")

cli.resume_transcript = resumed["transcript"]
builtins.input = lambda _prompt="": (_ for _ in ()).throw(KeyboardInterrupt())
cli.ensure_ollama = lambda warn=False: "test"
try:
    out = io.StringIO()
    with redirect_stdout(out):
        cli._repl_screen()
finally:
    builtins.input = original_input
    cli.ensure_ollama = original_ollama
check("[read_file] →" in out.getvalue() and "[read_file] ← content" in out.getvalue()
      and EN.t("cli.permission.decision.once") in out.getvalue(),
      "the REPL redraws executed actions in the resumed history")
cli.resume_transcript = []
try:
    app._load_session("../../etc/passwd")
    resume_safe = False
except ValueError:
    resume_safe = True
check(resume_safe, "--resume rejects IDs used as a path")

# Two sessions have to share the server Isaac started. The first one to leave
# must not take it down; only the last one ends the managed process.
original_runtime = os.environ.get("ISAACLI_RUNTIME_DIR")
original_ok = app._ollama_ok
original_which = app.shutil.which
original_popen = app.subprocess.Popen
original_identity = app._pid_identity
original_same = app._same_process
original_kill = app.os.kill
server = {"active": False}
identities = {101: "client-a", 202: "client-b"}
kills = []


class FakeOllamaProcess:
    pid = 999
    returncode = None

    def __init__(self, *_a, **kwargs):
        check(kwargs.get("start_new_session") is True,
              "the managed Ollama is born outside the terminal's group")
        server["active"] = True
        identities[self.pid] = "server"

    def poll(self):
        return None if server["active"] else 0

    def terminate(self):
        server["active"] = False

    def wait(self, timeout=None):
        return 0


try:
    os.environ["ISAACLI_RUNTIME_DIR"] = str(root / "runtime")
    app._ollama_ok = lambda timeout=2: "0.30-test" if server["active"] else None
    app.shutil.which = lambda _name: "/usr/bin/ollama"
    app.subprocess.Popen = FakeOllamaProcess
    app._pid_identity = lambda pid: identities.get(int(pid))
    app._same_process = lambda pid, start: identities.get(int(pid or -1)) == start

    def fake_kill(pid, signal_number):
        kills.append((pid, signal_number))
        if pid == 999:
            server["active"] = False
            identities.pop(999, None)

    app.os.kill = fake_kill
    cli_a = app.IsaacCLI("model", sub, 2, config_file=root / "cfg-a.json")
    cli_b = app.IsaacCLI("model", sub, 2, config_file=root / "cfg-b.json")
    cli_a._runtime_pid, cli_a._runtime_start = 101, "client-a"
    cli_b._runtime_pid, cli_b._runtime_start = 202, "client-b"
    check(cli_a.ensure_ollama() == "0.30-test", "the first session starts Ollama")
    check(cli_b.ensure_ollama() == "0.30-test", "the second session shares Ollama")
    cli_a.close()
    check(server["active"] and not kills,
          "closing one session preserves the Ollama another one is using")
    cli_b.close()
    check(not server["active"] and kills[-1][0] == 999,
          "the last session shuts the managed Ollama down")
finally:
    if original_runtime is None:
        os.environ.pop("ISAACLI_RUNTIME_DIR", None)
    else:
        os.environ["ISAACLI_RUNTIME_DIR"] = original_runtime
    app._ollama_ok = original_ok
    app.shutil.which = original_which
    app.subprocess.Popen = original_popen
    app._pid_identity = original_identity
    app._same_process = original_same
    app.os.kill = original_kill

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ISAAC CLI OK: workspace, model and basic output without Ollama")
