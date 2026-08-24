#!/usr/bin/env python3
"""Cheap tests for Isaac's CLI, without calling Ollama."""
import io
import builtins
import inspect
import json
import os
import pty
import re
import select
import subprocess
import sys
import tempfile
import termios
import time
from contextlib import redirect_stdout, nullcontext
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

# Point the configuration at a throwaway directory BEFORE importing the CLI.
# Without this the suite reads (and /permissions would write to) the real
# ~/.config/isaacli/config.json, so the assertions would depend on whatever
# language the person running the tests happens to have configured.
root = Path(tempfile.mkdtemp())
os.environ["XDG_CONFIG_HOME"] = str(root / "config-home")

import agent
import cli as app
import cli_ollama
import cli_presentation
import config
import installation
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


official_paths = {
    "/usr/local/bin/ollama", "/usr/local/lib/ollama",
    "/etc/systemd/system/ollama.service", "/usr/share/ollama",
}
ollama_plan = installation.official_ollama_plan(
    "/usr/local/bin/ollama",
    path_exists=lambda path: str(path) in official_paths,
    user_exists=True, group_exists=True,
)
check(ollama_plan[0] == ["sudo", "true"]
      and ["sudo", "rm", "-rf", "/usr/local/lib/ollama"] in ollama_plan
      and ["sudo", "rm", "-rf", "/usr/share/ollama"] in ollama_plan
      and installation.official_ollama_plan(
          "/usr/bin/ollama", path_exists=lambda _path: True,
          package_owned=False,
      ) is not None
      and installation.official_ollama_plan(
          "/usr/bin/ollama", path_exists=lambda _path: True,
          package_owned=True,
      ) is None,
      "Ollama purge recognizes official /usr/local and /usr layouts but not packages")
custom_models = root / "custom-models"
custom_service = root / "ollama.service"
custom_service.write_text(
    '[Service]\nEnvironment="OLLAMA_MODELS=/srv/ollama-models"\n',
    encoding="utf-8",
)
check(installation.custom_ollama_model_paths(
          home_dir=root / "home", environ={"OLLAMA_MODELS": str(custom_models)},
          service=root / "missing.service",
      ) == [custom_models],
      "Ollama purge detects custom model storage instead of claiming full removal")
check(installation.custom_ollama_model_paths(
          home_dir=root / "home", environ={}, service=custom_service,
      ) == [Path("/srv/ollama-models")]
      and installation.custom_ollama_model_paths(
          home_dir=root / "home", environ={"OLLAMA_MODELS": "relative-models"},
          service=root / "missing.service",
      ) == [Path("relative-models")],
      "systemd and relative custom model paths are both reported")


check(not hasattr(app, "FALLBACK_MODEL"),
      "the CLI has no hardcoded fallback model")

install_dir = root / "bin"
with redirect_stdout(io.StringIO()):
    first_install = app._install_launcher(install_dir)
    second_install = app._install_launcher(install_dir)
installed_launcher = install_dir / "isaacli"
installed_version = subprocess.run(
    [str(installed_launcher), "--version"], capture_output=True, text=True,
    check=False,
)
check(first_install == 0 and second_install == 0
      and installed_launcher.is_symlink()
      and installed_launcher.resolve() == HERE.parent / "isaacli"
      and installed_version.returncode == 0
      and "Isaac CLI v" in installed_version.stdout,
      "install creates an idempotent, executable per-user launcher symlink")
(install_dir / "isaacli").unlink()
(install_dir / "isaacli").write_text("another command", encoding="utf-8")
with redirect_stdout(io.StringIO()):
    conflicting_install = app._install_launcher(install_dir)
check(conflicting_install == 1
      and (install_dir / "isaacli").read_text(encoding="utf-8") == "another command",
      "install never overwrites an existing command")

purge_root = root / "purge"
purge_bin = purge_root / "bin"
purge_config = purge_root / "config" / "isaacli"
purge_sessions = purge_root / "cli_sessions"
purge_feedback = purge_root / "feedback"
purge_runtime = purge_root / "runtime" / "isaacli"
# Kaggle asset preparation stages its downloads in the cache directory, so this
# program creates it and purge has to be able to take it away again.
purge_cache = purge_root / "cache" / "isaacli"
for directory in (purge_config, purge_sessions, purge_feedback, purge_runtime,
                  purge_cache):
    directory.mkdir(parents=True)
    (directory / "state").write_text("private", encoding="utf-8")
untouched_clone = purge_root / "clone"
untouched_clone.mkdir()
with redirect_stdout(io.StringIO()):
    app._install_launcher(purge_bin)
    (purge_runtime / "ollama.json").write_text(json.dumps({"clients": [{
        "pid": os.getpid(), "start": app._pid_identity(os.getpid()),
    }]}), encoding="utf-8")
    active_purge = app._uninstall_launcher(
        purge=True, bin_dir=purge_bin, config_dir=purge_config,
        data_dirs=[purge_sessions, purge_feedback], runtime_dir=purge_runtime,
        cache_dir=purge_cache,
    )
    (purge_runtime / "ollama.json").write_text("{}", encoding="utf-8")
    purged = app._uninstall_launcher(
        purge=True, bin_dir=purge_bin, config_dir=purge_config,
        data_dirs=[purge_sessions, purge_feedback], runtime_dir=purge_runtime,
        cache_dir=purge_cache,
    )
check(active_purge == 1 and purged == 0 and not (purge_bin / "isaacli").exists()
      and not any(path.exists() for path in (
          purge_config, purge_sessions, purge_feedback, purge_runtime, purge_cache,
      )) and untouched_clone.exists(),
      "purge blocks active sessions, then removes private state and preserves the clone")

original_uninstall = app._uninstall_launcher
original_ollama_uninstall = app._uninstall_official_ollama
original_input = builtins.input
uninstall_calls = []
ollama_uninstall_calls = []
uninstall_order = []
try:
    def fake_uninstall(purge=False, check_only=False):
        uninstall_calls.append((purge, check_only))
        uninstall_order.append("check" if check_only else "isaac")
        return 0

    app._uninstall_launcher = fake_uninstall
    app._uninstall_official_ollama = (
        lambda: (ollama_uninstall_calls.append(True),
                 uninstall_order.append("ollama"), 0)[-1]
    )
    builtins.input = lambda _prompt: "n"
    with redirect_stdout(io.StringIO()):
        cancelled_purge = app.main(["uninstall", "--purge"])
    builtins.input = lambda _prompt: ""
    with redirect_stdout(io.StringIO()):
        cancelled_empty_purge = app.main(["uninstall", "--purge", "--ollama"])
    builtins.input = lambda _prompt: "y"
    with redirect_stdout(io.StringIO()):
        confirmed_purge = app.main(["uninstall", "--purge"])
        confirmed_ollama_purge = app.main(
            ["uninstall", "--purge", "--ollama"],
        )
finally:
    app._uninstall_launcher = original_uninstall
    app._uninstall_official_ollama = original_ollama_uninstall
    builtins.input = original_input
check(cancelled_purge == 130 and cancelled_empty_purge == 130
      and confirmed_purge == 0 and confirmed_ollama_purge == 0
      and uninstall_calls == [(True, False), (True, True), (True, False)]
      and ollama_uninstall_calls == [True]
      and uninstall_order[-3:] == ["check", "ollama", "isaac"],
      "a plain y/N confirmation, empty input stays a cancel, blocks a mistyped Enter")

original_interactive = terminal_ui.interactive
terminal_ui.interactive = lambda _input_fn=input: True
try:
    screen_sequences = io.StringIO()
    with redirect_stdout(screen_sequences), terminal_ui.alternate_screen():
        pass
    clear_sequences = io.StringIO()
    with redirect_stdout(clear_sequences):
        terminal_ui.clear()
finally:
    terminal_ui.interactive = original_interactive
# \033[3J drops the scrollback too. The REPL lives on the main screen, so its
# scrollback is the conversation: the shell that launched it must not sit one
# wheel turn above the first message.
check("\033[2J" in clear_sequences.getvalue()
      and "\033[3J" in clear_sequences.getvalue(),
      "clearing the screen also empties the scrollback")
check("terminal_ui.clear()" in inspect.getsource(app.IsaacCLI._initialize_repl),
      "the session opens on a screen of its own")
check("\033[?1049h" in screen_sequences.getvalue()
      and "\033[?1049l" in screen_sequences.getvalue()
      and "\033[?1007h" not in screen_sequences.getvalue()
      and "\033[?1000h" not in screen_sequences.getvalue(),
      "the full-screen menu enables no mouse reporting, which would block native selection")

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
(sub / "AGENTS.md").write_text("project-one-rule", encoding="utf-8")

cli = app.IsaacCLI("isaac-granite", sub, 4, autostart_ollama=False)
check(bool(app.SESSION_ID_UUID.fullmatch(cli.session_id)),
      "new sessions use a full UUIDv4")
check(tools.SANDBOX_ROOT == sub.resolve(), "the initial workspace becomes SANDBOX_ROOT")
check(str(sub.resolve()) in cli.history[0]["content"],
      "the system prompt states the workspace")
check("same language" in cli.history[0]["content"],
      "the system prompt requires answering in the user's language")
check("project-one-rule" in cli.history[0]["content"],
      "startup injects the workspace-root AGENTS.md")
check(cli.session_path.exists(), "the CLI creates the session's JSONL log")

# The window has to be declared before the workspace file is read, because
# every input cap is a share of it and the fallback ceiling is 32 KiB, which is
# about this machine's memory rather than the model's window. Measured on
# 2026-08-23 with the real config: an AGENTS.md of 21.082 characters went in
# whole under an 8.192 token profile and occupied 6.023 of those tokens, so the
# first "oi" of a fresh session left with 8.398 tokens and was refused. Nothing
# the user typed was to blame and nothing in the conversation could be
# compacted to fix it, which is why this is checked by effect on the prompt.
big = root / "bigproject"
big.mkdir()
(big / "AGENTS.md").write_text("x" * 21_000, encoding="utf-8")
narrow = app.IsaacCLI("isaac-granite", big, 4, autostart_ollama=False,
                      num_ctx=8192)
check("x" * 1000 not in narrow.history[0]["content"],
      "a workspace file larger than its share of the window stays out of the "
      f"prompt (prompt is {len(narrow.history[0]['content'])} characters)")
check(narrow._pending_workspace_instruction_warning,
      "and leaving it out is said out loud rather than done quietly")
check(agent.context_report(narrow.history, 8192)["used"]
      < agent.context_report(narrow.history, 8192)["budget"],
      "so an empty conversation starts inside its own budget")

# Switching to a profile with a different window has to re-read the file under
# the new one, or the larger model's copy is carried into the smaller window.
(big / "AGENTS.md").write_text("small-enough-rule", encoding="utf-8")
narrow.num_ctx = 32768
check("small-enough-rule" in narrow.history[0]["content"],
      "changing the profile's window re-reads the workspace file under it")

out = io.StringIO()
with redirect_stdout(out):
    cli.internal_command("/workspace")
check(str(sub.resolve()) in out.getvalue(), "/workspace with no argument shows the folder")

# Compacting the conversation is offered, never taken. What the screen has to
# carry at that moment is the number, the ceiling, which part of the number was
# measured, and what compacting would do, because it is the user answering.
context_report = {
    "num_ctx": 32_768, "used": 26_000, "measured": 24_000, "estimated": 2_000,
    "budget": 24_576, "headroom": 8_192, "over": True, "approaching": True,
    "compactable": 3,
}
original_inline = app.terminal_ui.select_inline
try:
    app.terminal_ui.select_inline = lambda options, **kwargs: 1
    kept_context = io.StringIO()
    with redirect_stdout(kept_context):
        kept_answer = cli._context_pressure(context_report)
    app.terminal_ui.select_inline = lambda options, **kwargs: 0
    with redirect_stdout(io.StringIO()):
        compact_answer = cli._context_pressure(context_report)
finally:
    app.terminal_ui.select_inline = original_inline
screen = kept_context.getvalue()
check(kept_answer is False and compact_answer is True,
      "the answer on the screen is what decides whether anything is compacted")
check("26000" in screen and "32768" in screen and "24576" in screen,
      "the offer carries where the conversation stands and where it has to stop")
check("24000" in screen and "2000" in screen,
      "the offer separates what the server counted from what is estimated")

compacted_out = io.StringIO()
with redirect_stdout(compacted_out):
    cli._context_note(context_report, ["one summary"])
check(f"--resume {cli.session_id}" in compacted_out.getvalue(),
      "what was compacted is recoverable, and the screen says with which command")

# The two checks above replace select_inline, which is the half of the screen a
# real terminal draws and a captured buffer never does: the line left behind
# after the answer only exists in the TTY branch. That is how this screen came
# to confirm a decision about the context window with the word "Permission",
# borrowed from the screen it was copied from. Drive the real menu on a real
# pseudo-terminal and read what stays on screen.
context_child = f"""
import sys
sys.path.insert(0, {str(HERE.parent / "tool_harness")!r})
import cli as app
from cli_i18n import set_language
set_language("en")
report = {{"num_ctx": 32768, "used": 26000, "measured": 24000,
           "estimated": 2000, "budget": 24576, "headroom": 8192,
           "over": True, "approaching": True, "compactable": 3}}
screen = app.IsaacCLI.__new__(app.IsaacCLI)
app.IsaacCLI._context_pressure(screen, report)
"""
def _drive_context_screen(keys):
    """Draw the offer on a real pseudo-terminal, type `keys`, return the screen."""
    drawn = b""
    master_fd, slave_fd = pty.openpty()
    try:
        child = subprocess.Popen(
            [sys.executable, "-c", context_child],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd, close_fds=True)
        os.close(slave_fd)
        slave_fd = None
        typed = False
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and child.poll() is None:
            ready, _, _ = select.select([master_fd], [], [], 0.2)
            if ready:
                try:
                    drawn += os.read(master_fd, 4096)
                except OSError:
                    break
            # Type only once the menu is on screen: tty.setraw flushes whatever
            # arrived before it, so a key sent too early is a key thrown away.
            if not typed and b"Compact now" in drawn:
                typed = True
                os.write(master_fd, keys)
        # Waiting, not polling. The read above ends with OSError the moment the
        # child closes its side of the pty, and asking poll() in that same
        # instant can find a process that has finished everything but has not
        # been reaped yet: measured on GitHub Actions 2026-08-22, that reported
        # a screen that had in fact been answered as one that never was. A
        # child that really is still running is killed, because waiting on it
        # unconditionally would raise out of the whole file and turn one failed
        # check into no report at all, and would leave a python holding this pty.
        try:
            child.wait(timeout=5)
            answered = True
        except subprocess.TimeoutExpired:
            answered = False
            child.kill()
            child.wait(timeout=5)
        drain = time.monotonic() + 1
        while time.monotonic() < drain:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if not ready:
                break
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            drawn += chunk
    finally:
        # The slave stays open here only if Popen itself failed, and then
        # nothing has anything to read: closing both is what keeps a check that
        # could not start from leaking the terminal it asked the kernel for.
        if slave_fd is not None:
            os.close(slave_fd)
        os.close(master_fd)
    return answered, drawn.decode(errors="replace")


answered, drawn = _drive_context_screen(b"\r")
check(answered, "the context offer answers a keypress on a TTY")
check("Context: Compact now" in drawn,
      "on a real terminal the answered offer is confirmed as a context decision")
check("Permission" not in drawn,
      "nothing about the window is announced as a permission")

# "k" moves the cursor up in every menu this program draws, and this screen used
# to bind it as a shortcut for "leave it as it is". Because shortcuts are read
# before navigation, pressing it to go up answered the question instead. Two of
# them wrap around a two-option menu and come back to where they started, so a
# screen that navigates ends on "compact" and a screen that answers on the first
# one does not.
answered_after_navigating, navigated = _drive_context_screen(b"kk\r")
check(answered_after_navigating and "Context: Compact now" in navigated,
      "the navigation keys move the cursor here instead of answering for the user")

# Fixing this caller alone would leave the trap armed for the next screen, in
# both of the ways it was armed here: an inherited default, and a shortcut that
# steals a navigation key.
inline_defaults = inspect.signature(terminal_ui.select_inline).parameters
check(all(word not in str(inline_defaults[name].default)
          for name in ("prompt", "chosen_label")
          for word in ("Permission", "w/g/n")),
      "the shared inline menu defaults to no particular screen's wording")
try:
    terminal_ui.select_inline(["a", "b"], shortcuts={"k": 1},
                              input_fn=lambda _p="": "1")
    refused_navigation_shortcut = False
except ValueError:
    refused_navigation_shortcut = True
check(refused_navigation_shortcut,
      "a screen cannot claim a navigation key as its answer shortcut")

out = io.StringIO()
(root / "AGENTS.md").write_text("project-two-rule", encoding="utf-8")
with redirect_stdout(out):
    cli.internal_command(f"/workspace {root}")
check(tools.SANDBOX_ROOT == root.resolve(), "/workspace swaps SANDBOX_ROOT")
check(str(root.resolve()) in out.getvalue(), "/workspace echoes the new folder")
check("project-two-rule" in cli.history[0]["content"]
      and "project-one-rule" not in cli.history[0]["content"],
      "/workspace replaces the previous project instructions")
check("previous workspace no longer apply" in cli.history[-1]["content"],
      "/workspace explicitly retires the previous instructions")

bad_workspace = root / "bad-workspace"
bad_workspace.mkdir()
(bad_workspace / "AGENTS.md").write_bytes(b"\xff")
bad_out = io.StringIO()
with redirect_stdout(bad_out):
    bad_cli = app.IsaacCLI("model", bad_workspace, 4, autostart_ollama=False)
    bad_cli._engine_label = lambda: "test"
    original_clear_bad = app.terminal_ui.clear
    app.terminal_ui.clear = lambda: None
    try:
        bad_cli._initialize_repl()
    finally:
        app.terminal_ui.clear = original_clear_bad
check(EN.t("cli.workspace.instructions_warning", path=bad_workspace / "AGENTS.md",
           reason=EN.t("cli.workspace.instructions.invalid_utf8")) in bad_out.getvalue()
      and len(bad_cli.history) == 1,
      "an invalid AGENTS.md remains visible after REPL startup")

pt_warning_config = root / "config-pt-warning.json"
pt_warning_data = config.empty_config()
pt_warning_data["language"] = "pt-BR"
config.save(pt_warning_data, pt_warning_config)
pt_warning_out = io.StringIO()
with redirect_stdout(pt_warning_out):
    pt_warning_cli = app.IsaacCLI(
        "model", bad_workspace, 4, autostart_ollama=False,
        config_file=pt_warning_config,
    )
    pt_warning_cli._show_workspace_instruction_warning()
check(PT.t("cli.workspace.instructions.invalid_utf8") in pt_warning_out.getvalue(),
      "the workspace instruction warning is rendered in Portuguese")
app.set_language("en")

empty_workspace = root / "empty-workspace"
empty_workspace.mkdir()
switch_cli = app.IsaacCLI("model", sub, 4, autostart_ollama=False)
switch_cli.set_workspace(empty_workspace)
check("project-one-rule" not in switch_cli.history[0]["content"],
      "/workspace to a folder without AGENTS.md removes the previous instructions")
invalid_switch_out = io.StringIO()
with redirect_stdout(invalid_switch_out):
    switch_cli.internal_command(f"/workspace {bad_workspace}")
check(EN.t("cli.workspace.instructions.invalid_utf8") in invalid_switch_out.getvalue(),
      "/workspace shows an instruction warning immediately")

snapshot = app.workspace_instructions.load_workspace_instructions(sub)
with patch.object(app.workspace_instructions, "load_workspace_instructions",
                  side_effect=AssertionError("unexpected second load")):
    resumed_constructor = app.IsaacCLI(
        "model", sub, 4, autostart_ollama=False,
        workspace_instructions_snapshot=snapshot,
    )
check(resumed_constructor.workspace_instructions is snapshot,
      "resume construction reuses exactly the snapshot already loaded")

out = io.StringIO()
with redirect_stdout(out):
    cli.internal_command("/model other")
check(cli.model == "other", "/model swaps the model")

# A bare model name is not a profile, and what the previous profile chose does
# not survive it. The temperature was the field that did: it was the one the
# four-line version of this branch did not clear, so a model picked by name
# ran at a temperature chosen for a different model.
leftover = app.IsaacCLI("m", sub, 4, autostart_ollama=False,
                        thinking="high", num_ctx=8192, temperature=0.7)
with redirect_stdout(io.StringIO()):
    leftover.internal_command("/model bare-name")
check((leftover.model, leftover.thinking, leftover.num_ctx,
       leftover.temperature) == ("bare-name", None, None, None),
      "a model named by hand starts from nothing, temperature included")

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

# The selector carries English defaults for the line it asks on and the line it
# answers a bad choice with. Every other screen in the program overrides both;
# this one did not, so a session in Portuguese was asked "Select: " in English.
# Recorded at the call site and then rendered for real, so what is checked is
# the text that reaches the screen and not the intention behind it.
app.set_language("pt-BR")
language_call = {}
terminal_ui.select = lambda *args, **kwargs: (
    language_call.update(args=args, kwargs=kwargs) or 1)
try:
    with redirect_stdout(io.StringIO()):
        cli_language.internal_command("/language")
finally:
    terminal_ui.select = original_select
language_answers = iter(["0", "1"])
with redirect_stdout(io.StringIO()) as language_screen:
    original_select(
        language_call["args"][0], language_call["args"][1],
        input_fn=lambda prompt="": (print(prompt, end="") or next(language_answers)),
        **{key: value for key, value in language_call["kwargs"].items()
           if key in {"prompt", "invalid"}})
language_drawn = language_screen.getvalue()
check("Selecione: " in language_drawn
      and "Select: " not in language_drawn
      and "Choose a number from 1 to" not in language_drawn,
      "the language screen asks and corrects in the language it is running in")
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
cleared = []
terminal_ui.interactive = lambda _input_fn=input: True
terminal_ui.clear = lambda _input_fn=input: cleared.append(True)
redraw_out = io.StringIO()
try:
    with redirect_stdout(redraw_out):
        cli_redraw.redraw_session("model changed")
finally:
    terminal_ui.interactive = original_interactive_redraw
    terminal_ui.clear = original_clear_redraw
# Closing the menu already puts the alternate buffer back: the conversation is
# on screen untouched. Clearing or reprinting it would erase or duplicate the
# terminal's scrollback.
check("model changed" in redraw_out.getvalue()
      and not cleared
      and "previous question" not in redraw_out.getvalue(),
      "returning from a menu announces the outcome without redrawing the session")

check("alternate_screen" not in inspect.getsource(app.IsaacCLI.repl),
      "the REPL stays on the main screen so the wheel scrolls the conversation")

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
(sub / "AGENTS.md").write_text("project-one-reloaded", encoding="utf-8")
with redirect_stdout(io.StringIO()):
    cli_new.internal_command("/new")
check(cli_new.session_id != previous_session and cli_new.session_path != previous_path,
      "/new creates another ID and another session file")
check(len(cli_new.history) == 1 and cli_new.turns == 0 and not cli_new.commands,
      "/new resets the context and counters without closing the CLI")
check(f'"next_session": "{cli_new.session_id}"' in previous_path.read_text()
      and cli_new.session_path.exists(),
      "/new closes the previous log and starts the new one with traceability")
check("project-one-reloaded" in cli_new.history[0]["content"],
      "/new reloads the current AGENTS.md")

cli_bad_new = app.IsaacCLI(
    "new-model", bad_workspace, 4, autostart_ollama=False,
    config_file=root / "config-bad-new.json",
)
cli_bad_new._show_workspace_instruction_warning()
bad_new_out = io.StringIO()
with redirect_stdout(bad_new_out):
    cli_bad_new.internal_command("/new")
check(EN.t("cli.workspace.instructions.invalid_utf8") in bad_new_out.getvalue(),
      "/new shows the reloaded instruction warning after clearing the screen")

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
# The step-limit message tells the user to raise the ceiling with this exact
# flag. While it was suppressed, --help denied that the flag existed, so the two
# halves of the program disagreed in front of whoever had just hit the ceiling.
# Whatever the message names, the help has to admit.
check("--max-steps" in out.getvalue(),
      "--max-steps appears in the help that the step-limit message sends people to")
check("--max-steps" in EN.t("cli.error.step_limit", steps=12)
      and "--max-steps" in PT.t("cli.error.step_limit", steps=12),
      "both languages point at the flag by the name the parser accepts")

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
    "long-model", "Ollama 0.30.10", sub, width=100,
)
check(all(app._visual_width(line) == 100 for line in panel)
      and f"Isaac CLI v{app.APP_VERSION}" in panel[0]
      and "Welcome back!" in "\n".join(panel)
      and "┬" in panel[0] and panel[1].count("│") == 3
      and all(line in "\n".join(panel) for line in app.WORDMARK_ISAAC)
      and EN.t("cli.welcome.shift_tab") in "\n".join(panel),
      "the welcome panel has the version, neutral greeting and stable alignment")
compact_panel = app._welcome_lines(
    "a-very-long-model-name", "engine", sub, width=40,
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

# A terminal wraps on the column, so it cuts words in half. What is checked
# here is the visible result: no rendered line goes past the width, and no word
# is split, which is the defect that was on screen.
wrap_width = 40


def visible_lines(rendered):
    return [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in rendered.splitlines()]


prose = ("Claro, vou criar um arquivo Python simples para demonstrar minha "
         "capacidade com **negrito atravessando** a quebra de linha.")
wrapped_prose = visible_lines(app._format_markdown_terminal(
    prose, colors=True, width=wrap_width, first_offset=len("isaac: ")))
check(all(len(line) <= wrap_width for line in wrapped_prose[1:])
      and len(wrapped_prose[0]) + len("isaac: ") <= wrap_width
      and " ".join(wrapped_prose).split() == re.sub(r"\*\*", "", prose).split(),
      "a long answer wraps by word, and the label's columns count on the first line")

structure = ("- item de lista bem comprido que precisa quebrar em mais de uma linha\n"
             "1. numerado tambem comprido o suficiente para quebrar em duas linhas\n"
             "> citacao comprida o bastante para quebrar em duas linhas no terminal\n"
             "## Um cabecalho bem comprido que tambem deveria quebrar por palavra")
wrapped_structure = visible_lines(app._format_markdown_terminal(
    structure, colors=True, width=wrap_width))
check(all(len(line) <= wrap_width for line in wrapped_structure)
      and [line[:3] for line in wrapped_structure] == [
          "• i", "  q", "1. ", "   ", "│ c", "│ q", "▌ U", "  d"],
      "list, quote and heading continuations line up under their own text")

# Code is the exception: breaking it produces a command nobody can copy.
code_block = "```sh\n" + "echo " + "a" * 80 + "\n```"
code_lines = visible_lines(app._format_markdown_terminal(
    code_block, colors=True, width=wrap_width))
check(any(len(line) > wrap_width for line in code_lines),
      "a code block is never rewrapped")

# An unbreakable word is left whole on its own line instead of being cut.
long_path = "/home/weverton/um/caminho/muito/longo/que/nao/pode/ser/cortado.txt"
path_lines = visible_lines(app._format_markdown_terminal(
    "final " + long_path, colors=True, width=wrap_width))
check(path_lines == ["final", long_path],
      "a path longer than the terminal stays whole on a line of its own")

# Everything that already fitted has to come out byte for byte as before, so
# alignment nobody asked us to touch is not collapsed.
short = "linha curta   com   espacos alinhados"
check(app._format_markdown_terminal(short, colors=True, width=wrap_width)
      == cli_presentation._markdown_inline(short, colors=True),
      "a line that already fits is returned untouched")

# The program's own notices are written as one long line in the catalogue on
# purpose, so the terminal decides where they break. They go out through say(),
# which decides it by word.
notice = EN.t("cli.error.step_limit", steps=12)
notice_out = io.StringIO()
with redirect_stdout(notice_out):
    cli_presentation.say(notice, "warn", width=wrap_width)
notice_lines = visible_lines(notice_out.getvalue().strip())
check(len(notice_lines) > 1
      and all(len(line) <= wrap_width for line in notice_lines)
      and " ".join(notice_lines) == notice,
      "a long notice is wrapped by word without a character being lost")

# Redirected output has no width to respect, and a command in a log has to
# survive being copied out of it whole.
piped = io.StringIO()
with redirect_stdout(piped):
    cli_presentation.say(notice)
check(piped.getvalue().strip() == notice,
      "output that is not a terminal stays on one line")

# Style has to survive the cut: the second half of a bold run stays bold.
split_bold = app._format_markdown_terminal(
    "um texto com **negrito bem comprido que atravessa a quebra** e o resto",
    colors=True, width=30).splitlines()
check(all(line.startswith("\033[1m") for line in split_bold[1:2])
      and split_bold[0].endswith("\033[0m"),
      "a bold run broken across lines closes and reopens instead of leaking")

out = io.StringIO()
with redirect_stdout(out):
    cli._show_working()
    cli._first_token_at = time.monotonic() - 1
    # Through the callback the session actually hands to `agent.run`. There was
    # a second one next to it, taking a thought and forwarding it here, that no
    # run ever passed: `on_progress` already carries the reasoning stream, so
    # wiring the other would have counted every thought twice.
    cli._generation_progress("do not show this reasoning")
check("tok/s" in out.getvalue()
      and "do not show this reasoning" not in out.getvalue(),
      "thinking updates tok/s without revealing the reasoning")

out = io.StringIO()
cli._working_visible = False
with redirect_stdout(out):
    cli._show_working()
    cli._show_working()
    cli._first_token_at = time.monotonic() - 1
    cli._generation_progress('{"path":"page.html","content":"generated"}')
rendered = out.getvalue()
check(rendered.count("\n") == 1 and "tok/s" in rendered
      and 'page.html' not in rendered and 'generated' not in rendered,
      "consecutive model steps reuse one live status line and tool arguments update it privately")

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
        "final": "Deleted successfully.", "calls": [], "usage": {}, "steps": 3,
    }
    out = io.StringIO()
    with redirect_stdout(out):
        cli.ask("delete the file")
finally:
    app.agent.run = original_agent_run
    cli.ensure_ollama = original_ensure
check("no changing tool was executed" in out.getvalue(),
      "the CLI contradicts a hallucinated success when no tool changed anything")
last_final = next(
    event for event in reversed([
        json.loads(line) for line in cli.session_path.read_text().splitlines()
    ]) if event.get("type") == "assistant_final"
)
check(last_final.get("steps") == 3,
      "the session log preserves the exact model step count")

# How a call was obtained is the only thing that separates "the model chose this
# tool" from "the schema constraint put the model back on the rails". It never
# reaches the screen, so the log is the only place it can be read, and both
# branches of _tool_after have to write it: run_command keeps its own.
with redirect_stdout(io.StringIO()):
    cli._tool_after("read_file", '{"path": "x"}', "content", "native")
    cli._tool_after("run_command", '{"cmd": "true"}', "exit code: 0", "constrained")
logged_via = [
    (event.get("name"), event.get("via"))
    for event in [json.loads(line)
                  for line in cli.session_path.read_text().splitlines()]
    if event.get("type") == "tool_result"
][-2:]
check(logged_via == [("read_file", "native"), ("run_command", "constrained")],
      "the session log records how each call was obtained, on both branches")
check(app._asks_for_mutation("vamos começar a criar um design.md")
      and app._asks_for_mutation("write a design file")
      and not app._asks_for_mutation("como criar um design.md?")
      and not app._asks_for_mutation("how do I create a design file?")
      and app._changing_tool_call("write_file", {})
      and not app._changing_tool_call("list_dir", {})
      and not app._changing_tool_call("run_command", {"cmd": "find . -name README.md"})
      and app._changing_tool_call("run_command", {"cmd": "mkdir site"})
      and app._changing_tool_succeeded("write_file", {}, "OK: wrote 1 byte")
      and not app._changing_tool_succeeded("write_file", {}, "ERROR: disk full")
      and app._changing_tool_succeeded(
          "run_command", {}, "$ mkdir site\n(exit code: 0)")
      and not app._changing_tool_succeeded(
          "run_command", {}, "$ mkdir site\n(exit code: 1)"),
      "mutation intent and changing tools stay distinct from read-only exploration")

try:
    cli.ensure_ollama = lambda warn=False: "test"
    app.agent.run = lambda *_a, **_kw: {
        "final": "", "calls": [("write_file", {}, "ERROR: missing path", "native")],
        "usage": {"eval_count": 3},
    }
    out = io.StringIO()
    with redirect_stdout(out):
        empty_answer_code = cli.ask("empty test")
finally:
    app.agent.run = original_agent_run
    cli.ensure_ollama = original_ensure
check(empty_answer_code == 1 and "no visible answer" in out.getvalue(),
      "the CLI reports an empty answer after a tool attempt instead of returning success")

# Running out of steps is unfinished work, not an empty or wrong answer, and
# "(step limit reached)" read like a freeze in whatever language the user is not
# using.
try:
    cli.ensure_ollama = lambda warn=False: "test"
    app.agent.run = lambda *_a, **_kw: {
        "final": None, "step_limit": 8,
        "calls": [("read_file", {}, "OK", "native")], "usage": {"eval_count": 3},
    }
    limit_out = io.StringIO()
    with redirect_stdout(limit_out):
        limit_code = cli.ask("long test")
finally:
    app.agent.run = original_agent_run
    cli.ensure_ollama = original_ensure
_limit_shown = limit_out.getvalue()
check(limit_code == 1 and "8" in _limit_shown and "--max-steps" in _limit_shown
      and "no visible answer" not in _limit_shown,
      "hitting the step ceiling says so and names the way forward")

def denied_agent(*_a, on_tool=None, **_kw):
    result = ("$ rm x\nDENIED BY USER: the command was not authorized.\n"
              "(exit code: 126)")
    on_tool("run_command", {"cmd": "rm x"}, result, "test")
    return {"final": "The deletion was denied.",
            "calls": [("run_command", {"cmd": "rm x"}, result, "test")],
            "usage": {}, "changing_calls": 1}


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

(root / "AGENTS.md").write_text("changed-after-workspace-switch", encoding="utf-8")
cli.history.append({"role": "user", "content": "junk"})
cli.internal_command("/clear")
check(len(cli.history) == 1, "/clear clears the history")
check(str(root.resolve()) in cli.history[0]["content"], "/clear keeps the current workspace")
check("project-two-rule" in cli.history[0]["content"]
      and "changed-after-workspace-switch" not in cli.history[0]["content"],
      "/clear keeps the session's instruction snapshot")

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

# "Recoverable" is a claim about a file on disk, so it is checked against the
# file: a conversation is compacted in memory and then rebuilt from its log,
# which still has to carry the whole result that was summarised away.
recoverable_id = "2026-08-22-090000-abcdef"
recoverable_path = app.SESSIONS_DIR / f"{recoverable_id}.jsonl"
recoverable_events = [
    {"type": "meta", "workspace": str(sub), "model": "resume-model"},
    {"type": "user", "workspace": str(sub), "model": "resume-model",
     "content": "fix the bug"},
    {"type": "tool_start", "workspace": str(sub), "model": "resume-model",
     "name": "read_file", "args": {"path": "client.ts"}},
    {"type": "tool_result", "workspace": str(sub), "model": "resume-model",
     "name": "read_file", "result": "y" * 60_000},
]
recoverable_path.write_text("\n".join(json.dumps(e, ensure_ascii=False)
                                      for e in recoverable_events) + "\n")
before_compaction = app._load_session(recoverable_id)
compacted = agent.fit_to_context(before_compaction["history"], 16_384,
                                 on_pressure=lambda report: True)
after_compaction = app._load_session(recoverable_id)
check(compacted and all(len(m.get("content") or "") < 60_000
                        for m in before_compaction["history"]),
      "the conversation in memory really was compacted")
check(any(m.get("role") == "tool" and len(m.get("content") or "") == 60_000
          for m in after_compaction["history"]),
      "the full result a summary replaced is still in the session log, and comes back")

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
original_ok = cli_ollama._ollama_ok
original_which = app.shutil.which
original_popen = app.subprocess.Popen
original_identity = cli_ollama._pid_identity
original_same = cli_ollama._same_process
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
    cli_ollama._ollama_ok = lambda timeout=2: "0.30-test" if server["active"] else None
    app.shutil.which = lambda _name: "/usr/bin/ollama"
    app.subprocess.Popen = FakeOllamaProcess
    cli_ollama._pid_identity = lambda pid: identities.get(int(pid))
    cli_ollama._same_process = lambda pid, start: identities.get(int(pid or -1)) == start

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
    cli_ollama._ollama_ok = original_ok
    app.shutil.which = original_which
    app.subprocess.Popen = original_popen
    cli_ollama._pid_identity = original_identity
    cli_ollama._same_process = original_same
    app.os.kill = original_kill


# The same lifecycle for a generic local server (llama-server or anything else
# speaking the compatible API) declared with "autostart" in the profile. Tested
# by effect: a fake process really goes up and down, and the assertions read the
# server state, never a message.
original_runtime = os.environ.get("ISAACLI_RUNTIME_DIR")
original_probe = cli_ollama._probe_health
original_popen = app.subprocess.Popen
original_identity = cli_ollama._pid_identity
original_same = cli_ollama._same_process
original_kill = app.os.kill
local_server = {"active": False}
local_identities = {303: "client-c", 404: "client-d"}
local_kills = []
launched = []

AUTOSTART_PROFILE = {
    "provider": "openai_compatible",
    "provider_name": "Llama Server",
    "base_url": "http://127.0.0.1:8080/v1",
    "autostart": {"cmd": ["llama-server", "-m", "model.gguf"],
                  "health_url": "http://127.0.0.1:8080/health"},
}


class FakeLocalServerProcess:
    pid = 888
    returncode = None

    def __init__(self, cmd, *_a, **kwargs):
        launched.append(cmd)
        check(kwargs.get("start_new_session") is True,
              "the managed local server is born outside the terminal's group")
        local_server["active"] = True
        local_identities[self.pid] = "local-server"

    def poll(self):
        return None if local_server["active"] else 0

    def terminate(self):
        local_server["active"] = False

    def wait(self, timeout=None):
        return 0


try:
    os.environ["ISAACLI_RUNTIME_DIR"] = str(root / "runtime-local")
    cli_ollama._probe_health = (
        lambda url, timeout=2: "ok" if local_server["active"] else None)
    app.subprocess.Popen = FakeLocalServerProcess
    cli_ollama._pid_identity = lambda pid: local_identities.get(int(pid))
    cli_ollama._same_process = (
        lambda pid, start: local_identities.get(int(pid or -1)) == start)

    def fake_local_kill(pid, signal_number):
        local_kills.append((pid, signal_number))
        if pid == 888:
            local_server["active"] = False
            local_identities.pop(888, None)

    app.os.kill = fake_local_kill
    cli_c = app.IsaacCLI("model", sub, 2, config_file=root / "cfg-c.json",
                         provider=dict(AUTOSTART_PROFILE))
    cli_d = app.IsaacCLI("model", sub, 2, config_file=root / "cfg-d.json",
                         provider=dict(AUTOSTART_PROFILE))
    cli_c._runtime_pid, cli_c._runtime_start = 303, "client-c"
    cli_d._runtime_pid, cli_d._runtime_start = 404, "client-d"
    check(cli_c.ensure_ollama() == "ok" and launched == [AUTOSTART_PROFILE["autostart"]["cmd"]],
          "the first session starts the configured local server, with its own command")
    check(cli_d.ensure_ollama() == "ok" and len(launched) == 1,
          "the second session shares the local server instead of starting another")
    cli_c.close()
    check(local_server["active"] and not local_kills,
          "closing one session preserves the local server another one is using")
    cli_d.close()
    check(not local_server["active"] and local_kills[-1][0] == 888,
          "the last session shuts the managed local server down")

    # A collision here would make one server's shutdown read the other's
    # clients and kill a process that still has users.
    runtime_files = {p.name for p in (root / "runtime-local" / "isaacli").iterdir()}
    check(any(name.startswith("autostart-llama-server") for name in runtime_files)
          and not any(name.startswith("ollama.") for name in runtime_files),
          "an autostart profile keeps its own lock and state, never Ollama's")
finally:
    if original_runtime is None:
        os.environ.pop("ISAACLI_RUNTIME_DIR", None)
    else:
        os.environ["ISAACLI_RUNTIME_DIR"] = original_runtime
    cli_ollama._probe_health = original_probe
    app.subprocess.Popen = original_popen
    cli_ollama._pid_identity = original_identity
    cli_ollama._same_process = original_same
    app.os.kill = original_kill

# A server on the user's own machine has no key to demand. Requiring one there
# blocked the whole local-first path through the guided setup.
local_no_key = app.IsaacCLI(
    "model", sub, 2, config_file=root / "cfg-e.json",
    provider={"provider": "openai_compatible", "provider_name": "Local",
              "base_url": "http://127.0.0.1:8080/v1", "api_key": ""})
remote_no_key = app.IsaacCLI(
    "model", sub, 2, config_file=root / "cfg-f.json",
    provider={"provider": "openai_compatible", "provider_name": "Groq",
              "base_url": "https://api.groq.com/openai/v1", "api_key": ""})
check(local_no_key.ensure_ollama() == "Local"
      and remote_no_key.ensure_ollama() is None,
      "a keyless local endpoint is usable while a keyless remote one is not")

# llama-server answers 503 while it is still reading the model file. Treating
# that as ready handed the user's first turn to a server that then refused it,
# with "API: Loading model". Tested by effect: the probe result itself changes.
import urllib.error as _urlerr


def _raise_http(code):
    def fake(url, timeout=0):
        raise _urlerr.HTTPError(url, code, "x", None, None)
    return fake


original_urlopen = cli_ollama.urllib.request.urlopen
try:
    cli_ollama.urllib.request.urlopen = _raise_http(503)
    loading = cli_ollama._probe_health("http://127.0.0.1:8080/v1/models", timeout=1)
    cli_ollama.urllib.request.urlopen = _raise_http(404)
    answering = cli_ollama._probe_health("http://127.0.0.1:8080/v1/models", timeout=1)
finally:
    cli_ollama.urllib.request.urlopen = original_urlopen
check(loading is None and answering == "ok",
      "a server still loading its model is not ready, while a 404 route still proves it is up")

# The Ollama budget does not transfer: its daemon answers at once and loads the
# model on demand, while llama-server reads the whole file before answering.
check(cli_ollama.AUTOSTART_TIMEOUT >= 60,
      "the autostart budget allows for a model that takes real time to load")

# An engine on this machine is brought up when the session opens, because the
# only thing that costs is a little VRAM. A remote endpoint is left alone: a
# Kaggle kernel burns quota by wall clock and is decided elsewhere, with the
# user's permission.
prewarm = app.IsaacCLI("model", sub, 2, config_file=root / "cfg-prewarm.json")
prewarm_calls = []
prewarm.ensure_ollama = lambda warn=False: (
    prewarm_calls.append(("ensure", warn)) or "test")
prewarm._preload_ollama_model = lambda: prewarm_calls.append(("preload",)) or True

prewarm.provider = {"provider": "openai_compatible", "provider_name": "Groq",
                    "base_url": "https://api.groq.com/openai/v1", "api_key": "k"}
with redirect_stdout(io.StringIO()):
    remote_prewarm = prewarm.prewarm_engine()
check(remote_prewarm is None and prewarm_calls == [],
      "a remote endpoint is not started when the session opens")

prewarm.provider = {"provider": "openai_compatible", "provider_name": "llama-server",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "autostart": {"cmd": ["llama-server"],
                                  "health_url": "http://127.0.0.1:8080/health"}}
with redirect_stdout(io.StringIO()):
    autostart_prewarm = prewarm.prewarm_engine()
check(autostart_prewarm == "test" and prewarm_calls == [("ensure", True)],
      "an autostart server is launched when the session opens, with no separate preload")

prewarm_calls.clear()
prewarm.provider = {"provider": "ollama"}
with redirect_stdout(io.StringIO()):
    ollama_prewarm = prewarm.prewarm_engine()
check(ollama_prewarm == "test"
      and prewarm_calls == [("ensure", True), ("preload",)],
      "Ollama gets its daemon and its weights before the first question")


def _interrupt(warn=False):
    raise KeyboardInterrupt


prewarm_calls.clear()
prewarm.ensure_ollama = _interrupt
with redirect_stdout(io.StringIO()):
    interrupted_prewarm = prewarm.prewarm_engine()
check(interrupted_prewarm is None,
      "Ctrl+C during the wait opens the prompt instead of raising out of startup")

# The daemon answering is not the model being loaded, so the preload has to
# reach Ollama itself. Tested by effect against a socket: what the server
# receives is the assertion.
import http.server
import threading

preload_seen = {}


class _PreloadHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        preload_seen["path"] = self.path
        preload_seen["body"] = json.loads(self.rfile.read(length) or b"{}")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"done": true}')

    def log_message(self, *_args):
        pass


preload_server = http.server.HTTPServer(("127.0.0.1", 0), _PreloadHandler)
preload_thread = threading.Thread(target=preload_server.serve_forever, daemon=True)
preload_thread.start()
original_agent_url = cli_ollama.agent.URL
try:
    host, port = preload_server.server_address
    cli_ollama.agent.URL = f"http://{host}:{port}/api/chat"
    loaded = app.IsaacCLI("qwen-test", sub, 2, config_file=root / "cfg-preload.json")
    preloaded = loaded._preload_ollama_model()
finally:
    cli_ollama.agent.URL = original_agent_url
    preload_server.shutdown()
    preload_server.server_close()
    preload_thread.join(timeout=5)
check(preloaded is True and preload_seen.get("path") == "/api/chat"
      and preload_seen.get("body") == {"model": "qwen-test"},
      "the preload asks Ollama to load exactly the session's model")

# A daemon that refuses the load is not the user's problem: the model still
# loads on the first question, so this reports false instead of crashing the
# opening screen.
original_preload_urlopen = cli_ollama.urllib.request.urlopen
try:
    cli_ollama.urllib.request.urlopen = _raise_http(404)
    refused_preload = loaded._preload_ollama_model()
finally:
    cli_ollama.urllib.request.urlopen = original_preload_urlopen
check(refused_preload is False,
      "a refused preload is reported as failed instead of breaking the session opening")

# --resume takes the session id, so it belongs on the opening panel, where it
# can still be copied before the conversation scrolls it away.
panel = app._welcome_lines("m", "Engine", "/tmp/w", "abc-123", width=100)
panel_without = app._welcome_lines("m", "Engine", "/tmp/w", width=100)
check(any("abc-123" in line for line in panel)
      and not any("abc-123" in line for line in panel_without)
      and len(panel) == len(panel_without) + 1,
      "the welcome panel shows the session id, and omits the row when there is none")

# A server that failed to start is not a missing credential. Sending the user
# to /setup to repair something that is not broken is worse than no message.
failed_autostart = app.IsaacCLI(
    "model", sub, 2, config_file=root / "cfg-g.json",
    provider=dict(AUTOSTART_PROFILE))
failed_autostart.ensure_ollama = lambda warn=False: None
out = io.StringIO()
with redirect_stdout(out):
    code = failed_autostart.ask("anything")
message = out.getvalue()
check(code == 1 and "Llama Server" in message and "--debug" in message
      and "/setup" not in message,
      "a local server that did not come up is reported as such, not as a bad credential")


# A profile that records `temperature` and is never read is a setting that
# silently does nothing. This checks the effect: the number the profile chose
# is the number that reaches agent.run, and a profile that chose nothing leaves
# the agent's own default alone.
captured_temperature = []
original_agent_run = app.agent.run


def fake_run(*_args, **kwargs):
    captured_temperature.append(kwargs.get("temperature", "__absent__"))
    return {"final": "done", "calls": [], "steps": 0, "usage": {},
            "thinking_adjusted": False, "changing_calls": 0,
            "successful_changes": 0}


try:
    app.agent.run = fake_run
    warm = app.IsaacCLI("m", ".", 4, temperature=0.7)
    warm.history = []
    warm.ensure_ollama = lambda warn=False: True
    with redirect_stdout(io.StringIO()):
        warm.ask("hi")
    cold = app.IsaacCLI("m", ".", 4)
    cold.history = []
    cold.ensure_ollama = lambda warn=False: True
    with redirect_stdout(io.StringIO()):
        cold.ask("hi")
finally:
    app.agent.run = original_agent_run

check(captured_temperature[:1] == [0.7],
      "the temperature a profile records is the temperature the request carries")
check(captured_temperature[1:2] == ["__absent__"],
      "a profile without temperature leaves the agent default untouched")

# Which of those settings a profile actually applies used to depend on which
# screen chose it: the five fields were assigned one by one in four places, so
# forgetting one meant a setting the user chose and the program ignores. There
# is one place now, and these two say so from both ends.
switching = app.IsaacCLI("m", ".", 4, thinking="high", num_ctx=8192,
                         temperature=0.7)
switching.apply_profile({"model": "other-model"})
check((switching.model, switching.thinking, switching.num_ctx,
       switching.temperature, switching.provider)
      == ("other-model", None, None, None, {"provider": "ollama"}),
      "a profile that chose nothing clears what the last one chose, "
      "rather than leaving a setting behind that belongs to a different model")
switching.apply_profile({"model": "third", "thinking": "low", "num_ctx": 4096,
                         "temperature": 0.1})
check((switching.model, switching.thinking, switching.num_ctx,
       switching.temperature) == ("third", "low", 4096, 0.1),
      "and every field a profile does choose is carried, not only the model")

profile_fields = []
for module in ("cli_commands.py", "cli_providers.py", "cli.py"):
    body = (HERE.parent / "tool_harness" / module).read_text(encoding="utf-8")
    if module == "cli_providers.py":
        # Everything after the one function that is allowed to do this.
        body = body.split("def apply_profile", 1)[1].split("\n    def ", 1)[1]
    for number, line in enumerate(body.splitlines(), 1):
        if re.search(r"self\.(model|thinking|num_ctx|temperature|provider)"
                     r"\s*=.*\bitem\b", line):
            profile_fields.append(f"{module}:{line.strip()}")
check(not profile_fields,
      "no screen unpacks a profile itself: " + (
          "; ".join(profile_fields) or "none does"))

# `/help` lists the commands in a single catalogue string, which is a copy of
# COMMANDS kept by hand, so a new command is invisible in it until somebody
# remembers. /config was, until this check existed. Both languages, because the
# copy is kept twice.
help_bodies = {
    name: Translator(name).t("cli.help.body")
    for name in ("en", "pt-BR")
}
undocumented = {
    name: [command for command in app.SLASH_COMMANDS
           if not re.search(rf"^  {re.escape(command)}(\s|$)", body, re.M)]
    for name, body in help_bodies.items()
}
check(not any(undocumented.values()),
      f"/help lists every slash command in every language ({undocumented})")

# The other direction, which is what a mangled row looks like: a line that
# opens like a command row but names nothing that exists. `  /configsession...`
# lived through a check that only asked whether /config appeared somewhere.
strays = {
    name: [line for line in body.splitlines()
           if line.startswith("  /")
           and line.split()[0] not in app.SLASH_COMMANDS]
    for name, body in help_bodies.items()
}
malformed = {
    name: [line for line in body.splitlines()
           # Not "no space in the line": the indent is always a space, which is
           # how the first version of this let a bare row through.
           if line.startswith("  /") and len(line.split()) < 2]
    for name, body in help_bodies.items()
}
check(not any(strays.values()) and not any(malformed.values()),
      f"every command row in /help names a real command ({strays}, {malformed})")

# ----------------------------------------------------------------------
# /config: the settings that used to be reachable only by editing the file.
#
# The screen is driven here the way a person drives it, by answering the menu,
# and what is asserted afterwards is the behaviour that changed, not the words
# the screen used to describe it. The one place words are asserted is the
# sentence that has to appear when context management is switched off, because
# saying it is the requirement.
# ----------------------------------------------------------------------
import cli_config

config_home = Path(tempfile.mkdtemp()) / "config.json"
config.save({
    "version": 1,
    "language": None,
    "default_profile": "local",
    "profiles": {
        # Exactly the state the real profile was found in: a window written
        # into the file during a test session, with nothing recording that a
        # screen put it there, because none did.
        "local": {"model": "qwen2.5-coder:3b", "num_ctx": 8192},
    },
    "permissions": {"global": [], "workspaces": {}},
}, config_home)

drive = app.IsaacCLI("qwen2.5-coder:3b", ".", 4, config_file=config_home,
                     num_ctx=8192)
drive.history = []

answers = []
drawn = []
original_select = cli_config.terminal_ui.select


def scripted_select(title, options, **_kwargs):
    drawn.append((title, list(options)))
    if not answers:
        raise AssertionError(f"the screen asked one question too many: {title}")
    return answers.pop(0)


def run_config(script):
    """Answer the menu with `script` and hand back what was drawn."""
    global answers
    answers = list(script)
    drawn.clear()
    cli_config.terminal_ui.select = scripted_select
    try:
        with redirect_stdout(io.StringIO()) as out:
            drive.config_screen()
    finally:
        cli_config.terminal_ui.select = original_select
    return drawn, out.getvalue()


# The command has to reach the screen. Everything below drives config_screen
# directly, which would keep passing with the dispatch never wired up.
dispatched = []
drive.config_screen = lambda: dispatched.append(True)
try:
    handled = drive.internal_command("/config")
finally:
    del drive.config_screen
check(handled and dispatched == [True],
      f"/config reaches the preferences screen ({handled}, {dispatched})")

# A number nobody chose must not be presented as a choice. This is the whole
# reason the origin column exists.
rows, _ = run_config([3])
check(any("8192" in option or "8K" in option for option in rows[0][1])
      and any(Translator("en").t("cli.config.origin.hand") in option
              for option in rows[0][1]),
      f"a window written into config.json by hand is labelled as such: {rows[0][1]}")

# Set it from the screen: 16384 is index 2 of WINDOW_CHOICES. The row that comes
# back has to stop calling it a hand edit, because now it is not one.
rows, _ = run_config([1, 2, 3])
saved = config.load(config_home)["profiles"]["local"]
check(saved.get("num_ctx") == 16_384 and drive.num_ctx == 16_384,
      f"the window chosen on the screen is what the profile and the session hold: {saved}")
rows, _ = run_config([3])
check(any(Translator("en").t("cli.config.origin.chosen") in option
          for option in rows[0][1])
      and not any(Translator("en").t("cli.config.origin.hand") in option
                  for option in rows[0][1]),
      f"a window chosen on the screen is labelled as chosen: {rows[0][1]}")

# Declaring no window is the case the task is really about: the local server is
# started by a script outside this repository, and with nothing saved here the
# honest row is that whatever that script passed is what holds. Clearing also
# has to forget that a screen was ever involved, otherwise the next hand edit
# would inherit the word "chosen".
rows, _ = run_config([1, len(cli_config.WINDOW_CHOICES), 3])
saved = config.load(config_home)["profiles"]["local"]
rows, _ = run_config([3])
check(saved.get("num_ctx") is None and drive.num_ctx is None
      and "num_ctx" not in (saved.get("chosen_in_isaacli") or []),
      f"clearing the window clears the record that a screen set it: {saved}")
check(any(Translator("en").t("cli.config.origin.inherited") in option
          for option in rows[0][1]),
      f"with no window saved the row says the server's own window holds: {rows[0][1]}")

# Temperature, the third setting that had no screen at all.
rows, _ = run_config([2, 1, 3])
saved = config.load(config_home)["profiles"]["local"]
check(saved.get("temperature") == 0.2 and drive.temperature == 0.2,
      f"the temperature chosen on the screen reaches the profile: {saved}")

# ---- context management, proven by effect ----------------------------------
# Turning it off through the screen has to change what the agent does, not just
# what the file says. The same oversized conversation is run through
# fit_to_context both ways, and what is asserted is whether anything was
# compacted.
def oversized():
    return [
        {"role": "system", "content": "contract"},
        {"role": "user", "content": "fix the bug"},
        {"role": "tool", "tool_call_id": "read_file",
         "content": "x" * 40_000},
        {"role": "tool", "tool_call_id": "read_file",
         "content": "y" * 40_000},
    ]


run_config([0, 1, 3])          # context management -> off
off_messages = oversized()
off_summaries = agent.fit_to_context(off_messages, 8192,
                                     manage=drive.manage_context)
run_config([0, 0, 3])          # context management -> on
on_messages = oversized()
on_summaries = agent.fit_to_context(on_messages, 8192,
                                    manage=drive.manage_context)
check(off_summaries == [] and len(off_messages[2]["content"]) == 40_000,
      "with context management off through the screen nothing is compacted "
      f"({len(off_summaries)} summaries)")
check(on_summaries and len(on_messages[2]["content"]) < 40_000,
      "with it back on through the screen the oversized results are compacted "
      f"({len(on_summaries)} summaries)")
check(config.load(config_home).get("context_management") is True,
      "the choice survives in the configuration file, not only in the session")

# Off is not "turn off the interruption": the screen has to say, at the moment
# of turning it off, that the request will now fail with its cause on screen.
# Asserted in Portuguese so a screen that quietly reverts to English fails here.
app.set_language("pt-BR")
try:
    rows, printed = run_config([0, 1, 3])
    toggle_title = rows[1][0]
finally:
    app.set_language("en")
run_config([0, 0, 3])
portuguese = Translator("pt-BR")
check(portuguese.t("cli.config.context.explain") in toggle_title,
      "the screen that offers to switch context management off explains, in "
      "the session's language, that off means failing out loud")
check(portuguese.t("cli.config.context.now_off") in printed,
      "switching it off says so again on the way back to the conversation")

# Ctrl+C is an answer to a menu, not an escape from the session. The previous
# version of a screen written inside an except block let it travel past the
# caller's own handler.
def interrupting_select(*_args, **_kwargs):
    raise KeyboardInterrupt


cli_config.terminal_ui.select = interrupting_select
try:
    with redirect_stdout(io.StringIO()):
        drive.config_screen()
    escaped = False
except KeyboardInterrupt:
    escaped = True
finally:
    cli_config.terminal_ui.select = original_select
check(not escaped, "Ctrl+C on the preferences list closes the screen and no more")

# The same key inside one setting goes back to the list instead of leaving, so
# the list is drawn a second time and the run ends on the Close row.
back_out = []


def interrupt_once(title, options, **_kwargs):
    back_out.append(title)
    if len(back_out) == 1:
        return 0
    if len(back_out) == 2:
        raise KeyboardInterrupt
    return len(options) - 1


cli_config.terminal_ui.select = interrupt_once
try:
    with redirect_stdout(io.StringIO()):
        drive.config_screen()
finally:
    cli_config.terminal_ui.select = original_select
check(len(back_out) == 3,
      f"backing out of one setting returns to the list ({len(back_out)} screens)")

# The window the server is really running is asked for, not assumed. A profile
# number is what isaacli sends; only the server knows what it was started with.
props_body = io.BytesIO(
    json.dumps({"default_generation_settings": {"n_ctx": 4096}}).encode())
props_body.__enter__ = lambda: props_body
props_body.__exit__ = lambda *_a: False
asked = []


def fake_urlopen(url, timeout=None):
    asked.append(url)
    return props_body


original_urlopen = cli_config.urllib.request.urlopen
try:
    cli_config.urllib.request.urlopen = fake_urlopen
    reported = cli_config.server_window(
        {"provider": "openai_compatible", "base_url": "http://127.0.0.1:8080/v1"})
finally:
    cli_config.urllib.request.urlopen = original_urlopen
check(reported == 4096 and asked == ["http://127.0.0.1:8080/props"],
      f"the local server is asked what window it is running ({reported}, {asked})")


def refusing_urlopen(_url, timeout=None):
    raise OSError("connection refused")


try:
    cli_config.urllib.request.urlopen = refusing_urlopen
    silent = cli_config.server_window(
        {"provider": "openai_compatible", "base_url": "http://127.0.0.1:8080/v1"})
finally:
    cli_config.urllib.request.urlopen = original_urlopen
check(silent is None,
      "a server that cannot be asked yields no number, so the screen says it "
      "does not know instead of presenting the saved one as the truth")
check(cli_config.server_window({"provider": "ollama"}) is None,
      "nothing is probed when the profile names no local endpoint")

# ----------------------------------------------------------------------
# The two catalogues have to stay one catalogue in two languages.
#
# Every screen goes through i18n, so a key added to one file and forgotten in
# the other shows the raw key to whoever runs in the other language, and a
# placeholder that survives in one file but not the other raises KeyError at the
# moment the message is needed. Neither shows up until somebody switches
# language, which is exactly when nobody is looking.
# ----------------------------------------------------------------------
locales = HERE.parent / "tool_harness" / "locales"
catalogues = {
    path.stem: json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(locales.glob("*.json"))
}
english = catalogues["en"]
missing = {
    name: sorted(set(english) ^ set(catalogue))
    for name, catalogue in catalogues.items() if set(english) != set(catalogue)
}


def placeholders(text):
    return sorted(re.findall(r"\{([a-z_]+)\}", str(text)))


mismatched = {
    f"{name}:{key}": (placeholders(english[key]), placeholders(catalogue[key]))
    for name, catalogue in catalogues.items()
    for key in set(english) & set(catalogue)
    if placeholders(english[key]) != placeholders(catalogue[key])
}
check(len(catalogues) > 1 and not missing,
      f"every language catalogue holds the same keys ({missing or 'aligned'})")
check(not mismatched,
      f"a message takes the same placeholders in every language ({mismatched or 'aligned'})")

# ----------------------------------------------------------------------
# A key nobody asks for is translated text that no screen can ever show. The
# comparison above keeps the two catalogues equal to each other, so an orphan
# stays perfectly aligned in both files and reads as supported wording to
# whoever writes the next screen. One did live that way, in both files, from the
# commit that introduced it until a repository-wide sweep in 2026-08-22. Naming
# it here would have put it back in the sources this check reads.
#
# Some keys are assembled at the call site (`t(f"model.origin.{name}")`), and a
# literal search cannot see those. Exempting the whole prefix, which is what
# this did until 2026-08-24, hides the orphan that costs most: a suffix that no
# code can produce any more keeps reading as used because a sibling of it is.
# `i18n_scan.orphan_keys` resolves the suffix instead, and it found two live
# orphans the day it was written, both left behind by the commit that redrew
# the model list. Neither is named here, and that is the point: a key written
# into this file is a key this check then reads as used, which is how the
# previous sweep's own note would have resurrected the orphan it described.
#
# Anything under a dot-directory is skipped, and that is load-bearing rather
# than tidiness: a git worktree lives in `.claude/worktrees/` and holds a whole
# second copy of this repository, so without the skip a key deleted here is
# still "asked for" by a checkout from last week, and every removal this check
# is supposed to catch passes.
# ----------------------------------------------------------------------
sources = {}
for path in sorted((HERE.parent).rglob("*")):
    if not path.is_file() or "locales" in path.parts:
        continue
    if any(part.startswith(".") for part in path.relative_to(HERE.parent).parts):
        continue
    if path.suffix not in (".py", ".tmpl", ".sh") and path.name != "isaacli":
        continue
    if "tasks" in path.parts or "__pycache__" in path.parts:
        continue
    try:
        sources[path] = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue

import i18n_scan

orphans = i18n_scan.orphan_keys(english, sources)
check(not orphans,
      f"every catalogue key is asked for by some screen ({orphans or 'all used'})")

# Proven by effect in both directions, because a sweep that never reports is
# indistinguishable from one that reports everything. The planted pair is the
# shape that actually occurs: one key written out in full and one assembled
# from a prefix, each with a sibling that no source can produce.
planted_sources = {
    "planted.py": 'def draw(t, kind):\n'
                  '    print(t("planted.title"))\n'
                  '    print(t(f"planted.kind.{kind}"))\n'
                  '    return "known"\n',
}
planted_catalogue = ["planted.title", "planted.kind.known",
                     "planted.kind.vanished", "planted.gone"]
check(i18n_scan.orphan_keys(planted_catalogue, planted_sources)
      == ["planted.gone", "planted.kind.vanished"],
      "the sweep resolves an assembled key to its suffix instead of exempting "
      "the whole prefix, and still sees a plain key nobody asks for")

# ----------------------------------------------------------------------
# A sentence that never entered a catalogue at all has no pair to be missing,
# so the comparison above is blind to it. `i18n_scan` closes that by reading
# the sources: literal text handed to a call that writes on the screen.
#
# The whole value of it is what it does NOT flag. This project writes English
# on purpose in two other places, and a scan that accused those would be turned
# off within the week: text the model reads (system prompt, tool description,
# tool result, sandbox refusal) is a contract and is never translated, and a
# `--debug` note is neither. So both directions are proven here, on one planted
# module that carries both kinds at once.
# ----------------------------------------------------------------------
import i18n_scan

planted = '''
import sys
import debug
import terminal_ui

SCHEMA = [{"function": {"name": "read_file",
                        "description": "Read a file from the workspace."}}]


def refuse():
    return "Refused: the sandbox does not reach outside the workspace."


def build(msgs):
    msgs.append({"role": "system",
                 "content": "You are a coding agent. Call one tool per step."})
    debug.note("planted", "the probe answered nothing at all")
    print("this went to the debug channel", file=sys.stderr)
    return terminal_ui.select(t("planted.title"), [t("planted.only")])


def screen():
    print("Nothing here can be undone later.")
    terminal_ui.select("Pick a server to talk to", [t("planted.only")])
'''
planted_offenders = i18n_scan.interface_literals("planted.py", planted)
planted_lines = planted.splitlines()
expected = [
    (planted_lines.index('    print("Nothing here can be undone later.")') + 1,
     "print()", "Nothing here can be undone later."),
    (planted_lines.index(
        '    terminal_ui.select("Pick a server to talk to", [t("planted.only")])') + 1,
     "terminal_ui.select()", "Pick a server to talk to"),
]
check(len(planted_offenders) == len(expected)
      and all(f"planted.py:{line} {sink}" in offender and repr(text) in offender
              for (line, sink, text), offender in zip(expected, planted_offenders)),
      "an interface literal is refused with its file, its line and its sink "
      f"({planted_offenders})")
# Same module, same scan, same run: the model's English and the debug note come
# through untouched. This is the assertion that keeps the check alive.
model_text = ("Read a file from the workspace.", "Refused: the sandbox",
              "You are a coding agent", "the probe answered nothing",
              "this went to the debug channel")
check(not any(text in offender for text in model_text
              for offender in planted_offenders),
      "text the model reads and a --debug note are not mistaken for interface "
      f"text ({planted_offenders})")

# The whole package as it stands, which is what makes this a check and not a
# demonstration.
live_offenders = []
for path in sorted((HERE.parent / "tool_harness").glob("*.py")):
    live_offenders += i18n_scan.interface_literals(
        path.name, path.read_text(encoding="utf-8"))
check(not live_offenders,
      f"no module writes interface text outside the catalogues: {live_offenders}")

# Every excuse in the table has to still describe something real. An entry that
# outlives the line it excused is how a table like this turns into the place a
# rule goes to be forgotten: it silently pre-approves whatever text is written
# next with those exact words.
declared_stale = sorted(
    f"{name}:{text!r}" for name, texts in i18n_scan.DECLARED.items()
    for text in texts
    if text not in {
        found for _line, _sink, found in i18n_scan.screen_literals(
            name, (HERE.parent / "tool_harness" / name).read_text(encoding="utf-8"))
    }
)
check(not declared_stale,
      f"every declared exception still names a line that exists: {declared_stale}")

# `--help` is the first thing anybody sees of this program, and every row of the
# commands section ran past 80 columns, the longest at 132. A terminal that
# narrow wraps them itself, wherever the character happens to land, which breaks
# the very column the alignment exists to make.
import importlib as _importlib
import os as _os
import re as _re

_cli_module = _importlib.import_module("cli")

_ANSI = _re.compile(r"\033\[[0-9;]*m")
for _width in (80, 100, 132):
    _os.environ["COLUMNS"] = str(_width)
    _too_wide = [
        line for line in _cli_module._commands_epilog().splitlines()
        if len(_ANSI.sub("", line)) > _width
    ]
    check(not _too_wide,
          f"the commands section of --help fits {_width} columns "
          f"({_too_wide[:1] or 'fits'})")
_os.environ["COLUMNS"] = "100"
_epilog = _cli_module._commands_epilog()
check("isaacli uninstall --purge --kaggle" in _ANSI.sub("", _epilog)
      and "authentication files" in _epilog,
      "wrapping the commands section keeps every command and its whole description")

# A Kaggle model row runs to 138 characters and the menu counts every option as
# one screen line. On an 80 column terminal the terminal wraps them itself, so
# the window the menu thinks it is drawing is not the window on screen: rows
# scroll off the top and the "more below" count is wrong.
_long_row = (
    "Qwen3 30B A3B Instruct 2507, Q4_K_M \u00b7 17.28 GiB \u00b7 T4 x2, 2 x 16 GB \u00b7 "
    "[reviewed here] Aider Polyglot 55.1, LiveCodeBench v6 45.2, GPQA 68.4")
_fitted = terminal_ui.fit(_long_row, 80)
check(len(_fitted) <= 80 and _fitted.startswith("Qwen3 30B A3B Instruct 2507"),
      "a row too wide for the terminal is cut on purpose, keeping what names it")
check(terminal_ui.fit("short row", 80) == "short row",
      "a row that already fits is left exactly as it is")
check(all(len(terminal_ui.fit(_long_row, width)) <= width
          for width in (20, 40, 61, 79, 80, 200)),
      "the cut holds at every width, including the narrow ones")

# The menu writes a cursor and two spaces before each row, so the budget is the
# width minus that prefix: a row that fits the terminal exactly still wraps once
# the cursor is in front of it.
_rendered = terminal_ui.option_lines(
    [_long_row, "short row"], width=80, cursor=0, disabled=set())
check(all(len(re.sub(r"\x1b\[[0-9;]*m", "", line)) <= 80 for line in _rendered)
      and len(_rendered) == 2,
      "each option is drawn as exactly one line that fits the terminal")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
# Two READMEs are two chances to describe a program that no longer exists. The
# text cannot be compared, but the shape can: a section added to one and not the
# other is the way they drift, and it is silent.
readmes = {
    name: [line for line in (HERE.parent / name).read_text(encoding="utf-8").splitlines()
           if line.startswith("#")]
    for name in ("README.md", "README.pt-BR.md")
}
check(len(readmes["README.md"]) == len(readmes["README.pt-BR.md"])
      and [line.split(" ", 1)[0] for line in readmes["README.md"]]
      == [line.split(" ", 1)[0] for line in readmes["README.pt-BR.md"]],
      "the two READMEs still have the same sections at the same depth")

# Sections were the only shape being compared, and drift does not arrive as a
# missing section. It arrives as a feature listed in one language and not the
# other, which is exactly how the AGENTS.md bullet was added on 2026-08-21: by
# hand, in both, with nothing that would have noticed had it landed in one.
readme_bodies = {
    name: (HERE.parent / name).read_text(encoding="utf-8")
    for name in ("README.md", "README.pt-BR.md")
}
readme_bullets = {name: [line for line in text.splitlines()
                         if line.startswith("- ")]
                  for name, text in readme_bodies.items()}
check(len(readme_bullets["README.md"]) == len(readme_bullets["README.pt-BR.md"]),
      "the two READMEs list the same number of bullet points")

# The label is translated, the destination is not. A link that exists in one
# README and not the other is a promise made to only half the readers, and a
# target that points nowhere is worse in either language.
readme_links = {
    # Each README links to the other on purpose, and that is the one target
    # that is supposed to differ, so it is the one target excluded here.
    name: sorted(target for target in re.findall(r"\]\(([^)]+)\)", text)
                 if target not in ("README.md", "README.pt-BR.md"))
    for name, text in readme_bodies.items()
}
check(readme_links["README.md"] == readme_links["README.pt-BR.md"],
      "the two READMEs point at exactly the same targets")
missing_targets = sorted(
    target for target in readme_links["README.md"]
    if not target.startswith(("http://", "https://", "#"))
    and not (HERE.parent / target.split("#", 1)[0]).exists()
)
check(not missing_targets,
      f"every local link in the READMEs resolves: {missing_targets}")

print("ISAAC CLI OK: workspace, model and basic output without Ollama")
