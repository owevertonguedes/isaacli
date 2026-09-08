#!/usr/bin/env python3
"""Every slash command, driven one by one through the door they all enter by.

The question this answers is the owner's: "is each command working as it should,
or is the logic wrong?" The trap is that "it did not crash" answers nothing. Two
defects reached his screen with the whole suite green, and both of them
answered: a screen hid Ollama on a machine without Ollama, and a list of ten
GGUF files came back holding one. Neither raised.

So the assertion here is not that a command returns. `internal_command` returns
True for a command nobody implemented too, because its last line says "unknown
command" and returns True, and that is indistinguishable from success at the
call site. What is asserted instead:

1. The table closes in both directions. Every name offered by the completer
   reaches a branch, and every name a branch answers to is offered.
2. Driven for real, no command lands on the unknown-command line.
3. Nothing takes longer than its ceiling, so one hanging command reports rather
   than hanging the suite with no diagnosis.
4. Nothing is written outside the throwaway home this file creates.
"""
import ast
import builtins
import io
import os
import re
import signal
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

root = Path(tempfile.mkdtemp())
# Before importing the CLI, exactly as check_cli.py does: config, session logs
# and feedback all resolve from these, and a check that reads the owner's real
# ones can be moved by a file of his that has nothing to do with it.
os.environ["XDG_CONFIG_HOME"] = str(root / "config-home")
os.environ["XDG_DATA_HOME"] = str(root / "data-home")
os.environ["HOME"] = str(root / "home")

import cli as app
import cli_commands
import config
import setup_ollama
import terminal_ui

# The model screen resolves each hf.co entry live, three requests apiece. With
# the network away those wait out their ceiling one after another, which is a
# screen taking minutes rather than a defect, and it is not what this file is
# measuring. Stubbed the way check_setup.py stubs it, so the sweep exercises the
# screen's own logic offline.
setup_ollama._resolve_live = lambda *_args, **_kwargs: None

failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


# --- 1. the table closes in both directions -----------------------------

def answered_names(source):
    """Every literal the dispatcher compares the typed command against.

    Read from the function rather than from a list somebody keeps by hand,
    because a list kept by hand is the thing that goes stale. Both shapes count:
    `cmd == "/status"` and `cmd in ("/exit", "/quit")`.
    """
    names = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        for operator, comparator in zip(node.ops, node.comparators):
            if isinstance(operator, ast.Eq) and isinstance(comparator, ast.Constant):
                if isinstance(comparator.value, str) and comparator.value.startswith("/"):
                    names.add(comparator.value)
            if isinstance(operator, ast.In) and isinstance(comparator, (ast.Tuple, ast.List)):
                for item in comparator.elts:
                    if (isinstance(item, ast.Constant)
                            and isinstance(item.value, str)
                            and item.value.startswith("/")):
                        names.add(item.value)
    return names


dispatcher = (HERE.parent / "tool_harness" / "cli_commands.py").read_text(
    encoding="utf-8")
answered = answered_names(dispatcher)
offered = set(cli_commands.SLASH_COMMANDS)
aliases = set(cli_commands.COMMAND_ALIASES)
# `/` and `/?` are answered before the dispatch table and are not offered by the
# completer on purpose: one is the empty prompt, the other is help's short form.
extra = {"/", "/?", "/quit"}

unreachable = sorted(offered - answered)
check(not unreachable,
      f"every command the completer offers reaches a branch: {unreachable}")

hidden = sorted(answered - offered - aliases - extra)
check(not hidden,
      f"every command with a branch is offered by the completer: {hidden}")

alias_targets = sorted(set(cli_commands.COMMAND_ALIASES.values()) - offered)
check(not alias_targets,
      f"every hidden alias points at a command that exists: {alias_targets}")


# --- 2 and 3. driven for real, each under its own ceiling ----------------

class Slow(Exception):
    pass


def _ring(_signum, _frame):
    raise Slow()


signal.signal(signal.SIGALRM, _ring)
COMMAND_CEILING_SECONDS = 20

workspace = root / "project"
workspace.mkdir()

config_file = root / "config-home" / "isaacli" / "config.json"
config.save({"language": "en", "profiles": {}, "default_profile": None},
            config_file)

# The unknown-command line is what a command that reaches nothing prints, and it
# is the only thing that separates "handled" from "handled by falling off the
# end". Rendered from the catalogue rather than typed, so it stays true in
# either language.
from cli_i18n import t  # noqa: E402  (after the path insert above)

# The part before the command name, not the sentence with the name cut out of
# it: removing the name from the middle leaves a double space that matches
# nothing, and a marker that matches nothing is a check that cannot fail.
UNKNOWN_MARKER = t("cli.unknown_command", cmd="__PROBE__").split("__PROBE__")[0].strip()


class OpenedScreen(Exception):
    """Raised the moment a command draws its first selector.

    There is no row that is safe to pick on every screen, and this file found
    that out by trying: answering with the last row walked the model screen into
    "configure a new API", which asked for a name at the keyboard; answering
    with the Back row does not work either, because /config closes with "Close"
    and /language, /setup, /kaggle and /model each open on a screen that is the
    first one and has nowhere to go back to.

    Inventing a rule those screens do not follow would be asserting a program
    that does not exist. So a command that opens a screen is stopped at the
    moment it opens, which still proves the thing worth proving: it reached its
    own screen rather than the unknown-command line, and the screen it reached
    is drawn with real options.
    """

    def __init__(self, title, options):
        super().__init__(title)
        self.title = title
        self.options = [str(option) for option in options]


def screen_stub(title, options, **_kwargs):
    raise OpenedScreen(title, options)


def backing_input(_prompt=""):
    """An empty line, for anything that reads the keyboard without a selector.

    It declines every prompt in this program and changes nothing. Most screens
    never reach it now, because they are stopped at their first selector, but a
    command that asks before drawing anything would otherwise read a stdin that
    is not there and end the sweep on an EOFError.
    """
    return ""


results = {}
screens = {}
for command in cli_commands.SLASH_COMMANDS:
    if command == "/exit":
        # Documented to end the program by raising, so returning at all would be
        # the defect. Asserted below rather than driven with the rest.
        continue
    cli = app.IsaacCLI("probe-model", workspace, 4, autostart_ollama=False,
                       config_file=config_file)
    original_select = terminal_ui.select
    original_input = builtins.input
    screen = io.StringIO()
    terminal_ui.select = screen_stub
    builtins.input = backing_input
    signal.setitimer(signal.ITIMER_REAL, COMMAND_CEILING_SECONDS)
    try:
        with redirect_stdout(screen):
            handled = cli.internal_command(command)
        results[command] = (handled, screen.getvalue(), None)
    except OpenedScreen as opened:
        screens[command] = opened
        results[command] = (True, screen.getvalue(), None)
    except Slow:
        results[command] = (None, screen.getvalue(),
                            f"took longer than {COMMAND_CEILING_SECONDS}s")
    except Exception as error:  # noqa: BLE001 - recorded, never raised onward
        # Reported as the result of this command rather than allowed to end the
        # file. A sweep that dies on command three proves nothing about the
        # nineteen after it, which is the whole reason for sweeping.
        results[command] = (None, screen.getvalue(),
                            f"{type(error).__name__}: {error}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal_ui.select = original_select
        builtins.input = original_input

raised = sorted(name for name, (_h, _o, error) in results.items() if error)
check(not raised,
      "no command raises or hangs when driven: "
      + ("none" if not raised else
         "; ".join(f"{name} ({results[name][2]})" for name in raised)))

not_handled = sorted(name for name, (handled, _o, error) in results.items()
                     if error is None and handled is not True)
check(not not_handled,
      f"every command answers as handled rather than falling through to the "
      f"model: {not_handled}")

fell_through = sorted(
    name for name, (_h, output, error) in results.items()
    if error is None and UNKNOWN_MARKER and UNKNOWN_MARKER.strip() in output)
check(not fell_through,
      f"no command lands on the unknown-command line, which also returns "
      f"handled: {fell_through}")

# The commands that open a screen are the ones the two known defects lived in,
# and both of them drew a screen that answered while being wrong. What is
# asserted here is what can be asserted without picking a row: the command
# reaches a screen of its own, and that screen is drawn with real text.
# Named, not counted. A count is a statement about the machine: `/kaggle` opens
# its screen where the Kaggle CLI is installed and answers with a message where
# it is not, so `>= 5` passed here and failed on a clean CI runner, which is CI
# doing its job. These four depend on nothing outside the program, so they must
# reach a screen wherever this runs.
ALWAYS_A_SCREEN = {"/config", "/language", "/model", "/setup"}
opened = sorted(screens)
missing_screen = sorted(ALWAYS_A_SCREEN - set(screens))
check(not missing_screen,
      f"the commands that own a screen reach it rather than a message: "
      f"missing {missing_screen}, opened {opened}")

empty_screens = sorted(
    name for name, screen_opened in screens.items()
    if not screen_opened.options
    or any(not option.strip() for option in screen_opened.options))
check(not empty_screens,
      f"no screen is drawn with an empty row: {empty_screens}")

# A catalogue key that reached the screen instead of its text. The key shape is
# dotted lowercase with no spaces, which no rendered label looks like.
key_shaped = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+){2,}$")
leaked_keys = sorted(
    f"{name}: {option}"
    for name, screen_opened in screens.items()
    for option in screen_opened.options
    if key_shaped.match(option.strip()))
check(not leaked_keys,
      f"no untranslated catalogue key reaches a screen: {leaked_keys}")

# The marker has to be a real one, or the check above is a check that cannot
# fail. Proven against a command nobody implements.
probe_cli = app.IsaacCLI("probe-model", workspace, 4, autostart_ollama=False,
                         config_file=config_file)
probe_screen = io.StringIO()
with redirect_stdout(probe_screen):
    probe_handled = probe_cli.internal_command("/definitely-not-a-command")
check(probe_handled is True
      and UNKNOWN_MARKER.strip() in probe_screen.getvalue(),
      "an unimplemented command is what the fall-through marker actually finds")

exit_raised = False
try:
    exit_cli = app.IsaacCLI("probe-model", workspace, 4, autostart_ollama=False,
                            config_file=config_file)
    with redirect_stdout(io.StringIO()):
        exit_cli.internal_command("/exit")
except EOFError:
    exit_raised = True
check(exit_raised, "/exit ends the program by raising rather than by returning")


# --- 4. isolation --------------------------------------------------------

# The package directory is where sessions and feedback used to be written, and
# a sweep that drives /good and /bad would fill the repository with them.
package = HERE.parent / "tool_harness"
strays = sorted(
    str(path.relative_to(package))
    for path in package.rglob("*")
    if path.is_file() and path.suffix == ".jsonl")
check(not strays,
      f"driving every command writes no session or feedback into the "
      f"package directory: {strays[:5]}")

real_home = Path(os.path.expanduser("~"))
check(str(real_home).startswith(str(root)),
      f"this sweep runs against a throwaway home, not the owner's: {real_home}")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)
print(f"COMMANDS OK: {len(results)} slash commands driven plus /exit asserted "
      f"on its own, {len(cli_commands.SLASH_COMMANDS)} in the table, "
      f"{len(screens)} of them reaching a screen")
