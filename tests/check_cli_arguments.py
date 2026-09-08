#!/usr/bin/env python3
"""Every command line the program answers, driven one by one through main().

The slash commands got this sweep in check_commands.py. The other half of the
surface, `isaacli <words and flags>`, never had one: nothing drove `isaacli
kaggle --stop`, `isaacli uninstall --purge --ollama` or `isaacli --model X
"do something"` and asserted where they landed. The owner said it plainly:
he has not typed all of them either, so nothing here can be closed by reading.

"It did not crash" answers nothing, the same way it answered nothing there.
Every branch in main() ends in `return <code>`, so a line that matched the
wrong branch, or matched none and fell into the usage message, returns just
as quietly as one that worked. What is asserted instead:

1. The table closes in both directions. Every command line the code answers is
   documented in the list `--help` prints, and every documented line is
   answered by a branch. This is what caught `uninstall --purge --llamacpp`
   existing in the code and in the usage string while being absent from the
   only list a user actually reads.
2. Driven for real, each line reaches its own branch and no other. The branch
   is proven by a stub that records the call, not by the exit code, because
   several branches return 0 and 0 says nothing about which one ran.
3. A line that is nearly right is refused with the usage message rather than
   running something close to it. Wrong order, an extra word, two flags that do
   not combine.
4. The destructive ladder asks before acting, and declining removes nothing.
5. --debug works from any position without changing which branch matches.
"""
import ast
import builtins
import io
import os
import signal
import sys
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

root = Path(tempfile.mkdtemp())
os.environ["XDG_CONFIG_HOME"] = str(root / "config-home")
os.environ["XDG_DATA_HOME"] = str(root / "data-home")
os.environ["HOME"] = str(root / "home")
# The request path reads this before the profile, and the owner's real
# environment having it set would send those cases down another branch.
os.environ.pop("ISAACLI_MODEL", None)

import cli as app  # noqa: E402
import cli_kaggle  # noqa: E402
import config  # noqa: E402
import debug  # noqa: E402
import setup_ollama  # noqa: E402
from cli_i18n import t  # noqa: E402

failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


# --- 1. the table closes in both directions ------------------------------

def documented_lines():
    """The command lines `--help` prints, read from the catalogue it renders."""
    lines = set()
    for row in t("cli.args.commands").split("\n"):
        words = row.split("|", 1)[0].split()
        if words and words[0] == "isaacli":
            lines.add(tuple(words[1:]))
    return lines


def answered_lines(source):
    """The command lines main() matches, read from the source of the matching.

    Two shapes, both literal: `arguments[0] == "kaggle"` names a command word,
    and `arguments[1:] == ["--purge", "--ollama"]` names the flags that follow
    it. Read rather than listed by hand, because a list kept by hand is the
    thing that goes stale, which is the whole point of this file.
    """
    tree = ast.parse(source)
    main = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    words = set()
    flag_groups = set()
    for node in ast.walk(main):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], ast.Eq):
            continue
        target, value = node.left, node.comparators[0]
        if not isinstance(target, ast.Subscript):
            continue
        if not (isinstance(target.value, ast.Name)
                and target.value.id == "arguments"):
            continue
        if isinstance(target.slice, ast.Constant) and target.slice.value == 0:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                words.add(value.value)
        elif isinstance(target.slice, ast.Slice) and isinstance(value, ast.List):
            items = [item.value for item in value.elts
                     if isinstance(item, ast.Constant)]
            if len(items) == len(value.elts):
                flag_groups.add(tuple(items))
    return words, flag_groups


source = (HERE.parent / "tool_harness" / "cli.py").read_text(encoding="utf-8")
command_words, flag_groups = answered_lines(source)
documented = documented_lines()

# A bare command word is a line of its own, and so is that word with each group
# of flags the code matches after it. Which word a group belongs to is decided
# by the documentation, because the code's matching is written as two separate
# conditions; a group nobody documents is reported against every word, which is
# exactly the report wanted: nowhere does it appear.
answered = {(word,) for word in command_words}
for word in command_words:
    for group in flag_groups:
        candidate = (word,) + group
        if candidate in documented:
            answered.add(candidate)
undocumented_groups = sorted(
    group for group in flag_groups
    if not any((word,) + group in documented for word in command_words))

missing_from_help = sorted(" ".join(line) for line in documented - answered)
check(not missing_from_help,
      f"every command line --help documents is matched by a branch: "
      f"{missing_from_help}")

check(not undocumented_groups,
      f"every flag combination the code answers is in the list --help prints: "
      f"{[' '.join(group) for group in undocumented_groups]}")


# --- 2 to 5. driven for real ---------------------------------------------

calls = []


class FakeCLI:
    """Stands in for IsaacCLI so the request path can be driven with no model.

    Constructed with what main() decided, which is the thing worth asserting:
    the model it resolved, the workspace, and the step ceiling.
    """

    def __init__(self, model, workspace, max_steps, **kwargs):
        self.model = model
        self.workspace = workspace
        self.max_steps = max_steps
        self.kwargs = kwargs
        self.history = []
        self.kaggle_profile = None
        calls.append(("IsaacCLI", model, str(workspace), max_steps))

    def _provider_from_profile(self, _profile):
        return None

    def ask(self, request):
        calls.append(("ask", request))
        return 0

    def repl(self):
        calls.append(("repl",))
        return 0


def recorder(name, code=0):
    def record(*args, **kwargs):
        calls.append((name, args, kwargs))
        return code
    return record


config_file = root / "config-home" / "isaacli" / "config.json"
config.save({"language": "en", "profiles": {}, "default_profile": None},
            config_file)

app.IsaacCLI = FakeCLI
app._install_launcher = recorder("install_launcher")
app._uninstall_launcher = recorder("uninstall_launcher")
app._uninstall_official_ollama = recorder("uninstall_ollama")
app._uninstall_managed_kaggle = recorder("uninstall_kaggle")
app._uninstall_managed_llamacpp = recorder("uninstall_llamacpp")
app._offer_remote_cleanup = recorder("offer_remote_cleanup", {})
app._close_without_interruption = lambda _cli: None
app._kaggle_release_session = recorder("kaggle_release")
app._kaggle_stop_session = recorder("kaggle_stop_session")
app._kaggle_ensure_session = recorder("kaggle_ensure", "absent")
def fake_run_setup():
    """Saves a profile, because a setup that saved nothing is a different case.

    `isaacli setup` does not end at the setup screen: it goes on to open the
    session the screen just configured, and that second half is what a stub
    returning a bare 0 would hide. Writing the profile the real screen writes
    is what lets this file assert the whole line rather than its first half.
    """
    calls.append(("run_setup", (), {}))
    config.save({"language": "en",
                 "profiles": {"probe": {"model": "probe-model"}},
                 "default_profile": "probe"}, config_file)
    return 0


setup_ollama.run_setup = fake_run_setup
setup_ollama.run_kaggle = recorder("run_kaggle")
setup_ollama.run_prepare_assets = recorder("run_prepare_assets")
cli_kaggle.run_stop_kernels = recorder("run_stop_kernels")


class Slow(Exception):
    pass


def _ring(_signum, _frame):
    raise Slow()


signal.signal(signal.SIGALRM, _ring)
LINE_CEILING_SECONDS = 20

prompts = []


def drive(argv, answers=()):
    """Run one command line and report what it did, never what it printed only.

    Returns the exit code, the names of the branches that recorded a call, and
    the screen, so an assertion can name the branch instead of trusting a 0.
    """
    del calls[:]
    del prompts[:]
    typed = list(answers)

    def fake_input(prompt=""):
        # Recorded rather than printed: input() writes its prompt straight to
        # the real stdout, so a check reading the screen for the question would
        # never find it and would pass on a program that never asked.
        prompts.append(prompt)
        return typed.pop(0) if typed else ""

    original_input = builtins.input
    builtins.input = fake_input
    screen = io.StringIO()
    # A ceiling, for the same reason check_commands.py has one: a branch that
    # reaches the network instead of its stub would otherwise hang the suite
    # with no diagnosis, and "it hung" has to arrive as a named failure.
    signal.setitimer(signal.ITIMER_REAL, LINE_CEILING_SECONDS)
    try:
        with redirect_stdout(screen), redirect_stderr(screen):
            code = app.main(argv)
    except SystemExit as exit_request:  # argparse leaves this way
        code = exit_request.code
    except Slow:
        code = f"took longer than {LINE_CEILING_SECONDS}s"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        builtins.input = original_input
    return code, [call[0] for call in calls], screen.getvalue(), list(calls)


USAGE_MARKER = t("cli.args.commands_header")


def routes(argv, branch, answers=(), expected_code=0):
    code, names, screen, _detail = drive(argv, answers)
    check(branch in names and code == expected_code,
          f"`isaacli {' '.join(argv)}` reaches {branch} "
          f"(reached {names or 'nothing'}, exit {code})")
    return screen


def refused(argv):
    code, names, screen, _detail = drive(argv)
    check(code == 2 and not names,
          f"`isaacli {' '.join(argv)}` is refused with the usage message "
          f"rather than running something close to it "
          f"(exit {code}, reached {names or 'nothing'})")
    check(USAGE_MARKER in screen,
          f"`isaacli {' '.join(argv)}` shows the command list when refusing")


EMPTY_CONFIG = {"language": "en", "profiles": {}, "default_profile": None}

_code, names, _screen, _detail = drive(["setup"])
check(names[:1] == ["run_setup"] and "repl" in names,
      f"`isaacli setup` runs the setup screen and then opens the session it "
      f"just configured: {names}")
config.save(EMPTY_CONFIG, config_file)

# A cancelled setup must not open a model anyway. This is the branch that
# exists so a cancellation is not answered by picking something.
setup_ollama.run_setup = recorder("run_setup", 130)
code, names, _screen, _detail = drive(["setup"])
check(code == 130 and "IsaacCLI" not in names,
      f"cancelling the setup screen ends the program rather than opening a "
      f"model nobody chose (exit {code}, {names})")
setup_ollama.run_setup = recorder("run_setup", 1)
code, names, _screen, _detail = drive(["setup"])
check(code == 1 and "IsaacCLI" not in names,
      f"a setup that failed is not hidden by opening a model anyway "
      f"(exit {code}, {names})")
setup_ollama.run_setup = fake_run_setup
config.save(EMPTY_CONFIG, config_file)

routes(["install"], "install_launcher")
routes(["kaggle"], "run_kaggle")
routes(["kaggle", "--prepare-assets"], "run_prepare_assets")
routes(["kaggle", "--stop"], "run_stop_kernels")
routes(["uninstall"], "uninstall_launcher")

# The flag that decides whether a GPU is requested. Routing to run_kaggle is not
# enough here: both lines route there, and only the argument separates them.
_code, _names, _screen, detail = drive(["kaggle", "--flow-validation-cpu"])
kaggle_calls = [call for call in detail if call[0] == "run_kaggle"]
check(bool(kaggle_calls) and kaggle_calls[0][2].get("validation_cpu") is True,
      f"`isaacli kaggle --flow-validation-cpu` asks for the CPU flow rather "
      f"than a GPU one: {kaggle_calls}")
_code, _names, _screen, detail = drive(["kaggle"])
kaggle_calls = [call for call in detail if call[0] == "run_kaggle"]
check(bool(kaggle_calls) and kaggle_calls[0][2].get("validation_cpu") is False,
      f"plain `isaacli kaggle` is not the CPU validation flow: {kaggle_calls}")

# 3. nearly right is refused, not approximated.
refused(["setup", "now"])
refused(["install", "--user"])
refused(["kaggle", "--sto"])
refused(["kaggle", "--stop", "--prepare-assets"])
refused(["uninstall", "--ollama", "--purge"])
refused(["uninstall", "--purge", "--everything"])
refused(["uninstall", "--purge", "--ollama", "--kaggle"])

# 4. the destructive ladder asks first, and declining removes nothing.
YES = t("cli.uninstall.confirm_yes")
for flags, branch in (
    (["--purge"], "uninstall_launcher"),
    (["--purge", "--ollama"], "uninstall_ollama"),
    (["--purge", "--kaggle"], "uninstall_kaggle"),
    (["--purge", "--llamacpp"], "uninstall_llamacpp"),
):
    argv = ["uninstall"] + flags
    screen = routes(argv, branch, answers=[YES])
    asked = list(prompts)
    check(any(t("cli.uninstall.confirm").strip()[:20] in prompt
              for prompt in asked),
          f"`isaacli {' '.join(argv)}` asks before removing anything: {asked}")
    check(t("cli.uninstall.purge_warning")[:20] in screen
          or any(t(key)[:20] in screen for key in (
              "cli.uninstall.ollama.warning", "cli.uninstall.kaggle.warning",
              "cli.uninstall.llamacpp.warning")),
          f"`isaacli {' '.join(argv)}` says what it is about to delete before "
          f"asking")
    code, names, _screen, _detail = drive(argv, answers=["no"])
    check(code == 130 and not names,
          f"declining `isaacli {' '.join(argv)}` removes nothing "
          f"(exit {code}, reached {names or 'nothing'})")

# The purge that runs is the purge that was asked for. `uninstall` alone must
# not delete config and sessions, and every rung of the ladder must.
_code, _names, _screen, detail = drive(["uninstall"])
launcher = [call for call in detail if call[0] == "uninstall_launcher"]
check(bool(launcher) and launcher[0][2].get("purge") is False,
      f"plain `isaacli uninstall` keeps config, sessions and feedback: "
      f"{launcher}")
for flags in (["--purge"], ["--purge", "--ollama"], ["--purge", "--kaggle"],
              ["--purge", "--llamacpp"]):
    _code, _names, _screen, detail = drive(["uninstall"] + flags,
                                           answers=[YES])
    final = [call for call in detail if call[0] == "uninstall_launcher"][-1:]
    check(bool(final) and final[0][2].get("purge") is True,
          f"`isaacli uninstall {' '.join(flags)}` purges the local data: "
          f"{final}")

# 5. --debug from either end, and it must not change the match.
for argv in (["--debug", "kaggle", "--stop"], ["kaggle", "--stop", "--debug"]):
    debug.enable(False)
    code, names, _screen, _detail = drive(argv)
    check("run_stop_kernels" in names and code == 0 and debug.ENABLED,
          f"`isaacli {' '.join(argv)}` still stops the session, with reporting "
          f"on (reached {names or 'nothing'}, debug={debug.ENABLED})")
debug.enable(False)

# The request path: what main() resolved is what the session is built with.
workspace = root / "project"
workspace.mkdir()
_code, names, _screen, detail = drive(
    ["--model", "probe-model", "--workspace", str(workspace),
     "--max-steps", "3", "explain", "this"])
built = [call for call in detail if call[0] == "IsaacCLI"]
asked = [call for call in detail if call[0] == "ask"]
check(built == [("IsaacCLI", "probe-model", str(workspace), 3)],
      f"a request line builds the session with the model, workspace and step "
      f"ceiling it was given: {built}")
check(asked == [("ask", "explain this")],
      f"the words after the flags are the request, joined back together: "
      f"{asked}")

_code, _names, _screen, detail = drive(["--dir", str(workspace),
                                        "--model", "probe-model", "hi"])
built = [call for call in detail if call[0] == "IsaacCLI"]
check(built and built[0][2] == str(workspace),
      f"--dir is the same flag as --workspace: {built}")

_code, names, _screen, _detail = drive(["--model", "probe-model"])
check("repl" in names,
      f"a line with no request opens the session rather than asking nothing: "
      f"{names}")

code, names, screen, _detail = drive(["--resume", "does-not-exist", "hello"])
check(code == 2 and not names,
      f"--resume with a request is refused, because the two say different "
      f"things about where the history comes from (exit {code}, {names})")

code, names, screen, _detail = drive(["--resume", "does-not-exist"])
check(code == 2 and "IsaacCLI" not in names,
      f"resuming a session that does not exist reports instead of opening a "
      f"fresh one that pretends to be it (exit {code}, {names})")

code, _names, screen, _detail = drive(["--version"])
check(app.APP_VERSION in screen and code in (0, None),
      f"--version prints the version the program was built as: {screen.strip()}")

code, _names, screen, _detail = drive(["--nope"])
check(code == 2 and USAGE_MARKER in screen,
      f"an unknown flag shows the command list next to the error rather than "
      f"leaving the user guessing (exit {code})")

# The refusal marker has to be a real one, or every `refused` above is a check
# that cannot fail.
check(USAGE_MARKER.strip() != "" and USAGE_MARKER not in drive(
    ["kaggle", "--stop"])[2],
      "the usage marker is absent from a line that works, so finding it means "
      "something")

real_home = Path(os.path.expanduser("~"))
check(str(real_home).startswith(str(root)),
      f"this sweep runs against a throwaway home, not the owner's: {real_home}")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)
print(f"CLI ARGUMENTS OK: {len(documented)} documented command lines, "
      f"{len(command_words)} command words and {len(flag_groups)} flag "
      f"combinations answered, each driven through main()")
