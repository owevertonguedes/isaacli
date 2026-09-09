#!/usr/bin/env python3
"""Slash commands driven past the screen they open, to the effect they have.

`check_commands.py` sweeps every command and stops the moment one draws a
selector. That was deliberate and it is still right for a sweep: there is no row
that is safe to pick on every screen, and inventing a rule those screens do not
follow would assert a program that does not exist. But stopping there proves
only that a screen appeared. The owner's question was the other half, and he
said so plainly: he has not tested the commands, so it is likely they are not
doing what they should.

So this file is the other half, and it is a table rather than a sweep: one entry
per screen, the row named rather than numbered, and an assertion about what
changed on disk or in the session afterwards. A row picked by number is not an
answer to a screen whose rows come from somewhere else, and every check here
would keep passing while pointing at the wrong row.

Where it stops, and why: nothing here picks a row that would install anything,
spend anybody's GPU quota, or reach the network. Those rows are named in the
table with the reason, so what is left out is visible instead of implied.
"""
import builtins
import io
import json
import os
import signal
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

import cli as app  # noqa: E402
import config  # noqa: E402
import terminal_ui  # noqa: E402
from i18n import SUPPORTED_LANGUAGES  # noqa: E402

failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


class Slow(Exception):
    pass


def _ring(_signum, _frame):
    raise Slow()


signal.signal(signal.SIGALRM, _ring)
SCREEN_CEILING_SECONDS = 20

root = Path(tempfile.mkdtemp(prefix="isaacli-screens-"))
# Before anything imports a module that reads them. These commands write
# configuration for a living, and a check that drives them against the real
# home is a check that edits the machine it runs on.
os.environ["XDG_CONFIG_HOME"] = str(root / "config-home")
os.environ["XDG_DATA_HOME"] = str(root / "data-home")
os.environ["HOME"] = str(root / "home")
workspace = root / "project"
workspace.mkdir()
config_file = root / "config-home" / "isaacli" / "config.json"
config.save({"language": "en", "profiles": {}, "default_profile": None},
            config_file)


class RowMissing(Exception):
    """The named row was not on the screen, which is the finding, not a crash."""


class Chooser:
    """Answers one screen by naming the row, and records every screen it saw.

    The row is found by its rendered text, so a screen that reorders its rows
    still gets the same answer, and a screen that stops offering the row fails
    loudly here instead of quietly picking its neighbour. Matching is on a
    substring because several rows carry their current value after the label.

    A name that fits more than one row is refused rather than resolved to the
    first, and that rule was written after being caught by it: answering the
    context screen with "Off" matched "**Off**er to compact" one row above the
    row meant, and the check that followed reported the program saving the
    opposite of what was chosen. The program was right and the answer was
    wrong, which is the failure that costs the most, because it reads exactly
    like a defect.
    """

    def __init__(self, *wanted):
        self.wanted = list(wanted)
        self.seen = []

    def __call__(self, title, options, **_kwargs):
        rendered = [str(option) for option in options]
        self.seen.append((str(title), rendered))
        if not self.wanted:
            raise RowMissing(
                f"the screen {title!r} was drawn with nothing left to answer it")
        want = self.wanted.pop(0)
        hits = [index for index, option in enumerate(rendered)
                if want.lower() in option.lower()]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise RowMissing(f"no row matching {want!r} on {title!r}: {rendered}")
        raise RowMissing(
            f"{want!r} matches {len(hits)} rows on {title!r}, so it names none "
            f"of them: {[rendered[index] for index in hits]}")


def drive(command, chooser, cli=None):
    """Run one command with that chooser, returning the session and the screen."""
    cli = cli or app.IsaacCLI("probe-model", workspace, 4,
                              autostart_ollama=False, config_file=config_file)
    original_select = terminal_ui.select
    original_input = builtins.input
    terminal_ui.select = chooser
    builtins.input = lambda _prompt="": ""
    screen = io.StringIO()
    error = None
    signal.setitimer(signal.ITIMER_REAL, SCREEN_CEILING_SECONDS)
    try:
        with redirect_stdout(screen):
            cli.internal_command(command)
    except (RowMissing, Slow) as raised:
        error = f"{type(raised).__name__}: {raised}"
    except Exception as raised:  # noqa: BLE001 - reported, never raised onward
        # Reported as this entry's result. A file that dies on the second screen
        # says nothing about the screens after it, which is the whole point of
        # having a table.
        error = f"{type(raised).__name__}: {raised}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal_ui.select = original_select
        builtins.input = original_input
    return cli, screen.getvalue(), error


def saved():
    return json.loads(config_file.read_text(encoding="utf-8"))


# --- /language: the row picked has to be the language saved ---------------
#
# The one screen in the program whose effect changes every screen after it, so
# it is also the one where picking the wrong row is least visible: the config
# still holds a language, the confirmation still names one, and only the next
# screen tells you it was the wrong one.
PT_LABEL = SUPPORTED_LANGUAGES["pt-BR"]
EN_LABEL = SUPPORTED_LANGUAGES["en"]

chooser = Chooser(PT_LABEL)
session, screen, error = drive("/language", chooser)
check(error is None, f"/language reaches its screen and answers it: {error}")
check(chooser.seen and PT_LABEL in " ".join(chooser.seen[0][1]),
      f"the language screen offers the language by name: "
      f"{chooser.seen[0][1] if chooser.seen else 'no screen'}")
check(saved().get("language") == "pt-BR",
      f"picking a language by name saves that language: {saved().get('language')}")
check(PT_LABEL in screen,
      "and the confirmation names the language that was picked")

# The next screen, which is where mechanism one of the handoff lives: checking
# the layer you touched is not the same as following the data into the screen
# after it. A language that is saved and not applied looks identical here until
# somebody opens a different menu.
#
# The language screen itself is the wrong place to look, and this file asserted
# the wrong thing before checking: `cli.language.title` is deliberately the same
# string in both catalogues, because somebody who cannot read the language in
# use still has to recognise the screen that changes it. So the question is put
# to the next screen that does translate.
from cli_i18n import Translator  # noqa: E402

after = Chooser("Fechar")
_session, _out, after_error = drive("/config", after)
after_titles = [title for title, _rows in after.seen]
check(after_error is None and bool(after_titles)
      and Translator("pt-BR").t("cli.config.title") in after_titles[0]
      and Translator("en").t("cli.config.title") not in after_titles[0],
      f"the next screen is drawn in the language just chosen: {after_titles[:1]}")
check(any("Fechar" in row for _title, rows in after.seen for row in rows),
      "including the row that closes it, which is how it was answered at all")

back = Chooser(EN_LABEL)
_session, back_screen, back_error = drive("/language", back)
check(back_error is None and saved().get("language") == "en",
      f"and it goes back the same way: {saved().get('language')} ({back_error})")


# --- /config: the closing row closes, and changes nothing -----------------
#
# Every other row on this screen opens a second one, and this is the row that
# has to be a no-op. A closing row that saved something would be invisible: the
# screen it leaves behind says "unchanged" either way.
before_config = saved()
closing = Chooser("close")
_session, config_screen_out, config_error = drive("/config", closing)
check(config_error is None,
      f"/config reaches its screen and its closing row answers it: {config_error}")
check(saved() == before_config,
      "closing the preferences screen writes nothing, so the row is the no-op it looks like")
check(bool(closing.seen) and len(closing.seen[0][1]) > 1,
      f"the preferences screen is drawn with its settings, not just a way out: "
      f"{len(closing.seen[0][1]) if closing.seen else 0} rows")


# --- /config: a setting driven two screens deep, to what it writes --------
#
# One row deep rather than none, and then out through the closing row, which is
# the whole path a person takes. The effect asserted is the value on disk, not
# the sentence left on the screen: a screen that draws the right row and saves
# the other one reads identically from the outside.
context_rows = Chooser("Context management", "Off:", "close")
_session, _out, context_error = drive("/config", context_rows)
check(context_error is None,
      f"a preferences row opens a screen of its own and answers it: {context_error}")
check(len(context_rows.seen) >= 3,
      f"the path is menu, setting, menu again: "
      f"{[title.splitlines()[0] for title, _rows in context_rows.seen]}")
check(saved().get("context_management") is False,
      f"choosing Off writes Off, which is the only part of this a screen cannot "
      f"fake: {saved().get('context_management')}")
back_on = Chooser("Context management", "On:", "close")
_session, _out, back_on_error = drive("/config", back_on)
check(back_on_error is None and saved().get("context_management") is True,
      f"and On writes On: {saved().get('context_management')} ({back_on_error})")


# --- /mode: no screen, but an effect nobody was checking ------------------
#
# It draws nothing at all, which is why the sweep that stops at the first
# selector could never say anything about it. What it does is flip the state
# that decides whether a command runs without being asked about, so it is worth
# more than most of the screens here.
mode_cli = app.IsaacCLI("probe-model", workspace, 4, autostart_ollama=False,
                        config_file=config_file)
started_as = mode_cli.permission_mode
_session, _out, mode_error = drive("/mode", Chooser(), cli=mode_cli)
flipped = mode_cli.permission_mode
_session, _out, mode_error_back = drive("/mode", Chooser(), cli=mode_cli)
check(mode_error is None and mode_error_back is None,
      f"/mode answers without a screen: {mode_error or mode_error_back}")
check(flipped != started_as and mode_cli.permission_mode == started_as,
      f"/mode really flips the permission mode and flips it back: "
      f"{started_as} to {flipped} to {mode_cli.permission_mode}")
check(flipped in {"safe", "authorized_only"},
      f"and it lands on a mode the sandbox knows: {flipped}")


# --- What is deliberately not driven, named rather than implied -----------
#
# `/model` and `/setup` open on a source screen whose rows install Ollama, build
# llama.cpp, or push a Kaggle kernel that spends the owner's weekly GPU budget.
# `/kaggle` is the same screen with the spending one row closer. Picking any of
# those here would make the suite buy something. They are covered where the cost
# can be stubbed: check_setup.py drives the same screens through
# `source_answer`, which resolves the row by name out of the very function the
# screen is built from.
# `/model` and `/setup` open on a source screen whose rows install Ollama, build
# llama.cpp, or push a Kaggle kernel that spends the owner's weekly GPU budget,
# and `/kaggle` is that screen with the spending one row closer. Picking any of
# them here would make the suite buy something. They are driven in
# check_setup.py, where every cost is a stub and the row is resolved by name out
# of the same function the screen is built from. This is a note rather than a
# check, because a check that reads another file to see whether it mentions a
# helper proves nothing about either.

print()
if failures:
    print(f"{len(failures)} check(s) failed")
    raise SystemExit(1)
print("all command screen checks passed")
