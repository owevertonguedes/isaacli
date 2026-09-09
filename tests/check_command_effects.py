#!/usr/bin/env python3
"""Slash commands driven to the effect they have, not to the moment they answer.

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

The commands that draw no screen at all are here for the same reason and are
worse off without it: a sweep that stops at the first selector could never say
anything about them, and "it returned handled" is what a command that reaches
nothing returns too. So they are driven for their effect as well, which for most
of them is a line written to a file.

Where it stops, and why: nothing here picks a row that would install anything,
spend anybody's GPU quota, or reach the network. Those rows are named in the
file with the reason, so what is left out is visible instead of implied.
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

# --- the commands that draw nothing, and write something ------------------
#
# Fourteen of them, and until now the only thing asserted about any of them was
# that they returned handled, which is also what a command that reaches nothing
# returns. Each one here is driven for the mark it leaves.
plain = app.IsaacCLI("probe-model", workspace, 4, autostart_ollama=False,
                     config_file=config_file)


def run(command, cli=plain):
    """One command, with no screen expected, returning what it printed."""
    return drive(command, Chooser(), cli=cli)[1]


# /tools is the only list the user ever sees of what the model may do on their
# machine, so a tool the agent exposes and this screen omits is a capability
# nobody was told about. Closed against the schema the agent is actually given.
import tools as tool_schema  # noqa: E402
import execution  # noqa: E402

tools_out = run("/tools")
missing_tools = sorted(
    schema["function"]["name"] for schema in tool_schema.SCHEMA
    if schema["function"]["name"] not in tools_out)
check(not missing_tools,
      f"/tools names every tool the model is given, so none is hidden from the "
      f"person whose machine it runs on: missing {missing_tools}")
missing_commands = sorted(name for name in execution.ALLOWED if name not in tools_out)
check(not missing_commands,
      f"and every terminal command the sandbox allows: missing {missing_commands}")

# /good, /bad and /score are the only commands that write a file the program
# reads back later, and none of them was checked past returning handled.
feedback_path = Path(plain.feedback_path)
before_lines = (feedback_path.read_text(encoding="utf-8").splitlines()
                if feedback_path.exists() else [])
run("/good it did the thing")
run("/bad it did not")
run("/score 7 about right")
written = [json.loads(line) for line in
           feedback_path.read_text(encoding="utf-8").splitlines()[len(before_lines):]]
check(len(written) == 3,
      f"three feedback commands write three records: {len(written)}")
check([record.get("score") for record in written] == [10, 0, 7],
      f"and each writes the score it stands for: "
      f"{[record.get('score') for record in written]}")
check([record.get("feedback_kind") for record in written] == ["good", "bad", "score"],
      f"named as what they are: {[record.get('feedback_kind') for record in written]}")
check(all(record.get("session_id") == plain.session_id for record in written),
      "each tied to the session it was given in, which is what makes it readable later")
check(written[0].get("comment") == "it did the thing",
      f"and the comment typed after the command is kept: {written[0].get('comment')}")

# A score that is not a number must not be filed as one. The file is what proves
# it: a refusal that still writes is indistinguishable on screen from one that
# does not.
before_junk = feedback_path.read_text(encoding="utf-8").splitlines()
junk_out = run("/score not-a-number")
after_junk = feedback_path.read_text(encoding="utf-8").splitlines()
check(len(after_junk) == len(before_junk),
      f"a score that is not a number writes nothing: "
      f"{len(after_junk) - len(before_junk)} line(s) added")
check(Translator("en").t("cli.score.not_integer").strip() in junk_out,
      "and refuses it by name rather than with any message at all")
# The same door on the other side: a number outside the scale is not a score.
before_range = feedback_path.read_text(encoding="utf-8").splitlines()
range_out = run("/score 44")
check(len(feedback_path.read_text(encoding="utf-8").splitlines()) == len(before_range)
      and Translator("en").t("cli.score.out_of_range").strip() in range_out,
      "a score outside the scale is refused and not filed either")

# /log is one line of output and it is a promise about the disk: the path it
# prints has to be the session actually being written.
log_out = run("/log").strip()
check(log_out == str(plain.session_path),
      f"/log prints the session path this session is writing: {log_out}")
check(Path(log_out).exists(),
      f"and that file is really there: {log_out}")

# /new has to move the session on, and must not take the old one with it.
previous_path = Path(plain.session_path)
previous_id = plain.session_id
run("/new")
check(plain.session_id != previous_id and Path(plain.session_path) != previous_path,
      f"/new starts a session that is not the one before it: "
      f"{previous_id} to {plain.session_id}")
check(previous_path.exists(),
      "and leaves the finished one on disk, because that is the record")

# /clear rebuilds the history rather than emptying it: the workspace
# instructions have to survive, or the next answer is given by a model that no
# longer knows where it is.
plain.history.append({"role": "user", "content": "something to forget"})
crowded = len(plain.history)
run("/clear")
check(len(plain.history) < crowded,
      f"/clear drops the conversation: {crowded} to {len(plain.history)}")
check(any(message.get("role") == "system" for message in plain.history),
      "and keeps the system message, so the model still knows where it is")

# /workspace with no argument reports, and with one moves. Both matter: the
# reporting form is what somebody types when they are not sure.
here = run("/workspace").strip()
check(here == str(plain.workspace),
      f"/workspace with nothing says where it is: {here}")
moved_to = root / "other-project"
moved_to.mkdir(exist_ok=True)
run(f"/workspace {moved_to}")
check(Path(plain.workspace) == moved_to,
      f"/workspace with a path really moves there: {plain.workspace}")
run(f"/workspace {workspace}")

# /permissions reads the config, so a rule saved there has to appear.
rules = saved()
rules.setdefault("permissions", {}).setdefault("global", []).append("ls")
config.save(rules, config_file)
permissions_out = run("/permissions", cli=app.IsaacCLI(
    "probe-model", workspace, 4, autostart_ollama=False, config_file=config_file))
check("ls" in permissions_out,
      "/permissions shows a rule that is in the config, rather than a fixed list")

# /status is the screen somebody reads when something is wrong, so the values on
# it have to be this session's rather than defaults.
status_out = run("/status")
check(plain.session_id in status_out and str(plain.session_path) in status_out,
      "/status names this session and its log, not a placeholder")
check("probe-model" in status_out,
      "and the model actually in use")

# --- /model <name>: the branch with no screen and the most to lose ---------
#
# `/model` with an argument never draws anything, so the sweep that stops at the
# first selector was never going to reach it, and it is the branch that decides
# where the next question is sent. The contract apply_profile states is that the
# five fields move together or not at all: a profile that carries a temperature
# and a path that does not read it is a setting the user chose and the program
# silently ignores.
switch = app.IsaacCLI("probe-model", workspace, 4, autostart_ollama=False,
                      config_file=config_file)
# The engine is not brought up here. Starting a server is what this command does
# for a living and it is exactly what a check must not do on somebody's machine.
switch.prewarm_engine = lambda *args, **kwargs: None
remote = {
    "provider": "openai_compatible", "provider_name": "Server",
    "base_url": "https://api.test/v1", "credential": "api:probe",
    "model": "remote/model", "thinking": "high", "num_ctx": 32768,
    "temperature": 0.3, "max_output_tokens": 4096,
}
profiles = saved()
profiles.setdefault("profiles", {})["probe-remote"] = remote
config.save(profiles, config_file)

run("/model probe-remote", cli=switch)
carried = (switch.model, switch.thinking, switch.num_ctx, switch.temperature,
           switch.max_output_tokens)
check(carried == ("remote/model", "high", 32768, 0.3, 4096),
      f"/model with a saved profile takes every field that profile chose: {carried}")
check(switch.provider.get("base_url") == "https://api.test/v1",
      f"including the endpoint, so the next question goes where the profile says: "
      f"{switch.provider.get('base_url')}")

# And the other half, which is the one that can send a question to the wrong
# machine: a bare name is a profile that chose nothing but the model, so
# everything the replaced profile chose has to go with it.
run("/model some-local-model", cli=switch)
kept = (switch.thinking, switch.num_ctx, switch.temperature,
        switch.max_output_tokens)
check(switch.model == "some-local-model",
      f"/model with a bare name switches to it: {switch.model}")
check(kept == (None, None, None, None),
      f"and drops what the previous profile chose, rather than lending its "
      f"settings to a model that never chose them: {kept}")
check(switch.provider.get("base_url") != "https://api.test/v1",
      f"and above all stops pointing at the replaced profile's endpoint: "
      f"{switch.provider.get('base_url')}")

# --- /model with no argument, driven through the screen to the session ------
#
# The expensive rows on this screen install software or spend GPU quota, and
# they are driven in check_setup where every cost is a stub. The row that costs
# nothing is an endpoint already configured, and following it is what closes the
# loop this task is about: the screen writes the configuration, and the session
# has to pick it up. A profile saved and not applied leaves the panel naming the
# model that was replaced and the next question going to the old one.
import setup_ollama  # noqa: E402

selector = app.IsaacCLI("probe-model", workspace, 4, autostart_ollama=False,
                        config_file=config_file)
selector.prewarm_engine = lambda *args, **kwargs: None
offered_models = ["remote/model", "remote/other-model"]
original_list = setup_ollama._list_api_models
original_validate = setup_ollama._validate_api
try:
    # The network is the one thing this screen reaches on its own, and it is
    # stubbed rather than allowed: the suite promises to stay offline.
    setup_ollama._list_api_models = lambda base_url, api_key: offered_models
    setup_ollama._validate_api = lambda url, key, model: None
    _session, selector_out, selector_error = drive(
        "/model",
        # The row is named as the screen draws it, provider and model, which is
        # not the key the profile is stored under.
        Chooser("Server · remote/model", "remote/other-model", "high"),
        cli=selector)
finally:
    setup_ollama._list_api_models = original_list
    setup_ollama._validate_api = original_validate
check(selector_error is None,
      f"/model reaches its screen, and the rows are answered by name: {selector_error}")
_name, chosen = config.profile(saved())
check((chosen or {}).get("model") == "remote/other-model",
      f"the model picked on the screen is the model written to the config: "
      f"{(chosen or {}).get('model')}")
check(selector.model == "remote/other-model",
      f"and the session picks it up, rather than answering with the one it had: "
      f"{selector.model}")
check(selector.thinking == "high",
      f"with the reasoning level chosen on the screen after it: {selector.thinking}")

# The four that only report are still driven, because a report that raises is a
# command that does not work, and because each has to reach real state.
sessions_out = run("/sessions")
check(previous_path.name in sessions_out or previous_id in sessions_out,
      f"/sessions lists a session that exists on disk: {previous_id}")
history_out = run("/history")
check(history_out.strip() != "", "/history answers with something")
feedback_out = run("/feedback")
check(feedback_out.strip() != "", "/feedback answers with something")
show_out = run("/show")
check(show_out.strip() != "", "/show answers with something even when nothing ran")


print()
if failures:
    print(f"{len(failures)} check(s) failed")
    raise SystemExit(1)
print("all command effect checks passed")
