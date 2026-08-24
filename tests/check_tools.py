#!/usr/bin/env python3
"""Cheap tests for the local tools."""
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

import agent
import context_budget
import tools

failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    tools.SANDBOX_ROOT = root

    # --- read/write/append -------------------------------------------------
    out = tools.write_file("notes.txt", "first line\n")
    check(out.startswith("OK:"), "write_file returns OK")
    check(tools.read_file("notes.txt") == "first line\n", "read_file returns what was written")

    tools.append_file("notes.txt", "second line")
    text = tools.read_file("notes.txt")
    check(text == "first line\nsecond line\n", "append_file preserves the previous content")
    check(text.endswith("\n"), "append_file guarantees a trailing newline")

    check(tools.read_file("missing.txt").startswith("ERROR:"),
          "read_file reports a missing file instead of raising")
    check(tools.list_dir(".").splitlines() == ["notes.txt"], "list_dir lists the folder")
    check(tools.list_dir("missing").startswith("ERROR:"), "list_dir rejects a non-directory")

    # A small model emits a literal '\n' thinking it is a line break.
    tools.write_file("escaped.txt", "a\\nb")
    check(tools.read_file("escaped.txt") == "a\nb",
          "write_file converts a literal \\n when there is no real line break")
    tools.write_file("kept.txt", "real\nbackslash \\n stays")
    check("\\n" in tools.read_file("kept.txt"),
          "write_file keeps a backslash when the text already has a real line break")
    out = tools.write_file("unicode.txt", "ação")
    check("wrote 6 bytes" in out and "after=6 bytes" in out,
          "mutation results report UTF-8 bytes rather than Python characters")
    out = tools.write_file("empty.txt", "")
    check("created new file; before=0 bytes; after=0 bytes" in out
          and "no line-level textual difference" in out,
          "an empty new file has explicit objective evidence instead of a blank diff")

    # --- exact replacement and bounded mutation evidence ------------------
    tools.write_file("exact.txt", "before\ntarget\nafter\n")
    out = tools.replace_text("exact.txt", "target", "replacement")
    check(out.startswith("OK:") and "CHANGE EVIDENCE" in out
          and "-target" in out and "+replacement" in out,
          "replace_text returns objective evidence of the exact change")
    check(tools.read_file("exact.txt") == "before\nreplacement\nafter\n",
          "replace_text preserves every surrounding byte")

    snapshot = tools.read_file("exact.txt")
    out = tools.replace_text("exact.txt", "absent", "x")
    check(out.startswith("ERROR:") and tools.read_file("exact.txt") == snapshot,
          "replace_text leaves the file untouched when old_text is absent")
    tools.write_file("ambiguous.txt", "same\nmiddle\nsame\n")
    snapshot = tools.read_file("ambiguous.txt")
    out = tools.replace_text("ambiguous.txt", "same", "changed")
    check("appears 2 times" in out and tools.read_file("ambiguous.txt") == snapshot,
          "replace_text refuses ambiguity without modifying the file")

    large = "\n".join(f"old-{i}" for i in range(200))
    out = tools.write_file("large.txt", large)
    out = tools.write_file("large.txt", large.replace("old-", "new-"))
    evidence = out.split("CHANGE EVIDENCE", 1)[1]
    check("DIFF TRUNCATED BY ISAACLI LIMITS" in evidence
          and len(evidence.encode("utf-8")) < tools.MAX_MUTATION_DIFF_BYTES + 200,
          "large mutation evidence is bounded and explicitly marked as truncated")

    # --- memory bounds -----------------------------------------------------
    # Tested by effect, not by message: difflib is replaced with something that
    # raises, so any attempt to diff an oversized file fails the test loudly
    # instead of quietly costing the user gigabytes of RAM.
    import difflib

    def exploding_diff(*_args, **_kwargs):
        raise AssertionError("diffed a file that is too large to diff")

    original_diff = difflib.unified_diff
    original_input_cap = tools.MAX_DIFF_INPUT_BYTES
    original_read_cap = context_budget.CEILINGS["read"]
    try:
        tools.MAX_DIFF_INPUT_BYTES = 500
        oversized = "x" * 4000
        tools.write_file("huge.txt", oversized)
        difflib.unified_diff = exploding_diff
        appended = tools.append_file("huge.txt", "one more line")
        overwritten = tools.write_file("huge.txt", "small now")
    finally:
        difflib.unified_diff = original_diff
        tools.MAX_DIFF_INPUT_BYTES = original_input_cap

    check("too large for a line-level diff" in appended
          and "before=4000 bytes" in appended
          and "after=4014 bytes" in appended,
          "appending to an oversized file reports exact bytes without building a diff")
    check("too large for a line-level diff" in overwritten
          and "before=4014 bytes" in overwritten,
          "overwriting an oversized file never reads or diffs the old content")
    check(tools.read_file("huge.txt") == "small now",
          "the oversized path still wrote the real bytes to disk")

    try:
        context_budget.CEILINGS["read"] = 100
        tools.write_file("long.txt", "y" * 900)
        partial = tools.read_file("long.txt")
    finally:
        context_budget.CEILINGS["read"] = original_read_cap
    check(partial.startswith("y" * 100) and "y" * 101 not in partial
          and "FILE TRUNCATED BY ISAACLI LIMITS" in partial
          and "of 900 bytes" in partial,
          "read_file stops at the cap and says so instead of loading the whole file")

    # --- --debug surfaces what the normal flow absorbs ---------------------
    # By effect: the same call is made twice and only the stderr differs. A
    # test that read the refusal string would pass even if debug did nothing.
    import io
    import debug as debug_module

    def exploding_impl(**_kwargs):
        raise RuntimeError("the real cause, normally hidden")

    original_impl = tools.IMPLS["list_dir"]
    original_stderr = sys.stderr
    try:
        tools.IMPLS["list_dir"] = exploding_impl
        debug_module.enable(False)
        sys.stderr = quiet_err = io.StringIO()
        quiet = tools.execute("list_dir", {})
        debug_module.enable(True)
        sys.stderr = loud_err = io.StringIO()
        loud = tools.execute("list_dir", {})
    finally:
        sys.stderr = original_stderr
        tools.IMPLS["list_dir"] = original_impl
        debug_module.enable(False)

    check(quiet_err.getvalue() == "" and quiet.startswith("ERROR while running"),
          "without --debug an absorbed exception stays absorbed and silent")
    check(loud == quiet
          and "Traceback" in loud_err.getvalue()
          and "the real cause, normally hidden" in loud_err.getvalue()
          and "tools.execute(list_dir)" in loud_err.getvalue(),
          "--debug prints the traceback and the site without changing the result")

    # --- replace_between ---------------------------------------------------
    document = (
        "<html>\n"
        "  <!-- FORM_START -->\n"
        "  <!-- FORM_END -->\n"
        "  <script>\n"
        "    /* RENDER_START */\n"
        "    /* RENDER_END */\n"
        "  </script>\n"
        "</html>\n"
    )
    tools.write_file("page.html", document)

    out = tools.replace_between("page.html", "FORM_START", "FORM_END", '<form id="f"></form>')
    text = tools.read_file("page.html")
    check(out.startswith("OK:"), "replace_between returns OK")
    check('<!-- FORM_START -->\n<form id="f"></form>\n  <!-- FORM_END -->' in text,
          "replace_between swaps the body and keeps the marker lines")

    tools.replace_between("page.html", "RENDER_START", "RENDER_END",
                          "document.body.dataset.ok = '1';")
    text = tools.read_file("page.html")
    check("/* RENDER_START */\ndocument.body.dataset.ok = '1';\n    /* RENDER_END */" in text,
          "replace_between works with markers inside a JS comment")

    out = tools.replace_between("page.html", "FORM_START", "RENDER_END", "x")
    check(out.startswith("ERROR: mismatched markers"),
          "replace_between refuses crossed markers")

    out = tools.replace_between("page.html", "/* RENDER_START */", "/* RENDER_END */",
                                "const a = 1;")
    check(out.startswith("OK:"),
          "replace_between accepts a marker wrapped in its comment")

    # The model often echoes the marker lines back inside the content it sends.
    tools.replace_between(
        "page.html", "FORM_START", "FORM_END",
        '<!-- FORM_START -->\n<input id="clean">\n<!-- FORM_END -->')
    text = tools.read_file("page.html")
    check(text.count("FORM_START") == 1 and text.count("FORM_END") == 1,
          "replace_between strips markers the model repeated in the content")

    out = tools.replace_between("page.html", "MISSING_START", "MISSING_END", "x")
    check(out.startswith("ERROR: start marker not found"),
          "replace_between reports a missing start marker")

    out = tools.replace_between("missing.html", "FORM_START", "FORM_END", "x")
    check(out.startswith("ERROR: file does not exist"),
          "replace_between reports a missing file")

    # --- execute() dispatch -------------------------------------------------
    check(tools.execute("read_file", '{"path": "notes.txt"}') == "first line\nsecond line\n",
          "execute dispatches with JSON arguments as a string")
    check(tools.execute("read_file", {"path": "notes.txt"}).startswith("first"),
          "execute also accepts arguments already parsed")
    check(tools.execute("nope", "{}").startswith("ERROR: unknown tool"),
          "execute reports an unknown tool")
    check(tools.execute("read_file", "{not json").startswith("ERROR: arguments are not valid JSON"),
          "execute reports invalid JSON")
    check(tools.execute("read_file", '{"wrong": 1}').startswith("ERROR: wrong arguments"),
          "execute reports the wrong argument names")
    missing = tools.execute("write_file", {"content": "draft"})
    check("missing required argument(s): path" in missing
          and "Required arguments: path, content" in missing,
          "execute reports the exact missing schema arguments to the model")

    names = {s["function"]["name"] for s in tools.SCHEMA}
    check(names == set(tools.IMPLS),
          f"the schema and the implementations describe the same tools ({names})")

# The boundary is tested by effect, not by the refusal it prints. A test that
# reads "path outside the sandbox" passes just as well against a version that
# refuses and writes the file anyway, and the file is what matters.
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "workspace"
    root.mkdir()
    tools.SANDBOX_ROOT = root
    outside = Path(tmp) / "outside.txt"
    outside.write_text("untouched\n")
    (root / "shortcut").symlink_to(outside)
    (root / "up").symlink_to(Path(tmp))

    escapes = [
        "../outside.txt",
        "shortcut",
        "up/outside.txt",
        "sub/../../outside.txt",
        str(outside),
        "/etc/passwd",
    ]
    for attempt in escapes:
        for call in (
                lambda name: tools.write_file(name, "OWNED"),
                lambda name: tools.append_file(name, "OWNED"),
                lambda name: tools.replace_text(name, "untouched", "OWNED"),
        ):
            try:
                call(attempt)
            except ValueError:
                pass
    check(outside.read_text() == "untouched\n",
          "no write, append or replace reaches a file outside the workspace")
    check(Path("/etc/passwd").read_text().find("OWNED") == -1,
          "an absolute system path is neutralised rather than followed")

    # Reading has the same boundary, and a leak there is a disclosure rather
    # than a corruption, so it is worth its own effect check.
    leaked = []
    for attempt in escapes:
        try:
            leaked.append(tools.read_file(attempt))
        except ValueError:
            pass
    check(not any("untouched" in text or "root:" in text for text in leaked),
          "no read reaches a file outside the workspace")

    # A name that cannot be a path at all has to be refused, not passed down to
    # the filesystem where the error would come back as something else.
    refused_nul = []
    for call in (tools.read_file,
                 lambda name: tools.write_file(name, "x")):
        try:
            call("notes\0.txt")
        except (ValueError, OSError):
            refused_nul.append(True)
    check(len(refused_nul) == 2, "a path holding a null byte is refused")

# One read of one ordinary source file used to be allowed to carry 200 KB, which
# is around 57.000 tokens, into a 32.768 token window. The endpoint then refuses
# the entire request and the turn dies with everything the model had done in it.
# That is what ended the first Part B case of task 036 after 56 seconds, so the
# cap is measured against the window instead of being a constant.
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    tools.SANDBOX_ROOT = root
    big = "x" * 200_000
    (root / "huge.ts").write_text(big)
    try:
        tools.set_read_budget(32_768)
        narrow = tools.read_file("huge.ts")
        check(len(narrow) < 40_000 and "TRUNCATED" in narrow,
              "one read cannot swallow a window it was told the size of")
        check(agent.estimate_tokens([{"role": "tool", "content": narrow}])
              < 32_768 * agent.CONTEXT_INPUT_SHARE,
              "what one read returns still leaves the window room to answer")
        tools.set_read_budget(None)
        check(len(tools.read_file("huge.ts")) == 200_000,
              "with no window declared the old absolute ceiling is what holds")
    finally:
        tools.set_read_budget(None)

# Every path that pushes bytes into the model's window used to decide its own
# limit, in bytes, written for a machine and a model nobody knew. Measured on
# 2026-08-22: one call of each carried 338.768 bytes, around 96.790 tokens,
# against windows of 8.192, 16.384 and 32.768. One place decides now, and the
# question a budget has to answer is whether one call of each can fit at once.

PATHS = ("read", "web", "command_output", "workspace_instructions",
         "mutation_diff")
try:
    context_budget.set_window(32_768)
    roomy = {name: context_budget.bytes_for(name) for name in PATHS}
    context_budget.set_window(8_192)
    tight = {name: context_budget.bytes_for(name) for name in PATHS}
finally:
    context_budget.set_window(None)
check(all(tight[name] < roomy[name] for name in PATHS),
      "every path that fills the window scales with the window, not with a constant")
check(sum(context_budget.tokens_for(name, 8_192) for name in PATHS)
      <= int(8_192 * agent.CONTEXT_INPUT_SHARE),
      "one call of each fits at once in the smallest window this program supports")

# A limit nobody can explain becomes a magic number again at the next review, so
# each one says what it came from, and it says it where diagnosis belongs.
budget_notes = []
original_budget_note = context_budget.debug.note
try:
    context_budget.debug.note = lambda source, message: budget_notes.append(message)
    context_budget.set_window(16_384)
    context_budget.bytes_for("web")
finally:
    context_budget.debug.note = original_budget_note
    context_budget.set_window(None)
check(any("16384" in note and "web" in note for note in budget_notes),
      "each derived limit says the window it came from, in --debug")

# The absolute ceilings are a different promise: they protect the memory of the
# machine this runs on, not the window of a model, so they hold when no window
# was declared at all.
check(context_budget.bytes_for("read") == tools.MAX_READ_BYTES
      and context_budget.bytes_for("web") == context_budget.CEILINGS["web"],
      "with no window declared the absolute ceilings are what hold")

# The rule has to outlive whoever remembers it, so it is a scan that fails with
# the file and the line, like the one that refuses a translator without a
# language. Proven both ways: it has to accept the module as it stands and
# refuse a new hand-written limit planted in it.
offenders = context_budget.undeclared_limits(
    "tools.py", "MAX_SOMETHING_BYTES = 40_000\nSANDBOX_ROOT = 1\n")
check(any("MAX_SOMETHING_BYTES" in offender for offender in offenders),
      "a new hand-written limit in a context path is refused with its name")
check(context_budget.undeclared_limits(
    "tools.py", "MAX_READ_BYTES = 200_000\n") == [],
      "the limits that already have a written reason are not flagged again")

real_offenders = []
for module in ("tools.py", "execution.py", "workspace_instructions.py", "agent.py"):
    real_offenders += context_budget.undeclared_limits(
        module,
        (HERE.parent / "tool_harness" / module).read_text(encoding="utf-8"))
check(real_offenders == [],
      f"no context path carries an unexplained hand-written limit today: {real_offenders}")

# Where the conversation stands against the window is measured, not guessed. The
# server answers it exactly in prompt_eval_count after every call, and dividing
# characters by 3.5 was both unnecessary and wrong in the dangerous direction:
# on code the real ratio is smaller, so the estimate undershoots and the warning
# would arrive after the request had already been refused.
measured_history = [
    {"role": "system", "content": "contract"},
    {"role": "user", "content": "fix the bug"},
]
measured_report = agent.context_report(measured_history, 32_768, measured=24_000,
                                       measured_upto=2)
check(measured_report["used"] == 24_000 and measured_report["estimated"] == 0
      and measured_report["measured"] == 24_000,
      "what the server counted is what is used, not a ratio of characters")

# Everything added since that count has no measurement yet, so it is estimated
# and the report says how much of the number is a guess.
grown_history = measured_history + [
    {"role": "tool", "tool_call_id": "a", "content": "y" * 7_000}]
grown_report = agent.context_report(grown_history, 32_768, measured=24_000,
                                    measured_upto=2)
check(grown_report["measured"] == 24_000 and grown_report["estimated"] > 0
      and grown_report["used"] == 24_000 + grown_report["estimated"],
      "the delta since the last count is estimated, and named as the estimated part")

# Compacting is the one thing here that changes what the model can see, so it is
# offered before it happens, while there is still room to act, and never after.
# The trigger is the window and the largest single thing a tool may add to it,
# both of which are already known, rather than a fraction chosen by hand.
pressure_history = [
    {"role": "system", "content": "contract"},
    {"role": "user", "content": "fix the bug"},
    {"role": "assistant", "tool_calls": [
        {"id": "a", "function": {"name": "read_file",
                                 "arguments": '{"path": "src/client.ts"}'}}]},
    {"role": "tool", "tool_call_id": "a", "content": "y" * 60_000},
    {"role": "tool", "tool_call_id": "b", "content": "z" * 60_000},
    {"role": "tool", "tool_call_id": "c", "content": "recent and small"},
]
asked = []
declined = agent.fit_to_context(
    [dict(m) for m in pressure_history], 32_768,
    on_pressure=lambda report: asked.append(report) or False)
check(len(asked) == 1 and asked[0]["num_ctx"] == 32_768
      and asked[0]["used"] > asked[0]["budget"],
      "the numbers reach whoever decides, with the ceiling next to them")
check(declined == [] and asked[0]["compactable"] == 3,
      "saying no compacts nothing, and the offer said how much was on the table")

kept_history = [dict(m) for m in pressure_history]
agent.fit_to_context(kept_history, 32_768, on_pressure=lambda report: False)
check([m.get("content") for m in kept_history]
      == [m.get("content") for m in pressure_history],
      "a refused compaction leaves every message exactly as it was")

# Compacting is summarising, not deleting. The model has to be told what it lost,
# not merely that it lost something: a result replaced by a bare note leaves it
# unable to decide whether reading the file again is worth a step.
history = [dict(m) for m in pressure_history]
summaries = agent.fit_to_context(history, 32_768, on_pressure=lambda report: True)
check(len(summaries) == 1 and agent.context_report(history, 32_768)["used"]
      <= agent.context_report(history, 32_768)["budget"],
      "accepting compacts exactly as much as it takes to fit and no more")
check("read_file" in history[3]["content"] and "client.ts" in history[3]["content"]
      and "60000" in history[3]["content"].replace(",", ""),
      "what replaces a result names the call, its arguments and the size that went")
check(history[0]["content"] == "contract" and history[1]["content"] == "fix the bug"
      and history[2].get("tool_calls") and history[-1]["content"] == "recent and small",
      "the contract, the task, the calls and the newest work all survive")
check(all(m.get("tool_call_id") for m in history if m["role"] == "tool"),
      "a compacted result keeps the id its call is waiting on")

# With nobody to ask there is still nobody to keep in the dark: the run is a
# measurement script, it compacts to survive, and the log says what went.
silent_history = [dict(m) for m in pressure_history]
noted = []
original_note = agent.debug.note
try:
    agent.debug.note = lambda source, message: noted.append(f"{source} {message}")
    auto = agent.fit_to_context(silent_history, 32_768)
finally:
    agent.debug.note = original_note
check(len(auto) == 1 and any("read_file" in line for line in noted),
      "with nobody at the terminal it compacts and the log names what it compacted")

# The task itself is never compactable, and a task that does not fit is a refusal
# with the numbers, not a silent shrinking of what the user asked for. This is
# the case that kept the dense model out of Part A of task 036.
oversized = [
    {"role": "system", "content": "contract"},
    {"role": "user", "content": "q" * 200_000},
]
oversized_report = agent.context_report(oversized, 32_768)
check(oversized_report["over"] and oversized_report["compactable"] == 0,
      "a request bigger than the window is over budget with nothing to compact")
untouched_request = [dict(m) for m in oversized]
check(agent.fit_to_context(untouched_request, 32_768,
                           on_pressure=lambda report: True) == []
      and untouched_request[1]["content"] == oversized[1]["content"],
      "the user's own request is never compacted, not even when it is what does not fit")

# The two limits are only worth anything if the window reaches them, so this
# runs the loop itself with a stubbed endpoint and reads the cap afterwards.
original_call = agent.call
try:
    agent.call = lambda *args, **kwargs: {"role": "assistant", "content": "done"}
    tools.set_read_budget(None)
    agent.run("anything", "some-model", max_steps=1, verbose=False, num_ctx=32_768)
    wired = context_budget.bytes_for("read")
finally:
    agent.call = original_call
    tools.set_read_budget(None)
check(wired == int(context_budget.tokens_for("read", 32_768)
                   * context_budget.CHARS_PER_TOKEN),
      "the window a turn runs in is what sets the read cap for that turn")

# A budget that only the function which sets it can see is worth nothing, so
# this runs the loop itself: the server's count from one step has to be what the
# next step measures against, instead of the loop going back to counting
# characters and warning too late.
loop_reports = []
loop_answers = [
    {"role": "assistant", "content": "",
     "tool_calls": [{"id": "z", "type": "function",
                     "function": {"name": "read_file",
                                  "arguments": '{"path": "gone.txt"}'}}],
     "_usage": {"prompt_eval_count": 30_000, "eval_count": 1}},
    {"role": "assistant", "content": "done",
     "_usage": {"prompt_eval_count": 30_100, "eval_count": 1}},
]
original_call = agent.call
try:
    agent.call = lambda *args, **kwargs: loop_answers.pop(0)
    agent.run("anything", "some-model", max_steps=2, verbose=False,
              num_ctx=32_768,
              on_context_pressure=lambda report: loop_reports.append(report) or False)
finally:
    agent.call = original_call
    tools.set_read_budget(None)
check(len(loop_reports) == 1 and loop_reports[0]["measured"] == 30_000,
      "the count the server returned is what the next step of the loop measures against")
check(loop_reports[0]["used"] >= 30_000
      and loop_reports[0]["used"] > loop_reports[0]["budget"],
      "a conversation the server says is nearly full is seen as nearly full")

untouched = [{"role": "tool", "tool_call_id": "a", "content": "y" * 60_000}]
check(agent.fit_to_context(untouched, None) == []
      and len(untouched[0]["content"]) == 60_000,
      "with no window declared nothing is compacted behind the model's back")

# A question whose only answer changes nothing is worse than no question. On
# 2026-08-23 a fresh session was opened, "oi" was typed, and the screen offered
# to compact while saying, in the same paragraph, that 0 results could be
# compacted. Answering yes led to the same refusal, because what filled the
# window was the fixed part of every request: the system prompt, the tool
# schema and a 21.082 character AGENTS.md, none of which a compaction touches.
nothing_to_compact = [
    {"role": "system", "content": "c" * 30_000},
    {"role": "user", "content": "oi"},
]
offered = []
noted = []
result = agent.fit_to_context(
    [dict(m) for m in nothing_to_compact], 8192,
    on_pressure=lambda report: offered.append(report) or True,
    on_note=lambda report, summaries: noted.append((report, summaries)))
pressure = agent.context_report(nothing_to_compact, 8192)
check(pressure["over"] and pressure["compactable"] == 0,
      "the reproduction really is under pressure with nothing to compact")
check(not offered,
      "compaction is not offered when no message in the conversation can be "
      f"compacted (asked {len(offered)} time(s))")
check(result == [], "and nothing is compacted, because there was nothing")
check(len(noted) == 1 and noted[0][0]["compactable"] == 0,
      "the pressure is still reported, so the turn does not fail in silence")

# The offer must keep coming when there is in fact something to give up.
still_offered = []
agent.fit_to_context(
    [dict(m) for m in pressure_history], 32_768,
    on_pressure=lambda report: still_offered.append(report) or False)
check(len(still_offered) == 1,
      "a conversation that does hold compactable results is still offered the "
      "choice")


print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("TOOLS OK")
