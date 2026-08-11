#!/usr/bin/env python3
"""Cheap tests for the local tools."""
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

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
    check([s["function"]["name"] for s in tools.filtered_schema(["read_file"])] == ["read_file"],
          "filtered_schema returns only what was asked for")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("TOOLS OK")
