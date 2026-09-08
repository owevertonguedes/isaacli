#!/usr/bin/env python3
"""Guards against what a public repository should never say about a private
machine, and against a feature that arrived in the code with no way in.

Four sweeps, all mechanical, all over `git ls-files` (the tracked tree, which
is what is actually public):

1. A literal pointer to something that is not versioned: the private notes
   directory, the two note files that used to sit at this repository's root,
   or a home directory named after a real person. Four of these had already
   been published before this sweep existed. The exact strings are in
   FORBIDDEN_SUBSTRINGS below, spelled by concatenation so that this file can
   name them without tripping its own scan, which is also why this file is
   scanned like every other one rather than exempting itself.
2. A relative markdown link that points at a file that does not exist in the
   tree. A reader who clicks it gets nothing.
3. A third-party program named in the unattended command allowlist
   (`tool_harness/execution.py`) or probed with `shutil.which` anywhere under
   `tool_harness/`, that no published document (README, README.pt-BR, docs/)
   ever mentions. This is the check task 065 asked for: the `graphify` tool
   lived in the allowlist and the system prompt for months with zero mentions
   in any document, and this comparison would have caught it the day it
   landed (removed in commit 9547e7b).
4. Published storage paths that still present the package directories used by
   old releases as the current destination instead of the XDG data directory.

Every failure names the file and, where it applies, the line: a report that
tells the reader where to look, not a traceback that ends the run instead of
finishing it.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [ROOT / line for line in out.splitlines() if line]


# --- 1. literal pointers to what is not public --------------------------

# Built by concatenation on purpose, and it is the only reason this file can be
# scanned like every other one. It used to exempt itself instead, and a guard
# that skips itself is a guard with one file it cannot see: this docstring was
# publishing a real home directory and the number of a private note, in the very
# file whose job is to stop exactly that.
#
# `CLAUDE.md` is deliberately NOT in this list: it is a documented, public
# concept in this project (isaacli recognises a workspace's own CLAUDE.md and
# explains that it is not an alias for AGENTS.md, in docs/USAGE.md and in
# tests/check_workspace_instructions.py), not a pointer into the private
# instructions file at this repository's own root. A real pointer to THIS
# repo's private CLAUDE.md would be a markdown link to it, which the broken
# link sweep below catches on its own, because that file is untracked.
FORBIDDEN_SUBSTRINGS = (
    "tasks" + "/",
    "CONTEXTO" + ".md",
    "HAND" + "OFF",
)

# Home directories that are test fixtures or placeholders, not the owner's
# real machine, and are fine to keep.
NEUTRAL_HOME_NAMES = {"user", "isaac"}
HOME_PATTERN = re.compile(r"/home/([A-Za-z0-9_.-]+)")


def scan_forbidden_pointers(files):
    hits = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(ROOT)
        for line_no, line in enumerate(text.splitlines(), start=1):
            for needle in FORBIDDEN_SUBSTRINGS:
                if needle in line:
                    hits.append(f"{rel}:{line_no}: contains {needle!r}: {line.strip()[:120]!r}")
            for match in HOME_PATTERN.finditer(line):
                name = match.group(1)
                if name not in NEUTRAL_HOME_NAMES:
                    hits.append(
                        f"{rel}:{line_no}: names a home directory "
                        f"({match.group(0)!r}) that is not a neutral fixture: "
                        f"{line.strip()[:120]!r}"
                    )
    return hits


# --- 2. relative markdown links that point at nothing --------------------

MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def scan_broken_links(files):
    # Checked against the TRACKED set, not raw filesystem existence: a link to
    # this project's own private CLAUDE.md resolves on the disk of whoever
    # wrote it, but nobody who clones the repository has that file, so the
    # link is exactly as broken for them as one pointing at a typo.
    tracked = {p.resolve() for p in files}
    hits = []
    md_files = [p for p in files if p.suffix == ".md"]
    for path in md_files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(ROOT)
        for line_no, line in enumerate(text.splitlines(), start=1):
            for target in MD_LINK.findall(line):
                target = target.split(" ", 1)[0].strip()
                if not target or target.startswith("#"):
                    continue
                if "://" in target or target.startswith("mailto:"):
                    continue
                clean = target.split("#", 1)[0]
                if not clean:
                    continue
                if clean.startswith("/"):
                    resolved = (ROOT / clean.lstrip("/")).resolve()
                else:
                    resolved = (path.parent / clean).resolve()
                if resolved not in tracked and not resolved.is_dir():
                    hits.append(
                        f"{rel}:{line_no}: relative link to a file that is "
                        f"not tracked in the repository ({target!r}): "
                        f"{line.strip()[:120]!r}"
                    )
    return hits


# --- 3. third-party program named nowhere in public docs -----------------

# Standard POSIX coreutils: present on any Linux box by definition, so a
# document does not need to introduce them the way it needs to introduce an
# optional dependency like `ollama` or `bwrap`. Anything in the allowlist
# that is NOT one of these is treated as worth documenting.
KNOWN_COREUTILS = {"ls", "cat", "head", "tail", "wc", "grep", "find"}

# Anchored at the start of a line so this matches the top-level `ALLOWED = {`
# and not `GIT_ALLOWED = {` or `GH_ALLOWED = {`, which hold subcommands, not
# program names.
ALLOWED_SET_PATTERN = re.compile(r"^ALLOWED\s*=\s*\{([^}]*)\}", re.MULTILINE)
WHICH_CALL_PATTERN = re.compile(r"shutil\.which\(\s*[\"']([^\"']+)[\"']")
QUOTED_NAME = re.compile(r"[\"']([^\"']+)[\"']")


def third_party_names(execution_text=None):
    execution_py = ROOT / "tool_harness" / "execution.py"
    text = (execution_py.read_text(encoding="utf-8")
            if execution_text is None else execution_text)
    names = set()

    match = ALLOWED_SET_PATTERN.search(text)
    if match is None:
        return None
    for name in QUOTED_NAME.findall(match.group(1)):
        if name not in KNOWN_COREUTILS:
            names.add(name)

    for py_file in (ROOT / "tool_harness").glob("*.py"):
        py_text = py_file.read_text(encoding="utf-8")
        for name in WHICH_CALL_PATTERN.findall(py_text):
            names.add(name)

    return names


def public_doc_text():
    doc_paths = [ROOT / "README.md", ROOT / "README.pt-BR.md"]
    doc_paths += sorted((ROOT / "docs").glob("*.md"))
    text = ""
    for path in doc_paths:
        if path.exists():
            text += path.read_text(encoding="utf-8") + "\n"
    return text


def _mentioned(name, docs):
    """Case-insensitive, and a hyphenated tool (dpkg-query) also counts as
    named when the doc says the family (dpkg) rather than the exact binary.
    """
    if re.search(r"\b" + re.escape(name) + r"\b", docs, re.IGNORECASE):
        return True
    if "-" in name:
        prefix = name.split("-", 1)[0]
        if re.search(r"\b" + re.escape(prefix) + r"\b", docs, re.IGNORECASE):
            return True
    return False


def scan_undocumented_third_party(execution_text=None):
    hits = []
    docs = public_doc_text()
    names = third_party_names(execution_text)
    if names is None:
        return [
            "tool_harness/execution.py: could not find the ALLOWED = {...} "
            "set; update ALLOWED_SET_PATTERN in this check"
        ]
    for name in sorted(names):
        if not _mentioned(name, docs):
            hits.append(
                f"tool_harness/execution.py or a shutil.which() probe names "
                f"{name!r} as a program isaacli runs, but no published "
                f"document (README.md, README.pt-BR.md, docs/*.md) mentions it"
            )
    return hits


# --- 4. current private-data locations in published docs -----------------

DATA_LOCATION_DOCS = (
    "docs/ARCHITECTURE.md", "docs/SECURITY.md", "docs/USAGE.md",
)


def scan_stale_data_locations():
    hits = []
    for relative in DATA_LOCATION_DOCS:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        if "$XDG_DATA_HOME/isaacli/cli_sessions" not in text:
            hits.append(
                f"{relative}: does not name $XDG_DATA_HOME/isaacli/cli_sessions "
                "as the current session location"
            )
        if "$XDG_DATA_HOME/isaacli/feedback" not in text and relative != "docs/USAGE.md":
            hits.append(
                f"{relative}: does not name $XDG_DATA_HOME/isaacli/feedback "
                "as the current feedback location"
            )
    return hits


def main():
    files = tracked_files()

    pointer_hits = scan_forbidden_pointers(files)
    check(not pointer_hits,
          "no tracked file points at an unpublished private file or names "
          "the owner's real home directory" if not pointer_hits else
          "found pointer(s) to private/unpublished paths:\n  "
          + "\n  ".join(pointer_hits))

    link_hits = scan_broken_links(files)
    check(not link_hits,
          "every relative markdown link resolves to a tracked file" if not link_hits
          else "found broken relative link(s):\n  " + "\n  ".join(link_hits))

    doc_hits = scan_undocumented_third_party()
    check(not doc_hits,
          "every third-party program in the allowlist or probed with "
          "shutil.which is mentioned in the published documentation"
          if not doc_hits else
          "found undocumented third-party program(s):\n  " + "\n  ".join(doc_hits))

    changed_shape_hits = scan_undocumented_third_party(
        'ALLOWED = frozenset({"python3"})')
    check(changed_shape_hits == [
        "tool_harness/execution.py: could not find the ALLOWED = {...} "
        "set; update ALLOWED_SET_PATTERN in this check"
    ],
          "an ALLOWED shape change is reported with its file and cause, "
          "without interrupting later checks")

    location_hits = scan_stale_data_locations()
    check(not location_hits,
          "published docs name the XDG data directory as current storage"
          if not location_hits else
          "found stale private-data location documentation:\n  "
          + "\n  ".join(location_hits))

    # No check may read the real HOME. In 2026-08 one weight landing in the
    # owner's download folder failed three checks that had nothing to do with
    # it, because `local_models.available()` scans that folder by default, and
    # the failure message carried the name of his file. Measured by effect on
    # 2026-09-07 as well, by planting two decoy weights there and running the
    # whole suite, which did not move; this is the standing version of that.
    #
    # A call is safe when it is told where to look, by any of the arguments
    # that redirect it. Naming only home_dir and environ reported five calls
    # that pass root= and target_dir= and were never in danger.
    scanning = {
        "local_models": {"available", "data_dir", "download_weight",
                         "downloaded_dir", "link_ollama_model", "linked_dir",
                         "linked_models", "ollama_manifests", "ollama_models",
                         "ollama_root"},
        "installation": {"custom_ollama_model_paths", "feedback_dir",
                         "sessions_dir", "uninstall_managed_llamacpp",
                         "uninstall_official_ollama"},
    }
    redirects = {"home_dir", "environ", "root", "target_dir", "path",
                 "record_path"}
    ambient = []
    for path in sorted(HERE.glob("check_*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (isinstance(function, ast.Attribute)
                    and isinstance(function.value, ast.Name)):
                continue
            if function.attr not in scanning.get(function.value.id, ()):
                continue
            if node.args:
                continue
            if not (redirects & {word.arg for word in node.keywords}):
                ambient.append(f"{path.name}:{node.lineno} "
                               f"{function.value.id}.{function.attr}()")
    check(not ambient,
          "no check reads the real HOME, so nothing in it can change an answer"
          if not ambient else
          "these calls fall back to the owner's own folders:\n  "
          + "\n  ".join(ambient))

    # A check file whose failure gate sits above some of its checks reports
    # [FAILED] and still exits zero, so the runner calls it OK and the suite
    # shrinks without anyone deciding to shrink it. That is what check_cli.py
    # was doing: the gate sat above the README checks, and everything appended
    # after them inherited the silence. Structural rather than by effect,
    # because proving it by effect means planting a failure in fifteen files.
    gate = re.compile(r"sys\.exit\(|raise SystemExit\(")
    call = re.compile(r"\s*check\(")
    unguarded = []
    for path in sorted((HERE).glob("check_*.py")):
        lines = path.read_text(encoding="utf-8").split("\n")
        calls = [index for index, line in enumerate(lines) if call.match(line)]
        if not calls:
            continue
        gates = [index for index, line in enumerate(lines) if gate.search(line)]
        if not gates:
            unguarded.append(f"{path.name}: uses check() and never exits non-zero")
            continue
        after = [index + 1 for index in calls if index > gates[-1]]
        if after:
            unguarded.append(
                f"{path.name}: {len(after)} check(s) after the gate, "
                f"first at line {after[0]}")
    check(not unguarded,
          "every check file's failure gate comes after its last check"
          if not unguarded else
          "a failure in these would print and still exit zero:\n  "
          + "\n  ".join(unguarded))

    print()
    if failures:
        print(f"PUBLIC HYGIENE: {len(failures)} check(s) failed")
        return 1
    print("PUBLIC HYGIENE OK: no private pointer, no broken link, no "
          "undocumented third-party program")
    return 0


if __name__ == "__main__":
    sys.exit(main())
