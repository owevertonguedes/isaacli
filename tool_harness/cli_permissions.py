"""Command classification for the approval prompt.

Pure functions: given a command string, decide what to show and how to warn.
This overlaps with execution.review(), which decides what runs without asking;
the two have similar lists and different purposes. Unifying them is a
behaviour change and belongs in its own commit, not this mechanical move.
"""
import shlex

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
