"""One place decides how many bytes each path may push into the model's window.

Every tool that puts bytes in front of the model used to carry its own limit,
written in bytes, decided for a machine and a model nobody knew. Measured on
2026-08-22, one call of each carried 338.768 bytes, roughly 96.790 tokens,
against windows of 8.192, 16.384 and 32.768: a single `read_file` on its own
occupied 348% of a 16K window, and that is what ended the first Part B case of
task 036 after 56 seconds. Limits that never sum, because nothing ever adds
them, are guaranteed to overflow whatever window they meet.

Two different promises live here and they must not be confused. A **share** is
this path's slice of the model's window, and it only exists once something says
how big that window is. A **ceiling** is an absolute number of bytes that
protects the memory of the machine this runs on, which no model window has
anything to say about, and it is what holds when nothing declared a window.
"""
import re

import debug

# The window the current turn runs in, or None when nothing declared one. It is
# process-wide for the same reason the read cap was: one turn runs at a time and
# every path has to agree on the same number for their sum to mean anything.
_window = None

# Only ever applied to bytes that have not been counted by any tokenizer. The
# server counts what it received exactly, in prompt_eval_count, and that is what
# `agent.context_report` uses for the part of the conversation already sent.
# The error here has a direction worth remembering: on source code the real
# ratio is smaller than 3.5, so this understates tokens, which means a share
# spends slightly more of the window than it claims. That is the reason the
# shares below sum well under what the window allows a request to occupy.
CHARS_PER_TOKEN = 3.5

# Each path's slice of the window, with the reason it is that size. They sum to
# 0.47, and the request as a whole may occupy 0.75 of the window, so one call of
# every path at once still leaves the conversation itself room to exist. That
# sum is not a comment: `check_tools.py` adds them against the smallest window
# this program supports and fails if they stop fitting.
SHARES = {
    # The largest single thing one step can add, because it is the whole point
    # of the tool: putting a file in front of the model.
    "read": 0.25,
    # A fetched page is mostly navigation and boilerplate around the part that
    # was wanted, so it earns less room than a file the model asked for by name.
    "web": 0.10,
    # Command output is evidence, and the useful part of it is the head and the
    # exit status. A build log that fills the window buys nothing.
    "command_output": 0.05,
    # Workspace instructions are read once and carried for the whole session, so
    # every token here is paid on every request, not once.
    "workspace_instructions": 0.05,
    # Mutation evidence proves a change happened. It is a receipt, not content.
    "mutation_diff": 0.02,
}

# What holds when no window was declared: bytes, about this machine's memory.
CEILINGS = {
    "read": 200_000,
    "web": 80_000,
    "command_output": 20_000,
    "workspace_instructions": 32 * 1024,
    "mutation_diff": 6_000,
}


def set_window(num_ctx):
    """Declare the window this turn runs in, or None to fall back to ceilings."""
    global _window
    _window = int(num_ctx) if num_ctx else None
    return _window


def window():
    return _window


def tokens_for(name, num_ctx=None):
    """This path's slice of the window, in tokens."""
    declared = num_ctx if num_ctx else _window
    return int(declared * SHARES[name]) if declared else 0


def bytes_for(name):
    """This path's limit right now, in bytes, and where that number came from."""
    ceiling = CEILINGS[name]
    if not _window:
        debug.note("context_budget",
                   f"{name} is capped at {ceiling} bytes by its absolute ceiling, "
                   f"because no context window was declared")
        return ceiling
    derived = int(tokens_for(name) * CHARS_PER_TOKEN)
    limit = min(ceiling, derived)
    debug.note("context_budget",
               f"{name} is capped at {limit} bytes: {SHARES[name]} of a {_window} "
               f"token window is {derived} bytes, against a ceiling of {ceiling}")
    return limit


# Every module-level constant in the context path whose number stays written by
# hand, and the reason it is not derived from the window. A name absent from
# here is refused by `undeclared_limits`, which is what keeps the next reviewer
# from having to remember this rule.
DECLARED = {
    "tools.py": {
        "MAX_READ_BYTES": "absolute ceiling on this machine's memory; the window "
                          "budget for reads lives in CEILINGS and SHARES here",
        "MAX_DIFF_INPUT_BYTES": "bounds what difflib is asked to build, which is "
                                "the user's RAM and not the model's window",
        "MAX_MUTATION_DIFF_LINES": "shape of the evidence on screen; the bytes "
                                   "that reach the model are bounded by the "
                                   "mutation_diff share",
        "MAX_WEB_BYTES": "kept as the name of the web ceiling for callers; the "
                         "effective limit is the web share",
        "MAX_MUTATION_DIFF_BYTES": "kept as the name of the mutation ceiling; "
                                   "the effective limit is the mutation share",
    },
    "execution.py": {
        "TIMEOUT_SECONDS": "containment policy, deliberate, not an estimate of "
                           "anything about a model",
        "OUTPUT_LIMIT": "kept as the name of the output ceiling; the effective "
                        "limit is the command_output share",
    },
    "workspace_instructions.py": {
        "MAX_INSTRUCTIONS_BYTES": "kept as the name of the ceiling; the effective "
                                  "limit is the workspace_instructions share",
    },
    "agent.py": {
        "CHARS_PER_TOKEN": "re-exported from here so there is one ratio, not two",
        "CONTEXT_INPUT_SHARE": "how much of the window the request may occupy, "
                               "which is the budget every share is measured "
                               "against and therefore belongs to the model, not "
                               "to a path",
        "RATE_LIMIT_RETRIES": "protocol, what the provider's 429 asks for",
        "RATE_LIMIT_MAX_WAIT": "protocol, what the provider's 429 asks for",
    },
}

# Names that look like a limit on what reaches the model. Deliberately narrow:
# a scan that flags everything gets switched off, and one that flags the family
# of defects that caused this already earns its place.
_LIMIT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_LIMIT_WORDS = ("BYTES", "LIMIT", "SHARE", "CHARS_PER", "CONTEXT")
_ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=\s*[-0-9]")


def undeclared_limits(filename, source):
    """Limits written by hand in a context path that nobody wrote a reason for.

    The pattern is the one already used to refuse a translator built without a
    language: a scan of the source that fails with the file and the line, rather
    than a rule somebody has to remember at review time.
    """
    allowed = DECLARED.get(filename, {})
    offenders = []
    for number, line in enumerate(source.splitlines(), 1):
        match = _ASSIGNMENT.match(line)
        if not match:
            continue
        name = match.group(1)
        if not any(word in name for word in _LIMIT_WORDS):
            continue
        if name in allowed:
            continue
        offenders.append(
            f"{filename}:{number} {name} is a hand-written limit on what reaches "
            f"the model. Derive it from the window in context_budget.SHARES, or "
            f"add it to context_budget.DECLARED with the reason it cannot be.")
    return offenders
