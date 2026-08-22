"""A scan that refuses interface text written outside the catalogs.

`check_cli.py` already compares `en.json` against `pt-BR.json` key by key and
placeholder by placeholder, which catches a key that lost its pair. It cannot
see a sentence that never entered a catalog at all, because a string with no
pair has no pair to miss. That was the hole left open when the quantization
screen came out in English inside a Portuguese session.

The hard part is not finding literals, it is knowing which ones are wrong. This
project writes three kinds of text and only one of them belongs in a catalog:

1. **What the user reads.** Catalog, always, through `cli_i18n.t`.
2. **What the model reads**: system prompt, tool description, tool result,
   sandbox refusal. Fixed English, always, never translated, because it is a
   contract with the model and not a preference of whoever is using it.
3. **Identifiers, keys, formats, `--debug` notes.** Neither.

A scan that flags the second kind is worth nothing by the following week,
because the only way to keep the suite green is to switch it off. So this one
decides by **where the string is going**, not by what it says: the sinks below
all write to the screen the user is looking at, and nothing else does. Text for
the model reaches `msgs`, `tools.SCHEMA` or a tool's return value, and text for
`--debug` reaches `debug.note`, none of which are sinks here.

Deliberately narrow, for the same reason `context_budget.undeclared_limits` is:
literal arguments of `print` and of `terminal_ui`. That already covers the
family of defects that motivated this. Widen it later, with a measured target.
"""
import ast
import re


# A `print` that writes somewhere other than stdout is not the screen. That is
# exactly what `debug.py` does, and its text is category 3 by definition.
_STDOUT = ("sys.stdout", "stdout")

# Placeholders carry runtime values, so their names are not words on screen, and
# an escape sequence is paint rather than text. Both go before anything is
# counted, otherwise `f"  {i}) {option}"` reads as a sentence.
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
_LETTERS = re.compile(r"[^\W\d_]")

# Text the user reads outside the catalogs, kept here with the reason it cannot
# be a key. Same shape as `context_budget.DECLARED`: keyed by the literal itself
# rather than by a line number, so moving the code does not silently empty the
# table, and adding an entry costs writing down why.
DECLARED = {
    "agent.py": {
        "[step {}] FINAL ANSWER:\n{}":
            "the verbose trace of the agent loop, which is developer output in "
            "the sense of debug.note and never part of a session's screen",
        "[step {}] {} TOOL_CALL -> {}({})":
            "same verbose trace, one line per tool call",
        "steps: {}  calls: {}":
            "the summary printed by `python3 agent.py`, a developer entry "
            "point that the CLI never runs",
    },
}


def _is_prose(text):
    """Two words separated by a space, which is what a sentence needs.

    A catalogue key (`cli.status.model`), an identifier, a separator or a
    number are all one token or none, so they never reach the report. The test
    is on purpose blunt: whatever it lets through is caught by the next scan
    somebody writes, and whatever it flags falsely would be noise, which is how
    a check gets switched off.
    """
    stripped = _ANSI.sub(" ", _PLACEHOLDER.sub(" ", text))
    words = [word for word in stripped.split()
             if len(_LETTERS.findall(word)) >= 2]
    return len(words) >= 2


def _literals(node):
    """Every literal that reaches an argument without passing through a call.

    A list of options and an `a or "b"` default are still literals written at
    the call site. A `t(...)` call is not, and neither is a name, which is the
    whole point: those already came from a catalog.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        # The f-string's constant halves are the part a translator would write.
        # Each interpolation becomes a bare `{}` rather than disappearing: it
        # keeps the reported text readable, it survives renaming the variable
        # inside it, and `_is_prose` strips it like any other placeholder.
        return ["".join(
            part.value if isinstance(part, ast.Constant)
            and isinstance(part.value, str) else "{}"
            for part in node.values)]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [text for element in node.elts for text in _literals(element)]
    if isinstance(node, ast.BoolOp):
        return [text for value in node.values for text in _literals(value)]
    return []


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _sink(call):
    """The name of the screen-writing call, or None if this is not one."""
    name = _dotted(call.func)
    if name == "print":
        for keyword in call.keywords:
            if keyword.arg == "file" and _dotted(keyword.value) not in _STDOUT:
                return None
        return "print"
    if name and name.startswith("terminal_ui."):
        return name
    return None


def screen_literals(filename, source):
    """Every sentence written literally at a call that draws on the screen.

    Yields `(line, sink, text)` without consulting `DECLARED`, which is what
    lets a test ask the other question: whether an entry in that table still
    describes a line that exists.
    """
    tree = ast.parse(source, filename=filename)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        sink = _sink(node)
        if sink is None:
            continue
        arguments = list(node.args) + [k.value for k in node.keywords]
        for argument in arguments:
            for text in _literals(argument):
                if _is_prose(text):
                    found.append((node.lineno, sink, text))
    return found


def interface_literals(filename, source):
    """Interface text written at a screen-writing call instead of in a catalog.

    Returns one message per offender, naming the file and the line, in the
    shape `context_budget.undeclared_limits` established: a scan that fails
    with a location beats a rule somebody has to remember at review time.
    """
    allowed = DECLARED.get(filename, {})
    return [
        f"{filename}:{line} {sink}() is handed the literal {text!r}. Text the "
        f"user reads goes through cli_i18n.t with a key in locales/en.json and "
        f"locales/pt-BR.json, or into i18n_scan.DECLARED with the reason it "
        f"cannot."
        for line, sink, text in screen_literals(filename, source)
        if text not in allowed
    ]
