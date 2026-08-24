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


# The calls that turn a key into a sentence, matched on the last segment of the
# name so that a translator reached through its module counts. Named rather than
# guessed, because the whole point below is to read their first argument.
#
# The last segment and not the whole name, because writing out the whole one is
# how this was first written and it missed `model_discovery.text`, which builds
# a key from a prefix. That miss was harmless only because a second call site
# builds the same prefix in a recognised shape; had it not, three live keys
# would have been reported as orphans, and the obvious way to make the check
# pass again is to delete them.
_TRANSLATORS = ("t", "_t", "translate", "text", "speak")


def _translated_key(call):
    """The first argument of a translating call, as (prefix, whole_key).

    `prefix` is set only when the key is assembled at runtime, and it is the
    literal half that always precedes the part chosen at the call:
    `t(f"model.origin.{name}")` and `translate("model.origin." + origin(model))`
    both give `model.origin.`. `whole_key` is set only when the key is written
    out in full.
    """
    name = _dotted(call.func)
    if not name or name.rsplit(".", 1)[-1] not in _TRANSLATORS:
        return None, None
    argument = call.args[0] if call.args else None
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return None, argument.value
    if isinstance(argument, ast.JoinedStr) and argument.values:
        head = argument.values[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str) \
                and head.value.endswith("."):
            return head.value, None
    if isinstance(argument, ast.BinOp) and isinstance(argument.op, ast.Add) \
            and isinstance(argument.left, ast.Constant) \
            and isinstance(argument.left.value, str) \
            and argument.left.value.endswith("."):
        return argument.left.value, None
    return None, None


def _assembled(sources):
    """(prefixes built at a translating call, every string literal in sight)."""
    prefixes, literals = set(), set()
    for filename, source in sources.items():
        if not str(filename).endswith(".py"):
            continue
        try:
            tree = ast.parse(source, filename=str(filename))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literals.add(node.value)
            elif isinstance(node, ast.Call):
                prefix, _whole = _translated_key(node)
                if prefix:
                    prefixes.add(prefix)
    return prefixes, literals


def orphan_keys(catalog, sources):
    """Catalog keys that no screen can ever ask for.

    A key nobody asks for is translated text no screen can show, and it stays
    perfectly aligned in both catalogs, so the comparison between them reads it
    as supported wording. One lived that way from the commit that introduced it
    until a repository-wide sweep found it.

    The hard half is the key assembled at the call site. Exempting its whole
    prefix, which is what this used to do, hides exactly the orphan that costs
    most: a suffix no code can produce any more goes on reading as used because
    a sibling of it is. So the suffix is resolved instead: a runtime key counts
    as reachable when the part after the prefix is written somewhere as a
    literal, which is where every such value in this repository comes from
    (`TASK_VALUES`, the returns of `model_discovery.origin`, an exception's
    `reason`).

    No orphan is ever named in prose, here or in the check that calls this.
    Every file this reads is a file it searches, so writing the key down is
    what makes it look asked-for, and a note explaining a removal would undo
    the removal. Both halves of this sweep were written naming their find, and
    both passed until the name came out.

    Erring towards keeping: a suffix that merely looks like the right word
    passes. What it will not do any more is pass a whole family unread.

    `sources` maps a filename to its text; only the `.py` ones are parsed for
    literals, and every one of them is searched for the key written in full.
    """
    prefixes, literals = _assembled(sources)
    blob = "\n".join(sources.values())
    orphans = []
    for key in catalog:
        # A plain substring match also fires on a key that is only the prefix
        # of a longer live key (`engine.ollama` inside `engine.ollama.found`),
        # which hid a dead key for as long as a live sibling of it existed. A
        # real occurrence never continues into more key characters, so the
        # character right after the match is checked and rejected when it
        # would extend the key instead of ending it.
        if re.search(re.escape(key) + r"(?![A-Za-z0-9_.])", blob):
            continue
        # Every matching prefix is tried, not the first one found. Two of them
        # nest here (`onboarding.task.` and `onboarding.task.ruler.`), and
        # asking only one of them turns a key the other assembles into an
        # orphan.
        if any(key.startswith(prefix) and key[len(prefix):] in literals
               for prefix in prefixes):
            continue
        orphans.append(key)
    return sorted(orphans)
