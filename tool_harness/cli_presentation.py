"""Terminal presentation: ANSI colour, Markdown rendering, the welcome panel.

Pure functions, entering and leaving strings. No session state lives here.
"""
import os
import re
import shutil
import sys
from pathlib import Path

from cli_i18n import t

APP_VERSION = "0.7.1"
WORDMARK_ISAAC = tuple(line.ljust(23) for line in (
    "╻  ┏━╸  ┏━┓  ┏━┓  ┏━╸",
    "┃  ┗━┓  ┣━┫  ┣━┫  ┃",
    "╹  ╺━┛  ╹ ╹  ╹ ╹  ┗━╸",
))
ANSI = {
    "prompt": "\033[1;36m",
    "assistant": "\033[1;32m",
    "tool": "\033[1;34m",
    "warn": "\033[1;33m",
    "bad": "\033[1;31m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "reset": "\033[0m",
    # Match argparse's own Python 3.13+ --help colours (captured from a real
    # isaacli --help run) so the "commands" epilog reads as one section with
    # "options:"/"positional arguments:", not flat text next to styled text.
    "help_header": "\033[1;34m",
    "help_positional": "\033[1;32m",
    "help_flag": "\033[1;36m",
}


def _uses_color():
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _color(text, name):
    if not _uses_color():
        return text
    return f"{ANSI[name]}{text}{ANSI['reset']}"


def _colored_prompt(text, name="prompt"):
    """Mark ANSI as non-printing so readline computes line wrapping correctly."""
    if not _uses_color():
        return text
    return f"\001{ANSI[name]}\002{text}\001{ANSI['reset']}\002"


def _terminal_safe_text(text):
    """Strip controls a model's answer must not be able to send to the terminal."""
    text = re.sub(
        r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)?|.)",
        "", str(text),
    )
    return "".join(
        character for character in text
        if character in "\n\t" or ord(character) >= 32 and ord(character) != 127
    )


def _markdown_inline(text, colors=True):
    """Render the most common inline Markdown subset LLMs emit."""
    if not colors:
        return text
    out = []
    i = 0
    styles = {"bold": False, "italic": False, "strike": False, "code": False}

    def toggle(name, ansi):
        styles[name] = not styles[name]
        return ansi if styles[name] else ANSI["reset"]

    while i < len(text):
        if text.startswith("**", i) or text.startswith("__", i):
            out.append(toggle("bold", "\033[1m"))
            i += 2
            continue
        if text.startswith("~~", i):
            out.append(toggle("strike", "\033[9m"))
            i += 2
            continue
        if text[i] == "`":
            out.append(toggle("code", "\033[38;5;81m"))
            i += 1
            continue
        if text[i] == "*":
            out.append(toggle("italic", "\033[3m"))
            i += 1
            continue
        if text[i] == "[":
            label_end = text.find("](https://", i)
            if label_end < 0:
                label_end = text.find("](http://", i)
            if label_end >= 0:
                url_end = text.find(")", label_end + 2)
                if url_end >= 0:
                    label = text[i + 1:label_end]
                    url = text[label_end + 2:url_end]
                    out.append(
                        f"\033[4m{label}{ANSI['reset']} {ANSI['dim']}<{url}>{ANSI['reset']}")
                    i = url_end + 1
                    continue
        out.append(text[i])
        i += 1
    if any(styles.values()):
        out.append(ANSI["reset"])
    return "".join(out)


_ANSI_SGR = re.compile(r"\033\[[0-9;]*m")


def _terminal_columns():
    return max(shutil.get_terminal_size((100, 24)).columns, 20)


def _wrap_ansi(text, width, subsequent_indent="", first_offset=0):
    """Break a styled line at spaces instead of letting the terminal cut words.

    A terminal wraps on the column, so it splits words in half. Wrapping here
    is the only place that can do it by word, because only here is it known
    which characters are visible: an ANSI sequence takes columns from nothing,
    and counting it would make every styled line wrap early.

    Style survives the break. The sequences active at the cut close at the end
    of the line and open again on the next one, so a break in the middle of a
    bold run does not leave the rest of the paragraph unstyled.

    A word longer than the whole width (a path, a URL) is never cut: it goes on
    its own line and the terminal does what it always did, which is the lesser
    evil against a broken path nobody can copy.
    """
    if width is None or width <= 1:
        return [text]
    # A line the terminal was never going to break comes back untouched. That
    # keeps every existing rendering exactly as it was, alignment included, and
    # limits this to the lines that were being cut in half.
    if first_offset + len(_ANSI_SGR.sub("", text)) <= width:
        return [text]
    words = _styled_words(text)
    if not words:
        return [text]

    lines = []
    line = ""
    column = first_offset
    indent_width = len(_ANSI_SGR.sub("", subsequent_indent))
    for word, visible, styles in words:
        separator = 1 if line else 0
        if line and column + separator + visible > width:
            lines.append(line + (ANSI["reset"] if styles else ""))
            line = subsequent_indent + "".join(styles) + word
            column = indent_width + visible
            continue
        line += (" " if separator else "") + word
        column += separator + visible
    lines.append(line)
    return lines


def _styled_words(text):
    """Split a styled line into (text, visible width, styles open at its start).

    The escape sequences stay attached to the word they were written against,
    so joining the words back with single spaces reproduces the original line.
    The third field is what a continuation line has to re-open after a break.
    """
    words = []
    active = []
    current, current_width, current_active = "", 0, None
    position = 0
    for match in list(_ANSI_SGR.finditer(text)) + [None]:
        chunk = text[position:match.start()] if match else text[position:]
        for piece in re.split(r"(\s+)", chunk):
            if not piece:
                continue
            if piece.isspace():
                if current_width:
                    words.append((current, current_width, current_active))
                    current, current_width, current_active = "", 0, None
                continue
            if current_active is None:
                current_active = list(active)
            current += piece
            current_width += len(piece)
        if match is None:
            break
        code = match.group()
        # Attached even with no visible character yet: it styles the word that
        # comes next, and dropping it here would drop the colour from the line.
        current += code
        if code in ("\033[0m", "\033[m"):
            active = []
        else:
            active.append(code)
        position = match.end()
    if current_width:
        words.append((current, current_width, current_active))
    elif current and words:
        # Trailing sequences with nothing after them, typically the reset that
        # closes the line. They belong to the last word, or styling leaks past
        # the end of the line.
        last, last_width, last_active = words[-1]
        words[-1] = (last + current, last_width, last_active)
    return words


def wrap_text(text, width=None):
    """A block of the program's own prose, broken at spaces to fit the screen.

    Line by line, never as one blob: a notice with a second paragraph in it
    has real newlines that mean something, and wrapping the whole string at
    once would swallow them and glue the paragraphs together.

    A width of None means no wrapping at all, which is what redirected output
    gets: there a command or a path has to survive being copied out of a log
    by grep, and nothing is choosing the column anyway.
    """
    if width is None:
        width = _terminal_columns() if sys.stdout.isatty() else None
    wrapped = []
    for line in str(text).split("\n"):
        # Wrapping joins the words back with single spaces, so a space somebody
        # put at the end on purpose is lost. On a question that is exactly what
        # separates the cursor from the colon it answers.
        body = line.rstrip()
        pieces = _wrap_ansi(body, width)
        pieces[-1] += line[len(body):]
        wrapped.extend(pieces)
    return "\n".join(wrapped)


def say(text, style=None, width=None, file=None, flush=False, end="\n"):
    """Print a sentence to the user, wrapped by word.

    For the program's own prose, and the reason it exists instead of `print`:
    a notice is written as one long line in the catalogue, on purpose, so that
    the screen decides where it breaks. Left to `print`, what decides is the
    terminal, which breaks on the column and cuts words in half. Anything
    already fitting comes out exactly as before.

    `file` and `flush` are here so that no caller has to choose between being
    wrapped and going to stderr.
    """
    body = wrap_text(text, width)
    print(_color(body, style) if style else body,
          **({"file": file} if file is not None else {}),
          flush=flush, end=end)


def _format_markdown_terminal(text, colors=None, width=None, first_offset=0):
    """Turn chat Markdown into a simple, safe ANSI presentation."""
    text = _terminal_safe_text(text)
    colors = _uses_color() if colors is None else colors
    if not colors:
        return text
    if width is None:
        width = _terminal_columns()
    pending_offset = first_offset

    def laid_out(prefix, prefix_width, content, indent):
        """One rendered block: its prefix, then the content wrapped by word.

        `pending_offset` is what the caller already printed on this line (the
        "isaac:" label), and it only counts against the very first line.
        """
        nonlocal pending_offset
        pieces = _wrap_ansi(content, width, subsequent_indent=indent,
                            first_offset=prefix_width + pending_offset)
        pending_offset = 0
        return prefix + "\n".join(pieces)

    lines = []
    in_code = False
    for line in text.splitlines(keepends=True):
        ending = "\n" if line.endswith("\n") else ""
        body = line[:-1] if ending else line
        fence = re.match(r"^\s*```\s*([^`]*)$", body)
        if fence:
            if in_code:
                lines.append(ANSI["reset"] + ending)
                in_code = False
            else:
                language = fence.group(1).strip()
                word = t("cli.markdown.code")
                label = f" {word} · {language} " if language else f" {word} "
                lines.append(f"{ANSI['dim']}──{label}────────────────{ANSI['reset']}" + ending)
                in_code = True
            continue
        if in_code:
            lines.append(f"\033[38;5;81m  {body}{ANSI['reset']}" + ending)
            continue
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+)$", body)
        if heading:
            lines.append(laid_out(
                f"{ANSI['assistant']}▌{ANSI['reset']} ", 2,
                f"\033[1m{_markdown_inline(heading.group(1), colors=True)}"
                f"{ANSI['reset']}", "  ") + ending)
            continue
        quote = re.match(r"^\s*>\s?(.*)$", body)
        if quote:
            bar = f"{ANSI['dim']}│{ANSI['reset']} "
            lines.append(laid_out(
                bar, 2, _markdown_inline(quote.group(1), colors=True), bar) + ending)
            continue
        item = re.match(r"^(\s*)[-+*]\s+(.*)$", body)
        if item:
            content = item.group(2)
            content = re.sub(r"^\[ \]\s*", "☐ ", content)
            content = re.sub(r"^\[[xX]\]\s*", "☑ ", content)
            spacing = item.group(1)
            lines.append(laid_out(
                f"{spacing}{ANSI['assistant']}•{ANSI['reset']} ", len(spacing) + 2,
                _markdown_inline(content, colors=True), spacing + "  ") + ending)
            continue
        numbered = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", body)
        if numbered:
            spacing, number = numbered.group(1), numbered.group(2)
            marker = len(spacing) + len(number) + 2
            lines.append(laid_out(
                f"{spacing}{ANSI['assistant']}{number}.{ANSI['reset']} ", marker,
                _markdown_inline(numbered.group(3), colors=True),
                " " * marker) + ending)
            continue
        if re.match(r"^\s*(?:---+|___+|\*\*\*+)\s*$", body):
            lines.append(f"{ANSI['dim']}{'─' * 32}{ANSI['reset']}" + ending)
            continue
        indent = re.match(r"^\s*", body).group()
        lines.append(laid_out(
            indent, len(indent),
            _markdown_inline(body[len(indent):], colors=True), indent) + ending)
    if in_code:
        lines.append(ANSI["reset"])
    return "".join(lines)


def _shorten(text, limit):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:max(1, limit - 1)] + "…"


def _visual_width(text):
    return len(str(text))


def _pad_visual(text, width, alignment="left"):
    missing = max(0, width - _visual_width(text))
    if alignment == "center":
        left = missing // 2
        return " " * left + text + " " * (missing - left)
    return text + " " * missing


def _friendly_path(path):
    text = str(path)
    home = str(Path.home())
    return "~" + text[len(home):] if text == home or text.startswith(home + os.sep) else text


def _short_context(value):
    if isinstance(value, int) and value % 1024 == 0:
        return f"{value // 1024}K"
    return str(value)


def _welcome_lines(model, engine, workspace, session=None, width=None):
    """Build the opening panel without ANSI so the alignment stays predictable."""
    columns = width if width is not None else shutil.get_terminal_size((100, 24)).columns - 2
    width = max(36, min(columns, 112))
    inner = width - 2
    lines = []

    def body(text=""):
        text = _shorten(text, inner - 2)
        lines.append("│ " + _pad_visual(text, inner - 2) + " │")

    if width >= 88:
        left_width = (inner - 5) // 2
        right_width = inner - left_width - 5
        title = f"─── Isaac CLI v{APP_VERSION} "
        lines.append(
            "╭" + title + "─" * max(0, left_width + 2 - len(title))
            + "┬" + "─" * (right_width + 2) + "╮"
        )
        left_side = [
            "",
            "",
            *WORDMARK_ISAAC,
            "",
            "",
            "",
        ]
        right_side = [
            t("cli.welcome.greeting"),
            "",
            t("cli.welcome.getting_started"),
            f"{'/help':<9} " + t("cli.cmd.help"),
            f"{'/setup':<9} " + t("cli.welcome.setup"),
            f"{'/status':<9} " + t("cli.welcome.status"),
            f"{'/history':<9} " + t("cli.welcome.history"),
            t("cli.welcome.shift_tab"),
        ]
        for a, b in zip(left_side, right_side):
            lines.append(
                "│ " + _pad_visual(_shorten(a, left_width), left_width, "center") + " │ "
                + _pad_visual(_shorten(b, right_width), right_width) + " │"
            )
        lines.append("├" + "─" * (left_width + 2) + "┴" + "─" * (right_width + 2) + "┤")
    else:
        title = f"╭─── Isaac CLI v{APP_VERSION} "
        lines.append(title + "─" * max(0, width - len(title) - 1) + "╮")
        body(t("cli.welcome.greeting"))
        for wordmark_line in WORDMARK_ISAAC:
            body(_pad_visual(wordmark_line, inner - 2, "center"))
        body(t("cli.welcome.compact"))
        lines.append("├" + "─" * inner + "┤")

    lines.extend(_summary_rows(model, engine, workspace, session, inner))
    lines.append("╰" + "─" * inner + "╯")
    return lines


def _summary_rows(model, engine, workspace, session, inner):
    """The label/value rows at the foot of the panel, without the frame's ends.

    Shared with _print_session_summary so the box a model change reprints says
    the same things, in the same order, as the one the session opened with.
    """
    value_width = inner - 13
    rows = [
        (t("cli.welcome.label.model"), model),
        (t("cli.welcome.label.engine"), engine),
        (t("cli.welcome.label.workspace"), _friendly_path(workspace)),
    ]
    # The session id is what --resume takes, so it belongs where the user can
    # copy it before the conversation scrolls it away.
    if session:
        rows.append((t("cli.welcome.label.session"), session))
    out = []
    for label, content in rows:
        text = _shorten(f"{label:<10} {_shorten(content, value_width)}", inner - 2)
        out.append("│ " + _pad_visual(text, inner - 2) + " │")
    return out


def _session_summary_lines(model, engine, workspace, session=None, width=None):
    """The panel's foot on its own, for reprinting after something in it changed.

    The opening panel is scrollback: once printed it cannot be edited, so a
    model chosen later left the old name sitting at the top of the screen as
    if nothing had happened. This prints the current answer where the user is
    looking, instead of asking them to trust the summary line.
    """
    columns = width if width is not None else shutil.get_terminal_size((100, 24)).columns - 2
    width = max(36, min(columns, 112))
    inner = width - 2
    return [
        "╭" + "─" * inner + "╮",
        *_summary_rows(model, engine, workspace, session, inner),
        "╰" + "─" * inner + "╯",
    ]


def _print_session_summary(model, engine, workspace, session=None):
    for line in _session_summary_lines(model, engine, workspace, session):
        decorated = line
        for glyph in "│╭╮╯╰─":
            decorated = decorated.replace(glyph, _color(glyph, "assistant"))
        print(decorated)


def _print_welcome(model, engine, workspace, session=None):
    for index, line in enumerate(_welcome_lines(model, engine, workspace, session)):
        if index == 0 or line.startswith(("├", "╰")):
            print(_color(line, "assistant"))
            continue
        # The divider and the frame use Isaac's green.
        decorated = line
        for glyph in "│╭╮╯╰─╱╲":
            decorated = decorated.replace(glyph, _color(glyph, "assistant"))
        # The wordmark is one visual unit: every letter gets the same colour.
        for piece in WORDMARK_ISAAC:
            decorated = decorated.replace(piece, _color(piece, "assistant"))
        print(decorated)
