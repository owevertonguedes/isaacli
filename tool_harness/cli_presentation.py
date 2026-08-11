"""Terminal presentation: ANSI colour, Markdown rendering, the welcome panel.

Pure functions, entering and leaving strings. No session state lives here.
"""
import os
import re
import shutil
import sys
from pathlib import Path

from cli_i18n import t

APP_VERSION = "0.4.0"
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


def _format_markdown_terminal(text, colors=None):
    """Turn chat Markdown into a simple, safe ANSI presentation."""
    text = _terminal_safe_text(text)
    colors = _uses_color() if colors is None else colors
    if not colors:
        return text
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
            lines.append(
                f"{ANSI['assistant']}▌{ANSI['reset']} \033[1m"
                f"{_markdown_inline(heading.group(1), colors=True)}{ANSI['reset']}" + ending
            )
            continue
        quote = re.match(r"^\s*>\s?(.*)$", body)
        if quote:
            lines.append(
                f"{ANSI['dim']}│{ANSI['reset']} "
                f"{_markdown_inline(quote.group(1), colors=True)}" + ending
            )
            continue
        item = re.match(r"^(\s*)[-+*]\s+(.*)$", body)
        if item:
            content = item.group(2)
            content = re.sub(r"^\[ \]\s*", "☐ ", content)
            content = re.sub(r"^\[[xX]\]\s*", "☑ ", content)
            lines.append(
                f"{item.group(1)}{ANSI['assistant']}•{ANSI['reset']} "
                f"{_markdown_inline(content, colors=True)}" + ending
            )
            continue
        numbered = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", body)
        if numbered:
            lines.append(
                f"{numbered.group(1)}{ANSI['assistant']}{numbered.group(2)}.{ANSI['reset']} "
                f"{_markdown_inline(numbered.group(3), colors=True)}" + ending
            )
            continue
        if re.match(r"^\s*(?:---+|___+|\*\*\*+)\s*$", body):
            lines.append(f"{ANSI['dim']}{'─' * 32}{ANSI['reset']}" + ending)
            continue
        lines.append(_markdown_inline(body, colors=True) + ending)
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


def _welcome_lines(model, engine, workspace, width=None):
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

    value_width = inner - 13
    for label, content in (
        (t("cli.welcome.label.model"), model),
        (t("cli.welcome.label.engine"), engine),
        (t("cli.welcome.label.workspace"), _friendly_path(workspace)),
    ):
        body(f"{label:<10} {_shorten(content, value_width)}")
    lines.append("╰" + "─" * inner + "╯")
    return lines


def _print_welcome(model, engine, workspace):
    for index, line in enumerate(_welcome_lines(model, engine, workspace)):
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
