"""Terminal UI with no third-party dependency: arrow keys on a TTY, numbers as
a fallback.

The wording every menu needs when a caller supplies none comes from the
catalog, like all other text the user reads. It is resolved inside the call and
never in a default argument: a default is evaluated once, when this module is
imported, which is before the session has chosen a language, so a `t(...)` in
the signature would freeze English into every screen.
"""
from contextlib import contextmanager
import os
import re
import shutil
import sys

from cli_i18n import t
from cli_presentation import wrap_text


_ALT_DEPTH = 0


def interactive(input_fn=input):
    return input_fn is input and sys.stdin.isatty() and sys.stdout.isatty()


def clear(input_fn=input):
    """Start from an empty screen *and* an empty scrollback.

    Without \\033[3J the shell session that came before stays one wheel turn
    above the conversation. Since the REPL lives on the main screen, its
    scrollback is the conversation: nothing else belongs in it.
    """
    if interactive(input_fn):
        sys.stdout.write("\033[H\033[2J\033[3J")
        sys.stdout.flush()


def dim(text, input_fn=input):
    return f"\033[2m{text}\033[0m" if interactive(input_fn) else text


@contextmanager
def busy_input(input_fn=input, fd=None):
    """Keep scrolling/keystrokes from being echoed while the agent works.

    ISIG stays on, so Ctrl+C still interrupts normally. On the way out we discard
    sequences that arrived while no prompt was active. ``fd`` exists so the
    behaviour can be tested in a pseudo-terminal.
    """
    active = fd is not None or interactive(input_fn)
    if not active:
        yield
        return

    import termios

    input_fd = sys.stdin.fileno() if fd is None else fd
    previous = termios.tcgetattr(input_fd)
    busy = termios.tcgetattr(input_fd)
    busy[3] &= ~(termios.ECHO | termios.ICANON)
    if hasattr(termios, "ECHOCTL"):
        busy[3] &= ~termios.ECHOCTL
    busy[3] |= termios.ISIG
    busy[6][termios.VMIN] = 0
    busy[6][termios.VTIME] = 0
    try:
        termios.tcsetattr(input_fd, termios.TCSADRAIN, busy)
        yield
    finally:
        # Do not hand the next prompt arrow keys/scrolling typed during generation.
        termios.tcflush(input_fd, termios.TCIFLUSH)
        termios.tcsetattr(input_fd, termios.TCSADRAIN, previous)


@contextmanager
def alternate_screen(input_fn=input):
    """Isolate the wizard from the terminal's visible history."""
    global _ALT_DEPTH
    active = interactive(input_fn)
    first = active and _ALT_DEPTH == 0
    if active:
        _ALT_DEPTH += 1
    if first:
        sys.stdout.write("\033[?1049h\033[H\033[2J")
        sys.stdout.flush()
    try:
        yield
    finally:
        if active:
            _ALT_DEPTH -= 1
        if first:
            sys.stdout.write("\033[?25h\033[0m\033[?1049l")
            sys.stdout.flush()


ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Both menus move the cursor with these as well as with the arrow keys, so no
# screen may claim them as an answer shortcut.
NAVIGATION_KEYS = ("j", "k")


def _wording(prompt, invalid, more_above, more_below):
    """The catalog's wording for whatever the caller left out.

    `count` is handed back as the literal `{count}` because both scroll hints
    are formatted again later, with the real number, by whoever draws them.
    """
    return (
        t("select.prompt") if prompt is None else prompt,
        t("select.invalid") if invalid is None else invalid,
        t("ui.more_above", count="{count}") if more_above is None else more_above,
        t("ui.more_below", count="{count}") if more_below is None else more_below,
    )


def fit(text, width):
    """One option on one line, cut on purpose rather than by the terminal.

    A Kaggle model row runs to 138 characters. The menu counts every option as
    one screen line, so on an 80 column terminal the terminal wraps them itself
    and the window the menu believes it is drawing stops being the window on
    screen: rows scroll off the top and the count of what is below is wrong.
    What gets cut is the tail, because the beginning is what names the option,
    and for the row that is finally chosen the evidence is printed in full
    underneath it anyway.
    """
    if width <= 0:
        return ""
    visible = ANSI.sub("", text)
    if len(visible) <= width:
        return text
    if visible != text:
        # Cutting inside an escape sequence would leave the rest of the screen
        # painted in whatever colour was half written.
        return text
    return text[: max(0, width - 1)] + "\u2026"


def option_lines(options, width, cursor, disabled, indent=3):
    """Exactly what the menu writes for each option, one line each."""
    lines = []
    for position, option in enumerate(options):
        body = fit(option, max(1, width - indent))
        if position in disabled:
            lines.append(f"   \033[2m{body}\033[0m")
            continue
        mark = "\u276f" if position == cursor else " "
        highlight = "\033[1;36m" if position == cursor else ""
        reset = "\033[0m" if highlight else ""
        lines.append(f" {mark} {highlight}{body}{reset}")
    return lines


def select(title, options, input_fn=input, prompt=None, invalid=None,
           initial=0, disabled=None, more_above=None, more_below=None):
    if not options:
        raise ValueError("select requires at least one option")
    prompt, invalid, more_above, more_below = _wording(
        prompt, invalid, more_above, more_below)
    disabled = set(disabled or ())
    selectable = [i for i in range(len(options)) if i not in disabled]
    if not selectable:
        raise ValueError("select requires at least one enabled option")
    if not interactive(input_fn):
        print(wrap_text(title))
        number_to_index = []
        for i, option in enumerate(options):
            if i in disabled:
                print(f"  {option}")
            else:
                number_to_index.append(i)
                print(f"  {len(number_to_index)}) {option}")
        while True:
            value = input_fn(prompt).strip()
            try:
                number = int(value) - 1
            except ValueError:
                number = -1
            if 0 <= number < len(number_to_index):
                return number_to_index[number]
            print(invalid)

    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    index = min(selectable, key=lambda i: abs(i - initial))

    def move(direction):
        position = selectable.index(index)
        return selectable[(position + direction) % len(selectable)]

    def render():
        # Redrawing the whole screen also works when a long option wraps onto
        # more than one physical line.
        size = shutil.get_terminal_size((80, 24))
        width = max(size.columns, 20)
        ansi = re.compile(r"\x1b\[[0-9;]*m")
        # The explanation under a title is a paragraph written as one line, so
        # it is broken here by word. Left to the terminal it breaks on the
        # column, in the middle of words, on the very screen where the user is
        # reading what a choice costs.
        shown = wrap_text(title, width)
        title_lines = sum(
            max(1, (len(ansi.sub("", line)) + width - 1) // width)
            for line in shown.splitlines()
        )
        capacity = max(5, size.lines - title_lines - 4)
        start = max(0, index - capacity // 2)
        end = min(len(options), start + capacity)
        start = max(0, end - capacity)
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.write(shown.replace("\n", "\r\n") + "\r\n")
        if start:
            label = more_above.format(count=start)
            sys.stdout.write(f"   \033[2m{label}\033[0m\r\n")
        for line in option_lines(
                options[start:end], width, index - start,
                {position - start for position in disabled}):
            sys.stdout.write(line + "\r\n")
        if end < len(options):
            label = more_below.format(count=len(options) - end)
            sys.stdout.write(f"   \033[2m{label}\033[0m\r\n")
        sys.stdout.flush()

    with alternate_screen(input_fn):
        try:
            tty.setraw(fd)
            sys.stdout.write("\033[?25l")
            render()
            while True:
                key = os.read(fd, 1)
                if key in (b"\r", b"\n"):
                    return index
                if key in (b"k", b"K"):
                    index = move(-1)
                    render()
                elif key in (b"j", b"J"):
                    index = move(1)
                    render()
                elif key == b"\x1b":
                    sequence = os.read(fd, 2)
                    if sequence == b"[A":
                        index = move(-1)
                        render()
                    elif sequence == b"[B":
                        index = move(1)
                        render()
                elif key == b"\x03":
                    raise KeyboardInterrupt
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous)
            sys.stdout.write("\033[?25h\033[0m")
            sys.stdout.flush()


def select_inline(options, shortcuts=None, input_fn=input, initial=0,
                  prompt=None, chosen_label="{option}"):
    """Arrow-key menu that does not clear the conversation already on screen.

    The defaults name no screen on purpose. They used to be the command
    approval screen's own wording, back when it was the only caller, and the
    second caller inherited it: the context screen confirmed a decision about
    the model's window with the word "Permission". Every caller passes its own
    translated wording, so a default that belongs to one of them is a trap and
    never a saving.
    """
    if not options:
        raise ValueError("inline select requires at least one option")
    prompt = t("select.prompt") if prompt is None else prompt
    shortcuts = shortcuts or {}
    # Shortcuts are read before navigation below, so binding one of these would
    # turn the key that moves the cursor into the key that answers. The context
    # screen bound "k" and pressing it, to go up, chose "leave it as it is" on
    # the spot. Refused loudly rather than quietly ignored: a shortcut that
    # silently does nothing is the same class of bug pointing the other way.
    colliding = sorted(set(shortcuts) & set(NAVIGATION_KEYS))
    if colliding:
        raise ValueError(
            "inline select shortcuts cannot use the navigation keys: "
            + ", ".join(colliding))
    if not interactive(input_fn):
        for i, option in enumerate(options, 1):
            print(f"  {i}) {option}")
        value = input_fn(prompt).strip().lower()
        if value == "":
            return initial
        if value in shortcuts:
            return shortcuts[value]
        try:
            index = int(value) - 1
        except ValueError:
            return len(options) - 1
        return index if 0 <= index < len(options) else len(options) - 1

    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    index = min(max(initial, 0), len(options) - 1)

    def draw(move_up=False):
        if move_up:
            sys.stdout.write(f"\033[{len(options)}A")
        for position, option in enumerate(options):
            cursor = "❯" if position == index else " "
            highlight = "\033[1;36m" if position == index else ""
            reset = "\033[0m" if highlight else ""
            sys.stdout.write(f"\r\033[2K {cursor} {highlight}{option}{reset}\r\n")
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        sys.stdout.write("\033[?25l")
        draw()
        while True:
            key = os.read(fd, 1)
            if key in (b"\r", b"\n"):
                break
            if key.decode(errors="ignore").lower() in shortcuts:
                index = shortcuts[key.decode(errors="ignore").lower()]
                break
            if key in (b"k", b"K"):
                index = (index - 1) % len(options)
                draw(move_up=True)
            elif key in (b"j", b"J"):
                index = (index + 1) % len(options)
                draw(move_up=True)
            elif key == b"\x1b":
                sequence = os.read(fd, 2)
                if sequence == b"[A":
                    index = (index - 1) % len(options)
                    draw(move_up=True)
                elif sequence == b"[B":
                    index = (index + 1) % len(options)
                    draw(move_up=True)
            elif key == b"\x03":
                raise KeyboardInterrupt
        sys.stdout.write(f"\033[{len(options)}A")
        for _ in options:
            sys.stdout.write("\r\033[2K\033[1B")
        sys.stdout.write(f"\033[{len(options)}A\r\033[2K")
        # The terminal is still raw: a bare \n does not return to column zero.
        sys.stdout.write(chosen_label.format(option=options[index]) + "\r\n")
        sys.stdout.flush()
        return index
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        sys.stdout.write("\033[?25h\033[0m")
        sys.stdout.flush()
