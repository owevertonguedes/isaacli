"""`/config`: the session preferences that used to exist only inside config.json.

Three settings were readable by the program and reachable by nobody. Context
management was on unless the configuration said otherwise, with no way to say
otherwise except opening the file. The local window and the temperature lived
in the active profile, were read on every request, and no screen ever showed
them. Editing JSON by hand is exactly what a preferences screen exists to stop.

Every row says where its value came from, because that is the question the file
could never answer. A number somebody typed into config.json in a test session
is not a number the user chose, and the screen must not present the two as the
same thing: that is why setting a value from here also records that it was set
from here.
"""
import json
import urllib.request

import config
import debug
import terminal_ui
from cli_i18n import t
from cli_presentation import _short_context

# Windows worth offering. Not a claim about any model: it is the set of round
# numbers a server is normally started with, and the last row exists so the
# answer "I do not want to declare one" stays reachable.
WINDOW_CHOICES = (4096, 8192, 16384, 32768, 65536, 131072)

# Steps, not a slider, for the same reason. 0 is the value every measurement in
# this repository was taken at, so it stays first.
TEMPERATURE_CHOICES = (0.0, 0.2, 0.5, 0.7, 1.0)

# Where a value came from. `hand` is the honest answer for a number that is in
# the file with nothing saying a screen put it there, which is the state
# `num_ctx: 8192` was found in.
ORIGIN_CHOSEN = "cli.config.origin.chosen"
ORIGIN_DEFAULT = "cli.config.origin.default"
ORIGIN_HAND = "cli.config.origin.hand"
# "Not set" does not mean the same thing for every row, and one label for both
# said, of the temperature, that the server's window is whatever started it.
# Each row names the key that describes its own absence.
ORIGIN_UNSET = "cli.config.origin.unset"
ORIGIN_INHERITED = "cli.config.origin.inherited"


def _origin(item, key, unset=ORIGIN_UNSET):
    """Whether a profile value was chosen on this screen or written by hand."""
    if item is None or item.get(key) is None:
        return unset
    sources = item.get("chosen_in_isaacli") or []
    return ORIGIN_CHOSEN if key in sources else ORIGIN_HAND


def _record_choice(item, key, value):
    """Store the value and the fact that a screen is what put it there."""
    sources = [name for name in (item.get("chosen_in_isaacli") or [])
               if name != key]
    if value is None:
        item.pop(key, None)
    else:
        item[key] = value
        sources.append(key)
    if sources:
        item["chosen_in_isaacli"] = sorted(sources)
    else:
        item.pop("chosen_in_isaacli", None)


def server_window(provider, timeout=2):
    """What the local server answers that it is actually running, or None.

    The profile's number is what isaacli sends; it is not evidence of what the
    server was started with. llama-server is normally started by a script
    outside this repository, and the `-c` in that script wins. `/props` is the
    only way to stop guessing, so it is asked, and when it cannot be asked the
    screen says so instead of presenting the profile's number as the truth.
    """
    base = (provider or {}).get("base_url")
    if not base or not config.is_local_endpoint(base):
        return None
    root = base.rstrip("/")
    for suffix in ("/v1", "/v1/"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    try:
        with urllib.request.urlopen(f"{root}/props", timeout=timeout) as answer:
            body = json.load(answer)
    except Exception:
        debug.swallowed("cli_config.server_window")
        return None
    settings = body.get("default_generation_settings")
    window = (settings or {}).get("n_ctx") if isinstance(settings, dict) else None
    if window is None:
        window = body.get("n_ctx")
    try:
        return int(window) if window else None
    except (TypeError, ValueError):
        debug.note("cli_config.server_window",
                   f"the server answered a window that is not a number: {window!r}")
        return None


def _menu(title, explanation, options, initial=0):
    return terminal_ui.select(
        f"{title}\n{terminal_ui.dim(explanation)}\n", options,
        prompt=t("select.prompt"), invalid=t("select.invalid"), initial=initial,
        more_above=t("ui.more_above", count="{count}"),
        more_below=t("ui.more_below", count="{count}"),
    )


class ConfigMixin:
    def config_screen(self):
        """The preferences screen. Returns the line to leave on the conversation."""
        try:
            data = config.load(self.config_file)
        except ValueError as e:
            print(t("cli.config.error", error=e))
            return
        message = t("cli.config.unchanged")
        cursor = 0
        while True:
            name, item = config.profile(data)
            rows = self._config_rows(data, item)
            options = [row for _handler, row in rows] + [t("cli.config.close")]
            try:
                cursor = _menu(t("cli.config.title"), t("cli.config.explain"),
                               options, initial=cursor)
            except (EOFError, KeyboardInterrupt):
                # Ctrl+C on the list leaves the screen. It must not travel any
                # further: the caller's own handler would read it as the user
                # abandoning the session rather than closing a menu.
                break
            if cursor >= len(rows):
                break
            try:
                outcome = rows[cursor][0](data, name, item)
            except (EOFError, KeyboardInterrupt):
                # Backing out of one setting is not backing out of the screen.
                continue
            if outcome:
                message = outcome
        self.redraw_session(message)

    def _config_rows(self, data, item):
        """One row per setting: what it is, what it is now, where that came from."""
        chosen = "context_management" in data
        window = (item or {}).get("num_ctx")
        temperature = (item or {}).get("temperature")
        return [
            (self._config_context_management, t(
                "cli.config.row",
                label=t("cli.config.label.context_management"),
                value=t("cli.config.value.on" if self.manage_context
                        else "cli.config.value.off"),
                origin=t(ORIGIN_CHOSEN if chosen else ORIGIN_DEFAULT))),
            (self._config_window, t(
                "cli.config.row",
                label=t("cli.config.label.window"),
                value=(_short_context(window) if window
                       else t("cli.config.value.unset")),
                origin=t(_origin(item, "num_ctx", ORIGIN_INHERITED)))),
            (self._config_temperature, t(
                "cli.config.row",
                label=t("cli.config.label.temperature"),
                value=(str(temperature) if temperature is not None
                       else t("cli.config.value.unset")),
                origin=t(_origin(item, "temperature")))),
        ]

    def _config_context_management(self, data, _name, _item):
        options = [t("cli.config.context.on"), t("cli.config.context.off")]
        index = _menu(t("cli.config.context.title"),
                      t("cli.config.context.explain"), options,
                      initial=0 if self.manage_context else 1)
        self.manage_context = index == 0
        data["context_management"] = self.manage_context
        config.save(data, self.config_file)
        self._log("meta", event="context_management", value=self.manage_context)
        # Saying it again on the way out is the point: off is not "turn off the
        # interruption", it is "let the request fail with its cause on screen".
        return t("cli.config.context.now_on" if self.manage_context
                 else "cli.config.context.now_off")

    def _config_window(self, data, name, item):
        if not item:
            return t("cli.config.no_profile")
        running = server_window(self.provider)
        explanation = (
            t("cli.config.window.server_says", tokens=running) if running
            else t("cli.config.window.server_silent"))
        options = [t("cli.config.window.option", tokens=_short_context(value))
                   for value in WINDOW_CHOICES] + [t("cli.config.window.unset")]
        current = item.get("num_ctx")
        initial = (WINDOW_CHOICES.index(current) if current in WINDOW_CHOICES
                   else len(options) - 1)
        index = _menu(t("cli.config.window.title"), explanation, options,
                      initial=initial)
        value = WINDOW_CHOICES[index] if index < len(WINDOW_CHOICES) else None
        _record_choice(item, "num_ctx", value)
        data["profiles"][name] = item
        config.save(data, self.config_file)
        self.num_ctx = value
        self._log("meta", event="num_ctx", profile=name, num_ctx=value)
        if value is None:
            return t("cli.config.window.cleared")
        return t("cli.config.window.set", tokens=_short_context(value))

    def _config_temperature(self, data, name, item):
        if not item:
            return t("cli.config.no_profile")
        options = [t("cli.config.temperature.option", value=value)
                   for value in TEMPERATURE_CHOICES]
        options.append(t("cli.config.temperature.unset"))
        current = item.get("temperature")
        initial = (TEMPERATURE_CHOICES.index(current)
                   if current in TEMPERATURE_CHOICES else len(options) - 1)
        index = _menu(t("cli.config.temperature.title"),
                      t("cli.config.temperature.explain"), options,
                      initial=initial)
        value = (TEMPERATURE_CHOICES[index] if index < len(TEMPERATURE_CHOICES)
                 else None)
        _record_choice(item, "temperature", value)
        data["profiles"][name] = item
        config.save(data, self.config_file)
        self.temperature = value
        self._log("meta", event="temperature", profile=name, temperature=value)
        if value is None:
            return t("cli.config.temperature.cleared")
        return t("cli.config.temperature.set", value=value)
