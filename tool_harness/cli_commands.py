"""Slash-command dispatch: the palette, completion, and every /command handler
except the ones that belong to sessions, Ollama or providers.

internal_command is a single giant dispatch. The natural boundary is a
command-to-handler table, but that is a real refactor, not a file move, so it
stays as one long if-chain here (task note: don't mix mechanical movement
with behaviour change).
"""
import json
import os

try:
    from prompt_toolkit.completion import Completer, Completion
except ImportError:  # pragma: no cover - optional dependency
    Completer = object
    Completion = None

import config
import terminal_ui
import tools
from cli_i18n import set_language, t
from cli_presentation import _color, _short_context
from cli_sessions import FEEDBACK_DIR, _build_history, _now

# The catalogue key for each slash command's description. The text itself lives
# in locales/*.json so the palette and /help speak the chosen language.
COMMANDS = (
    ("/help", "cli.cmd.help"),
    ("/setup", "cli.cmd.setup"),
    ("/status", "cli.cmd.status"),
    ("/tools", "cli.cmd.tools"),
    ("/sessions", "cli.cmd.sessions"),
    ("/history", "cli.cmd.history"),
    ("/show", "cli.cmd.show"),
    ("/log", "cli.cmd.log"),
    ("/feedback", "cli.cmd.feedback"),
    ("/good", "cli.cmd.good"),
    ("/bad", "cli.cmd.bad"),
    ("/score", "cli.cmd.score"),
    ("/workspace", "cli.cmd.workspace"),
    ("/model", "cli.cmd.model"),
    ("/permissions", "cli.cmd.permissions"),
    ("/mode", "cli.cmd.mode"),
    ("/language", "cli.cmd.language"),
    ("/clear", "cli.cmd.clear"),
    ("/new", "cli.cmd.new"),
    ("/exit", "cli.cmd.exit"),
)
SLASH_COMMANDS = [command for command, _key in COMMANDS]

# Portuguese names kept as hidden aliases: they were the original spelling and
# breaking them would break muscle memory for no gain.
COMMAND_ALIASES = {"/bom": "/good", "/ruim": "/bad", "/nota": "/score"}


def _score_command(query, command, description):
    """Rank prefixes first and accept fuzzy matching on the rest."""
    query = query.casefold().strip()
    command_normal = command.casefold()
    description_normal = description.casefold()
    if not query or query == "/":
        return (0, SLASH_COMMANDS.index(command))
    if command_normal.startswith(query):
        return (0, len(command))
    term = query.removeprefix("/")
    name = command_normal.removeprefix("/")
    if term in name:
        return (1, name.index(term), len(name))
    if term in description_normal:
        return (2, description_normal.index(term), len(description_normal))
    position = 0
    distance = 0
    for character in term:
        found = name.find(character, position)
        if found < 0:
            return None
        distance += found - position
        position = found + 1
    return (3, distance, len(name))


def _filter_commands(query):
    found = []
    for command, key in COMMANDS:
        description = t(key)
        points = _score_command(query, command, description)
        if points is not None:
            found.append((points, command, description))
    found.sort(key=lambda item: item[0])
    return [(command, description) for _points, command, description in found]


class _CommandCompleter(Completer):
    def get_completions(self, document, _event):
        text = document.text_before_cursor
        if not text.startswith("/") or any(c.isspace() for c in text):
            return
        for command, description in _filter_commands(text):
            yield Completion(
                command,
                start_position=-len(text),
                display=command,
                display_meta=description,
            )


def _install_autocomplete():
    try:
        import readline
    except ImportError:  # pragma: no cover - Windows/minimal environment
        return
    if readline is None:
        return

    def complete(text, state):
        options = [c for c in SLASH_COMMANDS if c.startswith(text)]
        try:
            return options[state] + " "
        except IndexError:
            return None

    readline.set_completer(complete)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")
    # Shift+Tab sends /mode as if the user had typed it. Works in GNU readline
    # without replacing the line editor (and its correct wrapping).
    readline.parse_and_bind('"\\e[Z": "/mode\\n"')


class CommandsMixin:
    def help_screen(self):
        print(t("cli.help.body"))

    def internal_command(self, text):
        if not text.startswith("/"):
            return False
        if text == "/":
            self.help_screen()
            return True
        parts = text.split(maxsplit=1)
        cmd = COMMAND_ALIASES.get(parts[0], parts[0])
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            raise EOFError
        if cmd in ("/help", "/?"):
            self.help_screen()
            return True
        if cmd == "/setup":
            import setup_ollama
            code = setup_ollama.run_setup(config_file=self.config_file)
            if code == 0:
                try:
                    data = config.load(self.config_file)
                    name, item = config.profile(data)
                except ValueError as e:
                    self.redraw_session(t("cli.setup.reread_failed", error=e))
                    return True
                if item:
                    self.model = item["model"]
                    self.thinking = item.get("thinking")
                    self.num_ctx = item.get("num_ctx")
                    self.provider = self._provider_from_profile(item)
                    self._log("meta", event="setup", profile=name,
                              model=self.model, thinking=self.thinking)
                    self.redraw_session(
                        t("cli.setup.profile_loaded", name=name, model=self.model))
            else:
                self.redraw_session(
                    t("cli.setup.cancelled") if code == 130
                    else t("cli.setup.incomplete")
                )
            return True
        if cmd == "/status":
            self.status()
            return True
        if cmd == "/tools":
            self.list_tools()
            return True
        if cmd == "/sessions":
            self.list_sessions()
            return True
        if cmd == "/history":
            self.show_history(arg)
            return True
        if cmd == "/show":
            self.show_command(arg or "last")
            return True
        if cmd == "/log":
            print(self.session_path)
            return True
        if cmd == "/feedback":
            self.feedback_help()
            return True
        if cmd == "/good":
            self.save_feedback("good", 10, arg)
            return True
        if cmd == "/bad":
            self.save_feedback("bad", 0, arg)
            return True
        if cmd == "/score":
            self.score_command(arg)
            return True
        if cmd == "/workspace":
            if not arg:
                print(self.workspace)
            else:
                self.set_workspace(arg)
                print(t("cli.workspace.now", path=self.workspace))
            return True
        if cmd == "/model":
            if not arg:
                self.select_model()
            else:
                try:
                    data = config.load(self.config_file)
                except ValueError:
                    data = config.empty_config()
                item = (data.get("profiles") or {}).get(arg)
                if item:
                    self.model = item["model"]
                    self.thinking = item.get("thinking")
                    self.num_ctx = item.get("num_ctx")
                    self.provider = self._provider_from_profile(item)
                    source = t("cli.model.source.profile", name=arg)
                else:
                    self.model = arg
                    self.thinking = None
                    self.num_ctx = None
                    self.provider = {"provider": "ollama"}
                    source = t("cli.model.source.ollama")
                self._log("meta", event="model", model=self.model,
                          thinking=self.thinking)
                print(t("cli.model.set", model=self.model, source=source))
            return True
        if cmd == "/mode":
            self.permission_mode = (
                "authorized_only" if self.permission_mode == "safe" else "safe"
            )
            print(t("cli.mode.changed", mode=self._permission_mode_label()))
            self._log("meta", event="permission_mode", mode=self.permission_mode)
            return True
        if cmd == "/permissions":
            try:
                data = config.load(self.config_file)
            except ValueError as e:
                print(t("cli.config.error", error=e))
                return True
            permissions = data.get("permissions") or {}
            if arg in ("clear workspace", "clear global"):
                if arg == "clear global":
                    permissions["global"] = []
                else:
                    (permissions.get("workspaces") or {}).pop(str(self.workspace), None)
                config.save(data, self.config_file)
                print(t("cli.permissions.cleared", scope=arg.removeprefix("clear ")))
                return True
            global_rules = permissions.get("global") or []
            local_rules = (permissions.get("workspaces") or {}).get(str(self.workspace), [])
            none = t("cli.permissions.none")
            print(t("cli.permissions.mode", mode=self._permission_mode_label()))
            print(t("cli.permissions.global",
                    rules=", ".join(global_rules) if global_rules else none))
            print(t("cli.permissions.workspace",
                    rules=", ".join(local_rules) if local_rules else none))
            print(t("cli.permissions.clear_hint"))
            return True
        if cmd == "/language":
            from i18n import SUPPORTED_LANGUAGES
            try:
                data = config.load(self.config_file)
            except ValueError:
                data = config.empty_config()
            codes = list(SUPPORTED_LANGUAGES)
            current = data.get("language") or "en"
            initial = codes.index(current) if current in codes else 0
            index = terminal_ui.select(
                t("cli.language.title"),
                [SUPPORTED_LANGUAGES[code] for code in codes],
                initial=initial,
                more_above=t("ui.more_above", count="{count}"),
                more_below=t("ui.more_below", count="{count}"),
            )
            data["language"] = codes[index]
            config.save(data, self.config_file)
            self.tr = set_language(codes[index])
            self._log("meta", event="language", language=codes[index])
            # Full-screen selector: come back through redraw_session like /setup
            # and /model, otherwise the leftover menu stays on screen and the
            # conversation is gone.
            self.redraw_session(
                t("cli.language.set", language=SUPPORTED_LANGUAGES[codes[index]]))
            return True
        if cmd == "/clear":
            self.history = _build_history(self.workspace)
            self._log("meta", event="clear")
            print(t("cli.clear.done"))
            return True
        if cmd == "/new":
            self.new_session()
            return True

        print(t("cli.unknown_command", cmd=cmd))
        return True

    def status(self):
        # Lazy: cli.py imports this module, so importing cli back here at
        # call time (not at module load) is what avoids a circular import.
        import cli

        if self.provider.get("provider") == "ollama":
            engine = cli._ollama_ok() or t("cli.engine.no_response")
        else:
            engine = self.provider.get("provider_name") or t("cli.engine.openai_compatible")
        duration_s = self.total_usage.get("total_duration", 0) / 1_000_000_000
        default = t("cli.status.model_default")
        print(t("cli.status.session", id=self.session_id))
        print(t("cli.status.log", path=self.session_path))
        print(t("cli.status.pid", pid=os.getpid()))
        print(t("cli.status.model", model=self.model))
        print(t("cli.status.reasoning",
                value=self.thinking if self.thinking is not None else default))
        print(t("cli.status.context",
                value=_short_context(self.num_ctx) if self.num_ctx else default))
        print(t("cli.status.workspace", path=self.workspace))
        print(t("cli.status.engine", engine=engine))
        print(t("cli.status.turns", turns=self.turns))
        print(t("cli.status.commands", commands=len(self.commands), failures=self.failures))
        print(t("cli.status.permissions", mode=self._permission_mode_label()))
        print(t("cli.status.feedback", count=self.ratings, path=self.feedback_path))
        print(t("cli.status.tokens",
                prompt=self.total_usage.get("prompt_eval_count", 0),
                response=self.total_usage.get("eval_count", 0),
                seconds=f"{duration_s:.2f}"))
        print(t("cli.status.slash", commands=" ".join(SLASH_COMMANDS)))
        self._log("status", turns=self.turns, commands=len(self.commands),
                  failures=self.failures, usage=self.total_usage)

    def list_tools(self):
        import execution

        names = [s["function"]["name"] for s in tools.SCHEMA]
        print(t("cli.tools.list", names=", ".join(names)))
        print(t("cli.tools.terminal", names=", ".join(sorted(execution.ALLOWED))))
        print(t("cli.tools.git", names=", ".join(sorted(execution.GIT_ALLOWED))))

    def show_command(self, which):
        if not self.commands:
            print(t("cli.show.none"))
            return
        if which == "last":
            item = self.commands[-1]
        else:
            try:
                number = int(which)
            except ValueError:
                print(t("cli.show.usage"))
                return
            item = next((c for c in self.commands if c["id"] == number), None)
            if item is None:
                print(t("cli.show.missing", number=number))
                return
        print(_color(t("cli.show.full", id=item["id"], cmd=item["cmd"]), "tool"))
        print(item["result"])

    def feedback_help(self):
        print(t("cli.feedback.body",
                feedback=self.feedback_path, session=self.session_path))

    def score_command(self, arg):
        if not arg:
            print(t("cli.score.usage"))
            return
        parts = arg.split(maxsplit=1)
        try:
            score = int(parts[0])
        except ValueError:
            print(t("cli.score.not_integer"))
            return
        if score < 0 or score > 10:
            print(t("cli.score.out_of_range"))
            return
        comment = parts[1] if len(parts) > 1 else ""
        self.save_feedback("score", score, comment)

    def save_feedback(self, kind, score, comment):
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": _now(),
            "feedback_kind": kind,
            "score": score,
            "comment": comment,
            "session_id": self.session_id,
            "session_path": str(self.session_path),
            "model": self.model,
            "workspace": str(self.workspace),
            "turns": self.turns,
            "commands": len(self.commands),
            "failures": self.failures,
            "usage": self.total_usage,
            "last_answer": self.last_answer,
        }
        with self.feedback_path.open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.ratings += 1
        self._log("feedback", **event)
        print(t("cli.feedback.saved", score=score, path=self.feedback_path))

    def feedback_reminder(self, had_command):
        if self.turns == 0:
            return
        if not had_command and self.turns % 3 != 0:
            return
        print()
        print(_color(t("cli.feedback.reminder"), "dim"))
