"""Local model server lifecycle: autostart, autostop, and sharing one server
across several isaacli sessions.

Covers Ollama, and any openai_compatible profile that carries an "autostart"
command (llama-server, or another server speaking the same API). Both use the
same locking primitive under their own key, so two servers never share state.

Invariant that must not break: a server started by isaacli is shared between
sessions, and the last session only stops the server isaacli itself started.
A server that already existed belongs to the user and is never touched.
tests/check_cli.py covers this for both paths.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import config
import debug
from cli_i18n import t
from cli_presentation import _color


# Seconds to wait for an autostarted local server to answer. Generous on
# purpose: the cost of waiting is a slow first turn, the cost of giving up too
# early is a turn that fails against a server that was about to be ready.
AUTOSTART_TIMEOUT = 180.0


def _runtime_ollama_dir():
    base = os.environ.get("ISAACLI_RUNTIME_DIR") or os.environ.get("XDG_RUNTIME_DIR")
    if base:
        return Path(base) / "isaacli"
    return Path("/tmp") / f"isaacli-{os.getuid()}"


def _autostart_key(provider):
    """Filesystem-safe key so two autostart profiles never share a lock or a
    state file with each other, or with Ollama's own "ollama" key."""
    base = provider.get("provider_name") or provider.get("base_url") or "local"
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-") or "local"
    return f"autostart-{slug}"


def _pid_identity(pid):
    """Stable identity so we never signal a PID that has been recycled."""
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().split()[21]
    except (OSError, ValueError, IndexError):
        return None


def _same_process(pid, identity):
    current = _pid_identity(pid)
    return bool(current and identity and current == str(identity))


@contextmanager
def _shared_local_state(key="ollama"):
    """Serialise autostart/autostop across several Isaac sessions, keyed per
    server so Ollama and an autostart profile never share state."""
    import fcntl

    folder = _runtime_ollama_dir()
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = folder / f"{key}.lock"
    state_path = folder / f"{key}.json"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            try:
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
            except (OSError, json.JSONDecodeError):
                state = {}
            yield state
            tmp = state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            os.replace(tmp, state_path)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _shared_ollama_state():
    """The historical entry point cli.py imports by name; equivalent to
    _shared_local_state("ollama")."""
    return _shared_local_state("ollama")


def _install_signals():
    def leave(_signum, _frame):
        raise SystemExit(130)

    # SIGINT has to become KeyboardInterrupt so the REPL can restore the screen
    # and print the session summary. HUP/TERM keep terminating immediately.
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (AttributeError, ValueError):
        pass
    for sig in (signal.SIGHUP, signal.SIGTERM):
        try:
            signal.signal(sig, leave)
        except (AttributeError, ValueError):
            pass


def _close_without_interruption(cli):
    """Finish the cleanup even when the user hits Ctrl+C again on the way out."""
    previous = {}
    for sig in (signal.SIGINT, signal.SIGHUP, signal.SIGTERM):
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, signal.SIG_IGN)
        except (AttributeError, ValueError):
            pass
    try:
        while True:
            try:
                cli.close()
                return
            except KeyboardInterrupt:
                # A SIGINT may already have been delivered at the instant the
                # finally block started. From here on new ones are ignored.
                continue
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (AttributeError, ValueError):
                pass


def _ollama_ok(timeout=2):
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=timeout) as r:
            return json.load(r).get("version") or "ok"
    except Exception:
        debug.swallowed("cli_ollama._ollama_ok")
        return None


def _probe_health(url, timeout=2):
    """Health probe for a non-Ollama server: any answer that is not a
    connection error counts as up. Unlike _ollama_ok it assumes nothing about
    the body, because llama-server and friends do not share Ollama's
    /api/version shape."""
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return "ok"
    except urllib.error.HTTPError as e:
        # Answering at all proves it is reachable, so a 404 on this particular
        # route still counts as up. The gateway codes are the exception: 503 is
        # exactly how llama-server says "still loading the model", and treating
        # that as ready hands the user's first request to a server that then
        # refuses it.
        if e.code in (502, 503, 504):
            debug.swallowed("cli_ollama._probe_health not ready")
            return None
        return "ok"
    except Exception:
        debug.swallowed("cli_ollama._probe_health")
        return None


class OllamaMixin:
    def ensure_ollama(self, warn=False):
        if self.provider.get("provider") != "ollama":
            autostart = self.provider.get("autostart")
            if autostart and autostart.get("cmd") and autostart.get("health_url"):
                return self._ensure_autostart_provider(autostart, warn=warn)
            base_url = self.provider.get("base_url")
            if not base_url:
                return None
            # A server on the user's own machine has no key to demand, so the
            # absence of one is not the absence of a provider.
            if not (self.provider.get("api_key") or config.is_local_endpoint(base_url)):
                return None
            return self.provider.get("provider_name") or "API"
        with _shared_local_state("ollama") as state:
            version = _ollama_ok()
            server_valid = (
                state.get("managed")
                and _same_process(state.get("server_pid"), state.get("server_start"))
            )
            clients = [
                item for item in state.get("clients", [])
                if _same_process(item.get("pid"), item.get("start"))
            ]
            if not server_valid:
                state.clear()
                clients = []
            if version:
                if server_valid:
                    current = {"pid": self._runtime_pid, "start": self._runtime_start}
                    clients = [c for c in clients if c.get("pid") != self._runtime_pid]
                    clients.append(current)
                    state["clients"] = clients
                    self._ollama_registered = True
                # Without valid state, the server belongs to the user/system.
                return version
            if not self.autostart_ollama:
                return None
            exe = shutil.which("ollama")
            if not exe:
                if warn:
                    print(_color(t("cli.ollama.not_found"), "bad"))
                return None

            if warn:
                print(_color(t("cli.ollama.starting"), "warn"))
            try:
                self.ollama_proc = subprocess.Popen(
                    [exe, "serve"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception as e:
                self._log("error", error=f"ollama_autostart: {e}")
                if warn:
                    print(_color(t("cli.ollama.start_failed", error=e), "bad"))
                return None

            self._log("meta", event="ollama_autostart", pid=self.ollama_proc.pid)
            for _ in range(40):
                time.sleep(0.25)
                version = _ollama_ok(timeout=1)
                if version:
                    state.update({
                        "managed": True,
                        "server_pid": self.ollama_proc.pid,
                        "server_start": _pid_identity(self.ollama_proc.pid),
                        "clients": [{"pid": self._runtime_pid,
                                     "start": self._runtime_start}],
                    })
                    self._ollama_registered = True
                    return version
                if self.ollama_proc.poll() is not None:
                    self._log("error", error=(
                        f"ollama serve exited with code {self.ollama_proc.returncode}"
                    ))
                    return None
            version = _ollama_ok(timeout=1)
            if version:
                state.update({
                    "managed": True,
                    "server_pid": self.ollama_proc.pid,
                    "server_start": _pid_identity(self.ollama_proc.pid),
                    "clients": [{"pid": self._runtime_pid,
                                 "start": self._runtime_start}],
                })
                self._ollama_registered = True
                return version
            if self.ollama_proc.poll() is None:
                self.ollama_proc.terminate()
                self.ollama_proc.wait(timeout=3)
            return None

    def _ensure_autostart_provider(self, autostart, warn=False):
        """The generic twin of the Ollama branch above, for an
        openai_compatible profile carrying an "autostart" command. Same
        shared-state and locking, under its own key.

        Deliberately a parallel path rather than an attempt to merge the two:
        the Ollama branch is the one covered by tests and proven in use, and
        rewriting it to serve both was the larger risk."""
        if not hasattr(self, "autostart_proc"):
            self.autostart_proc = None
            self._autostart_registered = False
        key = _autostart_key(self.provider)
        health_url = autostart["health_url"]
        name = self.provider.get("provider_name") or "the local server"
        with _shared_local_state(key) as state:
            version = _probe_health(health_url)
            server_valid = (
                state.get("managed")
                and _same_process(state.get("server_pid"), state.get("server_start"))
            )
            clients = [
                item for item in state.get("clients", [])
                if _same_process(item.get("pid"), item.get("start"))
            ]
            if not server_valid:
                state.clear()
                clients = []
            if version:
                if server_valid:
                    current = {"pid": self._runtime_pid, "start": self._runtime_start}
                    clients = [c for c in clients if c.get("pid") != self._runtime_pid]
                    clients.append(current)
                    state["clients"] = clients
                    self._autostart_registered = True
                # Without valid state the server belongs to the user, exactly
                # as with a pre-existing Ollama, and is never touched.
                return version
            if not self.autostart_ollama:
                return None
            if warn:
                print(_color(t("cli.local_server.starting", name=name), "warn"))
            try:
                self.autostart_proc = subprocess.Popen(
                    autostart["cmd"], stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception as e:
                self._log("error", error=f"autostart: {e}")
                if warn:
                    print(_color(t("cli.local_server.start_failed",
                                   name=name, error=e), "bad"))
                return None

            self._log("meta", event="autostart", key=key,
                      pid=self.autostart_proc.pid)
            # Ollama's 10s budget does not transfer: its daemon answers at once
            # and loads the model on demand, while llama-server and friends read
            # the whole model file before they answer anything. Measured here: a
            # 2 GB model takes well past 10s. The profile can raise it further
            # for a large model on slow storage.
            budget = float(autostart.get("timeout") or AUTOSTART_TIMEOUT)
            for _ in range(int(budget / 0.25)):
                time.sleep(0.25)
                version = _probe_health(health_url, timeout=1)
                if version:
                    state.update({
                        "managed": True,
                        "server_pid": self.autostart_proc.pid,
                        "server_start": _pid_identity(self.autostart_proc.pid),
                        "clients": [{"pid": self._runtime_pid,
                                     "start": self._runtime_start}],
                    })
                    self._autostart_registered = True
                    return version
                if self.autostart_proc.poll() is not None:
                    reason = f"exited with code {self.autostart_proc.returncode}"
                    self._log("error", error=f"autostart {key} {reason}")
                    if warn:
                        print(_color(t("cli.local_server.start_failed",
                                       name=name, error=reason), "bad"))
                    return None
            if self.autostart_proc.poll() is None:
                self.autostart_proc.terminate()
                self.autostart_proc.wait(timeout=3)
            return None

    def _close_autostart_provider(self):
        if not getattr(self, "_autostart_registered", False):
            return
        with _shared_local_state(_autostart_key(self.provider)) as state:
            clients = [
                item for item in state.get("clients", [])
                if item.get("pid") != self._runtime_pid
                and _same_process(item.get("pid"), item.get("start"))
            ]
            state["clients"] = clients
            server_pid = state.get("server_pid")
            server_valid = (
                state.get("managed")
                and _same_process(server_pid, state.get("server_start"))
            )
            if clients or not server_valid:
                if not server_valid:
                    state.clear()
                self._autostart_registered = False
                return
            try:
                os.kill(int(server_pid), signal.SIGTERM)
                deadline = time.monotonic() + 3
                while _same_process(server_pid, state.get("server_start")):
                    if time.monotonic() >= deadline:
                        os.kill(int(server_pid), signal.SIGKILL)
                        break
                    time.sleep(0.05)
            except (ProcessLookupError, PermissionError, ValueError, TypeError):
                pass
            state.clear()
            self._autostart_registered = False
            self._log("meta", event="autostart_stop", pid=server_pid)

    def close(self):
        self._close_autostart_provider()
        if not self._ollama_registered:
            return
        with _shared_local_state("ollama") as state:
            clients = [
                item for item in state.get("clients", [])
                if item.get("pid") != self._runtime_pid
                and _same_process(item.get("pid"), item.get("start"))
            ]
            state["clients"] = clients
            server_pid = state.get("server_pid")
            server_valid = (
                state.get("managed")
                and _same_process(server_pid, state.get("server_start"))
            )
            if clients or not server_valid:
                if not server_valid:
                    state.clear()
                self._ollama_registered = False
                return

            # The lock stays held until the process exits: a new session must not
            # see the server and register itself during the shutdown window.
            try:
                os.kill(int(server_pid), signal.SIGTERM)
                deadline = time.monotonic() + 3
                while _same_process(server_pid, state.get("server_start")):
                    if time.monotonic() >= deadline:
                        os.kill(int(server_pid), signal.SIGKILL)
                        break
                    time.sleep(0.05)
            except ProcessLookupError:
                pass
            state.clear()
            self._ollama_registered = False
            self._log("meta", event="ollama_autostop", pid=server_pid)
