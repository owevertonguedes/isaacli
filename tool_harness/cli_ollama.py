"""Ollama server lifecycle: autostart, autostop, and sharing one server across
several isaacli sessions.

Invariant that must not break: an Ollama started by isaacli is shared between
sessions, and the last session only stops the server isaacli itself started.
A server that already existed belongs to the user and is never touched.
tests/check_cli.py covers this.
"""
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from cli_i18n import t
from cli_presentation import _color


def _runtime_ollama_dir():
    base = os.environ.get("ISAACLI_RUNTIME_DIR") or os.environ.get("XDG_RUNTIME_DIR")
    if base:
        return Path(base) / "isaacli"
    return Path("/tmp") / f"isaacli-{os.getuid()}"


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
def _shared_ollama_state():
    """Serialise autostart/autostop across several Isaac sessions."""
    import fcntl

    folder = _runtime_ollama_dir()
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = folder / "ollama.lock"
    state_path = folder / "ollama.json"
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
        return None


class OllamaMixin:
    def ensure_ollama(self, warn=False):
        if self.provider.get("provider") != "ollama":
            return ((self.provider.get("provider_name") or "API")
                    if self.provider.get("api_key") and self.provider.get("base_url") else None)
        with _shared_ollama_state() as state:
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

    def close(self):
        if not self._ollama_registered:
            return
        with _shared_ollama_state() as state:
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
