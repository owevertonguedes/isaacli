#!/usr/bin/env python3
"""Destructive install lifecycle checks for a disposable Fedora container."""
import contextlib
import hashlib
import io
import json
import os
import pwd
import grp
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


CLONE = Path("/opt/isaacli")
HOME = Path("/home/isaac")
BIN = HOME / ".local" / "bin"
CONFIG = HOME / ".config" / "isaacli"
RUNTIME = Path("/tmp/isaacli-runtime") / "isaacli"
SESSIONS = CLONE / "tool_harness" / "cli_sessions"
FEEDBACK = CLONE / "tool_harness" / "feedback"
KEY = "isolated-secret-009-never-log"
ENV = {
    **os.environ,
    "HOME": str(HOME),
    "PATH": f"{BIN}:/usr/local/bin:/usr/bin:/bin",
    "ISAACLI_RUNTIME_DIR": str(RUNTIME.parent),
}
FAILURES = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        FAILURES.append(description)


def run(args, *, env=None, cwd=None, input_text=None):
    return subprocess.run(
        [str(item) for item in args], check=False, capture_output=True, text=True,
        cwd=cwd or HOME, env=env or ENV, input=input_text,
    )


def sudo(*args):
    result = run(["sudo", *args])
    if result.returncode:
        raise RuntimeError(f"sudo {' '.join(args)} failed: {result.stderr}")


def remove_official_fixture():
    run(["sudo", "systemctl", "disable", "--now", "ollama"])
    for path in (
        "/etc/systemd/system/ollama.service.d",
        "/etc/systemd/system/ollama.service",
        "/usr/local/lib/ollama", "/usr/local/bin/ollama",
        "/usr/share/ollama",
    ):
        run(["sudo", "rm", "-rf", path])
    run(["sudo", "userdel", "ollama"])
    run(["sudo", "groupdel", "ollama"])
    run(["sudo", "systemctl", "daemon-reload"])
    shutil.rmtree(HOME / ".ollama", ignore_errors=True)


def seed_official_fixture():
    remove_official_fixture()
    sudo("useradd", "--system", "--home-dir", "/usr/share/ollama",
         "--shell", "/sbin/nologin", "ollama")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        stub = root / "ollama"
        stub.write_text(
            "#!/bin/sh\n"
            "if [ \"${1:-}\" = serve ]; then exec sleep infinity; fi\n"
            "printf '%s\\n' 'ollama test stub'\n",
            encoding="utf-8",
        )
        unit = root / "ollama.service"
        unit.write_text(
            "[Unit]\nDescription=Disposable Ollama fixture\n"
            "[Service]\nExecStart=/usr/local/bin/ollama serve\nUser=ollama\nGroup=ollama\n"
            "[Install]\nWantedBy=multi-user.target\n",
            encoding="utf-8",
        )
        sudo("install", "-m", "0755", str(stub), "/usr/local/bin/ollama")
        sudo("install", "-m", "0644", str(unit),
             "/etc/systemd/system/ollama.service")
    sudo("mkdir", "-p", "/usr/local/lib/ollama", "/usr/share/ollama/models")
    sudo("touch", "/usr/local/lib/ollama/library-bait",
         "/usr/share/ollama/models/model-bait")
    sudo("chown", "-R", "ollama:ollama", "/usr/share/ollama")
    (HOME / ".ollama" / "models").mkdir(parents=True, exist_ok=True)
    (HOME / ".ollama" / "models" / "user-model-bait").write_text("bait")
    sudo("systemctl", "daemon-reload")
    sudo("systemctl", "enable", "--now", "ollama")


class ApiHandler(BaseHTTPRequestHandler):
    authorizations = []

    def log_message(self, _format, *_args):
        pass

    def do_GET(self):
        self.authorizations.append(self.headers.get("Authorization"))
        payload = json.dumps({"data": [{"id": "fixture-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        self.authorizations.append(self.headers.get("Authorization"))
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = (
            'data: {"choices":[{"delta":{"content":"fixture reply"}}],'
            '"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n'
            'data: [DONE]\n\n'
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(body)


@contextlib.contextmanager
def api_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), ApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def seed_private_state():
    CONFIG.mkdir(parents=True, exist_ok=True)
    (CONFIG / "state").write_text("private")
    for directory in (SESSIONS, FEEDBACK, RUNTIME):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "state").write_text("private")


def install():
    return run([CLONE / "isaacli", "install"])


def failure_matrix(launcher):
    fake_bin = Path(tempfile.mkdtemp(prefix="failing-sudo-"))
    fake_sudo = fake_bin / "sudo"
    fake_sudo.write_text(
        "#!/bin/sh\n"
        "case \" $* \" in *\"${FAIL_MATCH}\"*) exit 42 ;; esac\n"
        "exec /usr/bin/sudo \"$@\"\n",
        encoding="utf-8",
    )
    fake_sudo.chmod(0o755)
    cases = (
        ("true", {"service", "library", "binary", "shared", "user", "group"}),
        ("systemctl stop ollama",
         {"service", "library", "binary", "shared", "user", "group"}),
        ("rm -rf /usr/local/lib/ollama",
         {"library", "binary", "shared", "user", "group"}),
        ("userdel ollama", {"user", "group"}),
        ("groupdel ollama", set()),
    )
    results = {}
    for match, expected in cases:
        seed_official_fixture()
        if not launcher.exists():
            install()
        seed_private_state()
        env = {**ENV, "FAIL_MATCH": match,
               "PATH": f"{fake_bin}:{ENV['PATH']}"}
        result = run(["isaacli", "uninstall", "--purge", "--ollama"],
                     env=env, input_text="uninstall ollama\n")
        present = set()
        paths = {
            "service": Path("/etc/systemd/system/ollama.service"),
            "library": Path("/usr/local/lib/ollama"),
            "binary": Path("/usr/local/bin/ollama"),
            "shared": Path("/usr/share/ollama"),
        }
        present.update(name for name, path in paths.items() if path.exists())
        try:
            pwd.getpwnam("ollama")
            present.add("user")
        except KeyError:
            pass
        try:
            grp.getgrnam("ollama")
            present.add("group")
        except KeyError:
            pass
        results[match] = sorted(present)
        check(result.returncode == 1 and match in result.stdout
              and launcher.exists() and CONFIG.exists()
              and (HOME / ".ollama").exists()
              and expected.issubset(present),
              f"failure at {match!r} reports the command and preserves documented bait")
    print("[evidence] partial-removal matrix: " + json.dumps(results, sort_keys=True))


def lifecycle():
    os.chdir(HOME)
    shutil.rmtree(CONFIG, ignore_errors=True)
    shutil.rmtree(BIN, ignore_errors=True)
    shutil.rmtree(RUNTIME.parent, ignore_errors=True)
    check(not CONFIG.exists() and not SESSIONS.exists() and not FEEDBACK.exists(),
          "fresh archive contains no configuration, sessions, or feedback")

    first, second = install(), install()
    launcher = BIN / "isaacli"
    check(first.returncode == second.returncode == 0 and launcher.is_symlink()
          and launcher.resolve() == CLONE / "isaacli",
          "install creates exactly one idempotent per-user symlink")
    version = run(["isaacli", "--version"], cwd="/tmp")
    check(version.returncode == 0 and "Isaac CLI v0.4.0-dev" in version.stdout,
          "the PATH launcher works outside the clone and preserves arguments")

    launcher.unlink()
    launcher.write_text("other checkout")
    conflict_install = install()
    conflict_uninstall = run([CLONE / "isaacli", "uninstall"])
    check(conflict_install.returncode == conflict_uninstall.returncode == 1
          and launcher.read_text() == "other checkout",
          "another checkout's command is neither overwritten nor removed")
    launcher.unlink()
    check(install().returncode == 0, "the owned launcher can be restored")

    fake_bin = Path(tempfile.mkdtemp(prefix="flatpak-bin-"))
    flatpak_log = fake_bin / "calls.jsonl"
    fake_spawn = fake_bin / "flatpak-spawn"
    fake_spawn.write_text(
        "#!/bin/sh\n"
        "python3 -c 'import json,os,sys; open(os.environ[\"FLATPAK_LOG\"],\"a\").write(json.dumps({\"cwd\":os.getcwd(),\"args\":sys.argv[1:]})+\"\\n\")' \"$@\"\n"
        "[ \"$1\" = --host ] && shift\n"
        "unset FLATPAK_ID\nexec \"$@\"\n",
        encoding="utf-8",
    )
    fake_spawn.chmod(0o755)
    flatpak_env = {**ENV, "FLATPAK_ID": "fixture.app",
                   "FLATPAK_LOG": str(flatpak_log),
                   "PATH": f"{fake_bin}:{ENV['PATH']}"}
    flatpak = run(["isaacli", "--version"], env=flatpak_env, cwd="/tmp")
    calls = [json.loads(line) for line in flatpak_log.read_text().splitlines()]
    check(flatpak.returncode == 0 and len(calls) == 1
          and calls[0]["cwd"] == "/tmp"
          and calls[0]["args"][-2:] == [str(CLONE / "isaacli"), "--version"],
          "the Flatpak bridge runs once and preserves cwd and arguments")

    sys.path.insert(0, str(CLONE / "tool_harness"))
    import config
    import setup_ollama
    selections = iter([1, 2])  # API, then medium reasoning.
    original_select = setup_ollama.terminal_ui.select
    setup_ollama.terminal_ui.select = lambda *_a, **_kw: next(selections)
    output = io.StringIO()
    try:
        with api_server() as base_url, contextlib.redirect_stdout(output):
            answers = iter(["Fixture API", base_url, "fixture-model", KEY])
            setup_code = setup_ollama._run_setup(
                input_fn=lambda _prompt="": next(answers), initial_language="en",
            )
            via_path = run(["isaacli", "hello"], cwd="/tmp")
            via_absolute = run([CLONE / "isaacli", "hello"], cwd="/tmp")
    finally:
        setup_ollama.terminal_ui.select = original_select
    config_data = json.loads((CONFIG / "config.json").read_text())
    secret_data = json.loads((CONFIG / "secrets.json").read_text())
    secret_mode = stat.S_IMODE((CONFIG / "secrets.json").stat().st_mode)
    public_files = [CONFIG / "config.json", *SESSIONS.glob("*.jsonl")]
    leaked = any(KEY in path.read_text(errors="replace") for path in public_files)
    combined_output = output.getvalue() + via_path.stdout + via_path.stderr \
        + via_absolute.stdout + via_absolute.stderr
    check(setup_code == 0 and config_data["default_profile"]
          and KEY in secret_data.values() and secret_mode == 0o600,
          "first API setup stores one profile and a 0600 secret in isolated HOME")
    check(via_path.returncode == via_absolute.returncode == 0
          and "fixture reply" in via_path.stdout and "fixture reply" in via_absolute.stdout,
          "PATH and absolute launchers reload the same profile and credential")
    check(KEY not in combined_output and not leaked,
          "the API key is absent from output, config, sessions, and logs")

    before_cancel = hashlib.sha256((CONFIG / "config.json").read_bytes()).digest()
    cancel = run([CLONE / "isaacli", "setup"], input_text="")
    check(cancel.returncode != 0
          and hashlib.sha256((CONFIG / "config.json").read_bytes()).digest()
          == before_cancel,
          "EOF during setup preserves the last complete profile")

    seed_official_fixture()
    seed_private_state()
    normal = run(["isaacli", "uninstall"])
    normal_again = run([CLONE / "isaacli", "uninstall"])
    check(normal.returncode == normal_again.returncode == 0 and not launcher.exists()
          and all(path.exists() for path in (CONFIG, SESSIONS, FEEDBACK, RUNTIME))
          and Path("/usr/local/bin/ollama").exists()
          and (HOME / ".ollama").exists() and CLONE.exists(),
          "plain uninstall is idempotent and removes only its launcher")

    check(install().returncode == 0, "launcher reinstalls before purge")
    runtime_state = RUNTIME / "ollama.json"
    identity = Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]
    runtime_state.write_text(json.dumps({"clients": [{
        "pid": os.getpid(), "start": identity,
    }]}))
    active = run(["isaacli", "uninstall", "--purge"], input_text="uninstall\n")
    check(active.returncode == 1 and launcher.exists() and CONFIG.exists()
          and Path("/usr/local/bin/ollama").exists(),
          "a live session blocks purge before Ollama is touched")
    runtime_state.write_text("{}")
    purge = run(["isaacli", "uninstall", "--purge"], input_text="uninstall\n")
    purge_again = run([CLONE / "isaacli", "uninstall", "--purge"],
                      input_text="uninstall\n")
    check(purge.returncode == purge_again.returncode == 0 and not launcher.exists()
          and not any(path.exists() for path in (CONFIG, SESSIONS, FEEDBACK, RUNTIME))
          and Path("/usr/local/bin/ollama").exists()
          and (HOME / ".ollama").exists() and CLONE.exists(),
          "purge is idempotent, removes Isaac data, and preserves Ollama and clone")

    check(install().returncode == 0, "launcher reinstalls before strong purge")
    seed_private_state()
    custom_env = {**ENV, "OLLAMA_MODELS": "/srv/custom-ollama-models"}
    custom = run(["isaacli", "uninstall", "--purge", "--ollama"],
                 env=custom_env, input_text="uninstall ollama\n")
    check(custom.returncode == 1 and launcher.exists() and CONFIG.exists()
          and Path("/usr/local/bin/ollama").exists()
          and "/srv/custom-ollama-models" in custom.stdout,
          "known custom model storage refuses strong purge before any deletion")

    failure_matrix(launcher)
    seed_official_fixture()
    seed_private_state()

    strong = run(["isaacli", "uninstall", "--purge", "--ollama"],
                 input_text="uninstall ollama\n")
    strong_again = run([CLONE / "isaacli", "uninstall", "--purge", "--ollama"],
                       input_text="uninstall ollama\n")
    known = (
        launcher, CONFIG, SESSIONS, FEEDBACK, RUNTIME,
        Path("/usr/local/bin/ollama"), Path("/usr/local/lib/ollama"),
        Path("/etc/systemd/system/ollama.service"), Path("/usr/share/ollama"),
        HOME / ".ollama",
    )
    try:
        pwd.getpwnam("ollama")
        user_absent = False
    except KeyError:
        user_absent = True
    try:
        grp.getgrnam("ollama")
        group_absent = False
    except KeyError:
        group_absent = True
    check(strong.returncode == strong_again.returncode == 0
          and not any(path.exists() for path in known)
          and user_absent and group_absent and CLONE.exists(),
          "strong purge is idempotent and leaves only the clone")

    import installation
    check(installation.official_ollama_plan(
              "/usr/bin/ollama", path_exists=lambda _path: True,
              package_owned=True,
          ) is None,
          "a package-owned /usr/bin installation is refused")

    move_home = Path(tempfile.mkdtemp(prefix="isaacli-move-home-"))
    movable = Path(tempfile.mkdtemp(prefix="isaacli-movable-")) / "clone"
    shutil.copytree(CLONE, movable)
    move_env = {**ENV, "HOME": str(move_home),
                "PATH": f"{move_home / '.local/bin'}:/usr/bin:/bin"}
    moved_install = run([movable / "isaacli", "install"], env=move_env)
    moved_link = move_home / ".local" / "bin" / "isaacli"
    movable.rename(movable.with_name("clone-moved"))
    try:
        broken = run([moved_link, "--version"], env=move_env)
        broken_code = broken.returncode
    except FileNotFoundError:
        broken_code = 127
    check(moved_install.returncode == 0 and moved_link.is_symlink()
          and not moved_link.exists() and broken_code != 0,
          "moving a clone leaves a broken link that currently needs manual cleanup")


if __name__ == "__main__":
    try:
        lifecycle()
    finally:
        remove_official_fixture()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for failure in FAILURES:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("INSTALL LIFECYCLE OK: isolated install, setup, and three-level uninstall")
