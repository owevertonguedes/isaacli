"""Per-user launcher installation and explicit local-data removal."""
import grp
import json
import os
import pwd
import shutil
import subprocess
import sys
from pathlib import Path

import config
from cli_i18n import t
from cli_ollama import _runtime_ollama_dir, _same_process
from cli_sessions import FEEDBACK_DIR, SESSIONS_DIR

HERE = Path(__file__).resolve().parent


def install_launcher(bin_dir=None):
    """Install a per-user symlink without overwriting another command."""
    source = HERE.parent / "isaacli"
    target_dir = (
        Path(bin_dir) if bin_dir is not None else Path.home() / ".local" / "bin"
    )
    target = target_dir / "isaacli"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            if target.resolve(strict=False) == source.resolve():
                print(t("cli.install.already", path=target))
                return 0
            print(t("cli.install.conflict", path=target))
            return 1
        target.symlink_to(source)
    except OSError as error:
        print(t("cli.install.failed", path=target, error=error))
        return 1

    print(t("cli.install.success", path=target))
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(target_dir) not in path_entries:
        print(t("cli.install.path_hint", path=target_dir))
    return 0


def uninstall_launcher(
    purge=False, bin_dir=None, config_dir=None, data_dirs=None, runtime_dir=None,
):
    """Remove only this checkout's launcher and, when requested, its local data."""
    source = (HERE.parent / "isaacli").resolve()
    target_dir = (
        Path(bin_dir) if bin_dir is not None else Path.home() / ".local" / "bin"
    )
    target = target_dir / "isaacli"
    if target.is_symlink() or target.exists():
        if not target.is_symlink() or target.resolve(strict=False) != source:
            print(t("cli.uninstall.conflict", path=target))
            return 1

    purge_dirs = list(data_dirs) if data_dirs is not None else [
        SESSIONS_DIR, FEEDBACK_DIR, HERE / "curation",
    ]
    purge_dirs.extend([
        Path(config_dir) if config_dir is not None else config.config_path().parent,
        Path(runtime_dir) if runtime_dir is not None else _runtime_ollama_dir(),
    ])
    if purge:
        state_path = purge_dirs[-1] / "ollama.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        active = [
            item for item in state.get("clients", [])
            if isinstance(item, dict)
            and isinstance(item.get("pid"), (int, str))
            and _same_process(item.get("pid"), item.get("start"))
        ]
        if active:
            print(t("cli.uninstall.active", count=len(active)))
            return 1

    try:
        if target.is_symlink():
            target.unlink()
            print(t("cli.uninstall.removed", path=target))
        else:
            print(t("cli.uninstall.not_installed", path=target))
        if purge:
            for path in purge_dirs:
                if path.exists():
                    shutil.rmtree(path)
                    print(t("cli.uninstall.purged", path=path))
    except OSError as error:
        print(t("cli.uninstall.failed", error=error))
        return 1
    return 0


def official_ollama_plan(found=None, path_exists=None, user_exists=None,
                         group_exists=None):
    """Build the fixed-path teardown for the official Linux script layout."""
    found = shutil.which("ollama") if found is None else found
    path_exists = path_exists or (lambda path: path.exists())
    executable = Path(found).resolve() if found else None
    official_binary = Path("/usr/local/bin/ollama")
    official_library = Path("/usr/local/lib/ollama")
    service = Path("/etc/systemd/system/ollama.service")
    shared_data = Path("/usr/share/ollama")
    if executable != official_binary or not (
        path_exists(official_library) or path_exists(service)
    ):
        return None

    commands = [["sudo", "-v"]]
    if path_exists(service):
        commands.extend([
            ["sudo", "systemctl", "stop", "ollama"],
            ["sudo", "systemctl", "disable", "ollama"],
            ["sudo", "rm", "-f", str(service)],
            ["sudo", "systemctl", "daemon-reload"],
        ])
    if path_exists(official_library):
        commands.append(["sudo", "rm", "-rf", str(official_library)])
    if path_exists(official_binary):
        commands.append(["sudo", "rm", "-f", str(official_binary)])
    if path_exists(shared_data):
        commands.append(["sudo", "rm", "-rf", str(shared_data)])
    if user_exists is None:
        try:
            pwd.getpwnam("ollama")
            user_exists = True
        except KeyError:
            user_exists = False
    if user_exists:
        commands.append(["sudo", "userdel", "ollama"])
    if group_exists is None:
        try:
            grp.getgrnam("ollama")
            group_exists = True
        except KeyError:
            group_exists = False
    if group_exists:
        commands.append(["sudo", "groupdel", "ollama"])
    return commands


def uninstall_official_ollama(run_fn=subprocess.run, home_dir=None):
    """Remove Ollama only when it matches the official Linux script layout."""
    if not sys.platform.startswith("linux"):
        print(t("cli.uninstall.ollama.unsupported"))
        return 1
    found = shutil.which("ollama")
    commands = official_ollama_plan(found)
    if commands is None:
        print(t("cli.uninstall.ollama.unrecognized", path=found or "—"))
        return 1
    if shutil.which("sudo") is None:
        print(t("cli.uninstall.ollama.no_sudo"))
        return 1

    for command in commands:
        result = run_fn(command, check=False)
        allowed = {0, 6} if command[1] in {"userdel", "groupdel"} else {0}
        if result.returncode not in allowed:
            print(t("cli.uninstall.ollama.failed", command=" ".join(command)))
            return 1
    user_data = Path(home_dir) / ".ollama" if home_dir else Path.home() / ".ollama"
    try:
        if user_data.exists():
            shutil.rmtree(user_data)
    except OSError as error:
        print(t("cli.uninstall.failed", error=error))
        return 1
    print(t("cli.uninstall.ollama.removed"))
    return 0
