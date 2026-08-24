"""Per-user launcher installation and explicit local-data removal."""
import grp
import json
import os
import pwd
import re
import shutil
import subprocess
import sys
from pathlib import Path

import config
import debug
import local_models
import units
from cli_i18n import t
from cli_ollama import _runtime_ollama_dir, _same_process
from cli_sessions import FEEDBACK_DIR, SESSIONS_DIR

HERE = Path(__file__).resolve().parent
OFFICIAL_SERVICE = Path("/etc/systemd/system/ollama.service")
OFFICIAL_SHARED_DATA = Path("/usr/share/ollama")
OFFICIAL_LAYOUTS = {
    Path("/usr/local/bin/ollama"): Path("/usr/local/lib/ollama"),
    Path("/usr/bin/ollama"): Path("/usr/lib/ollama"),
}


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
    check_only=False, cache_dir=None,
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
        # Links into Ollama's blob store are ours and cost nothing to recreate,
        # so they go. Removing a link never touches what it points at, which is
        # the whole reason reuse was built out of links.
        local_models.linked_dir(),
    ]
    purge_dirs.extend([
        Path(config_dir) if config_dir is not None else config.config_path().parent,
        # Kaggle asset preparation stages downloads here, so this program creates
        # it and this program has to be able to take it away again. It stays
        # ahead of the runtime directory because the Ollama state file below is
        # read from the last entry.
        Path(cache_dir) if cache_dir is not None else config.cache_path(),
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

    if check_only:
        return 0

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
            _report_kept_weights()
    except OSError as error:
        print(t("cli.uninstall.failed", error=error))
        return 1
    return 0


def _report_kept_weights(models_dir=None):
    """Name the model weights a purge deliberately did not delete.

    These are gigabytes somebody waited on, and a download nobody wanted to
    repeat is exactly why reuse exists at all. Deleting them because a user
    asked to forget their sessions would be the wrong trade in the expensive
    direction. Leaving them without saying so would be worse, so the path and
    the size go on screen and the deletion stays the user's to make.
    """
    folder = Path(models_dir) if models_dir else local_models.downloaded_dir()
    # Every failure here is the failure of a courtesy message, and the removal
    # it reports on has already happened. Letting an OSError out would make a
    # completed uninstall report itself as failed, so the whole thing is
    # guarded rather than only the listing.
    try:
        weights = [path for path in folder.glob("*.gguf") if path.is_file()]
        if not weights:
            return None
        total = sum(path.stat().st_size for path in weights)
    except OSError:
        debug.swallowed("installation._report_kept_weights")
        return None
    print(t("cli.uninstall.weights_kept", path=folder, count=len(weights),
            size=units.gib(total)))
    return folder


def uninstall_managed_llamacpp(home_dir=None, record_path=None):
    """Remove the llama.cpp isaacli installed, saying in the user's language
    exactly what happened or exactly what was refused.

    The decision lives in llama_cpp.uninstall, which answers with a key rather
    than a sentence so that the module doing the refusing never has to know
    which language the session is in.
    """
    import llama_cpp

    code, key, values = llama_cpp.uninstall(
        record_path=record_path, home_dir=home_dir)
    print(t(key, **values))
    return code


def _package_owns(path):
    """Return whether RPM or dpkg reports ownership of an executable."""
    checks = (
        (["rpm", "-qf", str(path)], shutil.which("rpm")),
        (["dpkg-query", "-S", str(path)], shutil.which("dpkg-query")),
    )
    for command, executable in checks:
        if not executable:
            continue
        result = subprocess.run(
            command, check=False, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return True
    return False


def _service_model_paths(service=OFFICIAL_SERVICE):
    """Read absolute OLLAMA_MODELS paths declared by the unit or its drop-ins."""
    files = [service]
    drop_in = service.with_name(service.name + ".d")
    try:
        files.extend(sorted(drop_in.glob("*.conf")))
    except OSError:
        pass
    paths = []
    pattern = re.compile(r"OLLAMA_MODELS=([^\s\"']+|\"[^\"]+\"|'[^']+')")
    for path in files:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for match in pattern.finditer(content):
            value = match.group(1).strip("\"'")
            candidate = Path(value).expanduser()
            if candidate not in paths:
                paths.append(candidate)
    return paths


def custom_ollama_model_paths(home_dir=None, environ=None, service=OFFICIAL_SERVICE):
    """Return known model paths not covered by the fixed safe teardown."""
    environ = os.environ if environ is None else environ
    candidates = _service_model_paths(service)
    value = environ.get("OLLAMA_MODELS")
    if value:
        candidate = Path(value).expanduser()
        if candidate not in candidates:
            candidates.append(candidate)
    home = Path(home_dir) if home_dir else Path.home()
    covered = (OFFICIAL_SHARED_DATA.resolve(), (home / ".ollama").resolve())
    custom = []
    for path in candidates:
        if not path.is_absolute():
            custom.append(path)
            continue
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            custom.append(path)
            continue
        if not any(resolved == root or root in resolved.parents for root in covered):
            custom.append(path)
    return custom


def official_ollama_plan(found=None, path_exists=None, user_exists=None,
                         group_exists=None, package_owned=None):
    """Build the fixed-path teardown for the official Linux script layout."""
    found = shutil.which("ollama") if found is None else found
    path_exists = path_exists or (lambda path: path.exists())
    executable = Path(found).resolve() if found else None
    official_library = OFFICIAL_LAYOUTS.get(executable)
    if official_library is None:
        return None
    package_owned = _package_owns(executable) if package_owned is None else package_owned
    if package_owned:
        return None
    # /usr is also the package-manager prefix. Requiring both official-script
    # artifacts there avoids guessing that an RPM/DEB-like partial layout is ours.
    if executable == Path("/usr/bin/ollama"):
        recognized = path_exists(official_library) and path_exists(OFFICIAL_SERVICE)
    else:
        recognized = path_exists(official_library) or path_exists(OFFICIAL_SERVICE)
    if not recognized:
        return None

    # Unlike `sudo -v`, this also works for a valid NOPASSWD policy while still
    # prompting up front when the policy requires authentication.
    commands = [["sudo", "true"]]
    if path_exists(OFFICIAL_SERVICE):
        commands.extend([
            ["sudo", "systemctl", "stop", "ollama"],
            ["sudo", "systemctl", "disable", "ollama"],
            ["sudo", "rm", "-f", str(OFFICIAL_SERVICE)],
            ["sudo", "systemctl", "daemon-reload"],
        ])
    if path_exists(official_library):
        commands.append(["sudo", "rm", "-rf", str(official_library)])
    if path_exists(executable):
        commands.append(["sudo", "rm", "-f", str(executable)])
    if path_exists(OFFICIAL_SHARED_DATA):
        commands.append(["sudo", "rm", "-rf", str(OFFICIAL_SHARED_DATA)])
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
    custom_paths = custom_ollama_model_paths(home_dir=home_dir)
    if custom_paths:
        print(t("cli.uninstall.ollama.custom_models",
                paths=", ".join(str(path) for path in custom_paths)))
        return 1
    user_data = Path(home_dir) / ".ollama" if home_dir else Path.home() / ".ollama"
    found = shutil.which("ollama")
    commands = official_ollama_plan(found)
    if commands is None:
        known_paths = [*OFFICIAL_LAYOUTS, *OFFICIAL_LAYOUTS.values(),
                       OFFICIAL_SERVICE, OFFICIAL_SHARED_DATA, user_data]
        accounts_remain = any((
            _account_exists(pwd.getpwnam, "ollama"),
            _account_exists(grp.getgrnam, "ollama"),
        ))
        if (found is None and not accounts_remain
                and not any(path.exists() for path in known_paths)):
            print(t("cli.uninstall.ollama.not_installed"))
            return 0
        print(t("cli.uninstall.ollama.unrecognized", path=found or "?"))
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
    try:
        if user_data.exists():
            shutil.rmtree(user_data)
    except OSError as error:
        print(t("cli.uninstall.failed", error=error))
        return 1
    print(t("cli.uninstall.ollama.removed"))
    return 0


def _account_exists(lookup, name):
    try:
        lookup(name)
        return True
    except KeyError:
        return False
