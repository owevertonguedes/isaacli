"""isaacli's local configuration, outside the workspace and free of secrets."""
import json
import os
import tempfile
import urllib.parse
from pathlib import Path


CONFIG_VERSION = 1

# A server the user runs on their own machine (llama-server, Ollama's own
# compatible endpoint, anything else) normally has no authentication to
# demand. Requiring a key there would block the local-first path this project
# exists for. It stays required for anything reachable over the network.
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}


def is_local_endpoint(base_url) -> bool:
    host = urllib.parse.urlsplit(str(base_url or "")).hostname
    return (host or "").lower() in {h.strip("[]") for h in LOCAL_HOSTS}


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "isaacli" / "config.json"


def secrets_path() -> Path:
    return config_path().with_name("secrets.json")


def save_secret(name, value, path=None):
    target = Path(path) if path else secrets_path()
    data = {}
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    data[name] = value
    save(data, target)


def load_secret(name, path=None):
    target = Path(path) if path else secrets_path()
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8")).get(name)
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def empty_config():
    return {
        "version": CONFIG_VERSION,
        "language": None,
        "default_profile": None,
        "profiles": {},
        "permissions": {"global": [], "workspaces": {}},
    }


def load(path=None):
    target = Path(path) if path else config_path()
    if not target.exists():
        return empty_config()
    try:
        data = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"invalid configuration in {target}: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("profiles", {}), dict):
        raise ValueError(f"invalid configuration in {target}: unexpected format")
    data.setdefault("version", CONFIG_VERSION)
    data.setdefault("language", None)
    data.setdefault("default_profile", None)
    data.setdefault("profiles", {})
    permissions = data.setdefault("permissions", {})
    if not isinstance(permissions, dict):
        permissions = data["permissions"] = {}
    permissions.setdefault("global", [])
    permissions.setdefault("workspaces", {})
    return data


def permission_rules(data, workspace):
    permissions = data.get("permissions") or {}
    global_rules = set(permissions.get("global") or [])
    local_rules = set(
        (permissions.get("workspaces") or {}).get(str(Path(workspace).resolve()), []))
    return global_rules | local_rules


def add_permission(data, rule, workspace=None):
    permissions = data.setdefault("permissions", {"global": [], "workspaces": {}})
    if workspace is None:
        target = permissions.setdefault("global", [])
    else:
        key = str(Path(workspace).resolve())
        target = permissions.setdefault("workspaces", {}).setdefault(key, [])
    if rule not in target:
        target.append(rule)


def save(data, path=None):
    target = Path(path) if path else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix="config-", suffix=".tmp", dir=target.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as file:
            file.write(payload)
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


def profile(data, name=None):
    chosen = name or data.get("default_profile")
    item = (data.get("profiles") or {}).get(chosen)
    return chosen, item


def default_model(fallback=None, path=None):
    try:
        data = load(path)
    except ValueError:
        return fallback
    _name, item = profile(data)
    return item.get("model", fallback) if item else fallback


def model_thinking(data, model):
    for item in (data.get("profiles") or {}).values():
        if item.get("model") == model:
            return item.get("thinking")
    return None


def profile_for_model(data, model):
    for name, item in (data.get("profiles") or {}).items():
        if item.get("model") == model:
            return name, item
    return None, None
