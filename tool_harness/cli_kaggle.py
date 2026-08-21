"""Kaggle CLI installation and explicit remote-kernel orchestration."""
import csv
import fcntl
import hashlib
import io
import json
import os
import re
import select
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import config
import debug
import hardware
import terminal_ui
from cli_i18n import t
from installation import _package_owns


HERE = Path(__file__).resolve().parent
TEMPLATE_DIR = HERE.parent / "contrib" / "kaggle"
MODEL_CATALOG_PATH = HERE / "model_catalog.json"
TERMINAL_STATES = {"COMPLETE", "ERROR", "CANCELLED"}
URL_PATTERN = re.compile(r"TUNNEL_URL=(https://[-a-z0-9]+\.trycloudflare\.com)")
MODEL_CONTEXT = 16384
# There is no `kaggle kernels stop`, only `delete`, so a kernel nobody watches
# runs until Kaggle's own global maximum and spends quota that does not come
# back. Every push carries its own ceiling instead of relying on that maximum.
SESSION_TIMEOUT_SECONDS = 4 * 60 * 60
ACCELERATORS = {
    "NvidiaTeslaP100": {
        "label": "P100 16 GB", "vram_mb": 16384,
        "overhead_mb": hardware.DEFAULT_OVERHEAD_MB, "cuda_arch": "60",
        "gpu_count": 1,
    },
    "NvidiaTeslaT4": {
        "label": "T4 x2, 2 x 16 GB", "vram_mb": 32768,
        "overhead_mb": hardware.DEFAULT_OVERHEAD_MB * 2, "cuda_arch": "75",
        "gpu_count": 2,
    },
}
ACCELERATOR_PREFERENCE = ("NvidiaTeslaP100", "NvidiaTeslaT4")
BINARY_DATASET_SLUGS = {
    "75": "isaacli-llama-cuda-sm75-b10502",
}
MODEL_DATASET_SLUGS = {
    "qwen38-27b": "isaacli-qwen38-27b-ud-q4-k-m",
}
PREPARATION_TIMEOUT_SECONDS = 4 * 60 * 60


def _home(home_dir=None):
    return Path(home_dir) if home_dir is not None else Path.home()


def kaggle_install_record(path=None):
    return Path(path) if path else config.config_path().with_name("kaggle-install.json")


def _write_record(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    config.save(data, path)


def _secret_path(config_file=None):
    return Path(config_file).with_name("secrets.json") if config_file else None


def register_account(username, credential, config_file=None):
    """Store one Kaggle credential without placing it in the public config."""
    username = str(username).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise RuntimeError(t("cli.kaggle.accounts.username_invalid"))
    if not isinstance(credential, dict) or not (
            credential.get("token") or credential.get("key")):
        raise RuntimeError(t("cli.kaggle.accounts.credential_invalid"))
    secret_name = f"kaggle:{username}"
    config.save_secret(secret_name, json.dumps(credential), _secret_path(config_file))
    data = config.load(config_file)
    state = data.setdefault("kaggle", {})
    state.setdefault("accounts", {})[username] = {"credential": secret_name}
    state["selected_account"] = username
    config.save(data, config_file)
    return username


def _accounts_root(config_file=None):
    base = (Path(config_file).parent if config_file else config.config_path().parent)
    return base / "kaggle-accounts"


def _account_dir(username, config_file=None):
    digest = hashlib.sha256(username.encode("utf-8")).hexdigest()[:16]
    return _accounts_root(config_file) / digest


def _isolated_environment(account_dir):
    """A Kaggle environment that can only reach the credential in this folder.

    `KAGGLE_CONFIG_DIR` is not enough, and believing it was is what made account
    selection look real without being real. Read from the CLI 2.2.4 source and
    then confirmed by running it: `authenticate` tries the access token first,
    the legacy API key second and the OAuth credentials third, and both the
    token and the OAuth credentials are read from `~/.kaggle/access_token` and
    `~/.kaggle/credentials.json` through `expanduser`, which `KAGGLE_CONFIG_DIR`
    does not touch. Measured on this machine: an account folder holding a
    deliberately invalid credential still answered with the ambient account's
    real quota. Selecting the second account would have spent the first
    account's hours under the second account's name.

    `expanduser` does follow `HOME`, so the account folder becomes the home the
    CLI sees, and with it the credentials it can reach. `PYTHONUSERBASE` is
    pinned to the real home because a CLI installed with `pip --user` resolves
    its own package through `HOME`, and moving that without pinning stops the
    program from importing itself.
    """
    account_dir = Path(account_dir)
    (account_dir / ".kaggle").mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        environment.pop(name, None)
    environment.setdefault(
        "PYTHONUSERBASE", str(Path(os.environ.get("HOME", Path.home())) / ".local"))
    environment["HOME"] = str(account_dir)
    environment["KAGGLE_CONFIG_DIR"] = str(account_dir / ".kaggle")
    return environment


def _server_owner(executable, run_fn=subprocess.run, env=None):
    """The account Kaggle itself attributes the assets to, or None if it has none.

    `config view` cannot answer this. It prints the local configuration back,
    so a folder holding a kaggle.json that names one account and a key
    belonging to another reports the name in the file. A listing of what the
    authenticated user owns comes from the server, and its refs are prefixed
    with the real owner.
    """
    for noun in ("datasets", "kernels"):
        result = _run_capture(
            [str(executable), noun, "list", "--mine", "--csv",
             "--page-size", "1"], run_fn, env)
        if result.returncode != 0:
            continue
        for row in csv.DictReader(io.StringIO(result.stdout)):
            ref = row.get("ref") or row.get("Ref") or ""
            if "/" in ref:
                return ref.split("/", 1)[0]
    return None


def _verify_account(executable, username, environment, run_fn=subprocess.run):
    """Refuse to use an environment that does not answer as this account.

    Isolation is an argument about environment variables, and an argument is not
    evidence. Two things are checked, in this order. First the environment has
    to authenticate at all, which is what catches a folder whose credential
    expired or was revoked. Then, when the account owns anything at all, the
    owner Kaggle reports has to be this account, which is what catches a
    credential filed under the wrong name. An account that owns nothing yet
    cannot be confirmed that way, and that says so rather than passing quietly.
    """
    try:
        _quota(executable, run_fn, environment)
    except RuntimeError as error:
        raise RuntimeError(
            t("cli.kaggle.accounts.unverified", username=username, error=error))
    owner = _server_owner(executable, run_fn, environment)
    if owner is None:
        print(t("cli.kaggle.accounts.owner_unknown", username=username))
        return None
    if owner != username:
        raise RuntimeError(t("cli.kaggle.accounts.mismatch",
                             username=username, answered=owner))
    return owner


def _account_environment(username, config_file=None):
    """Materialize one selected credential for the Kaggle CLI only."""
    data = config.load(config_file)
    account = ((data.get("kaggle") or {}).get("accounts") or {}).get(username)
    if not account:
        raise RuntimeError(t("cli.kaggle.accounts.missing", username=username))
    account_dir = _account_dir(username, config_file)
    if account.get("browser_login"):
        # The Kaggle CLI wrote this folder itself during `auth login`, and it
        # owns whatever shape the credential has. Copying it into secrets.json
        # would mean this code deciding what a Kaggle credential looks like, and
        # then breaking the day that shape changes.
        if not account_dir.is_dir():
            raise RuntimeError(t("cli.kaggle.accounts.session_missing",
                                 username=username))
        return _isolated_environment(account_dir)
    raw = config.load_secret(account.get("credential"), _secret_path(config_file))
    try:
        credential = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(t("cli.kaggle.accounts.credential_invalid")) from error
    environment = _isolated_environment(account_dir)
    # These are the exact paths the CLI reads through `expanduser`, which now
    # resolves inside this account's folder. Nothing else has to be arranged,
    # and in particular no empty file is planted to neutralise a cached token:
    # an empty token file reads as no token at all and hands the decision
    # straight back to the ambient account.
    kaggle_dir = account_dir / ".kaggle"
    # The same value is offered to both mechanisms, and the CLI decides which
    # one it is. Read from the 2.2.4 source: `authenticate` tries the access
    # token first, and `_authenticate_with_access_token` introspects it against
    # Kaggle and returns False for one that does not check out, which falls
    # through to the legacy key. So a credential issued as an access token works
    # even when it arrives inside a kaggle.json, where only the legacy field
    # exists, and a real legacy key is not disturbed by being offered first.
    # This is not a guess about the credential's shape: guessing is what broke
    # it, because what Kaggle hands out at kaggle.com/settings today is an
    # access token and the legacy field cannot carry it.
    secret = str(credential.get("token") or credential.get("key")).strip()
    token_path = kaggle_dir / "access_token"
    token_path.write_text(secret + "\n", encoding="utf-8")
    token_path.chmod(0o600)
    if credential.get("key"):
        config.save({"username": username, "key": credential["key"]},
                    kaggle_dir / "kaggle.json")
    else:
        config.save({"username": username}, kaggle_dir / "kaggle.json")
    return environment


def _authenticated_username(executable, run_fn=subprocess.run, env=None):
    """The account the CLI is actually authenticated as, asked rather than typed."""
    result = _run_capture([str(executable), "config", "view"], run_fn, env)
    match = re.search(
        r"(?:username:\s*|- username:\s*)([^\s]+)", result.stdout or "", re.I)
    if result.returncode != 0 or not match:
        raise RuntimeError(t("cli.kaggle.username.failed"))
    return match.group(1)


def login_account(executable, config_file=None, run_fn=subprocess.run,
                  launch_browser=True):
    """Log in through the Kaggle CLI itself and register whoever answered.

    The CLI already has a browser flow, so asking the person to type a username
    and paste a token is doing by hand what `auth login` does for them. The name
    is read back from the CLI afterwards, because the CLI is the only thing that
    knows which account actually authenticated.
    """
    root = _accounts_root(config_file)
    root.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix="pending-", dir=str(root)))
    pending.chmod(0o700)
    environment = _isolated_environment(pending)
    # Without --force the CLI can decide it is already logged in and keep the
    # previous account, which is exactly how a second account ends up spending
    # the first account's quota under the wrong name.
    command = [str(executable), "auth", "login", "--force"]
    if not launch_browser:
        command.append("--no-launch-browser")
    print(t("cli.kaggle.accounts.login_starting"))
    try:
        result = run_fn(command, check=False, env=environment)
        if result.returncode != 0:
            raise RuntimeError(t("cli.kaggle.accounts.login_failed"))
        username = _authenticated_username(executable, run_fn, environment)
    except RuntimeError:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    target = _account_dir(username, config_file)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    pending.replace(target)
    target.chmod(0o700)
    data = config.load(config_file)
    state = data.setdefault("kaggle", {})
    state.setdefault("accounts", {})[username] = {"browser_login": True}
    state["selected_account"] = username
    config.save(data, config_file)
    print(t("cli.kaggle.accounts.login_done", username=username))
    return username


def register_api_key_file(source, config_file=None, executable=None,
                          run_fn=subprocess.run):
    """Register an account from whatever kaggle.com/settings handed the user.

    That is either the kaggle.json file, which carries the username with it, or
    a bare access token, which does not have to: the CLI introspects the token
    against Kaggle and answers with the account it belongs to, so the name is
    read back rather than typed, exactly as it is after a browser sign-in.
    """
    text = str(source).strip()
    path = Path(text).expanduser()
    try:
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
    except OSError:
        debug.swallowed("cli_kaggle.register_api_key_file read")
    try:
        payload = json.loads(text)
        return register_account(
            payload["username"], {"key": payload["key"]}, config_file)
    except (TypeError, ValueError, KeyError) as error:
        debug.note("cli_kaggle.register_api_key_file not a kaggle.json", error)
    if not text or any(character.isspace() for character in text):
        raise RuntimeError(t("cli.kaggle.accounts.api_key_invalid"))
    if executable is None:
        raise RuntimeError(t("cli.kaggle.accounts.api_key_invalid"))
    return _register_bare_token(text, executable, config_file, run_fn)


def _register_bare_token(token, executable, config_file=None,
                         run_fn=subprocess.run):
    """Ask the CLI who a token belongs to, then file it under that account."""
    root = _accounts_root(config_file)
    root.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix="pending-", dir=str(root)))
    pending.chmod(0o700)
    environment = _isolated_environment(pending)
    token_path = pending / ".kaggle" / "access_token"
    token_path.write_text(token + "\n", encoding="utf-8")
    token_path.chmod(0o600)
    try:
        username = _authenticated_username(executable, run_fn, environment)
    except RuntimeError:
        raise RuntimeError(t("cli.kaggle.accounts.token_rejected"))
    finally:
        shutil.rmtree(pending, ignore_errors=True)
    return register_account(username, {"token": token}, config_file)


def forget_account(username, config_file=None, executable=None,
                   run_fn=subprocess.run, revoke=False):
    """Remove one account from isaacli, optionally revoking it at Kaggle too.

    Forgetting is local by default. Revoking reaches Kaggle and cannot be
    undone from here, so it only happens when it was asked for explicitly.
    """
    data = config.load(config_file)
    state = data.get("kaggle") or {}
    accounts = state.get("accounts") or {}
    if username not in accounts:
        raise RuntimeError(t("cli.kaggle.accounts.missing", username=username))
    if revoke and executable:
        try:
            environment = _account_environment(username, config_file)
            result = _run_capture(
                [str(executable), "auth", "revoke"], run_fn, environment)
            if result.returncode != 0:
                print(t("cli.kaggle.accounts.revoke_failed",
                        error=(result.stderr or result.stdout).strip()))
            else:
                print(t("cli.kaggle.accounts.revoked", username=username))
        except RuntimeError as error:
            print(t("cli.kaggle.accounts.revoke_failed", error=error))
    secret = accounts[username].get("credential")
    if secret:
        config.delete_secret(secret, _secret_path(config_file))
    shutil.rmtree(_account_dir(username, config_file), ignore_errors=True)
    accounts.pop(username, None)
    if state.get("selected_account") == username:
        state["selected_account"] = next(iter(accounts), None)
    config.save(data, config_file)
    print(t("cli.kaggle.accounts.forgotten", username=username))
    return username


def _register_account_interactive(input_fn, config_file=None, executable=None,
                                  run_fn=subprocess.run):
    """Add an account the way the CLI supports, never by typing a username."""
    index = _choose(
        t("cli.kaggle.accounts.add_title"),
        [t("cli.kaggle.accounts.add_browser"), t("cli.kaggle.accounts.add_api_key")],
        input_fn)
    if index == 1:
        print(t("cli.kaggle.accounts.api_key_explain"))
        source = input_fn(t("cli.kaggle.accounts.api_key_prompt")).strip()
        return register_api_key_file(source, config_file, executable, run_fn)
    if executable is None:
        raise RuntimeError(t("cli.kaggle.accounts.login_unavailable"))
    return login_account(executable, config_file, run_fn)


def install_kaggle_cli(input_fn=None, run_fn=subprocess.run, which_fn=shutil.which,
                       home_dir=None, record_path=None):
    """Install Kaggle into an isolated per-user venv and record ownership."""
    found = which_fn("kaggle")
    if found:
        print(t("cli.kaggle.install.found", path=found))
        return Path(found)
    input_fn = input if input_fn is None else input_fn
    home = _home(home_dir)
    env_dir = home / ".local" / "share" / "isaacli" / "kaggle-cli"
    link = home / ".local" / "bin" / "kaggle"
    record = kaggle_install_record(record_path)
    print(t("cli.kaggle.install.plan", env=env_dir, link=link))
    if input_fn(t("cli.kaggle.confirm")).strip().lower() != t("cli.kaggle.confirm_yes"):
        print(t("cli.kaggle.cancelled"))
        return None
    if link.exists() or link.is_symlink():
        print(t("cli.kaggle.install.conflict", path=link))
        return None
    try:
        result = run_fn([sys.executable, "-m", "venv", str(env_dir)], check=False)
        if result.returncode != 0:
            raise RuntimeError(t("cli.kaggle.install.venv_failed"))
        pip = env_dir / "bin" / "python"
        result = run_fn([str(pip), "-m", "pip", "install", "kaggle"], check=False)
        if result.returncode != 0:
            raise RuntimeError(t("cli.kaggle.install.pip_failed"))
        executable = env_dir / "bin" / "kaggle"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(executable)
        result = run_fn([str(link), "--version"], check=False)
        if result.returncode != 0:
            raise RuntimeError(t("cli.kaggle.install.verify_failed"))
        _write_record({
            "version": 1,
            "installed_by": "isaacli",
            "executable": str(link),
            "environment": str(env_dir),
        }, record)
    except (OSError, RuntimeError) as error:
        try:
            if link.is_symlink():
                link.unlink()
            if env_dir.exists():
                shutil.rmtree(env_dir)
        except OSError:
            debug.swallowed("cli_kaggle.install_kaggle_cli cleanup")
        print(t("cli.kaggle.install.failed", error=error))
        return None
    print(t("cli.kaggle.install.success", path=link))
    return link


def _known_credentials(home_dir=None):
    root = _home(home_dir) / ".kaggle"
    return [root / name for name in ("credentials.json", "kaggle.json", "access_token")]


def uninstall_managed_kaggle(remove_credentials=False, home_dir=None,
                             record_path=None, package_owned_fn=_package_owns):
    """Remove only the isolated Kaggle installation recorded by isaacli."""
    record = kaggle_install_record(record_path)
    if not record.exists():
        print(t("cli.uninstall.kaggle.not_managed"))
        return 1
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
        executable = Path(data["executable"])
        environment = Path(data["environment"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        print(t("cli.uninstall.kaggle.invalid_record", error=error))
        return 1
    home = _home(home_dir).resolve()
    expected_environment = home / ".local" / "share" / "isaacli" / "kaggle-cli"
    expected_executable = home / ".local" / "bin" / "kaggle"
    if (environment.absolute() != expected_environment
            or executable.absolute() != expected_executable):
        print(t("cli.uninstall.kaggle.invalid_record",
                error=t("cli.uninstall.kaggle.unsafe_paths")))
        return 1
    if executable.exists() and package_owned_fn(executable):
        print(t("cli.uninstall.kaggle.package_owned", path=executable))
        return 1
    credentials = [path for path in _known_credentials(home) if path.exists()]
    if credentials and not remove_credentials:
        print(t("cli.uninstall.kaggle.credentials", paths=", ".join(map(str, credentials))))
        return 1
    try:
        if executable.is_symlink():
            executable.unlink()
        elif executable.exists():
            print(t("cli.uninstall.kaggle.changed", path=executable))
            return 1
        if environment.exists():
            shutil.rmtree(environment)
        for credential in credentials:
            credential.unlink()
        record.unlink()
    except OSError as error:
        print(t("cli.uninstall.failed", error=error))
        return 1
    print(t("cli.uninstall.kaggle.removed"))
    return 0


def _run_capture(command, run_fn=subprocess.run, env=None):
    return run_fn(command, check=False, capture_output=True, text=True, env=env)


def _quota(executable, run_fn=subprocess.run, env=None):
    result = _run_capture([str(executable), "quota"], run_fn, env)
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or result.stdout).strip() or t("cli.kaggle.quota.failed")
        )
    return result.stdout.strip()


def _quota_summary(text):
    """The one number that decides anything, out of the CLI's quota table.

    The raw table is four lines of column headers and dashes. Folding it into a
    single line with separators produced a wall nobody can read, and the only
    figure that changes a decision is how many GPU hours are left.
    """
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].upper() == "GPU":
            return t("cli.kaggle.accounts.quota_gpu",
                     remaining=parts[2], total=parts[3])
    return " ".join((text or "").split())


def _choose(title, options, input_fn, initial=0, disabled=None):
    """One selection screen, drawn the way the rest of isaacli draws them."""
    return terminal_ui.select(
        title, options, input_fn=input_fn,
        prompt=t("select.prompt"), invalid=t("select.invalid"), initial=initial,
        disabled=disabled,
        more_above=t("ui.more_above", count="{count}"),
        more_below=t("ui.more_below", count="{count}"),
    )


def _forget_account_interactive(names, input_fn, config_file=None, executable=None,
                                run_fn=subprocess.run):
    """Sign out of one registered account, and say what that does and does not do."""
    index = _choose(
        t("cli.kaggle.accounts.forget_title"),
        [*names, t("navigation.back")], input_fn)
    if index >= len(names):
        return None
    username = names[index]
    revoke = _choose(
        t("cli.kaggle.accounts.forget_explain", username=username),
        [t("cli.kaggle.accounts.forget_local"),
         t("cli.kaggle.accounts.forget_revoke")], input_fn) == 1
    return forget_account(username, config_file, executable, run_fn, revoke)


def _account_options(executable, names, run_fn, config_file):
    labels = []
    for username in names:
        try:
            quota = _quota_summary(
                _quota(executable, run_fn,
                       _account_environment(username, config_file)))
        except RuntimeError as error:
            quota = t("cli.kaggle.accounts.quota_unavailable", error=error)
        labels.append(t("cli.kaggle.accounts.option", username=username, quota=quota))
    return labels


def _select_account(executable, input_fn, run_fn=subprocess.run, config_file=None):
    """List account quotas and return the manually selected account and env."""
    data = config.load(config_file)
    accounts = ((data.get("kaggle") or {}).get("accounts") or {})
    if not accounts:
        print(t("cli.kaggle.accounts.none"))
        # Somebody who just signed in has already answered which account to use.
        # Drawing the picker at them again, with "add another account" on it,
        # reads as the sign-in not having worked.
        username = _register_account_interactive(
            input_fn, config_file, executable, run_fn)
        return _use_account(executable, username, run_fn, config_file)
    names = list(accounts)
    options = [
        *_account_options(executable, names, run_fn, config_file),
        t("cli.kaggle.accounts.add_option"),
        t("cli.kaggle.accounts.forget_option"),
    ]
    selected = ((data.get("kaggle") or {}).get("selected_account"))
    index = _choose(
        t("cli.kaggle.accounts.title"), options, input_fn,
        initial=names.index(selected) if selected in names else 0)
    if index == len(names):
        username = _register_account_interactive(
            input_fn, config_file, executable, run_fn)
        return _use_account(executable, username, run_fn, config_file)
    if index == len(names) + 1:
        _forget_account_interactive(names, input_fn, config_file, executable, run_fn)
        # Signing out changes the list this screen just printed, so it is drawn
        # again rather than acting on the stale numbering the user saw.
        return _select_account(executable, input_fn, run_fn, config_file)
    if not 0 <= index < len(names):
        raise RuntimeError(t("cli.kaggle.accounts.invalid"))
    return _use_account(executable, names[index], run_fn, config_file)


def _use_account(executable, username, run_fn=subprocess.run, config_file=None):
    """Check the account really answers as itself, then record it as selected."""
    environment = _account_environment(username, config_file)
    _verify_account(executable, username, environment, run_fn)
    data = config.load(config_file)
    data.setdefault("kaggle", {})["selected_account"] = username
    config.save(data, config_file)
    return username, environment


def _kernel_refs(executable, run_fn=subprocess.run, env=None):
    refs = []
    page = 1
    while True:
        result = _run_capture([
            str(executable), "kernels", "list", "--mine", "--csv",
            "--page", str(page), "--page-size", "100",
        ], run_fn, env)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        page_refs = [row.get("ref") or row.get("Ref") for row in rows]
        refs.extend(ref for ref in page_refs if ref)
        if len(rows) < 100:
            return refs
        page += 1


def live_kernels(executable, run_fn=subprocess.run, env=None):
    """Return every visible non-terminal kernel, querying each unique slug."""
    live = []
    for ref in _kernel_refs(executable, run_fn, env):
        result = _run_capture(
            [str(executable), "kernels", "status", ref], run_fn, env)
        output = (result.stdout + " " + result.stderr).strip()
        if result.returncode != 0:
            # A kernel that never opened a session answers 404 on
            # GetKernelSessionStatus. That is the answer to the question being
            # asked, not a failure: it is not running. Measured against a real
            # account, where notebooks with no session made the whole flow abort
            # before it could list anything. Anything else really is a failure
            # and still stops the flow, because a kernel we cannot ask about
            # might be spending quota right now.
            if "GetKernelSessionStatus" in output and "404" in output:
                debug.note(f"cli_kaggle.live_kernels {ref}", output)
                continue
            raise RuntimeError(output)
        state = next((name for name in TERMINAL_STATES if name in output.upper()), None)
        if state is None:
            live.append((ref, output))
    return live


def _load_model_candidates(path=MODEL_CATALOG_PATH):
    """Load benchmark-backed candidates before hardware fit is applied."""
    required = {
        "name", "repo", "file", "alias", "source", "model_bytes",
        "n_layers", "n_kv_heads", "head_dim", "benchmark",
        "benchmark_source", "scores", "active_ratio",
    }
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        candidates = data["kaggle"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(t("cli.kaggle.catalog.invalid", error=error)) from error
    if (not isinstance(candidates, list) or not candidates
            or any(not isinstance(item, dict) or not required <= item.keys()
                   for item in candidates)):
        raise RuntimeError(t("cli.kaggle.catalog.invalid", error="invalid entries"))
    # The seed is the reviewed list, and the screen says so on each row. A live
    # search result that happens to resolve the same fields does not become
    # reviewed by looking alike.
    return [{**item, "curated": True} for item in candidates]


def models_for_accelerator(machine_shape, catalog_path=MODEL_CATALOG_PATH):
    """Benchmark-backed candidates that fit this exact Kaggle accelerator."""
    accelerator = ACCELERATORS[machine_shape]
    selected = []
    for candidate in _load_model_candidates(catalog_path):
        kv_bytes = hardware.kv_cache_bytes(
            candidate["n_layers"], candidate["n_kv_heads"],
            candidate["head_dim"], MODEL_CONTEXT,
        )
        if hardware.fits(
                candidate["model_bytes"], kv_bytes, accelerator["vram_mb"],
                overhead_mb=accelerator["overhead_mb"]):
            model = dict(candidate)
            model.update({
                "machine_shape": machine_shape,
                "machine_label": accelerator["label"],
                "cuda_arch": accelerator["cuda_arch"],
                "gpu_count": accelerator["gpu_count"],
                "kv_bytes": kv_bytes,
            })
            selected.append(model)
    # Order is a recommendation whether or not it means to be, so it follows the
    # curated order rather than bytes per token. Sorting by speed buried the
    # strongest model in the list: on the two benchmarks both publish,
    # Qwen3.8-27B scores 90.3 and 89.2 against 45.2 and 68.4 for the mixture of
    # experts that reads a tenth of the bytes. Cheap to run is not the same as
    # good, the user said so plainly, and the cost per token is on screen next
    # to each option for whoever weighs it differently.
    return selected


def prepared_models(catalog_path=MODEL_CATALOG_PATH):
    """Compatibility name for models that can be prepared by their owner."""
    return recommended_models(catalog_path)


def recommended_models(catalog_path=MODEL_CATALOG_PATH):
    """Assign every fitting candidate to the smallest preferred accelerator."""
    candidates = _load_model_candidates(catalog_path)
    by_alias = {item["alias"]: item for item in candidates}
    selected = []
    assigned = set()
    for machine_shape in ACCELERATOR_PREFERENCE:
        for model in models_for_accelerator(machine_shape, catalog_path):
            alias = model["alias"]
            if alias in by_alias and alias not in assigned:
                selected.append(model)
                assigned.add(alias)
    return selected


def model_entry(model, no_score=None):
    """One model on one line, because a selection screen draws one line each.

    Three printed lines per model turned six candidates into a wall the user
    could not read. The evidence behind the chosen row is not dropped, it is
    printed once for the row that was actually chosen.
    """
    import model_discovery

    return t(
        "cli.kaggle.models.entry", name=model["name"],
        size=f"{model['model_bytes'] / 1024 ** 3:.2f}",
        machine=model.get("machine_label", ""),
        origin=model_discovery.origin_label(model, t),
        benchmark=(model.get("benchmark")
                   or (no_score if no_score is not None
                       else model_discovery.no_public_score(t))),
    )


def print_model_evidence(model):
    """The sources behind the row that was chosen, which do not fit on it."""
    if model.get("source"):
        print(model["source"])
    if model.get("benchmark_source"):
        print(f"{model['benchmark_source']} ({t('model.discovery.scope')})")


def prepared_weight_probe(executable, username, run_fn=subprocess.run, env=None):
    """Answer whether one exact weight file is already a dataset on this account.

    Precision decides the alias, the alias decides the dataset ref, and moving
    one row on the quantization screen therefore unhooks the prepared input with
    nothing on screen saying so. The account is asked once, and an account that
    cannot be asked answers unknown rather than stopping the screen.
    """
    cache = {}

    def prepared(model):
        if "refs" not in cache:
            try:
                cache["refs"] = _dataset_refs(executable, run_fn, env)
            except RuntimeError as error:
                debug.note("cli_kaggle.prepared_weight_probe", error)
                cache["refs"] = set()
        return _asset_refs(username, model)["model"] in cache["refs"]

    return prepared


def _select_model(input_fn, catalog_path=MODEL_CATALOG_PATH, prepared_fn=None):
    models = prepared_models(catalog_path)
    if not models:
        raise RuntimeError(t("cli.kaggle.models.none"))
    index = _choose(
        "\n\n".join((t("cli.kaggle.models.section"), t("cli.kaggle.models.title"))),
        [model_entry(model) for model in models], input_fn)
    model = models[index]
    print_model_evidence(model)
    return model


DATASET_SLUG_LIMIT = 50


def _model_dataset_slug(alias, limit=DATASET_SLUG_LIMIT):
    """A dataset name short enough for Kaggle that still says what it holds.

    The name was being cut at the limit from the end, and what lives at the end
    of an alias is the precision. `...-Q6_K` and `...-Q8_0` of one repository
    therefore collapsed onto the same name, so publishing the second would land
    on top of the first and the launch would attach whichever was there. The cut
    now happens in the middle, which is the only part that does not identify
    anything.
    """
    prefix = "isaacli-model-"
    slug = re.sub(r"[^a-z0-9-]+", "-", str(alias).lower()).strip("-")
    room = limit - len(prefix)
    if len(slug) <= room:
        return prefix + slug
    tail = slug[-min(len(slug) // 2, 20):]
    tail = tail.split("-", 1)[1] if "-" in tail[1:] else tail.lstrip("-")
    head = slug[:max(1, room - len(tail) - 1)]
    head = head.rsplit("-", 1)[0] if "-" in head[1:] else head
    return f"{prefix}{head.rstrip('-')}-{tail}"


def _asset_refs(username, model):
    binary_slug = BINARY_DATASET_SLUGS.get(
        model.get("cuda_arch"),
        f"isaacli-llama-cuda-sm{model.get('cuda_arch', 'unknown')}-b10502",
    )
    model_slug = MODEL_DATASET_SLUGS.get(
        model.get("alias"), _model_dataset_slug(model["alias"]))
    return {
        "binary": f"{username}/{binary_slug}",
        "model": f"{username}/{model_slug}",
    }


def _dataset_refs(executable, run_fn=subprocess.run, env=None):
    refs = set()
    page = 1
    while True:
        result = _run_capture([
            str(executable), "datasets", "list", "--mine", "--csv",
            "--page", str(page), "--page-size", "100",
        ], run_fn, env)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        for row in rows:
            ref = row.get("ref") or row.get("Ref")
            if ref:
                refs.add(ref)
        if len(rows) < 100:
            return refs
        page += 1


def _available_asset_refs(executable, username, model, run_fn=subprocess.run,
                          env=None):
    expected = _asset_refs(username, model)
    existing = _dataset_refs(executable, run_fn, env)
    return {kind: ref for kind, ref in expected.items() if ref in existing}


def _needs_every_gpu(model):
    """Whether the model really needs both cards, or would only be split.

    Splitting layers across two T4 is capacity, not speed: on a single request
    one card computes while the other waits, and the 026 measurement says so
    without theory, 13.4 tok/s against the 10.8 predicted from the bandwidth of
    ONE card. A model that fits on one card is therefore asked to stay on one.
    When the numbers behind the decision are missing, the split stays, because
    the failure of splitting needlessly is slower, and the failure of not
    splitting when it was needed is a launch that dies out of memory.
    """
    accelerator = ACCELERATORS.get(model.get("machine_shape"))
    if not accelerator or accelerator["gpu_count"] < 2:
        return False
    kv_bytes = model.get("kv_bytes")
    if not model.get("model_bytes") or kv_bytes is None:
        return True
    count = accelerator["gpu_count"]
    return not hardware.fits(
        model["model_bytes"], kv_bytes, accelerator["vram_mb"] // count,
        overhead_mb=accelerator["overhead_mb"] // count)


# Every marker below lands inside a bare double-quoted Python literal in a file
# Kaggle runs on the user's own account, so a value carrying a quote, a
# backslash or a newline stops being data and becomes code. The repository name
# and the selector a user types are already checked where they are typed, but
# the file name is not typed: it is whatever `siblings` says, straight from
# Hugging Face, and it only ever had to end in .gguf. Rendering is the one point
# every path goes through, curated, discovered, exact reference, sibling
# precision, remembered preference and flow validation alike, so the check lives
# here rather than in each of them.
KERNEL_VALUE_PATTERNS = {
    "__MODEL_REPO__": re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"),
    "__MODEL_FILE__": re.compile(r"[A-Za-z0-9_./+ -]+"),
    "__MODEL_ALIAS__": re.compile(r"[A-Za-z0-9_.-]+"),
    "__API_KEY__": re.compile(r"[A-Za-z0-9_-]+"),
    "__CUDA_ARCH__": re.compile(r"[0-9]+"),
    "__MACHINE_SHAPE__": re.compile(r"[A-Za-z0-9]+"),
    "__GPU_COUNT__": re.compile(r"[0-9]+"),
    "__SPLIT_MODE__": re.compile(r"[a-z]+"),
}


def _kernel_value(marker, value):
    """One substituted value, or a refusal naming which one it was.

    An empty value is the flow validation probe, which carries no model at all.
    `..` is refused on top of the pattern because it is legal in a file name and
    is a path traversal in the kernel's own download.
    """
    text = "" if value is None else str(value)
    if text == "":
        return ""
    if ".." in text or not KERNEL_VALUE_PATTERNS[marker].fullmatch(text):
        raise RuntimeError(t("cli.kaggle.kernel.unsafe_value",
                             marker=marker.strip("_").lower(), value=text[:80]))
    return text


def _render_kernel(folder, slug, model, api_key, validation_cpu=False,
                   dataset_sources=None):
    template_name = "flow-validation-cpu.py.tmpl" if validation_cpu else "gpu-server.py.tmpl"
    template = (TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
    values = {
        "__MODEL_REPO__": model["repo"],
        "__MODEL_FILE__": model["file"],
        "__MODEL_ALIAS__": model["alias"],
        "__API_KEY__": api_key,
        "__CUDA_ARCH__": model.get("cuda_arch", ""),
        "__MACHINE_SHAPE__": model.get("machine_shape", ""),
        "__GPU_COUNT__": str(model.get("gpu_count", 0)),
        "__SPLIT_MODE__": "layer" if _needs_every_gpu(model) else "none",
    }
    values = {marker: _kernel_value(marker, value)
              for marker, value in values.items()}
    for marker, value in values.items():
        template = template.replace(marker, value)
    code_name = f"{slug.rsplit('/', 1)[-1]}.py"
    (folder / code_name).write_text(template, encoding="utf-8")
    metadata = {
        "id": slug,
        "title": slug.rsplit("/", 1)[-1].replace("-", " "),
        "code_file": code_name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": not validation_cpu,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    if not validation_cpu:
        metadata["machine_shape"] = model["machine_shape"]
        metadata["dataset_sources"] = list(dataset_sources or [])
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )


def _render_preparation_kernel(folder, slug, cuda_arch):
    # This is the second file this program writes for Kaggle to run on the
    # user's own account, and it substitutes into a bare literal exactly like
    # the GPU one. The comment above `KERNEL_VALUE_PATTERNS` called rendering
    # the single point every value passes through, and that was only true of
    # one of the two renderers.
    template = (TEMPLATE_DIR / "prepare-assets-cpu.py.tmpl").read_text(
        encoding="utf-8")
    template = template.replace(
        "__CUDA_ARCH__", _kernel_value("__CUDA_ARCH__", cuda_arch))
    code_name = f"{slug.rsplit('/', 1)[-1]}.py"
    (folder / code_name).write_text(template, encoding="utf-8")
    metadata = {
        "id": slug,
        "title": slug.rsplit("/", 1)[-1].replace("-", " "),
        "code_file": code_name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _wait_for_kernel(executable, slug, run_fn=subprocess.run, env=None,
                     timeout=PREPARATION_TIMEOUT_SECONDS):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = _run_capture(
            [str(executable), "kernels", "status", slug], run_fn, env)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        output = (result.stdout + " " + result.stderr).upper()
        if "COMPLETE" in output:
            return
        if "ERROR" in output or "CANCELLED" in output:
            raise RuntimeError(t("cli.kaggle.prepare.kernel_failed", slug=slug))
        time.sleep(15)
    raise RuntimeError(t("cli.kaggle.prepare.kernel_timeout", slug=slug))


def _publish_private_dataset(executable, folder, ref, title,
                             run_fn=subprocess.run, env=None):
    metadata = {
        "id": ref,
        "title": title[:50],
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }
    (folder / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    command = [str(executable), "datasets", "create", "-p", str(folder)]
    result = run_fn(
        command, check=False, capture_output=True, text=True, env=env)
    output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    if result.returncode != 0 or "dataset creation error:" in output.lower():
        debug.note("cli_kaggle._publish_private_dataset", output)
        raise RuntimeError(t("cli.kaggle.prepare.publish_failed", ref=ref))


_free_bytes = shutil.disk_usage


def _scratch_root():
    """Stage hundreds of megabytes on disk, not in the machine's memory."""
    root = config.cache_path()
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def _require_space(directory, needed_bytes):
    """Refuse a download that cannot land, before it starts, with the numbers.

    The limit belongs at the entrance. Starting a 15 GiB transfer into a
    filesystem that has 7 GiB spends the whole transfer to find out, and the
    error at the end is about a write, not about the choice that caused it.
    """
    free = _free_bytes(directory).free
    if free >= needed_bytes:
        return
    raise RuntimeError(t(
        "cli.kaggle.prepare.no_space", path=directory,
        needed=f"{needed_bytes / 1024 ** 3:.2f}", free=f"{free / 1024 ** 3:.2f}"))


def _prepare_assets(executable, username, model, available, input_fn,
                    run_fn=subprocess.run, env=None):
    expected = _asset_refs(username, model)
    scratch = _scratch_root()
    if "binary" not in available:
        suffix = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        slug = f"{username}/isaacli-prepare-cpu-{suffix}"
        print(t("cli.kaggle.prepare.cpu", slug=slug))
        with tempfile.TemporaryDirectory(
                prefix="isaacli-kaggle-prepare-", dir=scratch) as temporary:
            folder = Path(temporary)
            _render_preparation_kernel(folder, slug, model["cuda_arch"])
            result = run_fn([
                str(executable), "kernels", "push", "-p", temporary,
                "-t", str(PREPARATION_TIMEOUT_SECONDS),
            ], check=False, env=env)
            if result.returncode != 0:
                raise RuntimeError(t("cli.kaggle.push.failed"))
            _wait_for_kernel(executable, slug, run_fn, env)
            output = folder / "output"
            output.mkdir()
            result = run_fn([
                str(executable), "kernels", "output", slug, "-p", str(output),
            ], check=False, env=env)
            if result.returncode != 0:
                raise RuntimeError(t("cli.kaggle.prepare.output_failed", slug=slug))
            archives = list(output.rglob("llama-cuda-*.tar.gz"))
            if len(archives) != 1:
                raise RuntimeError(t("cli.kaggle.prepare.output_missing"))
            dataset = folder / "binary-dataset"
            dataset.mkdir()
            # Moved, not copied: a second copy of the runtime archive buys
            # nothing and doubles what has to fit while it is being published.
            archives[0].replace(dataset / archives[0].name)
            _publish_private_dataset(
                executable, dataset, expected["binary"],
                f"isaacli CUDA runtime sm{model['cuda_arch']}", run_fn, env)
            print(t("cli.kaggle.prepare.created", ref=expected["binary"]))
            print(t("cli.kaggle.prepare.kernel_remove", slug=slug))
        available["binary"] = expected["binary"]
    if "model" not in available:
        size = model["model_bytes"] / 1024 ** 3
        print(t("cli.kaggle.prepare.weight", size=f"{size:.1f}", name=model["name"]))
        if input_fn(t("cli.kaggle.prepare.weight_confirm")).strip().lower() == t(
                "cli.kaggle.confirm_yes"):
            _require_space(scratch, model["model_bytes"])
            with tempfile.TemporaryDirectory(
                    prefix="isaacli-kaggle-weight-", dir=scratch) as temporary:
                folder = Path(temporary)
                target = folder / model["file"]
                url = model.get("file_url") or (
                    f"https://huggingface.co/{model['repo']}/resolve/main/{model['file']}")
                result = run_fn(
                    ["curl", "-fL", "-o", str(target), url], check=False, env=env)
                if result.returncode != 0:
                    raise RuntimeError(t("cli.kaggle.prepare.download_failed"))
                _publish_private_dataset(
                    executable, folder, expected["model"],
                    f"isaacli model {model['alias']}", run_fn, env)
                print(t("cli.kaggle.prepare.created", ref=expected["model"]))
            available["model"] = expected["model"]
    return available


# The reusable inputs remove the 34 minute build and download path. The same
# model took 43.5 seconds to load in the measured T4 x2 run. A CPU probe with
# the attached 15.33 GiB dataset took 11 minutes 26 seconds from push to status
# check after completion, although its script ran in one second. Thirty minutes
# covers input staging, scheduling, loading and tunnel startup.
URL_DISCOVERY_TIMEOUT = 30 * 60


def discover_tunnel_url(executable, slug, timeout=URL_DISCOVERY_TIMEOUT,
                        popen_fn=subprocess.Popen, env=None):
    process = popen_fn(
        [str(executable), "kernels", "logs", "-f", slug],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([process.stdout], [], [], 0.5)
            if not ready:
                if process.poll() is not None:
                    break
                continue
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                time.sleep(0.1)
                continue
            match = URL_PATTERN.search(line)
            if match:
                return match.group(1)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
    raise RuntimeError(t("cli.kaggle.url.failed", slug=slug))


def save_kaggle_profile(url, slug, model, api_key, config_file=None,
                        account=None):
    data = config.load(config_file)
    profile_name = f"kaggle-{slug.rsplit('/', 1)[-1]}"
    credential = f"{profile_name}-api-key"
    data.setdefault("profiles", {})[profile_name] = {
        "provider": "openai_compatible",
        "provider_name": "Kaggle",
        "base_url": url.rstrip("/") + "/v1",
        "model": model["alias"],
        "credential": credential,
        "num_ctx": 16384,
        "thinking": False,
    }
    data["default_profile"] = profile_name
    state = data.setdefault("kaggle", {})
    # What was chosen and the session it ran in are two different lifetimes. The
    # tunnel URL is minted per kernel and never comes back, so the record and the
    # profile built from it are thrown away when the kernel ends, and that used
    # to take the account, the model and the exact file with them. Keeping the
    # choice apart from the session is what turns the next launch back into one
    # keypress instead of the whole flow again.
    state["preference"] = {"account": account, "model": dict(model)}
    kernels = state.setdefault("kernels", [])
    kernels.append({
        "slug": slug, "url": url,
        "web_url": f"https://www.kaggle.com/code/{slug}",
        "profile": profile_name, "model": model["alias"], "account": account,
        # The window that spent the quota is holding it from this moment on, so
        # a second window opening later adds itself rather than inheriting it.
        "holders": [os.getpid()],
    })
    config.save(data, config_file)
    secret_path = Path(config_file).with_name("secrets.json") if config_file else None
    config.save_secret(credential, api_key, secret_path)
    return profile_name


def remote_leftovers(executable, config_file=None, run_fn=subprocess.run):
    """What this program put on Kaggle, per account, without deleting anything.

    Uninstalling has to reach the account too. Everything here is named by
    isaacli and owned by the authenticated user, and it is only ever reported:
    what happens to somebody's Kaggle account is not decided by an uninstall
    flag on their laptop.
    """
    accounts = ((config.load(config_file).get("kaggle") or {}).get("accounts") or {})
    found = {}
    for username in accounts:
        try:
            environment = _account_environment(username, config_file)
            kernels = [
                ref for ref in _kernel_refs(executable, run_fn, environment)
                if ref.split("/", 1)[-1].startswith("isaacli-")
            ]
            datasets = sorted(
                ref for ref in _dataset_refs(executable, run_fn, environment)
                if ref.split("/", 1)[-1].startswith("isaacli-")
            )
        except RuntimeError as error:
            # An account whose credential no longer works cannot be asked, and
            # that must not stop the local cleanup of every other account.
            debug.note(f"cli_kaggle.remote_leftovers {username}", error)
            continue
        if kernels or datasets:
            found[username] = {"kernels": kernels, "datasets": datasets}
    return found


def delete_remote_leftovers(executable, leftovers, config_file=None,
                            run_fn=subprocess.run):
    """Delete exactly what was listed and confirmed, reporting each failure."""
    removed, failed = [], []
    for username, items in leftovers.items():
        environment = _account_environment(username, config_file)
        for noun, refs in (("kernels", items["kernels"]),
                            ("datasets", items["datasets"])):
            for ref in refs:
                result = _run_capture(
                    [str(executable), noun, "delete", ref, "--yes"],
                    run_fn, environment)
                if result.returncode == 0:
                    removed.append(ref)
                else:
                    failed.append((ref, (result.stderr or result.stdout).strip()))
    return removed, failed


def stop_kernel(executable, slug, run_fn=subprocess.run, env=None):
    """End a session, which on Kaggle means deleting the kernel that owns it.

    There is no `kernels stop`, and for a long time this program said so and
    left the web interface as the only recourse while quota drained. Measured
    against a kernel that really was running: `kernels delete` returned success,
    the tunnel stopped answering with HTTP 530, and the remaining GPU hours
    stopped moving. Deleting is destructive and it is a third party account, so
    nothing here ever decides it on its own.
    """
    result = _run_capture(
        [str(executable), "kernels", "delete", slug, "--yes"], run_fn, env)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip()
                            or t("cli.kaggle.stop.failed", slug=slug))
    return slug


def run_stop_kernels(input_fn=None, run_fn=subprocess.run, config_file=None,
                     home_dir=None, record_path=None, which_fn=shutil.which):
    """Stop a session this account has running, chosen explicitly."""
    input_fn = input if input_fn is None else input_fn
    executable = install_kaggle_cli(
        input_fn=input_fn, run_fn=run_fn, which_fn=which_fn,
        home_dir=home_dir, record_path=record_path,
    )
    if executable is None:
        return 1
    try:
        account, environment = _select_account(
            executable, input_fn, run_fn, config_file)
        username = _authenticated_username(executable, run_fn, environment)
        live = live_kernels(executable, run_fn, environment)
    except RuntimeError as error:
        print(t("cli.kaggle.failed", error=error))
        return 1
    for record in _prune_dead_kernels(live, config_file, account, username):
        print(t("cli.kaggle.pruned", slug=record.get("slug", "?")))
    if not live:
        print(t("cli.kaggle.stop.none"))
        return 0
    slugs = [ref for ref, _state in live]
    index = _choose(
        t("cli.kaggle.stop.title"),
        [t("cli.kaggle.stop.option", slug=ref, state=state) for ref, state in live]
        + [t("navigation.back")], input_fn)
    if index >= len(slugs):
        print(t("cli.kaggle.cancelled"))
        return 130
    print(t("cli.kaggle.stop.explain", slug=slugs[index]))
    if input_fn(t("cli.kaggle.stop.confirm")).strip().lower() != t(
            "cli.kaggle.confirm_yes"):
        print(t("cli.kaggle.cancelled"))
        return 130
    try:
        stop_kernel(executable, slugs[index], run_fn, environment)
    except RuntimeError as error:
        print(t("cli.kaggle.failed", error=error))
        return 1
    print(t("cli.kaggle.stop.stopped", slug=slugs[index]))
    # The endpoint that kernel published cannot come back, so the record and the
    # profile built from it go with it instead of waiting for the next run.
    for record in _prune_dead_kernels(
            [item for item in live if item[0] != slugs[index]],
            config_file, account, username):
        print(t("cli.kaggle.pruned", slug=record.get("slug", "?")))
    return 0


def _drop_records(data, removed, config_file=None):
    """Remove the profile each dead record created, when it still owns it.

    A profile is only dropped when it still holds the exact URL that record
    created it with, so a profile the user edited or reused is left alone.
    """
    profiles = data.get("profiles") or {}
    for record in removed:
        profile = profiles.get(record.get("profile"))
        if not profile or not record.get("url"):
            continue
        if profile.get("base_url", "").rstrip("/") != record["url"].rstrip("/") + "/v1":
            continue
        profiles.pop(record["profile"])
        config.delete_secret(profile.get("credential"), _secret_path(config_file))
        if data.get("default_profile") == record["profile"]:
            data["default_profile"] = next(iter(profiles), None)


def _prune_dead_kernels(live, config_file=None, account=None, username=None):
    """Forget the kernels that are over, and the profiles they left behind.

    A tunnel URL is minted per session and never comes back, so a record whose
    kernel has no session names an address that cannot answer again, and the
    profile built from it is pointed at nothing. They accumulate: the first real
    account reached eight records. Only Kaggle decides what is over, only this
    account's records are touched, and a profile is only removed when it still
    holds the exact dead URL that record created it with, so a profile the user
    edited or reused is left alone.
    """
    data = config.load(config_file)
    state = data.get("kaggle") or {}
    live_refs = {ref for ref, _state in live}
    kept, removed = [], []
    for record in state.get("kernels") or []:
        slug = record.get("slug") or ""
        owned = record.get("account") == account or (
            record.get("account") is None
            and bool(username) and slug.startswith(f"{username}/"))
        (kept if slug in live_refs or not owned else removed).append(record)
    if not removed:
        return []
    _drop_records(data, removed, config_file)
    state["kernels"] = kept
    config.save(data, config_file)
    return removed


def _forget_kernel_record(slug, config_file=None):
    """Forget one kernel by slug, leaving every other record untouched.

    The prune above answers "which of my kernels are over" and needs the live
    list to say so. Stopping one kernel already knows the answer for that slug
    alone, and must not decide anything about the others: another isaacli may be
    serving from one of them right now.
    """
    data = config.load(config_file)
    state = data.setdefault("kaggle", {})
    records = state.get("kernels") or []
    removed = [item for item in records if item.get("slug") == slug]
    if not removed:
        return []
    _drop_records(data, removed, config_file)
    state["kernels"] = [item for item in records if item.get("slug") != slug]
    config.save(data, config_file)
    return removed


def run_prepare_assets(input_fn=None, run_fn=subprocess.run, config_file=None,
                       home_dir=None, record_path=None, which_fn=shutil.which):
    """Build the reusable Kaggle assets on their own, spending no GPU quota.

    Preparation only ever existed grafted onto the GPU launch, so the one step
    that costs nothing could not be reached without starting the one that costs
    hours. It is a CPU kernel and a private dataset, and it is what removes the
    34 minute compile and download from the GPU clock afterwards.
    """
    input_fn = input if input_fn is None else input_fn
    executable = install_kaggle_cli(
        input_fn=input_fn, run_fn=run_fn, which_fn=which_fn,
        home_dir=home_dir, record_path=record_path,
    )
    if executable is None:
        return 1
    try:
        _account, environment = _select_account(
            executable, input_fn, run_fn, config_file)
        username = _authenticated_username(executable, run_fn, environment)
        model = _select_model(
            input_fn,
            prepared_fn=prepared_weight_probe(
                executable, username, run_fn, environment))
        available = _available_asset_refs(
            executable, username, model, run_fn, environment)
    except RuntimeError as error:
        print(t("cli.kaggle.failed", error=error))
        return 1
    for ref in available.values():
        print(t("cli.kaggle.assets.available", ref=ref))
    missing = [kind for kind in ("binary", "model") if kind not in available]
    if not missing:
        print(t("cli.kaggle.prepare.nothing_missing", username=username))
        return 0
    # Announcing both steps when only one is missing promises a CPU kernel that
    # will not run, and the plan is the thing being consented to.
    print(t("cli.kaggle.prepare.plan", username=username))
    if "binary" in missing:
        print(t("cli.kaggle.prepare.plan_binary", arch=model["cuda_arch"]))
    if "model" in missing:
        print(t("cli.kaggle.prepare.plan_model",
                size=f"{model['model_bytes'] / 1024 ** 3:.1f}"))
    if input_fn(t("cli.kaggle.prepare.confirm")).strip().lower() != t(
            "cli.kaggle.confirm_yes"):
        print(t("cli.kaggle.cancelled"))
        return 130
    try:
        available = _prepare_assets(
            executable, username, model, available, input_fn, run_fn, environment)
    except (OSError, RuntimeError) as error:
        print(t("cli.kaggle.failed", error=error))
        return 1
    print(t("cli.kaggle.prepare.done", count=len(available)))
    return 0


def _probe_url(base_url):
    """The route that answers this question, which is not the one under /v1.

    Measured against llama-server b10502 launched with `--api-key`: `/v1/models`
    returns 200 with no credential at all, `/health` returns 200 with no
    credential, and `/props` returns 401 without the key and 200 with it. Only
    the last one answers what is being asked here.
    """
    root = profile_base = str(base_url).rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return (root or profile_base) + "/props"


def _endpoint_answers(profile, secret_path=None,
                      urlopen_fn=urllib.request.urlopen, timeout=10):
    """Whether the endpoint a saved profile names is serving this key now.

    "The endpoint responds" is not the question. A tunnel that is up answers
    `/v1/models` to anybody, so probing that route reactivated a profile whose
    stored key no longer opened anything, and the failure surfaced at the user's
    first real question instead of here, where it can still be explained.
    """
    key = config.load_secret(profile.get("credential"), secret_path)
    if not key:
        # Without a key there is nothing to prove, and a route that requires one
        # would refuse for the wrong reason. Say so rather than guess.
        debug.note("cli_kaggle._endpoint_answers key",
                   "the saved profile has no stored key to prove")
        return False
    request = urllib.request.Request(
        _probe_url(profile["base_url"]),
        headers={"Authorization": "Bearer " + key},
    )
    try:
        with urlopen_fn(request, timeout=timeout) as answer:
            if answer.status == 200:
                return True
            debug.note("cli_kaggle._endpoint_answers status",
                       f"saved endpoint returned HTTP {answer.status}")
    except urllib.error.HTTPError as error:
        # 401 here is the answer, not a transport failure: the tunnel is up and
        # the stored key does not open it.
        debug.note("cli_kaggle._endpoint_answers status",
                   f"saved endpoint refused with HTTP {error.code}")
    except (urllib.error.URLError, OSError, TimeoutError):
        debug.swallowed("cli_kaggle._endpoint_answers probe")
    return False


def _existing_executable(which_fn=shutil.which, home_dir=None):
    """The Kaggle CLI already on this machine. Never installs anything.

    Closing the program is the wrong moment to offer an installation: what is
    being asked for here is cleanup of something that only exists because the
    CLI was there a moment ago.
    """
    found = which_fn("kaggle")
    if found:
        return Path(found)
    link = _home(home_dir) / ".local" / "bin" / "kaggle"
    return link if link.exists() else None


def profile_kernel_record(profile_name, config_file=None):
    """The kernel this program launched for `profile_name`, or None."""
    if not profile_name:
        return None
    for record in reversed(
            (config.load(config_file).get("kaggle") or {}).get("kernels") or []):
        if record.get("profile") == profile_name and record.get("slug"):
            return record
    return None


def _record_environment(record, config_file=None):
    account = record.get("account") or (
        (config.load(config_file).get("kaggle") or {}).get("selected_account"))
    if not account:
        raise RuntimeError(t("cli.kaggle.accounts.none"))
    return _account_environment(account, config_file)


def _kernel_lock(config_file=None):
    """Serialise the read, change and write of the holder list across processes.

    Two windows opening at the same moment both read the record, both add
    themselves and one of the two writes last, so one holder is silently lost
    and the kernel it was using can be deleted underneath it. `config.save` is
    atomic, which keeps the file whole, and that is a different guarantee from
    keeping the update whole.
    """
    path = (Path(config_file) if config_file else config.config_path()).with_name(
        "kaggle-kernels.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except OSError:
        # A filesystem with no working locks is worse served by refusing to
        # close the session than by the race this was guarding against.
        debug.swallowed("cli_kaggle._kernel_lock flock")
    return handle


def _holder_alive(pid):
    """Whether the window that wrote this number is still running.

    A window that was killed rather than closed never removes itself, and
    treating its number as live would pin the kernel, and the quota it spends,
    on a process that no longer exists.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists and belongs to somebody else. Existing is the question.
        return True
    except OSError:
        return False
    return True


def _update_holders(profile_name, config_file, change):
    """Apply `change` to one record's holder list under the cross-process lock."""
    handle = _kernel_lock(config_file)
    try:
        data = config.load(config_file)
        for record in reversed((data.get("kaggle") or {}).get("kernels") or []):
            if record.get("profile") != profile_name or not record.get("slug"):
                continue
            holders = [
                number for number in record.get("holders") or []
                if _holder_alive(number)
            ]
            record["holders"] = change(holders)
            config.save(data, config_file)
            return record["slug"], record["holders"]
        return None, []
    finally:
        handle.close()


def hold_profile_session(profile_name, config_file=None, pid=None):
    """Record this window as a user of the kernel behind `profile_name`.

    Reusing an endpoint that already answers is free, so every window does it,
    and the first one to close used to delete the kernel the others were still
    talking to. Holding is what tells the last window out that it really is the
    last, which is also what keeps the quota brake inside the program.
    """
    pid = os.getpid() if pid is None else int(pid)
    slug, _holders = _update_holders(
        profile_name, config_file,
        lambda holders: [*[n for n in holders if n != pid], pid])
    return slug


def release_profile_session(profile_name, config_file=None, pid=None):
    """Drop this window from the record, and answer who is still holding it."""
    pid = os.getpid() if pid is None else int(pid)
    slug, holders = _update_holders(
        profile_name, config_file,
        lambda holders: [n for n in holders if n != pid])
    return slug, holders


def ensure_profile_session(profile_name, input_fn=None, config_file=None,
                           urlopen_fn=urllib.request.urlopen, run_kaggle_fn=None,
                           pid=None):
    """Reopen the saved Kaggle kernel, or ask before spending quota on another.

    Reactivating a kernel that is still serving costs nothing, so it happens
    without asking. Pushing a new one spends GPU quota by wall clock and does
    not come back, so it is never automatic: opening the program is not consent
    to spend hours of somebody's weekly allowance.
    """
    record = profile_kernel_record(profile_name, config_file)
    if record is None:
        return None
    profile = (config.load(config_file).get("profiles") or {}).get(profile_name)
    if not profile or not profile.get("base_url"):
        return None
    if _endpoint_answers(profile, _secret_path(config_file), urlopen_fn):
        debug.note("cli_kaggle.ensure_profile_session",
                   f"{record['slug']} is still serving, nothing was pushed")
        hold_profile_session(profile_name, config_file, pid)
        return "live"
    print(t("cli.kaggle.session.gone", slug=record["slug"]))
    _forget_kernel_record(record["slug"], config_file)
    input_fn = input if input_fn is None else input_fn
    try:
        answer = input_fn(t("cli.kaggle.session.relaunch")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        answer = ""
    if answer != t("cli.kaggle.confirm_yes"):
        print(t("cli.kaggle.session.declined"))
        return "declined"
    if run_kaggle_fn is None:
        # The same screens the `/kaggle` command uses, so relaunching offers the
        # live model list instead of a second, poorer version of it.
        import setup_ollama

        run_kaggle_fn = setup_ollama.run_kaggle
    code = run_kaggle_fn(config_file=config_file, input_fn=input_fn)
    return "relaunched" if code == 0 else "failed"


def stop_profile_session(profile_name, config_file=None, run_fn=subprocess.run,
                         which_fn=shutil.which, home_dir=None, pid=None):
    """End the kernel this program launched for the profile it just used.

    What isaacli starts, isaacli stops. A Kaggle session spends quota by wall
    clock until something deletes the kernel, and leaving that to the user put
    the only brake on the spending outside the program. Only the kernel behind
    the profile this process opened is touched: a second isaacli serving from
    another kernel of the same account must not have it deleted underneath it,
    and `isaacli kaggle --stop` is how those are reached.

    Two windows can be talking to the same kernel, because reusing an endpoint
    that already answers is free and both of them do it. Closing one of them
    stops nothing while the other still holds the record; the last one out is
    what ends the session, so the brake stays inside the program either way.
    """
    record = profile_kernel_record(profile_name, config_file)
    if record is None:
        return None
    _slug, holders = release_profile_session(profile_name, config_file, pid)
    if holders:
        print(t("cli.kaggle.session.still_used",
                slug=record["slug"], count=len(holders)))
        return None
    executable = _existing_executable(which_fn, home_dir)
    if executable is None:
        print(t("cli.kaggle.session.no_cli", slug=record["slug"]))
        return None
    print(t("cli.kaggle.session.stopping", slug=record["slug"]))
    try:
        environment = _record_environment(record, config_file)
        stop_kernel(executable, record["slug"], run_fn, environment)
    except RuntimeError as error:
        # Saying nothing here would leave quota draining behind a screen that
        # already said goodbye.
        print(t("cli.kaggle.session.stop_failed", slug=record["slug"], error=error))
        return None
    _forget_kernel_record(record["slug"], config_file)
    print(t("cli.kaggle.session.stopped", slug=record["slug"]))
    return record["slug"]


def _reactivate_live_profile(live, config_file=None, account=None,
                             urlopen_fn=urllib.request.urlopen):
    """Reactivate a saved profile only when its recorded kernel and API answer."""
    data = config.load(config_file)
    live_refs = {ref for ref, _state in live}
    secret_path = Path(config_file).with_name("secrets.json") if config_file else None
    for record in reversed((data.get("kaggle") or {}).get("kernels") or []):
        if account and record.get("account") not in (None, account):
            continue
        if record.get("slug") not in live_refs:
            continue
        profile_name = record.get("profile")
        if not profile_name and record.get("url"):
            expected_url = record["url"].rstrip("/") + "/v1"
            profile_name = next((
                name for name, item in (data.get("profiles") or {}).items()
                if item.get("base_url", "").rstrip("/") == expected_url
            ), None)
        profile = (data.get("profiles") or {}).get(profile_name)
        if not profile or profile.get("model") == "isaacli-flow-probe":
            continue
        if not _endpoint_answers(profile, secret_path, urlopen_fn):
            continue
        data["default_profile"] = profile_name
        config.save(data, config_file)
        return profile_name, record
    return None, None


def _live_holders(profile_name, config_file=None, pid=None):
    """The other windows holding this record right now, this one aside."""
    pid = os.getpid() if pid is None else int(pid)
    record = profile_kernel_record(profile_name, config_file)
    if record is None:
        return []
    return [
        number for number in record.get("holders") or []
        if number != pid and _holder_alive(number)
    ]


def _offer_switch(record, profile_name, input_fn, config_file=None):
    """Keep the kernel that is serving, or end it and choose another model.

    Reactivating whatever was already up and returning made changing model from
    inside the program impossible: the model list was never drawn, so the only
    way through was to know that `isaacli kaggle --stop` had to be run first.
    Both answers cost something and the screen says which: keeping it spends
    nothing, replacing it deletes a running kernel and pushes one that starts
    spending quota again.
    """
    holders = _live_holders(profile_name, config_file)
    index = _choose(
        t("cli.kaggle.switch.title",
          model=record.get("model", "?"), slug=record.get("slug", "?")),
        [t("cli.kaggle.switch.keep", model=record.get("model", "?")),
         t("cli.kaggle.switch.replace") if not holders
         else t("cli.kaggle.switch.replace_held", count=len(holders))],
        input_fn,
        # Ending it would take the session away from a window that is still
        # using it. The option stays visible, refusing with its reason, because
        # an option that quietly disappears reads as a program that forgot.
        disabled={1} if holders else None,
    )
    return index == 0


MODEL_FILE_URL_PREFIX = "https://huggingface.co/"


def _replayable_model(model):
    """Whether a remembered model is still the shape this program wrote.

    Everything else about a saved model is checked where it is used, but the
    saved copy skips the screens that produced it: it is read back out of
    `config.json`, which is a file a person edits. The values then reach a path
    join, a `curl` target, a dataset name and a literal in a kernel Kaggle runs
    on the user's account, and the preparation half of that runs before the
    launch is rendered. Checking the shape here is checking it at the entrance,
    which is where the rest of this program puts its limits.
    """
    for marker, key in (("__MODEL_ALIAS__", "alias"), ("__MODEL_REPO__", "repo"),
                        ("__MODEL_FILE__", "file"),
                        ("__CUDA_ARCH__", "cuda_arch"),
                        ("__MACHINE_SHAPE__", "machine_shape")):
        try:
            _kernel_value(marker, model.get(key))
        except RuntimeError:
            return key
    for key in ("model_bytes", "kv_bytes", "gpu_count", "n_layers",
                "n_kv_heads", "head_dim"):
        value = model.get(key)
        if value is not None and not isinstance(value, (int, float)):
            return key
    url = model.get("file_url")
    if url is not None and not str(url).startswith(MODEL_FILE_URL_PREFIX):
        # The only writer of this field builds it from that root, and the value
        # is handed to `curl -o`, where a `file://` would copy something off
        # this machine into a dataset instead of downloading a weight.
        return "file_url"
    return None


def stored_preference(config_file=None):
    """The account and model of the last launch, when both still make sense.

    An account that has been signed out of cannot be repeated, and offering it
    would be offering something that fails at the first command.
    """
    state = config.load(config_file).get("kaggle") or {}
    preference = state.get("preference") or {}
    account = preference.get("account")
    model = preference.get("model")
    if not account or not isinstance(model, dict) or not model.get("alias"):
        return None
    edited = _replayable_model(model)
    if edited:
        debug.note("cli_kaggle.stored_preference",
                   f"the remembered {edited} is not the shape this program "
                   "saved, so the last choice is not repeated")
        return None
    if account not in (state.get("accounts") or {}):
        debug.note("cli_kaggle.stored_preference",
                   f"{account} is no longer registered, so it is not offered")
        return None
    return preference


def _offer_preference(preference, input_fn):
    """Repeat the last choice, or open the screens that change it."""
    model = preference["model"]
    index = _choose(
        t("cli.kaggle.preference.title"),
        [t("cli.kaggle.preference.repeat",
           model=model.get("name") or model.get("alias"),
           account=preference["account"]),
         t("cli.kaggle.preference.change")],
        input_fn)
    return preference if index == 0 else None


def run_kaggle(validation_cpu=False, input_fn=None, run_fn=subprocess.run,
                popen_fn=subprocess.Popen, config_file=None, home_dir=None,
                record_path=None, which_fn=shutil.which,
                urlopen_fn=urllib.request.urlopen):
    """Run the user-confirmed Kaggle setup, push, discovery and profile cycle."""
    input_fn = input if input_fn is None else input_fn
    print(t("cli.kaggle.terms"))
    print(t("cli.kaggle.account"))
    executable = install_kaggle_cli(
        input_fn=input_fn, run_fn=run_fn, which_fn=which_fn,
        home_dir=home_dir, record_path=record_path,
    )
    if executable is None:
        return 1
    preference = None if validation_cpu else stored_preference(config_file)
    if preference:
        preference = _offer_preference(preference, input_fn)
    try:
        if preference:
            account, environment = _use_account(
                executable, preference["account"], run_fn, config_file)
        else:
            account, environment = _select_account(
                executable, input_fn, run_fn, config_file)
        quota = _quota(executable, run_fn, environment)
        # The raw answer is a four line table of headers and dashes. Only the
        # remaining GPU hours change a decision here, and the account picker
        # already shows that same figure the same way.
        debug.note("cli_kaggle.run_kaggle quota", quota)
        print(t("cli.kaggle.quota", quota=_quota_summary(quota)))
        username = _authenticated_username(executable, run_fn, environment)
        live = live_kernels(executable, run_fn, environment)
    except RuntimeError as error:
        print(t("cli.kaggle.failed", error=error))
        return 1
    for record in _prune_dead_kernels(live, config_file, account, username):
        print(t("cli.kaggle.pruned", slug=record.get("slug", "?")))
    if live:
        for ref, state in live:
            print(t("cli.kaggle.live", slug=ref, state=state,
                    url=f"https://www.kaggle.com/code/{ref}"))
        if not validation_cpu:
            profile, record = _reactivate_live_profile(
                live, config_file, account=account, urlopen_fn=urlopen_fn,
            )
            if profile:
                keep = _offer_switch(record, profile, input_fn, config_file)
                if keep:
                    print(t("cli.kaggle.reused", profile=profile,
                            url=record["url"] + "/v1"))
                    return 0
                try:
                    stop_kernel(executable, record["slug"], run_fn, environment)
                except RuntimeError as error:
                    print(t("cli.kaggle.failed", error=error))
                    return 1
                _forget_kernel_record(record["slug"], config_file)
                print(t("cli.kaggle.switch.stopped", slug=record["slug"]))
                live = [item for item in live if item[0] != record["slug"]]
            saved_refs = {
                item.get("slug") for item in
                (config.load(config_file).get("kaggle") or {}).get("kernels", [])
                if item.get("account") in (None, account)
            }
            if any(ref in saved_refs for ref, _state in live):
                print(t("cli.kaggle.unresponsive"))
                return 1
        # Ending the one kernel that was serving can leave the account with
        # nothing running, and then this is an ordinary first launch.
        if live:
            print(t("cli.kaggle.second_refused"))
            return 1
    if validation_cpu:
        print(t("cli.kaggle.cpu_only"))
        model = {"repo": "", "file": "", "alias": "isaacli-flow-probe"}
    else:
        if preference:
            # The screens are what cost the user time, not the choice. Naming
            # what is about to be launched is what keeps skipping them honest.
            model = preference["model"]
            print(t("cli.kaggle.preference.using",
                    model=model.get("name") or model.get("alias"),
                    machine=model.get("machine_label", "")))
        else:
            try:
                model = _select_model(
                    input_fn,
                    prepared_fn=prepared_weight_probe(
                        executable, username, run_fn, environment))
            except RuntimeError as error:
                print(error)
                return 1
    dataset_sources = []
    if not validation_cpu:
        try:
            available = _available_asset_refs(
                executable, username, model, run_fn, environment)
        except RuntimeError as error:
            print(t("cli.kaggle.failed", error=error))
            return 1
        missing = [kind for kind in ("binary", "model") if kind not in available]
        if missing:
            # What this costs depends on which input is missing, and the
            # difference is the whole point of having prepared them. Announcing
            # a 34 minute compile to an account that already owns the compiled
            # runtime overstates the price of the launch being consented to.
            print(t("cli.kaggle.assets.missing", username=username))
            if "binary" in missing:
                print(t("cli.kaggle.assets.cost_build"))
            if "model" in missing:
                print(t("cli.kaggle.assets.cost_download",
                        size=f"{model['model_bytes'] / 1024 ** 3:.1f}"))
            if input_fn(t("cli.kaggle.prepare.confirm")).strip().lower() == t(
                    "cli.kaggle.confirm_yes"):
                try:
                    available = _prepare_assets(
                        executable, username, model, available, input_fn,
                        run_fn, environment)
                except (OSError, RuntimeError) as error:
                    print(t("cli.kaggle.failed", error=error))
                    return 1
            if len(available) < 2:
                print(t("cli.kaggle.assets.self_contained"))
        dataset_sources = list(available.values())
        for ref in dataset_sources:
            print(t("cli.kaggle.assets.available", ref=ref))
    if input_fn(t("cli.kaggle.push.confirm")).strip().lower() != t("cli.kaggle.confirm_yes"):
        print(t("cli.kaggle.cancelled"))
        return 130
    suffix = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    kind = "flow-cpu" if validation_cpu else "gpu"
    slug = f"{username}/isaacli-{kind}-{suffix}"
    api_key = secrets.token_urlsafe(24)
    try:
        with tempfile.TemporaryDirectory(prefix="isaacli-kaggle-") as temporary:
            _render_kernel(
                Path(temporary), slug, model, api_key, validation_cpu,
                dataset_sources=dataset_sources)
            result = run_fn([str(executable), "kernels", "push", "-p", temporary,
                             "-t", str(SESSION_TIMEOUT_SECONDS)], check=False,
                            env=environment)
            if result.returncode != 0:
                raise RuntimeError(t("cli.kaggle.push.failed"))
        print(t("cli.kaggle.pushed", slug=slug,
                url=f"https://www.kaggle.com/code/{slug}"))
        url = discover_tunnel_url(
            executable, slug, popen_fn=popen_fn, env=environment)
        profile = save_kaggle_profile(
            url, slug, model, api_key, config_file, account=account)
    except (OSError, RuntimeError) as error:
        # Giving up here does not stop anything: the kernel was already pushed
        # and keeps spending quota that does not come back, so say that instead
        # of only offering a link.
        print(t("cli.kaggle.failed", error=error))
        print(t("cli.kaggle.stop_spending",
                url=f"https://www.kaggle.com/code/{slug}"))
        return 1
    print(t("cli.kaggle.ready", profile=profile, url=url + "/v1"))
    print(t("cli.kaggle.stop", url=f"https://www.kaggle.com/code/{slug}"))
    return 0
