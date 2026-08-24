"""Kaggle CLI installation and explicit remote-kernel orchestration."""
import csv
import fcntl
import hashlib
import io
import json
import os
import pwd
import re
import select
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import config
import debug
import hardware
import terminal_ui
import units
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
# The rungs the launch screen offers for that ceiling. The first is the default
# because the useful work of the run that overspent on 2026-08-23 fit in 38
# minutes, and one hour is the smallest rung that does not cut a slow load
# short: the longest load measured on this account, 22.53 GiB off an attached
# dataset, took over half an hour before the server answered.
SESSION_CEILING_HOURS = (1, 2, 3, 4)
# The client's heartbeat, and the silence the kernel reads as the client being
# gone. The period is not a preference: the kernel notices anything at the
# granularity of its own watch, which is the 30 s sleep in its serving loop, so
# beating faster buys nothing it can see and beating slower would let a live
# session look dead between two of its own checks.
#
# The tolerance is ten of those beats. A single missed beat is a network blip:
# measured on this account on 2026-08-21, the same request reported 9.12 tok/s
# at the server and delivered 8.79 tok/s through the tunnel, a gap of about 4%,
# so the round trip is far under a second and one failure in thirty is noise.
# Ten consecutive failures over five minutes is not noise. Five minutes is also
# the whole exposure of being wrong, against a smallest agreed ceiling of an
# hour, so a false positive costs 1.4% of the cheapest launch and a false
# negative used to cost four hours.
HEARTBEAT_SECONDS = 30
SESSION_IDLE_SECONDS = 10 * HEARTBEAT_SECONDS
ACCELERATORS = {
    "NvidiaTeslaP100": {
        "label": "P100 16 GB", "vram_mb": 16384,
        "overhead_mb": hardware.overhead_mb(1), "cuda_arch": "60",
        "gpu_count": 1,
        # What heads a column: short enough to sit above it, specific enough to
        # say which card the row was drawn against.
        "column": "P100",
        # Per card, never the pair added together: llama.cpp splits layers
        # across cards and decodes through them in turn, so a token still
        # crosses one bus at a time. The figure is the vendor's, which is what
        # makes the throughput column an estimate and says so in the legend.
        "bandwidth_gbs": hardware.gpu_bandwidth("Tesla P100"),
    },
    # 30720, not the 32768 two 16 GB cards suggest. Read inside a session on
    # 2026-08-22 with nvidia-smi: each T4 reports 15360 MiB total, so the pair
    # is 2048 MiB short of the nominal figure this used to carry. That matters
    # more than it looks. The 1536 MiB reserve subtracted below was meant to
    # keep room for the server's own buffers, and against a number 2048 MiB too
    # high it reserved nothing at all: it left the arithmetic 512 MiB past what
    # the cards physically hold. A launch on 2026-08-21 died exactly there,
    # allocating the last 3200 MiB of a cache the pair was said to have room
    # for.
    "NvidiaTeslaT4": {
        "label": "T4 x2, 2 x 16 GB", "vram_mb": 30720,
        "overhead_mb": hardware.overhead_mb(2), "cuda_arch": "75",
        "gpu_count": 2,
        "column": "T4 x2",
        "bandwidth_gbs": hardware.gpu_bandwidth("Tesla T4"),
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
# The first line of the help the Kaggle CLI prints, on stdout, when every
# authentication mechanism it knows has come up empty. It is matched rather
# than parsed because it is the whole difference between "this account is
# signed out" and "the call failed": both arrive here as a non-zero exit and a
# blob of text, and telling a person to check their credentials when the call
# actually failed for another reason is the error message blaming the wrong
# cause.
AUTH_HELP_MARKER = "Authentication required to call the Kaggle API"


def _home(home_dir=None):
    return Path(home_dir) if home_dir is not None else Path.home()


def _real_home():
    """The home of the account running isaacli, whatever HOME currently says.

    `_isolated_environment` moves HOME on purpose, and it used to derive the
    `PYTHONUSERBASE` pin from `os.environ["HOME"]`, which is the real home only
    while isaacli itself runs in it. Measured on 2026-08-23 by running the
    `--stop` flow with HOME already pointing at an account folder: the pin
    landed inside that folder, and the Kaggle CLI installed with `pip --user`
    answered `ModuleNotFoundError: No module named 'kaggle'` instead of running.
    The password database does not follow HOME, so it still answers correctly
    from inside an environment that has already been redirected once.
    """
    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except KeyError:
        # No passwd entry for this uid happens inside minimal containers. HOME
        # is then the only answer there is, and saying so beats guessing.
        debug.note("cli_kaggle._real_home",
                   f"no passwd entry for uid {os.getuid()}, falling back to HOME")
        return Path(os.environ.get("HOME") or Path.home())


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
    environment.setdefault("PYTHONUSERBASE", str(_real_home() / ".local"))
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


def _quota_remaining_hours(text):
    """The GPU hours left, as a number, or None when the table cannot say.

    The same row `_quota_summary` reads, parsed instead of formatted, because
    the ceiling screen has to compare against it rather than print it. A table
    that does not answer is not an error here: the ceiling is still chosen, the
    screen just does not get to say how much room is left.
    """
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].upper() == "GPU":
            try:
                return float(parts[2].rstrip("hH"))
            except ValueError:
                debug.note("cli_kaggle._quota_remaining_hours", line)
                return None
    return None


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
            # This goes on a row of a selection screen. The Kaggle CLI answers
            # a signed-out account with a thousand characters of instructions,
            # and folding that into a row turns the picker into a wall nobody
            # can read: the row only has to say why this account cannot be
            # measured. The text itself still reaches --debug.
            debug.note(f"cli_kaggle._account_options {username}", error)
            quota = t("cli.kaggle.accounts.quota_signed_out") if _signed_out(error) \
                else t("cli.kaggle.accounts.quota_unavailable",
                       error=" ".join(str(error).split())[:120])
        labels.append(t("cli.kaggle.accounts.option", username=username, quota=quota))
    return labels


def _select_account(executable, input_fn, run_fn=subprocess.run, config_file=None,
                    verify=True):
    """List account quotas and return the manually selected account and env.

    `verify=False` is for the paths that only ever delete. Verification is what
    keeps a launch from spending one account's hours under another account's
    name, and it costs a quota call that a signed-out account cannot answer, so
    demanding it on the brake would mean the brake fails exactly when it is
    needed. Registering a new account still verifies: that is a launch's
    precondition being set up, not a deletion.
    """
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
        return _select_account(executable, input_fn, run_fn, config_file, verify)
    if not 0 <= index < len(names):
        raise RuntimeError(t("cli.kaggle.accounts.invalid"))
    return _use_account(executable, names[index], run_fn, config_file, verify)


def _use_account(executable, username, run_fn=subprocess.run, config_file=None,
                 verify=True):
    """Check the account really answers as itself, then record it as selected."""
    environment = _account_environment(username, config_file)
    if verify:
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
    """Benchmark-backed candidates that fit this exact Kaggle accelerator.

    Whether a model fits a card is one question with one answer, so it is
    asked of `model_discovery.fit_report`, the same function every local
    screen asks. This used to compute the cache and call `hardware.fits`
    itself, which meant a correction to the fit arithmetic would have reached
    the screens that choose a model to run here and not the one that chooses a
    model to run on a borrowed GPU. Only the context differs, and it is passed.
    """
    import model_discovery

    accelerator = ACCELERATORS[machine_shape]
    selected = []
    for candidate in _load_model_candidates(catalog_path):
        report = model_discovery.fit_report(
            candidate, accelerator["vram_mb"],
            overhead_mb=accelerator["overhead_mb"], context=MODEL_CONTEXT,
        )
        if report["fits"]:
            model = dict(candidate)
            model.update({
                "machine_shape": machine_shape,
                "machine_label": accelerator["label"],
                "cuda_arch": accelerator["cuda_arch"],
                "gpu_count": accelerator["gpu_count"],
                "kv_bytes": report["kv_bytes"],
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


def accelerator_machine(machine_shape):
    """The hardware a Kaggle row is drawn against: the kernel's card, not ours.

    Quoting the throughput of the GTX 1650 under this desk on a screen that
    chooses what will run on a borrowed T4 is worse than quoting nothing, so
    the card is passed in explicitly and comes from the accelerator the row was
    assigned to.
    """
    import model_discovery

    accelerator = ACCELERATORS[machine_shape]
    return model_discovery.machine(
        vram_mb=accelerator["vram_mb"], gpu_count=accelerator["gpu_count"],
        bandwidth_gbs=accelerator["bandwidth_gbs"],
        name=accelerator["column"],
    )


def model_rows(models, fit=None):
    """The cells for each candidate, each against the card it was assigned.

    One line per model, because a selection screen draws one line each: three
    printed lines per model turned six candidates into a wall the user could
    not read. The evidence behind the chosen row is not dropped, it is printed
    once for the row that was actually chosen.

    `fit` is given by a screen whose rows all share one accelerator: there the
    card heads the column, so the cell answers whether the model fits it rather
    than naming the same card on every line.
    """
    import model_discovery

    return [
        model_discovery.model_row(
            dict(model, name=model_discovery.resolved_row_name(model)),
            accelerator_machine(model["machine_shape"]), translate=t,
            fit=fit or ACCELERATORS[model["machine_shape"]]["column"],
            # Everything on these screens was selected for fitting the
            # accelerator it was assigned, which is what makes the estimate
            # about a model that card actually holds.
            fits=True,
            # Nothing here is installed anywhere, so the last column answers
            # the question this screen does raise: what stands behind the row.
            state=model_discovery.origin_label(model, t),
        )
        for model in models
    ]


def model_table(models):
    """Header and rows for a catalogue whose rows do not share one card.

    Every model here is assigned to the smallest accelerator that holds it, so
    the card is a property of the row and heads no column: it becomes the cell
    under GPU, and the legend says once where the throughput came from.
    """
    import model_discovery

    return model_discovery.model_table(
        model_rows(models), translate=t,
        state_header=t("model.table.origin"),
        fit_header=t("cli.kaggle.models.gpu_header"),
        legend=t("model.table.legend.per_accelerator"),
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
    table = model_table(models)
    index = _choose(
        "\n\n".join((t("cli.kaggle.models.section"), t("cli.kaggle.models.title"),
                     table["header"], table["legend"])),
        table["rows"], input_fn)
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
    "__CONTEXT__": re.compile(r"[0-9]+"),
    "__SESSION_SECONDS__": re.compile(r"[0-9]+"),
    "__IDLE_SECONDS__": re.compile(r"[0-9]+"),
}

# 16384 is what every launch used to get, and it is a floor rather than a
# measurement: a model only enters the list at all if it fits at that size. A
# real task can exceed it without the model being at fault. Reading this
# project's own documentation set and then answering asked for 18075 tokens on
# 2026-08-21 and died against the ceiling, having already paid the whole load.
# So when the cards still have room after the weights, the ceiling rises to the
# largest rung that fits, decided by the same fit function that chose the model.
MODEL_CONTEXT_LADDER = (16384, 24576, 32768, 49152, 65536, 98304, 131072)
# The cache used to be allowed only half of what was free after the weights,
# and that half was never a measurement. It was compensating, without saying
# so, for a VRAM figure that was 2048 MiB larger than the cards: with the real
# figure in `ACCELERATORS`, the 1536 MiB reserve is a reserve again and the
# halving has nothing left to compensate for. The two launches that bound this
# were both measured on borrowed cards, and they are the reason the reserve is
# not smaller and not larger:
#
#   2026-08-21, refused now, died then. The MoE at Q6_K was handed 65536
#   tokens: 23930 MiB of weights and 6400 MiB of cache against 30720 MiB of
#   card, which leaves 390 MiB for everything llama.cpp needs on top. It died
#   with `cudaMalloc failed` on device 0. With the real figure and this
#   reserve the same request is refused before it costs anything.
#
#   2026-08-22, allowed now, served then. The dense at UD-Q6_K_L was handed
#   24576 tokens, which this arithmetic scores at 751 MiB free per card. It
#   loaded and answered, and nvidia-smi inside the session read 11657 MiB on
#   one card and 12985 MiB on the other, 2375 MiB still free on the fuller one.
#
# So a launch predicted to leave 195 MiB per card dies and one predicted to
# leave 751 MiB per card serves, and 768 MiB per card sits between them, which
# is what `hardware.DEFAULT_OVERHEAD_MB` already said.
#
# One thing this arithmetic gets wrong is written down rather than corrected,
# because correcting it would be guessing. For that dense launch it predicted
# 29217 MiB of weights plus cache and the pair really held 24642 MiB, an
# overshoot of 4575 MiB. The direction is safe, it refuses more than it must,
# and the cost is real: 24576 tokens is refused by 33 MiB on a model measured
# serving at exactly that size. Whatever explains the overshoot is in
# llama.cpp's own buffer sizes, which Kaggle's log keeps a few dozen records of
# and drops.


def _context_ceiling(model):
    """The largest context this model can be given on that accelerator.

    A ceiling, not a decision: what the borrowed memory allows, which the user
    then chooses inside. Never below the floor every launch used to get,
    because a model only reaches the list by fitting there, and never so high
    that the cache eats the headroom the server needs to run.
    """
    accelerator = ACCELERATORS.get(model.get("machine_shape"))
    required = ("n_layers", "n_kv_heads", "head_dim", "model_bytes")
    if not accelerator or any(model.get(key) is None for key in required):
        return MODEL_CONTEXT
    # Per card, because that is where running out of memory happens. A pair
    # with room to spare is no comfort to the card doing the allocating: the
    # 2026-08-21 death was `cudaMalloc failed` on device 0 while the aggregate
    # still looked fine. Weights and cache are split across the cards by
    # `--split-mode`, and the split is not exactly even, which the reserve
    # absorbs: the one launch measured card by card ended 1328 MiB heavier on
    # one card than the other.
    count = max(1, accelerator.get("gpu_count", 1))
    usable_per_card = max(
        0, accelerator["vram_mb"] - accelerator["overhead_mb"]) / count
    weights_per_card = model["model_bytes"] / count
    budget_per_card = usable_per_card * 1024 * 1024 - weights_per_card
    best = MODEL_CONTEXT
    for context in MODEL_CONTEXT_LADDER:
        kv_per_card = hardware.kv_cache_bytes(
            model["n_layers"], model["n_kv_heads"], model["head_dim"],
            context) / count
        if kv_per_card <= budget_per_card:
            best = max(best, context)
        else:
            break
    debug.note("cli_kaggle._context_ceiling",
               f"{model.get('alias')} allows up to {best} tokens, "
               f"{units.gib(budget_per_card)} GiB free for cache on each "
               f"of {count} card(s)")
    return best


def _kernel_context(model):
    """The one number the kernel and the profile both read.

    They must not diverge: a profile promising more than llama-server was
    started with fails at the end of a long turn, which is the most expensive
    moment to find out. A model carries the user's choice; the ceiling answers
    for the paths that never ask, which are the flow-validation probe and any
    record written before this screen existed.
    """
    chosen = model.get("context")
    return int(chosen) if chosen else _context_ceiling(model)


def _choose_session_ceiling(input_fn, remaining_hours=None):
    """How long this kernel may live before it ends itself, agreed at the push.

    A kernel spends GPU quota by wall clock until something deletes it, and
    every brake this program had needed somebody alive to pull it: the window
    that launched it, or a person typing `isaacli kaggle --stop`. On 2026-08-23
    the session that was conducting a run was cut off by an API limit at 23:23
    and came back at 03:31, and the kernel served nobody for four hours. What
    finally stopped it was the four-hour ceiling the push carries, which is
    this program taking the largest number it could and calling it a limit.

    So the number is chosen here instead of inherited, and it is carried twice:
    into the kernel, which watches its own clock and ends the session itself,
    and into `kernels push -t`, which is Kaggle enforcing the same figure if
    our own watch dies with the script. Neither of those needs anybody alive.
    """
    hours = [hour for hour in SESSION_CEILING_HOURS
             if hour * 3600 <= SESSION_TIMEOUT_SECONDS]
    options = [t("cli.kaggle.ceiling.option", hours=hour) for hour in hours]
    explanation = t("cli.kaggle.ceiling.explain")
    if remaining_hours is not None:
        explanation += "\n" + t("cli.kaggle.ceiling.remaining",
                                remaining=f"{remaining_hours:.2f}")
    options.append(t("navigation.back"))
    index = _choose(f"{t('cli.kaggle.ceiling.title')}\n\n{explanation}",
                    options, input_fn, initial=0)
    if index >= len(hours):
        return None
    return int(hours[index] * 3600)


def _choose_kernel_context(model, input_fn):
    """The context screen the local setup already draws, with a harder ceiling.

    On Ollama the ceiling is what the model was trained for, and asking for
    more is refused by something that costs nothing. Here it is borrowed VRAM,
    and going over it does not warn: it kills the kernel while allocating the
    cache, after the whole load has already been paid for out of a weekly
    budget. So the list offered is the filtered one, and a typed value above
    the ceiling is refused with the reason rather than accepted because the
    user asked for it.
    """
    import setup_ollama

    ceiling = _context_ceiling(model)
    levels = [(key, value) for key, value in setup_ollama.CONTEXT_LEVELS
              if value <= ceiling]
    if ceiling not in {value for _key, value in levels}:
        levels.append(("cli.kaggle.context.maximum", ceiling))
    options = [t(key, limit=setup_ollama.format_context(value))
               for key, value in levels]
    options += [t("context.manual"), t("navigation.back")]
    explanation = (t("context.explain") + "\n"
                   + t("cli.kaggle.context.limit",
                       limit=setup_ollama.format_context(ceiling),
                       machine=model.get("machine_label", "")))
    title = t("context.title")
    # The largest rung is highlighted because that is exactly what a launch was
    # given before this screen existed, so pressing Enter keeps the behaviour
    # the measurements were taken against.
    index = _choose(f"{title}\n\n{explanation}", options, input_fn,
                    initial=max(0, len(levels) - 1))
    if index == len(levels) + 1:
        return None
    if index < len(levels):
        return levels[index][1]
    while True:
        print(f"{title}\n\n{explanation}")
        value = setup_ollama.parse_context(input_fn(t("context.manual.prompt")))
        if setup_ollama.MIN_CONTEXT <= value <= ceiling:
            return value
        print(t("cli.kaggle.context.manual.invalid",
                limit=setup_ollama.format_context(ceiling)))


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
                   dataset_sources=None, session_seconds=SESSION_TIMEOUT_SECONDS):
    template_name = "flow-validation-cpu.py.tmpl" if validation_cpu else "gpu-server.py.tmpl"
    template = (TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
    values = {
        "__SESSION_SECONDS__": str(int(session_seconds)),
        "__IDLE_SECONDS__": str(int(SESSION_IDLE_SECONDS)),
        "__MODEL_REPO__": model["repo"],
        "__MODEL_FILE__": model["file"],
        "__MODEL_ALIAS__": model["alias"],
        "__API_KEY__": api_key,
        "__CUDA_ARCH__": model.get("cuda_arch", ""),
        "__MACHINE_SHAPE__": model.get("machine_shape", ""),
        "__GPU_COUNT__": str(model.get("gpu_count", 0)),
        "__SPLIT_MODE__": "layer" if _needs_every_gpu(model) else "none",
        "__CONTEXT__": str(_kernel_context(model)),
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
        needed=units.gib(needed_bytes), free=units.gib(free)))


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
        print(t("cli.kaggle.prepare.weight",
                size=units.gib(model["model_bytes"]), name=model["name"]))
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


# Lines the rendered kernel prints to name the step it is starting, so a wait
# that lasts half an hour shows what it is waiting for.
STAGE_PREFIX = "[setup]"
# Lines the rendered kernel prints with what the cards really hold and what is
# really on them. It is the one measurement of borrowed memory that exists, and
# it is diagnosis rather than work, so it goes to --debug and never to a screen
# somebody is watching for their URL.
VRAM_PREFIX = "[vram]"


def _kernel_state(executable, slug, run_fn=subprocess.run, env=None):
    """What Kaggle says about this kernel right now, as a bare word."""
    result = _run_capture(
        [str(executable), "kernels", "status", slug], run_fn, env)
    output = (result.stdout + " " + result.stderr).strip()
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
    return match.group(1) if match else ""


def discover_tunnel_url(executable, slug, timeout=SESSION_TIMEOUT_SECONDS,
                        popen_fn=subprocess.Popen, env=None,
                        run_fn=subprocess.run):
    """Wait for the kernel to publish its tunnel URL, for as long as it can.

    `kernels logs -f` returns immediately while the kernel is still queued,
    because there is nothing to follow yet. Treating the end of that stream as
    the end of the wait made a launch that had worked look like one that failed:
    measured on 2026-08-21, the push succeeded, this gave up in under a minute
    with the kernel in QUEUED, and the caller then told the user their kernel
    was spending quota and should be deleted. The deadline is a clock, not a
    process, so the stream is reopened until the clock runs out, and only a
    state Kaggle calls terminal ends the wait early.

    That clock used to be thirty minutes, calibrated against one 15.33 GiB
    weight, and it was the calibration that was wrong rather than the number:
    the same step measured 3 minutes for a 17.28 GiB weight and over 30 for a
    22.53 GiB one, so no rate derived from those points would be a measurement.
    What does bound the wait without inventing anything is the kernel's own
    life. It cannot publish a URL after the ceiling its own push carries, and
    it raises rather than hang if the server never answers, so waiting that long
    can only end in a URL or in a state Kaggle calls terminal. Twice on
    2026-08-22 the old clock ended a dense launch that was still loading, and
    left the kernel spending quota with nothing to show for it.
    """
    deadline = time.monotonic() + timeout
    announced = set()
    state = ""
    # Reading the follower through a pipe is what makes it buffer. The Kaggle
    # CLI is Python, and Python writing to a pipe rather than a terminal fills
    # a block before it flushes any of it, so the last few hundred bytes of the
    # log sit in the child and never arrive. Measured on 2026-08-22: the kernel
    # published its URL, this saw every line up to the one before it, and then
    # waited with the URL stuck in that buffer while the GPU billed. The kernel
    # goes quiet exactly when it starts serving, so nothing was ever going to
    # push that block out. Unbuffering the child is the whole fix, and it has
    # to be added to the account environment rather than replacing it, because
    # that environment is what isolates which Kaggle account this speaks for.
    stream_env = dict(os.environ if env is None else env)
    stream_env["PYTHONUNBUFFERED"] = "1"
    while time.monotonic() < deadline:
        process = popen_fn(
            [str(executable), "kernels", "logs", "-f", slug],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=stream_env,
        )
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
                stage = line.strip()
                # Reopening the stream replays what it already carried, so the
                # same stage must not be announced twice.
                if stage.startswith(STAGE_PREFIX) and stage not in announced:
                    announced.add(stage)
                    print(t("cli.kaggle.url.stage",
                            stage=stage[len(STAGE_PREFIX):].strip()))
                elif stage.startswith(VRAM_PREFIX) and stage not in announced:
                    announced.add(stage)
                    measurement = stage[len(VRAM_PREFIX):].strip()
                    # One site per moment measured, deliberately. `debug.note`
                    # reports once per site, so naming them all after this
                    # function would print the reading taken before the load
                    # and drop the one taken after it, which is the only one
                    # that says what the runtime costs.
                    moment = measurement.split(":", 1)[0].strip()
                    debug.note(f"cli_kaggle.kernel_vram[{moment}]", measurement)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
        state = _kernel_state(executable, slug, run_fn, env)
        if state in TERMINAL_STATES:
            raise RuntimeError(t("cli.kaggle.url.ended", slug=slug, state=state))
        debug.note("cli_kaggle.discover_tunnel_url",
                   f"{slug} has not published a URL yet, state {state or 'unknown'}")
        time.sleep(10)
    # Naming the end of the log stream describes a symptom that happens on every
    # queued kernel and reads as a kernel that died. The kernel is alive, Kaggle
    # says so, and it is spending quota right now: that is the cause and the
    # thing the user has to act on. The state is asked for again here rather
    # than reused from the last poll, because it is the answer being reported.
    state = _kernel_state(executable, slug, run_fn, env) or state
    raise RuntimeError(t(
        "cli.kaggle.url.expired", slug=slug, minutes=f"{timeout / 60:.0f}",
        state=state or t("cli.kaggle.url.state_unknown")))


def _settle_unfinished_kernel(executable, slug, input_fn, run_fn=subprocess.run,
                              env=None):
    """Decide what happens to a kernel whose launch never finished.

    A launch that fails after the push leaves a GPU kernel running with no URL,
    no profile and nobody watching it. Twice on 2026-08-22 that cost quota until
    it was noticed and deleted by hand, which is the opposite of the rule that
    what this program starts, this program can stop. Nothing is decided silently:
    whoever is at the terminal is asked, with the cost named. With nobody there
    the answer is to stop, because losing a launch costs one push and leaving a
    GPU kernel alone costs quota that does not come back.
    """
    page = f"https://www.kaggle.com/code/{slug}"
    try:
        state = _kernel_state(executable, slug, run_fn, env)
    except RuntimeError as error:
        # Not knowing is not the same as knowing it stopped, so say the kernel
        # may still be spending rather than deciding on its behalf.
        debug.note(f"cli_kaggle._settle_unfinished_kernel {slug}", error)
        print(t("cli.kaggle.stop_spending", url=page))
        return
    if state in TERMINAL_STATES:
        debug.note(f"cli_kaggle._settle_unfinished_kernel {slug}",
                   f"already {state}, nothing left to stop")
        return
    if terminal_ui.interactive(input_fn):
        try:
            index = _choose(
                t("cli.kaggle.unfinished.title", slug=slug, state=state),
                [t("cli.kaggle.unfinished.stop"), t("cli.kaggle.unfinished.keep")],
                input_fn)
        except (KeyboardInterrupt, EOFError):
            # This screen runs inside run_kaggle's error handler, so an
            # interrupt raised here escapes past the KeyboardInterrupt branch
            # that stops the kernel when a launch is cancelled: measured, it
            # left a traceback on screen and a GPU kernel spending with no
            # delete ever issued, which is the exact hole this function exists
            # to close. Interrupting the question is not an answer to it, so it
            # falls back to the answer given when there is nobody to ask.
            print()
            index = 0
        if index != 0:
            print(t("cli.kaggle.stop_spending", url=page))
            return
    else:
        print(t("cli.kaggle.unfinished.stopping", slug=slug, state=state))
    try:
        stop_kernel(executable, slug, run_fn, env)
    except (OSError, RuntimeError) as error:
        print(t("cli.kaggle.session.stop_failed", slug=slug, error=error))
        print(t("cli.kaggle.stop_spending", url=page))
    else:
        print(t("cli.kaggle.stop.stopped", slug=slug))


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
        # The same number the kernel was started with, read from the same
        # place, so the two cannot drift apart.
        "num_ctx": _kernel_context(model),
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


def _signed_out(error):
    """Whether the Kaggle CLI answered that nothing authenticated it at all."""
    return AUTH_HELP_MARKER in str(error)


def _report_brake_lost(error, config_file=None):
    """Say the brake is down, why, and where the quota can still be stopped.

    This is the message that cost 4.00 h of GPU against 2 h authorised on
    2026-08-23: `isaacli kaggle --stop` answered with the Kaggle CLI's own
    sign-in help, which reads as "your credential is missing" about a
    credential that is sitting right there in the account folder. It is not
    missing. Read from kagglesdk 2.2.4: an account registered through the
    browser holds only `credentials.json`, whose access token lives twelve
    hours and is renewed over the network from a refresh token, and when that
    renewal is refused the CLI deletes the file and prints this help. The real
    home also holds `~/.kaggle/access_token`, which is tried first and never
    expires, which is exactly why the same command answered there.

    Whatever the cause, the person is now holding a kernel that bills by wall
    clock with no working brake inside the program, so the last thing this
    prints is every Kaggle page it knows how to reach by hand.
    """
    if _signed_out(error):
        print(t("cli.kaggle.stop.signed_out"))
    else:
        print(t("cli.kaggle.failed", error=error))
    for record in (config.load(config_file).get("kaggle") or {}).get("kernels") or []:
        if record.get("slug"):
            print(t("cli.kaggle.stop_spending",
                    url=f"https://www.kaggle.com/code/{record['slug']}"))


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
        # Verification is a guard against spending under the wrong name, and
        # this path spends nothing: it deletes. Gating the brake on it put the
        # only way to stop the quota behind the very quota call that had just
        # failed. What replaces it is stronger for this one purpose anyway,
        # because `kernels list --mine` is answered by the server and every
        # slug it returns is prefixed with the owner Kaggle attributes it to.
        account, environment = _select_account(
            executable, input_fn, run_fn, config_file, verify=False)
        live = live_kernels(executable, run_fn, environment)
        # Nothing was verified above, so the name used to decide which local
        # records belong to this run comes from the server rather than from
        # `config view`, which only reads the local configuration back.
        username = _server_owner(executable, run_fn, environment) or account
    except RuntimeError as error:
        _report_brake_lost(error, config_file)
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
                size=units.gib(model["model_bytes"])))
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
    except OSError as error:
        # A filesystem with no working locks is worse served by refusing to
        # close the session than by the race this was guarding against. But a
        # layer that disappears in silence is worse than no layer, because
        # nobody goes looking for it afterwards, so it says so on the way past.
        debug.note("cli_kaggle._kernel_lock flock", error)
        print(t("cli.kaggle.session.no_lock", path=str(path)))
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


def _being_ended(record):
    """Whether a live process has already claimed the ending of this record.

    The claim is the pid that made it, not a flag, because a window killed
    between the claim and the delete would otherwise leave a record no window
    can ever adopt again, with the kernel behind it still spending.
    """
    claim = record.get("ending")
    return bool(claim) and _holder_alive(claim)


def _update_holders(profile_name, config_file, change, adopting=False,
                    after=None):
    """Apply `change` to one record's holder list under the cross-process lock."""
    handle = _kernel_lock(config_file)
    try:
        data = config.load(config_file)
        for record in reversed((data.get("kaggle") or {}).get("kernels") or []):
            if record.get("profile") != profile_name or not record.get("slug"):
                continue
            if adopting and _being_ended(record):
                # The last window out has already stepped away from this record
                # and the delete for it is on the wire. Joining now would be
                # holding a kernel that is being ended.
                debug.note("cli_kaggle._update_holders",
                           f"{record['slug']} is being ended, so it is not adopted")
                return None, []
            holders = [
                number for number in record.get("holders") or []
                if _holder_alive(number)
            ]
            record["holders"] = change(holders)
            if after is not None:
                after(record)
            config.save(data, config_file)
            return record["slug"], record["holders"]
        return None, []
    finally:
        handle.close()


def claim_session_end(profile_name, config_file=None, pid=None):
    """Step out of the record and, when nobody is left, claim its ending.

    Deciding that nobody holds the record and deleting the kernel afterwards are
    two operations, and between them the lock is open. A window opening in that
    gap read a record with an empty holder list, added itself, and went on
    talking to a kernel whose delete was already on its way. The emptiness and
    the claim are one locked write, so the other window either arrives first and
    is seen, or arrives second and finds the record already spoken for.
    """
    pid = os.getpid() if pid is None else int(pid)

    def claim(record):
        if not record["holders"]:
            record["ending"] = pid

    return _update_holders(
        profile_name, config_file,
        lambda holders: [number for number in holders if number != pid],
        after=claim)


def drop_session_claim(profile_name, config_file=None):
    """Give the record back when the ending did not happen after all."""
    _update_holders(profile_name, config_file, lambda holders: holders,
                    after=lambda record: record.pop("ending", None))


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
        lambda holders: [*[n for n in holders if n != pid], pid],
        adopting=True)
    return slug


def release_profile_session(profile_name, config_file=None, pid=None):
    """Drop this window from the record, and answer who is still holding it."""
    pid = os.getpid() if pid is None else int(pid)
    slug, holders = _update_holders(
        profile_name, config_file,
        lambda holders: [n for n in holders if n != pid])
    return slug, holders


_heartbeats = {}


def _kaggle_profile_name(config_file=None):
    """The profile a fresh launch just made the default, asked rather than kept.

    `run_kaggle` names the profile it saved on the screen, not to its caller,
    and the caller in cli.py rereads the configuration for exactly this reason.
    Reading it here keeps that one fact in one place instead of two.
    """
    name, _item = config.profile(config.load(config_file))
    return name


def start_session_heartbeat(profile_name, config_file=None,
                            urlopen_fn=urllib.request.urlopen,
                            interval=HEARTBEAT_SECONDS):
    """Keep telling the kernel this window is still here, until it is not.

    The wall ceiling caps what a launch can ever cost. It does not shorten the
    case that actually happened on 2026-08-23, where the window conducting a
    run was cut off while the kernel was serving and the kernel billed for four
    hours for nobody: the ceiling was the only thing that ever stopped it, and
    a ceiling is by definition the worst case rather than the right one.

    A dead-man's switch is the standard answer, and the standard shape of it is
    this: the side that must not outlive the other keeps sending a sign of life,
    and stopping is the signal. Everything the user asked to be covered is the
    same event from the kernel's side, which is what makes this one mechanism
    instead of four: the terminal closed, the terminal killed, the machine's
    network gone, Ctrl+C. In every one of them this process stops beating.

    It is a daemon thread on purpose, and that is not an implementation detail:
    a daemon thread cannot outlive the interpreter, so there is no way for this
    program to exit while still claiming to be alive. It also beats on a timer
    rather than on inference traffic, which is what keeps a user who is reading
    or thinking for ten minutes from being read as gone.
    """
    profile = (config.load(config_file).get("profiles") or {}).get(profile_name)
    if not profile or not profile.get("base_url"):
        debug.note("cli_kaggle.start_session_heartbeat",
                   f"{profile_name} names no endpoint to beat against")
        return None
    running = _heartbeats.get(profile_name)
    if running is not None and running.is_alive():
        return running
    stop = threading.Event()
    secret_path = _secret_path(config_file)

    def beat():
        # `wait` returns True the moment the session ends, so closing the
        # program never waits out a sleep before the process can go.
        while not stop.wait(interval):
            try:
                answered = _endpoint_answers(profile, secret_path, urlopen_fn)
            except Exception as error:
                # A heartbeat that can raise is a heartbeat that can take the
                # session down with it. Nothing here is worth that, and the
                # kernel already treats silence as the answer.
                debug.note("cli_kaggle.start_session_heartbeat beat", error)
                continue
            if not answered:
                debug.note("cli_kaggle.start_session_heartbeat beat",
                           f"{profile_name} did not answer this beat")

    thread = threading.Thread(
        target=beat, name=f"isaacli-heartbeat-{profile_name}", daemon=True)
    thread.stop_beating = stop
    _heartbeats[profile_name] = thread
    thread.start()
    return thread


def stop_session_heartbeat(profile_name):
    """Stop beating for a session that is over, so the kernel can go quietly."""
    thread = _heartbeats.pop(profile_name, None)
    if thread is not None:
        thread.stop_beating.set()
    return thread


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
        if hold_profile_session(profile_name, config_file, pid) is None:
            # The endpoint answered, and between that answer and this claim the
            # window that owns the record started ending it. Reporting it as
            # live would hand this session a kernel that is going away.
            debug.note("cli_kaggle.ensure_profile_session",
                       f"{record['slug']} is being ended by the window that "
                       "holds it, so it was not reused")
            return None
        debug.note("cli_kaggle.ensure_profile_session",
                   f"{record['slug']} is still serving, nothing was pushed")
        start_session_heartbeat(profile_name, config_file)
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
    if code != 0:
        return "failed"
    # A kernel this program just pushed is the one most exposed to the client
    # dying: nobody has used it yet, so nothing else would ever make a request
    # against it, and its silence would read as abandonment within minutes.
    start_session_heartbeat(_kaggle_profile_name(config_file), config_file)
    return "relaunched"


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
    # Beating for a session this window is leaving would be this program lying
    # about being there, and the lie would keep somebody else's kernel alive.
    stop_session_heartbeat(profile_name)
    record = profile_kernel_record(profile_name, config_file)
    if record is None:
        return None
    _slug, holders = claim_session_end(profile_name, config_file, pid)
    if holders:
        print(t("cli.kaggle.session.still_used",
                slug=record["slug"], count=len(holders)))
        return None
    executable = _existing_executable(which_fn, home_dir)
    if executable is None:
        # Nothing is going to be deleted, so the record has to go back to being
        # adoptable instead of staying spoken for by an ending that never ran.
        drop_session_claim(profile_name, config_file)
        print(t("cli.kaggle.session.no_cli", slug=record["slug"]))
        return None
    print(t("cli.kaggle.session.stopping", slug=record["slug"]))
    try:
        environment = _record_environment(record, config_file)
        stop_kernel(executable, record["slug"], run_fn, environment)
    except RuntimeError as error:
        # Saying nothing here would leave quota draining behind a screen that
        # already said goodbye.
        drop_session_claim(profile_name, config_file)
        # The Kaggle sign-in help is a thousand characters of instructions for
        # a credential that is present, and pasting it under "could not end the
        # session" hides the one thing that matters here: the kernel is still
        # billing and the program can no longer stop it.
        print(t("cli.kaggle.session.stop_failed", slug=record["slug"],
                error=t("cli.kaggle.stop.signed_out") if _signed_out(error)
                else error))
        print(t("cli.kaggle.stop_spending",
                url=f"https://www.kaggle.com/code/{record['slug']}"))
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
            # The context chosen that time travels inside the model, so
            # repeating a choice does not ask about it again either.
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
            context = _choose_kernel_context(model, input_fn)
            if context is None:
                print(t("cli.kaggle.cancelled"))
                return 130
            model = dict(model, context=context)
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
                        size=units.gib(model["model_bytes"])))
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
    # Asked before the push confirmation, because the ceiling is part of what is
    # being consented to: how many hours of a weekly budget this may cost at
    # most, whether or not anybody is left watching it.
    session_seconds = SESSION_TIMEOUT_SECONDS
    if not validation_cpu:
        session_seconds = _choose_session_ceiling(
            input_fn, _quota_remaining_hours(quota))
        if session_seconds is None:
            print(t("cli.kaggle.cancelled"))
            return 130
        print(t("cli.kaggle.ceiling.chosen", hours=session_seconds // 3600))
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
                dataset_sources=dataset_sources,
                session_seconds=session_seconds)
            # The same figure on both sides. The kernel watches its own clock,
            # which is the brake that needs nobody alive; `-t` is Kaggle holding
            # to the same agreement if the script dies before its watch fires.
            result = run_fn([str(executable), "kernels", "push", "-p", temporary,
                             "-t", str(session_seconds)], check=False,
                            env=environment)
            if result.returncode != 0:
                raise RuntimeError(t("cli.kaggle.push.failed"))
        print(t("cli.kaggle.pushed", slug=slug,
                url=f"https://www.kaggle.com/code/{slug}"))
        # Waiting is bounded by the life of the kernel, and that life is now the
        # number chosen above rather than the largest one this program can ask
        # for. Leaving the default here would have this window watching for four
        # hours for a URL from a kernel that ends itself after one.
        url = discover_tunnel_url(
            executable, slug, timeout=session_seconds,
            popen_fn=popen_fn, env=environment)
        profile = save_kaggle_profile(
            url, slug, model, api_key, config_file, account=account)
    except KeyboardInterrupt:
        # Ctrl+C means the launch is cancelled, not that the already-pushed
        # kernel should keep spending quota without a profile or owner in the
        # local lifecycle. The user explicitly approved this exact kernel push,
        # so cancelling its unfinished setup also ends that same kernel.
        print()
        try:
            stop_kernel(executable, slug, run_fn, environment)
        except (OSError, RuntimeError) as error:
            print(t("cli.kaggle.session.stop_failed", slug=slug, error=error))
            print(t("cli.kaggle.stop_spending",
                    url=f"https://www.kaggle.com/code/{slug}"))
        else:
            print(t("cli.kaggle.stop.stopped", slug=slug))
        print(t("cli.kaggle.cancelled"))
        return 130
    except (OSError, RuntimeError) as error:
        # Giving up here does not stop anything on its own: the kernel was
        # already pushed and keeps spending quota that does not come back.
        print(t("cli.kaggle.failed", error=error))
        _settle_unfinished_kernel(executable, slug, input_fn, run_fn, environment)
        return 1
    print(t("cli.kaggle.ready", profile=profile, url=url + "/v1"))
    print(t("cli.kaggle.stop", url=f"https://www.kaggle.com/code/{slug}"))
    return 0
