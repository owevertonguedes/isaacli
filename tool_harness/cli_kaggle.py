"""Kaggle CLI installation and explicit remote-kernel orchestration."""
import csv
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
from pathlib import Path

import config
import debug
from cli_i18n import t
from installation import _package_owns


HERE = Path(__file__).resolve().parent
TEMPLATE_DIR = HERE.parent / "contrib" / "kaggle"
TERMINAL_STATES = {"COMPLETE", "ERROR", "CANCELLED"}
URL_PATTERN = re.compile(r"TUNNEL_URL=(https://[-a-z0-9]+\.trycloudflare\.com)")
MODELS = (
    {
        "label_key": "cli.kaggle.model.qwen_coder",
        "repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "file": "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        "alias": "qwen3-coder-30b-a3b",
        "source": "https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
    },
    {
        "label_key": "cli.kaggle.model.qwen_instruct",
        "repo": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
        "file": "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
        "alias": "qwen3-30b-a3b-2507",
        "source": "https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
    },
)


def _home(home_dir=None):
    return Path(home_dir) if home_dir is not None else Path.home()


def kaggle_install_record(path=None):
    return Path(path) if path else config.config_path().with_name("kaggle-install.json")


def _write_record(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    config.save(data, path)


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


def _run_capture(command, run_fn=subprocess.run):
    return run_fn(command, check=False, capture_output=True, text=True)


def _quota(executable, run_fn=subprocess.run):
    result = _run_capture([str(executable), "quota"], run_fn)
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or result.stdout).strip() or t("cli.kaggle.quota.failed")
        )
    return result.stdout.strip()


def _authenticate(executable, run_fn=subprocess.run):
    try:
        return _quota(executable, run_fn)
    except RuntimeError as error:
        debug.swallowed("cli_kaggle._authenticate quota before login")
        print(t("cli.kaggle.login.required", error=error))
    result = run_fn([str(executable), "auth", "login"], check=False)
    if result.returncode != 0:
        raise RuntimeError(t("cli.kaggle.login.failed"))
    return _quota(executable, run_fn)


def _kernel_refs(executable, run_fn=subprocess.run):
    refs = []
    page = 1
    while True:
        result = _run_capture([
            str(executable), "kernels", "list", "--mine", "--csv",
            "--page", str(page), "--page-size", "100",
        ], run_fn)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        page_refs = [row.get("ref") or row.get("Ref") for row in rows]
        refs.extend(ref for ref in page_refs if ref)
        if len(rows) < 100:
            return refs
        page += 1


def live_kernels(executable, run_fn=subprocess.run):
    """Return every visible non-terminal kernel, querying each unique slug."""
    live = []
    for ref in _kernel_refs(executable, run_fn):
        result = _run_capture([str(executable), "kernels", "status", ref], run_fn)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        output = (result.stdout + " " + result.stderr).upper()
        state = next((name for name in TERMINAL_STATES if name in output), None)
        if state is None:
            live.append((ref, (result.stdout or result.stderr).strip()))
    return live


def _select_model(input_fn):
    print(t("cli.kaggle.models.title"))
    for index, model in enumerate(MODELS, 1):
        print(f"  {index}. {t(model['label_key'])}")
        print(f"     {model['source']}")
    answer = input_fn(t("cli.kaggle.models.prompt")).strip()
    try:
        return MODELS[int(answer) - 1]
    except (ValueError, IndexError):
        raise RuntimeError(t("cli.kaggle.models.invalid"))


def _render_kernel(folder, slug, model, api_key, validation_cpu=False):
    template_name = "flow-validation-cpu.py.tmpl" if validation_cpu else "gpu-server.py.tmpl"
    template = (TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
    values = {
        "__MODEL_REPO__": model["repo"],
        "__MODEL_FILE__": model["file"],
        "__MODEL_ALIAS__": model["alias"],
        "__API_KEY__": api_key,
    }
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
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )


def discover_tunnel_url(executable, slug, timeout=300, popen_fn=subprocess.Popen):
    process = popen_fn(
        [str(executable), "kernels", "logs", "-f", slug],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
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


def save_kaggle_profile(url, slug, model, api_key, config_file=None):
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
    kernels = state.setdefault("kernels", [])
    kernels.append({"slug": slug, "url": url, "web_url": f"https://www.kaggle.com/code/{slug}"})
    config.save(data, config_file)
    secret_path = Path(config_file).with_name("secrets.json") if config_file else None
    config.save_secret(credential, api_key, secret_path)
    return profile_name


def run_kaggle(validation_cpu=False, input_fn=None, run_fn=subprocess.run,
                popen_fn=subprocess.Popen, config_file=None, home_dir=None,
                record_path=None, which_fn=shutil.which):
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
    try:
        quota = _authenticate(executable, run_fn)
        print(t("cli.kaggle.quota", quota=quota))
        live = live_kernels(executable, run_fn)
    except RuntimeError as error:
        print(t("cli.kaggle.failed", error=error))
        return 1
    if live:
        for ref, state in live:
            print(t("cli.kaggle.live", slug=ref, state=state,
                    url=f"https://www.kaggle.com/code/{ref}"))
        print(t("cli.kaggle.second_refused"))
        return 1
    if validation_cpu:
        print(t("cli.kaggle.cpu_only"))
        model = {"repo": "", "file": "", "alias": "isaacli-flow-probe"}
    else:
        try:
            model = _select_model(input_fn)
        except RuntimeError as error:
            print(error)
            return 1
    if input_fn(t("cli.kaggle.push.confirm")).strip().lower() != t("cli.kaggle.confirm_yes"):
        print(t("cli.kaggle.cancelled"))
        return 130
    username_result = _run_capture([str(executable), "config", "view"], run_fn)
    username_match = re.search(r"username:\s*([^\s]+)", username_result.stdout, re.I)
    if username_result.returncode != 0 or not username_match:
        print(t("cli.kaggle.failed", error=t("cli.kaggle.username.failed")))
        return 1
    username = username_match.group(1)
    suffix = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    kind = "flow-cpu" if validation_cpu else "gpu"
    slug = f"{username}/isaacli-{kind}-{suffix}"
    api_key = secrets.token_urlsafe(24)
    try:
        with tempfile.TemporaryDirectory(prefix="isaacli-kaggle-") as temporary:
            _render_kernel(Path(temporary), slug, model, api_key, validation_cpu)
            result = run_fn([str(executable), "kernels", "push", "-p", temporary], check=False)
            if result.returncode != 0:
                raise RuntimeError(t("cli.kaggle.push.failed"))
        print(t("cli.kaggle.pushed", slug=slug,
                url=f"https://www.kaggle.com/code/{slug}"))
        url = discover_tunnel_url(executable, slug, popen_fn=popen_fn)
        profile = save_kaggle_profile(url, slug, model, api_key, config_file)
    except (OSError, RuntimeError) as error:
        print(t("cli.kaggle.failed", error=error))
        print(t("cli.kaggle.stop", url=f"https://www.kaggle.com/code/{slug}"))
        return 1
    print(t("cli.kaggle.ready", profile=profile, url=url + "/v1"))
    print(t("cli.kaggle.stop", url=f"https://www.kaggle.com/code/{slug}"))
    return 0
