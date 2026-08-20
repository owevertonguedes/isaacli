"""Kaggle CLI installation and explicit remote-kernel orchestration."""
import csv
import getpass
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


def _account_environment(username, config_file=None):
    """Materialize one selected credential for the Kaggle CLI only."""
    data = config.load(config_file)
    account = ((data.get("kaggle") or {}).get("accounts") or {}).get(username)
    if not account:
        raise RuntimeError(t("cli.kaggle.accounts.missing", username=username))
    raw = config.load_secret(account.get("credential"), _secret_path(config_file))
    try:
        credential = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(t("cli.kaggle.accounts.credential_invalid")) from error
    base = (Path(config_file).parent if config_file else config.config_path().parent)
    digest = hashlib.sha256(username.encode("utf-8")).hexdigest()[:16]
    account_dir = base / "kaggle-accounts" / digest
    account_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.pop("KAGGLE_USERNAME", None)
    environment.pop("KAGGLE_KEY", None)
    environment["KAGGLE_CONFIG_DIR"] = str(account_dir)
    if credential.get("token"):
        token_path = account_dir / "access_token"
        token_path.write_text(str(credential["token"]).strip() + "\n", encoding="utf-8")
        token_path.chmod(0o600)
        environment["KAGGLE_API_TOKEN"] = str(token_path)
        config.save({"username": username}, account_dir / "kaggle.json")
    else:
        config.save({"username": username, "key": credential["key"]},
                    account_dir / "kaggle.json")
        empty_token = account_dir / "no-access-token"
        empty_token.write_text("", encoding="utf-8")
        empty_token.chmod(0o600)
        environment["KAGGLE_API_TOKEN"] = str(empty_token)
    return environment


def _register_account_interactive(input_fn, config_file=None):
    username = input_fn(t("cli.kaggle.accounts.username_prompt")).strip()
    credential_input = getpass.getpass if input_fn is input else input_fn
    value = credential_input(t("cli.kaggle.accounts.credential_prompt")).strip()
    credential = {"token": value} if value.startswith("KGAT_") else {"key": value}
    return register_account(username, credential, config_file)


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


def _authenticate(executable, run_fn=subprocess.run, env=None):
    try:
        return _quota(executable, run_fn, env)
    except RuntimeError as error:
        debug.swallowed("cli_kaggle._authenticate quota before login")
        print(t("cli.kaggle.login.required", error=error))
    result = run_fn([str(executable), "auth", "login"], check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(t("cli.kaggle.login.failed"))
    return _quota(executable, run_fn, env)


def _select_account(executable, input_fn, run_fn=subprocess.run, config_file=None):
    """List account quotas and return the manually selected account and env."""
    data = config.load(config_file)
    accounts = ((data.get("kaggle") or {}).get("accounts") or {})
    if not accounts:
        print(t("cli.kaggle.accounts.none"))
        _register_account_interactive(input_fn, config_file)
        data = config.load(config_file)
        accounts = ((data.get("kaggle") or {}).get("accounts") or {})
    names = list(accounts)
    print(t("cli.kaggle.accounts.title"))
    for index, username in enumerate(names, 1):
        environment = _account_environment(username, config_file)
        try:
            quota = _quota(executable, run_fn, environment).replace("\n", " | ")
        except RuntimeError as error:
            quota = t("cli.kaggle.accounts.quota_unavailable", error=error)
        print(t("cli.kaggle.accounts.option", index=index, username=username,
                quota=quota))
    print(t("cli.kaggle.accounts.add_option", index=len(names) + 1))
    selected = ((data.get("kaggle") or {}).get("selected_account"))
    default = names.index(selected) + 1 if selected in names else 1
    answer = input_fn(t("cli.kaggle.accounts.prompt", default=default)).strip()
    try:
        index = int(answer or default) - 1
    except ValueError as error:
        raise RuntimeError(t("cli.kaggle.accounts.invalid")) from error
    if index == len(names):
        username = _register_account_interactive(input_fn, config_file)
    elif 0 <= index < len(names):
        username = names[index]
    else:
        raise RuntimeError(t("cli.kaggle.accounts.invalid"))
    data = config.load(config_file)
    data.setdefault("kaggle", {})["selected_account"] = username
    config.save(data, config_file)
    return username, _account_environment(username, config_file)


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
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        output = (result.stdout + " " + result.stderr).upper()
        state = next((name for name in TERMINAL_STATES if name in output), None)
        if state is None:
            live.append((ref, (result.stdout or result.stderr).strip()))
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
    return candidates


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


def _select_model(input_fn, catalog_path=MODEL_CATALOG_PATH):
    models = prepared_models(catalog_path)
    if not models:
        raise RuntimeError(t("cli.kaggle.models.none"))
    print(t("cli.kaggle.models.title"))
    for index, model in enumerate(models, 1):
        size_gib = model["model_bytes"] / 1024 ** 3
        print(t(
            "cli.kaggle.models.option", index=index, name=model["name"],
            size=f"{size_gib:.1f}", machine=model["machine_label"],
            benchmark=model["benchmark"],
        ))
        print(f"     {model['source']}")
        print(f"     {model['benchmark_source']}")
    answer = input_fn(t("cli.kaggle.models.prompt")).strip()
    try:
        return models[int(answer) - 1]
    except (ValueError, IndexError):
        raise RuntimeError(t("cli.kaggle.models.invalid"))


def _asset_refs(username, model):
    binary_slug = BINARY_DATASET_SLUGS.get(
        model.get("cuda_arch"),
        f"isaacli-llama-cuda-sm{model.get('cuda_arch', 'unknown')}-b10502",
    )
    model_slug = MODEL_DATASET_SLUGS.get(
        model.get("alias"),
        "isaacli-model-" + re.sub(r"[^a-z0-9-]+", "-", model["alias"].lower()),
    )
    return {
        "binary": f"{username}/{binary_slug}",
        "model": f"{username}/{model_slug}"[:50 + len(username) + 1],
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
    if not validation_cpu:
        metadata["machine_shape"] = model["machine_shape"]
        metadata["dataset_sources"] = list(dataset_sources or [])
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )


def _render_preparation_kernel(folder, slug, cuda_arch):
    template = (TEMPLATE_DIR / "prepare-assets-cpu.py.tmpl").read_text(
        encoding="utf-8")
    template = template.replace("__CUDA_ARCH__", cuda_arch)
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


def _prepare_assets(executable, username, model, available, input_fn,
                    run_fn=subprocess.run, env=None):
    expected = _asset_refs(username, model)
    if "binary" not in available:
        suffix = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        slug = f"{username}/isaacli-prepare-cpu-{suffix}"
        print(t("cli.kaggle.prepare.cpu", slug=slug))
        with tempfile.TemporaryDirectory(prefix="isaacli-kaggle-prepare-") as temporary:
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
            shutil.copy2(archives[0], dataset / archives[0].name)
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
            with tempfile.TemporaryDirectory(
                    prefix="isaacli-kaggle-weight-") as temporary:
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
    kernels = state.setdefault("kernels", [])
    kernels.append({
        "slug": slug, "url": url,
        "web_url": f"https://www.kaggle.com/code/{slug}",
        "profile": profile_name, "model": model["alias"], "account": account,
    })
    config.save(data, config_file)
    secret_path = Path(config_file).with_name("secrets.json") if config_file else None
    config.save_secret(credential, api_key, secret_path)
    return profile_name


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
        key = config.load_secret(profile.get("credential"), secret_path)
        request = urllib.request.Request(
            profile["base_url"].rstrip("/") + "/models",
            headers={"Authorization": "Bearer " + key} if key else {},
        )
        try:
            with urlopen_fn(request, timeout=10) as answer:
                if answer.status != 200:
                    debug.note(
                        "cli_kaggle._reactivate_live_profile status",
                        f"saved endpoint returned HTTP {answer.status}",
                    )
                    continue
        except (urllib.error.URLError, OSError, TimeoutError):
            debug.swallowed("cli_kaggle._reactivate_live_profile probe")
            continue
        data["default_profile"] = profile_name
        config.save(data, config_file)
        return profile_name, record
    return None, None


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
    try:
        account, environment = _select_account(
            executable, input_fn, run_fn, config_file)
        quota = _quota(executable, run_fn, environment)
        print(t("cli.kaggle.quota", quota=quota))
        username_result = _run_capture(
            [str(executable), "config", "view"], run_fn, environment)
        username_match = re.search(
            r"(?:username:\s*|- username:\s*)([^\s]+)",
            username_result.stdout, re.I)
        if username_result.returncode != 0 or not username_match:
            raise RuntimeError(t("cli.kaggle.username.failed"))
        username = username_match.group(1)
        live = live_kernels(executable, run_fn, environment)
    except RuntimeError as error:
        print(t("cli.kaggle.failed", error=error))
        return 1
    if live:
        for ref, state in live:
            print(t("cli.kaggle.live", slug=ref, state=state,
                    url=f"https://www.kaggle.com/code/{ref}"))
        if not validation_cpu:
            profile, record = _reactivate_live_profile(
                live, config_file, account=account, urlopen_fn=urlopen_fn,
            )
            if profile:
                print(t("cli.kaggle.reused", profile=profile,
                        url=record["url"] + "/v1"))
                return 0
            saved_refs = {
                item.get("slug") for item in
                (config.load(config_file).get("kaggle") or {}).get("kernels", [])
                if item.get("account") in (None, account)
            }
            if any(ref in saved_refs for ref, _state in live):
                print(t("cli.kaggle.unresponsive"))
                return 1
        print(t("cli.kaggle.second_refused"))
        return 1
    if not validation_cpu:
        saved = [
            item for item in
            (config.load(config_file).get("kaggle") or {}).get("kernels", [])
            if item.get("account") in (None, account)
        ]
        if saved:
            print(t("cli.kaggle.dead", slug=saved[-1].get("slug", "?")))
    if validation_cpu:
        print(t("cli.kaggle.cpu_only"))
        model = {"repo": "", "file": "", "alias": "isaacli-flow-probe"}
    else:
        try:
            model = _select_model(input_fn)
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
            print(t("cli.kaggle.assets.missing", username=username))
            print(t("cli.kaggle.assets.measured_cost"))
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
