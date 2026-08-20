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
CUDA_BINARY_DATASETS = {
    "75": "owevertonguedes/isaacli-llama-cuda-sm75-b10502",
}
MODEL_DATASETS = {
    "qwen38-27b": "owevertonguedes/isaacli-qwen38-27b-ud-q4-k-m",
}


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


def _load_model_candidates(path=MODEL_CATALOG_PATH):
    """Load benchmark-backed candidates before hardware fit is applied."""
    required = {
        "name", "repo", "file", "alias", "source", "model_bytes",
        "n_layers", "n_kv_heads", "head_dim", "benchmark",
        "benchmark_source", "active_ratio",
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
    return sorted(selected, key=lambda item: (item["active_ratio"], item["model_bytes"]))


def prepared_models(catalog_path=MODEL_CATALOG_PATH):
    """Models whose exact weight and architecture binary are reusable inputs."""
    return [
        model for model in recommended_models(catalog_path)
        if model["alias"] in MODEL_DATASETS
        and model["cuda_arch"] in CUDA_BINARY_DATASETS
    ]


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


def _render_kernel(folder, slug, model, api_key, validation_cpu=False):
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
        try:
            metadata["dataset_sources"] = [
                CUDA_BINARY_DATASETS[model["cuda_arch"]],
                MODEL_DATASETS[model["alias"]],
            ]
        except KeyError as error:
            raise RuntimeError(
                f"no reusable Kaggle asset is published for {error.args[0]}"
            ) from error
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )


# The reusable inputs remove the 34 minute build and download path. The same
# model took 43.5 seconds to load in the measured T4 x2 run. A CPU probe with
# the attached 15.33 GiB dataset took 11 minutes 26 seconds from push to status
# check after completion, although its script ran in one second. Thirty minutes
# covers input staging, scheduling, loading and tunnel startup.
URL_DISCOVERY_TIMEOUT = 30 * 60


def discover_tunnel_url(executable, slug, timeout=URL_DISCOVERY_TIMEOUT,
                        popen_fn=subprocess.Popen):
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
    kernels.append({
        "slug": slug, "url": url,
        "web_url": f"https://www.kaggle.com/code/{slug}",
        "profile": profile_name, "model": model["alias"],
    })
    config.save(data, config_file)
    secret_path = Path(config_file).with_name("secrets.json") if config_file else None
    config.save_secret(credential, api_key, secret_path)
    return profile_name


def _reactivate_live_profile(live, config_file=None,
                             urlopen_fn=urllib.request.urlopen):
    """Reactivate a saved profile only when its recorded kernel and API answer."""
    data = config.load(config_file)
    live_refs = {ref for ref, _state in live}
    secret_path = Path(config_file).with_name("secrets.json") if config_file else None
    for record in reversed((data.get("kaggle") or {}).get("kernels") or []):
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
        if not validation_cpu:
            profile, record = _reactivate_live_profile(
                live, config_file, urlopen_fn=urlopen_fn,
            )
            if profile:
                print(t("cli.kaggle.reused", profile=profile,
                        url=record["url"] + "/v1"))
                return 0
            saved_refs = {
                item.get("slug") for item in
                (config.load(config_file).get("kaggle") or {}).get("kernels", [])
            }
            if any(ref in saved_refs for ref, _state in live):
                print(t("cli.kaggle.unresponsive"))
                return 1
        print(t("cli.kaggle.second_refused"))
        return 1
    if not validation_cpu:
        saved = (config.load(config_file).get("kaggle") or {}).get("kernels", [])
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
            result = run_fn([str(executable), "kernels", "push", "-p", temporary,
                             "-t", str(SESSION_TIMEOUT_SECONDS)], check=False)
            if result.returncode != 0:
                raise RuntimeError(t("cli.kaggle.push.failed"))
        print(t("cli.kaggle.pushed", slug=slug,
                url=f"https://www.kaggle.com/code/{slug}"))
        url = discover_tunnel_url(executable, slug, popen_fn=popen_fn)
        profile = save_kaggle_profile(url, slug, model, api_key, config_file)
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
