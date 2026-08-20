#!/usr/bin/env python3
"""Kaggle lifecycle checks with isolated local state and no network."""
import io
import builtins
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))
root = Path(tempfile.mkdtemp())
os.environ["HOME"] = str(root / "home")
os.environ["XDG_CONFIG_HOME"] = str(root / "config")

import cli_kaggle
import cli
import config
import setup_ollama


failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


home = Path(os.environ["HOME"])
record = root / "managed.json"
commands = []


def install_run(command, check=False, **kwargs):
    commands.append(command)
    if command[1:3] == ["-m", "venv"]:
        env = Path(command[-1])
        (env / "bin").mkdir(parents=True)
        for name in ("python", "kaggle"):
            (env / "bin" / name).write_text("executable", encoding="utf-8")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


with redirect_stdout(io.StringIO()):
    executable = cli_kaggle.install_kaggle_cli(
        input_fn=lambda _prompt: "y", run_fn=install_run,
        which_fn=lambda _name: None, home_dir=home, record_path=record,
    )
managed = json.loads(record.read_text(encoding="utf-8"))
check(executable is not None and managed["installed_by"] == "isaacli",
      "install records that isaacli owns the isolated Kaggle CLI")
check(not any("sudo" in part for command in commands for part in command),
      "the Kaggle install path never invokes sudo")

record_without_ownership = root / "missing-record.json"
managed_link = home / ".local" / "bin" / "kaggle"
with redirect_stdout(io.StringIO()):
    refused_without_record = cli_kaggle.uninstall_managed_kaggle(
        remove_credentials=True, home_dir=home, record_path=record_without_ownership,
    )
check(refused_without_record == 1 and managed_link.is_symlink(),
      "Kaggle removal has no effect without an isaacli ownership record")

with redirect_stdout(io.StringIO()):
    refused_package = cli_kaggle.uninstall_managed_kaggle(
        remove_credentials=True, home_dir=home, record_path=record,
        package_owned_fn=lambda _path: True,
    )
check(refused_package == 1 and managed_link.is_symlink(),
      "package-manager ownership refuses removal and leaves the executable")

credential = home / ".kaggle" / "credentials.json"
credential.parent.mkdir(parents=True)
credential.write_text("private", encoding="utf-8")
with redirect_stdout(io.StringIO()):
    refused_credentials = cli_kaggle.uninstall_managed_kaggle(
        remove_credentials=False, home_dir=home, record_path=record,
        package_owned_fn=lambda _path: False,
    )
check(refused_credentials == 1 and credential.exists() and managed_link.is_symlink(),
      "authentication data without an explicit warning blocks every deletion")

with redirect_stdout(io.StringIO()):
    removed = cli_kaggle.uninstall_managed_kaggle(
        remove_credentials=True, home_dir=home, record_path=record,
        package_owned_fn=lambda _path: False,
    )
check(removed == 0 and not managed_link.exists() and not managed_link.is_symlink()
      and not credential.exists() and not record.exists(),
      "an explicitly confirmed managed removal deletes the CLI and credential")


live_commands = []


def live_run(command, check=False, capture_output=False, text=False, **kwargs):
    live_commands.append(command)
    joined = " ".join(map(str, command))
    if " quota" in joined:
        return SimpleNamespace(returncode=0, stdout="GPU quota: 30 hours", stderr="")
    if "kernels list" in joined:
        return SimpleNamespace(returncode=0, stdout="ref,title\nuser/running,Running\n", stderr="")
    if "kernels status" in joined:
        return SimpleNamespace(returncode=0, stdout="KernelWorkerStatus.RUNNING", stderr="")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


with redirect_stdout(io.StringIO()):
    second_code = cli_kaggle.run_kaggle(
        input_fn=lambda _prompt: (_ for _ in ()).throw(AssertionError("no prompt expected")),
        run_fn=live_run, which_fn=lambda _name: "/fake/kaggle",
        config_file=root / "second.json", home_dir=home,
    )
check(second_code == 1 and not any("kernels push" in " ".join(map(str, c))
                                   for c in live_commands),
      "a visible live kernel refuses a second push by effect")

# There is no way to stop a kernel through the API, so an unattended one spends
# quota until Kaggle's own maximum. Every push has to carry its own ceiling.
push_commands = []


def push_run(command, check=False, capture_output=False, text=False, **kwargs):
    push_commands.append([str(part) for part in command])
    joined = " ".join(str(part) for part in command)
    if " quota" in joined:
        return SimpleNamespace(returncode=0, stdout="GPU quota: 30 hours", stderr="")
    if "kernels list" in joined:
        return SimpleNamespace(returncode=0, stdout="ref,title\n", stderr="")
    if "config view" in joined:
        return SimpleNamespace(returncode=0, stdout="username: tester\n", stderr="")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


answers = iter(["1", "y"])
with redirect_stdout(io.StringIO()):
    cli_kaggle.run_kaggle(
        input_fn=lambda _prompt: next(answers), run_fn=push_run,
        popen_fn=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no discovery")),
        which_fn=lambda _name: "/fake/kaggle",
        config_file=root / "timeout.json", home_dir=home,
    )
pushes = [c for c in push_commands if "push" in c]
check(bool(pushes) and all(
    "-t" in c and c[c.index("-t") + 1].isdigit() and int(c[c.index("-t") + 1]) > 0
    for c in pushes),
      "every push carries a session ceiling so an unattended kernel cannot run on")

profile_file = root / "profile" / "config.json"
model = {"alias": "test-model"}
profile_name = cli_kaggle.save_kaggle_profile(
    "https://public.trycloudflare.com", "user/kernel", model, "secret", profile_file,
)
profile_data = config.load(profile_file)
profile = profile_data["profiles"][profile_name]
agent_source = (HERE.parent / "tool_harness" / "agent.py").read_text(encoding="utf-8").lower()
check(profile["provider"] == "openai_compatible"
      and profile["base_url"] == "https://public.trycloudflare.com/v1",
      "the discovered URL is saved in an openai_compatible profile")
check("kaggle" not in agent_source,
      "the request adapter remains unaware of Kaggle")
module_source = (HERE.parent / "tool_harness" / "cli_kaggle.py").read_text(encoding="utf-8")
check('"sudo"' not in module_source and "'sudo'" not in module_source,
      "no Kaggle orchestration command can invoke sudo")

# The kernel builds CUDA and downloads more than 15 GB before it can answer, and
# it only publishes the URL once the server does. A five minute wait guarantees
# the command gives up while the kernel keeps spending quota, which is exactly
# how the first real attempt failed. Measured cost from push to first token was
# 34 minutes, so the ceiling has to leave real room above that.
import inspect

discovery_default = inspect.signature(
    cli_kaggle.discover_tunnel_url).parameters["timeout"].default
check(discovery_default >= 45 * 60,
      "waiting for the tunnel outlasts a real build, download and model load")

gpu_dir = root / "gpu-render"
gpu_t4_dir = root / "gpu-t4-render"
cpu_dir = root / "cpu-render"
gpu_dir.mkdir()
gpu_t4_dir.mkdir()
cpu_dir.mkdir()
recommended = cli_kaggle.recommended_models()
p100_model = next(
    model for model in recommended
    if model["machine_shape"] == "NvidiaTeslaP100"
)
t4_model = next(
    model for model in recommended
    if model["machine_shape"] == "NvidiaTeslaT4"
)
cli_kaggle._render_kernel(gpu_dir, "user/gpu", p100_model, "key", False)
cli_kaggle._render_kernel(gpu_t4_dir, "user/gpu-t4", t4_model, "key", False)
cli_kaggle._render_kernel(
    cpu_dir, "user/cpu", {"repo": "", "file": "", "alias": "probe"}, "key", True,
)
gpu_metadata = json.loads((gpu_dir / "kernel-metadata.json").read_text())
t4_metadata = json.loads((gpu_t4_dir / "kernel-metadata.json").read_text())
cpu_metadata = json.loads((cpu_dir / "kernel-metadata.json").read_text())
check(gpu_metadata["enable_gpu"] is True and cpu_metadata["enable_gpu"] is False,
      "the normal template always requests GPU and only flow validation requests CPU")
check(gpu_metadata["machine_shape"] == "NvidiaTeslaP100"
      and t4_metadata["machine_shape"] == "NvidiaTeslaT4",
      "kernel metadata requests the accelerator derived from the selected model")

p100_models = cli_kaggle.models_for_accelerator("NvidiaTeslaP100")
t4_models = cli_kaggle.models_for_accelerator("NvidiaTeslaT4")
p100_aliases = {model["alias"] for model in p100_models}
t4_aliases = {model["alias"] for model in t4_models}
check(t4_model["alias"] not in p100_aliases,
      "a model that does not fit the selected accelerator is not offered")
check(len(recommended) > 2 and p100_aliases != t4_aliases
      and p100_aliases < t4_aliases,
      "the catalog is larger than two and changing accelerator changes the fit list")

# `TUNNEL_URL=` is what isaacli turns into a ready profile, so the GPU template
# must not print it until the server answers. cloudflared publishes its URL in
# seconds while loading tens of gigabytes takes minutes, and announcing the
# tunnel first hands the user a profile whose first request fails. This path
# cannot be exercised without spending GPU quota, so what is pinned here is the
# order in the rendered file, not a live run.
gpu_code = next(path for path in gpu_dir.iterdir() if path.suffix == ".py")
gpu_source = gpu_code.read_text(encoding="utf-8")
t4_code = next(path for path in gpu_t4_dir.iterdir() if path.suffix == ".py")
t4_source = t4_code.read_text(encoding="utf-8")
check('CUDA_ARCH = "60"' in gpu_source and 'MACHINE_SHAPE = "NvidiaTeslaP100"' in gpu_source
      and 'CUDA_ARCH = "75"' in t4_source
      and 'MACHINE_SHAPE = "NvidiaTeslaT4"' in t4_source,
      "the rendered CUDA architecture matches each requested machine shape")
check("-DCUDA_cuda_driver_LIBRARY=" in gpu_source
      and "/usr/local/nvidia/lib64/libcuda.so.1" in gpu_source,
      "the GPU build passes the mounted CUDA driver library explicitly to CMake")
check("/kaggle/temp" not in gpu_source and "tempfile.gettempdir()" in gpu_source,
      "the GPU template uses Kaggle's writable temporary mount")
check("/v1/models" in gpu_source
      and gpu_source.index("/v1/models") < gpu_source.index('"TUNNEL_URL="'),
      "the GPU template probes the server before publishing the tunnel URL")
check(compile(gpu_source, str(gpu_code), "exec") is not None,
      "the rendered GPU kernel is valid Python before it ever reaches Kaggle")
check(compile(t4_source, str(t4_code), "exec") is not None,
      "the rendered T4 kernel is valid Python before it ever reaches Kaggle")

# `/model` always shows Kaggle, even before a Kaggle profile exists. Selecting
# it must call cli_kaggle.run_kaggle itself, the same function object imported
# by the top-level `isaacli kaggle` command, rather than a copied flow.
selector_config = root / "selector" / "config.json"
config.save({"language": "en", "profiles": {}, "default_profile": None}, selector_config)
original_select = setup_ollama._select
original_run_kaggle = cli_kaggle.run_kaggle
kaggle_setup_calls = []
try:
    setup_ollama._select = lambda _tr, _title, options, *_args, **_kwargs: (
        check(any("Kaggle" in option and "not installed" in option for option in options),
              "Kaggle appears in /model before it is configured"),
        1,
    )[-1]
    cli_kaggle.run_kaggle = lambda **kwargs: (
        kaggle_setup_calls.append(kwargs), 130,
    )[-1]
    selector_result = setup_ollama._select_configured_api(
        lambda _prompt: "", selector_config, "en", setup_ollama.Translator("en"),
    )
finally:
    setup_ollama._select = original_select
    cli_kaggle.run_kaggle = original_run_kaggle
check(selector_result == 130 and len(kaggle_setup_calls) == 1
      and kaggle_setup_calls[0]["config_file"] == selector_config
      and cli._run_kaggle is original_run_kaggle,
      "/model and isaacli kaggle reach the same configuration implementation")

original_launcher_uninstall = cli._uninstall_launcher
original_kaggle_uninstall = cli._uninstall_managed_kaggle
original_input = builtins.input
calls = []
try:
    cli._uninstall_launcher = lambda purge=False, check_only=False: (
        calls.append(("isaac", purge, check_only)), 0,
    )[-1]
    cli._uninstall_managed_kaggle = lambda remove_credentials=False: (
        calls.append(("kaggle", remove_credentials)), 0,
    )[-1]
    builtins.input = lambda _prompt: "y"
    with redirect_stdout(io.StringIO()):
        purge_code = cli.main(["uninstall", "--purge", "--kaggle"])
finally:
    cli._uninstall_launcher = original_launcher_uninstall
    cli._uninstall_managed_kaggle = original_kaggle_uninstall
    builtins.input = original_input
check(purge_code == 0 and calls == [
    ("isaac", True, True), ("kaggle", True), ("isaac", True, False),
], "the explicit Kaggle purge validates, removes Kaggle, then purges isaacli data")

if failures:
    print(f"\n{len(failures)} check(s) failed")
    raise SystemExit(1)
print("\nAll Kaggle checks passed")
