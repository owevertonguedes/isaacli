#!/usr/bin/env python3
"""Kaggle lifecycle checks with isolated local state and no network."""
import io
import ast
import builtins
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))
root = Path(tempfile.mkdtemp())
os.environ["HOME"] = str(root / "home")
os.environ["XDG_CONFIG_HOME"] = str(root / "config")
# A signed-in shell is the case that matters. Asserting these are absent from a
# process that never had them proves nothing, and pointing KAGGLE_CONFIG_DIR at
# a fresh folder is not on its own enough to change account: the CLI answered
# from a cached token and reported another account's quota.
os.environ["KAGGLE_USERNAME"] = "ambient-account"
os.environ["KAGGLE_KEY"] = "ambient-key"

import cli_kaggle
import cli
import config
import setup_ollama


failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


def add_account(path, username="tester", key="key"):
    config.save({"language": "en", "profiles": {}, "default_profile": None}, path)
    cli_kaggle.register_account(username, {"key": key}, path)


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
    if "config view" in joined:
        return SimpleNamespace(returncode=0, stdout="username: user\n", stderr="")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


second_file = root / "second.json"
add_account(second_file)
with redirect_stdout(io.StringIO()):
    second_code = cli_kaggle.run_kaggle(
        input_fn=lambda _prompt: "1",
        run_fn=live_run, which_fn=lambda _name: "/fake/kaggle",
        config_file=second_file, home_dir=home,
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


timeout_file = root / "timeout.json"
add_account(timeout_file)
answers = iter(["1", "1", "n", "y"])
timeout_output = io.StringIO()
with redirect_stdout(timeout_output):
    cli_kaggle.run_kaggle(
        input_fn=lambda _prompt: next(answers), run_fn=push_run,
        popen_fn=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no discovery")),
        which_fn=lambda _name: "/fake/kaggle",
        config_file=timeout_file, home_dir=home,
    )
pushes = [c for c in push_commands if "push" in c]
check(bool(pushes) and all(
    "-t" in c and c[c.index("-t") + 1].isdigit() and int(c[c.index("-t") + 1]) > 0
    for c in pushes),
      "every push carries a session ceiling so an unattended kernel cannot run on")
# Nothing is prepared in this scenario, so both costs apply and both are named.
# The 34 minute figure belongs to compiling, and quoting it at an account that
# already owns the compiled runtime would overstate what it is consenting to.
check("34 minutes" in timeout_output.getvalue()
      and "GiB inside the GPU kernel" in timeout_output.getvalue()
      and "produced inside this GPU kernel" in timeout_output.getvalue(),
      "an unprepared account is told both costs before the self-contained push")

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

# Attached inputs leave model loading as the long step. The measured T4 x2 load
# took 43.5 seconds, while a real attached-weight CPU probe took 11 minutes 26
# seconds wall clock despite one second of script time. The ceiling also has to
# cover scheduling and tunnel startup without retaining the old 75 minutes.
import inspect

discovery_default = inspect.signature(
    cli_kaggle.discover_tunnel_url).parameters["timeout"].default
check(20 * 60 <= discovery_default <= 40 * 60,
      "the discovery ceiling fits attached-input startup with scheduling room")

gpu_dir = root / "gpu-render"
gpu_t4_dir = root / "gpu-t4-render"
self_contained_dir = root / "gpu-self-contained-render"
cpu_dir = root / "cpu-render"
gpu_dir.mkdir()
gpu_t4_dir.mkdir()
self_contained_dir.mkdir()
cpu_dir.mkdir()
recommended = cli_kaggle.recommended_models()
t4_model = next(
    model for model in cli_kaggle.prepared_models()
    if model["machine_shape"] == "NvidiaTeslaT4"
    and model["alias"] not in {
        item["alias"] for item in cli_kaggle.models_for_accelerator("NvidiaTeslaP100")
    })
user_assets = list(cli_kaggle._asset_refs("account-one", t4_model).values())
other_assets = list(cli_kaggle._asset_refs("account-two", t4_model).values())
cli_kaggle._render_kernel(
    gpu_dir, "account-one/gpu", t4_model, "key", False, user_assets)
cli_kaggle._render_kernel(
    gpu_t4_dir, "account-two/gpu-t4", t4_model, "key", False, other_assets)
cli_kaggle._render_kernel(
    self_contained_dir, "account-one/self-contained", t4_model, "key", False, [])
cli_kaggle._render_kernel(
    cpu_dir, "user/cpu", {"repo": "", "file": "", "alias": "probe"}, "key", True,
)
gpu_metadata = json.loads((gpu_dir / "kernel-metadata.json").read_text())
t4_metadata = json.loads((gpu_t4_dir / "kernel-metadata.json").read_text())
cpu_metadata = json.loads((cpu_dir / "kernel-metadata.json").read_text())
self_contained_metadata = json.loads(
    (self_contained_dir / "kernel-metadata.json").read_text())
check(gpu_metadata["enable_gpu"] is True and cpu_metadata["enable_gpu"] is False,
      "the normal template always requests GPU and only flow validation requests CPU")
check(gpu_metadata["machine_shape"] == "NvidiaTeslaT4"
      and t4_metadata["machine_shape"] == "NvidiaTeslaT4",
      "kernel metadata requests the accelerator derived from the selected model")
check(gpu_metadata["dataset_sources"] == user_assets
      and t4_metadata["dataset_sources"] == other_assets
      and user_assets != other_assets,
      "the authenticated owner determines every attached dataset")
check(self_contained_metadata["dataset_sources"] == [],
      "a user without prepared assets gets a self-contained kernel")


def rendered_sources_for_account(username):
    account_file = root / f"authenticated-{username}" / "config.json"
    add_account(account_file, username, f"key-{username}")
    rendered = []

    def authenticated_run(command, check=False, capture_output=False, text=False,
                          env=None, **kwargs):
        joined = " ".join(map(str, command))
        credential = json.loads(
            (Path(env["KAGGLE_CONFIG_DIR"]) / "kaggle.json").read_text())
        if " quota" in joined:
            return SimpleNamespace(returncode=0, stdout="GPU 30h remaining", stderr="")
        if "config view" in joined:
            return SimpleNamespace(
                returncode=0,
                stdout=f"username: {credential['username']}\n", stderr="")
        if "kernels list" in joined:
            return SimpleNamespace(returncode=0, stdout="ref,title\n", stderr="")
        if "datasets list" in joined:
            model = cli_kaggle.prepared_models()[0]
            refs = cli_kaggle._asset_refs(credential["username"], model)
            rows = "\n".join(f"{ref},asset" for ref in refs.values())
            return SimpleNamespace(returncode=0, stdout=f"ref,title\n{rows}\n", stderr="")
        if "kernels push" in joined:
            folder = Path(command[command.index("-p") + 1])
            rendered.append(json.loads((folder / "kernel-metadata.json").read_text()))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    answers = iter(["1", "1", "y"])
    with redirect_stdout(io.StringIO()):
        cli_kaggle.run_kaggle(
            input_fn=lambda _prompt: next(answers), run_fn=authenticated_run,
            popen_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("stop after render")),
            which_fn=lambda _name: "/fake/kaggle", config_file=account_file,
            home_dir=home,
        )
    return rendered[0]["dataset_sources"]


authenticated_first = rendered_sources_for_account("authenticated-one")
authenticated_second = rendered_sources_for_account("authenticated-two")
check(authenticated_first != authenticated_second
      and all(ref.startswith("authenticated-one/") for ref in authenticated_first)
      and all(ref.startswith("authenticated-two/") for ref in authenticated_second),
      "changing kaggle config view changes the datasets in rendered kernel metadata")

# The model screen is the one the user reaches last and complained about first:
# three printed lines per candidate turned six models into a wall. What is
# pinned is the effect, not a string. The selector is replaced by a recorder, so
# a screen that goes back to print plus input never reaches it and this stays
# empty. Each option also has to be a single line, because that is what the
# selector draws and what its capacity arithmetic assumes.
drawn = []
original_select = cli_kaggle.terminal_ui.select
try:
    cli_kaggle.terminal_ui.select = lambda title, options, **kwargs: (
        drawn.append((title, options)) or 0)
    with redirect_stdout(io.StringIO()) as chosen_output:
        chosen_model = cli_kaggle._select_model(lambda _prompt: "1")
finally:
    cli_kaggle.terminal_ui.select = original_select
check(len(drawn) == 1 and len(drawn[0][1]) == len(cli_kaggle.prepared_models())
      and all("\n" not in option for option in drawn[0][1])
      and chosen_model["name"] in drawn[0][1][0],
      "the Kaggle model screen is drawn by the shared selector, one line per model")
check(chosen_model["benchmark_source"] in chosen_output.getvalue()
      and chosen_model["source"] in chosen_output.getvalue(),
      "the evidence behind the chosen model is still shown, without the wall")

# The prepared runtime is found by name, and the name carries both the compute
# architecture and the llama.cpp build. A P100 launch used to extract
# llama-cuda-sm60-b10502 and then look for the server inside a directory named
# sm75, which is a crash after the archive was already attached. And if any of
# the four places that spell the build tag drifts from the others, the GPU
# kernel silently stops finding what was prepared and recompiles for half an
# hour instead, which is the failure the prepared asset exists to remove.
p100_dir = root / "p100-render"
p100_dir.mkdir()
p100_model = next(item for item in cli_kaggle.recommended_models()
                  if item["cuda_arch"] == "60")
cli_kaggle._render_kernel(p100_dir, "account/p100", p100_model, "key", False, [])
p100_source = next(path for path in p100_dir.iterdir()
                   if path.suffix == ".py").read_text(encoding="utf-8")
check("sm75" not in p100_source and 'CUDA_ARCH = "60"' in p100_source,
      "a kernel rendered for one architecture never names another one")

# A prepared runtime is copied out of the dataset into a directory that is not
# the one it was compiled in, so the RPATH inside the binary points at nothing
# and every directory holding a shared object has to be on the search path.
# Measured on Kaggle: naming lib/ alone left llama-server unable to open
# libllama-server-impl.so, which lives in bin/, after the weights had already
# been downloaded. The function is lifted out of the rendered kernel and run
# against a tree shaped like the published dataset, so this exercises the code
# that ships rather than a copy of it.
runtime_fixture = root / "runtime-fixture" / "llama-cuda-prepared"
for relative in ("bin/llama-server", "bin/libllama-server-impl.so",
                 "bin/libggml-cuda.so.0", "lib/libcudart.so.12",
                 "cloudflared-linux-amd64"):
    target = runtime_fixture / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"binary")
rendered_tree = ast.parse(p100_source)
library_path_node = next(
    node for node in rendered_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "library_path")
library_namespace = {"os": os}
exec(compile(ast.Module(body=[library_path_node], type_ignores=[]),
             "<rendered kernel>", "exec"), library_namespace)
searched = library_namespace["library_path"](runtime_fixture).split(os.pathsep)
check(str(runtime_fixture / "bin") in searched
      and str(runtime_fixture / "lib") in searched,
      "the rendered kernel searches every directory of the runtime that holds a library")

templates = {
    path.name: path.read_text(encoding="utf-8")
    for path in (HERE.parent / "contrib" / "kaggle").glob("*.tmpl")
}
build_tags = {
    name: set(re.findall(r"b(\d{5,})", source))
    for name, source in templates.items() if "b10502" in source
}
build_tags["cli_kaggle"] = set(re.findall(
    r"b(\d{5,})", inspect.getsource(cli_kaggle)))
check(len(build_tags) >= 3
      and len(set.union(*build_tags.values())) == 1
      and all(len(tags) == 1 for tags in build_tags.values()),
      f"every place that names the llama.cpp build names the same one ({build_tags})")

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
check('CUDA_ARCH = "75"' in gpu_source and 'MACHINE_SHAPE = "NvidiaTeslaT4"' in gpu_source
      and 'CUDA_ARCH = "75"' in t4_source
      and 'MACHINE_SHAPE = "NvidiaTeslaT4"' in t4_source,
      "the rendered CUDA architecture matches each requested machine shape")
check('"git", "clone"' in gpu_source and '"curl"' in gpu_source
      and "huggingface.co" in gpu_source
      and "optional_input(MODEL_FILE)" in gpu_source
      and 'optional_input("llama-server")' in gpu_source,
      "the rendered GPU kernel retains the announced self-contained path")
check("/v1/models" in gpu_source
      and gpu_source.index("/v1/models") < gpu_source.index('"TUNNEL_URL="'),
      "the GPU template probes the server before publishing the tunnel URL")
check(compile(gpu_source, str(gpu_code), "exec") is not None,
      "the rendered GPU kernel is valid Python before it ever reaches Kaggle")
check(compile(t4_source, str(t4_code), "exec") is not None,
      "the rendered T4 kernel is valid Python before it ever reaches Kaggle")

# A kernel recorded by isaacli is reused only after its authenticated endpoint
# answers. This is pinned by the absence of a push, not by presentation text.
reuse_file = root / "reuse" / "config.json"
reuse_profile = "kaggle-existing"
reuse_slug = "user/existing"
config.save({
    "language": "en", "default_profile": "other",
    "profiles": {
        reuse_profile: {
            "provider": "openai_compatible", "provider_name": "Kaggle",
            "base_url": "https://live.trycloudflare.com/v1",
            "model": "qwen38-27b", "credential": "reuse-key",
        },
    },
    "kaggle": {"kernels": [{
        "slug": reuse_slug, "url": "https://live.trycloudflare.com",
        "model": "qwen38-27b",
    }]},
}, reuse_file)
config.save_secret("reuse-key", "secret", reuse_file.with_name("secrets.json"))
cli_kaggle.register_account("user", {"key": "account-key"}, reuse_file)
reuse_commands = []


def reuse_run(command, check=False, capture_output=False, text=False, **kwargs):
    reuse_commands.append([str(part) for part in command])
    joined = " ".join(map(str, command))
    if " quota" in joined:
        return SimpleNamespace(returncode=0, stdout="GPU quota", stderr="")
    if "kernels list" in joined:
        return SimpleNamespace(returncode=0, stdout=f"ref,title\n{reuse_slug},Existing\n", stderr="")
    if "kernels status" in joined:
        return SimpleNamespace(returncode=0, stdout="KernelWorkerStatus.RUNNING", stderr="")
    if "config view" in joined:
        return SimpleNamespace(returncode=0, stdout="username: user\n", stderr="")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


class HealthyAnswer:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


with redirect_stdout(io.StringIO()):
    reuse_code = cli_kaggle.run_kaggle(
        input_fn=lambda _prompt: "1",
        run_fn=reuse_run, which_fn=lambda _name: "/fake/kaggle",
        config_file=reuse_file, home_dir=home,
        urlopen_fn=lambda _request, timeout=0: HealthyAnswer(),
    )
check(reuse_code == 0 and config.load(reuse_file)["default_profile"] == reuse_profile
      and not any("kernels push" in " ".join(command) for command in reuse_commands),
      "a live responsive isaacli kernel reactivates its profile without a push")

dead_file = root / "dead" / "config.json"
config.save({
    "language": "en", "default_profile": "other",
    "profiles": {"other": {"model": "local"}, reuse_profile: {
        "provider": "openai_compatible", "provider_name": "Kaggle",
        "base_url": "https://dead.trycloudflare.com/v1",
        "model": "qwen38-27b", "credential": "dead-key",
    }},
    "kaggle": {"kernels": [{
        "slug": "user/dead", "url": "https://dead.trycloudflare.com",
        "profile": reuse_profile, "model": "qwen38-27b",
    }]},
}, dead_file)
cli_kaggle.register_account("user", {"key": "account-key"}, dead_file)
dead_commands = []


def dead_run(command, check=False, capture_output=False, text=False, **kwargs):
    dead_commands.append([str(part) for part in command])
    joined = " ".join(map(str, command))
    if " quota" in joined:
        return SimpleNamespace(returncode=0, stdout=REAL_QUOTA_TABLE, stderr="")
    if "kernels list" in joined:
        return SimpleNamespace(returncode=0, stdout="ref,title\n", stderr="")
    if "config view" in joined:
        return SimpleNamespace(returncode=0, stdout="username: user\n", stderr="")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


REAL_QUOTA_TABLE = (
    "resource  used    remaining  total   refreshAt            \n"
    "--------  ------  ---------  ------  -------------------  \n"
    "GPU       17.82h  12.18h     30.00h  2026-08-22T00:00:00  \n"
    "TPU       0.00h   20.00h     20.00h  2026-08-22T00:00:00"
)

# Kaggle already writes the unit onto every figure in that table. Appending
# another one produced "12.18h h restantes de 30.00h h" on the real account,
# which is what the person actually reads before deciding whether to spend the
# hours. Pinning the exact rendering is what catches a unit being added back.
summary = cli_kaggle._quota_summary(REAL_QUOTA_TABLE)
check(summary == "GPU 12.18h left of 30.00h",
      "the quota summary carries Kaggle's own figures with one unit each")

dead_answers = iter(["1", "1", "n", "n"])
dead_output = io.StringIO()
with redirect_stdout(dead_output):
    dead_code = cli_kaggle.run_kaggle(
        input_fn=lambda _prompt: next(dead_answers), run_fn=dead_run,
        which_fn=lambda _name: "/fake/kaggle", config_file=dead_file, home_dir=home,
    )
check(dead_code == 130 and config.load(dead_file)["default_profile"] == "other"
      and "user/dead" in dead_output.getvalue()
      and not any("kernels push" in " ".join(command) for command in dead_commands),
      "a dead kernel is reported and its broken profile is not reactivated")
check("12.18h" in dead_output.getvalue()
      and "refreshAt" not in dead_output.getvalue()
      and "--------" not in dead_output.getvalue(),
      "the run shows the hours left, not the raw quota table a second time")

dead_state = config.load(dead_file)
check(not (dead_state.get("kaggle") or {}).get("kernels")
      and reuse_profile not in dead_state["profiles"]
      and "other" in dead_state["profiles"],
      "a kernel Kaggle no longer runs is forgotten, with the profile it created")

# Pruning has to be decided by Kaggle and bounded by ownership. A record that is
# still running, one belonging to another account, and one whose profile no
# longer holds that dead URL are three different reasons to keep something, and
# each of them has been a way to delete somebody's working configuration.
keep_file = root / "keep" / "config.json"
config.save({
    "language": "en", "default_profile": "kept-live",
    "profiles": {
        "kept-live": {"provider": "openai_compatible",
                       "base_url": "https://live.trycloudflare.com/v1"},
        "kept-edited": {"provider": "openai_compatible",
                         "base_url": "https://elsewhere.example/v1"},
    },
    "kaggle": {"kernels": [
        {"slug": "owner/live", "url": "https://live.trycloudflare.com",
         "profile": "kept-live", "account": "owner"},
        {"slug": "other/over", "url": "https://gone.trycloudflare.com",
         "profile": "kept-edited", "account": "other"},
        {"slug": "owner/over", "url": "https://gone2.trycloudflare.com",
         "profile": "kept-edited", "account": "owner"},
    ]},
}, keep_file)
pruned = cli_kaggle._prune_dead_kernels(
    [("owner/live", "KernelWorkerStatus.RUNNING")], keep_file, "owner", "owner")
kept_state = config.load(keep_file)
kept_slugs = [item["slug"] for item in kept_state["kaggle"]["kernels"]]
check([record["slug"] for record in pruned] == ["owner/over"]
      and kept_slugs == ["owner/live", "other/over"]
      and set(kept_state["profiles"]) == {"kept-live", "kept-edited"}
      and kept_state["default_profile"] == "kept-live",
      "pruning removes only this account's finished kernels, and no repointed profile")

accounts_file = root / "accounts" / "config.json"
add_account(accounts_file, "first-account", "first-key")
cli_kaggle.register_account("second-account", {"key": "second-key"}, accounts_file)
account_calls = []


def account_run(command, check=False, capture_output=False, text=False, env=None,
                answers_as=None, **kwargs):
    # The stand-in answers from the credential it can reach through the
    # environment it was handed, which is the same thing the real CLI does and
    # the only reason this proves anything about isolation.
    credential = json.loads(
        (Path(env["KAGGLE_CONFIG_DIR"]) / "kaggle.json").read_text())
    parts = list(map(str, command))
    account_calls.append((parts, credential, dict(env)))
    if parts[1:3] == ["config", "view"]:
        return SimpleNamespace(
            returncode=0, stdout=f"- username: {credential['username']}\n", stderr="")
    if parts[2:4] == ["list", "--mine"]:
        # Kaggle prefixes what it owns with the real owner, which is the part a
        # local config file cannot fake.
        owner = answers_as or credential["username"]
        return SimpleNamespace(
            returncode=0, stdout=f"ref,title\n{owner}/thing,Thing\n", stderr="")
    # The real `kaggle quota` answers with a four line table. Standing in with
    # a friendly sentence would have this check pass against a parser that
    # cannot read what Kaggle actually sends.
    hours = "1.00" if credential["username"] == "first-account" else "5.00"
    remaining = "29.00" if credential["username"] == "first-account" else "25.00"
    return SimpleNamespace(
        returncode=0,
        stdout=(
            "resource  used    remaining  total   refreshAt\n"
            "--------  ------  ---------  ------  -------------------\n"
            f"GPU       {hours}h  {remaining}h    30.00h  2026-08-22T00:00:00\n"
            "TPU       0.00h   20.00h     20.00h  2026-08-22T00:00:00\n"
        ),
        stderr="",
    )


accounts_output = io.StringIO()
with redirect_stdout(accounts_output):
    selected_account, selected_env = cli_kaggle._select_account(
        Path("/fake/kaggle"), lambda _prompt: "2", account_run, accounts_file)
selected_credential = json.loads(
    (Path(selected_env["KAGGLE_CONFIG_DIR"]) / "kaggle.json").read_text())
check(selected_account == "second-account"
      and selected_credential == {"username": "second-account", "key": "second-key"}
      and account_calls[-1][1]["username"] == "second-account"
      and "KAGGLE_USERNAME" not in selected_env
      and "KAGGLE_KEY" not in selected_env
      and "29.00" in accounts_output.getvalue()
      and "25.00" in accounts_output.getvalue()
      and "refreshAt" not in accounts_output.getvalue(),
      "every account shows its own hours left, not the raw table mashed onto one line")

# Read from the CLI 2.2.4 source and then confirmed by running it: the access
# token and the OAuth credentials come from `~/.kaggle/`, through `expanduser`,
# which KAGGLE_CONFIG_DIR does not redirect. An account folder holding a
# deliberately invalid credential still answered with the ambient account's real
# quota on this machine. Only HOME moves those two files.
check(selected_env["HOME"] == str(
          cli_kaggle._account_dir("second-account", accounts_file))
      and selected_env["KAGGLE_CONFIG_DIR"].startswith(selected_env["HOME"])
      and "KAGGLE_API_TOKEN" not in selected_env,
      "the account folder becomes the home the CLI sees, which is what moves its token")
check(selected_env.get("PYTHONUSERBASE"),
      "moving HOME keeps a pip --user CLI able to import itself")

# Isolation is an argument about environment variables, and an argument is not
# evidence. What makes the selection real is asking the CLI who it is.
mismatch_error = None
try:
    with redirect_stdout(io.StringIO()):
        cli_kaggle._select_account(
            Path("/fake/kaggle"), lambda _prompt: "2",
            lambda *a, **k: account_run(*a, answers_as="someone-else", **k),
            accounts_file)
except RuntimeError as error:
    mismatch_error = str(error)
check(mismatch_error and "someone-else" in mismatch_error
      and "second-account" in mismatch_error,
      "an account that answers as somebody else is refused, not quietly used")


def empty_account_run(command, check=False, capture_output=False, text=False,
                      env=None, **kwargs):
    parts = list(map(str, command))
    if parts[2:4] == ["list", "--mine"]:
        return SimpleNamespace(returncode=0, stdout="ref,title\n", stderr="")
    return account_run(command, check, capture_output, text, env, **kwargs)


# A brand new account owns nothing, so the server has nothing to attribute and
# the identity cannot be confirmed. A layer that cannot run has to say so.
fresh_output = io.StringIO()
with redirect_stdout(fresh_output):
    fresh_account, _fresh_env = cli_kaggle._select_account(
        Path("/fake/kaggle"), lambda _prompt: "2", empty_account_run, accounts_file)
check(fresh_account == "second-account" and "NOTE:" in fresh_output.getvalue(),
      "an account Kaggle cannot attribute anything to is announced, not confirmed")

private_dir = root / "private-dataset"
private_dir.mkdir()
(private_dir / "asset.bin").write_bytes(b"asset")
private_commands = []


def private_run(command, **kwargs):
    private_commands.append(list(map(str, command)))
    return SimpleNamespace(returncode=0, stdout="", stderr="")


cli_kaggle._publish_private_dataset(
    Path("/fake/kaggle"), private_dir, "owner/private-asset",
    "private asset", private_run, {})
private_metadata = json.loads((private_dir / "dataset-metadata.json").read_text())
check(private_metadata["isPrivate"] is True
      and all("-u" not in command for command in private_commands),
      "asset preparation publishes a private dataset and never requests public access")

preparation_dir = root / "preparation-render"
preparation_dir.mkdir()
cli_kaggle._render_preparation_kernel(
    preparation_dir, "owner/prepare", "75")
preparation_metadata = json.loads(
    (preparation_dir / "kernel-metadata.json").read_text())
check(preparation_metadata["enable_gpu"] is False
      and preparation_metadata["is_private"] is True,
      "asset compilation runs in a private CPU kernel")

owned_sources = [HERE.parent / "tool_harness", HERE]
forbidden_hits = []
for source_root in owned_sources:
    for path in source_root.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".json", ".md"}:
            forbidden_name = "owe" + "vertonguedes"
            if forbidden_name in path.read_text(encoding="utf-8", errors="ignore"):
                forbidden_hits.append(str(path))
check(not forbidden_hits,
      "no personal Kaggle account name is fixed in tool_harness or tests")

# Nothing isaacli creates in someone's Kaggle account may be world readable.
# The CLI already defaults to private, so the danger is a stray --public or a
# metadata file that says otherwise, and both are cheap to forbid outright.
public_flag_hits = []
private_flag_hits = []
public_marks = ('"--pub' + 'lic"', '"' + '-u"', "'--pub" + "lic'")
private_marks = ('"isPri' + 'vate": False', '"is_pri' + 'vate": False')
for source_root in [HERE.parent / "tool_harness", HERE.parent / "contrib"]:
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".tmpl", ".json"}:
            continue
        body = path.read_text(encoding="utf-8", errors="ignore")
        if any(mark in body for mark in public_marks):
            public_flag_hits.append(str(path))
        if any(mark in body for mark in private_marks):
            private_flag_hits.append(str(path))
check(not public_flag_hits,
      "no Kaggle path anywhere asks for public visibility")
check(not private_flag_hits,
      "no Kaggle dataset or kernel is ever declared non-private")

# Registering an account writes a real Kaggle credential to disk. It must not
# land in the public config, and the file the CLI reads has to be unreadable by
# anyone else on the machine. Checked by effect, on the bytes and the mode.
credential_config = root / "credentials" / "config.json"
config.save({"language": "en"}, credential_config)
cli_kaggle.register_account(
    "tester", {"key": "SECRET-UNDER-TEST"}, credential_config)
account_env = cli_kaggle._account_environment("tester", credential_config)
account_files = sorted(Path(account_env["KAGGLE_CONFIG_DIR"]).iterdir())
carrying_secret = [
    path for path in account_files
    if "SECRET-UNDER-TEST" in path.read_text(encoding="utf-8", errors="ignore")
]
check(bool(carrying_secret)
      and all((path.stat().st_mode & 0o777) == 0o600 for path in carrying_secret),
      "the credential the Kaggle CLI reads is written unreadable by other users")
check("SECRET-UNDER-TEST" not in credential_config.read_text(encoding="utf-8"),
      "a Kaggle credential never reaches the config file that carries no secrets")

# This check used to require KAGGLE_API_TOKEN to be set, on the belief that
# pointing it at an empty file neutralised the cached token. Reading the CLI
# 2.2.4 source and then running it showed the opposite: an empty token file
# reads as no token at all, so the lookup falls through to
# `~/.kaggle/access_token`, which `expanduser` resolves from HOME and which
# KAGGLE_CONFIG_DIR never touches. Measured here: an account folder holding a
# deliberately invalid credential still answered with the ambient account's real
# quota. HOME is the variable that moves those files, so HOME is what this pins.
check(account_env.get("HOME")
      and account_env.get("KAGGLE_CONFIG_DIR", "").startswith(account_env["HOME"])
      and "KAGGLE_API_TOKEN" not in account_env
      and "KAGGLE_USERNAME" not in account_env
      and "KAGGLE_KEY" not in account_env,
      "account selection moves HOME, which is what the CLI reads its token through")

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

# ----------------------------------------------------------------------
# Preparing the reusable assets on its own, which is the step that costs
# nothing and until now could only be reached by starting the step that costs
# hours. What is checked is the effect: every kernel this command pushes has to
# declare no accelerator, and no GPU kernel may be pushed at all.
# ----------------------------------------------------------------------
prepare_file = root / "prepare" / "config.json"
add_account(prepare_file, "preparer", "prepare-key")
prepare_pushes = []
prepare_datasets = []
prepare_paths = []
prepared_model = cli_kaggle.prepared_models()[0]
prepared_refs = cli_kaggle._asset_refs("preparer", prepared_model)
prepared_archive = prepared_refs["binary"].split("/", 1)[1].replace("isaacli-", "")


def prepare_run(command, check=False, capture_output=False, text=False, env=None,
                **kwargs):
    parts = list(map(str, command))
    joined = " ".join(parts)
    if " quota" in joined:
        return SimpleNamespace(returncode=0, stdout=REAL_QUOTA_TABLE, stderr="")
    if "config view" in joined:
        return SimpleNamespace(returncode=0, stdout="username: preparer\n", stderr="")
    if "kernels list" in joined or "datasets list" in joined:
        return SimpleNamespace(returncode=0, stdout="ref,title\npreparer/x,X\n", stderr="")
    if "kernels push" in joined:
        folder = Path(parts[parts.index("-p") + 1])
        prepare_paths.append(folder)
        prepare_pushes.append(json.loads(
            (folder / "kernel-metadata.json").read_text()))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    if "kernels status" in joined:
        return SimpleNamespace(
            returncode=0, stdout="KernelWorkerStatus.COMPLETE", stderr="")
    if "kernels output" in joined:
        # The real command writes the archive the CPU kernel produced, named
        # for the architecture that kernel was told to compile for.
        folder = Path(parts[parts.index("-p") + 1])
        (folder / f"{prepared_archive}.tar.gz").write_bytes(b"runtime")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    if "datasets create" in joined:
        folder = Path(parts[parts.index("-p") + 1])
        prepare_datasets.append((
            json.loads((folder / "dataset-metadata.json").read_text()),
            sorted(path.name for path in folder.iterdir()),
        ))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


# "1" selects the account, "1" the model, "y" confirms preparation, "n" declines
# publishing the weight, which is the part that downloads gigabytes locally.
prepare_answers = iter(["1", "1", "y", "n"])
prepare_output = io.StringIO()
with redirect_stdout(prepare_output):
    prepare_code = cli_kaggle.run_prepare_assets(
        input_fn=lambda _prompt: next(prepare_answers), run_fn=prepare_run,
        which_fn=lambda _name: "/fake/kaggle", config_file=prepare_file,
        home_dir=home,
    )
prepare_metadata, prepare_files = prepare_datasets[0] if prepare_datasets else ({}, [])
check(prepare_code == 0 and len(prepare_pushes) == 1
      and prepare_pushes[0]["enable_gpu"] is False
      and prepare_pushes[0]["enable_tpu"] is False
      and "machine_shape" not in prepare_pushes[0],
      "preparing assets pushes one kernel and it asks for no accelerator")
check(len(prepare_datasets) == 1 and prepare_metadata["isPrivate"] is True
      and prepare_metadata["id"] == prepared_refs["binary"]
      and f"{prepared_archive}.tar.gz" in prepare_files,
      "the compiled CUDA runtime is published as a private dataset of that account")
check(config.load(prepare_file).get("default_profile") is None
      and not (config.load(prepare_file).get("kaggle") or {}).get("kernels"),
      "preparation leaves no endpoint behind, because it never started a server")

# The plan is the thing being consented to, so it has to name the steps that
# will really run. With the runtime already published, announcing a CPU kernel
# promises work that is skipped, and the run must not push one either.
partial_file = root / "partial" / "config.json"
add_account(partial_file, "preparer", "prepare-key")
partial_pushes = []


def partial_run(command, check=False, capture_output=False, text=False, env=None,
                **kwargs):
    parts = list(map(str, command))
    joined = " ".join(parts)
    if "kernels push" in joined:
        partial_pushes.append(parts)
    if "datasets list" in joined:
        return SimpleNamespace(
            returncode=0,
            stdout=f"ref,title\n{prepared_refs['binary']},Runtime\n", stderr="")
    return prepare_run(command, check, capture_output, text, env, **kwargs)


# "1" account, "1" model, "n" declines, so nothing is built and only the plan
# that was going to be carried out is on screen.
partial_answers = iter(["1", "1", "n"])
with redirect_stdout(io.StringIO()) as partial_output:
    partial_code = cli_kaggle.run_prepare_assets(
        input_fn=lambda _prompt: next(partial_answers), run_fn=partial_run,
        which_fn=lambda _name: "/fake/kaggle", config_file=partial_file,
        home_dir=home,
    )
partial_text = partial_output.getvalue()
check(partial_code == 130 and not partial_pushes
      and prepared_refs["binary"] in partial_text
      and "CPU kernel" not in partial_text
      and f"{prepared_model['model_bytes'] / 1024 ** 3:.1f} GiB" in partial_text,
      "the plan names only the assets that are actually missing")

# Staging belongs on a disk, and the refusal belongs before the transfer. /tmp
# is a tmpfs on a normal desktop, so a 15 GiB weight staged there is written
# into RAM. What is checked is the effect: the scratch directory is not the
# system temp root, and a weight that cannot fit is refused without curl ever
# being invoked, rather than after spending the whole download to find out.
space_file = root / "space" / "config.json"
add_account(space_file, "preparer", "prepare-key")
space_calls = []
original_free = cli_kaggle._free_bytes
try:
    cli_kaggle._free_bytes = lambda _path: SimpleNamespace(
        total=0, used=0, free=prepared_model["model_bytes"] - 1)

    def space_run(command, check=False, capture_output=False, text=False, env=None,
                  **kwargs):
        space_calls.append(list(map(str, command)))
        return prepare_run(command, check, capture_output, text, env, **kwargs)

    refused = ""
    with redirect_stdout(io.StringIO()) as space_output:
        # The runtime is already published, so only the weight is left to decide.
        cli_kaggle._prepare_assets(
            "/fake/kaggle", "preparer", prepared_model, {"binary": "x"},
            lambda _prompt: "y", space_run, {},
        )
except RuntimeError as error:
    refused = str(error)
finally:
    cli_kaggle._free_bytes = original_free
check("GiB" in refused and not any("curl" in parts[0] for parts in space_calls),
      "a weight that cannot fit is refused with the numbers, before any download")
check(all(Path(path).is_relative_to(config.cache_path()) for path in prepare_paths)
      and prepare_paths
      and Path(cli_kaggle._scratch_root()) == config.cache_path(),
      "large staging follows the cache location, not the system temp filesystem")

# Uninstalling has to reach the account, not only the disk. A kernel left in a
# Kaggle account can still be spending quota, and the credential that could
# delete it is about to be removed from this machine. It is a third party
# account, so the list goes on screen and refusing must not block the local
# cleanup.
remote_file = root / "remote" / "config.json"
add_account(remote_file, "leaver", "leave-key")
remote_commands = []


def remote_run(command, check=False, capture_output=False, text=False, env=None,
               **kwargs):
    parts = list(map(str, command))
    remote_commands.append(parts)
    joined = " ".join(parts)
    if "kernels list" in joined:
        return SimpleNamespace(
            returncode=0,
            stdout="ref,title\nleaver/isaacli-gpu-1,Gpu\nleaver/my-own-notebook,Mine\n",
            stderr="")
    if "datasets list" in joined:
        return SimpleNamespace(
            returncode=0,
            stdout="ref,title\nleaver/isaacli-llama-cuda-sm75-b10502,Runtime\n"
                   "leaver/my-own-dataset,Mine\n",
            stderr="")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


found = cli_kaggle.remote_leftovers(Path("/fake/kaggle"), remote_file, remote_run)
check(found == {"leaver": {
          "kernels": ["leaver/isaacli-gpu-1"],
          "datasets": ["leaver/isaacli-llama-cuda-sm75-b10502"]}},
      "what this program left on Kaggle is listed, and the user's own work is not")

kept_output = io.StringIO()
with redirect_stdout(kept_output):
    kept = cli._offer_remote_cleanup(
        input_fn=lambda _prompt: "n", which_fn=lambda _name: "/fake/kaggle",
        run_fn=remote_run, config_file=remote_file)
check(kept == {}
      and not any("delete" in parts for parts in remote_commands)
      and "leaver/isaacli-gpu-1" in kept_output.getvalue(),
      "refusing the remote cleanup shows the list and deletes nothing")

with redirect_stdout(io.StringIO()):
    cli._offer_remote_cleanup(
        input_fn=lambda _prompt: "y", which_fn=lambda _name: "/fake/kaggle",
        run_fn=remote_run, config_file=remote_file)
remote_deletes = [parts[1:4] for parts in remote_commands if "delete" in parts]
check(remote_deletes == [["kernels", "delete", "leaver/isaacli-gpu-1"],
                         ["datasets", "delete",
                          "leaver/isaacli-llama-cuda-sm75-b10502"]],
      "confirming deletes exactly what was listed, on Kaggle, and nothing else")

# Stopping is the only thing that stops the quota, and for a long time the
# program said it was impossible. Measured against a kernel that really was
# running: `kernels delete` succeeded, the tunnel answered HTTP 530 afterwards
# and the remaining GPU hours stopped moving. What is checked here is that it
# never happens without being asked for, that it really goes through delete,
# and that the endpoint it published is forgotten with it.
stop_file = root / "stop" / "config.json"
config.save({
    "language": "en", "default_profile": "kaggle-live",
    "profiles": {"kaggle-live": {
        "provider": "openai_compatible",
        "base_url": "https://live.trycloudflare.com/v1",
        "model": "qwen38-27b", "credential": "live-key"}},
    "kaggle": {"kernels": [{
        "slug": "stopper/live", "url": "https://live.trycloudflare.com",
        "profile": "kaggle-live", "model": "qwen38-27b", "account": "stopper"}]},
}, stop_file)
cli_kaggle.register_account("stopper", {"key": "stop-key"}, stop_file)
stop_commands = []


def stop_run(command, check=False, capture_output=False, text=False, env=None,
             **kwargs):
    parts = list(map(str, command))
    stop_commands.append(parts)
    joined = " ".join(parts)
    if " quota" in joined:
        return SimpleNamespace(returncode=0, stdout=REAL_QUOTA_TABLE, stderr="")
    if "config view" in joined:
        return SimpleNamespace(returncode=0, stdout="username: stopper\n", stderr="")
    if "kernels list" in joined:
        return SimpleNamespace(
            returncode=0, stdout="ref,title\nstopper/live,Live\n", stderr="")
    if "datasets list" in joined:
        return SimpleNamespace(
            returncode=0, stdout="ref,title\nstopper/thing,Thing\n", stderr="")
    if "kernels status" in joined:
        return SimpleNamespace(
            returncode=0, stdout="KernelWorkerStatus.RUNNING", stderr="")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


declined_answers = iter(["1", "1", "n"])
with redirect_stdout(io.StringIO()):
    declined_stop = cli_kaggle.run_stop_kernels(
        input_fn=lambda _prompt: next(declined_answers), run_fn=stop_run,
        which_fn=lambda _name: "/fake/kaggle", config_file=stop_file, home_dir=home)
check(declined_stop == 130
      and not any("kernels delete" in " ".join(parts) for parts in stop_commands)
      and (config.load(stop_file)["kaggle"]["kernels"] or [{}])[0]["slug"]
      == "stopper/live",
      "declining to stop deletes nothing and keeps the recorded session")

stop_answers = iter(["1", "1", "y"])
with redirect_stdout(io.StringIO()) as stop_output:
    stopped = cli_kaggle.run_stop_kernels(
        input_fn=lambda _prompt: next(stop_answers), run_fn=stop_run,
        which_fn=lambda _name: "/fake/kaggle", config_file=stop_file, home_dir=home)
deletes = [parts for parts in stop_commands if "delete" in parts]
stop_state = config.load(stop_file)
check(stopped == 0 and len(deletes) == 1
      and deletes[0][1:] == ["kernels", "delete", "stopper/live", "--yes"],
      "stopping a session goes through the only command Kaggle has for it")
check(not (stop_state.get("kaggle") or {}).get("kernels")
      and "kaggle-live" not in stop_state["profiles"],
      "the endpoint a stopped kernel published is forgotten with the kernel")

# The command has to reach preparation through the same wrapper the launch uses,
# because that wrapper is what lends it the live model screen. Routing it
# straight at cli_kaggle would give this command a different, smaller list.
routed = []
original_prepare = setup_ollama.run_prepare_assets
original_launch = setup_ollama.run_kaggle
try:
    setup_ollama.run_prepare_assets = lambda **kwargs: routed.append("prepare") or 0
    setup_ollama.run_kaggle = lambda **kwargs: routed.append(
        ("launch", kwargs.get("validation_cpu"))) or 0
    with redirect_stdout(io.StringIO()):
        prepare_route = cli.main(["kaggle", "--prepare-assets"])
        launch_route = cli.main(["kaggle"])
        bogus_route = cli.main(["kaggle", "--prepare"])
finally:
    setup_ollama.run_prepare_assets = original_prepare
    setup_ollama.run_kaggle = original_launch
check(prepare_route == 0 and launch_route == 0 and bogus_route == 2
      and routed == ["prepare", ("launch", False)],
      "isaacli kaggle --prepare-assets prepares and never falls through to a launch")

# ----------------------------------------------------------------------
# Adding an account the way the Kaggle CLI already supports.
#
# Typing a username and pasting a token is doing by hand what `auth login`
# does through the browser, and the username is something the CLI can be
# asked for. Both of these drive the real functions; the browser round trip
# itself is the one part a test cannot perform.
# ----------------------------------------------------------------------
login_file = root / "login" / "config.json"
config.save({"language": "en", "profiles": {}, "default_profile": None}, login_file)
login_commands = []


def refuse_input(_prompt=""):
    raise AssertionError("the account flow asked the user to type something")


def login_run(command, check=False, capture_output=False, text=False, env=None,
              **kwargs):
    command = list(map(str, command))
    login_commands.append((command, dict(env or {})))
    if command[1:3] == ["config", "view"]:
        directory = Path(env["KAGGLE_CONFIG_DIR"])
        # The real CLI answers with whoever finished the browser flow. Standing
        # in for that means writing the session into the folder it was told to
        # use, which is also what proves the folder is the one being used.
        (directory / "kaggle-session").write_text("session", encoding="utf-8")
        return SimpleNamespace(
            returncode=0, stdout="- username: browser-user\n", stderr="")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


with redirect_stdout(io.StringIO()):
    logged_in = cli_kaggle.login_account(
        Path("/fake/kaggle"), login_file, login_run)
login_data = config.load(login_file)
login_command, login_env = login_commands[0]
session_dir = cli_kaggle._account_dir("browser-user", login_file)
secrets_file = login_file.with_name("secrets.json")

check(logged_in == "browser-user"
      and login_data["kaggle"]["accounts"]["browser-user"] == {"browser_login": True}
      and login_data["kaggle"]["selected_account"] == "browser-user",
      "the browser sign-in registers the account the CLI says answered")
check(login_command[1:] == ["auth", "login", "--force"],
      "the sign-in forces a fresh login instead of accepting the cached one")
check(login_env["KAGGLE_CONFIG_DIR"].startswith(
          str(cli_kaggle._accounts_root(login_file)))
      and "KAGGLE_USERNAME" not in login_env and "KAGGLE_KEY" not in login_env,
      "the sign-in runs in a folder of its own with the ambient account cleared")
check((session_dir / ".kaggle" / "kaggle-session").exists(),
      "the session the CLI wrote is kept, under the folder for that account")
check(not secrets_file.exists()
      or "kaggle:browser-user" not in json.loads(secrets_file.read_text()),
      "a browser session is not copied into secrets.json in a shape we invented")

browser_env = cli_kaggle._account_environment("browser-user", login_file)
check(browser_env["HOME"] == str(session_dir)
      and browser_env["KAGGLE_CONFIG_DIR"] == str(session_dir / ".kaggle")
      and "KAGGLE_API_TOKEN" not in browser_env,
      "using a browser account points the CLI at its own session and adds nothing")

key_json = root / "downloaded-kaggle.json"
key_json.write_text(json.dumps({"username": "key-user", "key": "downloaded-key"}),
                    encoding="utf-8")
cli_kaggle.register_api_key_file(str(key_json), login_file)
key_data = config.load(login_file)
check("key-user" in key_data["kaggle"]["accounts"]
      and "downloaded-key" not in login_file.read_text()
      and json.loads(json.loads(
          secrets_file.read_text())["kaggle:key-user"])["key"] == "downloaded-key",
      "an API key file registers its own username and keeps the key out of the config")
check(cli_kaggle._account_environment("key-user", login_file)["KAGGLE_CONFIG_DIR"]
      != browser_env["KAGGLE_CONFIG_DIR"],
      "two registered accounts never share one credential folder")

pasted = json.dumps({"username": "pasted-user", "key": "pasted-key"})
cli_kaggle.register_api_key_file(pasted, login_file)
check("pasted-user" in config.load(login_file)["kaggle"]["accounts"],
      "the contents of kaggle.json can be pasted instead of a path")

rejected = False
try:
    cli_kaggle.register_api_key_file("{\"username\": \"no-key\"}", login_file)
except RuntimeError:
    rejected = True
check(rejected and "no-key" not in config.load(login_file)["kaggle"]["accounts"],
      "a file without a key is refused instead of registering half an account")

# What kaggle.com/settings hands out is an access token, and the legacy field
# of kaggle.json cannot carry it: measured against a real credential, that file
# answered "Authentication required to call the Kaggle API" while the identical
# value in access_token authenticated. Read from the CLI 2.2.4 source,
# `authenticate` tries the access token first and ignores one that fails
# introspection, falling through to the legacy key, so the same value is offered
# to both mechanisms and the CLI decides which one it is.
key_kaggle_dir = cli_kaggle._account_dir("key-user2", login_file) / ".kaggle"
cli_kaggle.register_api_key_file(
    json.dumps({"username": "key-user2", "key": "downloaded-key"}), login_file)
cli_kaggle._account_environment("key-user2", login_file)
check((key_kaggle_dir / "access_token").read_text(encoding="utf-8").strip()
      == "downloaded-key"
      and json.loads((key_kaggle_dir / "kaggle.json").read_text())
      == {"username": "key-user2", "key": "downloaded-key"}
      and (key_kaggle_dir / "access_token").stat().st_mode & 0o077 == 0,
      "a credential from kaggle.json is offered to both mechanisms, and stays private")

# A bare token carries no username, and it does not have to: the CLI introspects
# it against Kaggle and answers with the account, which is the same rule the
# browser sign-in follows. Typing a name would be inventing the one fact that
# decides whose quota gets spent.
token_commands = []


def token_run(command, check=False, capture_output=False, text=False, env=None,
              **kwargs):
    parts = list(map(str, command))
    token_commands.append((parts, dict(env or {})))
    if parts[1:3] == ["config", "view"]:
        token = (Path(env["KAGGLE_CONFIG_DIR"]) / "access_token").read_text().strip()
        if token != "KGAT_real":
            return SimpleNamespace(returncode=1, stdout="", stderr="not authenticated")
        return SimpleNamespace(
            returncode=0,
            stdout="- username: token-user\n- auth_method: ACCESS_TOKEN\n", stderr="")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


registered_token = cli_kaggle.register_api_key_file(
    "KGAT_real", login_file, Path("/fake/kaggle"), token_run)
token_accounts = config.load(login_file)["kaggle"]["accounts"]
check(registered_token == "token-user" and "token-user" in token_accounts
      and json.loads(json.loads(secrets_file.read_text())[
          "kaggle:token-user"])["token"] == "KGAT_real"
      and "KGAT_real" not in login_file.read_text(),
      "a pasted access token registers under the account Kaggle says it belongs to")

token_refused = False
try:
    cli_kaggle.register_api_key_file(
        "KGAT_wrong", login_file, Path("/fake/kaggle"), token_run)
except RuntimeError:
    token_refused = True
pending_left = [path for path in cli_kaggle._accounts_root(login_file).iterdir()
                if path.name.startswith("pending-")]
check(token_refused and not pending_left
      and len(config.load(login_file)["kaggle"]["accounts"]) == len(token_accounts),
      "a token Kaggle does not recognise registers nothing and leaves no folder behind")

# Signing out is local unless revoking was asked for. Revoking cannot be undone
# from here, so it must never be the thing that happens by default.
revoke_commands = []


def revoke_run(command, check=False, capture_output=False, text=False, env=None,
               **kwargs):
    revoke_commands.append(list(map(str, command)))
    return SimpleNamespace(returncode=0, stdout="", stderr="")


with redirect_stdout(io.StringIO()):
    cli_kaggle.forget_account("key-user", login_file, Path("/fake/kaggle"), revoke_run)
after_forget = config.load(login_file)
check("key-user" not in after_forget["kaggle"]["accounts"]
      and "kaggle:key-user" not in json.loads(secrets_file.read_text())
      and not cli_kaggle._account_dir("key-user", login_file).exists(),
      "signing out removes the account, its credential and its folder")
check(not revoke_commands,
      "signing out does not revoke the token at Kaggle unless that was asked for")

with redirect_stdout(io.StringIO()):
    cli_kaggle.forget_account(
        "browser-user", login_file, Path("/fake/kaggle"), revoke_run, revoke=True)
check(any(command[1:] == ["auth", "revoke"] for command in revoke_commands)
      and not session_dir.exists()
      and "browser-user" not in config.load(login_file)["kaggle"]["accounts"],
      "an explicitly confirmed sign-out revokes the token and clears the session")

ghost_file = root / "ghost" / "config.json"
config.save({"language": "en", "profiles": {}, "default_profile": None}, ghost_file)
with redirect_stdout(io.StringIO()):
    cli_kaggle.login_account(Path("/fake/kaggle"), ghost_file, login_run)
shutil.rmtree(cli_kaggle._account_dir("browser-user", ghost_file))
ghost_error = None
try:
    cli_kaggle._account_environment("browser-user", ghost_file)
except RuntimeError as error:
    ghost_error = str(error)
check(ghost_error and "browser-user" in ghost_error,
      "a browser account whose session is gone says so instead of using another one")

# The username is never typed. If any of this asked for input, the flow would
# have raised, because the input function used above refuses to answer.
with redirect_stdout(io.StringIO()):
    silent = cli_kaggle.login_account(
        Path("/fake/kaggle"), root / "silent" / "config.json", login_run)
check(silent == "browser-user", "signing in never asks the user for a username")
del refuse_input

# A notebook that never opened a session answers 404 on GetKernelSessionStatus.
# Treating that as fatal aborted the whole flow against a real account that had
# such notebooks, before it could list anything at all.
def dormant_run(command, check=False, capture_output=False, text=False, env=None,
                **kwargs):
    parts = list(map(str, command))
    if parts[1:3] == ["kernels", "list"]:
        return SimpleNamespace(
            returncode=0,
            stdout="ref,title\nuser/dormant,Dormant\nuser/running,Running\n",
            stderr="")
    if parts[1:3] == ["kernels", "status"]:
        if parts[3] == "user/dormant":
            return SimpleNamespace(
                returncode=1, stdout="",
                stderr="404 Client Error: Not Found for url: "
                       "https://api.kaggle.com/v1/kernels.KernelsApiService/"
                       "GetKernelSessionStatus")
        return SimpleNamespace(
            returncode=0, stdout="KernelWorkerStatus.RUNNING", stderr="")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


with redirect_stdout(io.StringIO()):
    dormant_live = cli_kaggle.live_kernels(Path("/fake/kaggle"), dormant_run)
check([ref for ref, _state in dormant_live] == ["user/running"],
      "a notebook with no session counts as not running, not as a failure")


def refused_run(command, check=False, capture_output=False, text=False, env=None,
                **kwargs):
    parts = list(map(str, command))
    if parts[1:3] == ["kernels", "list"]:
        return SimpleNamespace(returncode=0, stdout="ref,title\nuser/one,One\n", stderr="")
    return SimpleNamespace(returncode=1, stdout="", stderr="403 Forbidden")


# A kernel we cannot ask about might be spending quota right now, so anything
# that is not the session-less answer still stops the flow.
still_raises = False
try:
    with redirect_stdout(io.StringIO()):
        cli_kaggle.live_kernels(Path("/fake/kaggle"), refused_run)
except RuntimeError:
    still_raises = True
check(still_raises,
      "a status failure that is not a missing session still stops the flow")

# ----------------------------------------------------------------------
# The session lifecycle: what isaacli starts on Kaggle, isaacli ends.
#
# A kernel spends quota by wall clock until it is deleted, so leaving the
# stop to the user put the only brake on the spending outside the program.
# These drive the two entry points the CLI calls when it opens and when it
# closes, and they are judged by what reached the Kaggle CLI and what is left
# in the config, never by the message on screen.
# ----------------------------------------------------------------------
def refuse_typing(_prompt=""):
    raise AssertionError("a live session asked the user to type something")


def session_config(path, slug="owner/isaacli-gpu-1", extra_kernels=()):
    config.save({
        "language": "en",
        "default_profile": "kaggle-one",
        "profiles": {"kaggle-one": {
            "provider": "openai_compatible",
            "base_url": "https://one.trycloudflare.com/v1",
            "model": "qwen38-27b", "credential": "kaggle-one-api-key"}},
        "kaggle": {"kernels": [
            {"slug": slug, "url": "https://one.trycloudflare.com",
             "profile": "kaggle-one", "model": "qwen38-27b", "account": "owner"},
            *extra_kernels,
        ]},
    }, path)
    # `save_kaggle_profile` always stores the key it minted, and the liveness
    # probe now proves that key rather than only that the tunnel is up, so a
    # fixture without it would be describing a profile this program never writes.
    config.save_secret("kaggle-one-api-key", "session-key",
                       path.with_name("secrets.json"))
    cli_kaggle.register_account("owner", {"key": "owner-key"}, path)
    # register_account marks the account selected, which rewrites the file. The
    # kernels and the profile above have to survive that.
    return path


def answering_endpoint(_request, timeout=None):
    class Answer:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    return Answer()


def dead_endpoint(_request, timeout=None):
    raise OSError("no route to host")


live_file = session_config(root / "session-live" / "config.json")
with redirect_stdout(io.StringIO()):
    live_state = cli_kaggle.ensure_profile_session(
        "kaggle-one", input_fn=refuse_typing, config_file=live_file,
        urlopen_fn=answering_endpoint,
        run_kaggle_fn=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("a live kernel was relaunched")))
check(live_state == "live"
      and (config.load(live_file)["kaggle"]["kernels"] or [{}])[0]["slug"]
      == "owner/isaacli-gpu-1",
      "an endpoint that still answers is reused without pushing a kernel")

declined_file = session_config(root / "session-declined" / "config.json")
with redirect_stdout(io.StringIO()):
    declined_state = cli_kaggle.ensure_profile_session(
        "kaggle-one", input_fn=lambda _prompt: "n", config_file=declined_file,
        urlopen_fn=dead_endpoint,
        run_kaggle_fn=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("quota was spent without consent")))
declined_state_data = config.load(declined_file)
check(declined_state == "declined"
      and not (declined_state_data.get("kaggle") or {}).get("kernels")
      and "kaggle-one" not in declined_state_data["profiles"],
      "opening the program never pushes a kernel on its own, and forgets the dead one")

relaunch_file = session_config(root / "session-relaunch" / "config.json")
relaunched = []
with redirect_stdout(io.StringIO()):
    relaunch_state = cli_kaggle.ensure_profile_session(
        "kaggle-one", input_fn=lambda _prompt: "y", config_file=relaunch_file,
        urlopen_fn=dead_endpoint,
        run_kaggle_fn=lambda **kwargs: relaunched.append(kwargs) or 0)
check(relaunch_state == "relaunched" and len(relaunched) == 1
      and relaunched[0]["config_file"] == relaunch_file,
      "a dead kernel is replaced only after the user says so")

end_file = session_config(
    root / "session-end" / "config.json",
    extra_kernels=[{"slug": "owner/isaacli-gpu-2",
                    "url": "https://two.trycloudflare.com",
                    "profile": "kaggle-two", "model": "qwen38-27b",
                    "account": "owner"}])
end_commands = []


def end_run(command, check=False, capture_output=False, text=False, env=None,
            **kwargs):
    end_commands.append(list(map(str, command)))
    return SimpleNamespace(returncode=0, stdout="", stderr="")


with redirect_stdout(io.StringIO()):
    ended = cli_kaggle.stop_profile_session(
        "kaggle-one", config_file=end_file, run_fn=end_run,
        which_fn=lambda _name: "/fake/kaggle", home_dir=home)
end_state = config.load(end_file)
end_slugs = [item["slug"] for item in (end_state.get("kaggle") or {}).get("kernels", [])]
check(ended == "owner/isaacli-gpu-1"
      and [parts[1:] for parts in end_commands]
      == [["kernels", "delete", "owner/isaacli-gpu-1", "--yes"]]
      and end_slugs == ["owner/isaacli-gpu-2"]
      and "kaggle-one" not in end_state["profiles"],
      "closing the program ends its own kernel and leaves the other one alone")

quiet_commands = []
with redirect_stdout(io.StringIO()):
    nothing = cli_kaggle.stop_profile_session(
        None, config_file=end_file,
        run_fn=lambda command, **kwargs: quiet_commands.append(command),
        which_fn=lambda _name: "/fake/kaggle", home_dir=home)
check(nothing is None and not quiet_commands,
      "a run with no Kaggle kernel of its own never reaches Kaggle on the way out")

# The wiring itself, by effect: a run that opens a Kaggle profile has to end
# its kernel on the way out, whatever the answer to the question was.
wired_file = config.config_path()
session_config(wired_file)
wired_stops = []
original_ensure = cli._kaggle_ensure_session
original_stop = cli._kaggle_stop_session
try:
    cli._kaggle_ensure_session = lambda name, **kwargs: "live"
    cli._kaggle_stop_session = lambda name: wired_stops.append(name)
    with redirect_stdout(io.StringIO()):
        wired_code = cli.main(["say", "something"])
finally:
    cli._kaggle_ensure_session = original_ensure
    cli._kaggle_stop_session = original_stop
check(wired_code == 1 and wired_stops == ["kaggle-one"],
      "a run that opened a Kaggle profile ends that kernel when it closes")

# A delete that fails must not look like a stop that worked: the kernel is
# still spending, and the record has to stay so the next run can try again.
failed_file = session_config(root / "session-failed" / "config.json")


def failing_run(command, check=False, capture_output=False, text=False, env=None,
                **kwargs):
    return SimpleNamespace(returncode=1, stdout="", stderr="403 Forbidden")


with redirect_stdout(io.StringIO()) as failed_output:
    failed_stop = cli_kaggle.stop_profile_session(
        "kaggle-one", config_file=failed_file, run_fn=failing_run,
        which_fn=lambda _name: "/fake/kaggle", home_dir=home)
check(failed_stop is None
      and "--stop" in failed_output.getvalue()
      and (config.load(failed_file)["kaggle"]["kernels"] or [{}])[0]["slug"]
      == "owner/isaacli-gpu-1",
      "a failed stop says so, keeps the record, and names the command that retries")

# Measured against llama-server b10502 started with --api-key: /v1/models answers
# 200 with no credential at all, while /props answers 401 without the key and 200
# with it. Probing /v1/models therefore proves the tunnel is up and proves
# nothing about our key, so a profile holding the wrong key was reactivated as
# live and only failed at the first real question.
probe_file = root / "probe" / "config.json"
config.save({
    "language": "en",
    "profiles": {"kaggle-probe": {
        "provider": "openai_compatible", "provider_name": "Kaggle",
        "base_url": "https://probe.trycloudflare.com/v1",
        "model": "qwen38-27b", "credential": "probe-key",
    }},
}, probe_file)
config.save_secret("probe-key", "wrong", probe_file.with_name("secrets.json"))
probe_paths = []


def probe_urlopen(request, timeout=0):
    probe_paths.append(request.full_url)
    authorized = request.headers.get("Authorization") == "Bearer right"
    if request.full_url.endswith("/v1/models"):
        return HealthyAnswer()
    if authorized:
        return HealthyAnswer()
    raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)


probe_profile = config.load(probe_file)["profiles"]["kaggle-probe"]
with redirect_stdout(io.StringIO()):
    wrong_key_live = cli_kaggle._endpoint_answers(
        probe_profile, probe_file.with_name("secrets.json"), probe_urlopen)
check(not wrong_key_live,
      "a saved endpoint holding the wrong key is not reported as live")
check(probe_paths and not any(path.endswith("/v1/models") for path in probe_paths),
      "the liveness probe uses a route the server refuses without the key")

# Changing model while a kernel is serving used to be impossible from inside the
# program: the flow reactivated the saved profile and returned before the model
# list was ever drawn, so the only way through was to guess that
# `isaacli kaggle --stop` had to be run first.
def switch_config(path):
    config.save({
        "language": "en", "default_profile": "kaggle-live",
        "profiles": {"kaggle-live": {
            "provider": "openai_compatible", "provider_name": "Kaggle",
            "base_url": "https://switch.trycloudflare.com/v1",
            "model": "qwen38-27b", "credential": "switch-key"}},
        "kaggle": {"kernels": [{
            "slug": "user/isaacli-gpu-live",
            "url": "https://switch.trycloudflare.com",
            "profile": "kaggle-live", "model": "qwen38-27b", "account": "user"}]},
    }, path)
    config.save_secret("switch-key", "k", path.with_name("secrets.json"))
    cli_kaggle.register_account("user", {"key": "account-key"}, path)
    return path


def switch_run_factory(commands):
    def run(command, check=False, capture_output=False, text=False, env=None,
            **kwargs):
        commands.append(list(map(str, command)))
        joined = " ".join(map(str, command))
        if " quota" in joined:
            return SimpleNamespace(returncode=0, stdout=REAL_QUOTA_TABLE, stderr="")
        if "kernels list" in joined:
            return SimpleNamespace(
                returncode=0,
                stdout="ref,title\nuser/isaacli-gpu-live,Live\n", stderr="")
        if "kernels status" in joined:
            return SimpleNamespace(
                returncode=0, stdout="KernelWorkerStatus.RUNNING", stderr="")
        if "config view" in joined:
            return SimpleNamespace(returncode=0, stdout="username: user\n", stderr="")
        if "datasets list" in joined:
            return SimpleNamespace(returncode=0, stdout="ref,title\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return run


switch_screens = []
original_switch_select = cli_kaggle.terminal_ui.select


def recording_select(title, options, **kwargs):
    # Only the switch screen is being answered here. Every other screen keeps
    # its real behaviour, otherwise this would be answering the account picker
    # too and testing something nobody wrote.
    if "serving" not in title:
        return original_switch_select(title, options, **kwargs)
    switch_screens.append((title, list(options), kwargs.get("disabled")))
    return switch_choice[0]


keep_file = switch_config(root / "switch-keep" / "config.json")
keep_commands = []
switch_choice = [0]
try:
    cli_kaggle.terminal_ui.select = recording_select
    with redirect_stdout(io.StringIO()):
        keep_code = cli_kaggle.run_kaggle(
            input_fn=lambda _prompt: "1", run_fn=switch_run_factory(keep_commands),
            which_fn=lambda _name: "/fake/kaggle", config_file=keep_file,
            home_dir=home, urlopen_fn=lambda _r, timeout=0: HealthyAnswer())
finally:
    cli_kaggle.terminal_ui.select = original_switch_select
check(keep_code == 0
      and not any("kernels delete" in " ".join(c) for c in keep_commands)
      and not any("kernels push" in " ".join(c) for c in keep_commands)
      and switch_screens
      and config.load(keep_file)["default_profile"] == "kaggle-live",
      "a live kernel offers a choice, and keeping it spends nothing")

replace_file = switch_config(root / "switch-replace" / "config.json")
replace_commands = []
switch_screens.clear()
switch_choice = [1]
pushed_models = []
original_select_model = cli_kaggle._select_model
try:
    cli_kaggle.terminal_ui.select = recording_select
    cli_kaggle._select_model = lambda _input, catalog_path=None, prepared_fn=None: (
        pushed_models.append(prepared_fn) or {
            "repo": "org/Repo-GGUF", "file": "Model-Q4_K_M.gguf",
            "alias": "model-q4-k-m", "name": "Model", "model_bytes": 10 * 1024 ** 3,
            "machine_shape": "NvidiaTeslaT4", "cuda_arch": "75", "gpu_count": 2})
    with redirect_stdout(io.StringIO()) as replace_output:
        replace_code = cli_kaggle.run_kaggle(
            # Only the push is being answered yes here. Preparing assets is a
            # different question with its own cost, and answering it by accident
            # would send a CPU kernel this check never meant to send.
            input_fn=lambda prompt: (
                "y" if "Push" in prompt
                else "1" if "Select" in prompt
                else "n"),
            run_fn=switch_run_factory(replace_commands),
            which_fn=lambda _name: "/fake/kaggle", config_file=replace_file,
            home_dir=home, urlopen_fn=lambda _r, timeout=0: HealthyAnswer(),
            popen_fn=lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("tunnel discovery not exercised here")))
finally:
    cli_kaggle.terminal_ui.select = original_switch_select
    cli_kaggle._select_model = original_select_model
check(any("kernels delete" in " ".join(c) for c in replace_commands)
      and any("kernels push" in " ".join(c) for c in replace_commands)
      and pushed_models,
      "choosing another model ends the live kernel and offers the model list")

# Ending it would take the session away from a window that is still using it, so
# that option is shown and refused rather than quietly missing.
held_file = switch_config(root / "switch-held" / "config.json")
held_commands = []
switch_screens.clear()
switch_choice = [0]
held_process = subprocess.Popen(["sleep", "120"])
try:
    cli_kaggle.hold_profile_session(
        "kaggle-live", config_file=held_file, pid=held_process.pid)
    cli_kaggle.terminal_ui.select = recording_select
    with redirect_stdout(io.StringIO()):
        cli_kaggle.run_kaggle(
            input_fn=lambda _prompt: "1", run_fn=switch_run_factory(held_commands),
            which_fn=lambda _name: "/fake/kaggle", config_file=held_file,
            home_dir=home, urlopen_fn=lambda _r, timeout=0: HealthyAnswer())
finally:
    cli_kaggle.terminal_ui.select = original_switch_select
    held_process.kill()
    held_process.wait()
check(switch_screens and switch_screens[0][2] == {1},
      "replacing is refused while another window is using that kernel")

# The kernel is a Python file this program writes and Kaggle runs on the user's
# own account, and every marker lands inside a bare double-quoted literal. The
# repository name and the selector a user types are checked, but the file name
# is whatever the repository publishes: `siblings` comes from Hugging Face and
# only had to end in .gguf. A name carrying a quote or a newline leaves the
# string and becomes code. Proven by parsing what gets written, not by reading a
# refusal message.
import ast as kernel_ast

hostile_values = [
    ("file", 'Model".gguf'),
    ("file", 'Model\n__import__("os").system("id")\n#.gguf'),
    ("repo", 'org/Repo"+__import__("os").popen("id").read()+"'),
    ("alias", 'alias"\nSTOLEN = 1\n#'),
]
escaped = []
for field, value in hostile_values:
    model = {
        "repo": "org/Repo-GGUF", "file": "Model-Q4_K_M.gguf",
        "alias": "model-q4-k-m", "machine_shape": "NvidiaTeslaT4",
        "cuda_arch": "75", "gpu_count": 2,
    }
    model[field] = value
    folder = Path(tempfile.mkdtemp())
    try:
        cli_kaggle._render_kernel(
            folder, "user/isaacli-gpu-x", model, "api-key", dataset_sources=[])
    except (RuntimeError, ValueError):
        continue
    rendered = (folder / "isaacli-gpu-x.py").read_text(encoding="utf-8")
    try:
        tree = kernel_ast.parse(rendered)
    except SyntaxError:
        # Leaving the literal at all is the escape. Whether the result happens
        # to parse is the attacker's problem, not evidence of containment.
        escaped.append((field, value))
        continue
    planted = [
        node for node in kernel_ast.walk(tree)
        if isinstance(node, kernel_ast.Call)
        and isinstance(node.func, kernel_ast.Name)
        and node.func.id == "__import__"
    ] + [
        node for node in kernel_ast.walk(tree)
        if isinstance(node, kernel_ast.Assign)
        and any(isinstance(target, kernel_ast.Name) and target.id == "STOLEN"
                for target in node.targets)
    ]
    if planted:
        escaped.append((field, value))
check(not escaped,
      "a hostile file or repository name cannot become code inside the kernel")

# And the honest values still render, otherwise the guard has just broken the
# feature it was protecting.
plain_folder = Path(tempfile.mkdtemp())
cli_kaggle._render_kernel(
    plain_folder, "user/isaacli-gpu-ok",
    {"repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
     "file": "Q4_K_M/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
     "alias": "qwen3-coder-30b-a3b", "machine_shape": "NvidiaTeslaT4",
     "cuda_arch": "75", "gpu_count": 2},
    "kkQ9-_ab", dataset_sources=[])
plain = (plain_folder / "isaacli-gpu-ok.py").read_text(encoding="utf-8")
check("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf" in plain
      and kernel_ast.parse(plain) is not None,
      "an ordinary repository, subfolder and key still render into a valid kernel")

# The tunnel URL dies with the kernel, so the profile built from it is thrown
# away and every launch asked for the account, the model and the exact file
# again. What the user chose is durable even though the session is not, so it is
# kept apart from it and offered back as one keypress.
preference_file = root / "preference" / "config.json"
config.save({"language": "en"}, preference_file)
cli_kaggle.register_account("user", {"key": "account-key"}, preference_file)
cli_kaggle.save_kaggle_profile(
    "https://pref.trycloudflare.com", "user/isaacli-gpu-pref",
    {"alias": "model-q4-k-m", "name": "Model, Q4_K_M", "repo": "org/Repo-GGUF",
     "file": "Model-Q4_K_M.gguf", "model_bytes": 10 * 1024 ** 3,
     "machine_shape": "NvidiaTeslaT4", "cuda_arch": "75", "gpu_count": 2},
    "api-key", preference_file, account="user")
stored_preference = (config.load(preference_file).get("kaggle") or {}).get(
    "preference") or {}
check(stored_preference.get("account") == "user"
      and (stored_preference.get("model") or {}).get("file") == "Model-Q4_K_M.gguf"
      and (stored_preference.get("model") or {}).get("machine_shape")
      == "NvidiaTeslaT4",
      "what was chosen is remembered even though the session it ran in is not")

# The kernel is gone, so the record and the profile went with it. The preference
# has to survive that, because it is the answer to a different question.
cli_kaggle._forget_kernel_record("user/isaacli-gpu-pref", preference_file)
survived = (config.load(preference_file).get("kaggle") or {}).get("preference")
check(survived and survived.get("account") == "user",
      "forgetting a dead kernel does not forget what the user chose")

repeat_commands = []
repeat_screens = []


def preference_run(command, check=False, capture_output=False, text=False,
                   env=None, **kwargs):
    repeat_commands.append(list(map(str, command)))
    joined = " ".join(map(str, command))
    if " quota" in joined:
        return SimpleNamespace(returncode=0, stdout=REAL_QUOTA_TABLE, stderr="")
    if "kernels list" in joined or "datasets list" in joined:
        return SimpleNamespace(returncode=0, stdout="ref,title\n", stderr="")
    if "config view" in joined:
        return SimpleNamespace(returncode=0, stdout="username: user\n", stderr="")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


original_preference_select = cli_kaggle.terminal_ui.select


def preference_select(title, options, **kwargs):
    repeat_screens.append(title)
    return 0


try:
    cli_kaggle.terminal_ui.select = preference_select
    cli_kaggle._select_model = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("the model list was drawn again"))
    with redirect_stdout(io.StringIO()) as repeat_output:
        cli_kaggle.run_kaggle(
            input_fn=lambda prompt: "y" if "Push" in prompt else "n",
            run_fn=preference_run, which_fn=lambda _name: "/fake/kaggle",
            config_file=preference_file, home_dir=home,
            popen_fn=lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("tunnel discovery not exercised here")))
finally:
    cli_kaggle.terminal_ui.select = original_preference_select
    cli_kaggle._select_model = original_select_model
check(len(repeat_screens) == 1
      and any("kernels push" in " ".join(c) for c in repeat_commands)
      and "Model, Q4_K_M" in repeat_output.getvalue(),
      "repeating the last choice names it and asks once, not for account and model again")
check("12.18h" in repeat_output.getvalue(),
      "repeating still shows the quota before anything is spent")

# The quantization screen has to know which precision this account already has,
# because the alias is built from the file name: moving one row unhooks the
# prepared dataset and the kernel goes back to downloading the weight.
probe_calls = []


def dataset_listing_run(command, check=False, capture_output=False, text=False,
                        env=None, **kwargs):
    probe_calls.append(list(map(str, command)))
    return SimpleNamespace(
        returncode=0,
        stdout="ref,title\nowner/isaacli-model-qwen38-27b-ud-q4-k-m,Weight\n",
        stderr="")


weight_probe = cli_kaggle.prepared_weight_probe(
    "/fake/kaggle", "owner", dataset_listing_run, {})
check(weight_probe({"alias": "qwen38-27b-ud-q4-k-m", "cuda_arch": "75"})
      and not weight_probe({"alias": "qwen38-27b-ud-q8-0", "cuda_arch": "75"})
      and sum(1 for command in probe_calls if "datasets" in command) == 1,
      "the account is asked once which precisions it already has prepared")


def refusing_listing_run(command, check=False, capture_output=False, text=False,
                         env=None, **kwargs):
    return SimpleNamespace(returncode=1, stdout="", stderr="503")


refusing_probe = cli_kaggle.prepared_weight_probe(
    "/fake/kaggle", "owner", refusing_listing_run, {})
with redirect_stdout(io.StringIO()) as refusing_output:
    refused = refusing_probe({"alias": "qwen38-27b-ud-q4-k-m", "cuda_arch": "75"})
check(refused is False and refusing_output.getvalue() == "",
      "an account that cannot be asked answers unknown, quietly")

# Two isaacli windows on the same kernel. Reusing an endpoint that already
# answers costs nothing, so both of them do it, and the one that closes first
# used to delete the kernel underneath the other. Proven by effect: the delete
# command is not issued while somebody else still holds the record.
shared_file = session_config(root / "session-shared" / "config.json")
other_process = subprocess.Popen(["sleep", "120"])
shared_commands = []


def shared_run(command, check=False, capture_output=False, text=False, env=None,
               **kwargs):
    shared_commands.append(list(map(str, command)))
    return SimpleNamespace(returncode=0, stdout="", stderr="")


with redirect_stdout(io.StringIO()):
    cli_kaggle.hold_profile_session(
        "kaggle-one", config_file=shared_file, pid=other_process.pid)
    held_state = cli_kaggle.ensure_profile_session(
        "kaggle-one", input_fn=refuse_typing, config_file=shared_file,
        urlopen_fn=answering_endpoint,
        run_kaggle_fn=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("a live kernel was relaunched")))
with redirect_stdout(io.StringIO()) as shared_output:
    shared_stop = cli_kaggle.stop_profile_session(
        "kaggle-one", config_file=shared_file, run_fn=shared_run,
        which_fn=lambda _name: "/fake/kaggle", home_dir=home)
shared_state = config.load(shared_file)
check(held_state == "live" and shared_stop is None
      and not any("kernels delete" in " ".join(command)
                  for command in shared_commands)
      and (shared_state["kaggle"]["kernels"] or [{}])[0]["slug"]
      == "owner/isaacli-gpu-1",
      "closing one window leaves the kernel another window is still using")

# And the last one out does end it, otherwise the brake on quota is gone.
with redirect_stdout(io.StringIO()):
    cli_kaggle.release_profile_session(
        "kaggle-one", config_file=shared_file, pid=other_process.pid)
    last_stop = cli_kaggle.stop_profile_session(
        "kaggle-one", config_file=shared_file, run_fn=shared_run,
        which_fn=lambda _name: "/fake/kaggle", home_dir=home)
other_process.kill()
other_process.wait()
check(last_stop == "owner/isaacli-gpu-1"
      and any("kernels delete" in " ".join(command)
              for command in shared_commands)
      and not (config.load(shared_file)["kaggle"] or {}).get("kernels"),
      "the last window out ends the kernel and forgets it")

# A window that was killed rather than closed leaves its number behind. Holding
# the kernel forever because of a process that no longer exists would put the
# quota brake back out of reach.
stale_file = session_config(root / "session-stale" / "config.json")
stale_commands = []


def stale_run(command, check=False, capture_output=False, text=False, env=None,
              **kwargs):
    stale_commands.append(list(map(str, command)))
    return SimpleNamespace(returncode=0, stdout="", stderr="")


dead_process = subprocess.Popen(["sleep", "0.01"])
dead_pid = dead_process.pid
dead_process.wait()
with redirect_stdout(io.StringIO()):
    cli_kaggle.hold_profile_session(
        "kaggle-one", config_file=stale_file, pid=dead_pid)
    cli_kaggle.hold_profile_session("kaggle-one", config_file=stale_file)
    stale_stop = cli_kaggle.stop_profile_session(
        "kaggle-one", config_file=stale_file, run_fn=stale_run,
        which_fn=lambda _name: "/fake/kaggle", home_dir=home)
check(stale_stop == "owner/isaacli-gpu-1"
      and any("kernels delete" in " ".join(command) for command in stale_commands),
      "a window that was killed does not keep the kernel alive forever")

config.save_secret("probe-key", "right", probe_file.with_name("secrets.json"))
with redirect_stdout(io.StringIO()):
    right_key_live = cli_kaggle._endpoint_answers(
        probe_profile, probe_file.with_name("secrets.json"), probe_urlopen)
check(right_key_live,
      "a saved endpoint holding the right key is still reported as live")

# The dataset name is the only thing that tells two prepared weights apart on
# the account, and Kaggle caps it. Cutting from the end threw away the precision,
# which is what identifies the file, so two precisions of one repository landed
# on the same name and the second publication would sit on top of the first.
long_alias = "unsloth-qwen3-8-27b-gguf-qwen3-8-27b-ud-{}"
six = cli_kaggle._model_dataset_slug(long_alias.format("q6-k-l"))
eight = cli_kaggle._model_dataset_slug(long_alias.format("q8-k-l"))
check(six != eight and six.endswith("q6-k-l") and eight.endswith("q8-k-l")
      and max(len(six), len(eight)) <= cli_kaggle.DATASET_SLUG_LIMIT,
      "two precisions of one repository get two names, and both fit Kaggle's limit")
check(cli_kaggle._model_dataset_slug("short-one") == "isaacli-model-short-one",
      "a name that already fits is not mangled to make room it does not need")
catalogued = {"alias": "qwen3-coder-30b-a3b", "cuda_arch": "75"}
check(cli_kaggle._asset_refs("owner", catalogued)["model"]
      == "owner/isaacli-model-qwen3-coder-30b-a3b",
      "a catalogued alias keeps the exact name its dataset was published under")

if failures:
    print(f"\n{len(failures)} check(s) failed")
    raise SystemExit(1)
print("\nAll Kaggle checks passed")
