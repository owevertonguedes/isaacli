#!/usr/bin/env python3
"""Offline checks for live model discovery, exact resolution and fit gates."""
import io
import json
import sys
import tempfile
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

import model_discovery
import setup_ollama


failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


class FakeResponse:
    def __init__(self, payload=None, length=None, status=200):
        self.payload = payload
        self.status = status
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def read(self, _size=-1):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


GB = 1024 ** 3
calls = []
repos = {
    "test/MoE-30B-GGUF": {
        "id": "test/MoE-30B-GGUF", "downloads": 10,
        "cardData": {"base_model": "source/MoE-30B"},
        "siblings": [{"rfilename": "moe-Q4_K_M.gguf"}],
    },
    "test/Dense-20B-GGUF": {
        "id": "test/Dense-20B-GGUF", "downloads": 999999,
        "cardData": {"base_model": "source/Dense-20B"},
        "siblings": [{"rfilename": "dense-Q4_K_M.gguf"}],
    },
    "test/Huge-GGUF": {
        "id": "test/Huge-GGUF", "downloads": 1,
        "cardData": {"base_model": "source/Huge"},
        "siblings": [{"rfilename": "huge-Q4_K_M.gguf"}],
    },
}
configs = {
    "source/MoE-30B": {
        "num_hidden_layers": 48, "num_key_value_heads": 4,
        "head_dim": 128, "num_experts": 128, "num_experts_per_tok": 8,
    },
    "source/Dense-20B": {
        "num_hidden_layers": 40, "num_key_value_heads": 8, "head_dim": 128,
    },
    "source/Huge": {
        "num_hidden_layers": 64, "num_key_value_heads": 8, "head_dim": 128,
    },
}
sizes = {
    "moe-Q4_K_M.gguf": 18 * GB,
    "dense-Q4_K_M.gguf": 12 * GB,
    "huge-Q4_K_M.gguf": 40 * GB,
}


def fake_urlopen(request, timeout=None):
    url = request.full_url
    calls.append((url, request.get_method(), timeout))
    if url.startswith(model_discovery.HF_API + "?"):
        return FakeResponse([
            {"id": "test/Dense-20B-GGUF", "downloads": 999999},
            {"id": "test/MoE-30B-GGUF", "downloads": 10},
        ])
    prefix = model_discovery.HF_API + "/"
    if url.startswith(prefix):
        return FakeResponse(repos[url[len(prefix):]])
    if url.endswith("/config.json"):
        repo = url.split("huggingface.co/", 1)[1].split("/resolve/", 1)[0]
        return FakeResponse(configs[repo])
    if request.get_method() == "HEAD":
        return FakeResponse(length=sizes[url.rsplit("/", 1)[-1]])
    raise AssertionError(f"unexpected fake URL {url}")


catalog = setup_ollama.MODEL_CATALOG_PATH
discovered, discovery_errors = model_discovery.discover_models(
    catalog, urlopen_fn=fake_urlopen,
)
by_repo = {item["repo"]: item for item in discovered}
# The search answers Dense first and MoE second, and the MoE reads a tenth of
# the bytes per token. Ordering by that would put the cheap model on top, which
# is a recommendation whether it means to be one or not. Speed and capability
# point opposite ways here, and the ordering must not quietly pick speed.
check(not discovery_errors and [item["repo"] for item in discovered] == [
    "test/Dense-20B-GGUF", "test/MoE-30B-GGUF",
], "discovery keeps the source order instead of promoting the cheapest model")
check(by_repo["test/MoE-30B-GGUF"]["active_ratio"] == 8 / 128
      and by_repo["test/Dense-20B-GGUF"]["active_ratio"] == 1.0,
      "MoE active ratio comes from config.json while a dense model uses 1.0")
check(all(timeout and timeout <= model_discovery.DEFAULT_TIMEOUT
          for _url, _method, timeout in calls),
      "every Hugging Face request has a finite timeout")
check(by_repo["test/MoE-30B-GGUF"]["benchmark"] == model_discovery.NO_PUBLIC_SCORE
      and by_repo["test/MoE-30B-GGUF"]["benchmark_source"] is None,
      "an uncurated model reports no public accepted score instead of inventing one")


original_local_vram = model_discovery.local_vram
try:
    model_discovery.local_vram = lambda: (4096, 1)
    huge_out = io.StringIO()
    with redirect_stdout(huge_out):
        refused = setup_ollama._resolve_custom_ollama(
            "hf.co/test/Huge-GGUF:Q4_K_M", lambda _prompt: "no",
            catalog, fake_urlopen,
        )
    shown = huge_out.getvalue()
    check(refused is None and "Fits: no" in shown and "Weights 40.00 GiB" in shown
          and "KV cache at 16K" in shown and "total" in shown,
          "a named model that does not fit shows the numbers and is not installed")
finally:
    model_discovery.local_vram = original_local_vram


class FakeClient:
    def __init__(self):
        self.installed = []

    def models(self):
        return [{"name": item} for item in self.installed]

    def show(self, model):
        return {
            "capabilities": ["completion", "tools"],
            "model_info": {"test.context_length": 32768},
        }


original_client = setup_ollama.OllamaLocal
original_which = setup_ollama.shutil.which
original_server = setup_ollama._ensure_server
original_download = setup_ollama._download_model
original_other = setup_ollama._choose_other_ollama
downloads = []
client = FakeClient()
reference = "hf.co/test/MoE-30B-GGUF:Q4_K_M"
try:
    setup_ollama.OllamaLocal = lambda: client
    setup_ollama.shutil.which = lambda _name: "/usr/bin/ollama"
    setup_ollama._ensure_server = lambda *_args: ("test", None)

    def choose_other(input_fn, tr):
        model_discovery.local_vram = lambda: (32768, 2)
        return setup_ollama._resolve_custom_ollama(
            reference, input_fn, catalog, fake_urlopen,
        )

    def download(executable, model, tr=None):
        downloads.append((executable, model))
        client.installed.append(model)

    setup_ollama._choose_other_ollama = choose_other
    setup_ollama._download_model = download
    # Language, the onboarding task skipped, the "other model" entry, then the
    # download confirmation.
    answers = iter(["1", "4", "1", "7", "1", "1"])
    with redirect_stdout(io.StringIO()):
        code = setup_ollama.run_setup(
            lambda _prompt="": next(answers),
            config_file=Path(tempfile.mkdtemp()) / "config.json",
        )
    check(code == 0 and downloads == [("/usr/bin/ollama", reference)],
          "a fitting exact Ollama reference reaches ollama pull unchanged")
finally:
    model_discovery.local_vram = original_local_vram
    setup_ollama.OllamaLocal = original_client
    setup_ollama.shutil.which = original_which
    setup_ollama._ensure_server = original_server
    setup_ollama._download_model = original_download
    setup_ollama._choose_other_ollama = original_other


def offline(_request, timeout=None):
    raise urllib.error.URLError("network disabled by test")


offline_out = io.StringIO()
offline_answers = iter(["2"])
with redirect_stdout(offline_out):
    result = setup_ollama._choose_other_ollama(
        lambda _prompt="": next(offline_answers),
        setup_ollama.Translator("en"), catalog, offline,
    )
check(result is None and "network disabled by test" in offline_out.getvalue()
      and len(setup_ollama._recommended_catalog()) == 5,
      "offline discovery shows the cause while the versioned catalog still responds")


original_discover = model_discovery.discover_models
try:
    model_discovery.discover_models = lambda *_args, **_kwargs: ([], [])
    kaggle_answers = iter([
        "5",
        "https://huggingface.co/test/MoE-30B-GGUF/blob/main/moe-Q4_K_M.gguf",
    ])
    with redirect_stdout(io.StringIO()):
        kaggle_model = setup_ollama._dynamic_kaggle_selector(
            lambda _prompt="": next(kaggle_answers), catalog, fake_urlopen,
        )
    folder = Path(tempfile.mkdtemp())
    setup_ollama._render_dynamic_kaggle_kernel(
        folder, "tester/dynamic-model", kaggle_model, "secret",
    )
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    generated = (folder / "dynamic-model.py").read_text()
    check(kaggle_model["repo"] == "test/MoE-30B-GGUF"
          and kaggle_model["file"] == "moe-Q4_K_M.gguf"
          and kaggle_model["model_bytes"] == 18 * GB,
          "an exact Kaggle GGUF reaches the launch path with resolved values")
    check(metadata["machine_shape"] == "NvidiaTeslaT4"
          and metadata["dataset_sources"] == []
          and f'MODEL_REPO = "{kaggle_model["repo"]}"' in generated
          and f'MODEL_FILE = "{kaggle_model["file"]}"' in generated,
          "the dynamic Kaggle kernel is self-contained and uses the exact HF file URL")

    huge_answers = iter([
        "5",
        "https://huggingface.co/test/Huge-GGUF/blob/main/huge-Q4_K_M.gguf",
        "no",
    ])
    refused_kaggle = False
    try:
        with redirect_stdout(io.StringIO()):
            setup_ollama._dynamic_kaggle_selector(
                lambda _prompt="": next(huge_answers), catalog, fake_urlopen,
            )
    except RuntimeError:
        refused_kaggle = True
    check(refused_kaggle,
          "a Kaggle model that exceeds T4 x2 does not reach kernel rendering silently")
finally:
    model_discovery.discover_models = original_discover


# ----------------------------------------------------------------------
# Ordering by the ruler the declared task selects.
# ----------------------------------------------------------------------
kaggle_catalog = json.loads(Path(catalog).read_text(encoding="utf-8"))["kaggle"]
curated = [item["alias"] for item in kaggle_catalog]

check([item["alias"] for item in model_discovery.order_for_task(kaggle_catalog, None)]
      == curated,
      "no declared task leaves the curated order exactly as it is")
check([item["alias"] for item in
       model_discovery.order_for_task(kaggle_catalog, "unknown_task")] == curated,
      "a task with no ruler does not silently invent an ordering")

# Devstral scores 53.6 on SWE-bench Verified and Qwen3.8-27B scores 61.7 on
# SWE-bench Pro. Pro is the harder set, so ranking Verified above it because
# Verified is the preferred ruler would demote the stronger model while looking
# rigorous. Scores are only ever compared inside one ruler.
fix_order = [item["alias"] for item in
             model_discovery.order_for_task(kaggle_catalog, "fix_bug")]
check(fix_order.index("qwen38-27b") < fix_order.index("devstral-small-2507"),
      "scores from different rulers are not compared against each other")
check(fix_order[-1] == "qwen3-coder-30b-a3b",
      "a model with no score on the chosen ruler falls to the end, not out of the list")
check(sorted(fix_order) == sorted(curated),
      "ordering for a task keeps every model in the list")

build_order = [item["alias"] for item in
               model_discovery.order_for_task(kaggle_catalog, "build_new")]
check(build_order[:2] == ["qwen38-27b", "qwen3-30b-a3b-2507"],
      "build_new promotes the models scored on its rulers, best ruler first")

# Inside one ruler the number does decide, and the curated order does not
# override it: the lower score is listed first in the catalog on purpose here.
same_ruler = [
    {"alias": "weaker", "scores": {"swebench_verified": 20.0}},
    {"alias": "stronger", "scores": {"swebench_verified": 70.0}},
]
check([item["alias"] for item in
       model_discovery.order_for_task(same_ruler, "fix_bug")]
      == ["stronger", "weaker"],
      "two models on the same ruler are ordered by that ruler's score")

check(model_discovery.matched_ruler(
          {"scores": {"gpqa_diamond": 89.2}}, "explain_code") == "gpqa_diamond"
      and model_discovery.matched_ruler({"scores": {}}, "fix_bug") is None,
      "the ruler a model is judged by is reported, or None when it has no score")


# ----------------------------------------------------------------------
# A modified build does not inherit the score of the model it was built from.
#
# Seen on screen against the live API: an uncensored rebuild of Qwen3.8-27B
# declared the official model as its base and was listed with the official
# model's LiveCodeBench, GPQA and SWE-bench numbers, none of which had ever
# been measured on it.
# ----------------------------------------------------------------------
catalogued = json.loads(Path(catalog).read_text(encoding="utf-8"))["kaggle"][0]
official_gguf = catalogued["repo"]
official_upstream = "/".join(
    catalogued["benchmark_source"].split("huggingface.co/")[1].split("/")[:2])
derivative = f"someone/{official_upstream.split('/')[-1]}-Uncensored-GGUF"
attribution_repos = {
    official_gguf: {
        "id": official_gguf,
        "cardData": {"base_model": official_upstream},
        "siblings": [{"rfilename": "official-Q4_K_M.gguf"}],
    },
    derivative: {
        "id": derivative,
        "cardData": {"base_model": official_upstream},
        "siblings": [{"rfilename": "derivative-Q4_K_M.gguf"}],
    },
}


def attribution_urlopen(request, timeout=None):
    url = request.full_url
    prefix = model_discovery.HF_API + "/"
    if url.startswith(prefix):
        return FakeResponse(attribution_repos[url[len(prefix):]])
    if url.endswith("/config.json"):
        return FakeResponse({
            "num_hidden_layers": 64, "num_key_value_heads": 4, "head_dim": 256,
        })
    if request.get_method() == "HEAD":
        return FakeResponse(length=16 * GB)
    raise AssertionError(f"unexpected fake URL {url}")


official = model_discovery.resolve_hf_model(
    official_gguf, catalog_path=catalog, urlopen_fn=attribution_urlopen)
modified = model_discovery.resolve_hf_model(
    derivative, catalog_path=catalog, urlopen_fn=attribution_urlopen)
check(official["scores"] == catalogued["scores"] and official["scores"],
      "a plain requantization of the catalogued model keeps its published score")
check(modified["scores"] == {}
      and modified["benchmark"] == model_discovery.NO_PUBLIC_SCORE,
      "a modified build reports no public score instead of borrowing the original's")
check(modified["n_layers"] == official["n_layers"],
      "the modified build still reads its geometry from the base architecture")
check(model_discovery._is_plain_quantization("unsloth/Model-X-GGUF", "org/Model-X")
      and not model_discovery._is_plain_quantization(
          "someone/Model-X-abliterated-GGUF", "org/Model-X")
      and not model_discovery._is_plain_quantization("unsloth/Model-X-GGUF", None),
      "only a name that adds nothing but the format counts as the same weights")


print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("ISAAC MODEL DISCOVERY OK: offline seed, metadata, fit gates and ordering")
