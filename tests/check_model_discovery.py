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
check(not discovery_errors and [item["repo"] for item in discovered] == [
    "test/MoE-30B-GGUF", "test/Dense-20B-GGUF",
], "discovery sorts by bytes read per token, putting the larger MoE first")
check(discovered[0]["active_ratio"] == 8 / 128
      and discovered[1]["active_ratio"] == 1.0,
      "MoE active ratio comes from config.json while a dense model uses 1.0")
check(all(timeout and timeout <= model_discovery.DEFAULT_TIMEOUT
          for _url, _method, timeout in calls),
      "every Hugging Face request has a finite timeout")
check(discovered[0]["benchmark"] == model_discovery.NO_PUBLIC_SCORE
      and discovered[0]["benchmark_source"] is None,
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
    answers = iter(["1", "1", "7", "1", "1"])
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
          and len(metadata["dataset_sources"]) == 1
          and kaggle_model["file_url"] in generated,
          "the dynamic Kaggle kernel uses the T4 binary and the exact HF file URL")

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


print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("ISAAC MODEL DISCOVERY OK: offline seed, metadata, fit gates and ordering")
