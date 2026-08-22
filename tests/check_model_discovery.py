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
    # The catalogued Kaggle model, so the screens that offer its other
    # precisions can be exercised against the shelf it really has.
    "unsloth/Qwen3.8-27B-GGUF": {
        "id": "unsloth/Qwen3.8-27B-GGUF", "downloads": 500,
        "cardData": {"base_model": "Qwen/Qwen3.8-27B"},
        "siblings": [
            {"rfilename": "Qwen3.8-27B-UD-Q4_K_M.gguf"},
            {"rfilename": "Qwen3.8-27B-UD-IQ4_XS.gguf"},
            {"rfilename": "Qwen3.8-27B-BF16-00001-of-00003.gguf"},
        ],
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
    "Qwen/Qwen3.8-27B": {
        "num_hidden_layers": 64, "num_key_value_heads": 4, "head_dim": 256,
    },
}
sizes = {
    "moe-Q4_K_M.gguf": 18 * GB,
    "dense-Q4_K_M.gguf": 12 * GB,
    "huge-Q4_K_M.gguf": 40 * GB,
    "Qwen3.8-27B-UD-Q4_K_M.gguf": 16464440224,
    "Qwen3.8-27B-UD-IQ4_XS.gguf": 14 * GB,
    "Qwen3.8-27B-BF16-00001-of-00003.gguf": 18 * GB,
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
check(by_repo["test/MoE-30B-GGUF"]["benchmark"] == ""
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


# A cause printed just before a screen is drawn lands on the page the alternate
# screen replaces, so it has to arrive on the screen itself. The recorder keeps
# the title the selector was handed, which is the only thing the user reads.
screens = []
original_ui_select = setup_ollama.terminal_ui.select
try:
    setup_ollama.terminal_ui.select = lambda title, options, **kwargs: (
        screens.append((title, options)) or len(options) - 1)
    offline_out = io.StringIO()
    with redirect_stdout(offline_out):
        result = setup_ollama._choose_other_ollama(
            lambda _prompt="": "", setup_ollama.Translator("en"), catalog, offline,
        )
    offline_title = screens[-1][0]

    # A single candidate that fails explains a shorter list and nothing else, so
    # it goes to --debug rather than onto a screen about choosing a model.
    original_discover_noise = model_discovery.discover_models
    try:
        model_discovery.discover_models = lambda *_a, **_k: (
            [{"name": "Something", "benchmark": "none", "model_bytes": GB}],
            ["test/Repo-GGUF: no Q4_K_M file was found"])
        with redirect_stdout(io.StringIO()) as quiet_out:
            setup_ollama._choose_other_ollama(
                lambda _prompt="": "", setup_ollama.Translator("en"), catalog, offline)
    finally:
        model_discovery.discover_models = original_discover_noise
finally:
    setup_ollama.terminal_ui.select = original_ui_select

check(result is None and "network disabled by test" in offline_title
      and len(setup_ollama._recommended_catalog()) == 5,
      "discovery that returned nothing puts the cause on the screen, not behind it")
check("Q4_K_M" not in screens[-1][0] and "Q4_K_M" not in quiet_out.getvalue()
      and len(screens[-1][1]) == 3,
      "one failed candidate explains a shorter list in --debug, not on the screen")


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

    # The Kaggle screen is the same screen as the local one and has to be drawn
    # the same way. Replacing the selector with a recorder is what proves it: a
    # screen that goes back to print plus input never reaches it. The declared
    # task decided this order, so the ruler it used belongs on it, and a list
    # ordered by an unstated rule reads as an arbitrary one.
    import cli_kaggle

    drawn = []
    original_select = cli_kaggle.terminal_ui.select
    try:
        cli_kaggle.terminal_ui.select = lambda title, options, **kwargs: (
            drawn.append((title, options)) or 0)
        with redirect_stdout(io.StringIO()):
            task_model = setup_ollama._dynamic_kaggle_selector(
                lambda _prompt="": "1", catalog, fake_urlopen,
                onboarding_task="fix_bug",
            )
    finally:
        cli_kaggle.terminal_ui.select = original_select
    ruler = model_discovery.text("onboarding.task.ruler.fix_bug")
    # Every row on this screen reads T4 x2, because that is what it assigns, so
    # the explanation above them must not promise a smaller accelerator.
    check(all(cli_kaggle.ACCELERATORS["NvidiaTeslaT4"]["label"] in option
              for option in drawn[0][1][:-1])
          and "P100" not in drawn[0][0],
          "the screen describes the accelerator it actually requests")
    check(all("\n" not in option for option in drawn[0][1])
          and drawn[0][1][-1] == model_discovery.text("model.discovery.exact"),
          "the Kaggle discovery screen is drawn by the shared selector too")
    # Choosing the model and choosing how much of it to keep are two questions.
    # The second one only exists because the repository publishes more than one
    # file, and a split weight is not one of the answers: its Content-Length is
    # one part, so offering it would misstate both fit and speed.
    check(len(drawn) == 2 and "IQ4_XS" in " ".join(drawn[1][1])
          and not any("00001-of-00003" in option for option in drawn[1][1])
          and task_model["file"] == "Qwen3.8-27B-UD-Q4_K_M.gguf",
          "the other precisions of the chosen model are offered, split files aside")
    check(ruler in drawn[0][0],
          "the screen names the public ruler that decided the order it shows")

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

# The local list is where the declared task was asked about and then did
# nothing, because all five entries carried an empty score. Asking a question,
# announcing the ruler and producing the identical list is worse than not
# asking. What is checked is the effect on the list the user actually sees.
local_catalog = json.loads(Path(catalog).read_text(encoding="utf-8"))["local"]
local_curated = [item["reference"] for item in local_catalog]


def local_order(task):
    return [item["reference"]
            for item in model_discovery.order_for_task(local_catalog, task)]


check(any(item["scores"] for item in local_catalog)
      and all(item["benchmark_source"] for item in local_catalog if item["scores"])
      and all(not item["scores"] for item in local_catalog
              if not item["benchmark_source"]),
      "a local model with a score cites where it comes from, and none has a loose number")
check(local_order(None) == local_curated
      and local_order("explain_code") != local_curated
      and local_order("fix_bug") != local_curated,
      "the declared task really reorders the local list, not only the Kaggle one")
# fix_bug and build_new coincide today, and that is a fact about the data rather
# than a broken ruler: the same model leads both, and the only model scored on
# Aider is behind both models scored on LiveCodeBench. Pinning it means a future
# catalogue edit that separates them is a visible change, not a silent one.
check(local_order("fix_bug") == local_order("build_new")
      and local_order("explain_code") != local_order("fix_bug"),
      "reading code selects a different local order than fixing or building does")

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
      and modified["benchmark"] == "",
      "a modified build reports no public score instead of borrowing the original's")
check(modified["n_layers"] == official["n_layers"],
      "the modified build still reads its geometry from the base architecture")
check(model_discovery._is_plain_quantization("unsloth/Model-X-GGUF", "org/Model-X")
      and not model_discovery._is_plain_quantization(
          "someone/Model-X-abliterated-GGUF", "org/Model-X")
      and not model_discovery._is_plain_quantization("unsloth/Model-X-GGUF", None),
      "only a name that adds nothing but the format counts as the same weights")


# ----------------------------------------------------------------------
# What a list of suggestions accepts.
#
# The first real use put `Qwen3.8-27B-Uncensored` and `-OBLITERATED` on a
# screen titled recommendation. The rule is not a list of forbidden words:
# a repository that declares a base model and is not simply a requantization
# of it is a changed model, so nothing published about the base describes it.
# It stays reachable by exact reference and leaves the suggestions.
# ----------------------------------------------------------------------
def suggestion_urlopen(request, timeout=None):
    url = request.full_url
    if url.startswith(model_discovery.HF_API + "?"):
        return FakeResponse([{"id": official_gguf}, {"id": derivative}])
    return attribution_urlopen(request, timeout)


suggested, suggestion_errors = model_discovery.discover_models(
    catalog, urlopen_fn=suggestion_urlopen)
with_derived, _errors = model_discovery.discover_models(
    catalog, urlopen_fn=suggestion_urlopen, include_derived=True)
check([item["repo"] for item in suggested] == [official_gguf]
      and any(derivative in message for message in suggestion_errors)
      and [item["repo"] for item in with_derived] == [official_gguf, derivative],
      "a modified build leaves the suggestions and its reason goes to --debug")
check(modified["derived"] and modified["derived_from"] == official_upstream
      and not official["derived"],
      "being a changed model is decided by the declared base, not by the name")
check(model_discovery.origin(official) == "curated"
      and model_discovery.origin(
          {"scores": {"swebench_verified": 1.0}}) == "scored"
      and model_discovery.origin({"scores": {}}) == "discovered",
      "every row can say whether it was reviewed, scored, or merely found")


# A gated repository answers its metadata and refuses its files. Finding that
# out after a kernel is up costs quota; finding it out here costs nothing.
def gated_urlopen(request, timeout=None):
    url = request.full_url
    if url.startswith(model_discovery.HF_API + "/"):
        return FakeResponse({
            "id": "org/Gated-GGUF", "gated": "auto",
            "siblings": [{"rfilename": "gated-Q4_K_M.gguf"}],
        })
    raise AssertionError("a gated repository was read past its metadata")


gated_error = None
try:
    model_discovery.resolve_hf_model(
        "org/Gated-GGUF", catalog_path=catalog, urlopen_fn=gated_urlopen)
except model_discovery.DiscoveryError as error:
    gated_error = str(error)
check(gated_error and "org/Gated-GGUF" in gated_error,
      "a gated repository is refused while it is still free to find out")

# The local discovery screen answers "what can I run", so it is drawn against
# this machine and the ones that fit come first.
fit_screens = []
original_local_vram = model_discovery.local_vram
original_ui = setup_ollama.terminal_ui.select
try:
    model_discovery.local_vram = lambda: (16384, 1)
    setup_ollama.terminal_ui.select = lambda title, options, **kwargs: (
        fit_screens.append(options) or len(options) - 1)
    with redirect_stdout(io.StringIO()):
        setup_ollama._choose_other_ollama(
            lambda _prompt="": "", setup_ollama.Translator("en"), catalog,
            fake_urlopen)
finally:
    model_discovery.local_vram = original_local_vram
    setup_ollama.terminal_ui.select = original_ui
rows = fit_screens[-1][:-2]
check(len(rows) == 2 and "Fits" in rows[0] and "Does not fit" in rows[1]
      and "found live" in rows[0],
      "the discovery screen says what fits this machine, fitting models first")

# Filling the accelerator is the point of borrowing it: the hour costs the same
# whether two thirds of the card sit idle or not. The heaviest that still leaves
# room comes first, and what is not a weight at all never appears.
shelf = {
    "org/Model-GGUF": {
        "id": "org/Model-GGUF",
        "siblings": [
            {"rfilename": "Model-Q4_K_M.gguf"},
            {"rfilename": "Model-Q6_K.gguf"},
            {"rfilename": "Model-Q6_K_L.gguf"},
            {"rfilename": "Model-Q8_0.gguf"},
            {"rfilename": "Model-BF16-00001-of-00002.gguf"},
            {"rfilename": "imatrix_unsloth.gguf"},
            {"rfilename": "mmproj-BF16.gguf"},
            {"rfilename": "MTP/mtp-Model-Q4_0.gguf"},
        ],
    },
}
shelf_sizes = {
    "Model-Q4_K_M.gguf": 10 * GB, "Model-Q6_K.gguf": 13 * GB,
    "Model-Q6_K_L.gguf": 14 * GB, "Model-Q8_0.gguf": 19 * GB,
    "imatrix_unsloth.gguf": GB // 100, "mmproj-BF16.gguf": GB,
    "MTP/mtp-Model-Q4_0.gguf": GB,
}


def shelf_urlopen(request, timeout=None):
    url = request.full_url
    prefix = model_discovery.HF_API + "/"
    if url.startswith(prefix):
        return FakeResponse(shelf[url[len(prefix):]])
    if url.endswith("/config.json"):
        return FakeResponse({
            "num_hidden_layers": 40, "num_key_value_heads": 8, "head_dim": 128,
        })
    if request.get_method() == "HEAD":
        name = url.split("/resolve/main/", 1)[1]
        return FakeResponse(length=shelf_sizes[name])
    raise AssertionError(f"unexpected fake URL {url}")


shelf_model = model_discovery.resolve_hf_model(
    "org/Model-GGUF", "Model-Q4_K_M.gguf", urlopen_fn=shelf_urlopen)
shelf_variants = model_discovery.quantization_variants(
    shelf_model, urlopen_fn=shelf_urlopen)
offered = sorted(item["file"] for item in shelf_variants)
check(offered == ["Model-Q6_K.gguf", "Model-Q6_K_L.gguf", "Model-Q8_0.gguf"],
      "only real precisions of this model are offered, not projectors or calibration")
check(model_discovery.quantization_label("Model-Q6_K.gguf") == "Q6_K"
      and model_discovery.quantization_label("Model-Q6_K_L.gguf") == "Q6_K_L"
      and model_discovery.quantization_label("m-UD-Q4_K_M.gguf") == "UD-Q4_K_M",
      "each precision reads as itself instead of collapsing onto a shorter name")

shelf_screens = []
original_shelf_ui = setup_ollama.terminal_ui.select
try:
    setup_ollama.terminal_ui.select = lambda title, options, **kwargs: (
        shelf_screens.append((options, kwargs.get("initial"))) or kwargs.get("initial", 0))
    with redirect_stdout(io.StringIO()):
        picked = setup_ollama._choose_quantization(
            shelf_model, lambda _prompt="": "", setup_ollama.Translator("en"),
            shelf_urlopen, vram_mb=16384, overhead_mb=768)
finally:
    setup_ollama.terminal_ui.select = original_shelf_ui
rows, cursor = shelf_screens[-1]
# 16 GiB minus overhead leaves 15.25 GiB. Q8_0 at 19 GiB does not fit; Q6_K_L at
# 14 GiB plus 2.5 GiB of cache does not either; Q6_K at 13 GiB is the heaviest
# that fits, and it still has to leave a tenth of the card free.
check([row.split(" ")[0] for row in rows]
      == ["Q8_0", "Q6_K_L", "Q6_K", "Q4_K_M"],
      "the precisions are drawn heaviest first, so the card gets filled")
check(picked["file"] == "Model-Q4_K_M.gguf" and cursor == 3,
      "the cursor starts on the heaviest precision that still leaves room")

# Picking a precision the account has no dataset for silently changes the alias,
# so the prepared input stops matching and the kernel goes back to downloading
# tens of gigabytes with nothing on screen saying so. The row that is already
# prepared says so and is where the cursor starts.
def quantization_screen(prepared_fn, vram_mb=24576, overhead_mb=768):
    screens = []
    original = setup_ollama.terminal_ui.select
    try:
        setup_ollama.terminal_ui.select = lambda title, options, **kwargs: (
            screens.append((options, kwargs.get("initial")))
            or kwargs.get("initial", 0))
        with redirect_stdout(io.StringIO()):
            picked = setup_ollama._choose_quantization(
                shelf_model, lambda _prompt="": "", setup_ollama.Translator("en"),
                shelf_urlopen, vram_mb=vram_mb, overhead_mb=overhead_mb,
                prepared_fn=prepared_fn)
    finally:
        setup_ollama.terminal_ui.select = original
    return picked, screens[-1][0], screens[-1][1]


# On 24 GiB the headroom rule alone would stop on Q6_K_L, so a cursor that lands
# on Q4_K_M is the prepared weight winning and not a coincidence of ordering.
_default_pick, _default_rows, default_cursor = quantization_screen(None)
prepared_pick, prepared_rows, prepared_cursor = quantization_screen(
    lambda item: item["file"] == "Model-Q4_K_M.gguf")
check(default_cursor == 1 and prepared_pick["file"] == "Model-Q4_K_M.gguf"
      and prepared_cursor == 3,
      "the cursor starts on the precision this account already has prepared")
check(sum(1 for row in prepared_rows if "prepared" in row) == 1
      and "prepared" in prepared_rows[3],
      "only the prepared precision is marked as prepared")

# A prepared weight that will not load is still worth marking and is the wrong
# place to leave the cursor: the launch would spend the hour and fail.
too_big_pick, too_big_rows, too_big_cursor = quantization_screen(
    lambda item: item["file"] == "Model-Q8_0.gguf", vram_mb=16384)
check(too_big_pick["file"] == "Model-Q4_K_M.gguf" and too_big_cursor == 3
      and "prepared" in too_big_rows[0],
      "a prepared precision that does not fit is marked but not preselected")

# A lookup that cannot answer is unknown, not an error: the screen still draws,
# and it falls back to the rule it had before anybody asked about datasets.
def refusing_lookup(_item):
    raise RuntimeError("kaggle is unreachable")


unknown_pick, unknown_rows, unknown_cursor = quantization_screen(
    refusing_lookup, vram_mb=16384)
check(unknown_pick["file"] == "Model-Q4_K_M.gguf" and unknown_cursor == 3
      and not any("prepared" in row for row in unknown_rows),
      "a lookup that fails leaves the screen unmarked instead of breaking it")

# Every scored row in the catalogue is a quantized GGUF whose number was measured
# on the original weights: all six `benchmark_source` values are the upstream
# model page. The sentence that says so was only printed after the choice, and
# the choice happens on the list, where the number sat on the row of a file
# nobody had scored. That is the same error as inheriting a score for a
# derivative build, and the rule against it does not care which direction the
# inheritance runs.
scored_row = model_discovery.benchmark_cell({"benchmark": "SWE-bench Verified 73.4"})
unscored_row = model_discovery.benchmark_cell({"benchmark": ""})
check("73.4" in scored_row and scored_row != "SWE-bench Verified 73.4",
      "a score on a row never appears without saying whose score it is")
check(unscored_row == model_discovery.no_public_score()
      and "(" not in unscored_row,
      "a model with no score is not given an owner it does not have")

catalogue = json.loads(
    (HERE.parent / "tool_harness" / "model_catalog.json").read_text(encoding="utf-8"))
unsourced = [item["name"] for section in catalogue.values() for item in section
             if item.get("benchmark") and not item.get("benchmark_source")]
check(not unsourced,
      "no row carries a number without a source to check it against"
      + (f" (found {', '.join(unsourced)})" if unsourced else ""))

# The Kaggle path used to hand `_choose_quantization` a translator built on the
# spot, which is always English, so this one screen came out in English inside a
# Portuguese session while every screen around it was translated. Comparing the
# two catalogs cannot see that: both catalogs have the key and both are correct.
# Only running the screen in a Portuguese session can, so that is what this does.
import cli_i18n
import cli_kaggle

kaggle_titles = []
original_select = setup_ollama.terminal_ui.select
original_choose = cli_kaggle._choose
original_candidates = cli_kaggle._load_model_candidates
original_discover = model_discovery.discover_models
try:
    cli_i18n.set_language("pt-BR")
    setup_ollama.terminal_ui.select = lambda title, options, **kwargs: (
        kaggle_titles.append(title) or kwargs.get("initial", 0))
    cli_kaggle._choose = lambda _title, _options, _input_fn: 0
    model_discovery.discover_models = lambda *args, **kwargs: ([], [])
    cli_kaggle._load_model_candidates = lambda *args, **kwargs: [shelf_model]

    with redirect_stdout(io.StringIO()):
        setup_ollama._dynamic_kaggle_selector(
            lambda _prompt="": "", urlopen_fn=shelf_urlopen)
finally:
    setup_ollama.terminal_ui.select = original_select
    cli_kaggle._choose = original_choose
    cli_kaggle._load_model_candidates = original_candidates
    model_discovery.discover_models = original_discover
    cli_i18n.set_language("en")

pt_title = setup_ollama.Translator("pt-BR").t("model.quantization.title")
en_title = setup_ollama.Translator("en").t("model.quantization.title")
check(kaggle_titles and pt_title in kaggle_titles[-1]
      and en_title not in kaggle_titles[-1] and pt_title != en_title,
      "the Kaggle quantization screen speaks the language the session chose")

# The same mistake anywhere else in the module would be just as invisible, so
# the rule is the guard: only cli_i18n decides what a bare translator is.
sources = sorted((HERE.parent / "tool_harness").glob("*.py"))
bare = [f"{path.name}:{number}" for path in sources if path.name != "cli_i18n.py"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "Translator()" in line]
check(not bare,
      "no module builds a language-less translator behind the session's back"
      + (f" (found {', '.join(bare)})" if bare else ""))

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("ISAAC MODEL DISCOVERY OK: offline seed, metadata, fit gates and ordering")
