#!/usr/bin/env python3
"""Tests for the guided setup, with no network, downloads or a real Ollama."""
import io
import json
import sys
import tempfile
import stat
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

import config
import model_discovery
import setup_ollama


failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


class FakeClient:
    def __init__(self, installed, infos):
        self.installed = installed
        self.infos = infos

    def models(self):
        return [{"name": name} for name in self.installed]

    def show(self, model):
        return self.infos[model]


def answers(*values):
    items = iter(values)
    return lambda _prompt="": next(items)


original_which = setup_ollama.shutil.which
original_client = setup_ollama.OllamaLocal
original_server = setup_ollama._ensure_server
original_download = setup_ollama._download_model
original_validate_api = setup_ollama._validate_api
original_list_api_models = setup_ollama._list_api_models

try:
    root = Path(tempfile.mkdtemp())
    config_file = root / "config.json"
    downloads = []
    qwen36 = "hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ1_M"
    infos = {
        qwen36: {
            "capabilities": ["completion", "tools", "thinking"],
            "model_info": {"qwen3.context_length": 262144},
        },
        "gpt-oss:20b": {
            "capabilities": ["completion", "tools", "thinking"],
            "model_info": {
                "gptoss.context_length": 131072,
                "gptoss.rope.scaling.original_context_length": 4096,
            },
        },
        "granite4:micro-h": {
            "capabilities": ["completion", "tools"],
            "model_info": {"granitehybrid.context_length": 1048576},
        },
    }
    client = FakeClient(
        [qwen36, "gpt-oss:20b", "granite4:micro-h", "test-model:7b"], infos,
    )
    setup_ollama.shutil.which = lambda _name: "/usr/bin/ollama"
    setup_ollama.OllamaLocal = lambda: client
    setup_ollama._ensure_server = lambda _client, _exe, _tr=None: ("test", None)
    setup_ollama._download_model = lambda exe, name, tr=None: downloads.append((exe, name))

    pt = setup_ollama.Translator("pt-BR")

    out = io.StringIO()
    with redirect_stdout(out):
        code = setup_ollama.run_setup(
            answers("1", "4", "1", "1", "6", "12K", "1"), config_file=config_file,
        )
    data = json.loads(config_file.read_text())
    qwen_profile = data["profiles"][data["default_profile"]]
    check(code == 0, "the recommended Qwen3.6 setup completes")
    check(qwen_profile["num_ctx"] == 12288, "manual mode accepts the friendly 12K context")
    check(qwen_profile["thinking"] == "low",
          "thinking is detected from the manifest, not from the catalog")
    check(data["language"] == "pt-BR", "setup saves the interface language")
    check(qwen_profile["model"] == qwen36,
          "the context lives in the profile without creating a derived model copy")
    check(not downloads, "an installed model is not downloaded again")
    recommended_menu = setup_ollama._recommended_catalog()
    local_menu = setup_ollama._installed_models(client.installed)
    check([item["base_model"] for item in recommended_menu] == setup_ollama.RECOMMENDED,
          "the recommendations section preserves the curation and its order")
    check(any(item["base_model"] == "test-model:7b" for item in local_menu),
          "the installed section includes models queried live from Ollama")
    check(any(item["base_model"] == "granite4:micro-h" for item in local_menu)
          and "granite4:micro-h" not in setup_ollama.RECOMMENDED,
          "Micro H shows up because it is installed, with no recommendation badge")
    check(pt.t("model.section.recommended") in out.getvalue()
          and pt.t("model.section.installed", count=len(local_menu)).split("(")[0]
          in out.getvalue()
          and "test-model:7b" in out.getvalue(),
          "the menu shows recommendations and every installed model on one screen")

    selector_config = root / "selector-config.json"
    client.installed.append("isaac-qwen-legacy-16k")
    selector_data = dict(config.empty_config(), language="pt-BR")
    selector_data["profiles"]["qwen-legacy-16k"] = {
        "provider": "ollama", "model": "isaac-qwen-legacy-16k",
        "base_model": qwen36, "num_ctx": 16384,
    }
    selector_data["default_profile"] = "qwen-legacy-16k"
    config.save(selector_data, selector_config)
    selector_out = io.StringIO()
    with redirect_stdout(selector_out):
        code = setup_ollama.run_model_selector(
            answers("1", "6", "1"), config_file=selector_config,
        )
    _, micro_profile = config.profile(config.load(selector_config))
    check(code == 0 and micro_profile["model"] == "granite4:micro-h",
          "/model finds Micro H in the live local list, not in the curation")
    check(micro_profile["num_ctx"] == 8192 and micro_profile["thinking"] is False,
          "/model asks for the context afterwards and detects the absence of thinking")
    check("isaac-qwen-legacy-16k" not in selector_out.getvalue(),
          "/model hides only the context copies the configuration recognises")
    client.installed.remove("isaac-qwen-legacy-16k")

    with redirect_stdout(io.StringIO()):
        code = setup_ollama.run_setup(
            answers("1", "4", "1", "5", "3", "3"), config_file=config_file,
        )
    data = config.load(config_file)
    gpt_profile = data["profiles"][data["default_profile"]]
    check(code == 0, "the GPT-OSS setup completes")
    check(gpt_profile["num_ctx"] == 32768, "the GPT-OSS long preset is 32K")
    check(gpt_profile["thinking"] == "high", "GPT-OSS saves thinking high separately")
    check(gpt_profile["temperature"] == 0,
          "setup does not inject a hardcoded GPT-OSS-specific tweak")
    check(len(data["profiles"]) == 2, "a new profile preserves the previous one")

    before_failure = config_file.read_text()
    original_qwen_info = client.infos[qwen36]
    client.infos[qwen36] = {
        "capabilities": ["completion"],
        "model_info": {"qwen3.context_length": 262144},
    }
    with redirect_stdout(io.StringIO()):
        code = setup_ollama.run_setup(answers("1", "4", "1", "1"), config_file=config_file)
    check(code == 1 and config_file.read_text() == before_failure,
          "a model without tools is refused without touching the previous profile")
    client.infos[qwen36] = original_qwen_info

    setup_ollama.shutil.which = lambda _name: None
    missing = root / "missing.json"
    with redirect_stdout(io.StringIO()):
        code = setup_ollama.run_setup(answers("1", "4", "1"), config_file=missing)
    check(code == 2 and not missing.exists(),
          "a missing Ollama gives instructions and writes no partial config")

    setup_ollama._validate_api = lambda url, key, model: None
    api_config = root / "api-config.json"
    with redirect_stdout(io.StringIO()):
        code = setup_ollama.run_setup(
            answers("1", "4", "2", "Groq", "https://api.groq.com/openai/v1",
                    "openai/gpt-oss-20b", "test-secret", "3"),
            config_file=api_config,
        )
    api_data = config.load(api_config)
    _, api_profile = config.profile(api_data)
    api_secret = config.load_secret(
        api_profile["credential"], api_config.with_name("secrets.json"))
    check(code == 0 and api_profile["provider"] == "openai_compatible",
          "setup creates a compatible API profile with no hardcoded provider")
    check(api_profile["base_url"] == "https://api.groq.com/openai/v1"
          and api_profile["model"] == "openai/gpt-oss-20b",
          "the API endpoint and model are configurable data")
    check(api_secret == "test-secret" and "test-secret" not in api_config.read_text(),
          "the API key stays out of config.json")
    check(stat.S_IMODE(api_config.with_name("secrets.json").stat().st_mode) == 0o600,
          "the secrets file uses 0600 permissions")

    setup_ollama._list_api_models = lambda base_url, api_key: (
        ["openai/gpt-oss-20b", "qwen/qwen3.6-27b"] if api_key == "test-secret" else []
    )
    with redirect_stdout(io.StringIO()):
        swap_result = setup_ollama._select_configured_api(
            answers("3", "2", "1"), api_config, "pt-BR", pt,
        )
    _, swapped_profile = config.profile(config.load(api_config))
    check(swap_result == 0 and swapped_profile["model"] == "qwen/qwen3.6-27b",
          "swapping the model of a configured API uses the live list, without redoing setup")
    check(swapped_profile["base_url"] == "https://api.groq.com/openai/v1"
          and swapped_profile["credential"] == api_profile["credential"],
          "swapping the model preserves the saved endpoint and credential")
    setup_ollama._list_api_models = original_list_api_models

    attempts = []

    def validate_on_second(url, key, model):
        attempts.append((url, key, model))
        if len(attempts) == 1:
            raise RuntimeError("HTTP 401: invalid key")

    setup_ollama._validate_api = validate_on_second
    api_retry_config = root / "api-retry-config.json"
    with redirect_stdout(io.StringIO()):
        code = setup_ollama.run_setup(
            answers(
                "1", "4", "2", "Server", "https://api.test/v1/chat/completions",
                "test-model", "wrong-key", "1",
                "Server", "https://api.test/v1", "test-model", "right-key", "1",
            ),
            config_file=api_retry_config,
        )
    _, retry_profile = config.profile(config.load(api_retry_config))
    check(code == 0 and len(attempts) == 2,
          "a validation failure lets you fix the data without restarting setup")
    check(retry_profile["base_url"] == "https://api.test/v1",
          "a full endpoint is normalized before being saved")
    setup_ollama._validate_api = lambda url, key, model: None

    setup_ollama.shutil.which = lambda _name: "/usr/bin/ollama"
    back_config = root / "back-config.json"
    with redirect_stdout(io.StringIO()):
        code = setup_ollama.run_setup(
            answers("1", "4", "1", "8", "4", "2", "Server", "https://api.test/v1",
                    "test-model", "key", "1"),
            config_file=back_config,
        )
    _, back_profile = config.profile(config.load(back_config))
    check(code == 0 and back_profile["provider"] == "openai_compatible",
          "going back in the model menu returns to the engine without repeating the language")
    check(setup_ollama.RECOMMENDED[0].endswith(":UD-IQ1_M"),
          "Qwen3.6-35B-A3B UD-IQ1_M is the first recommendation")
    # The reference is a repository and a precision rather than an Ollama tag,
    # and that is load-bearing rather than cosmetic. `phi4-mini:latest` names
    # whatever that tag points at today, so no measurement can ever be pinned
    # to it: a report is about one file with one digest. The two rows that now
    # carry a measurement taken on this machine had to become file-exact to
    # carry it.
    check(len(setup_ollama.RECOMMENDED) == 5
          and "hf.co/unsloth/Phi-4-mini-instruct-GGUF:Q4_K_M"
          in setup_ollama.RECOMMENDED,
          "the official Phi-4 Mini is among the five recommendations, by file")
    floating = [item["reference"] for item in setup_ollama.LOCAL_CATALOG
                if item.get("measured_here")
                and not item["reference"].startswith("hf.co/")]
    check(not floating,
          "no measured row is recommended under a tag that can move under it"
          + (f" (found {', '.join(floating)})" if floating else ""))
    check("qwen3:4b-instruct-2507-q4_K_M" not in setup_ollama.RECOMMENDED,
          "the old test Qwen does not appear in the curation")

    # The suggestion list is the screen the choice is made on. A measurement
    # that reaches only the discovery screen and the line printed after the
    # choice is a measurement nobody used, which is what happened the first
    # time: the numbers landed in `benchmark_cell`, the suggestion row is built
    # by `_model_label`, and the two never met.
    import terminal_ui

    for language in ("en", "pt-BR"):
        speak = setup_ollama.Translator(language)
        # `_resolved_local_catalog` and not `_recommended_catalog`: the second
        # is what the screen actually calls, and testing the first is how a
        # check ends up proving something no user path runs. `_resolve_live` is
        # stubbed out so this exercises the screen's code offline.
        live = setup_ollama._resolve_live
        setup_ollama._resolve_live = lambda *_a, **_k: None
        try:
            rows = {
                item["base_model"]: setup_ollama._model_label(item, [], speak)
                for item in setup_ollama._resolved_local_catalog(
                    None, {"gpus": [{"vram_mb": 4096}]}, speak)
            }
        finally:
            setup_ollama._resolve_live = live
        for item in setup_ollama.LOCAL_CATALOG:
            measured = item.get("measured_here")
            if not measured:
                continue
            row = rows[item["reference"]]
            check(measured["humaneval"] in row
                  and f"{measured['tokens_per_second']:.0f}" in row,
                  f"[{language}] the suggestion row for {item['reference']} "
                  f"carries what was measured here ({row})")
            check(speak.t("model.origin.measured") in row
                  and speak.t("model.origin.curated") not in row,
                  f"[{language}] a measured row says measured, not reviewed")
            # Everything after the fit text is the first thing an 80-column
            # terminal throws away, so the measurement cannot live there.
            narrow = terminal_ui.fit(row, 77)
            check(speak.t("model.origin.measured") in narrow
                  and measured["humaneval"] in narrow,
                  f"[{language}] the measurement survives an 80-column "
                  f"terminal ({narrow})")

        unmeasured = [item["reference"] for item in setup_ollama.LOCAL_CATALOG
                      if not item.get("measured_here")]
        for reference in unmeasured:
            check(speak.t("model.origin.measured") not in rows[reference],
                  f"[{language}] {reference} is not called measured, because "
                  "nobody measured it")
    check(
        setup_ollama._normalize_api_url(
            "https://api.groq.com/openai/v1/chat/completions/"
        ) == "https://api.groq.com/openai/v1",
        "setup fixes an endpoint pasted with /chat/completions",
    )

    check(
        setup_ollama.max_context(infos["gpt-oss:20b"]) == 131072,
        "it detects the nominal context and ignores original_context_length",
    )
    check(setup_ollama.parse_context("16K") == 16384, "the human input 16K becomes tokens")
    context_answers = answers("6", "4K", "12K")
    with redirect_stdout(io.StringIO()):
        context = setup_ollama._choose_context(262144, context_answers, pt)
    check(context == 12288, "the manual context refuses 4K and accepts 12K")
    with redirect_stdout(io.StringIO()):
        back = setup_ollama._choose_context(262144, answers("7"), pt)
        back_thinking = setup_ollama._choose_thinking(
            dict(setup_ollama._model_item("test"), thinking_kind="levels"),
            answers("4"), pt,
        )
    check(back is None and back_thinking == "__context__",
          "the context and reasoning menus allow going back")

    # ------------------------------------------------------------------
    # Task-oriented onboarding.
    #
    # Every check below drives the real screens. The Hugging Face seam is
    # replaced by a fake that answers with numbers no live repository has, so a
    # code path that went around the seam and reached the network would report
    # different sizes and fail here. That is deliberate: the screen once made
    # six serial requests on every draw, which both froze the setup on a bad
    # link and made this file depend on the network its own first line promises
    # it does not use.
    # ------------------------------------------------------------------
    class FakeResponse:
        def __init__(self, payload=None, length=None):
            self.payload = payload
            self.headers = {"Content-Length": str(length)} if length else {}

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    QWEN_REPO = "unsloth/Qwen3.6-35B-A3B-GGUF"
    QWEN_FILE = "Qwen3.6-35B-A3B-UD-IQ1_M.gguf"
    QWEN_BYTES = 2 * 1024 ** 3
    hf_requests = []

    def fake_hf(request, timeout=None):
        url = request.full_url
        hf_requests.append(url)
        if url == f"{model_discovery.HF_API}/{QWEN_REPO}":
            return FakeResponse({
                "id": QWEN_REPO,
                "siblings": [{"rfilename": QWEN_FILE}],
                "cardData": {"base_model": "Qwen/Qwen3.6-35B-A3B"},
            })
        if url.endswith("/config.json") and "Qwen3.6-35B-A3B/" in url:
            return FakeResponse({
                "num_hidden_layers": 4, "num_key_value_heads": 2,
                "head_dim": 64, "num_attention_heads": 8,
            })
        if request.get_method() == "HEAD" and url.endswith(QWEN_FILE):
            return FakeResponse(length=QWEN_BYTES)
        raise urllib.error.URLError("not in the fake index")

    original_detect = setup_ollama.hardware.detect
    original_seam = setup_ollama.LOCAL_RESOLUTION_URLOPEN
    setup_ollama.LOCAL_RESOLUTION_URLOPEN = fake_hf
    try:
        setup_ollama.hardware.detect = lambda: {
            "gpus": [{"name": "NVIDIA GeForce GTX 1650", "vram_mb": 4096,
                      "bandwidth_gbs": 128.0}],
            "ram_mb": 15813, "cpu_cores": 12,
        }
        setup_ollama._LOCAL_RESOLUTION_CACHE.clear()
        task_config = root / "task-config.json"
        task_out = io.StringIO()
        with redirect_stdout(task_out):
            code = setup_ollama.run_setup(
                answers("1", "1", "1", "6", "12K", "1"), config_file=task_config,
            )
        task_data = config.load(task_config)
        screen = task_out.getvalue()
        check(code == 0 and task_data["onboarding"]["task"] == "fix_bug",
              "the onboarding records the declared task alongside the profile")
        check(pt.t("onboarding.task.ruler.fix_bug") in screen,
              "the declared task names the public ruler it selected, on screen")
        check(pt.t("model.benchmark.scope") in screen,
              "the screen that shows scores also says they are pre-quantization")
        check("NVIDIA GeForce GTX 1650" in screen and "15.4" in screen,
              "the detected machine is stated instead of being assumed")
        check(f"{QWEN_BYTES / 1024 ** 3:.2f}" in screen
              and any(QWEN_FILE in url for url in hf_requests),
              "the recommended list reports the size the seam returned, not a guess")
        check(all("huggingface.co" in url for url in hf_requests),
              "resolution goes through the injected seam and nowhere else")

        # An entry the fake index refuses stands for a model whose size nothing
        # records. Saying "does not fit" there would be inventing the number
        # that decides it.
        check(pt.t("model.fit.unknown") in screen,
              "a model whose size cannot be resolved says so instead of guessing")

        setup_ollama._LOCAL_RESOLUTION_CACHE.clear()
        skip_config = root / "skip-config.json"
        skip_out = io.StringIO()
        with redirect_stdout(skip_out):
            code = setup_ollama.run_setup(
                answers("1", "4", "1", "1", "6", "12K", "1"),
                config_file=skip_config,
            )
        skip_data = config.load(skip_config)
        check(code == 0 and "onboarding" not in skip_data,
              "skipping the task question stores nothing rather than a default")
        check(pt.t("onboarding.task.ruler.fix_bug") not in skip_out.getvalue(),
              "a skipped task claims no ruler")

        # Preselection is what makes `isaacli setup` the way to redo the
        # onboarding: the stored answer has to come back as the default.
        preselected = []
        original_select = setup_ollama._select

        def recording_select(tr_, title, options, input_fn, explanation=None,
                             initial=0, disabled=None):
            preselected.append(initial)
            return initial

        setup_ollama._select = recording_select
        try:
            chosen_task = setup_ollama._choose_task(
                config.load(task_config), answers(), pt,
            )
        finally:
            setup_ollama._select = original_select
        check(chosen_task == "fix_bug" and preselected == [0],
              "running the onboarding again defaults to the task already stored")

        # No GPU is a normal machine. Reporting "does not fit" against zero VRAM
        # answers a question nobody asked and hides the real one, which is that
        # it would run on the CPU.
        setup_ollama.hardware.detect = lambda: {
            "gpus": [], "ram_mb": 15813, "cpu_cores": 12,
        }
        setup_ollama._LOCAL_RESOLUTION_CACHE.clear()
        headless_out = io.StringIO()
        with redirect_stdout(headless_out):
            code = setup_ollama.run_setup(
                answers("1", "1", "1", "6", "12K", "1"),
                config_file=root / "headless-config.json",
            )
        headless = headless_out.getvalue()
        check(code == 0 and pt.t("hardware.local.no_gpu", ram="15.4", cores=12)
              in headless,
              "a machine with no GPU is reported as such, not as a failure")
        check(pt.t("model.fit.no_gpu_sized", weights=f"{QWEN_BYTES / 1024 ** 3:.2f}")
              in headless and pt.t("model.fit.does_not_fit") not in headless,
              "with no GPU the screen says it runs on the CPU instead of does not fit")

        # Detection that blows up must not take the setup with it.
        def exploding_detect():
            raise OSError("nvidia-smi is not speaking to the driver")

        setup_ollama.hardware.detect = exploding_detect
        setup_ollama._LOCAL_RESOLUTION_CACHE.clear()
        broken_out = io.StringIO()
        with redirect_stdout(broken_out):
            code = setup_ollama.run_setup(
                answers("1", "1", "1", "6", "12K", "1"),
                config_file=root / "broken-detect-config.json",
            )
        check(code == 0 and "Traceback" not in broken_out.getvalue()
              and pt.t("hardware.local.no_gpu", ram="0.0", cores=0)
              in broken_out.getvalue(),
              "hardware detection that raises degrades to a line, not a traceback")
    finally:
        setup_ollama.hardware.detect = original_detect
        setup_ollama.LOCAL_RESOLUTION_URLOPEN = original_seam
        setup_ollama._LOCAL_RESOLUTION_CACHE.clear()

    def interrupt(_prompt=""):
        raise KeyboardInterrupt

    setup_ollama.shutil.which = lambda _name: "/usr/bin/ollama"
    with redirect_stdout(io.StringIO()):
        code = setup_ollama.run_setup(interrupt, config_file=config_file)
    check(code == 130, "Ctrl+C cancels setup without a traceback")
finally:
    setup_ollama.shutil.which = original_which
    setup_ollama.OllamaLocal = original_client
    setup_ollama._ensure_server = original_server
    setup_ollama._download_model = original_download
    setup_ollama._validate_api = original_validate_api


# A local endpoint is the only one isaacli can start, and the only one where a
# missing key is normal rather than a mistake.
check(config.is_local_endpoint("http://127.0.0.1:8080/v1")
      and config.is_local_endpoint("http://localhost:11434/v1")
      and config.is_local_endpoint("http://[::1]:8080/v1")
      and not config.is_local_endpoint("https://api.groq.com/openai/v1")
      and not config.is_local_endpoint("https://127.0.0.1.evil.example/v1")
      and not config.is_local_endpoint(""),
      "only a real loopback host counts as local, including a lookalike domain")

tr = setup_ollama.Translator("en")
with redirect_stdout(io.StringIO()):
    autostart = setup_ollama._ask_autostart(
        "http://127.0.0.1:8080/v1",
        lambda _prompt: 'llama-server -m "/models/a b.gguf" -c 8192', tr)
    skipped = setup_ollama._ask_autostart(
        "http://127.0.0.1:8080/v1", lambda _prompt: "   ", tr)
    unbalanced = setup_ollama._ask_autostart(
        "http://127.0.0.1:8080/v1", lambda _prompt: 'llama-server -m "unclosed', tr)

check(autostart == {"cmd": ["llama-server", "-m", "/models/a b.gguf", "-c", "8192"],
                    "health_url": "http://127.0.0.1:8080/v1/models"},
      "the autostart command is split like a shell would, quoted paths included")
check(skipped is None and unbalanced is None,
      "an empty or unparsable command saves nothing instead of saving something broken")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("ISAAC SETUP OK: profiles, context and reasoning kept separate")
