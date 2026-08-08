#!/usr/bin/env python3
"""Tests for the guided setup, with no network, downloads or a real Ollama."""
import io
import json
import sys
import tempfile
import stat
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config
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
            answers("1", "1", "1", "6", "12K", "1"), config_file=config_file,
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
            answers("1", "1", "5", "3", "3"), config_file=config_file,
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
        code = setup_ollama.run_setup(answers("1", "1", "1"), config_file=config_file)
    check(code == 1 and config_file.read_text() == before_failure,
          "a model without tools is refused without touching the previous profile")
    client.infos[qwen36] = original_qwen_info

    setup_ollama.shutil.which = lambda _name: None
    missing = root / "missing.json"
    with redirect_stdout(io.StringIO()):
        code = setup_ollama.run_setup(answers("1", "1"), config_file=missing)
    check(code == 2 and not missing.exists(),
          "a missing Ollama gives instructions and writes no partial config")

    setup_ollama._validate_api = lambda url, key, model: None
    api_config = root / "api-config.json"
    with redirect_stdout(io.StringIO()):
        code = setup_ollama.run_setup(
            answers("1", "2", "Groq", "https://api.groq.com/openai/v1",
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
            answers("2", "2", "1"), api_config, "pt-BR", pt,
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
                "1", "2", "Server", "https://api.test/v1/chat/completions",
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
            answers("1", "1", "8", "2", "Server", "https://api.test/v1",
                    "test-model", "key", "1"),
            config_file=back_config,
        )
    _, back_profile = config.profile(config.load(back_config))
    check(code == 0 and back_profile["provider"] == "openai_compatible",
          "going back in the model menu returns to the engine without repeating the language")
    check(setup_ollama.RECOMMENDED[0].endswith(":UD-IQ1_M"),
          "Qwen3.6-35B-A3B UD-IQ1_M is the first recommendation")
    check(len(setup_ollama.RECOMMENDED) == 5
          and "phi4-mini:latest" in setup_ollama.RECOMMENDED,
          "the official Phi-4 Mini is among the five recommendations")
    check("qwen3:4b-instruct-2507-q4_K_M" not in setup_ollama.RECOMMENDED,
          "the old test Qwen does not appear in the curation")
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


print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("ISAAC SETUP OK: profiles, context and reasoning kept separate")
