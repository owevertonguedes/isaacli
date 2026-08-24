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
import terminal_ui

# The curated references, in catalog order. Derived here rather than read from
# a module constant, because the constant this used to read was a list nothing
# in the program consumed: the screens all draw from LOCAL_CATALOG itself, so
# the only thing keeping the constant correct was this file asserting on it.
CURATED_REFERENCES = [item["reference"] for item in setup_ollama.LOCAL_CATALOG]


def engine_answer(key, which=lambda _name: "/usr/bin/ollama"):
    """The menu position of one engine, resolved by name.

    The engine screen is built from what the machine actually has, so its
    entries move when a machine gains or loses an engine. A check that answers
    it with a fixed number keeps passing while silently exercising a different
    engine, which is the kind of green that hides a regression. This asks the
    same function the screen asks.
    """
    entries, _notes = setup_ollama._detect_engines(
        setup_ollama.Translator("en"), which_fn=which)
    return str(next(index for index, (name, _label) in enumerate(entries, 1)
                    if name == key))


def source_answer(key, config_file, tr=None, which=lambda _name: "/usr/bin/ollama"):
    """The /model source screen's position for one entry, resolved by name.

    Entered through model_source_entries, which is the function the screen
    itself runs. A check that entered through anything else would be proving a
    list nobody draws, which is the trap task 055 names.
    """
    entries, _options, _initial = setup_ollama.model_source_entries(
        config.load(config_file), tr or setup_ollama.Translator("en"),
        which_fn=which)
    for index, entry in enumerate(entries, 1):
        if entry == key or (isinstance(entry, tuple) and entry[0] == key):
            return str(index)
    raise AssertionError(f"no {key} entry in the model source screen")


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
            answers("1", "4", engine_answer("ollama"), "1", "6", "12K", "1"), config_file=config_file,
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
    # Through the function the screen runs, with the live resolution stubbed
    # out so it works offline. The orphan it used to call built the same list
    # from the same catalog and was reached by no user path at all, which is
    # how a check ends up proving something nobody runs.
    live = setup_ollama._resolve_live
    setup_ollama._resolve_live = lambda *_a, **_k: None
    try:
        recommended_menu = setup_ollama._resolved_local_catalog(
            None, {"gpus": []}, pt)
    finally:
        setup_ollama._resolve_live = live
    local_menu = setup_ollama._installed_models(client.installed)
    check([item["base_model"] for item in recommended_menu] == CURATED_REFERENCES,
          "the recommendations section preserves the curation and its order")
    check(any(item["base_model"] == "test-model:7b" for item in local_menu),
          "the installed section includes models queried live from Ollama")
    check(any(item["base_model"] == "granite4:micro-h" for item in local_menu)
          and "granite4:micro-h" not in CURATED_REFERENCES,
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
            answers(engine_answer("ollama"), "6", "1"), config_file=selector_config,
        )
    _, micro_profile = config.profile(config.load(selector_config))
    check(code == 0 and micro_profile["model"] == "granite4:micro-h",
          "/model finds Micro H in the live local list, not in the curation")
    check(micro_profile["num_ctx"] == 8192 and micro_profile["thinking"] is False,
          "/model asks for the context afterwards and detects the absence of thinking")
    check("isaac-qwen-legacy-16k" not in selector_out.getvalue(),
          "/model hides only the context copies the configuration recognises")
    client.installed.remove("isaac-qwen-legacy-16k")

    # Everything /model prints is printed onto the alternate screen, which is
    # torn down on the way out. A source that failed said why, the screen took
    # the sentence with it, and the REPL then said only "model unchanged": a
    # Kaggle launch that reported a real error looked, from outside, like the
    # command doing nothing at all. The reason is held until it has been read.
    original_source = setup_ollama._select_configured_api
    original_interactive = terminal_ui.interactive
    acknowledged = []
    try:
        setup_ollama._select_configured_api = (
            lambda *_args, **_kwargs: print("the Kaggle launch failed: no quota") or 1)
        terminal_ui.interactive = lambda _input_fn=None: True
        failing_out = io.StringIO()

        def reader(prompt):
            acknowledged.append(prompt)
            return ""

        with redirect_stdout(failing_out):
            failed_code = setup_ollama.run_model_selector(
                reader, config_file=selector_config)
    finally:
        setup_ollama._select_configured_api = original_source
        terminal_ui.interactive = original_interactive
    check(failed_code == 1 and len(acknowledged) == 1,
          f"a failed /model waits for the reason to be read before the screen "
          f"closes, and still reports the failure (code {failed_code})")
    check("no quota" in failing_out.getvalue(),
          "and the reason itself is what was on that screen")

    with redirect_stdout(io.StringIO()):
        code = setup_ollama.run_setup(
            answers("1", "4", engine_answer("ollama"), "5", "3", "3"), config_file=config_file,
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
        code = setup_ollama.run_setup(answers("1", "4", engine_answer("ollama"), "1"), config_file=config_file)
    check(code == 1 and config_file.read_text() == before_failure,
          "a model without tools is refused without touching the previous profile")
    client.infos[qwen36] = original_qwen_info

    # A machine without Ollama is still offered Ollama. Hiding it left a screen
    # whose absences read as a program that never had the engine, so the menu
    # is fixed and the label carries the evidence. What the entry must not do
    # is promise an installation this program does not perform, so it is asked
    # for both halves: the entry is there, and it says it is not installed.
    setup_ollama.shutil.which = lambda _name: None
    bare_entries, _bare_notes = setup_ollama._detect_engines(
        pt, which_fn=lambda _name: None)
    bare_keys = [key for key, _label in bare_entries]
    check(bare_keys == ["llamacpp", "ollama", "api", "kaggle"],
          f"the four sources are offered whatever this machine has "
          f"(menu was {bare_keys})")
    # Read rather than indexed, so that a menu which lost the entry reports the
    # loss on every line that depended on it instead of ending the file in a
    # traceback with the remaining checks unrun.
    bare_ollama = dict(bare_entries).get("ollama")
    check(bare_ollama == pt.t("engine.ollama.install"),
          f"and the Ollama entry says it is not installed instead of promising "
          f"an install this program does not do ({bare_ollama!r})")

    # By effect, not by reading the label: choosing it on a machine with no
    # Ollama has to land on the instructions and leave without writing a
    # profile. An entry that leads nowhere is worse than no entry.
    missing_config = root / "no-ollama.json"
    printed = io.StringIO()
    code = None
    if bare_ollama is not None:
        with redirect_stdout(printed):
            code = setup_ollama.run_setup(
                answers("1", "4", engine_answer("ollama", which=lambda _name: None)),
                config_file=missing_config,
            )
    instructions = printed.getvalue()
    check(code not in (None, 0) and "ollama.com" in instructions
          and pt.t("ollama.missing.title") in instructions,
          f"choosing it prints the official instructions and stops (code {code})")
    check(not missing_config.exists()
          or not config.load(missing_config).get("profiles"),
          "and no profile is written for an engine that is not there")

    # The same entry from the other screen, which reaches the instructions
    # through a different path: /model does not ask the language or the task
    # again, so an entry that works in setup can still land nowhere here.
    model_missing = root / "no-ollama-model.json"
    config.save(dict(config.empty_config(), language="pt-BR"), model_missing)
    model_printed = io.StringIO()
    model_code = None
    try:
        with redirect_stdout(model_printed):
            model_code = setup_ollama.run_model_selector(
                answers(source_answer("ollama", model_missing, pt,
                                      which=lambda _name: None)),
                config_file=model_missing,
            )
    except (AssertionError, StopIteration):
        # Reported as a failed check below rather than as a traceback that
        # would take the rest of this file with it.
        pass
    check(model_code not in (None, 0)
          and pt.t("ollama.missing.title") in model_printed.getvalue(),
          f"/model reaches the same instructions and comes back (code "
          f"{model_code})")

    ollama_entries, ollama_notes = setup_ollama._detect_engines(
        pt, which_fn=lambda name: "/usr/bin/ollama" if name == "ollama" else None)
    check("ollama" in [key for key, _label in ollama_entries],
          "a machine that has Ollama keeps being offered it, and nothing is migrated")
    check(all("tok" not in note and "/s" not in note for note in ollama_notes),
          "and the line suggesting the other engine claims no speed nobody measured")

    # A feature that works in setup and cannot be reached from /model is half a
    # feature. Both screens are asked, through the functions they themselves
    # run, whether the local engine is there.
    source_entries, _options, _initial = setup_ollama.model_source_entries(
        config.load(config_file), pt, which_fn=lambda _name: "/usr/bin/ollama")
    source_keys = [entry[0] if isinstance(entry, tuple) else entry
                   for entry in source_entries]
    check("llamacpp" in source_keys,
          f"/model offers the local llama.cpp engine, not only Ollama and APIs "
          f"(got {source_keys})")
    setup_entries, _notes = setup_ollama._detect_engines(
        pt, which_fn=lambda _name: "/usr/bin/ollama")
    check("llamacpp" in [key for key, _label in setup_entries],
          "and setup offers the same engine, from the same detection")

    # The same fixed menu on the other screen, on the same bare machine, and in
    # both catalogues: a source offered in setup and missing from /model is the
    # defect this pair of screens has already produced once.
    for language in ("pt-BR", "en"):
        catalogue = setup_ollama.Translator(language)
        bare_source, bare_options, _bare_initial = setup_ollama.model_source_entries(
            config.load(config_file), catalogue, which_fn=lambda _name: None)
        bare_source_keys = [entry[0] if isinstance(entry, tuple) else entry
                            for entry in bare_source]
        check("ollama" in bare_source_keys and "llamacpp" in bare_source_keys,
              f"/model in {language} offers both local engines with neither "
              f"installed (got {bare_source_keys})")

        # One engine, one name. The screen used to carry `llama.cpp local` and
        # `Llama Server · qwen2.5-coder-3b` at once, which is one server listed
        # twice under two names, and the second of them offered an install of
        # what was already running.
        aliases = ("llama.cpp", "llama-server", "llama server", "llamacpp")
        wearing = [option for option in bare_options
                   if sum(alias in option.casefold() for alias in aliases)]
        check(not wearing,
              f"and no entry names the local engine in the jargon of its build "
              f"({wearing})")

    # A llama.cpp profile speaks the OpenAI protocol, so it lands in the same
    # config section as a remote endpoint. Offered as one, /model would try to
    # list its models over HTTP against a server that is not running.
    local_config = root / "local-engine.json"
    local_data = dict(config.empty_config(), language="pt-BR")
    local_data["profiles"]["llamacpp-fixture"] = {
        "provider": "openai_compatible",
        "provider_name": setup_ollama.LLAMACPP_PROVIDER_NAME,
        "base_url": "http://127.0.0.1:8080/v1", "model": "fixture",
    }
    local_data["default_profile"] = "llamacpp-fixture"
    config.save(local_data, local_config)
    local_entries, local_options, local_initial = setup_ollama.model_source_entries(
        config.load(local_config), pt, which_fn=lambda _name: "/usr/bin/ollama")
    check(not any(isinstance(entry, tuple) and entry[1] == "llamacpp-fixture"
                  for entry in local_entries),
          "a saved llama.cpp profile is not listed as somebody's remote API")
    check(local_entries[local_initial] == "llamacpp",
          "and opening /model on it starts the cursor on the local engine")

    # The same profile written by hand, before this program existed, carries the
    # user's own name for the engine. Recognised by the provider name alone it
    # came out as a remote API next to the engine entry, which is a local server
    # listed twice, once under a name the screen offers to install.
    hand_root = root / "hand-built"
    hand_root.mkdir(exist_ok=True)
    hand_server = hand_root / "llama-server"
    hand_server.write_text("#!/bin/sh\n", encoding="utf-8")
    hand_server.chmod(0o755)
    launcher = hand_root / "start-llama-server.sh"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    launcher.chmod(0o755)
    hand_config = root / "hand-made-engine.json"
    hand_data = dict(config.empty_config(), language="pt-BR")
    hand_data["profiles"]["llama-server-fixture"] = {
        "provider": "openai_compatible",
        "provider_name": "Llama Server",
        "base_url": "http://127.0.0.1:8080/v1", "model": "fixture",
        "autostart": {"cmd": [str(launcher)],
                      "health_url": "http://127.0.0.1:8080/v1/models"},
    }
    hand_data["default_profile"] = "llama-server-fixture"
    config.save(hand_data, hand_config)
    hand_entries, hand_options, hand_initial = setup_ollama.model_source_entries(
        config.load(hand_config), pt, which_fn=lambda _name: None)
    check(not any(isinstance(entry, tuple) and entry[1] == "llama-server-fixture"
                  for entry in hand_entries),
          "a hand-made local server profile is not listed a second time as an API")
    check(hand_entries[hand_initial] == "llamacpp",
          "and the cursor starts on the engine it belongs to")
    engine_label = hand_options[hand_entries.index("llamacpp")]
    check(engine_label == pt.t("engine.llamacpp.found"),
          f"and the entry reports the server it already runs rather than offering "
          f"to install one ({engine_label!r})")

    # Back to the machine the rest of this file describes. engine_answer asks
    # the same question the screen will ask, so the two have to be looking at
    # the same machine when they ask it.
    setup_ollama.shutil.which = lambda _name: "/usr/bin/ollama"

    setup_ollama._validate_api = lambda url, key, model: None
    api_config = root / "api-config.json"
    with redirect_stdout(io.StringIO()):
        code = setup_ollama.run_setup(
            answers("1", "4", engine_answer("api"), "Groq", "https://api.groq.com/openai/v1",
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
            answers(source_answer("api", api_config), "2", "1"), api_config, "pt-BR", pt,
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
                "1", "4", engine_answer("api"), "Server", "https://api.test/v1/chat/completions",
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
            answers("1", "4", engine_answer("ollama"), "8", "4", engine_answer("api"), "Server", "https://api.test/v1",
                    "test-model", "key", "1"),
            config_file=back_config,
        )
    _, back_profile = config.profile(config.load(back_config))
    check(code == 0 and back_profile["provider"] == "openai_compatible",
          "going back in the model menu returns to the engine without repeating the language")
    check(CURATED_REFERENCES[0].endswith(":UD-IQ1_M"),
          "Qwen3.6-35B-A3B UD-IQ1_M is the first recommendation")
    # The reference is a repository and a precision rather than an Ollama tag,
    # and that is load-bearing rather than cosmetic. `phi4-mini:latest` names
    # whatever that tag points at today, so no measurement can ever be pinned
    # to it: a report is about one file with one digest. The two rows that now
    # carry a measurement taken on this machine had to become file-exact to
    # carry it.
    check(len(CURATED_REFERENCES) == 5
          and "hf.co/unsloth/Phi-4-mini-instruct-GGUF:Q4_K_M"
          in CURATED_REFERENCES,
          "the official Phi-4 Mini is among the five recommendations, by file")
    floating = [item["reference"] for item in setup_ollama.LOCAL_CATALOG
                if item.get("measured_here")
                and not item["reference"].startswith("hf.co/")]
    check(not floating,
          "no measured row is recommended under a tag that can move under it"
          + (f" (found {', '.join(floating)})" if floating else ""))
    check("qwen3:4b-instruct-2507-q4_K_M" not in CURATED_REFERENCES,
          "the old test Qwen does not appear in the curation")

    # The suggestion list is the screen the choice is made on. A measurement
    # that reaches only the discovery screen and the line printed after the
    # choice is a measurement nobody used, which is what happened the first
    # time: the numbers landed in a cell no suggestion row drew, and the two
    # never met.
    import model_discovery

    profile = {"gpus": [{"vram_mb": 4096, "bandwidth_gbs": 128.0,
                         "name": "NVIDIA GeForce GTX 1650"}]}
    for language in ("en", "pt-BR"):
        speak = setup_ollama.Translator(language)
        # `_resolved_local_catalog` is what the screen actually calls, which is
        # the whole point: entering anywhere else is how a check ends up
        # proving something no user path runs. `_resolve_live` is stubbed out
        # so this exercises the screen's code offline.
        live = setup_ollama._resolve_live
        setup_ollama._resolve_live = lambda *_a, **_k: None
        try:
            items = setup_ollama._resolved_local_catalog(None, profile, speak)
        finally:
            setup_ollama._resolve_live = live
        # Every call below is the call the screen makes, with the same
        # arguments, so what passes here is what a person sees.
        row_machine = setup_ollama._row_machine(profile)
        cells = [setup_ollama._model_cells(item, [], speak, row_machine)
                 for item in items]
        table = model_discovery.model_table(
            cells, row_machine, translate=speak.t,
            state_header=speak.t("model.table.installed"))
        drawn = dict(zip([item["base_model"] for item in items], table["rows"]))
        by_reference = dict(zip([item["base_model"] for item in items], cells))
        # Offline nothing resolves, so every fit cell is a dash and the column
        # collapses into the note. The card still names it exactly once, and
        # trimmed: what must never happen is the card appearing per row.
        drawn_once = table["header"] + "\n" + table["legend"]
        check("GTX 1650" in drawn_once and "NVIDIA" not in drawn_once,
              f"[{language}] the card names its column once, trimmed to the "
              f"part that names the part ({drawn_once})")
        check(not any("GTX" in row for row in table["rows"]),
              f"[{language}] and never once per row ({table['rows']})")
        widest = max(len(line) for line in [table["header"], *table["rows"]])
        check(widest <= 80,
              f"[{language}] the suggestion table fits in 80 columns "
              f"(widest was {widest})")
        for item in setup_ollama.LOCAL_CATALOG:
            measured = item.get("measured_here")
            if not measured:
                continue
            cell = by_reference[item["reference"]]
            check(cell["tps"] == f"{measured['tokens_per_second']:.0f}",
                  f"[{language}] the suggestion row for {item['reference']} "
                  f"carries the throughput measured here ({cell['tps']})")
            check(measured["humaneval"] in cell["rankings"],
                  f"[{language}] and its ranking, with its owner "
                  f"({cell['rankings']})")
            check(measured["humaneval"] in drawn[item["reference"]],
                  f"[{language}] and both survive being drawn into the row "
                  f"({drawn[item['reference']]})")
        measured_names = [cell["name"] for cell in cells
                          if cell.get("measured")]
        check(measured_names
              and all(name in table["legend"] for name in measured_names),
              f"[{language}] the legend names what was measured here, since "
              f"the cell is a bare number ({table['legend']})")

        for item in setup_ollama.LOCAL_CATALOG:
            if item.get("measured_here"):
                continue
            cell = by_reference[item["reference"]]
            # Offline there is no live size, so there is nothing to estimate
            # from, and an invented number would read like a measured one.
            check(cell["tps"] == model_discovery.EMPTY_CELL,
                  f"[{language}] {item['reference']} shows no throughput, "
                  f"because nobody measured it and nothing was resolved "
                  f"({cell['tps']})")
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

    # MIN_CONTEXT is the smallest rung worth offering when the card has room for
    # it, not a floor the hardware has to clear. Treated as a floor it emptied
    # the menu of every rung and then refused every typed value, the ceiling
    # itself included: a screen reading "use a value between 8K and 6,730" whose
    # only working key was Back. Answered here the way the owner answered it,
    # by typing the ceiling and expecting it to be taken.
    tight_screens = []
    original_tight_select = setup_ollama.terminal_ui.select

    def record_tight(title, options, **kwargs):
        tight_screens.append(options)
        # The row before "back" is the one that takes a typed value.
        return len(options) - 2

    try:
        setup_ollama.terminal_ui.select = record_tight
        with redirect_stdout(io.StringIO()) as tight_out:
            tight = setup_ollama._choose_context(
                6730, answers("8K", "6730"), pt)
    except StopIteration:
        # The defect is an unsatisfiable prompt, so the screen asks forever and
        # the scripted answers run out. Caught and named here: a check that let
        # the StopIteration through reported this as a traceback and took the
        # rest of the file down with it, which hides the result instead of
        # showing it.
        tight = "the screen refused every value, including the ceiling"
    finally:
        setup_ollama.terminal_ui.select = original_tight_select
    check(tight == 6730,
          f"a ceiling under the smallest rung still accepts the ceiling, "
          f"instead of refusing every value there is ({tight})")
    check(any(pt.t("context.maximum", limit="6.730") in option
              for option in tight_screens[-1]),
          f"and it is on the menu as a row, so the screen is never a list with "
          f"no context on it ({tight_screens[-1]})")
    check(pt.t("context.manual.only", limit="6.730") in tight_out.getvalue(),
          "a range of one value says so instead of naming it twice")

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
                answers("1", "1", engine_answer("ollama"), "6", "12K", "1"), config_file=task_config,
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
        check(pt.t("model.row.size", size=f"{QWEN_BYTES / 1024 ** 3:.1f}") in screen
              and any(QWEN_FILE in url for url in hf_requests),
              "the recommended list reports the size the seam returned, not a guess")
        check(all("huggingface.co" in url for url in hf_requests),
              "resolution goes through the injected seam and nowhere else")

        # An entry the fake index refuses stands for a model whose size nothing
        # records. Saying "does not fit" there would be inventing the number
        # that decides it, and so would printing it as 0.0 GiB.
        check(pt.t("model.row.size", size="0.0") not in screen,
              "a model whose size cannot be resolved is never drawn as empty")

        setup_ollama._LOCAL_RESOLUTION_CACHE.clear()
        skip_config = root / "skip-config.json"
        skip_out = io.StringIO()
        with redirect_stdout(skip_out):
            code = setup_ollama.run_setup(
                answers("1", "4", engine_answer("ollama"), "1", "6", "12K", "1"),
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
                answers("1", "1", engine_answer("ollama"), "6", "12K", "1"),
                config_file=root / "headless-config.json",
            )
        headless = headless_out.getvalue()
        check(code == 0 and pt.t("hardware.local.no_gpu", ram="15.4", cores=12)
              in headless,
              "a machine with no GPU is reported as such, not as a failure")
        # Through the function the screen runs, because what the rows say is
        # decided there and a screen-wide substring cannot tell one row from
        # another.
        headless_items = setup_ollama._resolved_local_catalog(
            None, {"gpus": []}, pt)
        check(all(item["fit_cell"] == model_discovery.EMPTY_CELL
                  for item in headless_items)
              and pt.t("model.table.no_gpu") in headless,
              "with no GPU no row claims a fit, and the column says CPU")
        check(all(pt.t("model.fit.no_gpu") in item["fit_label"]
                  or pt.t("model.fit.no_gpu_sized", weights="0.00")[:40]
                  in item["fit_label"]
                  for item in headless_items),
              "and the sentence that says why stays reachable behind --debug")

        # Detection that blows up must not take the setup with it.
        def exploding_detect():
            raise OSError("nvidia-smi is not speaking to the driver")

        setup_ollama.hardware.detect = exploding_detect
        setup_ollama._LOCAL_RESOLUTION_CACHE.clear()
        broken_out = io.StringIO()
        with redirect_stdout(broken_out):
            code = setup_ollama.run_setup(
                answers("1", "1", engine_answer("ollama"), "6", "12K", "1"),
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
