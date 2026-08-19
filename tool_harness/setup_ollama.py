"""Guided setup for local Ollama models."""
import json
import getpass
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import agent
import config
import debug
import terminal_ui
from i18n import SUPPORTED_LANGUAGES, Translator


MODEL_CATALOG_PATH = Path(__file__).resolve().parent / "model_catalog.json"


def _load_catalog(key, path=MODEL_CATALOG_PATH):
    """Load a curated section; installed models still come from Ollama."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        models = data[key]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise RuntimeError(
            Translator().t("setup.catalog.invalid", path=path, error=e)) from e
    if not isinstance(models, list) or not models or not all(
            isinstance(item, str) and item.strip() for item in models):
        raise RuntimeError(Translator().t("setup.catalog.not_list", path=path))
    return models


RECOMMENDED = _load_catalog("recommended")

CONTEXT_LEVELS = [
    ("context.compact", 8192),
    ("context.standard", 16384),
    ("context.long", 32768),
    ("context.extended", 65536),
]
MIN_CONTEXT = 8192


def _base_url():
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")


class OllamaLocal:
    def __init__(self, base_url=None):
        self.base_url = (base_url or _base_url()).rstrip("/")

    def _request(self, method, path, payload=None, timeout=10):
        payload_bytes = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base_url + path, data=payload_bytes, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)

    def version(self):
        return self._request("GET", "/api/version", timeout=2).get("version")

    def models(self):
        return self._request("GET", "/api/tags").get("models", [])

    def show(self, model):
        return self._request("POST", "/api/show", {"model": model})


def max_context(show):
    candidates = []
    for key, value in (show.get("model_info") or {}).items():
        if key.endswith(".context_length") and "original_context" not in key:
            try:
                candidates.append(int(value))
            except (TypeError, ValueError):
                pass
    return max(candidates) if candidates else None


def format_context(value):
    if value % 1024 == 0:
        return f"{value // 1024}K"
    return f"{value:,}".replace(",", ".")


def parse_context(texto):
    value = texto.strip().lower().replace(".", "")
    multiplier = 1024 if value.endswith("k") else 1
    if multiplier != 1:
        value = value[:-1]
    try:
        return int(value) * multiplier
    except ValueError:
        return 0


def _title(tr, title, explanation=None):
    parts = [title]
    if explanation:
        parts.extend(("", explanation))
    return "\n".join(parts)


def _select(tr, title, options, input_fn=input, explanation=None, initial=0,
                disabled=None):
    return terminal_ui.select(
        _title(tr, title, explanation), options, input_fn=input_fn,
        prompt=tr.t("select.prompt"), invalid=tr.t("select.invalid"), initial=initial,
        disabled=disabled,
        more_above=tr.t("ui.more_above", count="{count}"),
        more_below=tr.t("ui.more_below", count="{count}"),
    )


def _choose_language(input_fn):
    languages = list(SUPPORTED_LANGUAGES)
    index = _select(
        Translator("en"), "Isaac CLI · Language / Idioma",
        [SUPPORTED_LANGUAGES[code] for code in languages], input_fn,
        "Use ↑/↓ e Enter · Use ↑/↓ and Enter",
    )
    return languages[index]


def _ollama_install_instructions(tr):
    terminal_ui.clear()
    print(tr.t("ollama.missing.title"))
    print(tr.t("ollama.missing.explain"))
    key = {"linux": "ollama.install.linux", "darwin": "ollama.install.macos",
             "windows": "ollama.install.windows"}.get(platform.system().lower(),
                                                       "ollama.install.other")
    print(tr.t(key))
    print(tr.t("ollama.install.retry"))


def _ensure_server(client, ollama_exe, tr=None):
    tr = tr or Translator()
    try:
        return client.version(), None
    except Exception:
        debug.swallowed("setup_ollama._ensure_server probe")
    proc = subprocess.Popen([ollama_exe, "serve"], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        time.sleep(0.25)
        try:
            return client.version(), proc
        except Exception:
            if proc.poll() is not None:
                break
    if proc.poll() is None:
        proc.terminate()
    raise RuntimeError(tr.t("ollama.server.failed"))


def _download_model(ollama_exe, model, tr=None):
    tr = tr or Translator()
    print(tr.t("model.download.running", model=model))
    result = subprocess.run([ollama_exe, "pull", model], check=False)
    if result.returncode != 0:
        raise RuntimeError(tr.t("model.download.failed", code=result.returncode))


def _model_item(model, recommended=False):
    normalized = model.removesuffix(":latest")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-") or "local"
    return {
        "id": slug, "name": model, "base_model": model,
        "temperature": 0, "thinking_kind": "detect", "recommended": recommended,
    }


def _recommended_catalog():
    return [_model_item(model, recommended=True) for model in RECOMMENDED]


def _installed_models(installed):
    return [_model_item(model) for model in sorted(installed, key=str.casefold)]


def _is_installed(model, installed):
    wanted = model.removesuffix(":latest").casefold()
    return any(item.removesuffix(":latest").casefold() == wanted for item in installed)


def _model_label(item, installed, tr):
    base = item["base_model"]
    state = ("model.installed" if _is_installed(base, installed)
              else "model.not_installed")
    return f"{base} [{tr.t(state)}]"


def _choose_context(limit, input_fn, tr):
    levels = [level for level in CONTEXT_LEVELS if not limit or level[1] <= limit]
    if limit and limit >= MIN_CONTEXT and limit not in {value for _, value in levels}:
        levels.append(("context.maximum", limit))
    options = ([tr.t(key, limit=format_context(value)) for key, value in levels]
              + [tr.t("context.manual"), tr.t("navigation.back")])
    explanation = tr.t("context.explain")
    if limit:
        explanation += "\n" + tr.t("context.limit", limit=format_context(limit))
    index = _select(tr, tr.t("context.title"), options, input_fn, explanation)
    if index == len(levels) + 1:
        return None
    if index < len(levels):
        return levels[index][1]
    while True:
        print(_title(tr, tr.t("context.title"), explanation))
        value = parse_context(input_fn(tr.t("context.manual.prompt")))
        if value >= MIN_CONTEXT and (not limit or value <= limit):
            return value
        ceiling = format_context(limit) if limit else "∞"
        print(tr.t("context.manual.invalid", limit=ceiling))


def _choose_thinking(item, input_fn, tr):
    if item["thinking_kind"] == "none":
        return False
    index = _select(
        tr, tr.t("thinking.gpt.title"),
        [tr.t("thinking.low"), tr.t("thinking.medium"), tr.t("thinking.high"),
         tr.t("navigation.back")],
        input_fn, tr.t("thinking.gpt.explain"), initial=1,
    )
    return ["low", "medium", "high", "__context__"][index]


def _save_ollama_profile(data, item, num_ctx, limit, thinking):
    """Persist a logical model; context does not create another model/profile."""
    base = item["base_model"]
    profiles = data.setdefault("profiles", {})
    duplicates = [
        name for name, profile in profiles.items()
        if profile.get("provider", "ollama") == "ollama"
        and (profile.get("base_model") or profile.get("model")) == base
    ]
    for name in duplicates:
        profiles.pop(name, None)
    name = item["id"]
    profiles[name] = {
        "provider": "ollama", "model": base, "base_model": base,
        "num_ctx": num_ctx, "context_limit": limit, "thinking": thinking,
        "temperature": item["temperature"],
    }
    data["default_profile"] = name
    return name


def _normalize_api_url(base_url):
    url = base_url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/models"):
        if url.endswith(suffix):
            url = url[:-len(suffix)]
    parts = urllib.parse.urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise RuntimeError(Translator().t("api.url.invalid"))
    return url


def _api_http_message(error):
    detail = ""
    try:
        body = error.read().decode("utf-8", errors="replace")
        data = json.loads(body)
        item = data.get("error", data)
        detail = item.get("message", "") if isinstance(item, dict) else str(item)
    except (OSError, ValueError, AttributeError):
        pass
    detail = re.sub(r"(?i)(api[_ -]?key\s*[=:]?\s*)\S+", r"\1[oculta]", detail)
    return f"HTTP {error.code}" + (f": {detail}" if detail else "")


def _list_api_models(base_url, api_key):
    """Query {base_url}/models live: the same call used to validate and to
    para trocar de model sem reinformar endpoint/key (ver _setup_api)."""
    req = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json",
                 "User-Agent": agent.USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(_api_http_message(e)) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            Translator().t("api.connect.failed", reason=e.reason)) from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise RuntimeError(Translator().t("api.models.invalid_json")) from e
    return sorted(item.get("id") for item in payload.get("data", [])
                  if isinstance(item, dict) and item.get("id"))


def _validate_api(base_url, api_key, model):
    models = _list_api_models(base_url, api_key)
    if model not in models:
        close_matches = sorted(m for m in models if model.lower() in m.lower() or
                          m.lower() in model.lower())[:5]
        suggestion = (Translator().t("api.model.suggestion",
                                     options=", ".join(close_matches))
                      if close_matches else "")
        raise RuntimeError(Translator().t("api.model.unavailable",
                                          model=model, suggestion=suggestion))


def _api_profile_name(provider_name, model):
    slug = re.sub(r"[^a-z0-9]+", "-", (provider_name or "api").lower()).strip("-") or "api"
    return f"{slug}-{re.sub(r'[^a-z0-9]+', '-', model.lower()).strip('-')}"


def _ask_autostart(base_url, input_fn, tr):
    """Offer to manage a server the user runs themselves (llama-server or any
    other compatible one), the way isaacli already manages Ollama.

    Only offered for a local endpoint: there is nothing to start on a machine
    that is not this one. An empty answer keeps the previous behaviour, where
    the user starts the server before opening isaacli."""
    print(_title(tr, tr.t("api.autostart.title"), tr.t("api.autostart.explain")))
    raw = input_fn(tr.t("api.autostart.prompt")).strip()
    if not raw:
        return None
    try:
        cmd = shlex.split(raw)
    except ValueError as e:
        print(tr.t("api.autostart.invalid", error=e))
        return None
    if not cmd:
        return None
    # /models is the route every OpenAI-compatible server has to answer, and
    # the probe counts any HTTP answer as up, so even a 404 proves reachability.
    return {"cmd": cmd, "health_url": base_url.rstrip("/") + "/models"}


def _setup_api(language, input_fn, config_file, tr):
    field_error = None
    while True:
        terminal_ui.clear()
        explanation = tr.t("api.explain")
        if field_error:
            explanation += "\n\n" + field_error
        print(_title(tr, tr.t("api.title"), explanation))
        field_error = None
        name = input_fn(tr.t("api.name.prompt")).strip()
        base_url = input_fn(tr.t("api.url.prompt")).strip()
        model = input_fn(tr.t("api.model.prompt")).strip()
        if not name or not base_url or not model:
            field_error = tr.t("api.fields.missing")
            continue
        try:
            base_url = _normalize_api_url(base_url)
        except RuntimeError as e:
            field_error = tr.t("api.validation.failed", error=e)
            continue
        local_endpoint = config.is_local_endpoint(base_url)
        prompt_key = ("api.key.prompt.local" if local_endpoint
                      else "api.key.prompt")
        key = (getpass.getpass(tr.t(prompt_key)) if input_fn is input
                 else input_fn(tr.t(prompt_key))).strip()
        if not key and not local_endpoint:
            field_error = tr.t("api.key.missing")
            continue
        print(tr.t("api.validating"))
        try:
            _validate_api(base_url, key, model)
        except RuntimeError as e:
            validation_error = tr.t("api.validation.failed", error=e)
            action = _select(
                tr, tr.t("api.retry.title"),
                [tr.t("api.retry.yes"), tr.t("api.save.unverified"),
                 tr.t("navigation.back")], input_fn, validation_error,
            )
            if action == 0:
                continue
            if action == 2:
                return "__engine__"
        break
    index = _select(
        tr, tr.t("thinking.api.title"),
        [tr.t("thinking.disabled"), tr.t("thinking.low"),
         tr.t("thinking.medium"), tr.t("thinking.high")], input_fn,
        tr.t("thinking.api.explain"), initial=2,
    )
    thinking = [None, "low", "medium", "high"][index]
    autostart = _ask_autostart(base_url, input_fn, tr) if local_endpoint else None
    profile_name = _api_profile_name(name, model)
    credential = f"api:{profile_name}"
    secret_path = (Path(config_file).with_name("secrets.json")
                    if config_file else None)
    config.save_secret(credential, key, secret_path)
    data = config.load(config_file)
    data["language"] = language
    data["profiles"][profile_name] = {
        "provider": "openai_compatible", "provider_name": name,
        "base_url": base_url, "model": model, "thinking": thinking,
        "credential": credential, "temperature": 0,
    }
    if autostart:
        data["profiles"][profile_name]["autostart"] = autostart
    data["default_profile"] = profile_name
    config.save(data, config_file)
    return 0


def _run_setup(input_fn=input, config_file=None, initial_language=None,
                    ollama_only=False):
    if initial_language:
        language = initial_language
    else:
        try:
            language = _choose_language(input_fn)
        except KeyboardInterrupt:
            print("\n" + Translator("en").t("setup.cancelled"))
            return 130
    tr = Translator(language)
    print(tr.t("setup.title"), "\n")

    try:
        if not ollama_only:
            engine = _select(
                tr, tr.t("engine.title"),
                [tr.t("engine.ollama"), tr.t("engine.api")], input_fn,
                tr.t("engine.explain"),
            )
            if engine == 1:
                api_result = _setup_api(language, input_fn, config_file, tr)
                if api_result == "__engine__":
                    return _run_setup(input_fn, config_file, initial_language=language)
                return api_result
    except (RuntimeError, ValueError, urllib.error.URLError) as e:
        print(tr.t("setup.error", error=e))
        return 1
    except KeyboardInterrupt:
        print("\n" + tr.t("setup.cancelled"))
        return 130

    ollama_exe = shutil.which("ollama")
    if not ollama_exe:
        _ollama_install_instructions(tr)
        return 2

    client = OllamaLocal()
    started = None
    try:
        version, started = _ensure_server(client, ollama_exe, tr)
        installed = {m.get("name", "").removesuffix(":latest") for m in client.models()}
        current_config = config.load(config_file)
        configured_derivatives = {
            profile.get("model", "").removesuffix(":latest").casefold()
            for profile in (current_config.get("profiles") or {}).values()
            if profile.get("provider", "ollama") == "ollama"
            and profile.get("base_model")
            and profile.get("model") != profile.get("base_model")
        }
        while True:
            recommended_items = _recommended_catalog()
            catalogued = {
                item["base_model"].removesuffix(":latest").casefold()
                for item in recommended_items
            }
            local_items = [
                item for item in _installed_models(installed)
                if item["base_model"].removesuffix(":latest").casefold() not in catalogued
                and item["base_model"].removesuffix(":latest").casefold()
                not in configured_derivatives
            ]
            entries = [None, *recommended_items, None, *local_items, "__back__"]
            options = [
                tr.t("model.section.recommended"),
                *[_model_label(item, installed, tr) for item in recommended_items],
                tr.t("model.section.installed", count=len(local_items)),
                *[item["base_model"] for item in local_items],
                tr.t("navigation.back"),
            ]
            headers = {0, len(recommended_items) + 1}
            _current_profile, current_item = config.profile(current_config)
            modelo_atual = (
                (current_item.get("base_model") or current_item.get("model", ""))
                .removesuffix(":latest").casefold()
                if current_item else ""
            )
            initial = next(
                (i for i, entrada in enumerate(entries)
                 if isinstance(entrada, dict)
                 and entrada["base_model"].removesuffix(":latest").casefold()
                 == modelo_atual),
                0,
            )
            model_index = _select(
                tr, tr.t("model.title"), options, input_fn,
                terminal_ui.dim(
                    tr.t("model.recommended.explain", version=version), input_fn,
                ),
                initial=initial,
                disabled=headers,
            )
            chosen = entries[model_index]
            if chosen == "__back__":
                if ollama_only:
                    return 130
                return _run_setup(input_fn, config_file, initial_language=language)
            base = chosen["base_model"]
            if not _is_installed(base, installed):
                index = _select(
                    tr, tr.t("model.download.confirm", model=base),
                    [tr.t("model.download.yes"), tr.t("navigation.back")], input_fn,
                )
                if index:
                    continue
                _download_model(ollama_exe, base, tr)
                installed.add(base.removesuffix(":latest"))

            info = client.show(base)
            if "tools" not in set(info.get("capabilities") or []):
                print(tr.t("model.tools.missing", model=base))
                return 1
            if chosen.get("thinking_kind") == "detect":
                chosen["thinking_kind"] = (
                    "levels" if "thinking" in set(info.get("capabilities") or []) else "none"
                )
            limit = max_context(info)
            voltar_modelo = False
            while True:
                num_ctx = _choose_context(limit, input_fn, tr)
                if num_ctx is None:
                    voltar_modelo = True
                    break
                thinking = _choose_thinking(chosen, input_fn, tr)
                if thinking == "__context__":
                    continue
                if thinking == "__model__":
                    voltar_modelo = True
                    break
                break
            if not voltar_modelo:
                break

        data = config.load(config_file)
        data["language"] = language
        _save_ollama_profile(data, chosen, num_ctx, limit, thinking)
        config.save(data, config_file)
        return 0
    except (RuntimeError, ValueError, urllib.error.URLError) as e:
        print(tr.t("setup.error", error=e))
        return 1
    except KeyboardInterrupt:
        print("\n" + tr.t("setup.cancelled"))
        return 130
    finally:
        if started and started.poll() is None:
            started.terminate()
            try:
                started.wait(timeout=3)
            except subprocess.TimeoutExpired:
                started.kill()
                started.wait(timeout=3)


def run_setup(input_fn=input, config_file=None):
    # A single alternate buffer keeps the main terminal from showing between
    # steps. Without Ollama, the instructions have to stay visible.
    if not shutil.which("ollama") or not terminal_ui.interactive(input_fn):
        return _run_setup(input_fn, config_file)
    with terminal_ui.alternate_screen(input_fn):
        code = _run_setup(input_fn, config_file)
    if code != 0:
        try:
            language = config.load(config_file).get("language", "en")
        except ValueError:
            language = "en"
        key = "setup.cancelled" if code == 130 else "setup.incomplete"
        print(Translator(language).t(key))
    return code


def _select_configured_api(input_fn, config_file, language, tr):
    data = config.load(config_file)
    api_profiles = [
        (name, item) for name, item in (data.get("profiles") or {}).items()
        if item.get("provider") == "openai_compatible"
    ]
    options = [tr.t("engine.ollama")]
    options.extend(
        f"{item.get('provider_name') or 'API'} · {item.get('model')}"
        for _nome, item in api_profiles
    )
    options.append(tr.t("api.configure.new"))
    current = data.get("default_profile")
    initial = next(
        (i + 1 for i, (name, _item) in enumerate(api_profiles) if name == current),
        0,
    )
    index = _select(
        tr, tr.t("model.source.title"), options, input_fn,
        tr.t("model.source.explain"), initial=initial,
    )
    if index == 0:
        return "__ollama__"
    if index == len(options) - 1:
        return _setup_api(language, input_fn, config_file, tr)

    name, original = api_profiles[index - 1]
    item = dict(original)
    secret_path = Path(config_file).with_name("secrets.json") if config_file else None
    try:
        key = config.load_secret(item.get("credential"), secret_path)
        models = _list_api_models(item["base_url"], key)
    except (RuntimeError, KeyError) as e:
        print(tr.t("api.model.fetch.failed", error=e))
        models = None
    if models:
        atual_modelo = item.get("model")
        inicial_modelo = models.index(atual_modelo) if atual_modelo in models else 0
        escolha_modelo = _select(
            tr, tr.t("api.model.select.title"), models, input_fn,
            tr.t("api.model.select.explain"), initial=inicial_modelo,
        )
        novo_modelo = models[escolha_modelo]
        if novo_modelo != atual_modelo:
            item["model"] = novo_modelo
            # The profile name follows the model (same convention as
            # _setup_api); without this the UI would show the old model's slug
            # after the swap.
            novo_nome = _api_profile_name(item.get("provider_name"), novo_modelo)
            if novo_nome != name and novo_nome not in data["profiles"]:
                data["profiles"].pop(name, None)
                name = novo_nome
    valores = [None, "low", "medium", "high"]
    current = item.get("thinking")
    initial = valores.index(current) if current in valores else 0
    choice = _select(
        tr, tr.t("thinking.api.title"),
        [tr.t("thinking.disabled"), tr.t("thinking.low"),
         tr.t("thinking.medium"), tr.t("thinking.high")],
        input_fn, tr.t("thinking.api.explain"), initial=initial,
    )
    item["thinking"] = valores[choice]
    data["profiles"][name] = item
    data["default_profile"] = name
    config.save(data, config_file)
    return 0


def run_model_selector(input_fn=input, config_file=None):
    """Troca model sem repetir language, workspace ou o restante do setup."""
    try:
        data = config.load(config_file)
        language = data.get("language") or "en"
        tr = Translator(language)
        with terminal_ui.alternate_screen(input_fn):
            source = _select_configured_api(input_fn, config_file, language, tr)
            if source == "__ollama__":
                return _run_setup(
                    input_fn, config_file, initial_language=language, ollama_only=True,
                )
            return source
    except (RuntimeError, ValueError, urllib.error.URLError) as e:
        print(Translator("en").t("setup.error", error=e))
        return 1
    except KeyboardInterrupt:
        print("\n" + Translator("en").t("setup.cancelled"))
        return 130
