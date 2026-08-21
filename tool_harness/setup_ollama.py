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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import agent
import config
import debug
import hardware
import model_discovery
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
    required = {"name", "reference", "benchmark", "benchmark_source", "scores"}
    if not isinstance(models, list) or not models or not all(
            isinstance(item, dict) and required <= item.keys()
            and isinstance(item["name"], str) and item["name"].strip()
            and isinstance(item["reference"], str) and item["reference"].strip()
            and isinstance(item["scores"], dict)
            for item in models):
        raise RuntimeError(Translator().t("setup.catalog.not_list", path=path))
    # This section is the reviewed list, and every screen that draws it says so
    # on the row itself, next to rows that were merely found.
    return [{**item, "curated": True} for item in models]


LOCAL_CATALOG = _load_catalog("local")
RECOMMENDED = [item["reference"] for item in LOCAL_CATALOG]
TASK_VALUES = ("fix_bug", "build_new", "explain_code")
# How much of the accelerator the suggested precision is allowed to claim. A
# launch that runs out of memory still spends the hour it was given.
QUANTIZATION_HEADROOM = 0.9
_UNCHANGED = object()
_LOCAL_RESOLUTION_CACHE = {}
# The model screen has to draw fast and has to work with the network down. Each
# hf.co entry costs three requests, so resolving them one after another at the
# default eight second ceiling can freeze the screen for the best part of a
# minute on a bad link. They go together, under a short ceiling, and whatever
# misses it degrades to the honest "size unknown" state instead of stalling.
# The urlopen seam exists so the suite stays offline, which its first line
# promises.
LOCAL_RESOLUTION_TIMEOUT = 4
LOCAL_RESOLUTION_URLOPEN = urllib.request.urlopen

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


def _model_item(model, recommended=False, catalog=None):
    normalized = model.removesuffix(":latest")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-") or "local"
    item = {
        "id": slug, "name": model, "base_model": model,
        "temperature": 0, "thinking_kind": "detect", "recommended": recommended,
    }
    if catalog:
        item["catalog"] = catalog
    return item


def _recommended_catalog(task=None):
    ordered = model_discovery.order_for_task(LOCAL_CATALOG, task)
    return [
        _model_item(item["reference"], recommended=True, catalog=item)
        for item in ordered
    ]


def _installed_models(installed):
    return [_model_item(model) for model in sorted(installed, key=str.casefold)]


def _is_installed(model, installed):
    wanted = model.removesuffix(":latest").casefold()
    return any(item.removesuffix(":latest").casefold() == wanted for item in installed)


def _model_label(item, installed, tr):
    base = item["base_model"]
    state = ("model.installed" if _is_installed(base, installed)
              else "model.not_installed")
    fit = item.get("fit_label") or tr.t("model.fit.unknown")
    return tr.t("model.option", model=base, state=tr.t(state),
                origin=_origin_label(item, tr), fit=fit)


def _origin_label(item, tr):
    """What stands behind this row: curation, a public score, or nothing.

    A model already installed is neither suggested nor discovered: the user put
    it there, and saying "found live, no evidence" about their own model would
    be describing this program's search instead of their machine.
    """
    evidence = item.get("catalog") or item.get("resolved")
    if evidence is None:
        return tr.t("model.origin.installed")
    return model_discovery.origin_label(evidence, tr.t)


def _task_ruler(task, tr):
    return tr.t(f"onboarding.task.ruler.{task}") if task else ""


def _choose_task(data, input_fn, tr):
    stored = (data.get("onboarding") or {}).get("task")
    initial = TASK_VALUES.index(stored) if stored in TASK_VALUES else 3
    index = _select(
        tr, tr.t("onboarding.task.title"),
        [tr.t(f"onboarding.task.{value}") for value in TASK_VALUES]
        + [tr.t("onboarding.task.skip")],
        input_fn, tr.t("onboarding.task.explain"), initial=initial,
    )
    return TASK_VALUES[index] if index < len(TASK_VALUES) else None


def _store_onboarding(data, task):
    if task:
        data.setdefault("onboarding", {})["task"] = task
        return
    onboarding = data.get("onboarding")
    if isinstance(onboarding, dict):
        onboarding.pop("task", None)
        if not onboarding:
            data.pop("onboarding", None)


def _machine_profile(tr):
    try:
        profile = hardware.detect()
        if not isinstance(profile, dict):
            raise TypeError("hardware.detect returned a non-object")
        ram_mb = float(profile.get("ram_mb") or 0)
        cores = int(profile.get("cpu_cores") or 0)
        gpus = []
        for raw in profile.get("gpus") or []:
            if not isinstance(raw, dict):
                continue
            vram_mb = float(raw.get("vram_mb") or 0)
            if vram_mb <= 0:
                continue
            gpus.append({
                "name": raw.get("name") or tr.t("hardware.unknown"),
                "vram_mb": vram_mb,
            })
    except Exception:
        debug.swallowed("setup_ollama._machine_profile")
        ram_mb = 0
        cores = 0
        gpus = []
    profile = {"gpus": gpus, "ram_mb": ram_mb, "cpu_cores": cores}
    ram_gib = ram_mb / 1024
    if not gpus:
        line = tr.t("hardware.local.no_gpu", ram=f"{ram_gib:.1f}", cores=cores)
    else:
        gpu_labels = tr.t("hardware.local.separator").join(
            tr.t(
                "hardware.local.gpu_item", name=item["name"],
                vram=f"{item['vram_mb'] / 1024:.1f}",
            )
            for item in gpus
        )
        line = tr.t(
            "hardware.local.summary", gpus=gpu_labels, ram=f"{ram_gib:.1f}", cores=cores,
        )
    return profile, line


def _resolve_live(reference):
    """Live metadata for one catalogued reference, or None when unresolvable.

    Returning None is a real answer here: it becomes the "size unknown" state on
    screen, which is the honest thing to say about a model nobody measured.
    """
    if reference in _LOCAL_RESOLUTION_CACHE:
        return _LOCAL_RESOLUTION_CACHE[reference]
    try:
        live = model_discovery.resolve_hf_model(
            reference, catalog_path=MODEL_CATALOG_PATH,
            urlopen_fn=LOCAL_RESOLUTION_URLOPEN,
            timeout=LOCAL_RESOLUTION_TIMEOUT,
        )
    except model_discovery.DiscoveryError as error:
        debug.note(f"setup_ollama._resolve_live {reference}", str(error))
        live = None
    except Exception:
        debug.swallowed(f"setup_ollama._resolve_live {reference}")
        live = None
    # A failure is cached too. Without that, every redraw pays the timeout again
    # for the reference that already failed, which is the slowest case asking to
    # be repeated.
    _LOCAL_RESOLUTION_CACHE[reference] = live
    return live


def _resolved_local_catalog(task, profile, tr):
    vram_mb = sum(
        int(item.get("vram_mb") or 0)
        for item in profile.get("gpus") or [] if isinstance(item, dict)
    )
    gpu_count = len(profile.get("gpus") or [])
    overhead_mb = hardware.DEFAULT_OVERHEAD_MB * max(1, gpu_count)
    ordered = model_discovery.order_for_task(LOCAL_CATALOG, task)
    pending = [
        item["reference"] for item in ordered
        if item["reference"].startswith("hf.co/")
        and item["reference"] not in _LOCAL_RESOLUTION_CACHE
    ]
    if pending:
        with ThreadPoolExecutor(max_workers=len(pending)) as executor:
            list(executor.map(_resolve_live, pending))
    items = []
    for catalog in ordered:
        reference = catalog["reference"]
        resolved = dict(catalog)
        live = _resolve_live(reference) if reference.startswith("hf.co/") else None
        if live:
            resolved.update(live)
            # The live payload carries its own benchmark evidence, derived from
            # the Kaggle seed. The curated entry is the authority for this
            # model, so it wins rather than being overwritten by a neighbour.
            resolved.update({
                "benchmark": catalog["benchmark"],
                "benchmark_source": catalog["benchmark_source"],
                "scores": catalog["scores"],
            })
        complete = all(
            key in resolved
            for key in ("model_bytes", "n_layers", "n_kv_heads", "head_dim")
        )
        item = _model_item(reference, recommended=True, catalog=catalog)
        if not gpu_count:
            # Without a GPU there is no VRAM to fit into, so "does not fit" would
            # be answering a question nobody asked. What the user needs to know
            # is that it runs on CPU and system RAM, and that this is slow.
            item["fit_label"] = (
                tr.t("model.fit.no_gpu_sized",
                     weights=f"{resolved['model_bytes'] / 1024 ** 3:.2f}")
                if complete else tr.t("model.fit.no_gpu")
            )
        elif complete:
            report = model_discovery.fit_report(
                resolved, vram_mb, overhead_mb=overhead_mb,
            )
            item["fit_report"] = report
            item["fit_label"] = model_discovery.format_fit(
                report, tr.t, state_key="model.fit.report",
                fit_yes_key="model.fit.fits", fit_no_key="model.fit.does_not_fit",
            )
        else:
            item["fit_label"] = tr.t("model.fit.unknown")
        items.append(item)
    return items


def _confirm_model_fit(model, input_fn, vram_mb=None, overhead_mb=None):
    """Show exact fit inputs and require consent when the model does not fit."""
    if vram_mb is None:
        vram_mb, gpu_count = model_discovery.local_vram()
        overhead_mb = hardware.DEFAULT_OVERHEAD_MB * max(1, gpu_count)
    report = model_discovery.fit_report(
        model, vram_mb, overhead_mb=overhead_mb or hardware.DEFAULT_OVERHEAD_MB,
    )
    print(model_discovery.format_fit(report))
    print(model_discovery.benchmark_line(model))
    if report["fits"]:
        return report
    answer = input_fn(model_discovery.text("model.discovery.continue")).strip().casefold()
    return report if answer == model_discovery.text("model.discovery.yes") else None


def _choose_quantization(model, input_fn, tr, urlopen_fn=urllib.request.urlopen,
                         vram_mb=None, overhead_mb=None):
    """Offer the other precisions of the model that was just chosen.

    Which model and how much of it are two different questions, and only the
    first was being asked. Fewer bits mean fewer bytes read per token and less
    memory, at a cost in quality nobody here has measured, so the screen states
    size and fit and says nothing about quality.
    """
    try:
        if "files" not in model and model.get("repo") and model.get("file"):
            # A curated row carries the one file that was reviewed, not the
            # shelf it came from. The shelf is one request away, and the
            # evidence stays the curated one.
            live = model_discovery.resolve_hf_model(
                model["repo"], model["file"], urlopen_fn=urlopen_fn)
            model = {**model, "files": live.get("files") or []}
        variants = model_discovery.quantization_variants(
            model, urlopen_fn=urlopen_fn)
    except model_discovery.DiscoveryError as error:
        debug.note("setup_ollama._choose_quantization", str(error))
        return model
    if not variants:
        return model
    if vram_mb is None:
        vram_mb, gpu_count = model_discovery.local_vram()
        overhead_mb = hardware.DEFAULT_OVERHEAD_MB * max(1, gpu_count)
    # Heaviest first, and the cursor starts on the heaviest that fits. Filling
    # the accelerator is the point: a borrowed GPU costs the same hour whether
    # two thirds of it sit idle or not, and more bits of the same weights is
    # quality that this hour is already paying for. What it costs is bytes read
    # per token, so the row says the size and the fit and nothing about speed.
    ranked = sorted([model, *variants],
                    key=lambda item: -item["model_bytes"])
    options, entries, initial, tight = [], [], None, None
    for item in ranked:
        report = model_discovery.fit_report(
            item, vram_mb, overhead_mb=overhead_mb or hardware.DEFAULT_OVERHEAD_MB)
        if report["fits"]:
            available = max(0, report["vram_mb"] - report["overhead_mb"]) * 1024 ** 2
            used = report["model_bytes"] + report["kv_bytes"]
            # The cursor stops on the heaviest that still leaves room, not on
            # the heaviest that arithmetic allows. The one measured overshoot
            # of this estimate was 16% on the safe side, but a launch that runs
            # out of memory spends the hour anyway, so the default keeps a
            # tenth of the card in reserve and the tighter rows stay one keypress
            # away for whoever wants them.
            if initial is None and used <= available * QUANTIZATION_HEADROOM:
                initial = len(entries)
            if tight is None:
                tight = len(entries)
        entries.append(item)
        options.append(tr.t(
            "model.quantization.option",
            quantization=model_discovery.quantization_label(item["file"]),
            size=f"{item['model_bytes'] / 1024 ** 3:.2f}",
            fit=tr.t("model.fit.fits") if report["fits"]
            else tr.t("model.fit.does_not_fit")))
    index = _select(
        tr, tr.t("model.quantization.title"), options, input_fn,
        tr.t("model.quantization.explain"),
        # Nothing leaves room: fall back to whatever fits at all, and if not
        # even that, to the smallest, which is the only one with a chance.
        initial=(initial if initial is not None
                 else tight if tight is not None else len(entries) - 1))
    return entries[index]


def _resolve_custom_ollama(reference, input_fn, catalog_path=MODEL_CATALOG_PATH,
                           urlopen_fn=urllib.request.urlopen, tr=None):
    """Resolve an Ollama reference before pull, or disclose what is unknown."""
    if reference.startswith(("hf.co/", "https://huggingface.co/",
                             "http://huggingface.co/")):
        model = model_discovery.resolve_hf_model(
            reference, catalog_path=catalog_path, urlopen_fn=urlopen_fn,
        )
        chosen = _choose_quantization(
            model, input_fn, tr or Translator(), urlopen_fn)
        report = _confirm_model_fit(chosen, input_fn)
        if report is None:
            return None
        # What the user typed is what they meant, and Ollama accepts it. Only a
        # file changed on the quantization screen has to be renamed, because
        # what they typed no longer names the file they picked.
        item = _model_item(
            reference if chosen["file"] == model["file"]
            else model_discovery.ollama_reference(chosen))
        item["resolved"] = report
        return item
    print(model_discovery.text(
        "model.discovery.unresolved",
        error=model_discovery.text("model.discovery.ollama_unresolved"),
    ))
    answer = input_fn(model_discovery.text("model.discovery.continue")).strip().casefold()
    return (_model_item(reference) if
            answer == model_discovery.text("model.discovery.yes") else None)


def _choose_other_ollama(input_fn, tr, catalog_path=MODEL_CATALOG_PATH,
                         urlopen_fn=urllib.request.urlopen):
    """One secondary screen for live discovery and exact Ollama references."""
    try:
        discovered, errors = model_discovery.discover_models(
            catalog_path, urlopen_fn=urlopen_fn,
        )
    except model_discovery.DiscoveryError as error:
        discovered, errors = [], [str(error)]
    # Printing the causes here and then drawing a screen put them on the page the
    # alternate screen replaces, so on a real terminal nobody ever read them.
    # A candidate that failed explains why the list is shorter and belongs in
    # --debug; a discovery that returned nothing at all is the answer to what the
    # user just asked for, so its cause goes on the screen they are looking at.
    for error in errors:
        debug.note("setup_ollama._choose_other_ollama discovery", error)
    explanation = None if discovered else "\n".join(
        model_discovery.text("model.discovery.failed", error=error)
        for error in errors) or None
    # What the user asked is "what can I run", so the answer is drawn against
    # their machine, the way the curated screen already is. A model that does
    # not fit still appears, saying so: hiding it would make the list look like
    # the whole of Hugging Face fits in this GPU.
    vram_mb, gpu_count = model_discovery.local_vram()
    overhead_mb = hardware.DEFAULT_OVERHEAD_MB * max(1, gpu_count)
    ranked = []
    for item in discovered:
        complete = all(key in item for key in
                       ("model_bytes", "n_layers", "n_kv_heads", "head_dim"))
        if not complete:
            ranked.append((dict(item, fits=False), tr.t("model.fit.unknown")))
            continue
        report = model_discovery.fit_report(item, vram_mb, overhead_mb=overhead_mb)
        if not gpu_count:
            label = tr.t("model.fit.no_gpu_sized",
                         weights=f"{item['model_bytes'] / 1024 ** 3:.2f}")
        else:
            label = tr.t("model.fit.fits") if report["fits"] else tr.t(
                "model.fit.does_not_fit")
        ranked.append((report, label))
    ranked.sort(key=lambda entry: not entry[0]["fits"])
    discovered = [report for report, _label in ranked]
    entries = [*discovered, "__exact__", "__back__"]
    options = [
        *[
            tr.t(
                "model.discovery.entry", name=report["name"],
                size=f"{report['model_bytes'] / 1024 ** 3:.2f}",
                origin=model_discovery.origin_label(report, tr.t),
                benchmark=report["benchmark"], fit=label)
            for report, label in ranked
        ],
        model_discovery.text("model.discovery.exact"),
        model_discovery.text("model.discovery.back"),
    ]
    index = _select(
        tr, model_discovery.text("model.discovery.section"), options, input_fn,
        explanation,
    )
    chosen = entries[index]
    if chosen == "__back__":
        return None
    if chosen == "__exact__":
        reference = input_fn(model_discovery.text("model.discovery.prompt")).strip()
        if not reference:
            return None
        try:
            return _resolve_custom_ollama(
                reference, input_fn, catalog_path, urlopen_fn, tr,
            )
        except model_discovery.DiscoveryError as error:
            print(model_discovery.text("model.discovery.unresolved", error=error))
            return None
    chosen = _choose_quantization(chosen, input_fn, tr, urlopen_fn)
    report = _confirm_model_fit(chosen, input_fn)
    if report is None:
        return None
    item = _model_item(model_discovery.ollama_reference(chosen))
    item["resolved"] = report
    return item


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


def _setup_api(language, input_fn, config_file, tr, onboarding_task=_UNCHANGED):
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
    if onboarding_task is not _UNCHANGED:
        _store_onboarding(data, onboarding_task)
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


def _setup_kaggle(language, input_fn, config_file, onboarding_task=_UNCHANGED):
    """One Kaggle configuration path shared by setup and the model selector."""
    from cli_i18n import set_language
    import cli_kaggle

    data = config.load(config_file)
    data["language"] = language
    if onboarding_task is not _UNCHANGED:
        _store_onboarding(data, onboarding_task)
    config.save(data, config_file)
    set_language(language)
    return run_kaggle(
        input_fn=input_fn, config_file=config_file, onboarding_task=onboarding_task,
    )


def _kaggle_fit(model, accelerator):
    """Restate fit for this accelerator after the file underneath changed."""
    report = model_discovery.fit_report(
        model, accelerator["vram_mb"], overhead_mb=accelerator["overhead_mb"])
    report.update({
        "machine_shape": next(
            name for name, item in _kaggle_accelerators().items()
            if item is accelerator),
        "machine_label": accelerator["label"],
        "cuda_arch": accelerator["cuda_arch"],
        "gpu_count": accelerator["gpu_count"],
    })
    return report


def _kaggle_accelerators():
    import cli_kaggle

    return cli_kaggle.ACCELERATORS


def _dynamic_kaggle_selector(input_fn, catalog_path=MODEL_CATALOG_PATH,
                             urlopen_fn=urllib.request.urlopen, onboarding_task=None):
    """Combine the offline seed with live GGUF discovery for Kaggle."""
    import cli_kaggle

    seeded = cli_kaggle._load_model_candidates(catalog_path)
    try:
        discovered, errors = model_discovery.discover_models(
            catalog_path, urlopen_fn=urlopen_fn,
        )
    except model_discovery.DiscoveryError as error:
        discovered, errors = [], [str(error)]
    # A repository that publishes no Q4_K_M is not a problem the user has to
    # read about while choosing a model. It is why one candidate is missing from
    # a list they never saw, which is exactly what --debug is for.
    for error in errors:
        debug.note("setup_ollama._dynamic_kaggle_selector discovery", error)
    merged = {}
    for item in [*seeded, *discovered]:
        # The curated entry wins a collision. It carries the reviewed name and
        # the benchmark evidence, and letting live discovery overwrite it
        # replaced a curated row with a duplicate of itself under a repo path.
        merged.setdefault((item["repo"].casefold(), item["file"].casefold()), item)
    accelerator = cli_kaggle.ACCELERATORS["NvidiaTeslaT4"]
    models = []
    for item in merged.values():
        report = model_discovery.fit_report(
            item, accelerator["vram_mb"],
            overhead_mb=accelerator["overhead_mb"],
        )
        if report["fits"]:
            report.update({
                "machine_shape": "NvidiaTeslaT4",
                "machine_label": accelerator["label"],
                "cuda_arch": accelerator["cuda_arch"],
                "gpu_count": accelerator["gpu_count"],
            })
            models.append(report)
    models = model_discovery.order_for_task(models, onboarding_task)
    # The ruler belongs on the screen whose order it decided. Announcing it in
    # setup and then drawing a differently ordered list without it leaves the
    # order looking arbitrary.
    # This screen assigns T4 x2 to every candidate, so it may not describe the
    # P100 the seeded catalogue screen can still choose. Saying a smaller
    # accelerator is used when the model fits it, next to rows that all read
    # "T4 x2", would be describing a path this screen does not take.
    explanation = [model_discovery.text("cli.kaggle.models.title_t4")]
    if onboarding_task:
        explanation.append(model_discovery.text(
            f"onboarding.task.ruler.{onboarding_task}"))
    selected = cli_kaggle._choose(
        "\n\n".join([
            model_discovery.text("cli.kaggle.models.section"), *explanation]),
        [
            *[cli_kaggle.model_entry(item, model_discovery.NO_PUBLIC_SCORE)
              for item in models],
            model_discovery.text("model.discovery.exact"),
        ],
        input_fn,
    )
    if selected < len(models):
        model = _choose_quantization(
            models[selected], input_fn, Translator(), urlopen_fn,
            vram_mb=accelerator["vram_mb"],
            overhead_mb=accelerator["overhead_mb"])
        model = _kaggle_fit(model, accelerator)
        cli_kaggle.print_model_evidence(model)
        return model
    reference = input_fn(model_discovery.text("model.discovery.prompt")).strip()
    try:
        repo, selected_file = model_discovery.parse_hf_reference(reference)
        if selected_file is None:
            selected_file = input_fn(
                model_discovery.text("model.discovery.file_prompt")
            ).strip()
        model = model_discovery.resolve_hf_model(
            repo, selected_file, catalog_path, urlopen_fn=urlopen_fn,
        )
    except model_discovery.DiscoveryError as error:
        raise RuntimeError(str(error)) from error
    model = _choose_quantization(
        model, input_fn, Translator(), urlopen_fn,
        vram_mb=accelerator["vram_mb"], overhead_mb=accelerator["overhead_mb"])
    report = model_discovery.fit_report(
        model, accelerator["vram_mb"], overhead_mb=accelerator["overhead_mb"],
    )
    print(model_discovery.format_fit(report))
    print(model_discovery.benchmark_line(model))
    if not report["fits"]:
        answer = input_fn(model_discovery.text("model.discovery.continue")).strip().casefold()
        if answer != model_discovery.text("model.discovery.yes"):
            raise RuntimeError(model_discovery.text("model.discovery.not_selected"))
    report.update({
        "machine_shape": "NvidiaTeslaT4",
        "machine_label": accelerator["label"],
        "cuda_arch": accelerator["cuda_arch"],
        "gpu_count": accelerator["gpu_count"],
    })
    return report


def _render_dynamic_kaggle_kernel(folder, slug, model, api_key,
                                  validation_cpu=False):
    """Render the shared self-contained kernel for a discovered model."""
    import cli_kaggle

    return cli_kaggle._render_kernel(
        folder, slug, model, api_key, validation_cpu, dataset_sources=[])


def run_kaggle(**kwargs):
    """Run cli_kaggle through the shared dynamic selector without duplicating flow."""
    import cli_kaggle

    return _with_dynamic_selector(
        cli_kaggle.run_kaggle, kwargs,
        kwargs.get("urlopen_fn", urllib.request.urlopen))


def run_prepare_assets(**kwargs):
    """Prepare the reusable assets from the same screens the launch uses.

    Preparation reaches Hugging Face to draw the model screen, but never reaches
    a saved endpoint, so the probe function is for the screen alone and does not
    travel into the entry point.
    """
    import cli_kaggle

    return _with_dynamic_selector(
        cli_kaggle.run_prepare_assets, kwargs,
        kwargs.pop("urlopen_fn", urllib.request.urlopen))


def _with_dynamic_selector(entry, kwargs, urlopen_fn):
    """Lend one Kaggle entry point the merged offline plus live model screen."""
    import cli_kaggle

    original_select = cli_kaggle._select_model
    onboarding_task = kwargs.pop("onboarding_task", _UNCHANGED)
    if onboarding_task is _UNCHANGED:
        try:
            onboarding_task = (
                config.load(kwargs.get("config_file")).get("onboarding") or {}
            ).get("task")
        except ValueError:
            debug.swallowed("setup_ollama._with_dynamic_selector onboarding")
            onboarding_task = None
    cli_kaggle._select_model = lambda _input, catalog_path=MODEL_CATALOG_PATH: (
        _dynamic_kaggle_selector(
            _input, catalog_path=catalog_path, urlopen_fn=urlopen_fn,
            onboarding_task=onboarding_task,
        )
    )
    try:
        return entry(**kwargs)
    finally:
        cli_kaggle._select_model = original_select


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
    from cli_i18n import set_language
    set_language(language)
    tr = Translator(language)
    print(tr.t("setup.title"), "\n")
    onboarding_task = _UNCHANGED
    ruler_line = ""

    try:
        if not ollama_only:
            setup_data = config.load(config_file)
            previous_task = (setup_data.get("onboarding") or {}).get("task")
            onboarding_task = _choose_task(setup_data, input_fn, tr)
            _store_onboarding(setup_data, onboarding_task)
            if onboarding_task or previous_task:
                config.save(setup_data, config_file)
            ruler_line = _task_ruler(onboarding_task, tr)
            engine_explanation = tr.t("engine.explain")
            if ruler_line:
                engine_explanation += "\n" + ruler_line
            engine = _select(
                tr, tr.t("engine.title"),
                [tr.t("engine.ollama"), tr.t("engine.api"),
                 tr.t("engine.kaggle")], input_fn,
                engine_explanation,
            )
            if engine == 1:
                api_result = _setup_api(
                    language, input_fn, config_file, tr, onboarding_task,
                )
                if api_result == "__engine__":
                    return _run_setup(input_fn, config_file, initial_language=language)
                return api_result
            if engine == 2:
                return _setup_kaggle(
                    language, input_fn, config_file, onboarding_task,
                )
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
        machine, machine_line = _machine_profile(tr)
        # /model does not ask the task again, but it must not contradict the
        # answer either: reordering the same list differently in the two screens
        # would read as the recommendation having changed.
        if onboarding_task is _UNCHANGED:
            screen_task = (current_config.get("onboarding") or {}).get("task")
            ruler_line = _task_ruler(screen_task, tr)
        else:
            screen_task = onboarding_task
        recommended_items = _resolved_local_catalog(screen_task, machine, tr)
        configured_derivatives = {
            profile.get("model", "").removesuffix(":latest").casefold()
            for profile in (current_config.get("profiles") or {}).values()
            if profile.get("provider", "ollama") == "ollama"
            and profile.get("base_model")
            and profile.get("model") != profile.get("base_model")
        }
        while True:
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
            entries = [
                None, *recommended_items, None, *local_items,
                "__back__", "__other__",
            ]
            options = [
                tr.t("model.section.recommended"),
                *[_model_label(item, installed, tr) for item in recommended_items],
                tr.t("model.section.installed", count=len(local_items)),
                *[item["base_model"] for item in local_items],
                tr.t("navigation.back"),
                model_discovery.text("model.discovery.other"),
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
                    "\n".join(filter(None, [
                        tr.t("model.recommended.explain", version=version),
                        machine_line,
                        ruler_line,
                        tr.t("model.benchmark.scope"),
                    ])), input_fn,
                ),
                initial=initial,
                disabled=headers,
            )
            chosen = entries[model_index]
            if chosen == "__back__":
                if ollama_only:
                    return 130
                return _run_setup(input_fn, config_file, initial_language=language)
            if chosen == "__other__":
                chosen = _choose_other_ollama(input_fn, tr)
                if chosen is None:
                    continue
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
        if onboarding_task is not _UNCHANGED:
            _store_onboarding(data, onboarding_task)
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
    kaggle_profiles = [
        (name, item) for name, item in (data.get("profiles") or {}).items()
        if item.get("provider") == "openai_compatible"
        and item.get("provider_name") == "Kaggle"
    ]
    api_profiles = [
        (name, item) for name, item in (data.get("profiles") or {}).items()
        if item.get("provider") == "openai_compatible"
        and item.get("provider_name") != "Kaggle"
    ]
    kaggle_state = "model.configured" if kaggle_profiles else "model.not_installed"
    options = [
        tr.t("engine.ollama"),
        tr.t("engine.kaggle.state", state=tr.t(kaggle_state)),
    ]
    options.extend(
        f"{item.get('provider_name') or 'API'} · {item.get('model')}"
        for _nome, item in api_profiles
    )
    options.append(tr.t("api.configure.new"))
    current = data.get("default_profile")
    if any(name == current for name, _item in kaggle_profiles):
        initial = 1
    else:
        initial = next(
            (i + 2 for i, (name, _item) in enumerate(api_profiles) if name == current),
            0,
        )
    index = _select(
        tr, tr.t("model.source.title"), options, input_fn,
        tr.t("model.source.explain"), initial=initial,
    )
    if index == 0:
        return "__ollama__"
    if index == 1:
        return _setup_kaggle(language, input_fn, config_file)
    if index == len(options) - 1:
        return _setup_api(language, input_fn, config_file, tr)

    name, original = api_profiles[index - 2]
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
