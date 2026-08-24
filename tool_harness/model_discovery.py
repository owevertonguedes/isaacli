"""Discover GGUF models and resolve exact Hugging Face metadata.

The curated JSON remains the offline seed and benchmark authority. Live
discovery can add candidates, but it cannot manufacture quality evidence.
"""
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import agent
import debug
import hardware
import units
from cli_i18n import t


HF_ROOT = "https://huggingface.co"
HF_API = HF_ROOT + "/api/models"
DEFAULT_TIMEOUT = 8
DEFAULT_CONTEXT = 16384
# A benchmark string is a proper noun and a number and stays as published. The
# absence of one is a sentence this program writes, so it is translated like
# every other sentence: a Portuguese screen was printing "no public score on the
# accepted coding benchmarks" in the middle of it.
NO_PUBLIC_SCORE_KEY = "model.score.none"


def no_public_score(translate=None):
    return (translate or text)(NO_PUBLIC_SCORE_KEY)
TASK_RULERS = {
    "fix_bug": (
        "swebench_verified", "swebench_lite", "swebench_pro", "aider_polyglot",
    ),
    "build_new": ("aider_polyglot", "livecodebench_v6"),
    "explain_code": ("gpqa_diamond",),
}


class DiscoveryError(RuntimeError):
    pass


def text(key, **values):
    return t(key, **values)


def _json_request(url, timeout=DEFAULT_TIMEOUT, urlopen_fn=urllib.request.urlopen):
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": agent.USER_AGENT},
    )
    try:
        with urlopen_fn(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise DiscoveryError(text(
            "model.discovery.error.http", code=error.code, url=url,
        )) from error
    except urllib.error.URLError as error:
        raise DiscoveryError(text(
            "model.discovery.error.request", reason=error.reason, url=url,
        )) from error
    except (OSError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DiscoveryError(text(
            "model.discovery.error.request", reason=error, url=url,
        )) from error


def _content_length(url, timeout=DEFAULT_TIMEOUT,
                    urlopen_fn=urllib.request.urlopen):
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": agent.USER_AGENT},
    )
    try:
        with urlopen_fn(request, timeout=timeout) as response:
            value = response.headers.get("Content-Length")
    except urllib.error.HTTPError as error:
        raise DiscoveryError(text(
            "model.discovery.error.size_http", code=error.code,
        )) from error
    except urllib.error.URLError as error:
        raise DiscoveryError(text(
            "model.discovery.error.size_request", reason=error.reason,
        )) from error
    except (OSError, TimeoutError) as error:
        raise DiscoveryError(text(
            "model.discovery.error.size_request", reason=error,
        )) from error
    try:
        size = int(value)
    except (TypeError, ValueError) as error:
        raise DiscoveryError(text("model.discovery.error.size_missing")) from error
    if size <= 0:
        raise DiscoveryError(text("model.discovery.error.size_invalid"))
    return size


def parse_hf_reference(reference, file_name=None):
    """Return (repo, file or selector) for supported HF and Ollama syntax."""
    raw = str(reference or "").strip()
    if not raw:
        raise DiscoveryError(text("model.discovery.error.empty"))
    if raw.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(raw)
        if parsed.hostname not in {"huggingface.co", "www.huggingface.co"}:
            raise DiscoveryError(text("model.discovery.error.host"))
        parts = [urllib.parse.unquote(item) for item in parsed.path.split("/") if item]
        if len(parts) < 2:
            raise DiscoveryError(text("model.discovery.error.url_repo"))
        repo = "/".join(parts[:2])
        if len(parts) >= 5 and parts[2] in {"blob", "resolve"}:
            return repo, "/".join(parts[4:])
        return repo, file_name
    if raw.startswith("hf.co/"):
        raw = raw[len("hf.co/"):]
        repo, separator, selector = raw.partition(":")
        if len(repo.split("/")) != 2:
            raise DiscoveryError(text("model.discovery.error.hf_syntax"))
        return repo, selector if separator else file_name
    parts = raw.split()
    repo = parts[0]
    selected_file = file_name or (parts[1] if len(parts) == 2 else None)
    if len(repo.split("/")) != 2 or len(parts) > 2:
        raise DiscoveryError(text("model.discovery.error.syntax"))
    return repo, selected_file


def _model_id(payload):
    return payload.get("id") or payload.get("modelId")


def _base_model(payload):
    card = payload.get("cardData") or {}
    base = card.get("base_model") or payload.get("base_model")
    if isinstance(base, list):
        base = next((item for item in base if isinstance(item, str)), None)
    if isinstance(base, dict):
        base = base.get("name") or base.get("id")
    return base if isinstance(base, str) and len(base.split("/")) == 2 else None


# What a repository publishes is not a name this program chose, and the file
# name ends up inside a Python literal in the kernel Kaggle runs on the user's
# account. Rendering refuses an unusable name, which is the guarantee; leaving
# it out here as well means it is never offered in the first place, so the
# refusal cannot arrive after a model has been chosen.
SAFE_GGUF_NAME = re.compile(r"[A-Za-z0-9_./+ -]+")


def _gguf_files(payload):
    names = []
    for sibling in payload.get("siblings") or []:
        name = sibling.get("rfilename") if isinstance(sibling, dict) else None
        if not isinstance(name, str) or not name.lower().endswith(".gguf"):
            continue
        if ".." in name or not SAFE_GGUF_NAME.fullmatch(name):
            debug.note("model_discovery._gguf_files",
                       f"skipping a file whose name cannot be used safely: {name[:80]}")
            continue
        names.append(name)
    return names


def _select_gguf(files, selector=None):
    if not files:
        raise DiscoveryError(text("model.discovery.error.no_gguf"))
    if selector:
        wanted = selector.casefold()
        exact = [name for name in files if name.casefold() == wanted]
        if exact:
            return exact[0]
        normalized = re.sub(r"[^a-z0-9]", "", wanted)
        matches = [name for name in files
                   if normalized in re.sub(r"[^a-z0-9]", "", name.casefold())]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise DiscoveryError(text(
                "model.discovery.error.no_match", selector=selector,
            ))
        raise DiscoveryError(text(
            "model.discovery.error.many_matches", selector=selector,
        ))
    preferred = [name for name in files if "q4_k_m" in name.casefold()]
    if preferred:
        return min(preferred, key=len)
    raise DiscoveryError(text("model.discovery.error.file_required"))


def _geometry(config_payload):
    payload = config_payload.get("text_config") or config_payload
    try:
        layers = int(payload["num_hidden_layers"])
        kv_heads = int(payload["num_key_value_heads"])
        head_dim = payload.get("head_dim")
        if head_dim is None:
            head_dim = int(payload["hidden_size"]) // int(payload["num_attention_heads"])
        head_dim = int(head_dim)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise DiscoveryError(text("model.discovery.error.geometry")) from error
    experts = payload.get("num_experts", payload.get("num_local_experts"))
    active = payload.get("num_experts_per_tok", payload.get("num_selected_experts"))
    if experts is None and active is None:
        ratio = 1.0
    else:
        try:
            ratio = float(active) / float(experts)
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise DiscoveryError(text("model.discovery.error.experts")) from error
        if not 0 < ratio <= 1:
            raise DiscoveryError(text("model.discovery.error.ratio"))
    return layers, kv_heads, head_dim, ratio


def _seed_maps(catalog_path):
    try:
        items = json.loads(Path(catalog_path).read_text(encoding="utf-8"))["kaggle"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise DiscoveryError(text(
            "model.discovery.error.catalog", error=error,
        )) from error
    by_gguf = {}
    by_upstream = {}
    for item in items:
        repo = item.get("repo")
        source = item.get("benchmark_source", "")
        upstream = "/".join(urllib.parse.urlparse(source).path.strip("/").split("/")[:2])
        evidence = {
            "benchmark": item.get("benchmark") or "",
            "benchmark_source": source or None,
            "upstream_repo": upstream or None,
            "scores": item.get("scores") or {},
        }
        if repo:
            by_gguf[repo.casefold()] = evidence
        if upstream:
            by_upstream[upstream.casefold()] = evidence
    return by_gguf, by_upstream


def _is_plain_quantization(repo, upstream):
    """Whether `repo` is just `upstream` requantized, rather than a new model.

    A repackaging keeps the model name and adds only a format suffix, so
    `unsloth/Qwen3.8-27B-GGUF` is the same weights as `Qwen/Qwen3.8-27B`. Any
    extra word, `Uncensored`, `abliterated`, `MTP` and the rest, describes a
    model that was changed and therefore was never the one that was measured.
    """
    if not upstream:
        return False
    name = re.sub(r"[-_.]?gguf$", "", str(repo).split("/")[-1], flags=re.I)
    return name.casefold() == str(upstream).split("/")[-1].casefold()


def resolve_hf_model(reference, file_name=None, catalog_path=None,
                     urlopen_fn=urllib.request.urlopen, timeout=DEFAULT_TIMEOUT):
    """Resolve one exact GGUF without downloading its body."""
    repo, selector = parse_hf_reference(reference, file_name)
    if (not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
            or selector and (".." in selector or not re.fullmatch(
                r"[A-Za-z0-9_./+ -]+", selector))):
        raise DiscoveryError(text("model.discovery.error.unsafe"))
    quoted_repo = urllib.parse.quote(repo, safe="/")
    repo_payload = _json_request(
        f"{HF_API}/{quoted_repo}", timeout=timeout, urlopen_fn=urlopen_fn,
    )
    if repo_payload.get("gated"):
        # Hugging Face answers the metadata of a gated repository and refuses
        # its files, so the download only fails after the user has chosen it and
        # a kernel is already spending quota. Ask now, while it is still free.
        raise DiscoveryError(text("model.discovery.error.gated", repo=repo))
    selected_file = _select_gguf(_gguf_files(repo_payload), selector)
    upstream = _base_model(repo_payload)
    # A repository that declares a base model and does not merely requantize it
    # is a changed model: Uncensored, abliterated, a merge, an aggressive MTP
    # graft. It may be exactly what somebody wants, and it stays reachable by
    # exact reference, but it was never the model anybody measured, so it is not
    # something this program puts in front of a user as a suggestion.
    derived = bool(upstream) and not _is_plain_quantization(repo, upstream)
    derived_from = upstream if derived else None
    by_gguf, by_upstream = _seed_maps(catalog_path) if catalog_path else ({}, {})
    evidence = by_gguf.get(repo.casefold())
    if evidence and not upstream:
        upstream = evidence.get("upstream_repo")
    if not upstream:
        guessed = re.sub(r"-gguf$", "", repo, flags=re.I)
        upstream = guessed if guessed.casefold() in by_upstream else None
    config_repo = upstream or repo
    config_url = f"{HF_ROOT}/{urllib.parse.quote(config_repo, safe='/')}/resolve/main/config.json"
    config_payload = _json_request(config_url, timeout=timeout, urlopen_fn=urlopen_fn)
    layers, kv_heads, head_dim, active_ratio = _geometry(config_payload)
    # Geometry may come from the upstream model, because a derivative keeps the
    # architecture. A score may not. An uncensored or otherwise modified build
    # declares the official model as its base, and inheriting the base model's
    # numbers would put "LiveCodeBench v6 90.3" next to a model nobody measured.
    # Seen on screen against the live Hugging Face API, which is how it was
    # caught. Only a plain quantization of the catalogued model keeps the score.
    if evidence is None and not _is_plain_quantization(repo, upstream):
        upstream = None
    file_url = (f"{HF_ROOT}/{quoted_repo}/resolve/main/"
                f"{urllib.parse.quote(selected_file, safe='/')}")
    model_bytes = _content_length(file_url, timeout=timeout, urlopen_fn=urlopen_fn)
    evidence = evidence or by_upstream.get((upstream or "").casefold()) or {}
    benchmark = evidence.get("benchmark") or ""
    alias = re.sub(
        r"[^a-z0-9]+", "-", f"{repo}-{Path(selected_file).stem}".casefold(),
    ).strip("-")
    return {
        "name": f"{repo}, {Path(selected_file).stem}",
        "repo": repo,
        "file": selected_file,
        "alias": alias,
        "source": f"{HF_ROOT}/{repo}",
        "model_bytes": model_bytes,
        "n_layers": layers,
        "n_kv_heads": kv_heads,
        "head_dim": head_dim,
        "active_ratio": active_ratio,
        "benchmark": benchmark,
        "benchmark_source": evidence.get("benchmark_source"),
        "scores": evidence.get("scores") or {},
        "benchmark_scope": "original weights, not quantized GGUF",
        "upstream_repo": upstream,
        "downloads": repo_payload.get("downloads", 0),
        "file_url": file_url,
        "files": _gguf_files(repo_payload),
        "curated": bool(by_gguf.get(repo.casefold())),
        "derived": derived,
        "derived_from": derived_from,
    }


SPLIT_GGUF = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.I)
# The precision always sits at the end of the file name, so the pattern is
# anchored there. Anchoring is what keeps `UD-Q6_K_L` from being read as
# `UD-Q6_K`: three different files were collapsing onto one label, and the
# screen showed three identical rows with three different sizes.
QUANTIZATION = re.compile(
    r"(?i)(?:^|[-_.])((?:ud[-_.])?(?:i?q\d+|bf16|f16|f32)(?:[-_.][a-z0-9]+)*)"
    r"\.gguf$")


def quantization_label(file_name):
    match = QUANTIZATION.search(file_name)
    return match.group(1) if match else Path(file_name).stem


def _weight_stem(file_name):
    """The model name a weight file carries, without its precision.

    A repository holds more than weights: `imatrix_unsloth.gguf` is calibration
    data, `mmproj-BF16.gguf` is a projector, `MTP/mtp-*.gguf` is a speculative
    head. All three were being offered as if they were the model, one of them at
    0.01 GiB and reported as fitting. What separates them is not a list of
    forbidden names, which would age: a real precision of this model carries
    this model's name and a precision, and lives beside it.
    """
    match = QUANTIZATION.search(file_name)
    if not match or "/" in file_name:
        return None
    return file_name[:match.start()].strip("-_.").casefold()


def quantization_variants(model, urlopen_fn=urllib.request.urlopen,
                          timeout=DEFAULT_TIMEOUT, limit=24):
    """The same weights at the other precisions the repository publishes.

    Choosing a model and choosing how much of it to keep are two questions, and
    only the first was ever asked: discovery picked one Q4_K_M per repository
    and the rest of the shelf was invisible unless the exact file name was
    typed. Fewer bits is fewer bytes read per token and less memory, at a cost
    in quality that this program has not measured and does not claim.

    Geometry and evidence belong to the weights, so they are carried over
    unchanged; only the file, its size and what follows from the size differ.
    Split files are left out: their `Content-Length` is one part of a model, and
    presenting a part as the whole would misstate both fit and speed.
    """
    stem = _weight_stem(model["file"])
    others = [
        name for name in model.get("files") or []
        if name != model["file"] and not SPLIT_GGUF.search(name)
        and _weight_stem(name) == stem and stem is not None
    ][:limit]
    if not others:
        return []
    repo = urllib.parse.quote(model["repo"], safe="/")
    variants = []
    with ThreadPoolExecutor(max_workers=min(6, len(others))) as executor:
        pending = {}
        for name in others:
            url = f"{HF_ROOT}/{repo}/resolve/main/{urllib.parse.quote(name, safe='/')}"
            pending[executor.submit(
                _content_length, url, timeout=timeout, urlopen_fn=urlopen_fn,
            )] = (name, url)
        for future in as_completed(pending):
            name, url = pending[future]
            try:
                size = future.result()
            except DiscoveryError as error:
                debug.note(f"model_discovery.quantization_variants {name}", str(error))
                continue
            variant = dict(model)
            variant.update({
                "file": name,
                "file_url": url,
                "model_bytes": size,
                "name": f"{model['repo']}, {Path(name).stem}",
                "alias": re.sub(
                    r"[^a-z0-9]+", "-",
                    f"{model['repo']}-{Path(name).stem}".casefold()).strip("-"),
            })
            variants.append(variant)
    order = {name: position for position, name in enumerate(others)}
    variants.sort(key=lambda item: order[item["file"]])
    return variants


def origin(model):
    """Where a row's authority comes from: curation, a public score, or nothing.

    The user reads a list of suggestions and has no way to tell which rows this
    program stands behind. Saying it on each row is what keeps a live search
    result from borrowing the standing of a reviewed one.

    A row measured on this machine outranks a curated one, because curation is
    somebody's judgement and a measurement is a number anybody can re-run.
    """
    if local_measurement(model):
        return "measured"
    if model.get("curated"):
        return "curated"
    return "scored" if model.get("scores") else "discovered"


def origin_label(model, translate=None):
    translate = translate or text
    return translate("model.origin." + origin(model))


def discover_models(catalog_path, search=None, limit=6,
                    urlopen_fn=urllib.request.urlopen, timeout=DEFAULT_TIMEOUT,
                    include_derived=False):
    """Discover and resolve live candidates while preserving search order."""
    query = {"filter": "gguf", "limit": str(limit)}
    if search:
        query["search"] = search
    payload = _json_request(
        HF_API + "?" + urllib.parse.urlencode(query),
        timeout=timeout, urlopen_fn=urlopen_fn,
    )
    if not isinstance(payload, list):
        raise DiscoveryError(text("model.discovery.error.search_json"))
    resolved = {}
    errors = []
    repos = [_model_id(item) for item in payload if isinstance(item, dict)]
    repos = [repo for repo in repos if repo]
    with ThreadPoolExecutor(max_workers=min(6, len(repos) or 1)) as executor:
        pending = {
            executor.submit(
                resolve_hf_model, repo, catalog_path=catalog_path,
                urlopen_fn=urlopen_fn, timeout=timeout,
            ): repo
            for repo in repos
        }
        for future in as_completed(pending):
            repo = pending[future]
            try:
                resolved[repo] = future.result()
            except DiscoveryError as error:
                errors.append(f"{repo}: {error}")
    models = []
    for repo in repos:
        model = resolved.get(repo)
        if model is None:
            continue
        if model.get("derived") and not include_derived:
            errors.append(text(
                "model.discovery.error.derived", repo=repo,
                upstream=model.get("derived_from") or "?"))
            continue
        models.append(model)
    return models, errors


def fit_report(model, vram_mb, overhead_mb=hardware.DEFAULT_OVERHEAD_MB,
               context=DEFAULT_CONTEXT):
    kv_bytes = hardware.kv_cache_bytes(
        model["n_layers"], model["n_kv_heads"], model["head_dim"], context,
    )
    result = dict(model)
    result.update({
        "kv_bytes": kv_bytes,
        "context": context,
        "vram_mb": vram_mb,
        "overhead_mb": overhead_mb,
        "fits": hardware.fits(
            model["model_bytes"], kv_bytes, vram_mb, overhead_mb=overhead_mb,
        ),
        "bytes_per_token": hardware.bytes_read_per_token(
            model["model_bytes"], model.get("active_ratio", 1.0),
        ),
    })
    return result


def format_fit(report, translate=None, state_key="model.discovery.fit",
               fit_yes_key="model.discovery.fit_yes",
               fit_no_key="model.discovery.fit_no"):
    translate = translate or text
    available = max(0, report["vram_mb"] - report["overhead_mb"]) * 1024 ** 2
    return translate(
        state_key,
        fits=translate(fit_yes_key) if report["fits"] else translate(fit_no_key),
        weights=units.gib(report["model_bytes"]),
        kv=units.gib(report["kv_bytes"]),
        total=units.gib(report["model_bytes"] + report["kv_bytes"]),
        available=units.gib(available),
    )


def benchmark_line(model):
    """The evidence behind a model, printed once the choice has been made.

    Both halves are printed when both exist, and each keeps its own owner. The
    line used to carry only the public score, which meant a model measured here
    and scored nowhere read as "no public score" and nothing else, hiding the
    only evidence anybody can re-run.
    """
    lines = []
    measured = model.get("measured_here")
    if measured:
        lines.append(text(
            "model.discovery.measured_line",
            humaneval=measured["humaneval"],
            tps=units.tps(measured["tokens_per_second"]),
            gpu=measured["gpu"],
            date=measured["date"],
            tools=text("model.score.measured_tools_yes"
                       if measured.get("native_tool_call")
                       else "model.score.measured_tools_no"),
            report=measured["report"]))
    lines.append(text("model.discovery.score",
                      score=model.get("benchmark") or no_public_score()))
    return "\n".join(lines)


def carried_measurement(catalog, resolved_file):
    """The catalogue's local measurement, but only for the file it measured.

    A catalogue row names a repository and a precision, and live resolution
    turns that into one file. Nothing guarantees the file it picks today is the
    file the benchmark ran on: a repository can gain a second file matching the
    same precision, and then the row would carry a measurement of a different
    artifact, which is the derivative-score mistake wearing a local disguise.
    So the file name recorded by the run has to match, or the measurement is
    dropped and the row goes back to having no number.
    """
    measured = (catalog or {}).get("measured_here")
    if not measured:
        return None
    if resolved_file and measured.get("file") != resolved_file:
        return None
    return measured


def machine(vram_mb=None, gpu_count=None, bandwidth_gbs=None, name=None):
    """The hardware a row is drawn against.

    Passed explicitly because the answer changes per screen: choosing a model
    to run here means this card, and choosing one to run on a borrowed kernel
    means that kernel's card. Quoting a GTX 1650's throughput on a screen where
    the model will run on a T4 is worse than quoting nothing.
    """
    if vram_mb is None or gpu_count is None:
        local = hardware.summarise(hardware.detect().get("gpus"))
        vram_mb = local["vram_mb"]
        gpu_count = local["gpu_count"]
        if bandwidth_gbs is None:
            bandwidth_gbs = local["bandwidth_gbs"]
        if name is None:
            name = local["name"]
    return {"vram_mb": vram_mb or 0, "gpu_count": gpu_count or 0,
            "bandwidth_gbs": bandwidth_gbs, "name": name}


def local_measurement(model):
    """The report for this exact file, whether the row is resolved or curated."""
    return model.get("measured_here") or (
        model.get("catalog") or {}).get("measured_here")


def throughput_cell(model, machine_profile, translate=None, fits=None):
    """Tokens per second as a bare number, or a dash.

    No word inside the cell. A column of numbers each carrying "measured" or
    "estimated" is the prose the table exists to remove, so the origin is said
    once, in the legend under it.

    The dash is not a formatting choice: when `gpu_bandwidth` has no figure for
    this card there is nothing to compute from, and a plausible invented number
    reads on screen exactly like a measured one.

    `fits=False` is the same refusal for a different reason. The estimate is
    bytes per token over the bus of the card that holds the weights, so it only
    describes a model the card holds: a 9.4 GiB model on a 4 GiB card was
    reading 243 tok/s on this screen, next to a cell saying it does not fit.
    What it would really do depends on how much spills to system RAM, which
    nothing here knows.
    """
    measured = local_measurement(model)
    if measured:
        return units.tps_short(measured["tokens_per_second"])
    bandwidth = (machine_profile or {}).get("bandwidth_gbs")
    if fits is False or not bandwidth or not model.get("model_bytes"):
        return EMPTY_CELL
    estimate = hardware.estimate_tokens_per_second(
        hardware.bytes_read_per_token(
            model["model_bytes"], model.get("active_ratio", 1.0)),
        bandwidth,
    )
    return units.tps_short(estimate) if estimate else EMPTY_CELL


# What a benchmark is called, as against the key this program files it under.
# `swebench_verified` is a field name; on a row it costs seventeen columns to
# say what "SWE-bench" says. Verified is the plain name because it is the main
# set; Lite and Pro keep their qualifier, because the three do not share a
# scale and a row that dropped it would compare numbers that cannot be
# compared. A ruler with no entry here keeps its key, which is ugly and honest,
# rather than being renamed to something nobody published.
RULER_LABELS = {
    "swebench_verified": "SWE-bench",
    "swebench_lite": "SWE-bench Lite",
    "swebench_pro": "SWE-bench Pro",
    "aider_polyglot": "Aider",
    "livecodebench_v6": "LiveCodeBench",
    "gpqa_diamond": "GPQA",
}


def ranking_cell(model, translate=None):
    """The strongest public ranking this model has, short, with its owner.

    Never a number on its own. A score belongs to whoever published it and to
    the weights they ran it on, and a quantized GGUF that inherited a figure
    from the model it was made from was measured by nobody.
    """
    translate = translate or text
    measured = local_measurement(model)
    if measured:
        return translate("model.row.rank.measured",
                         humaneval=measured["humaneval"])
    scores = model.get("scores") or {}
    if not scores:
        return ""
    ruler, value = next(iter(scores.items()))
    return translate("model.row.rank.public",
                     ruler=RULER_LABELS.get(ruler, ruler), score=value)


def _row_name(model):
    """Name and precision, with the precision written exactly once.

    A weight file is normally named after its own quantization, so pasting the
    precision next to that file name produced rows reading
    "LFM2-1.2B-Tool-Q4_K_M Q4_K_M". The suffix comes off the name, and the
    precision stays as its own field where every row carries it in the same
    place and the eye can compare down the column.
    """
    name = str(model.get("name") or "")
    quantization = model.get("quantization")
    if quantization:
        trimmed = re.sub(
            r"[-_. ]*" + re.escape(quantization) + r"$", "", name, flags=re.I)
        # A file called just "Q4_K_M" has no name apart from its precision, so
        # the cell is the precision once. Falling back to the untrimmed name
        # here is what produced "Q4_K_M Q4_K_M".
        return f"{trimmed} {quantization}" if trimmed else quantization
    return name


def resolved_row_name(model):
    """A resolved GGUF named the way a person names it, not the way it is got.

    `name` on a resolved model is "org/repo-GGUF, file-stem", which is the path
    it is downloaded by. On a row that spends forty columns repeating the
    organisation, the word GGUF and the precision that already has its own
    column, and it pushes every other field off the screen.
    """
    repo = str(model.get("repo") or "")
    leaf = re.sub(r"[-_.]?gguf$", "", repo.split("/")[-1], flags=re.I)
    if not leaf:
        return _row_name(model)
    return _row_name({
        "name": leaf,
        "quantization": quantization_label(model.get("file") or ""),
    })


# The columns, in order, and the only order any screen draws them in.
MODEL_COLUMNS = ("name", "size", "fit", "tps", "rankings", "state")
EMPTY_CELL = "-"


def model_row(model, machine_profile=None, translate=None, fit=None, state=None,
              fits=None):
    """One model as its cells, never as a finished line.

    Returning fields rather than text is what lets the header be written once
    at the top instead of repeated inside every row. The old line ran to 240
    characters because each row spelled out in prose what a column heading says
    once: which card it fits in, what the number means, whether it is installed.

    No screen assembles this into text. `model_table` does, for the whole list
    at once, because a column width is a property of the list and not of a row.
    """
    translate = translate or text
    machine_profile = machine_profile or machine()
    return {
        "name": _row_name(model),
        # A model whose size nobody resolved has no size, and "0.0 GiB" is not
        # a smaller way of saying that: it is a number, and it reads like one.
        "size": translate(
            "model.row.size",
            size=units.gib_short(model["model_bytes"]),
        ) if model.get("model_bytes") else EMPTY_CELL,
        "fit": fit or EMPTY_CELL,
        # `fits` is the yes/no behind the fit cell, which is text a screen chose
        # and cannot be read back. The throughput needs the answer, not the
        # wording: an estimate is only about a model the card actually holds.
        "tps": throughput_cell(model, machine_profile, translate, fits),
        "rankings": ranking_cell(model, translate) or EMPTY_CELL,
        "state": EMPTY_CELL if state is None else state,
        # Read by model_table for the legend, never drawn as a column.
        "measured": bool(local_measurement(model)),
    }


# What a driver calls a card and what a person calls it differ by a marketing
# prefix. The prefix is the same on every card the machine has, so it
# distinguishes nothing and only widens the column it heads.
GPU_PREFIXES = re.compile(
    r"^(nvidia|amd|intel|advanced micro devices)\b[\s.,]*"
    r"(geforce|radeon|corporation|\(r\)|\(tm\))?[\s.,]*", re.I)


def machine_label(machine_profile, translate=None):
    """What to call this hardware in a column heading.

    The card's name goes at the top once, which is what stops every row from
    having to explain which machine its "fits" refers to. It is trimmed to the
    part that names the part: "NVIDIA GeForce GTX 1650" heads a column 23
    characters wide, and 15 of those say nothing a reader did not already know.
    """
    translate = translate or text
    name = (machine_profile or {}).get("name")
    if not name:
        return translate("model.table.no_gpu")
    trimmed = GPU_PREFIXES.sub("", str(name)).strip()
    return trimmed or str(name)


def model_table(rows, machine_profile=None, translate=None, state_header=None,
                columns=MODEL_COLUMNS, fit_header=None, legend=None):
    """Header and rows as aligned text, with widths taken from the whole list.

    The header is drawn here, by the same code and from the same widths as the
    rows. Drawn anywhere else it would agree with them today and drift apart at
    the first field that changes width.

    Returns {"header": str, "rows": [str], "legend": str}. The legend is where
    the origin of the throughput column lives, because the cell itself is a
    bare number: a "measured" or "estimated" word repeated down a column is the
    prose this table exists to remove.

    `fit_header` and `legend` exist for the one list that is not drawn against a
    single card: the Kaggle catalogue assigns each model to the smallest
    accelerator that holds it, so the card is a property of the row and cannot
    head the column. Each row is still built against its own accelerator, which
    is what keeps a P100's throughput off a row that will run on a T4.
    """
    translate = translate or text
    if fit_header is None or legend is None:
        machine_profile = machine_profile or machine()
    headings = {
        "name": translate("model.table.name"),
        "size": translate("model.table.size"),
        "fit": fit_header or machine_label(machine_profile, translate),
        "tps": translate("model.table.tps"),
        "rankings": translate("model.table.rankings"),
        "state": state_header or translate("model.table.state"),
    }
    # A column whose every row says the same thing is a word repeated down the
    # page, which is the prose this table exists to remove. It is dropped and
    # said once in the note above the table instead.
    uniform = {}
    drawn = []
    for key in columns:
        values = {str(row.get(key, "")) for row in rows}
        # Two rows are the least that can repeat anything. With one row every
        # column is trivially "identical on every row", and collapsing them all
        # turned the table into a legend listing each field of the only model
        # in it, which is the opposite of what this does.
        if len(rows) > 1 and len(values) == 1 and key not in ("name", "size"):
            uniform[headings[key]] = values.pop()
            continue
        drawn.append(key)
    columns = tuple(drawn)
    widths = {
        key: max([len(headings[key])] + [len(str(row.get(key, ""))) for row in rows])
        for key in columns
    }

    def draw(values):
        # The last column is not padded: trailing spaces on a selectable row
        # widen it for no reason and show up as a highlight running past the
        # text when the cursor lands on it.
        cells = [str(values.get(key, "")).ljust(widths[key])
                 for key in columns[:-1]]
        cells.append(str(values.get(columns[-1], "")))
        return "  ".join(cells).rstrip()

    if legend is None:
        measured = [row["name"] for row in rows if row.get("measured")]
        if measured:
            legend = translate("model.table.legend.measured",
                               models=", ".join(measured),
                               gpu=machine_label(machine_profile, translate))
        elif (machine_profile or {}).get("bandwidth_gbs"):
            legend = translate("model.table.legend.estimated",
                               gpu=machine_label(machine_profile, translate))
        else:
            # No published bandwidth for this card, so every cell in the column
            # is a dash. Saying why beats a column of dashes nobody explains.
            legend = translate("model.table.legend.none",
                               gpu=machine_label(machine_profile, translate))
    if uniform:
        legend = "\n".join([
            translate("model.table.uniform",
                      fields="; ".join(f"{name} {value}"
                                       for name, value in uniform.items())),
            legend,
        ])
    return {"header": draw(headings), "rows": [draw(row) for row in rows],
            "legend": legend, "uniform": uniform}


def matched_ruler(model, task):
    """The first ruler of this task the model has a score on, or None."""
    scores = model.get("scores") or {}
    for ruler in TASK_RULERS.get(task, ()):
        if ruler in scores:
            return ruler
    return None


def order_for_task(models, task):
    """Put scored models first without disturbing the remaining curation.

    Scores are only ever compared inside one ruler. Numbers from different
    benchmarks do not share a scale: 61.7 on SWE-bench Pro is a stronger result
    than 53.6 on SWE-bench Verified, so ranking one above the other by the digit
    alone would demote the better model while looking rigorous. Models sharing a
    ruler are ordered by that ruler; everything else keeps the curated order,
    which is the order a human chose and the only defensible tie-break.
    """
    if not TASK_RULERS.get(task):
        return list(models)
    groups = {}
    for position, model in enumerate(models):
        groups.setdefault(matched_ruler(model, task), []).append((position, model))
    ordered = []
    for ruler, members in sorted(
            (item for item in groups.items() if item[0]),
            key=lambda item: item[1][0][0]):
        ordered.extend(
            model for _position, model in sorted(
                members, key=lambda entry: -float(entry[1]["scores"][ruler]))
        )
    ordered.extend(model for _position, model in groups.get(None, []))
    return ordered


def local_vram():
    local = hardware.summarise(hardware.detect().get("gpus"))
    return local["vram_mb"], local["gpu_count"]


def ollama_reference(model):
    match = re.search(
        r"(?i)(?:^|[-_.])(iq\d+_[a-z0-9]+|q\d+_[a-z0-9]+)(?:[-_.]|\.gguf$)",
        model["file"],
    )
    selector = match.group(1) if match else Path(model["file"]).stem
    return f"hf.co/{model['repo']}:{selector}"
