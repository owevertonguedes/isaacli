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
from cli_i18n import t


HF_ROOT = "https://huggingface.co"
HF_API = HF_ROOT + "/api/models"
DEFAULT_TIMEOUT = 8
DEFAULT_CONTEXT = 16384
NO_PUBLIC_SCORE = "no public score on the accepted coding benchmarks"
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


def _gguf_files(payload):
    names = []
    for sibling in payload.get("siblings") or []:
        name = sibling.get("rfilename") if isinstance(sibling, dict) else None
        if isinstance(name, str) and name.lower().endswith(".gguf"):
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
            "benchmark": item.get("benchmark") or NO_PUBLIC_SCORE,
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
    benchmark = evidence.get("benchmark") or NO_PUBLIC_SCORE
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
    """
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
    gib = 1024 ** 3
    available = max(0, report["vram_mb"] - report["overhead_mb"]) * 1024 ** 2
    return translate(
        state_key,
        fits=translate(fit_yes_key) if report["fits"] else translate(fit_no_key),
        weights=f"{report['model_bytes'] / gib:.2f}",
        kv=f"{report['kv_bytes'] / gib:.2f}",
        total=f"{(report['model_bytes'] + report['kv_bytes']) / gib:.2f}",
        available=f"{available / gib:.2f}",
    )


def benchmark_line(model):
    return text("model.discovery.score", score=model.get("benchmark") or NO_PUBLIC_SCORE)


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
    gpus = hardware.detect().get("gpus") or []
    return sum(item.get("vram_mb", 0) for item in gpus), len(gpus)


def ollama_reference(model):
    match = re.search(
        r"(?i)(?:^|[-_.])(iq\d+_[a-z0-9]+|q\d+_[a-z0-9]+)(?:[-_.]|\.gguf$)",
        model["file"],
    )
    selector = match.group(1) if match else Path(model["file"]).stem
    return f"hf.co/{model['repo']}:{selector}"
