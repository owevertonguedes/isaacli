"""Model weights that already exist on this machine, and what they can run.

Ollama answers "what do I have" from a running daemon. llama.cpp has no daemon
to ask: a model is a GGUF file somewhere on disk, and somebody has to find it,
size it, and read its geometry before the machine can say whether it fits. That
is this module.

It also answers the more valuable question, which is what the user already
downloaded through Ollama. Those weights are GGUF files too, sitting in Ollama's
blob store under their digest. Re-downloading gigabytes the machine already
holds would be the worst thing this change could do, so they are reused where
they lie, by link, and never copied and never written to.

Two directories, never one:

    <data>/models/downloaded/   what isaacli fetched, and may therefore remove
    <data>/models/from-ollama/  links into Ollama's store, which belong to Ollama

The separation is the whole point. isaacli must be able to remove what it
installed without ever reaching into somebody else's store, and a user who
uninstalls isaacli must keep every model Ollama downloaded for them.
"""
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import debug
import gguf
import model_discovery

# Read size for the download loop. Not a limit on anything: how much is allowed
# through is the size the repository declared, checked as it arrives.
_DOWNLOAD_BLOCK_BYTES = 1 << 20
DOWNLOAD_TIMEOUT = 60 * 60

# Ollama names the layer that holds the weights; the rest of the manifest is
# the template, the parameters and the licence. Matching the media type rather
# than "the biggest layer" is what keeps a licence blob from being served as a
# model.
OLLAMA_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"

# A digest is the file name in Ollama's blob store, and it is also a path
# fragment this program builds. Constraining it to the shape of a digest is what
# stops a crafted manifest from naming ../../ anything.
DIGEST = re.compile(r"^[a-z0-9]+[:-][a-f0-9]{64}$")

GGUF_SUFFIX = ".gguf"


def data_dir(home_dir=None):
    """Where this program keeps what it installed for itself."""
    base = os.environ.get("XDG_DATA_HOME")
    if base:
        root = Path(base).expanduser()
    else:
        root = (Path(home_dir) if home_dir else Path.home()) / ".local" / "share"
    return root / "isaacli"


def downloaded_dir(home_dir=None):
    return data_dir(home_dir) / "models" / "downloaded"


def linked_dir(home_dir=None):
    return data_dir(home_dir) / "models" / "from-ollama"


def ollama_root(environ=None, home_dir=None):
    """Ollama's own store, wherever the user pointed it.

    OLLAMA_MODELS is how Ollama itself is redirected, and installation.py
    already reads it for the opposite reason: to refuse to delete a store it
    does not recognise. Reading it here means a user who moved their models to
    another disk still gets them offered.
    """
    environ = os.environ if environ is None else environ
    value = environ.get("OLLAMA_MODELS")
    if value:
        return Path(value).expanduser()
    home = Path(home_dir) if home_dir else Path.home()
    return home / ".ollama" / "models"


def _quantization(path):
    """The precision this file carries, from its name, or None when unnamed."""
    label = model_discovery.quantization_label(path.name)
    return label if label != Path(path.name).stem else None


def describe(path, name=None, source=None, reference=None):
    """Describe one GGUF file the way the fit calculation and the screens want.

    Shaped like model_discovery.resolve_hf_model on purpose: the same fit
    report, the same origin labelling and the same screen render a file on disk
    and a repository on Hugging Face, so neither grows a second code path.

    Raises gguf.GGUFError when the header cannot be read, because a file this
    program cannot describe is not a model it can offer.
    """
    path = Path(path)
    stat = path.stat()
    shape = gguf.geometry(path)
    missing = [key for key in ("n_layers", "n_kv_heads", "head_dim")
               if not shape.get(key)]
    item = {
        "name": name or shape["name"],
        "path": str(path),
        "file": path.name,
        "model_bytes": stat.st_size,
        "architecture": shape["architecture"],
        "context_length": shape["context_length"],
        "chat_template": shape["chat_template"],
        "quantization": _quantization(path),
        # Nothing on disk carries a published benchmark. Saying so is the rule:
        # a number that has no source to check does not get shown, and a local
        # file has no source at all.
        "benchmark": "",
        "benchmark_source": None,
        "scores": {},
        "origin": "local",
        "source": source or str(path.parent),
        "reference": reference,
        # The expert ratio decides bytes read per token, and reading it exactly
        # would mean summing the expert tensors, which means a size table for
        # every ggml quantization. Claiming a ratio nobody computed would be an
        # unmeasured performance claim, so this stays at the dense value and
        # the screens show fit, which is arithmetic on the file size.
        "active_ratio": 1.0,
    }
    if missing:
        item["geometry_missing"] = missing
    else:
        item.update({
            "n_layers": shape["n_layers"],
            "n_kv_heads": shape["n_kv_heads"],
            "head_dim": shape["head_dim"],
        })
    return item


_CACHE = {}


def describe_cached(path, **values):
    """describe(), memoised against the file's own identity.

    A screen redraws, and re-reading every header on every redraw would make
    the list cost a second each time. The key is the file's size and
    modification time, so a weight that changed underneath is read again rather
    than answered from a stale entry.
    """
    path = Path(path)
    try:
        stat = path.stat()
    except OSError as error:
        raise gguf.GGUFError(str(error)) from error
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _CACHE:
        _CACHE[key] = describe(path, **values)
    return dict(_CACHE[key])


def scan(directories, recursive=True):
    """Find every GGUF under these directories, described or explained.

    Returns (models, problems). A file whose header cannot be read is a
    problem, not a silent omission: a shorter list with no reason is how a
    missing model becomes invisible.
    """
    models = []
    problems = []
    seen = set()
    for directory in directories:
        directory = Path(directory).expanduser()
        if not directory.is_dir():
            continue
        pattern = "**/*" if recursive else "*"
        try:
            candidates = sorted(directory.glob(pattern + GGUF_SUFFIX))
        except OSError as error:
            problems.append(f"{directory}: {error}")
            continue
        for path in candidates:
            # A weight split across shards is one model, and llama.cpp is given
            # the first shard and finds the rest. Listing every shard would show
            # one model five times, each row claiming a fifth of its real size.
            if model_discovery.SPLIT_GGUF.search(path.name):
                if "-00001-of-" not in path.name:
                    continue
            try:
                resolved = path.resolve()
            except OSError as error:
                problems.append(f"{path}: {error}")
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                models.append(describe_cached(path))
            except gguf.GGUFError as error:
                problems.append(f"{path.name}: {error}")
    return models, problems


def ollama_manifests(root=None, environ=None, home_dir=None):
    """Every model Ollama has downloaded, as (name, manifest path) pairs.

    The store is read, never written. What lives there is the user's, obtained
    through another program, and this one has no business changing it.
    """
    root = Path(root) if root else ollama_root(environ, home_dir)
    manifests = root / "manifests"
    if not manifests.is_dir():
        return []
    found = []
    try:
        candidates = sorted(path for path in manifests.rglob("*") if path.is_file())
    except OSError as error:
        debug.note("local_models.ollama_manifests", str(error))
        return []
    for path in candidates:
        parts = path.relative_to(manifests).parts
        # registry/namespace/model/tag is the layout Ollama writes. Anything
        # shallower is not a manifest and is skipped rather than guessed at.
        if len(parts) < 3:
            continue
        model = "/".join(parts[2:-1]) if len(parts) > 3 else parts[-2]
        namespace = parts[1]
        # "library" is Ollama's own namespace and it never appears in the name
        # the user typed, so repeating it here would show every model under a
        # name that does not match `ollama list`.
        name = model if namespace == "library" else f"{namespace}/{model}"
        found.append((f"{name}:{parts[-1]}", path))
    return found


def _blob_path(root, digest):
    if not isinstance(digest, str) or not DIGEST.match(digest):
        return None
    return root / "blobs" / digest.replace(":", "-")


def ollama_models(root=None, environ=None, home_dir=None):
    """Describe the weights Ollama already holds, without copying any of them.

    Returns (models, problems). Each model carries the blob path it lives at,
    so linking it is a later, explicit step rather than something this does
    behind the user's back.
    """
    root = Path(root) if root else ollama_root(environ, home_dir)
    models = []
    problems = []
    for name, manifest_path in ollama_manifests(root):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            layers = manifest["layers"]
            if not isinstance(layers, list):
                raise TypeError("layers is not a list")
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            problems.append(f"{name}: {error}")
            continue
        weights = next(
            (layer for layer in layers if isinstance(layer, dict)
             and layer.get("mediaType") == OLLAMA_MODEL_MEDIA_TYPE), None)
        if weights is None:
            problems.append(f"{name}: the manifest declares no model layer")
            continue
        blob = _blob_path(root, weights.get("digest"))
        if blob is None:
            problems.append(f"{name}: the model layer declares no usable digest")
            continue
        if not blob.exists():
            # Ollama can hold a manifest whose blob was garbage collected. That
            # is "not downloaded", not a broken installation.
            problems.append(f"{name}: the weights are not in the blob store")
            continue
        try:
            item = describe_cached(blob, name=name, source=str(root),
                                   reference=name)
        except gguf.GGUFError as error:
            problems.append(f"{name}: {error}")
            continue
        item["origin"] = "ollama"
        item["ollama_name"] = name
        # Ollama's template layer is not read, and that is the decision rather
        # than an omission: it is Go text/template, and llama.cpp's
        # --chat-template-file expects Jinja, so a server handed one would
        # start, answer, and format every conversation wrong. What decides
        # whether a weight can be talked to is the chat template inside the
        # GGUF, which `describe_cached` already reports and setup_llamacpp
        # refuses on. The path to that layer used to be recorded here as
        # "for --debug", and no --debug note or any other line ever read it.
        models.append(item)
    return models, problems


def _link_name(name):
    """A file name for a model reference, safe to build a path out of."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-._")
    return (slug or "model") + GGUF_SUFFIX


def link_ollama_model(item, home_dir=None, target_dir=None):
    """Point at weights Ollama already downloaded, without copying them.

    A link, because the alternative is asking somebody to download the same
    gigabytes twice onto the same disk. The link lives in this program's own
    directory, so removing isaacli removes the link and leaves Ollama's store
    exactly as it was.

    Returns the link path. Re-linking an unchanged model is not an error; a
    link that already points somewhere else is, because silently repointing it
    would change which weights a saved profile serves.
    """
    blob = Path(item["path"]).resolve()
    if not blob.exists():
        raise FileNotFoundError(blob)
    folder = Path(target_dir) if target_dir else linked_dir(home_dir)
    folder.mkdir(parents=True, exist_ok=True)
    link = folder / _link_name(item.get("ollama_name") or item["name"])
    if link.is_symlink():
        if link.resolve(strict=False) == blob:
            return link
        raise FileExistsError(link)
    if link.exists():
        raise FileExistsError(link)
    link.symlink_to(blob)
    return link


def available(home_dir=None, extra_dirs=(), environ=None, ollama_store=None):
    """Every weight this machine can serve right now, from every source.

    Returns (models, problems), the models carrying an "origin" of:

        downloaded  this program fetched it, so this program may remove it
        local       a folder the user pointed at, which is theirs
        ollama      Ollama downloaded it, and it is reused where it lies

    Deduplicated by the file the paths actually resolve to, so a model reached
    through both a link and its folder appears once instead of twice with two
    different names.
    """
    models = []
    problems = []
    seen = set()

    def add(items, origin=None):
        for item in items:
            try:
                key = Path(item["path"]).resolve()
            except OSError:
                key = Path(item["path"])
            if key in seen:
                continue
            seen.add(key)
            if origin:
                item["origin"] = origin
            models.append(item)

    linked, trouble = linked_models(home_dir)
    add(linked)
    problems.extend(trouble)

    found, trouble = scan([downloaded_dir(home_dir)])
    add(found, "downloaded")
    problems.extend(trouble)

    if extra_dirs:
        found, trouble = scan(extra_dirs)
        add(found, "local")
        problems.extend(trouble)

    # Offered last because these are not linked yet: choosing one is what makes
    # the link, and that is a decision the user takes rather than something
    # this does to their store while building a list.
    found, trouble = ollama_models(root=ollama_store, environ=environ,
                                   home_dir=home_dir)
    for item in found:
        item["needs_link"] = True
    add(found)
    problems.extend(trouble)
    return models, problems


class DownloadError(RuntimeError):
    """A weight could not be fetched, with the reason attached."""


def download_weight(model, home_dir=None, target_dir=None, progress=None,
                    urlopen_fn=urllib.request.urlopen, timeout=DOWNLOAD_TIMEOUT):
    """Fetch one GGUF into this program's own model directory.

    Downloads land in their own directory, never beside anything Ollama owns,
    so what this program may later remove is exactly what this program fetched.

    A partial file is written under a temporary name and only becomes the real
    one once the declared number of bytes has arrived. Without that, an
    interrupted download leaves a truncated GGUF that looks like a model, gets
    listed, gets chosen, and fails at the moment somebody asks a question.
    """
    url = model.get("file_url")
    if not url:
        raise DownloadError("this model carries no download URL")
    declared = model.get("model_bytes") or 0
    folder = Path(target_dir) if target_dir else downloaded_dir(home_dir)
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / Path(model.get("file") or "model.gguf").name
    if final.exists():
        return final
    partial = final.with_name(final.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "isaacli"})
    received = 0
    try:
        with urlopen_fn(request, timeout=timeout) as response, \
                partial.open("wb") as handle:
            while True:
                block = response.read(_DOWNLOAD_BLOCK_BYTES)
                if not block:
                    break
                received += len(block)
                if declared and received > declared:
                    raise DownloadError(
                        f"the download passed the {declared} bytes the repository "
                        "declared, so it was stopped")
                handle.write(block)
                if progress:
                    progress(received, declared)
        if declared and received != declared:
            raise DownloadError(
                f"the download ended at {received} bytes, not the {declared} "
                "the repository declared")
        os.replace(partial, final)
    except urllib.error.HTTPError as error:
        raise DownloadError(f"HTTP {error.code} downloading {url}") from error
    except urllib.error.URLError as error:
        raise DownloadError(str(error.reason)) from error
    except OSError as error:
        raise DownloadError(str(error)) from error
    finally:
        try:
            if partial.exists():
                partial.unlink()
        except OSError:
            debug.swallowed("local_models.download_weight cleanup")
    return final


def linked_models(home_dir=None, target_dir=None):
    """The links this program made, and which of them still resolve.

    Returns (models, problems). Ollama removing a model leaves a link pointing
    nowhere, and that has to be said out loud: a profile saved against it would
    otherwise fail at the moment the user asks a question.
    """
    folder = Path(target_dir) if target_dir else linked_dir(home_dir)
    if not folder.is_dir():
        return [], []
    models = []
    problems = []
    for link in sorted(folder.glob("*" + GGUF_SUFFIX)):
        if not link.is_symlink():
            problems.append(f"{link.name}: not a link this program made")
            continue
        if not link.exists():
            problems.append(
                f"{link.name}: the weights it points at are gone from Ollama's store")
            continue
        try:
            item = describe_cached(link, name=link.stem)
        except gguf.GGUFError as error:
            problems.append(f"{link.name}: {error}")
            continue
        item["origin"] = "ollama"
        models.append(item)
    return models, problems
