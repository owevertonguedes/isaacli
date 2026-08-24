"""The local path that needs no third-party daemon: llama.cpp serving a GGUF.

What this adds over the Ollama path is not speed, which nobody here has
measured in llama.cpp's favour, and the screens say nothing about it. It is
control and reach: the context is decided by this program against the card it
just measured instead of frozen into a launch script nobody sees, the
quantization is on screen, and a weight sitting in a folder is a model this
program can offer instead of something it cannot see.

Weights Ollama already downloaded are reused where they lie. Asking somebody to
download the same gigabytes a second time because a different program wants to
serve them would be the worst thing this path could do.
"""
import json
import re
import urllib.request
from pathlib import Path

import config
import debug
import hardware
import llama_cpp
import local_models
import model_discovery
import terminal_ui
import units
from cli_presentation import say

# The port the local server listens on. First free one from here, because a
# second isaacli, or the user's own llama-server, may already hold it.
BASE_PORT = 8080
PORT_ATTEMPTS = 16
# Asking a local server which file it has open. Short because the screen is
# drawn after it: a server that is not answering must cost a blink, not a wait.
SERVED_PROBE_TIMEOUT = 2



def _t(tr, key, **values):
    return tr.t(key, **values)


def _select(tr, title, options, input_fn, explanation=None, initial=0,
            disabled=None):
    from setup_ollama import _select as select
    return select(tr, title, options, input_fn, explanation, initial=initial,
                  disabled=disabled)


def _short_context(tokens):
    """A context as a round number of K, because the row has no room for
    "110.791" and nobody chooses between 110 791 and 110 000.

    Below a thousand tokens the K form would read "0K", which says the model
    holds no context at all when it holds a little. Those are the rows most
    worth being exact about, so they keep their real number.
    """
    return f"{tokens // 1024}K" if tokens >= 1024 else str(tokens)


def free_port(base=BASE_PORT, attempts=PORT_ATTEMPTS, is_free=None):
    """A port nothing is listening on, so two engines never collide.

    The alternative is assuming 8080 and producing a profile that silently
    talks to whatever already answers there, which on this machine could be a
    llama-server holding a completely different model.
    """
    import socket

    def probe(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", port)) != 0

    is_free = probe if is_free is None else is_free
    for offset in range(attempts):
        if is_free(base + offset):
            return base + offset
    return None


# --- getting a server -------------------------------------------------------

def _install_plan(tr, urlopen_fn=urllib.request.urlopen):
    """What would be installed and from where, for the screen that asks.

    Nothing is downloaded before that screen is answered. What it names is the
    exact asset, its size and the backend, because "install llama.cpp" without
    those three is asking somebody to approve something they cannot see. The
    asking is `ensure_server`'s, and this used to take the `input_fn` for it
    and never call it.
    """
    try:
        tag, assets = llama_cpp.available_builds(urlopen_fn=urlopen_fn)
    except llama_cpp.InstallError as error:
        say(_t(tr, "llamacpp.install.unreachable", error=error))
        return None
    if not assets:
        say(_t(tr, "llamacpp.install.no_build"))
        return None
    order = llama_cpp.backend_order()
    asset, skipped = llama_cpp.choose_asset(assets, order)
    if asset is None:
        # Every backend this machine could use is missing from the release, and
        # the CPU build is always in `order`, so this means the release itself
        # is incomplete rather than the machine being unusual.
        say(_t(tr, "llamacpp.install.no_backend",
               backends=", ".join(skipped)))
        return None
    for backend in skipped:
        debug.note("setup_llamacpp._install_plan",
                   f"{backend} is not published for this platform in {tag}")
    return asset, tag


def ensure_server(tr, input_fn, urlopen_fn=urllib.request.urlopen,
                  candidates=()):
    """A llama-server to serve with, installing one only after being told to.

    Returns (executable, owner) or (None, None). A llama-server the user
    already has is used exactly as it is and never recorded as ours, including
    the one a saved profile already launches from outside PATH: offering to
    install a copy of the server currently answering requests is the screen
    saying it cannot see what the user is looking at.
    """
    executable, owner = llama_cpp.find_server(candidates=candidates)
    if executable:
        return executable, owner

    plan = _install_plan(tr, urlopen_fn)
    if plan is None:
        return None, None
    asset, tag = plan
    root = llama_cpp.install_root()
    index = _select(
        tr, _t(tr, "llamacpp.install.title"),
        [_t(tr, "llamacpp.install.yes"), _t(tr, "navigation.back")], input_fn,
        _t(tr, "llamacpp.install.explain", backend=asset["backend"], build=tag,
           size=f"{(asset.get('size') or 0) / 1024 ** 2:.0f}", path=root),
    )
    if index:
        return None, None
    say(_t(tr, "llamacpp.install.working", backend=asset["backend"]))
    try:
        result = llama_cpp.install(asset, urlopen_fn=urlopen_fn)
    except llama_cpp.InstallError as error:
        say(_t(tr, "llamacpp.install.failed", error=error))
        return None, None
    devices = ", ".join(item["name"] for item in result["devices"]) or "-"
    say(_t(tr, "llamacpp.install.done", path=result["executable"],
           devices=devices))
    return result["executable"], "isaacli"


# --- choosing a model -------------------------------------------------------

def _origin_key(item):
    return {
        "ollama": "llamacpp.model.origin.ollama",
        "downloaded": "llamacpp.model.origin.downloaded",
    }.get(item.get("origin"), "llamacpp.model.origin.local")


def _fit_label(item, tr, vram_mb, overhead_mb, gpu_count):
    """How much context this card holds for this model, not a yes or a no.

    A plain "fits" is decided at one fixed context, and against a 4 GB card
    that produced rows saying "does not fit" directly above a screen offering
    them seven thousand tokens. Both statements were true and together they
    were useless. What this path is choosing is the context, so the context is
    what the row reports.
    """
    from setup_ollama import MIN_CONTEXT

    if item.get("geometry_missing"):
        return _t(tr, "model.fit.unknown")
    if not gpu_count:
        return _t(tr, "llamacpp.model.fit.cpu")
    ceiling, _reason = llama_cpp.context_ceiling(
        item, vram_mb, overhead_mb=overhead_mb)
    item["context_ceiling"] = ceiling
    if not ceiling:
        return _t(tr, "model.fit.does_not_fit")
    # Short on purpose. The sentence that spells out weights, cache and total
    # belongs after the choice and in --debug; on the row it has to leave space
    # for the four other fields, and a row nobody can read is worse than a row
    # that says less.
    key = ("llamacpp.model.fit.tight" if ceiling < MIN_CONTEXT
           else "llamacpp.model.fit.context")
    return _t(tr, key, context=_short_context(ceiling))


def _row(item, tr, vram_mb, overhead_mb, gpu_count, machine=None):
    """The cells for one model, from the one function every screen uses.

    The format is not this screen's to choose, and neither is the assembly:
    this returns fields, and model_table turns the whole list into aligned text
    with one header. A screen that built its own line would drift from the
    others at the first column anybody changed.
    """
    machine = machine or model_discovery.machine(
        vram_mb=vram_mb, gpu_count=gpu_count)
    fit = _fit_label(item, tr, vram_mb, overhead_mb, gpu_count)
    return model_discovery.model_row(
        item, machine, translate=tr.t,
        fit=fit,
        # A card that holds no context of this model holds no weights of it
        # either, so it will not decode at this card's bandwidth and there is
        # nothing to estimate. Unknown geometry is unknown, not a no.
        fits=(None if not gpu_count or item.get("geometry_missing")
              else bool(item.get("context_ceiling"))),
        # The last column answers where the weights are, which on this screen
        # is the same question as "is it installed" and additionally says whose
        # copy it is: choosing an Ollama one is what creates the link.
        state=_t(tr, _origin_key(item)),
    )


def _configured_dirs(config_file=None):
    """Folders the user told this program to look in, if any."""
    try:
        data = config.load(config_file)
    except ValueError:
        return []
    dirs = data.get("model_dirs")
    return [Path(item).expanduser() for item in dirs] if isinstance(dirs, list) else []


def _served_weight(item, urlopen_fn=urllib.request.urlopen):
    """The weight file one saved profile is serving, or None.

    Three sources, most certain first, because a machine can be serving a model
    this minute and still have nothing on screen to choose: the field this
    program writes; the `-m` of a launch command it can read; and, last, the
    running server itself, which is the only one that knows when the command is
    a wrapper script. Asking is not guessing: /props answers with the path of
    the file it has open.
    """
    weights = item.get("weights")
    if weights and Path(weights).is_file():
        return Path(weights)

    command = [str(part) for part in
               ((item.get("autostart") or {}).get("cmd") or [])]
    for flag in ("-m", "--model"):
        if flag in command:
            position = command.index(flag) + 1
            if position < len(command) and Path(command[position]).is_file():
                return Path(command[position])

    base_url = item.get("base_url")
    if not base_url or not config.is_local_endpoint(base_url):
        # Somebody else's endpoint serves a file on somebody else's disk, and
        # nothing here could open it.
        return None
    url = base_url.rstrip("/").removesuffix("/v1") + "/props"
    try:
        with urlopen_fn(url, timeout=SERVED_PROBE_TIMEOUT) as response:
            path = json.load(response).get("model_path")
    except Exception as error:  # noqa: BLE001 - a server that is down is not news
        debug.note("setup_llamacpp._served_weight", f"{url}: {error}")
        return None
    return Path(path) if path and Path(path).is_file() else None


def served_weight_dirs(config_file=None, urlopen_fn=urllib.request.urlopen):
    """The folders holding the weights the saved profiles serve.

    The folder, not the file: somebody who keeps one GGUF there keeps the rest
    there too, and a screen that offered only the model already loaded would be
    a list of one on a disk full of models. This is derived every time rather
    than written into the configuration, so a weight that moves stops being
    offered instead of becoming a stale entry nobody remembers adding.
    """
    try:
        data = config.load(config_file)
    except ValueError as error:
        debug.note("setup_llamacpp.served_weight_dirs", str(error))
        return []
    folders = []
    for item in (data.get("profiles") or {}).values():
        weight = _served_weight(item, urlopen_fn)
        if weight and weight.parent not in folders:
            folders.append(weight.parent)
    return folders


def _add_directory(tr, input_fn, config_file=None):
    """Remember one more folder to look for weights in."""
    raw = input_fn(_t(tr, "llamacpp.model.folder.prompt")).strip()
    if not raw:
        return False
    folder = Path(raw).expanduser()
    if not folder.is_dir():
        say(_t(tr, "llamacpp.model.folder.missing", path=folder))
        return False
    data = config.load(config_file)
    dirs = data.setdefault("model_dirs", [])
    if str(folder) not in dirs:
        dirs.append(str(folder))
        config.save(data, config_file)
    return True


def search_dirs(config_file=None, urlopen_fn=urllib.request.urlopen):
    """Every folder the model screen looks in, beyond this program's own.

    One function because there are two callers that must not drift: the screen
    itself, and the check that asks what the screen would find. Composed at the
    call site, a check proves a list nobody draws.
    """
    return (_configured_dirs(config_file)
            + served_weight_dirs(config_file, urlopen_fn))


def remember_folder(data, model, home_dir=None):
    """Keep the folder a chosen weight was found in, as a place to look.

    The folders searched are otherwise derived from the weight each profile is
    serving, and only that: choosing a model that lives one subfolder deeper
    than the last one moves the search to that subfolder and the rest of the
    collection leaves the screen. It happened on the developer's machine, where
    a list of ten became a list of one the moment the model he picked came from
    a subfolder.

    So the folder that was searched is written down, once, at the moment a
    model is taken out of it. Only a folder of the user's: what this program
    downloads into and what Ollama holds are already searched by name, and
    writing them down would mean a configuration file naming this program's own
    directories. Deriving stays too, which is what finds a folder nobody has
    registered yet.

    Returns True when the configuration gained a folder.
    """
    root = model.get("search_root")
    if not root or model.get("origin") != "local":
        return False
    root = Path(root).expanduser()
    if root in (local_models.downloaded_dir(home_dir),
                local_models.linked_dir(home_dir)):
        return False
    dirs = data.setdefault("model_dirs", [])
    known = [Path(item).expanduser() for item in dirs]
    # An ancestor already registered covers this folder, because the scan is
    # recursive. Adding the child anyway would list the same weights twice in
    # the configuration and change nothing on screen.
    if any(folder == root or folder in root.parents for folder in known):
        return False
    dirs.append(str(root))
    return True


def _choose_from_hub(tr, input_fn, urlopen_fn=urllib.request.urlopen):
    """What Hugging Face is publishing that this card can hold, as a list.

    This screen used to be a bare prompt asking for a "model reference" and
    nothing else, which is a question only somebody who already knows the answer
    can answer: the link, the name, the id, and in which of the four accepted
    spellings. The list the Ollama path has always drawn is drawn here too, from
    the same function, led by the reviewed catalogue, and typing a reference
    stays available for whoever has one, now with an example beside the prompt.
    """
    from setup_ollama import (
        MODEL_CATALOG_PATH, _choose_quantization, curated_gguf_models)

    curated = curated_gguf_models()
    try:
        found, errors = model_discovery.discover_models(
            MODEL_CATALOG_PATH, urlopen_fn=urlopen_fn)
    except model_discovery.DiscoveryError as error:
        found, errors = [], [str(error)]
    for error in errors:
        # A candidate that failed explains a shorter list and nothing more.
        debug.note("setup_llamacpp._choose_from_hub", error)
    # The reviewed rows first, then whatever the live search added that is not
    # already among them. Both keep the origin column that says which is which,
    # so a reviewed row is never passed off as a live find or the other way.
    seen = {(item["repo"], item["file"]) for item in curated}
    merged = curated + [item for item in found
                        if (item.get("repo"), item.get("file")) not in seen]
    discovered, rows, header, legend = model_discovery.rank_against_machine(
        merged, translate=tr.t)
    entries = [*discovered, "__exact__", "__back__"]
    options = [*rows,
               model_discovery.text("model.discovery.exact"),
               model_discovery.text("model.discovery.back")]
    explanation = "\n".join(filter(None, [
        # A discovery that returned nothing at all is the answer to what was
        # just asked, so its cause goes on the screen, not behind it.
        None if discovered else "\n".join(
            model_discovery.text("model.discovery.failed", error=error)
            for error in errors) or None,
        header, legend,
    ])) or None
    index = _select(tr, _t(tr, "llamacpp.hub.title"), options, input_fn,
                    explanation)
    chosen = entries[index]
    if chosen == "__back__":
        return None
    if chosen == "__exact__":
        say(model_discovery.text("model.discovery.prompt.explain"))
        reference = input_fn(model_discovery.text("model.discovery.prompt")).strip()
        if not reference:
            return None
        try:
            return model_discovery.resolve_hf_model(
                reference, catalog_path=MODEL_CATALOG_PATH,
                urlopen_fn=urlopen_fn)
        except model_discovery.DiscoveryError as error:
            say(model_discovery.text("model.discovery.unresolved", error=error))
            return None
    # Which model and how much of it are two different questions, and the second
    # one decides both the download size and whether it fits.
    return _choose_quantization(chosen, input_fn, tr, urlopen_fn=urlopen_fn)


def _download_from_hub(tr, input_fn,
                       urlopen_fn=urllib.request.urlopen):
    """Choose a model on Hugging Face and fetch its weights here."""
    model = _choose_from_hub(tr, input_fn, urlopen_fn)
    if model is None:
        return None
    say(_t(tr, "llamacpp.model.download.start", name=model["name"],
           size=units.gib(model["model_bytes"])))

    shown = [0]

    def progress(received, declared):
        if not declared:
            return
        percent = int(received * 100 / declared)
        # One line per whole percent. Printing per block turned a download into
        # thousands of lines of scrollback.
        if percent > shown[0]:
            shown[0] = percent
            say(_t(tr, "llamacpp.model.download.progress", percent=percent),
                end="\r", flush=True)

    try:
        path = local_models.download_weight(model, progress=progress)
    except local_models.DownloadError as error:
        print()
        say(_t(tr, "llamacpp.model.download.failed", error=error))
        return None
    print()
    try:
        return local_models.describe(path)
    except Exception as error:  # noqa: BLE001 - reported, never swallowed
        say(_t(tr, "llamacpp.model.unreadable", path=path, error=error))
        return None


def choose_model(tr, input_fn, config_file=None,
                 urlopen_fn=urllib.request.urlopen):
    """The screen that answers "what can I run", against this machine.

    Every row says its size, its precision, where it came from and whether it
    fits, and the rows that fit come first. A model that does not fit still
    appears saying so, because hiding it would make the list look like
    everything on this disk runs on this card.
    """
    # The whole summary, not only the two fields the fit needs. Built from those
    # two, the table still draws and two of its columns go quietly wrong: the
    # throughput has no bandwidth to estimate from and empties, and the fit
    # column loses the card's name and is headed "CPU" on a machine with a GPU,
    # under a legend explaining why the CPU has no published bandwidth.
    local = hardware.summarise(hardware.detect().get("gpus"))
    vram_mb, gpu_count = local["vram_mb"], local["gpu_count"]
    overhead_mb = hardware.overhead_mb(gpu_count)
    while True:
        models, problems = local_models.available(
            extra_dirs=search_dirs(config_file, urlopen_fn))
        for problem in problems:
            # A weight that could not be read explains why the list is shorter.
            # That is why the list looks the way it does, not something the
            # user asked for, so it goes where the rest of the mechanism goes.
            debug.note("setup_llamacpp.choose_model", problem)
        machine = model_discovery.machine(**local)
        cells = [(item, _row(item, tr, vram_mb, overhead_mb, gpu_count, machine))
                 for item in models]
        # Most usable context first, so the machine's best option is the one
        # the cursor starts on. A model that holds nothing sorts to the bottom
        # and still appears, saying so.
        cells.sort(key=lambda entry: -(entry[0].get("context_ceiling") or 0))
        table = model_discovery.model_table(
            [row for _item, row in cells], machine, translate=tr.t,
            state_header=_t(tr, "llamacpp.model.state_header"))
        entries = [item for item, _row_cells in cells] + [
            "__folder__", "__hub__", "__back__"]
        options = table["rows"] + [
            _t(tr, "llamacpp.model.folder"),
            _t(tr, "llamacpp.model.hub"),
            _t(tr, "navigation.back"),
        ]
        # The header and the legend belong above the list, not in it: they are
        # written once, and a selectable row that is a column heading is a row
        # somebody can choose by mistake.
        explanation = "\n".join(filter(None, [
            _t(tr, "llamacpp.model.explain"),
            _t(tr, "llamacpp.model.none") if not models else "",
            "",
            table["header"] if models else "",
            table["legend"] if models else "",
        ]))
        index = _select(tr, _t(tr, "llamacpp.model.title"), options, input_fn,
                        terminal_ui.dim(explanation, input_fn))
        chosen = entries[index]
        if chosen == "__back__":
            return None
        if chosen == "__folder__":
            _add_directory(tr, input_fn, config_file)
            continue
        if chosen == "__hub__":
            chosen = _download_from_hub(tr, input_fn, urlopen_fn)
            if chosen is None:
                continue
        if chosen.get("geometry_missing"):
            say(_t(tr, "llamacpp.model.no_geometry",
                   parts=", ".join(chosen["geometry_missing"])))
        if not chosen.get("chat_template"):
            # llama-server renders the conversation from the template inside
            # the GGUF. A file without one cannot be talked to, and saying so
            # here costs a screen; not saying it costs a session of nonsense.
            say(_t(tr, "llamacpp.model.no_template", name=chosen["name"]))
            continue
        if chosen.get("needs_link"):
            try:
                link = local_models.link_ollama_model(chosen)
            except (OSError, FileExistsError) as error:
                say(_t(tr, "llamacpp.model.link_failed", error=error))
                continue
            say(_t(tr, "llamacpp.model.linked", path=link))
            chosen["path"] = str(link)
        return chosen


# --- device and context -----------------------------------------------------

def choose_device(tr, input_fn, executable):
    """Which compute device serves this model, offered rather than assumed.

    The launch script on the developer's machine names -dev Vulkan1 by hand,
    and the number is only right until a device is added. This asks the build
    itself what it can see.
    """
    try:
        devices = llama_cpp.list_devices(executable)
    except llama_cpp.InstallError as error:
        debug.note("setup_llamacpp.choose_device", str(error))
        return None
    if not devices:
        return None
    if len(devices) == 1:
        return devices[0]["id"]
    preferred = llama_cpp.preferred_device(devices)
    # Free without total made a 4 GB card read as a 1.5 GB one, under an
    # integrated GPU claiming 7.4 GiB of borrowed system RAM: the row said the
    # discrete card was the smaller device. Both numbers, and the row says which
    # of them is memory of its own.
    dedicated = llama_cpp.dedicated_devices(devices)
    options = [
        _t(tr, "llamacpp.device.option" if item["id"] in dedicated
              else "llamacpp.device.option.shared",
           id=item["id"], name=item["name"],
           free=f"{(item.get('free_mb') or item['total_mb']) / 1024:.1f}",
           total=f"{item['total_mb'] / 1024:.1f}")
        for item in devices
    ] + [_t(tr, "llamacpp.device.auto")]
    initial = next((index for index, item in enumerate(devices)
                    if preferred and item["id"] == preferred["id"]), 0)
    index = _select(tr, _t(tr, "llamacpp.device.title"), options, input_fn,
                    _t(tr, "llamacpp.device.explain"), initial=initial)
    return devices[index]["id"] if index < len(devices) else None


def choose_context(tr, input_fn, model, device_free_mb=None):
    """Offer a context this machine can actually hold, and say what capped it.

    This is the number that stopped being a stranger's decision. The script
    outside this repository froze -c 8192, which falsified two of this
    project's own measurements before anybody noticed it was there.
    """
    from setup_ollama import _choose_context

    vram_mb, gpu_count = model_discovery.local_vram()
    overhead_mb = hardware.overhead_mb(gpu_count)
    ceiling, reason = llama_cpp.context_ceiling(
        model, vram_mb, overhead_mb=overhead_mb, device_free_mb=device_free_mb)
    # Named one by one rather than built from `reason`. A key assembled by
    # interpolation is a key no scan can find, and this project's catalogue
    # check exists precisely to catch a string that no screen appears to ask
    # for; hiding from it would mean a missing translation ships unnoticed.
    shown = f"{ceiling:,}".replace(",", " ")
    if reason == "does_not_fit":
        say(_t(tr, "llamacpp.context.does_not_fit", name=model["name"]))
    elif reason == "memory":
        say(_t(tr, "llamacpp.context.capped.memory", context=shown))
    elif reason == "trained":
        say(_t(tr, "llamacpp.context.capped.trained", context=shown))
    return _choose_context(ceiling or None, input_fn, tr)


# --- saving the profile -----------------------------------------------------

def _profile_name(model):
    slug = re.sub(r"[^a-z0-9]+", "-", str(model["name"]).lower()).strip("-")
    return f"llamacpp-{slug or 'model'}"


def save_profile(data, model, executable, context, device, port, alias=None):
    """Write the profile, with the launch command this program decided.

    Saved as openai_compatible with an autostart, which is the same shape the
    developer's own hand-made profile already has. The difference is that every
    number in the command was measured here rather than typed once into a file
    outside the repository.
    """
    from setup_ollama import LLAMACPP_PROVIDER_NAME

    alias = alias or model["name"]
    command = llama_cpp.server_command(
        executable, model["path"], context, device=device, port=port,
        alias=alias)
    base_url = f"http://127.0.0.1:{port}/v1"
    name = _profile_name(model)
    data.setdefault("profiles", {})[name] = {
        "provider": "openai_compatible",
        "provider_name": LLAMACPP_PROVIDER_NAME,
        "base_url": base_url,
        "model": alias,
        "thinking": None,
        "temperature": 0,
        "num_ctx": context,
        "context_limit": model.get("context_length"),
        "weights": model["path"],
        "autostart": {"cmd": command, "health_url": base_url + "/models"},
    }
    data["default_profile"] = name
    return name


def run(language, input_fn, config_file, tr, onboarding_task=None,
        urlopen_fn=urllib.request.urlopen, release_fn=None):
    """The whole local path: a server, a model, a device, a context, a profile.

    `release_fn` stops whatever local server this session is holding. It is a
    callback rather than a call because the state it releases belongs to the
    running CLI, and setup runs with no CLI behind it.
    """
    from setup_ollama import _store_onboarding, _UNCHANGED

    executable, _owner = ensure_server(
        tr, input_fn, urlopen_fn,
        candidates=llama_cpp.profile_servers(config.load(config_file)))
    if executable is None:
        return "__engine__"
    model = choose_model(tr, input_fn, config_file, urlopen_fn)
    if model is None:
        return "__engine__"
    # Every number from here on is read off the card, and the server this
    # session started for the outgoing model is still holding it. Measured
    # around it, a 4 GB card reported 1.5 GB free, the ceiling came out at 6 730
    # tokens for a model whose real ceiling on that card is 32 768, and the
    # screen offered a context no rung could satisfy. The outgoing server is
    # released here, before anything is measured, because it is released before
    # the new one starts either way.
    if release_fn is not None:
        release_fn()
    device = choose_device(tr, input_fn, executable)
    free_mb = None
    try:
        for item in llama_cpp.list_devices(executable):
            if device and item["id"] == device:
                free_mb = item.get("free_mb")
    except llama_cpp.InstallError:
        debug.swallowed("setup_llamacpp.run free memory")
    context = choose_context(tr, input_fn, model, device_free_mb=free_mb)
    if context is None:
        return "__engine__"
    port = free_port()
    if port is None:
        say(_t(tr, "llamacpp.port.none", base=BASE_PORT))
        return 1
    data = config.load(config_file)
    data["language"] = language
    if onboarding_task is not None and onboarding_task is not _UNCHANGED:
        _store_onboarding(data, onboarding_task)
    remember_folder(data, model)
    save_profile(data, model, executable, context, device, port)
    config.save(data, config_file)
    return 0
