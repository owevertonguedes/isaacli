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

# The port the local server listens on. First free one from here, because a
# second isaacli, or the user's own llama-server, may already hold it.
BASE_PORT = 8080
PORT_ATTEMPTS = 16



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


def _gib(value):
    return f"{(value or 0) / 1024 ** 3:.2f}"


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

def _install_plan(tr, input_fn, urlopen_fn=urllib.request.urlopen):
    """Show what would be installed, from where, and ask.

    Nothing is downloaded before this screen is answered. What it names is the
    exact asset, its size and the backend, because "install llama.cpp" without
    those three is asking somebody to approve something they cannot see.
    """
    try:
        tag, assets = llama_cpp.available_builds(urlopen_fn=urlopen_fn)
    except llama_cpp.InstallError as error:
        print(_t(tr, "llamacpp.install.unreachable", error=error))
        return None
    if not assets:
        print(_t(tr, "llamacpp.install.no_build"))
        return None
    order = llama_cpp.backend_order()
    asset, skipped = llama_cpp.choose_asset(assets, order)
    if asset is None:
        # Every backend this machine could use is missing from the release, and
        # the CPU build is always in `order`, so this means the release itself
        # is incomplete rather than the machine being unusual.
        print(_t(tr, "llamacpp.install.no_backend",
                 backends=", ".join(skipped)))
        return None
    for backend in skipped:
        debug.note("setup_llamacpp._install_plan",
                   f"{backend} is not published for this platform in {tag}")
    return asset, tag


def ensure_server(tr, input_fn, config_file=None,
                  urlopen_fn=urllib.request.urlopen):
    """A llama-server to serve with, installing one only after being told to.

    Returns (executable, owner) or (None, None). A llama-server the user
    already has is used exactly as it is and never recorded as ours.
    """
    executable, owner = llama_cpp.find_server()
    if executable:
        return executable, owner

    plan = _install_plan(tr, input_fn, urlopen_fn)
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
    print(_t(tr, "llamacpp.install.working", backend=asset["backend"]))
    try:
        result = llama_cpp.install(asset, urlopen_fn=urlopen_fn)
    except llama_cpp.InstallError as error:
        print(_t(tr, "llamacpp.install.failed", error=error))
        return None, None
    devices = ", ".join(item["name"] for item in result["devices"]) or "-"
    print(_t(tr, "llamacpp.install.done", path=result["executable"],
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


def _add_directory(tr, input_fn, config_file=None):
    """Remember one more folder to look for weights in."""
    raw = input_fn(_t(tr, "llamacpp.model.folder.prompt")).strip()
    if not raw:
        return False
    folder = Path(raw).expanduser()
    if not folder.is_dir():
        print(_t(tr, "llamacpp.model.folder.missing", path=folder))
        return False
    data = config.load(config_file)
    dirs = data.setdefault("model_dirs", [])
    if str(folder) not in dirs:
        dirs.append(str(folder))
        config.save(data, config_file)
    return True


def _download_from_hub(tr, input_fn, config_file=None,
                       urlopen_fn=urllib.request.urlopen):
    """Resolve an exact Hugging Face reference and fetch its weights here."""
    from setup_ollama import MODEL_CATALOG_PATH

    reference = input_fn(model_discovery.text("model.discovery.prompt")).strip()
    if not reference:
        return None
    try:
        model = model_discovery.resolve_hf_model(
            reference, catalog_path=MODEL_CATALOG_PATH, urlopen_fn=urlopen_fn)
    except model_discovery.DiscoveryError as error:
        print(model_discovery.text("model.discovery.unresolved", error=error))
        return None
    print(_t(tr, "llamacpp.model.download.start", name=model["name"],
             size=_gib(model["model_bytes"])))

    shown = [0]

    def progress(received, declared):
        if not declared:
            return
        percent = int(received * 100 / declared)
        # One line per whole percent. Printing per block turned a download into
        # thousands of lines of scrollback.
        if percent > shown[0]:
            shown[0] = percent
            print(_t(tr, "llamacpp.model.download.progress", percent=percent),
                  end="\r", flush=True)

    try:
        path = local_models.download_weight(model, progress=progress)
    except local_models.DownloadError as error:
        print()
        print(_t(tr, "llamacpp.model.download.failed", error=error))
        return None
    print()
    try:
        return local_models.describe(path)
    except Exception as error:  # noqa: BLE001 - reported, never swallowed
        print(_t(tr, "llamacpp.model.unreadable", path=path, error=error))
        return None


def choose_model(tr, input_fn, config_file=None,
                 urlopen_fn=urllib.request.urlopen):
    """The screen that answers "what can I run", against this machine.

    Every row says its size, its precision, where it came from and whether it
    fits, and the rows that fit come first. A model that does not fit still
    appears saying so, because hiding it would make the list look like
    everything on this disk runs on this card.
    """
    vram_mb, gpu_count = model_discovery.local_vram()
    overhead_mb = hardware.DEFAULT_OVERHEAD_MB * max(1, gpu_count)
    while True:
        models, problems = local_models.available(
            extra_dirs=_configured_dirs(config_file))
        for problem in problems:
            # A weight that could not be read explains why the list is shorter.
            # That is why the list looks the way it does, not something the
            # user asked for, so it goes where the rest of the mechanism goes.
            debug.note("setup_llamacpp.choose_model", problem)
        machine = model_discovery.machine(vram_mb=vram_mb, gpu_count=gpu_count)
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
            chosen = _download_from_hub(tr, input_fn, config_file, urlopen_fn)
            if chosen is None:
                continue
        if chosen.get("geometry_missing"):
            print(_t(tr, "llamacpp.model.no_geometry",
                     parts=", ".join(chosen["geometry_missing"])))
        if not chosen.get("chat_template"):
            # llama-server renders the conversation from the template inside
            # the GGUF. A file without one cannot be talked to, and saying so
            # here costs a screen; not saying it costs a session of nonsense.
            print(_t(tr, "llamacpp.model.no_template", name=chosen["name"]))
            continue
        if chosen.get("needs_link"):
            try:
                link = local_models.link_ollama_model(chosen)
            except (OSError, FileExistsError) as error:
                print(_t(tr, "llamacpp.model.link_failed", error=error))
                continue
            print(_t(tr, "llamacpp.model.linked", path=link))
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
    options = [
        _t(tr, "llamacpp.device.option", id=item["id"], name=item["name"],
           free=f"{(item.get('free_mb') or item['total_mb']) / 1024:.1f}")
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
    overhead_mb = hardware.DEFAULT_OVERHEAD_MB * max(1, gpu_count)
    ceiling, reason = llama_cpp.context_ceiling(
        model, vram_mb, overhead_mb=overhead_mb, device_free_mb=device_free_mb)
    # Named one by one rather than built from `reason`. A key assembled by
    # interpolation is a key no scan can find, and this project's catalogue
    # check exists precisely to catch a string that no screen appears to ask
    # for; hiding from it would mean a missing translation ships unnoticed.
    shown = f"{ceiling:,}".replace(",", " ")
    if reason == "does_not_fit":
        print(_t(tr, "llamacpp.context.does_not_fit", name=model["name"]))
    elif reason == "memory":
        print(_t(tr, "llamacpp.context.capped.memory", context=shown))
    elif reason == "trained":
        print(_t(tr, "llamacpp.context.capped.trained", context=shown))
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
        urlopen_fn=urllib.request.urlopen):
    """The whole local path: a server, a model, a device, a context, a profile."""
    from setup_ollama import _store_onboarding, _UNCHANGED

    executable, _owner = ensure_server(tr, input_fn, config_file, urlopen_fn)
    if executable is None:
        return "__engine__"
    model = choose_model(tr, input_fn, config_file, urlopen_fn)
    if model is None:
        return "__engine__"
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
        print(_t(tr, "llamacpp.port.none", base=BASE_PORT))
        return 1
    data = config.load(config_file)
    data["language"] = language
    if onboarding_task is not None and onboarding_task is not _UNCHANGED:
        _store_onboarding(data, onboarding_task)
    save_profile(data, model, executable, context, device, port)
    config.save(data, config_file)
    return 0
