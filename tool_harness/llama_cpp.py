"""Installing, recognising and removing a llama.cpp this program put here.

The rule this module exists to obey: what isaacli installs, isaacli uninstalls.
So the install writes down, at the moment it happens, that it was us; the
removal refuses anything that record does not cover, anything the package
manager owns, and anything that would need sudo, because needing sudo proves we
did not put it there. installation.py and cli_kaggle.install_kaggle_cli are the
pattern, and this follows them rather than inventing a third shape.

Nothing here compiles anything. Upstream publishes built binaries per platform
and per backend, and the asset names carry both, so the choice of backend is a
choice among files that already exist rather than a build the user waits on.
"""
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from fractions import Fraction
import zipfile
from pathlib import Path

import config
import debug
import hardware
import local_models

RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"

# Every upstream build is published as a prerelease, so "the latest release" is
# not a question the API answers for this repository: asking for /latest returns
# a tag with no binaries in it. The newest tag that actually carries binaries is
# the answer, and that is what this reads. Checked live 2026-08-23.
RELEASE_PAGE_SIZE = 8

DEFAULT_TIMEOUT = 20
# A binary release is tens of megabytes and the connection is somebody's home
# link, so the download gets its own budget rather than the metadata one.
DOWNLOAD_TIMEOUT = 600
# Read size for the download loop. Not a limit on anything: how much is allowed
# through is the size the release declared, checked below.
_DOWNLOAD_BLOCK_BYTES = 1 << 16

SERVER_NAME = "llama-server"

# --cache-type-k / --cache-type-v values llama.cpp accepts, mapped to the bytes
# per cache element that type spends. f16 is what the server uses when neither
# flag is given, so it is the reference the ceiling is measured against by
# default. A q8_0 block stores 32 quantized values plus one f16 scale, so it
# spends 34 bytes per 32 elements. That makes the KV cache smaller and lets a
# fixed memory budget hold more context. That arithmetic is what
# `context_ceiling` runs, not a claim about output quality, which this project
# has not measured for either type.
CACHE_TYPES = {"f16": 2, "q8_0": Fraction(34, 32)}
DEFAULT_CACHE_TYPE = "f16"

# llama-b10595-bin-ubuntu-vulkan-x64.tar.gz, and the same shape for every other
# platform. The backend section is optional: a plain "ubuntu-x64" is the CPU
# build. Parsing the name instead of listing the names means a backend upstream
# adds tomorrow is offered without changing this file.
ASSET = re.compile(
    r"^llama-(?P<build>b\d+)-bin-(?P<system>[a-z0-9]+)"
    r"(?:-(?P<backend>[a-z0-9][a-z0-9.-]*?))?"
    r"-(?P<arch>x64|arm64|s390x)\.(?P<extension>tar\.gz|zip)$"
)

# PCI vendor identifiers, as /sys reports them. This is the only place that
# knows which company made the card, and it is read rather than guessed.
PCI_VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}

# Which upstream backend to offer for which vendor, best first. These are not
# performance rankings, which this program does not have measurements for; they
# are which build can address that vendor's hardware at all. Vulkan is the
# vendor-neutral one and therefore the last resort that still uses the GPU.
#
# NVIDIA is the case worth naming: upstream publishes CUDA binaries for Windows
# only, so on Linux the published build that reaches an NVIDIA card is Vulkan.
# Confirmed against the live asset list, 2026-08-23.
VENDOR_BACKENDS = {
    "nvidia": ["cuda", "vulkan"],
    "amd": ["rocm", "vulkan"],
    "intel": ["sycl", "vulkan"],
}
CPU_BACKEND = "cpu"


class InstallError(RuntimeError):
    """Something in the install could not be done, with the reason attached."""


def install_record(path=None):
    return Path(path) if path else config.config_path().with_name(
        "llamacpp-install.json")


def install_root(home_dir=None):
    return local_models.data_dir(home_dir) / "llama.cpp"


def _target(system=None, machine=None):
    """This platform, in the words upstream uses in its asset names."""
    system = (system or sys.platform).lower()
    machine = (machine or platform.machine()).lower()
    if machine in ("x86_64", "amd64", "x64"):
        arch = "x64"
    elif machine in ("aarch64", "arm64"):
        arch = "arm64"
    elif machine == "s390x":
        arch = "s390x"
    else:
        arch = machine
    if system.startswith("linux"):
        return "ubuntu", arch
    if system == "darwin":
        return "macos", arch
    if system.startswith("win"):
        return "win", arch
    return system, arch


def gpu_vendors(drm_root="/sys/class/drm"):
    """Which GPU vendors this machine has, read from the kernel's own view.

    hardware.gpus() asks nvidia-smi, which by construction can only ever answer
    "NVIDIA". Choosing a backend needs to know about the other two vendors as
    well, and /sys states the PCI vendor without installing anything.
    """
    vendors = []
    try:
        cards = sorted(Path(drm_root).glob("card[0-9]*"))
    except OSError:
        debug.swallowed("llama_cpp.gpu_vendors")
        return vendors
    for card in cards:
        try:
            raw = (card / "device" / "vendor").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        vendor = PCI_VENDORS.get(raw.lower())
        if vendor and vendor not in vendors:
            vendors.append(vendor)
    return vendors


def backend_order(vendors=None, drm_root="/sys/class/drm", system=None):
    """Backends worth offering on this machine, best first.

    macOS is the exception with no choice in it: upstream ships one build per
    architecture and Metal is inside it, so offering a backend menu there would
    be inventing a decision the platform does not have.
    """
    system, _arch = _target(system)
    if system == "macos":
        return []
    vendors = gpu_vendors(drm_root) if vendors is None else list(vendors)
    order = []
    for vendor in vendors:
        for backend in VENDOR_BACKENDS.get(vendor, []):
            if backend not in order:
                order.append(backend)
    order.append(CPU_BACKEND)
    return order


def parse_asset(name):
    """Split an upstream asset name into what it is built for, or None."""
    match = ASSET.match(str(name))
    if not match:
        return None
    parts = match.groupdict()
    parts["backend"] = parts["backend"] or CPU_BACKEND
    return parts


def _json_request(url, timeout=DEFAULT_TIMEOUT, urlopen_fn=urllib.request.urlopen):
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "isaacli",
    })
    try:
        with urlopen_fn(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise InstallError(f"HTTP {error.code} from {url}") from error
    except urllib.error.URLError as error:
        raise InstallError(str(error.reason)) from error
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
        raise InstallError(str(error)) from error


def available_builds(urlopen_fn=urllib.request.urlopen, timeout=DEFAULT_TIMEOUT,
                     system=None, machine=None):
    """The newest published build that has binaries for this platform.

    Returns (build_tag, {backend: asset}). A platform upstream does not build
    for comes back empty, which is a real answer and gets said on screen rather
    than turning into a download of the wrong architecture.
    """
    payload = _json_request(
        f"{RELEASES_API}?per_page={RELEASE_PAGE_SIZE}", timeout=timeout,
        urlopen_fn=urlopen_fn,
    )
    if not isinstance(payload, list):
        raise InstallError("the release listing was not a list")
    want_system, want_arch = _target(system, machine)
    for release in payload:
        if not isinstance(release, dict):
            continue
        assets = {}
        for asset in release.get("assets") or []:
            if not isinstance(asset, dict):
                continue
            parts = parse_asset(asset.get("name"))
            if not parts:
                continue
            if parts["system"] != want_system or parts["arch"] != want_arch:
                continue
            assets.setdefault(parts["backend"], {
                "name": asset.get("name"),
                "url": asset.get("browser_download_url"),
                "size": asset.get("size"),
                "digest": asset.get("digest"),
                "backend": parts["backend"],
                "build": parts["build"],
                "extension": parts["extension"],
            })
        if assets:
            return release.get("tag_name"), assets
    return None, {}


def choose_asset(assets, order):
    """The first asset in this machine's backend order that upstream publishes.

    A backend nothing was built for is skipped with its reason recorded, not
    silently swapped: the caller shows what was chosen and why.
    """
    reasons = []
    for backend in order:
        if backend in assets:
            return assets[backend], reasons
        reasons.append(backend)
    return None, reasons


def _verify_digest(path, declared):
    """Match the bytes on disk against the digest the API declared for them."""
    if not declared:
        raise InstallError("the release published no digest for this file")
    algorithm, _, expected = str(declared).partition(":")
    try:
        digest = hashlib.new(algorithm)
    except ValueError as error:
        raise InstallError(f"unknown digest algorithm {algorithm}") from error
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise InstallError(
            "the downloaded file does not match the digest the release declared")


def download(asset, destination, urlopen_fn=urllib.request.urlopen,
             timeout=DOWNLOAD_TIMEOUT):
    """Fetch one asset and prove it is the file the release described."""
    url = asset.get("url")
    if not url:
        raise InstallError("the release entry carries no download URL")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The release listing already said how big this file is, so the size is a
    # limit on the way in rather than a discovery on the way out. Without it,
    # the amount written to somebody's disk is decided by whoever is answering
    # the connection. That is not theoretical: a stub that kept answering the
    # same bytes filled this machine's /tmp, which is RAM here, until nothing
    # on it could run.
    declared = asset.get("size")
    request = urllib.request.Request(url, headers={"User-Agent": "isaacli"})
    try:
        with urlopen_fn(request, timeout=timeout) as response, \
                destination.open("wb") as handle:
            written = 0
            while True:
                block = response.read(_DOWNLOAD_BLOCK_BYTES)
                if not block:
                    break
                written += len(block)
                if declared and written > declared:
                    raise InstallError(
                        f"the download exceeded the {declared} bytes the release "
                        "declared, so it was stopped")
                handle.write(block)
            if declared and written != declared:
                raise InstallError(
                    f"the download ended at {written} bytes, not the {declared} "
                    "the release declared")
    except urllib.error.HTTPError as error:
        raise InstallError(f"HTTP {error.code} downloading {url}") from error
    except urllib.error.URLError as error:
        raise InstallError(str(error.reason)) from error
    except OSError as error:
        raise InstallError(str(error)) from error
    _verify_digest(destination, asset.get("digest"))
    return destination


def _safe_members(names, root):
    """Refuse any archive entry that would write outside the target directory.

    An archive is a list of paths chosen by whoever built it, so an entry named
    ../../.bashrc is a write to the home directory dressed as an extraction.
    """
    for name in names:
        target = (root / name).resolve()
        if target != root and root not in target.parents:
            raise InstallError(f"the archive names a path outside the target: {name}")


def extract(archive, root):
    """Unpack a release archive into `root`, flattening its top directory.

    Upstream wraps the binaries in a build/bin directory on some platforms and
    not on others. What every consumer needs is one directory that holds
    llama-server next to the shared libraries it loads, so this puts them there
    rather than making every caller learn both layouts.
    """
    archive = Path(archive)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="llamacpp-", dir=root.parent))
    try:
        try:
            if archive.name.endswith(".zip"):
                with zipfile.ZipFile(archive) as bundle:
                    _safe_members(bundle.namelist(), staging.resolve())
                    bundle.extractall(staging)
            else:
                with tarfile.open(archive) as bundle:
                    _safe_members(bundle.getnames(), staging.resolve())
                    # The data filter refuses absolute paths, links out of the
                    # tree and device nodes. The check above is not made
                    # redundant by it: one is this program's rule, the other is
                    # the library's, and either one alone would be a single
                    # point of failure for the same attack.
                    bundle.extractall(staging, filter="data")
        except (tarfile.TarError, zipfile.BadZipFile, EOFError, OSError) as error:
            # Whatever the archive did wrong, the caller is asking this module
            # whether the install can proceed, and the answer is no with a
            # reason. Letting a library's own exception type escape turned a
            # refusal into a traceback, which is a crash dressed as security.
            raise InstallError(f"the archive could not be unpacked: {error}") from error
        found = next(
            (path.parent for path in staging.rglob(SERVER_NAME)
             if path.is_file()), None)
        if found is None:
            raise InstallError(
                f"the archive contains no {SERVER_NAME}, so it is not a build "
                "this program can serve models with")
        # Listed before moving. iterdir() is a generator over a directory that
        # this loop is emptying, and mutating a directory while walking it is
        # how a file gets skipped without anybody noticing which one.
        for item in sorted(found.iterdir()):
            destination = root / item.name
            if destination.exists() or destination.is_symlink():
                if destination.is_dir() and not destination.is_symlink():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(item), str(destination))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    server = root / SERVER_NAME
    if server.exists():
        server.chmod(server.stat().st_mode | 0o111)
    return server


def _environment(root, environ=None):
    """The library path a relocated build needs to load its own shared objects.

    The RPATH upstream compiles in names the machine it was built on, and the
    shared libraries sit beside the binary rather than in a lib directory. This
    is the same fact that made the first prepared Kaggle runtime fail to load.
    """
    environ = dict(os.environ if environ is None else environ)
    existing = environ.get("LD_LIBRARY_PATH", "")
    environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(root)] + ([existing] if existing else []))
    return environ


def list_devices(executable, run_fn=subprocess.run, environ=None):
    """The compute devices this build can actually see.

    This is the verification that matters. --version proves a binary runs;
    --list-devices proves the backend inside it found hardware to run on, which
    is the difference between an installed Vulkan build and a usable one.
    """
    executable = Path(executable)
    try:
        result = run_fn(
            [str(executable), "--list-devices"], check=False,
            capture_output=True, text=True, timeout=60,
            env=_environment(executable.parent, environ),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise InstallError(str(error)) from error
    if result.returncode != 0:
        raise InstallError(
            (result.stderr or result.stdout or "").strip()
            or f"{SERVER_NAME} --list-devices failed")
    devices = []
    for line in (result.stdout or "").splitlines():
        # "  Vulkan1: NVIDIA GeForce GTX 1650 (4342 MiB, 1547 MiB free)"
        match = re.match(
            r"\s+(?P<id>\S+):\s+(?P<name>.+?)\s+\((?P<total>\d+)\s*MiB"
            r"(?:,\s*(?P<free>\d+)\s*MiB free)?\)\s*$", line)
        if match:
            devices.append({
                "id": match.group("id"),
                "name": match.group("name"),
                "total_mb": int(match.group("total")),
                "free_mb": int(match.group("free")) if match.group("free") else None,
            })
    return devices


def dedicated_devices(devices, gpus=None):
    """The ids a second source confirms as cards with memory of their own.

    Reported memory does not tell these apart: an integrated GPU on this machine
    advertises 11859 MiB of shared system RAM against a discrete card's 4342 MiB,
    so the largest number is the slower device. What settles it is whether the
    card is also reported by nvidia-smi, which is a separate observation of the
    same hardware rather than a preference written here.
    """
    known = [str(item.get("name", "")).casefold()
             for item in (hardware.gpus() if gpus is None else gpus)]
    return {
        device["id"] for device in devices or []
        if any(entry and (entry in str(device.get("name", "")).casefold()
                          or str(device.get("name", "")).casefold() in entry)
               for entry in known)
    }


def preferred_device(devices, gpus=None):
    """The device to offer first: a card of its own, not the biggest number."""
    if not devices:
        return None
    ours = dedicated_devices(devices, gpus)
    return next((device for device in devices if device["id"] in ours), devices[0])


def context_ceiling(model, vram_mb, overhead_mb=None, device_free_mb=None,
                    cache_type=DEFAULT_CACHE_TYPE):
    """How much context this machine can actually hold for this model.

    Returns (ceiling, reason). The ceiling is the smaller of what the weights
    leave room for and what the model was trained for, and the reason names
    which of the two won, because those are different problems: one is solved
    by a smaller quantization and the other cannot be solved at all.

    `cache_type` is the KV cache precision this ceiling is measured at, one of
    the keys in `CACHE_TYPES`. A quantized cache spends fewer bytes per token,
    which is memory arithmetic, not a claim about what a smaller cache does to
    output quality; nothing here measures that.

    This is the number a hand-written launch script cannot know. The one on
    this machine froze -c 8192 into place, and that single frozen value
    falsified two measurements before anybody noticed it was there.
    """
    trained = model.get("context_length") or 0
    missing = [key for key in ("n_layers", "n_kv_heads", "head_dim")
               if not model.get(key)]
    if missing:
        return (trained or 0), "unknown_geometry"
    if cache_type not in CACHE_TYPES:
        raise ValueError(f"unknown cache type: {cache_type!r}")
    bytes_per_element = CACHE_TYPES[cache_type]
    overhead_mb = hardware.DEFAULT_OVERHEAD_MB if overhead_mb is None else overhead_mb
    # Free memory beats total memory when the driver reports it: a desktop
    # session already spent some of the card before this program asked.
    usable_mb = device_free_mb if device_free_mb else vram_mb
    room = hardware.max_context_that_fits(
        model["model_bytes"], model["n_layers"], model["n_kv_heads"],
        model["head_dim"], usable_mb, overhead_mb=overhead_mb,
        bytes_per_element=bytes_per_element,
    )
    if not room:
        return 0, "does_not_fit"
    if trained and trained <= room:
        return trained, "trained"
    return room, "memory"


def server_command(executable, model_path, context, device=None, host="127.0.0.1",
                   port=8080, alias=None, gpu_layers=99,
                   cache_type_k=None, cache_type_v=None):
    """The exact invocation isaacli uses to serve a local model.

    Everything decided here was previously frozen in a shell script outside the
    repository: the context, the device, the offload. --jinja is not optional
    and not a preference, it is what makes llama-server render the chat
    template out of the GGUF, which is the only reason a model can be talked to
    at all.

    `cache_type_k` and `cache_type_v` are the KV cache precisions the caller
    decided (see `context_ceiling`), one of the keys in `CACHE_TYPES`. Left
    None, llama-server keeps its own default, f16, and neither flag is
    emitted; a caller that knows the profile's cache type passes it so the
    running server matches the ceiling that was shown for it.
    """
    command = [str(executable), "-m", str(model_path)]
    if alias:
        command += ["--alias", str(alias)]
    if device:
        command += ["-dev", str(device)]
    command += [
        "-ngl", str(gpu_layers),
        "-c", str(int(context)),
    ]
    if cache_type_k:
        command += ["--cache-type-k", str(cache_type_k)]
    if cache_type_v:
        command += ["--cache-type-v", str(cache_type_v)]
    command += [
        "--host", str(host), "--port", str(port),
        "--jinja",
    ]
    return command


def installed(record_path=None):
    """The llama.cpp this program installed, or None.

    Reads the record rather than looking for a binary: a llama-server found on
    PATH belongs to the user, and this program must never remove or claim it.
    """
    record = install_record(record_path)
    if not record.exists():
        return None
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
        root = Path(data["root"])
        executable = Path(data["executable"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        debug.note("llama_cpp.installed", str(error))
        return None
    if not executable.exists():
        debug.note("llama_cpp.installed",
                   f"the recorded executable is gone: {executable}")
        return None
    data["root"] = root
    data["executable"] = executable
    return data


def profile_servers(data):
    """llama-server binaries reachable through a saved profile's autostart.

    A build somebody compiled themselves is usually not on PATH: it sits in the
    working directory it was built in, launched by a wrapper script next to it.
    PATH alone therefore reports "no llama.cpp here" on a machine that has been
    serving GGUF files all week, and the screen then offers to install a second
    copy of what is already running. The profile names the command it launches,
    so the binary is read off that command instead of guessed: the command
    itself when it is the server, otherwise the server sitting beside it.
    """
    found = []
    for item in (data.get("profiles") or {}).values():
        command = (item.get("autostart") or {}).get("cmd") or []
        if not command or not str(command[0]).strip():
            continue
        # An empty first word would make the sibling candidate a bare relative
        # name, and then whatever the working directory happens to hold decides
        # which server this program is about to run.
        first = Path(str(command[0]))
        for candidate in (first, first.parent / SERVER_NAME):
            if candidate.name == SERVER_NAME and _executable(candidate):
                found.append(candidate)
                break
    return found


def _executable(path):
    """A file this machine can run. A directory answers X_OK too, and a
    directory handed to Popen is an exception, not a server."""
    try:
        return Path(path).is_file() and os.access(path, os.X_OK)
    except OSError:
        debug.note("llama_cpp._executable", f"unreadable: {path}")
        return False


def find_server(record_path=None, which_fn=shutil.which, candidates=()):
    """A usable llama-server and who it belongs to.

    Returns (path, owner) with owner "isaacli" or "user", or (None, None). The
    user's own build wins nothing and loses nothing by being found here; it is
    simply used as it is, and never recorded as ours.
    """
    ours = installed(record_path)
    if ours:
        return ours["executable"], "isaacli"
    found = which_fn(SERVER_NAME)
    if found:
        return Path(found), "user"
    for candidate in candidates:
        if _executable(candidate):
            return Path(candidate), "user"
    return None, None


def write_record(data, record_path=None):
    """Record ownership, and only after the install has been proven to work.

    Written at the moment of installing because it cannot be inferred later:
    a directory full of binaries says nothing about who put it there.
    """
    record = install_record(record_path)
    record.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(record, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    return record


def install(asset, record_path=None, home_dir=None,
            urlopen_fn=urllib.request.urlopen, run_fn=subprocess.run,
            cache_dir=None):
    """Download, unpack and prove one build, then record that we installed it.

    The order is the whole contract. Nothing is recorded until --list-devices
    has answered, so a half-finished install never leaves behind a claim of
    ownership over a directory that does not work; and a failure takes its own
    directory back out rather than leaving debris for the next attempt to trip
    over.
    """
    root = install_root(home_dir)
    if root.exists() and any(root.iterdir()) and not installed(record_path, home_dir):
        # Something is already there that this program did not put there, or
        # did not finish putting there. Refusing beats overwriting somebody's
        # directory, and the message says which directory to look at.
        raise InstallError(
            f"{root} already has files in it and no record says this program "
            "installed them")
    cache = Path(cache_dir) if cache_dir else config.cache_path() / "llama.cpp"
    archive = cache / str(asset.get("name") or "llama.cpp-build")
    created = not root.exists()
    try:
        download(asset, archive, urlopen_fn=urlopen_fn)
        executable = extract(archive, root)
        devices = list_devices(executable, run_fn=run_fn)
        record = write_record({
            "version": 1,
            "installed_by": "isaacli",
            "root": str(root),
            "executable": str(executable),
            "build": asset.get("build"),
            "backend": asset.get("backend"),
            "asset": asset.get("name"),
            "digest": asset.get("digest"),
        }, record_path)
    except InstallError:
        if created:
            shutil.rmtree(root, ignore_errors=True)
        raise
    finally:
        try:
            if archive.exists():
                archive.unlink()
            # And the directory it was staged in, when nothing else is using
            # it. Leaving an empty folder behind after a successful install is
            # a small trace, and small traces are the ones nobody comes back
            # for. rmdir refuses a directory with anything in it, which is
            # exactly the guard wanted here.
            if cache.is_dir():
                cache.rmdir()
        except OSError:
            debug.swallowed("llama_cpp.install archive cleanup")
    return {"executable": executable, "devices": devices, "record": record,
            "backend": asset.get("backend"), "build": asset.get("build")}


def uninstall(record_path=None, home_dir=None, package_owned_fn=None):
    """Remove the llama.cpp this program installed, and refuse anything else.

    Returns (code, reason_key, values) so the caller says it in the user's
    language. Every refusal names what it refused and why.
    """
    from installation import _package_owns
    package_owned_fn = _package_owns if package_owned_fn is None else package_owned_fn
    record = install_record(record_path)
    if not record.exists():
        return 1, "cli.uninstall.llamacpp.not_managed", {}
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
        root = Path(data["root"])
        executable = Path(data["executable"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        return 1, "cli.uninstall.llamacpp.invalid_record", {"error": error}

    expected = install_root(home_dir)
    if root.absolute() != expected.absolute():
        return 1, "cli.uninstall.llamacpp.unsafe_path", {
            "path": root, "expected": expected}
    if executable.exists() and package_owned_fn(executable):
        return 1, "cli.uninstall.llamacpp.package_owned", {"path": executable}
    # If taking it away needs an administrator, an administrator put it there,
    # and it is not ours to take away. Checked by asking the filesystem rather
    # than by running sudo to find out.
    if root.exists() and not os.access(root.parent, os.W_OK):
        return 1, "cli.uninstall.llamacpp.needs_root", {"path": root}
    if root.exists():
        owner = root.stat().st_uid
        if owner != os.getuid():
            return 1, "cli.uninstall.llamacpp.needs_root", {"path": root}
    try:
        if root.exists():
            shutil.rmtree(root)
        record.unlink()
    except OSError as error:
        return 1, "cli.uninstall.failed", {"error": error}
    return 0, "cli.uninstall.llamacpp.removed", {"path": root}
