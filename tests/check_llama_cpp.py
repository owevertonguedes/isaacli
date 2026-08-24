#!/usr/bin/env python3
"""Offline checks for the llama.cpp install, recognition and removal lifecycle.

Nothing here reaches the network and nothing here touches the real
~/.local/share or the developer's own llama.cpp. The release listing is served
from a fixture, the archive is built by this check, and every path is inside a
temporary directory, because a removal check that runs against a real home
directory is a removal check that removes somebody's real files once.
"""
import io
import json
import os
import stat
import sys
import tarfile
import tempfile
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

import hardware
import llama_cpp
import model_discovery


failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


root = Path(tempfile.mkdtemp(prefix="isaacli-llamacpp-"))
home = root / "home"
home.mkdir()
os.environ.pop("XDG_DATA_HOME", None)

# --- reading upstream's asset names -----------------------------------------

check(llama_cpp.parse_asset("llama-b10595-bin-ubuntu-vulkan-x64.tar.gz") == {
    "build": "b10595", "system": "ubuntu", "backend": "vulkan",
    "arch": "x64", "extension": "tar.gz"},
    "a backend build is read out of its name")
check(llama_cpp.parse_asset("llama-b10595-bin-ubuntu-x64.tar.gz")["backend"] == "cpu",
      "a name with no backend section is the CPU build, not an unparsable one")
check(llama_cpp.parse_asset(
    "llama-b10595-bin-ubuntu-rocm-7.14-x64.tar.gz")["backend"] == "rocm-7.14",
    "a versioned backend keeps its version instead of being cut in half")
check(llama_cpp.parse_asset("cudart-llama-bin-win-cuda-12.4-x64.zip") is None,
      "the CUDA runtime archive is not mistaken for a llama.cpp build")
check(llama_cpp.parse_asset("llama-b10595-xcframework.zip") is None,
      "an asset that is not a platform build is skipped")

check(llama_cpp._target("linux", "x86_64") == ("ubuntu", "x64")
      and llama_cpp._target("darwin", "arm64") == ("macos", "arm64"),
      "this platform is named the way upstream names its assets")

# --- choosing a backend for the hardware that is here ------------------------

drm = root / "drm"
for index, vendor in enumerate(["0x8086", "0x10de"]):
    device = drm / f"card{index}" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text(vendor + "\n", encoding="utf-8")
check(llama_cpp.gpu_vendors(drm) == ["intel", "nvidia"],
      "every GPU vendor on the machine is read from the kernel, not only NVIDIA")
check(llama_cpp.gpu_vendors(root / "nothing-here") == [],
      "a machine with no drm directory reports no vendors instead of failing")

order = llama_cpp.backend_order(drm_root=drm, system="linux")
check(order[-1] == "cpu" and "vulkan" in order,
      f"the CPU build is always the last resort and never absent (got {order})")
check(llama_cpp.backend_order(vendors=[], system="linux") == ["cpu"],
      "a machine with no GPU is offered the CPU build and nothing else")
check(llama_cpp.backend_order(vendors=["nvidia"], system="darwin") == [],
      "macOS is not offered a backend menu it does not have")

# Upstream publishes no Linux CUDA build, so an NVIDIA machine has to land on
# the build that can actually address the card.
nvidia_assets = {"vulkan": {"backend": "vulkan"}, "cpu": {"backend": "cpu"}}
chosen, skipped = llama_cpp.choose_asset(
    nvidia_assets, llama_cpp.backend_order(vendors=["nvidia"], system="linux"))
check(chosen["backend"] == "vulkan" and skipped == ["cuda"],
      "an NVIDIA machine gets the Vulkan build, and the missing CUDA build is named")
check(llama_cpp.choose_asset({}, ["vulkan", "cpu"]) == (None, ["vulkan", "cpu"]),
      "a platform with no published build says so instead of choosing nothing quietly")

# --- the release listing -----------------------------------------------------

def release_payload(build="b10595"):
    def asset(name, digest="sha256:" + "0" * 64):
        return {"name": name, "browser_download_url": f"https://example/{name}",
                "size": 1, "digest": digest}
    return [
        # The newest tag upstream publishes carries no binaries at all, which is
        # why asking the API for "the latest release" answers the wrong thing.
        {"tag_name": "v0.2.0", "assets": [asset("nightly-tag.txt")]},
        {"tag_name": build, "assets": [
            asset(f"llama-{build}-bin-ubuntu-vulkan-x64.tar.gz"),
            asset(f"llama-{build}-bin-ubuntu-x64.tar.gz"),
            asset(f"llama-{build}-bin-win-cuda-13.3-x64.zip"),
            asset(f"llama-{build}-bin-macos-arm64.tar.gz"),
        ]},
    ]


class FakeResponse(io.BytesIO):
    """A real stream, because a fake one hid a real defect.

    An earlier version of this file answered read() with the whole payload
    however many bytes were asked for and never signalled the end. The download
    loop did what it was told and wrote until this machine's /tmp, which is RAM,
    was full. Deriving from BytesIO means read() cannot lie about its size or
    its end, so what this exercises is the code and not the stub.
    """

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class EndlessResponse:
    """A server that keeps answering, to prove the download stops anyway."""

    def __init__(self, block):
        self.block = block

    def read(self, size=-1):
        return self.block[:size] if size and size > 0 else self.block

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def listing_urlopen(_request, timeout=None):
    return FakeResponse(json.dumps(release_payload()).encode())


tag, assets = llama_cpp.available_builds(
    urlopen_fn=listing_urlopen, system="linux", machine="x86_64")
check(tag == "b10595" and sorted(assets) == ["cpu", "vulkan"],
      f"the newest tag that actually has binaries is the one used (got {tag}, {sorted(assets)})")
check("cuda" not in assets,
      "a build for another operating system is not offered to this one")

_tag, mac_assets = llama_cpp.available_builds(
    urlopen_fn=listing_urlopen, system="darwin", machine="arm64")
check(sorted(mac_assets) == ["cpu"],
      "macOS gets its single build, which is the one with Metal inside it")

_tag, none_assets = llama_cpp.available_builds(
    urlopen_fn=listing_urlopen, system="linux", machine="s390x")
check(none_assets == {},
      "a platform upstream did not build for comes back empty, not wrong")

# --- downloading, and refusing bytes that are not the published ones ---------

def build_archive(path, server_body="#!/bin/sh\n", nested=True, include_server=True):
    """A release archive shaped the way upstream ships one."""
    staging = Path(tempfile.mkdtemp(dir=root))
    inner = staging / "build" / "bin" if nested else staging
    inner.mkdir(parents=True, exist_ok=True)
    if include_server:
        server = inner / "llama-server"
        server.write_text(server_body, encoding="utf-8")
        server.chmod(0o755)
    (inner / "libggml.so").write_text("library", encoding="utf-8")
    with tarfile.open(path, "w:gz") as bundle:
        bundle.add(staging, arcname=".")
    return Path(path)


archive = build_archive(root / "release.tar.gz")
digest = "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest()
archive_size = archive.stat().st_size


def download_urlopen(_request, timeout=None):
    return FakeResponse(archive.read_bytes())


def endless_urlopen(_request, timeout=None):
    return EndlessResponse(archive.read_bytes())


def published(**overrides):
    entry = {"url": "https://example/x.tar.gz", "digest": digest,
             "size": archive_size}
    entry.update(overrides)
    return entry


def refuses_download(asset, name, description, urlopen_fn=download_urlopen,
                     expect=None):
    try:
        llama_cpp.download(asset, root / name, urlopen_fn=urlopen_fn)
    except llama_cpp.InstallError as error:
        check(expect is None or expect in str(error), description)
    else:
        check(False, f"{description} (it accepted the file instead)")


fetched = llama_cpp.download(published(), root / "fetched.tar.gz",
                             urlopen_fn=download_urlopen)
check(fetched.read_bytes() == archive.read_bytes(),
      "a download that matches its published digest is accepted")

refuses_download(
    published(digest="sha256:" + "ff" * 32), "bad.tar.gz",
    "bytes that do not match the published digest are refused, with the reason",
    expect="digest")
refuses_download(
    published(digest=None), "nodigest.tar.gz",
    "a release that published no digest is refused rather than trusted")

# A connection that never ends must not decide how much of somebody's disk gets
# written. This is the defect that filled this machine's RAM while this check
# was being written, so it is a check and not a comment.
refuses_download(
    published(), "endless.tar.gz",
    "a server that keeps answering is cut off at the size the release declared",
    urlopen_fn=endless_urlopen, expect="exceeded")
check((root / "endless.tar.gz").stat().st_size <= archive_size,
      "and nothing beyond that declared size ever reached the disk")

refuses_download(
    published(size=archive_size * 2), "short.tar.gz",
    "a download that ends early is refused instead of unpacked half-written",
    expect="ended at")

# --- unpacking, including an archive built to escape ------------------------

target = root / "unpacked"
server = llama_cpp.extract(archive, target)
check(server.exists() and (target / "libggml.so").exists(),
      "the binary and the libraries it loads land in one directory")
check(os.access(server, os.X_OK),
      "the extracted server is executable, whatever the archive recorded")

escaping = root / "escaping.tar.gz"
victim = root / "victim.txt"
victim.write_text("original", encoding="utf-8")
with tarfile.open(escaping, "w:gz") as bundle:
    info = tarfile.TarInfo("../victim.txt")
    payload = b"overwritten"
    info.size = len(payload)
    bundle.addfile(info, io.BytesIO(payload))
def refuses_extract(path, target, description, expect=None):
    """Report how the refusal came out instead of letting it abort the run.

    A bare `except InstallError` here turned a wrong exception type into a
    traceback that ended the whole check, so the run said nothing at all about
    the other twenty assertions. A check has to report its result, including
    when the result is that the code failed in an unexpected way.
    """
    try:
        llama_cpp.extract(path, target)
    except llama_cpp.InstallError as error:
        check(expect is None or expect in str(error), description)
    except Exception as error:  # noqa: BLE001 - the type is exactly the point
        check(False, f"{description} (raised {type(error).__name__} instead)")
    else:
        check(False, f"{description} (it unpacked the archive instead)")


refuses_extract(escaping, root / "escape-target",
                "an archive naming a path outside the target is refused")
check(victim.read_text(encoding="utf-8") == "original",
      "and the file it aimed at is untouched")

no_server = build_archive(root / "no-server.tar.gz", include_server=False)
refuses_extract(no_server, root / "no-server-target",
                "an archive without llama-server is refused by name, not installed empty",
                expect="llama-server")

# --- proving the build runs before claiming it -------------------------------

DEVICE_OUTPUT = (
    "Available devices:\n"
    "  Vulkan0: Intel(R) UHD Graphics 630 (CFL GT2) (11859 MiB, 6735 MiB free)\n"
    "  Vulkan1: NVIDIA GeForce GTX 1650 (4342 MiB, 1547 MiB free)\n"
)


class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def working_run(command, **kwargs):
    return FakeResult(stdout=DEVICE_OUTPUT)


def failing_run(command, **kwargs):
    return FakeResult(returncode=1, stderr="vulkan: no usable devices found")


devices = llama_cpp.list_devices(server, run_fn=working_run)
check([d["id"] for d in devices] == ["Vulkan0", "Vulkan1"]
      and devices[1]["free_mb"] == 1547,
      "every device the build can see is read, with its free memory")

# Reported size ranks these the wrong way round: the integrated GPU advertises
# 11859 MiB of shared system RAM against the discrete card's 4342 MiB.
preferred = llama_cpp.preferred_device(
    devices, gpus=[{"name": "NVIDIA GeForce GTX 1650", "vram_mb": 4096}])
check(preferred["id"] == "Vulkan1",
      "the discrete card is preferred over the larger-looking integrated one")
check(llama_cpp.preferred_device(devices, gpus=[])["id"] == "Vulkan0",
      "with no second source to confirm a card, the first device is offered, not a guess")
check(llama_cpp.preferred_device([], gpus=[]) is None,
      "no devices at all is None, not an index error")

try:
    llama_cpp.list_devices(server, run_fn=failing_run)
except llama_cpp.InstallError as error:
    check("no usable devices" in str(error),
          "a backend that finds no hardware fails with what it printed")
else:
    check(False, "a backend that finds no hardware fails with what it printed")

# --- the whole install, and what it records ----------------------------------

record_path = root / "llamacpp-install.json"
install_home = root / "install-home"
install_home.mkdir()
asset = published(name="llama-b10595-bin-ubuntu-vulkan-x64.tar.gz",
                  backend="vulkan", build="b10595")

result = llama_cpp.install(
    asset, record_path=record_path, home_dir=install_home,
    urlopen_fn=download_urlopen, run_fn=working_run, cache_dir=root / "cache")
check(result["executable"].exists() and result["devices"],
      "a finished install leaves a server that answered --list-devices")
recorded = json.loads(record_path.read_text(encoding="utf-8"))
check(recorded["installed_by"] == "isaacli" and recorded["backend"] == "vulkan",
      "the record says we installed it, and which build, at the moment it happened")
check(stat.S_IMODE(record_path.stat().st_mode) == 0o600,
      "the ownership record is written for this user only")
check(not (root / "cache" / asset["name"]).exists(),
      "the downloaded archive is not left behind after it has been unpacked")
check(not (root / "cache").exists(),
      "and neither is the empty directory it was staged in")

# A failure must not leave a claim of ownership behind.
failed_record = root / "failed-install.json"
failed_home = root / "failed-home"
failed_home.mkdir()
try:
    llama_cpp.install(asset, record_path=failed_record, home_dir=failed_home,
                      urlopen_fn=download_urlopen, run_fn=failing_run,
                      cache_dir=root / "cache")
except llama_cpp.InstallError:
    check(not failed_record.exists()
          and not llama_cpp.install_root(failed_home).exists(),
          "an install that could not prove itself records nothing and leaves nothing")
else:
    check(False, "an install that could not prove itself records nothing and leaves nothing")

# --- recognising whose llama-server this is ----------------------------------

found, owner = llama_cpp.find_server(record_path=record_path)
check(owner == "isaacli" and found == result["executable"],
      "the build this program installed is recognised as ours")
found, owner = llama_cpp.find_server(
    record_path=root / "absent.json",
    which_fn=lambda name: "/usr/bin/llama-server")
check(owner == "user" and str(found) == "/usr/bin/llama-server",
      "a llama-server the user already had is used, and never claimed as ours")
found, owner = llama_cpp.find_server(
    record_path=root / "absent.json",
    which_fn=lambda name: None)
check((found, owner) == (None, None),
      "no llama-server anywhere is an answer, not a failure")

# A build somebody compiled themselves is not on PATH: the developer's own is
# launched by a wrapper script sitting beside it. PATH alone reported "nothing
# here" on a machine that had been serving GGUF files all week, and /model then
# offered to install a second copy of the server it was talking to.
hand_built = root / "hand-built"
hand_built.mkdir()
(hand_built / "llama-server").write_text("#!/bin/sh\n", encoding="utf-8")
(hand_built / "llama-server").chmod(0o755)
(hand_built / "start-llama-server.sh").write_text("#!/bin/sh\n", encoding="utf-8")
(hand_built / "start-llama-server.sh").chmod(0o755)
by_script = llama_cpp.profile_servers({"profiles": {
    "hand-made": {"autostart": {"cmd": [str(hand_built / "start-llama-server.sh")]}},
}})
check([str(item) for item in by_script] == [str(hand_built / "llama-server")],
      "the server beside a profile's wrapper script is found")
by_binary = llama_cpp.profile_servers({"profiles": {
    "ours": {"autostart": {"cmd": [str(hand_built / "llama-server"), "-m", "x.gguf"]}},
}})
check([str(item) for item in by_binary] == [str(hand_built / "llama-server")],
      "and so is one the profile launches directly")
check(llama_cpp.profile_servers({"profiles": {
          "remote": {"base_url": "https://api.example/v1"},
          "gone": {"autostart": {"cmd": [str(root / "absent" / "llama-server")]}},
      }}) == [],
      "a profile with no autostart, or one naming a path that is gone, offers nothing")
found, owner = llama_cpp.find_server(
    record_path=root / "absent.json", which_fn=lambda name: None,
    candidates=by_script)
check(owner == "user" and found == hand_built / "llama-server",
      "and find_server uses it, off PATH, without ever claiming it as ours")

# --- removal, and every refusal it owes the user -----------------------------

code, key, _values = llama_cpp.uninstall(
    record_path=root / "absent.json", home_dir=install_home,
    package_owned_fn=lambda path: False)
check(code == 1 and key.endswith("not_managed"),
      "removal refuses what it has no record of installing")

code, key, _values = llama_cpp.uninstall(
    record_path=record_path, home_dir=install_home,
    package_owned_fn=lambda path: True)
check(code == 1 and key.endswith("package_owned")
      and llama_cpp.install_root(install_home).exists(),
      "removal refuses an executable the package manager owns, and removes nothing")

elsewhere = root / "elsewhere.json"
elsewhere.write_text(json.dumps({
    "root": "/usr/local/lib/llama.cpp",
    "executable": str(result["executable"]),
}), encoding="utf-8")
code, key, _values = llama_cpp.uninstall(
    record_path=elsewhere, home_dir=install_home,
    package_owned_fn=lambda path: False)
check(code == 1 and key.endswith("unsafe_path"),
      "a record pointing outside this program's own directory is refused")

code, key, _values = llama_cpp.uninstall(
    record_path=record_path, home_dir=install_home,
    package_owned_fn=lambda path: False)
check(code == 0 and not llama_cpp.install_root(install_home).exists()
      and not record_path.exists(),
      "removal takes away the directory it installed and the record with it")

code, key, _values = llama_cpp.uninstall(
    record_path=record_path, home_dir=install_home,
    package_owned_fn=lambda path: False)
check(code == 1 and key.endswith("not_managed"),
      "repeating a completed removal says so instead of failing differently")

# --- the context this program decides, instead of a script freezing one ------

# Grouped-query attention is what decides whether memory binds at all: this is
# the same weight file with eight KV heads instead of two, and the cache it
# needs is four times larger for every token of context.
model = {"model_bytes": 2_007_400_000, "n_layers": 36, "n_kv_heads": 8,
         "head_dim": 128, "context_length": 32768}
ceiling, reason = llama_cpp.context_ceiling(model, vram_mb=4096)
check(reason == "memory" and 0 < ceiling < 32768,
      f"on a 4 GB card the memory decides the context, not the training ({ceiling}, {reason})")
check(hardware.fits(model["model_bytes"],
                    hardware.kv_cache_bytes(36, 8, 128, ceiling), 4096),
      "the context this returns is one that actually fits, checked by the same arithmetic")
check(not hardware.fits(model["model_bytes"],
                        hardware.kv_cache_bytes(36, 8, 128, ceiling + 1), 4096),
      "and it is the largest one that does, not a cautious fraction of it")

ceiling, reason = llama_cpp.context_ceiling(model, vram_mb=24576)
check((ceiling, reason) == (32768, "trained"),
      "a card with room stops at what the model was trained for, and says so")

# The model and the card this project actually measures on, with the geometry
# read from the real file: qwen2.5-coder-3b Q4_K_M on a GTX 1650. The launch
# script outside the repository froze -c 8192 into place, and that frozen value
# falsified two measurements. What the card can hold is several times that, and
# this is the check that says so in numbers rather than in a comment.
this_machine = {"model_bytes": 2_104_932_800, "n_layers": 36, "n_kv_heads": 2,
                "head_dim": 128, "context_length": 32768}
ceiling, reason = llama_cpp.context_ceiling(this_machine, vram_mb=4096)
check(ceiling >= 32768 and reason == "trained",
      f"the 3B this project measures on holds its whole trained context on a 4 GB "
      f"card, not the 8192 a hand-written script froze in (got {ceiling}, {reason})")

ceiling, reason = llama_cpp.context_ceiling(
    {"model_bytes": 40_000_000_000, "n_layers": 36, "n_kv_heads": 2,
     "head_dim": 128, "context_length": 32768}, vram_mb=4096)
check((ceiling, reason) == (0, "does_not_fit"),
      "weights that do not fit at all report that, rather than a context of zero with no reason")

ceiling, reason = llama_cpp.context_ceiling(
    {"model_bytes": 1, "context_length": 8192}, vram_mb=4096)
check((ceiling, reason) == (8192, "unknown_geometry"),
      "a model whose geometry could not be read falls back to what it was trained for, and names why")

# Free memory beats total: the desktop session already spent some of the card.
tight, _reason = llama_cpp.context_ceiling(model, vram_mb=4096, device_free_mb=1547)
check(tight < llama_cpp.context_ceiling(model, vram_mb=4096)[0],
      "the context is decided against memory that is actually free, when that is known")

command = llama_cpp.server_command(
    "/bin/llama-server", "/models/x.gguf", 16384, device="Vulkan1", alias="x")
check(command[:3] == ["/bin/llama-server", "-m", "/models/x.gguf"]
      and "--jinja" in command and "-c" in command
      and command[command.index("-c") + 1] == "16384"
      and command[command.index("-dev") + 1] == "Vulkan1",
      "the launch command carries the chosen context, the chosen device and --jinja")

# --- the table the choice is actually made on -------------------------------

import setup_llamacpp
from i18n import Translator

tr = Translator("en")
GTX1650, OVERHEAD = 4096, 768
gtx = model_discovery.machine(vram_mb=GTX1650, gpu_count=1,
                              bandwidth_gbs=128.0, name="NVIDIA GeForce GTX 1650")


def fixture(**overrides):
    return dict({"name": "fixture", "model_bytes": 2_007_400_000, "n_layers": 36,
                 "n_kv_heads": 8, "head_dim": 128, "context_length": 32768,
                 "quantization": "Q4_K_M", "origin": "local"}, **overrides)


def cells_for(**overrides):
    item = fixture(**overrides)
    return item, setup_llamacpp._row(item, tr, GTX1650, OVERHEAD, 1, gtx)


def table_for(items, **kwargs):
    """Through the same call the screen makes, never through a parallel path.

    Task 055 names this trap by name: a check that entered through an orphan
    function proved a list no screen draws, twice.
    """
    rows = [setup_llamacpp._row(item, tr, GTX1650, OVERHEAD, 1, gtx)
            for item in items]
    return model_discovery.model_table(rows, gtx, translate=tr.t, **kwargs)


# A yes/no computed at one fixed context produced rows reading "does not fit"
# directly above a screen offering that same model seven thousand tokens.
item, row = cells_for()
check(row["fit"] == "9K ctx" and item["context_ceiling"] == 10052,
      f"the fit cell states the context this card holds, and the launch command carries it ({row['fit']})")
check(row["tps"].isdigit() and int(row["tps"]) > 0,
      f"the throughput cell is a bare number, with no word inside it ({row['tps']})")
check(row["name"] == "fixture Q4_K_M",
      "the precision is written once, not repeated out of the file name")

_item, tight = cells_for(model_bytes=3_400_000_000)
check("tight" in tight["fit"] and "0K" not in tight["fit"],
      f"a scrap of context is marked as tight and never rounded down to nothing ({tight['fit']})")
_item, huge = cells_for(model_bytes=40_000_000_000)
check(huge["fit"].lower().startswith("does not fit"),
      f"a model with no room at all still appears, saying it does not fit ({huge['fit']})")
# The estimate is bytes per token over the bus of the card that holds the
# weights, so it says nothing about a model this card does not hold. A row
# reading "does not fit" beside a confident throughput was on screen.
check(huge["tps"] == model_discovery.EMPTY_CELL,
      f"and it claims no throughput on a card that cannot hold it ({huge['tps']})")
_item, blind = cells_for(geometry_missing=["n_layers"])
check(blind["fit"] == tr.t("model.fit.unknown"),
      "a model whose geometry could not be read claims no context it cannot compute")

check(model_discovery._row_name({"name": "Q4_K_M", "quantization": "Q4_K_M"})
      == "Q4_K_M",
      "a file named only after its precision keeps a name at all")

# --- the header, drawn by the piece that draws the rows ----------------------

table = table_for([fixture(name="granite-4.1-3b"),
                   fixture(name="Hermes-3-Llama-3.2-3B", model_bytes=1_900_000_000),
                   fixture(name="LFM2.5-2.6B", model_bytes=700_000_000)],
                  state_header="WHERE")
lines = [table["header"], *table["rows"]]
check("GTX 1650" in table["header"] and "NVIDIA" not in table["header"],
      f"the card names the column once, trimmed to the part that names the part ({table['header']})")
check(all(len(line) <= 80 for line in lines),
      f"every line fits in 80 columns (widest was {max(len(line) for line in lines)})")

# Alignment is the point of a table, and it has to come from the whole list.
positions = [line.index("GiB") for line in table["rows"]]
check(len(set(positions)) == 1,
      f"the columns line up because the width came from the whole list, not from each row ({positions})")
tps_column = table["header"].index("TOK/S")
check(all(row[tps_column:].split()[0].isdigit() for row in table["rows"]),
      "and the header sits over the column it names, drawn from the same widths")

# A column whose every row says the same thing is a word repeated down the page.
check("WHERE" not in table["header"] and "your folder" in table["legend"],
      f"a column identical on every row is said once above the table instead ({table['header']})")

mixed = table_for([fixture(name="a"), fixture(name="b", origin="ollama")],
                  state_header="WHERE")
check("WHERE" in mixed["header"],
      "and it comes back as a column as soon as the rows differ")

# --- where the throughput comes from, said once and never in the cell --------

check("estimated" in table["legend"] and "GTX 1650" in table["legend"],
      f"the legend says the numbers are estimates and from which card ({table['legend']})")

no_bandwidth = model_discovery.machine(vram_mb=4096, gpu_count=1,
                                       bandwidth_gbs=None, name="Some GPU")
dark = model_discovery.model_table(
    [model_discovery.model_row(fixture(), no_bandwidth, translate=tr.t)],
    no_bandwidth, translate=tr.t)
check("-" in dark["rows"][0] and "no memory bandwidth is published" in dark["legend"],
      "a card with no published bandwidth gets dashes and a reason, never a guessed number")

benched = model_discovery.model_row(
    fixture(name="m", measured_here={"tokens_per_second": 36.2, "humaneval": "17/20"}),
    no_bandwidth, translate=tr.t)
check(benched["tps"] == "36" and benched["measured"] is True
      and "17/20" in benched["rankings"],
      f"a measurement of this exact file outranks any estimate ({benched})")
measured_table = model_discovery.model_table(
    [benched, model_discovery.model_row(fixture(name="other"), gtx, translate=tr.t)],
    gtx, translate=tr.t)
check("m" in measured_table["legend"] and "measured here" in measured_table["legend"],
      f"and the legend names which model was measured ({measured_table['legend']})")

# --- the weights a profile is already serving --------------------------------
#
# The screen that lists what this computer can run opened with nothing on it,
# on a machine that was serving a model at that very moment. It scanned only the
# folder this program downloads into, the Ollama store, and folders the user had
# registered by hand, so a weight in anybody's own directory was invisible even
# while it was answering requests.

import config

sys.path.insert(0, str(HERE))
from gguf_fixture import write_gguf, dense_keys

served_root = root / "served"
served_root.mkdir()
weight = write_gguf(served_root / "a-model-Q4_K_M.gguf", dense_keys())
sibling = write_gguf(served_root / "another-model-Q4_K_M.gguf", dense_keys())

check(setup_llamacpp._served_weight({"weights": str(weight)}) == weight,
      "the weights field a saved profile carries names the file being served")
check(setup_llamacpp._served_weight(
          {"autostart": {"cmd": ["llama-server", "-m", str(weight), "-c", "8192"]}})
      == weight,
      "and so does the -m of a launch command this program can read")
check(setup_llamacpp._served_weight({"weights": str(served_root / "gone.gguf")}) is None,
      "a recorded weight that is no longer on disk names nothing")


class FakeProps:
    """The /props answer of a llama-server, which is the only thing that knows
    the file when the launch command is a wrapper script."""

    def __init__(self, payload):
        self.payload = payload
        self.asked = []

    def __call__(self, url, timeout=None):
        self.asked.append(url)
        return self

    def __enter__(self):
        return io.BytesIO(json.dumps(self.payload).encode())

    def __exit__(self, *_exception):
        return False


props = FakeProps({"model_path": str(weight)})
wrapper = {"base_url": "http://127.0.0.1:8080/v1",
           "autostart": {"cmd": [str(served_root / "start-llama-server.sh")]}}
check(setup_llamacpp._served_weight(wrapper, props) == weight
      and props.asked == ["http://127.0.0.1:8080/props"],
      f"and a wrapper script is resolved by asking the running server ({props.asked})")

remote = FakeProps({"model_path": str(weight)})
check(setup_llamacpp._served_weight(
          {"base_url": "https://api.example.invalid/v1"}, remote) is None
      and remote.asked == [],
      "somebody else's endpoint is never asked: its file is on somebody else's disk")


def refuses(_url, timeout=None):
    raise OSError("connection refused")


check(setup_llamacpp._served_weight(wrapper, refuses) is None,
      "a server that is not answering costs the screen nothing and raises nothing")

served_config = root / "served-config.json"
config.save(dict(config.empty_config(), profiles={"served": wrapper}), served_config)
folders = setup_llamacpp.served_weight_dirs(served_config, props)
check(folders == [served_root],
      f"the folder is what is scanned, not the one file, so the models kept "
      f"beside it are offered too ({folders})")
import local_models

listed, _problems = local_models.available(extra_dirs=folders)
check({item["path"] for item in listed} == {str(weight), str(sibling)},
      "and both of them reach the list the choice is made on")

# --- the download screen ----------------------------------------------------
#
# It used to be a bare prompt asking for a "model reference", which is a
# question only somebody who already knows the answer can answer. What Hugging
# Face answers to "what is popular in GGUF" is also not an answer to "what can I
# run here": measured live on 2026-08-24 it was a page of 27B models, one of
# which survived resolution and did not fit the card asking. So the reviewed
# catalogue leads the list.
import setup_ollama

hub_screens = []
reviewed = dict(fixture(name="Reviewed-3B"), repo="org/Reviewed-3B-GGUF",
                file="reviewed-Q4_K_M.gguf", curated=True)
live_find = dict(fixture(name="Found-27B", model_bytes=15 * 1024 ** 3),
                 repo="org/Found-27B-GGUF", file="found-Q4_K_M.gguf")
original_curated = setup_ollama.curated_gguf_models
original_discover = model_discovery.discover_models
original_quantization = setup_ollama._choose_quantization
original_select = setup_llamacpp.terminal_ui.select
try:
    setup_ollama.curated_gguf_models = lambda *_a, **_k: [reviewed]
    model_discovery.discover_models = lambda *_a, **_k: ([live_find, reviewed], [])
    setup_ollama._choose_quantization = lambda model, *_a, **_k: model
    setup_llamacpp.terminal_ui.select = lambda title, options, **kwargs: (
        hub_screens.append((title, options)) or 0)
    picked = setup_llamacpp._choose_from_hub(tr, lambda _prompt="": "")
finally:
    setup_ollama.curated_gguf_models = original_curated
    model_discovery.discover_models = original_discover
    setup_ollama._choose_quantization = original_quantization
    setup_llamacpp.terminal_ui.select = original_select

_hub_title, hub_options = hub_screens[-1]
check(len(hub_options) == 4 and "Reviewed-3B" in hub_options[0],
      f"the download screen offers models, with the reviewed one first, "
      f"instead of an empty prompt ({hub_options})")
check("Found-27B" in hub_options[1]
      and (picked or {}).get("repo") == reviewed["repo"]
      and (picked or {}).get("file") == reviewed["file"],
      f"a live find is listed after it, once, and choosing a row returns the "
      f"file that row names ({(picked or {}).get('file')})")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("ISAAC LLAMA.CPP OK: backend choice, verified install, refusals and removal")
