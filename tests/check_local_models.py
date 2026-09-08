#!/usr/bin/env python3
"""Offline checks for GGUF header reading, disk discovery and Ollama reuse.

The fixtures are GGUF files this check writes itself. A check that needed the
developer's 2 GB weights would pass on one machine and be skipped everywhere
else, including CI, which is the same as not existing.
"""
import json
import struct
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tool_harness"))

import gguf
import local_models


failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


from gguf_fixture import (
    write_gguf, dense_keys, _string, _kv, UINT32, UINT64, STRING, ARRAY)


root = Path(tempfile.mkdtemp(prefix="isaacli-local-models-"))

# --- the header reader ------------------------------------------------------

dense = write_gguf(root / "dense-Q4_K_M.gguf", dense_keys())
shape = gguf.geometry(dense)
check(shape["n_layers"] == 28 and shape["n_kv_heads"] == 8
      and shape["head_dim"] == 128 and shape["context_length"] == 131072,
      "a dense header yields the geometry the fit calculation needs")
check(shape["chat_template"] is True,
      "the chat template is read while the vocabulary around it is skipped")
check(shape["name"] == "dense-Q4_K_M",
      "the name shown is the file name, not whatever general.name happens to hold")

# The vocabulary walk refills its block from the file, so a header bigger than
# one block has to come out identical to one that fits in a block. Without this,
# the fast path would be correct only for small fixtures.
big = write_gguf(root / "big-Q4_K_M.gguf", dense_keys(vocabulary=200_000))
check(big.stat().st_size > local_models.gguf._SKIP_BLOCK_BYTES,
      "the large fixture really is bigger than one read block")
check(gguf.geometry(big)["n_layers"] == 28,
      "a header spanning several read blocks is walked without losing its place")

# A hybrid model writes head_count_kv per layer, with 0 where the layer keeps no
# cache. Counting every block as an attention layer overstated LFM2's KV cache
# by three times on this machine.
hybrid_keys = dense_keys(architecture="lfm2", layers=30)
per_layer = [0, 0, 8] * 10
hybrid_keys["lfm2.attention.head_count_kv"] = (
    ARRAY, struct.pack("<I", UINT32) + struct.pack("<Q", len(per_layer))
    + b"".join(struct.pack("<I", value) for value in per_layer))
# The key that sizes what those 20 layers do hold. A real lfm2 header always
# carries it, and without it this fixture would now be exercising the branch
# that declines the discount rather than the one it was written for.
hybrid_keys["lfm2.shortconv.l_cache"] = (UINT32, struct.pack("<I", 3))
hybrid = write_gguf(root / "hybrid-Q4_K_M.gguf", hybrid_keys)
hybrid_shape = gguf.geometry(hybrid)
check(hybrid_shape["n_layers"] == 10 and hybrid_shape["block_count"] == 30,
      "only the layers that hold a cache are counted as attention layers")
check(hybrid_shape["n_kv_heads"] == 8,
      "a per-layer head count still yields the one head count the layers share")

# The other half of that same correction, which this path was missing: the 20
# layers just dropped out of the cache are not free, they hold a fixed state.
# Sizing them is the difference between an estimate that refuses a profile that
# would have loaded and one that saves a profile that will not, and only the
# second costs a session.
check(hybrid_shape["recurrent_bytes"] == 20 * 4 * 2 * 4096 * 4,
      f"the 20 layers the discount just dropped are billed for the state they "
      f"do hold: {hybrid_shape['recurrent_bytes']} bytes")

# And a hybrid whose state cannot be sized from its header declines the
# discount altogether, which is the pessimistic answer and the safe one: the
# other direction saves a profile that will not load.
unsizable_keys = dense_keys(architecture="mystery", layers=30)
unsizable_keys["mystery.attention.head_count_kv"] = (
    ARRAY, struct.pack("<I", UINT32) + struct.pack("<Q", len(per_layer))
    + b"".join(struct.pack("<I", value) for value in per_layer))
unsizable_shape = gguf.geometry(
    write_gguf(root / "unsizable-Q4_K_M.gguf", unsizable_keys))
check(unsizable_shape["n_layers"] == 30
      and unsizable_shape["recurrent_bytes"] == 0,
      f"an unsizable hybrid is billed for every layer rather than given a "
      f"discount with nothing added back: {unsizable_shape['n_layers']}/30, "
      f"{unsizable_shape['recurrent_bytes']} bytes")

# A short-convolution hybrid, which is the shape that can be sized from the
# header. The numbers are LFM2-1.2B's own, so the expectation below is what
# llama.cpp b10865 allocated for that file on 2026-09-08:
#   llama_memory_recurrent: size = 0.62 MiB (4 cells, 16 layers, 4 seqs)
shortconv_keys = dense_keys(architecture="lfm2", layers=16, embedding=2048)
shortconv_per_layer = [0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0]
shortconv_keys["lfm2.attention.head_count_kv"] = (
    ARRAY, struct.pack("<I", UINT32) + struct.pack("<Q", len(shortconv_per_layer))
    + b"".join(struct.pack("<I", value) for value in shortconv_per_layer))
shortconv_keys["lfm2.shortconv.l_cache"] = (UINT32, struct.pack("<I", 3))
shortconv = write_gguf(root / "shortconv-Q4_K_M.gguf", shortconv_keys)
shortconv_shape = gguf.geometry(shortconv)
check(shortconv_shape["n_layers"] == 6,
      f"the six caching layers of a short-convolution hybrid are the six "
      f"billed for cache: {shortconv_shape['n_layers']}")
check(shortconv_shape["recurrent_bytes"] == 655360,
      f"and its ten convolution layers are billed the 0.62 MiB llama.cpp "
      f"allocates for them, not zero: {shortconv_shape['recurrent_bytes']}")

# The discount is for hybrids and must not reach a dense model, where every
# block caches and there is no recurrent state to add back. Planting a discount
# on its own does not reach here, and that is the design working rather than
# this assertion being dead: a dense architecture cannot be sized, so the branch
# above declines the discount and hands back the full layer count. It takes both
# halves of the old defect at once, an escaping discount and a state worth zero,
# to make this line fail, and that is exactly the state this task found.
dense_shape = gguf.geometry(dense)
check(dense_shape["n_layers"] == dense_shape["block_count"] == 28
      and dense_shape["recurrent_bytes"] == 0,
      f"a dense model is billed for every layer and holds no recurrent state: "
      f"{dense_shape['n_layers']}/{dense_shape['block_count']}, "
      f"{dense_shape['recurrent_bytes']} bytes")

# And the state is resident whatever the context is, so it has to come off the
# top rather than out of the per-token division. Asserted by effect, through
# the ceiling the screen itself uses.
import llama_cpp  # noqa: E402  (only this block needs it)

shortconv_model = dict(shortconv_shape, model_bytes=700 * 1024 * 1024)
with_state, _reason = llama_cpp.context_ceiling(shortconv_model, vram_mb=2048)
without_state, _reason = llama_cpp.context_ceiling(
    dict(shortconv_model, recurrent_bytes=0), vram_mb=2048)
check(0 < with_state < without_state,
      f"the recurrent state comes off the memory the context is measured in: "
      f"{with_state} against {without_state} with the state ignored")

# --- refusing what it cannot read -------------------------------------------

def refuses(path, description):
    try:
        gguf.geometry(path)
    except gguf.GGUFError:
        check(True, description)
    except Exception as error:  # noqa: BLE001 - the point is that it does not
        check(False, f"{description} (raised {type(error).__name__}: {error})")
    else:
        check(False, f"{description} (it returned instead)")


refuses(write_gguf(root / "wrong-magic.gguf", dense_keys(), magic=b"GGUL"),
        "a file that is not GGUF is refused by name, not misread")
refuses(write_gguf(root / "old-version.gguf", dense_keys(), version=2),
        "an unsupported GGUF version is refused instead of parsed with the wrong layout")

truncated = root / "truncated.gguf"
truncated.write_bytes(dense.read_bytes()[:40])
refuses(truncated, "a header that ends early is refused instead of raising struct errors")

# A length read out of the file cannot be trusted to be smaller than the file.
# Without the bound, this line allocates 16 exabytes.
hostile = root / "hostile.gguf"
hostile.write_bytes(
    b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 1)
    + struct.pack("<Q", 2 ** 63) + b"key")
refuses(hostile, "a declared length larger than the file is refused, not allocated")

no_architecture = write_gguf(root / "no-arch.gguf", {
    "general.name": (STRING, _string("nameless")),
})
refuses(no_architecture, "a header that declares no architecture is refused")

# --- describing a file for the screens --------------------------------------

item = local_models.describe(dense)
check(item["model_bytes"] == dense.stat().st_size,
      "the size on screen is the size of the file, measured not estimated")
check(item["quantization"] == "Q4_K_M",
      "the precision is read off the file name the user sees")
check(item["benchmark"] == "" and item["scores"] == {},
      "a file on disk carries no published score, and claims none")
check(item["chat_template"] is True and item["origin"] == "local",
      "a local file is labelled as local and says whether it can be served")

no_geometry_keys = dense_keys()
del no_geometry_keys["llama.block_count"]
partial = local_models.describe(
    write_gguf(root / "partial-Q4_K_M.gguf", no_geometry_keys))
check(partial.get("geometry_missing") == ["n_layers"] and "n_layers" not in partial,
      "a model whose geometry is incomplete says which part is missing rather than guessing")

# --- scanning a folder ------------------------------------------------------

shelf = root / "shelf"
(shelf / "nested").mkdir(parents=True)
write_gguf(shelf / "one-Q4_K_M.gguf", dense_keys())
write_gguf(shelf / "nested" / "two-Q6_K.gguf", dense_keys())
(shelf / "notes.txt").write_text("not a model", encoding="utf-8")
(shelf / "broken-Q4_K_M.gguf").write_bytes(b"nope")
for index in (1, 2, 3):
    write_gguf(shelf / f"split-Q4_K_M-{index:05d}-of-00003.gguf", dense_keys())

models, problems = local_models.scan([shelf])
names = sorted(model["name"] for model in models)
check(names == ["one-Q4_K_M", "split-Q4_K_M-00001-of-00003", "two-Q6_K"],
      f"the scan finds nested weights and counts a split model once (got {names})")
check(any("broken-Q4_K_M.gguf" in problem for problem in problems),
      "a weight that cannot be read is reported with its reason, not dropped in silence")
check(not any("notes.txt" in problem for problem in problems),
      "a file that was never a model is not reported as a broken one")
check(local_models.scan([root / "does-not-exist"]) == ([], []),
      "a directory that is not there is not a failure")

# --- reusing what Ollama already downloaded ---------------------------------

store = root / "ollama" / "models"
blobs = store / "blobs"
blobs.mkdir(parents=True)
digest = "sha256:" + "ab" * 32
weights = write_gguf(blobs / digest.replace(":", "-"), dense_keys())
template_digest = "sha256:" + "cd" * 32
(blobs / template_digest.replace(":", "-")).write_text("{{ .Prompt }}", encoding="utf-8")


def write_manifest(namespace, name, tag, layers):
    path = store / "manifests" / "registry.ollama.ai" / namespace / name / tag
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schemaVersion": 2, "layers": layers}),
                    encoding="utf-8")
    return path


write_manifest("library", "qwen2.5-coder", "3b", [
    {"mediaType": local_models.OLLAMA_MODEL_MEDIA_TYPE, "digest": digest,
     "size": weights.stat().st_size},
    {"mediaType": "application/vnd.ollama.image.template", "digest": template_digest},
])
write_manifest("hf.co", "somebody/custom", "latest", [
    {"mediaType": local_models.OLLAMA_MODEL_MEDIA_TYPE, "digest": digest},
])
write_manifest("library", "collected", "latest", [
    {"mediaType": local_models.OLLAMA_MODEL_MEDIA_TYPE,
     "digest": "sha256:" + "ef" * 32},
])
write_manifest("library", "traversal", "latest", [
    {"mediaType": local_models.OLLAMA_MODEL_MEDIA_TYPE,
     "digest": "sha256:../../../../etc/passwd"},
])
write_manifest("library", "no-weights", "latest", [
    {"mediaType": "application/vnd.ollama.image.template", "digest": template_digest},
])
(store / "manifests" / "registry.ollama.ai" / "stray.json").write_text("{}", encoding="utf-8")

found, store_problems = local_models.ollama_models(root=store)
found_names = sorted(model["ollama_name"] for model in found)
check(found_names == ["hf.co/somebody/custom:latest", "qwen2.5-coder:3b"],
      f"models Ollama downloaded are found under the names it shows (got {found_names})")
check(all(model["origin"] == "ollama" for model in found),
      "a reused weight says it came from Ollama, so no screen can call it ours")
check(any("collected" in problem for problem in store_problems),
      "a manifest whose blob was collected is reported as missing weights")
# Named by its own reason, not merely by the model name. Dropping the digest
# bound still produced a problem line mentioning "traversal", because the path
# it built does not exist either, so a looser assertion passed while the guard
# was gone. Only "no usable digest" proves the digest was refused as a digest.
check(any("traversal" in problem and "no usable digest" in problem
          for problem in store_problems),
      "a digest that is not a digest is refused as a digest, before becoming a path")
check(any("no-weights" in problem for problem in store_problems),
      "a manifest with no model layer is reported rather than served")

before = sorted(path.name for path in blobs.iterdir())
link = local_models.link_ollama_model(found[0], target_dir=root / "linked")
check(link.is_symlink() and link.resolve() == weights.resolve(),
      "reuse is a link to the weights Ollama holds, not a second copy")
check(sorted(path.name for path in blobs.iterdir()) == before,
      "linking writes nothing at all into Ollama's store")
check(local_models.link_ollama_model(found[0], target_dir=root / "linked") == link,
      "linking the same model again is not an error")

listed, link_problems = local_models.linked_models(target_dir=root / "linked")
check(len(listed) == 1 and listed[0]["origin"] == "ollama" and not link_problems,
      "a link resolves back to a model the screens can describe")

weights.unlink()
listed, link_problems = local_models.linked_models(target_dir=root / "linked")
check(not listed and any("gone from Ollama's store" in problem
                         for problem in link_problems),
      "a model Ollama removed underneath is reported, not silently served")

# --- what a purge takes and what it deliberately leaves ---------------------

import installation

purge_home = root / "purge-home"
kept = purge_home / "models" / "downloaded"
kept.mkdir(parents=True)
write_gguf(kept / "expensive-Q4_K_M.gguf", dense_keys())
(kept / "notes.txt").write_text("ignore me", encoding="utf-8")

import io
from contextlib import redirect_stdout

reported = io.StringIO()
with redirect_stdout(reported):
    folder = installation._report_kept_weights(models_dir=kept)
shown = reported.getvalue()
check(folder == kept and str(kept) in shown,
      "a purge names the model weights it is leaving behind, with their folder")
check("1" in shown and (kept / "expensive-Q4_K_M.gguf").exists(),
      "the weights are still there afterwards, because they are a download nobody wants to repeat")

silent = io.StringIO()
with redirect_stdout(silent):
    empty = installation._report_kept_weights(models_dir=root / "no-weights-here")
check(empty is None and silent.getvalue() == "",
      "with no weights to keep, nothing is said about weights")

# The message must never turn a finished removal into a reported failure.
with redirect_stdout(io.StringIO()):
    unreadable = installation._report_kept_weights(models_dir=Path("/proc/1/root/nope"))
check(unreadable is None,
      "a folder that cannot be read ends the courtesy message, not the uninstall")

check(installation.local_models.linked_dir(home_dir=purge_home)
      != installation.local_models.downloaded_dir(home_dir=purge_home),
      "links and downloads live in separate directories, so one purge cannot take both")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("ISAAC LOCAL MODELS OK: gguf headers, disk discovery and Ollama reuse")
