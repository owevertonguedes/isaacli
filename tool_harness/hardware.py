"""Local hardware facts and the arithmetic that turns them into a fit/speed estimate.

This module only reports and computes. It never formats anything a user reads,
and it never raises: a machine with no GPU is a normal machine, not an error,
so absent hardware becomes an empty field and the caller decides what to say.

The speed side of the arithmetic exists because batch-1 decode is bound by
memory bandwidth, not by compute: every token requires reading the weights
once. That makes tokens per second predictable from bytes read and bandwidth,
without installing the model first.
"""
import os
import shutil
import subprocess

import debug


# nvidia-smi reports the name and the total memory but not the memory bus
# width, so bandwidth cannot be derived from it. A guessed number would read
# exactly like a measured one to the user, so a GPU outside this table gets
# None and the caller shows fit only. Values below are the vendor's published
# figures for these three parts.
GPU_BANDWIDTH = {
    "Tesla P100": 732.0,
    "Tesla T4": 300.0,
    "GTX 1650": 128.0,
}

# CUDA context plus compute buffers sit next to the weights and the KV cache.
# The reserve for one card; `overhead_mb` is what a caller with a machine asks.
DEFAULT_OVERHEAD_MB = 768

# Fraction of theoretical bandwidth a real decode loop achieves. The one
# empirical constant in the whole calculation, so it is measured, not guessed.
#
# Calibrated 2026-08-20 against a single real run: Qwen2.5-Coder-3B-Instruct
# Q4_K_M (2,104,932,800 bytes) fully offloaded to a GTX 1650 (128 GB/s) through
# llama.cpp b10502 on the Vulkan backend generated 36.2 tok/s, which puts the
# utilisation at 0.595.
#
# One point on one backend and one card. It is a better default than a guess,
# and it is still not enough to put a per-model number in front of a user on
# hardware nobody has measured. Widen this with more measurements before the
# onboarding starts quoting throughput.
DEFAULT_MBU = 0.595

# llama-server serves this many requests at once unless told otherwise, and the
# recurrent state of a hybrid model is allocated once per slot. Measured, not
# assumed: the 2026-09-08 Kaggle run of Qwen3.8-27B logged
# `llama_kv_cache: size = 3072.00 MiB ( 49152 cells, 16 layers, 4/1 seqs)` from
# a server started with no `-np`. The KV cache itself is not multiplied by it,
# because its cells are the context and the slots divide them.
DEFAULT_PARALLEL_SEQUENCES = 4


def overhead_mb(gpu_count=1):
    """The runtime's own reserve on a machine with this many cards.

    Every card carries its own context and its own compute buffers, so the
    reserve multiplies by the number of cards and not by anything else. This
    is a fact about hardware, and it used to be written as
    `DEFAULT_OVERHEAD_MB * max(1, gpu_count)` in six places across two
    interface modules, which meant changing it required remembering all six.

    The floor of one card is not defensive arithmetic. A machine with no GPU
    has no VRAM to reserve out of, and `vram_mb - 0` on such a machine reads
    as free memory that does not exist; keeping one card's reserve makes the
    fit answer no, which is the true answer.
    """
    return DEFAULT_OVERHEAD_MB * max(1, int(gpu_count or 0))


def _query_nvidia_smi():
    """Raw `name, memory.total` rows, or an empty list when there is no GPU."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    result = subprocess.run(
        [executable, "--query-gpu=name,memory.total",
         "--format=csv,noheader,nounits"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=10, check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def gpu_bandwidth(name):
    """GB/s for a known GPU, None for anything else. Never a guess."""
    lowered = str(name or "").lower()
    for known, bandwidth in GPU_BANDWIDTH.items():
        if known.lower() in lowered:
            return bandwidth
    return None


def gpus():
    """Every NVIDIA GPU visible, each with its VRAM and its bandwidth if known."""
    found = []
    try:
        rows = _query_nvidia_smi()
    except Exception:
        debug.swallowed("hardware.detect gpu")
        return found
    for row in rows:
        parts = row.split(",")
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        try:
            vram_mb = int(float(parts[1].strip()))
        except ValueError:
            debug.swallowed("hardware.detect gpu")
            continue
        found.append({
            "name": name,
            "vram_mb": vram_mb,
            "bandwidth_gbs": gpu_bandwidth(name),
        })
    return found


def summarise(gpus):
    """One machine's worth of cards, reduced to what a screen has to know.

    Four callers were doing this reduction by hand, with three different
    treatments of a card whose VRAM field is missing, so a list that read as
    "4 GB, one card" in one screen could read as "0 GB, one card" in another.
    There is one treatment here: an entry that is not a card at all is
    dropped, and a card whose VRAM cannot be read is still a card, with zero
    VRAM, because the count is what decides the reserve.

    `bandwidth_gbs` is filled only when there is exactly one card. Two cards
    do not add their buses together for a single decode loop, which reads one
    layer at a time from whichever card holds it, so summing them would
    inflate every throughput estimate drawn against this machine.
    """
    cards = [item for item in (gpus or []) if isinstance(item, dict)]
    total_mb = 0
    for card in cards:
        try:
            total_mb += int(float(card.get("vram_mb") or 0))
        except (TypeError, ValueError):
            debug.swallowed("hardware.summarise vram")
    return {
        "vram_mb": total_mb,
        "gpu_count": len(cards),
        "bandwidth_gbs": cards[0].get("bandwidth_gbs") if len(cards) == 1 else None,
        "name": cards[0].get("name") if cards else None,
    }


def ram_mb():
    """System RAM in MB, 0 when /proc/meminfo is unreadable."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemTotal:"):
                    # MemTotal is reported in kB.
                    return int(line.split()[1]) // 1024
    except Exception:
        debug.swallowed("hardware.detect ram")
    return 0


def cpu_cores():
    return os.cpu_count() or 0


def detect():
    """Everything the onboarding needs to know about this machine."""
    return {
        "gpus": gpus(),
        "ram_mb": ram_mb(),
        "cpu_cores": cpu_cores(),
    }


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, context, bytes_per_element=2):
    """Size of the KV cache at a given context length.

    The leading 2 is key plus value. `n_kv_heads` is deliberately not
    `n_heads`: with grouped-query attention the two differ by up to 8x, and
    using the query count overestimates the cache by that same factor.

    `n_layers` is the number of layers that actually hold a cache, which on a
    dense model is every layer and on a hybrid model is a fraction of them.
    Getting that wrong is the largest error this file ever made: billing all 64
    layers of Qwen3.8-27B, when 16 of them cache and 48 are recurrent,
    overstated the whole footprint by 45% at a 49152 context. Callers get the
    caching count from the model's own declaration; see `_geometry` in
    model_discovery.py and `_attention_layers` in gguf.py.
    """
    return int(2 * n_layers * n_kv_heads * head_dim * context * bytes_per_element)


def recurrent_state_bytes(n_recurrent_layers, key_heads, value_heads,
                          key_head_dim, value_head_dim, conv_kernel,
                          n_seqs=DEFAULT_PARALLEL_SEQUENCES,
                          bytes_per_element=4):
    """The fixed state the non-attention layers of a hybrid model hold.

    This is the other half of the same correction as `kv_cache_bytes`. Dropping
    48 layers out of the cache term without adding back what those layers do
    keep would trade a 45% overestimate for an underestimate, and only one of
    those two errors costs somebody a profile that will not load.

    It does not scale with context, which is the whole point of a recurrent
    layer: the state is a fixed-size matrix per head plus a short convolution
    window, no matter how long the conversation gets. It does scale with the
    number of slots the server runs.

    Checked against the numbers llama.cpp declared for Qwen3.8-27B on 2026-09-08
    (48 recurrent layers, 16 key heads, 48 value heads, 128/128 head dims,
    convolution kernel 4, four slots):

        llama_memory_recurrent: CUDA0 RS buffer size = 311.72 MiB
        llama_memory_recurrent: CUDA1 RS buffer size = 286.78 MiB

    which is 598.50 MiB across the two cards, and this returns 598.5 MiB.
    """
    per_layer_per_seq = (
        value_heads * key_head_dim * value_head_dim
        + max(0, conv_kernel - 1) * (key_heads * key_head_dim * 2
                                     + value_heads * value_head_dim)
    )
    return int(n_recurrent_layers * max(1, n_seqs)
               * per_layer_per_seq * bytes_per_element)


def fits(model_bytes, kv_bytes, vram_mb, overhead_mb=DEFAULT_OVERHEAD_MB,
         fixed_bytes=0):
    """Whether weights, cache and the runtime's own buffers fit in VRAM.

    `fixed_bytes` is anything resident that does not grow with the context: the
    recurrent state of a hybrid model, and nothing else so far. It is separate
    from `model_bytes` because it is not part of the weight file, and separate
    from `kv_bytes` because inverting the fit for a context has to divide by a
    per-token cost and this one has none.
    """
    return (model_bytes + kv_bytes + fixed_bytes) <= max(
        0, vram_mb - overhead_mb) * 1024 * 1024


def max_context_that_fits(model_bytes, n_layers, n_kv_heads, head_dim, vram_mb,
                          overhead_mb=DEFAULT_OVERHEAD_MB, bytes_per_element=2,
                          fixed_bytes=0):
    """The largest context whose cache still fits beside the weights, or 0.

    `fits` answers yes or no about a context somebody already chose. This
    inverts it, which is what a program needs to decide the context itself
    instead of inheriting whatever number a hand-written launch script froze
    into place.

    The cache grows linearly in context, so this is division, not a search.
    """
    free_bytes = (max(0, vram_mb - overhead_mb) * 1024 * 1024
                  - model_bytes - fixed_bytes)
    per_token = 2 * n_layers * n_kv_heads * head_dim * bytes_per_element
    if free_bytes <= 0 or per_token <= 0:
        return 0
    return int(free_bytes // per_token)


def bytes_read_per_token(model_bytes, active_ratio=1.0):
    """Weight bytes crossing the memory bus for one token.

    A mixture-of-experts model reads only its active experts, so `active_ratio`
    is active parameters over total parameters. It buys bandwidth per token,
    never memory: the whole file still has to be resident somewhere, which is
    why it belongs here and not in `fits`.
    """
    return float(model_bytes) * float(active_ratio)


def estimate_tokens_per_second(bytes_per_token, bandwidth_gbs, mbu=DEFAULT_MBU):
    """Decode throughput, or None when the bandwidth of the part is unknown.

    This estimates generation only. Prefill is compute-bound and follows a
    different calculation; mixing the two produces a number the user never
    feels.
    """
    if bandwidth_gbs is None:
        return None
    if not bytes_per_token:
        return None
    return (float(bandwidth_gbs) * 1e9 * float(mbu)) / float(bytes_per_token)
