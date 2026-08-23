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
    """
    return int(2 * n_layers * n_kv_heads * head_dim * context * bytes_per_element)


def fits(model_bytes, kv_bytes, vram_mb, overhead_mb=DEFAULT_OVERHEAD_MB):
    """Whether weights, cache and the runtime's own buffers fit in VRAM."""
    return (model_bytes + kv_bytes) <= max(0, vram_mb - overhead_mb) * 1024 * 1024


def max_context_that_fits(model_bytes, n_layers, n_kv_heads, head_dim, vram_mb,
                          overhead_mb=DEFAULT_OVERHEAD_MB, bytes_per_element=2):
    """The largest context whose cache still fits beside the weights, or 0.

    `fits` answers yes or no about a context somebody already chose. This
    inverts it, which is what a program needs to decide the context itself
    instead of inheriting whatever number a hand-written launch script froze
    into place.

    The cache grows linearly in context, so this is division, not a search.
    """
    free_bytes = max(0, vram_mb - overhead_mb) * 1024 * 1024 - model_bytes
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
