#!/usr/bin/env python3
"""Tests for hardware detection and the fit/throughput arithmetic.

Nothing here runs nvidia-smi for real: the point is that the module behaves
the same on a machine with a GPU, without one, and with a broken nvidia-smi.
"""
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

import hardware


failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


def fake_run(stdout):
    def run(_argv, **_kwargs):
        return subprocess.CompletedProcess(_argv, 0, stdout=stdout, stderr="")
    return run


original_which = hardware.shutil.which
original_run = hardware.subprocess.run

try:
    # No nvidia-smi at all: the common laptop, and not an error.
    hardware.shutil.which = lambda _name: None
    detected = hardware.detect()
    check(set(detected) == {"gpus", "ram_mb", "cpu_cores"},
          "detect returns the same three fields with no GPU present")
    check(detected["gpus"] == [], "a machine without nvidia-smi reports no GPU")
    check(isinstance(detected["ram_mb"], int) and detected["ram_mb"] > 0,
          "RAM comes from /proc/meminfo as a positive integer of MB")
    check(isinstance(detected["cpu_cores"], int) and detected["cpu_cores"] > 0,
          "the core count is a positive integer")

    # nvidia-smi exists but fails: still a dict, still no exception.
    hardware.shutil.which = lambda _name: "/usr/bin/nvidia-smi"

    def explode(_argv, **_kwargs):
        raise subprocess.CalledProcessError(9, _argv)

    hardware.subprocess.run = explode
    broken = hardware.detect()
    check(broken["gpus"] == [] and set(broken) == {"gpus", "ram_mb", "cpu_cores"},
          "a failing nvidia-smi degrades to an empty GPU list, not an exception")

    hardware.subprocess.run = fake_run(
        "NVIDIA GeForce GTX 1650, 4096\nQuadro RTX 8000, 49152\n")
    listed = hardware.detect()["gpus"]
    check(len(listed) == 2 and listed[0]["name"] == "NVIDIA GeForce GTX 1650"
          and listed[0]["vram_mb"] == 4096,
          "a GPU row parses into its name and its VRAM in MB")
    check(listed[0]["bandwidth_gbs"] == 128.0,
          "a GPU in the table carries the table's bandwidth")
    check(listed[1]["bandwidth_gbs"] is None,
          "a GPU outside the table reports no bandwidth instead of a guess")

    hardware.subprocess.run = fake_run("Tesla T4, not-a-number\n")
    check(hardware.detect()["gpus"] == [],
          "an unparsable VRAM value is dropped rather than faked")
finally:
    hardware.shutil.which = original_which
    hardware.subprocess.run = original_run

check(hardware.gpu_bandwidth("tesla p100-pcie-16gb") == 732.0
      and hardware.gpu_bandwidth("Tesla T4") == 300.0,
      "table matching is by substring and case-insensitive")
check(hardware.gpu_bandwidth("NVIDIA RTX 4090") is None
      and hardware.gpu_bandwidth("") is None
      and hardware.gpu_bandwidth(None) is None,
      "an unknown, empty or missing GPU name gives None")

# Qwen3-30B-A3B's real shape: 48 layers, 32 query heads against 4 KV heads,
# head_dim 128. Using n_heads here would overstate the cache by 8x.
gqa = hardware.kv_cache_bytes(48, 4, 128, 4096)
mha = hardware.kv_cache_bytes(48, 32, 128, 4096)
check(gqa == 2 * 48 * 4 * 128 * 4096 * 2 == 402653184,
      "kv_cache_bytes follows the formula exactly (384 MiB for this shape)")
check(mha == gqa * 8,
      "the same shape with 32 KV heads is exactly 8x larger, which is the GQA trap")
check(hardware.kv_cache_bytes(48, 4, 128, 8192) == gqa * 2,
      "doubling the context doubles the cache")
check(hardware.kv_cache_bytes(48, 4, 128, 4096, bytes_per_element=1) == gqa // 2,
      "a one-byte KV element halves the cache")
q8 = hardware.kv_cache_bytes(
    48, 4, 128, 4096, bytes_per_element=Fraction(34, 32))
check(q8 == 213909504,
      f"q8_0 charges 34 bytes per 32 elements without early truncation ({q8})")

GB = 1024 ** 3
# 4 GB card, 768 MB overhead: 3328 MB usable.
check(hardware.fits(3 * GB, 200 * 1024 * 1024, 4096) is True,
      "3 GB of weights plus 200 MB of cache fit a 4 GB card")
check(hardware.fits(3 * GB, 700 * 1024 * 1024, 4096) is False,
      "the same weights with a larger cache no longer fit")
check(hardware.fits(3 * GB, 700 * 1024 * 1024, 4096, overhead_mb=0) is True,
      "the overhead is what decides that borderline case, and it is a parameter")
check(hardware.fits(1, 0, 100) is False,
      "a card smaller than the overhead fits nothing instead of going negative")

q8_context = hardware.max_context_that_fits(
    2_497_281_120, 36, 8, 128, 4096,
    bytes_per_element=Fraction(34, 32))
check(q8_context == 12668,
      f"the exact q8_0 fraction is kept through the final context floor ({q8_context})")

empty = hardware.summarise([])
check(empty == {"vram_mb": 0, "gpu_count": 0, "bandwidth_gbs": None, "name": None},
      "no card summarises to zero VRAM, zero cards and no bandwidth")
check(hardware.summarise(None) == empty,
      "a missing GPU list is the same machine as an empty one")
one = hardware.summarise(
    [{"name": "GTX 1650", "vram_mb": 4096, "bandwidth_gbs": 128.0}])
check(one == {"vram_mb": 4096, "gpu_count": 1, "bandwidth_gbs": 128.0,
              "name": "GTX 1650"},
      "a single card lends its VRAM, its name and its bandwidth")
pair = hardware.summarise([{"name": "Tesla T4", "vram_mb": 15360,
                            "bandwidth_gbs": 300.0}] * 2)
check(pair["vram_mb"] == 30720 and pair["gpu_count"] == 2
      and pair["bandwidth_gbs"] is None,
      "two cards add their VRAM but never their buses: one decode reads one")
# The three treatments that used to disagree, now one. A card whose VRAM the
# driver did not report still counts as a card, because the count is what
# decides the reserve; what is not a card at all is dropped.
check(hardware.summarise([{"name": "a"}, {"vram_mb": None},
                          {"vram_mb": "junk"}])
      == {"vram_mb": 0, "gpu_count": 3, "bandwidth_gbs": None, "name": "a"},
      "an unreadable VRAM field is zero VRAM, not a card that stopped existing")
check(hardware.summarise([{"vram_mb": 4096.0}, "not a card", None])["vram_mb"]
      == 4096
      and hardware.summarise([{"vram_mb": 4096.0}, "not a card"])["gpu_count"] == 1,
      "a float VRAM is an integer of MB, and a non-card is not counted")

check(hardware.overhead_mb(1) == hardware.DEFAULT_OVERHEAD_MB
      and hardware.overhead_mb(2) == 2 * hardware.DEFAULT_OVERHEAD_MB,
      "the reserve multiplies by the number of cards, because each card has one")
check(hardware.overhead_mb(0) == hardware.overhead_mb()
      == hardware.overhead_mb(None) == hardware.DEFAULT_OVERHEAD_MB,
      "no card, no count and no argument all keep one card's reserve")
check(hardware.fits(3 * GB, 700 * 1024 * 1024, 4096,
                    overhead_mb=hardware.overhead_mb(0)) is False,
      "a machine with no GPU answers no, instead of counting VRAM it lacks")

# The rule above used to be written as `DEFAULT_OVERHEAD_MB * max(1, gpu_count)`
# in six places across two interface modules, which is a hardware rule living
# where hardware rules are not read. This refuses the seventh.
handwritten = []
summed = []
for source in sorted((HERE.parent / "tool_harness").glob("*.py")):
    if source.name == "hardware.py":
        continue
    body = source.read_text(encoding="utf-8")
    for number, line in enumerate(body.splitlines(), 1):
        if "DEFAULT_OVERHEAD_MB" in line and "*" in line.split("DEFAULT_OVERHEAD_MB")[1]:
            handwritten.append(f"{source.name}:{number}")
    # Across lines and across the parentheses of the `.get()` inside, because
    # the version this replaced was a four-line generator expression and both
    # a one-line pattern and a paren-free one would have missed it.
    for match in re.finditer(r"sum\(.{0,200}?vram_mb", body, re.S):
        summed.append(f"{source.name}:{body.count(chr(10), 0, match.start()) + 1}")
check(not handwritten,
      "no module multiplies the per-card reserve by hand: " + (
          ", ".join(handwritten) or "none does"))
check(not summed,
      "no module adds a machine's VRAM up by hand: " + (
          ", ".join(summed) or "none does"))

check(hardware.bytes_read_per_token(16 * GB) == float(16 * GB),
      "a dense model reads its whole file per token")
check(hardware.bytes_read_per_token(16 * GB, active_ratio=0.5) == float(8 * GB),
      "an active ratio scales the bytes read")

dense = hardware.estimate_tokens_per_second(
    hardware.bytes_read_per_token(4 * GB), 128.0)
check(abs(dense - (128.0 * 1e9 * hardware.DEFAULT_MBU) / float(4 * GB)) < 1e-9,
      "the dense estimate follows bandwidth times utilisation over bytes per token")
check(hardware.estimate_tokens_per_second(float(4 * GB), 128.0,
                                          mbu=hardware.DEFAULT_MBU / 2)
      == dense / 2,
      "halving the utilisation halves the estimate")

# The utilisation constant is not a guess: it reproduces the one run that was
# actually measured on this hardware. A regression here means someone changed
# the constant without a new measurement behind it.
measured_file_bytes = 2_104_932_800   # Qwen2.5-Coder-3B-Instruct Q4_K_M
measured_tokens_per_second = 36.2     # GTX 1650, llama.cpp b10502, Vulkan
check(abs(hardware.estimate_tokens_per_second(
          hardware.bytes_read_per_token(measured_file_bytes), 128.0)
          - measured_tokens_per_second) < 0.15,
      "the default utilisation reproduces the measured run on the reference card")

# MoE: same file, only the active experts cross the bus.
model_bytes = 16.8 * GB
moe = hardware.estimate_tokens_per_second(
    hardware.bytes_read_per_token(model_bytes, active_ratio=0.11), 300.0)
full = hardware.estimate_tokens_per_second(
    hardware.bytes_read_per_token(model_bytes, active_ratio=1.0), 300.0)
check(moe is not None and full is not None and abs(moe / full - 9.0909) < 0.01,
      "an active ratio of 0.11 estimates about 9x the throughput of the dense read")

check(hardware.estimate_tokens_per_second(float(4 * GB), None) is None,
      "an unknown bandwidth yields no throughput number at all")
check(hardware.estimate_tokens_per_second(float(4 * GB), 0.0) == 0.0,
      "a zero bandwidth is a number, not the unknown case")
check(hardware.estimate_tokens_per_second(0, 128.0) is None,
      "zero bytes per token gives None instead of dividing by zero")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("ISAAC HARDWARE OK: detection degrades, unknown bandwidth stays unknown")
