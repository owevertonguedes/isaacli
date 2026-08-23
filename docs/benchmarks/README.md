# What was actually measured, and on what

Every score isaacli used to show came from somebody else: measured on other
hardware, at full precision, inside another agent harness. That answers *where
the number came from*. It never answered *why this model is on this list*.

This folder is the other half. Each report here is one artifact, measured on
one machine, on a date, by `scripts/bench_local.py`. Nothing in a report is
estimated: a number that could not be measured is absent rather than guessed.

## The reports

| report | artifact | HumanEval (20 of 164) | tok/s | native tool call |
| --- | --- | --- | --- | --- |
| [granite-4.1-3b Q4_K_M](2026-08-23-granite-4-1-3b-q4-k-m.md) | `granite-4.1-3b-Q4_K_M.gguf` | 17/20 | 33.0 | yes |
| [qwen2.5-coder-3b-instruct Q4_K_M](2026-08-23-qwen2-5-coder-3b-instruct-q4-k-m.md) | `qwen2.5-coder-3b-instruct-q4_k_m.gguf` | 18/20 | 36.9 | no |
| [Phi-4-mini-instruct Q4_K_M](2026-08-23-phi-4-mini-instruct-q4-k-m.md) | `Phi-4-mini-instruct-Q4_K_M.gguf` | 15/20 | 29.4 | no |

All three on the same machine, an NVIDIA GeForce GTX 1650 with 4096 MiB of
VRAM, through llama.cpp Vulkan build 10502, on 2026-08-23.

**These numbers describe those files on that machine.** They are not a ranking
of models, they do not carry to another quantization of the same weights, and
they say nothing about how any of this behaves on a larger card. A newer or
better model may simply not run here, which is a fact about the card and not
about the model.

## Reading the table

The HumanEval column and the tool-call column disagree, and the disagreement is
the most useful thing in the table. `qwen2.5-coder-3b` writes the best Python
of the three and **does no work in isaacli**: asked to create a file, it prints
a JSON tool call inside a markdown fence instead of emitting the `<tool_call>`
its own chat template defines, so the server never converts it and the harness
never sees a call. `granite-4.1-3b` writes slightly worse Python and actually
drives the tools. For an agent, the second column decides.

That is also why HumanEval alone is not enough to justify a recommendation, and
why the tool-call case is run through the real `agent.py` loop with the real
tool schema, judged by the bytes that land on disk rather than by the model
claiming it wrote the file.

## Recommended models that were not measured here, because they do not fit

The three remaining entries in the local catalogue cannot run on this card, by
the program's own arithmetic (`model_discovery.fit_report`, 16384 context,
3.25 GiB usable after the runtime's own buffers):

| model | weights | KV cache | verdict |
| --- | --- | --- | --- |
| `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ1_M` | 9.36 GiB | 1.25 GiB | does not fit |
| `unsloth/North-Mini-Code-1.0-GGUF:IQ3_S` | 11.89 GiB | 1.53 GiB | does not fit |
| `gpt-oss:20b` (MXFP4) | 11.28 GiB | | does not fit |

They stay on the list because the list is not only for this machine, and the
discovery screen already tells each user what fits theirs. What they do not
have is a measurement, and no row claims one.

`Phi-4-mini-instruct-Q4_K_M` fits but not at 16384: llama.cpp failed the KV
allocation with `ErrorOutOfDeviceMemory`, so it was served at 8192 and its
report says so. HumanEval prompts are a few hundred tokens, so the shorter
window does not touch its score, but it is the kind of difference that has
falsified measurements in this project before and it is recorded rather than
smoothed over.

## Reproducing a report

Start any OpenAI-compatible server on the model, then:

```bash
scripts/bench_local.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model <the name the server answers to> \
  --artifact /path/to/the.gguf \
  --backend "llama.cpp Vulkan build 10502 (0adcc3bb5)" \
  --context 16384 --cases 20
```

It writes a markdown report and the raw JSON beside it, both named for the date
and the model.

## What the rulers are, and what they are not

- **HumanEval** comes from `openai/human-eval` as the released
  `HumanEval.jsonl.gz`, and every report records the SHA-256 of the exact bytes
  it graded against. The graded problems are a fixed stride over the 164 sorted
  task ids, so every model meets the same twenty. The judge is the dataset's
  own `check()`, run inside the project's sandbox and never on the host.
- **Throughput** is the server's own `predicted_per_second`, reported as the
  median over the twenty generations. It excludes this client and the socket.
- **The tool-call case** is ours, and the reports say so. It is one request
  through `agent.py`, judged by the bytes on disk.

Neither ruler is SWE-bench. SWE-bench needs a container per instance and hours
per model, and a battery nobody re-runs is a battery that goes stale. What this
measures is narrower and repeatable, and it is dated.

## Why the public leaderboards are not scraped instead

Investigated on 2026-08-23, and the short answer is that a SWE-bench number is
not a property of a model. In `swe-bench/experiments`, Devstral Small 2507 with
the SWE-agent scaffold resolves 190 of 500 instances (38.0%), while Mistral's
own model card reports 53.6% for the same weights under OpenHands, and the
leaderboard's Devstral Small 2505 entry at 234/500 (46.8%) matches that card's
own OpenHands figure exactly. Same weights, 15.6 points apart, decided by the
harness. Qwen3-Coder-30B-A3B-Instruct appears twice under one agent, at 261/500
and 302/500.

So every score in `model_catalog.json` now names the harness that produced it,
and a check refuses any SWE-bench number that does not.
