# local-llm-field-notes

**Field notes, with numbers, from an experiment on running code agents on a local
LLM — on deliberately weak hardware.**

*[Leia em português](README.pt-BR.md).*

Hardware: a laptop with a **GTX 1650 (4 GB VRAM)** and 15 GB of RAM. Not an AI
machine, and that was the point — find out what you can actually do with what you
have, not with what would be ideal.

> **This is an experiment, not a product.** It is not maintained and was not built
> for production. The value here is in the **measurements** and the **documented
> methodology mistakes** — treat it as a reference and copy whatever parts help.
> [AGPLv3](LICENSE): free to use, study and modify. Offering it as a closed
> network service requires a commercial license ([details](LICENSING.md)).

---

## What this repository is for

The question under test was: *can a small LLM, running locally, become a reliable
code agent without depending on managed cloud?*

The full answer is below, but the practical use of this repo doesn't depend on it.
It's useful for **four concrete things**, and at least one probably applies to you:

### 1. Deciding whether local is worth it, without burning a weekend finding out

The VRAM, speed, and quality numbers are already measured here. If you're on
similar hardware and considering this setup, the
[technical findings](#technical-findings-that-cost-real-time) answer in five
minutes what took days. Especially the VRAM cliff, and the fact that **MoE saves
compute, not memory** — which is where most blog posts get it wrong.

### 2. A checklist of measurement bugs, applicable to any LLM evaluation

This is the most transferable part, and it **applies equally to cloud models**.
These are six failure modes where the *ruler* lies, not the model — and every one
of them produces exactly the same `ERR` on screen that a genuine failure would.

If you maintain an eval, an LLM-as-judge, or an internal benchmark, run
[the list](#the-most-expensive-lesson-the-ruler-lies-before-the-model-does)
against it. It's cheap, and the odds of finding something are high.

### 3. A ready-made sandbox for executing agent code you don't trust

[`tool_harness/tools.py`](tool_harness/tools.py) and
[`isaac_cli.py`](tool_harness/isaac_cli.py) implement containment in three
independent layers: **direct execve** (no shell, so there's no injection via `;`
or `$()`), a **short allowlist** of binaries, and **`bwrap`** with the entire disk
read-only, networking closed, and only the working directory writable. It was
tested with bait planted outside the directory, to see whether it would escape.

This is reusable in any project that executes model-generated code — local or
cloud — and it's the part of this repo that ages the slowest.

### 4. A template for an evaluation harness that doesn't fool itself

[`bancada/`](bancada/) (*"the bench"*) holds coding problems with `assert`s that
**actually execute**, plus — the part that matters — a validator that checks two
things per problem: that the reference solution passes, **and that the naive
solution is rejected**. A ruler that approves the naive solution is measuring
nothing, and you only discover that by testing the ruler itself.

The same idea applies to
[`validar_datasets_lora.py`](tool_harness/validar_datasets_lora.py), which blocks
training examples that call nonexistent tools or claim success without evidence —
before they enter training and get baked in as hallucination.

---

## The answer the experiment produced

**Yes for specialization and reliability. No for raw capability.**

At 4 GB of VRAM, the model is the piece that won't move: a 3B doesn't become a
frontier model, and that's a limit of scale, not of effort. A good deal of the
work here ended up being infrastructure built around a stuck part.

But the split that emerged is the most useful framing of the whole project:

> **Raw intelligence isn't something you add.** It comes from pretraining, costs
> millions, and you download it ready-made. **Specialization and reliability are
> things you do add** — and that's demonstrated below, with numbers.

Judging a project like this by the first column guarantees frustration. By the
second, it delivers.

---

## What worked (measured)

### Teaching tool-calling to a model that couldn't do it: 0/8 → 6/8

`Qwen2.5-Coder-3B` looked simply incapable of calling tools. It wasn't.

`<tool_call>` is the **single token `151657`**. The component that picks output
tokens is the `lm_head` — and it was outside the LoRA's `target_modules`. The
model was being trained on everything *except* the layer that decides to emit the
tag. With `modules_to_save=["lm_head"]`, loss dropped to ~1e-06 and the tag
appeared.

**If you're training tool-calling with LoRA, that one detail may save you days.**
And the lesson generalizes: distinguish *"the model doesn't know"* from *"the
serving layer doesn't understand"*. Dig to the root cause before writing off a
model.

Code in [`qwen_tools_lora/`](qwen_tools_lora/).

### Distillation with a mechanical gate: 22 requests → 16 approved

A larger model generates examples, and a gate that **executes the code** decides
whether an example enters the dataset. A rejected example is **discarded, never
hand-fixed** — an error you repair becomes hallucination in the weights, because
you've taught the model to produce something it wouldn't produce on its own.

A ~73% approval rate is a useful planning number: expect to lose about a third of
whatever the teacher generates.

### Scaffolding beats a bigger model: `pass@1` 40% vs `pass@8` 75%

That gap is capability **already present in the weights** that doesn't surface in
a single attempt. The ceiling is fixed *per single pass*; the ceiling of the
**system** (model + retries + verifier + decomposition into steps) is not.

The practical consequence, and maybe the most actionable thing here: **when
hardware is the ceiling, investing in scaffolding pays more than swapping
models.**

---

## What didn't work

Worth as much as the rest — it saves you from repeating it:

- **No trained adapter was ever approved.** The best one took one workflow from
  1/5 to 5/5 but left another at 4/6, still emitting nonexistent tool names.
  Nothing was merged into the base weights. Reports for every run are in
  [`reports/lora/`](reports/lora/), **including the failed ones** — which is where
  the honest information lives.
- **Piling rules into the system prompt does not teach a small model.** Measured:
  zero improvement. What changes small-model behavior is LoRA and external
  scaffolding, not a longer instruction.
- **"Asking the model to review itself" doesn't raise capability.** `pass@1` went
  from 40% to 65%, but `pass@8` went from 75% to **77%**. It improves reliability,
  not the ceiling — still useful, as long as you don't confuse the two.
- **Letting the model "learn on its own from the internet"** was never
  implemented, on purpose: model collapse and prompt injection. The version that
  works is a trusted teacher plus an automatic quality gate before data gets in.

---

## The most expensive lesson: the ruler lies before the model does

In a single day, **six measurement bugs** surfaced. Five made results look
*worse* than reality. The sixth made them look *better*.

1. `skip_special_tokens=True` stripped `<tool_call>` from the string being
   evaluated. The model was right; the ruler couldn't see it.
2. A dataset teaching tools that didn't exist → trains hallucination into weights.
3. Output budget consumed by reasoning tokens → empty response with **HTTP 200**.
   No error on screen, no content either.
4. One Ollama response path returned reasoning in a **separate field**
   (`thinking`), not in `<think>`. The cleanup regex never had anything to clean.
5. A wrong reference answer in the bench: `match_wildcard('acdcb', 'a*c?b')` was
   marked `True`; the correct answer is `False`. **A wrong assert fails a correct
   model.**
6. And the worst: the LoRA evaluation matched **substrings against generated
   text** instead of running the agent and checking final state. It inflated the
   scoreboard — and a lax judge teaches the model to game the judge.

> **The rule that stuck: when a result looks bad, suspect the ruler before the
> model.** `the model was wrong` and `I measured wrong` produce the identical
> `ERR` on screen. Dump the raw output before reporting any verdict.

If you take one thing from this repo, take that. It has nothing to do with local
models — it applies to any LLM evaluation you run.

---

## Technical findings that cost real time

- **A Granite 4/Ollama endpoint mismatch was observed during the experiment, but
  did not reproduce later.** At the time, native `/api/chat` returned tool calls
  while OpenAI-compatible `/v1/chat/completions` returned `content=""` with no
  `tool_calls` and no error. Retesting on 2026-07-27 with Ollama `0.30.10` and
  the local `granite4:micro` returned tool calls on both endpoints. Treat this as
  a diagnostic lesson — dump raw responses and record versions — not as a claim
  of a current Ollama bug.
- **Past ~4 GB of VRAM, performance doesn't degrade — it falls off a cliff.** A
  6.6 GB model ran at **3.5 tok/s** on this machine, i.e. unusable. Plan around
  what **fits**, not around what's good.
- **MoE saves compute, not memory.** All weights stay resident. A model a blog
  advertised as "fits in 16 GB" measured 24 GB in reality. Check file size at the
  source, not in the post.
- **Efficient architecture beats a bigger model when hardware is the ceiling.** A
  well-designed 2.1 GB model (hybrid Mamba-2/Transformer) beat a 6.6 GB one.
  Before reaching for "a bigger model", ask whether there's a *better-designed
  model of the same size*.
- **Reasoning distillation doesn't transfer across domains.** An 8B model
  distilled for reasoning scored **1/4** on a coding bench — worse than a 3B.
- **Beware of capability that shipped from the factory.** One model tested already
  called tools natively (`ollama show` → `capabilities: tools`). The wrapper only
  *connected* it; it taught nothing. **Don't mistake integration for learning** —
  it's easy to convince yourself you taught something that was already there.

---

## Repository map

| directory | contents |
|---|---|
| [`bancada/`](bancada/) | *"the bench"* — coding problems with executing `assert`s, and the validator that tests the reference answers themselves |
| [`tool_harness/`](tool_harness/) | the agent: CLI, tools, three-layer sandbox, dataset validator, and tests for each piece |
| [`qwen_tools_lora/`](qwen_tools_lora/) | the tool-calling training run that worked (0/8 → 6/8), including the `lm_head` fix |
| [`datasets/`](datasets/) | 30 curated and validated examples, with READMEs explaining the criteria for each set |
| [`finetune_test/`](finetune_test/) | LoRA scripts and the before/after comparison |
| [`reports/`](reports/) | raw measurements from every run, passed and failed alike |

A note on scale: the bench has 25 problems and was built for one specific
question. **It is not a serious benchmark — don't compare models publicly with
it.** What's worth copying is the method, not the scoreboard.

### A note on language

Code, filenames, and inline comments are in Portuguese; the documentation is in
English. Renaming everything risked breaking working, tested code for no real
gain, so here's the glossary instead:

| term | meaning |
|---|---|
| `bancada` | the bench (evaluation harness) |
| `ferramenta` / `tools` | tool (as in tool-calling) |
| `juiz` | judge / verifier |
| `validar`, `testar`, `checar` | validate, test, check |
| `bancada/teto.py` | `teto` = ceiling |
| `aprender`, `ensinar` | learn, teach |
| `curadoria`, `curar` | curation, to curate |
| `andaime` | scaffolding |
| `regua` | ruler (as in measuring stick) |
| `isaac` | the name given to the local agent |

---

## If you want a working tool, not a set of notes

These are maintained and tested, and will get you further faster than rebuilding
what's here:

| tool | for what |
|---|---|
| [Ollama](https://ollama.com) | running the models. Lightweight daemon; it's the base this project uses |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | more control over quantization and layer offload; better when VRAM is the bottleneck |
| [Aider](https://aider.chat) | code agent in the terminal, with Git integration. Mature |
| [OpenCode](https://github.com/sst/opencode) / [Continue](https://continue.dev) | agent in the terminal and the editor, with local model support |
| [LM Studio](https://lmstudio.ai) | if you prefer a GUI over a terminal |
| [Unsloth](https://github.com/unslothai/unsloth) | training LoRA on a small GPU, far more efficient than the path taken here |
| [PEFT](https://github.com/huggingface/peft) + [TRL](https://github.com/huggingface/trl) | the standard fine-tuning route, if you want to assemble it yourself |

They solve *running*. This repository is about *measuring* — the two complement
each other well.

---

## Contributing

**Use, study and adapt it freely**, copy pieces, build on top. [AGPLv3](LICENSE):
derivatives stay open under the same license, including when served over a
network. To offer this as a closed service, a commercial license is available:
see [LICENSING.md](LICENSING.md).

Suggestions, corrections, and issues are welcome — especially if you find another
measurement bug that slipped through, since that's the kind of error you only
catch with a second pair of eyes.

Just know up front: **this isn't maintained**, so replies may be slow or may not
come, and parts of the code were written to answer one question and left as they
were. Expect loose ends. If your idea is good, it probably deserves its own
repository rather than a PR here.
