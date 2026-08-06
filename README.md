# isaacli

**A local-first CLI coding agent that actually executes, on 4 GB of VRAM.**

Most agent harnesses assume a frontier cloud model. Point them at a small local
model and they tend to do one of two things: invent tool names that do not exist,
or describe the work instead of doing it. `isaacli` is a small harness built the
other way around, for the model you can actually run.

It reads and writes files, runs shell commands inside a three-layer sandbox, and
finishes the task or tells you it failed. Nothing leaves the machine.

> [AGPLv3](LICENSE). Free to use, study and modify. Offering it as a closed
> network service requires a commercial license: see [LICENSING.md](LICENSING.md).

---

## The measurement

Same model, same machine, same context window. `granite4:micro` (2.1 GB) served by
Ollama at `num_ctx=16384`, on a GTX 1650 with 4 GB of VRAM. Two trivial tasks,
verified on disk rather than in the transcript.

These runs used `granite4:micro`. The default base has since moved to
`granite4:micro-h`, which is smaller and hybrid-architecture, and the comparison
has not been re-run on it yet. The numbers below are left as measured rather than
restated against a model that did not produce them.

| Harness | create a file | list + run command + append | tokens |
|---|---|---|---|
| **isaacli** | pass | pass | **2.3k / 4.9k** |
| codex-cli 0.146.0 | fail | fail | 34.7k / 52.2k |
| ollama run `--experimental` | fail | not run | n/a |
| aider 0.86.2 | pass, after a config fix | out of scope by design | n/a |

Codex did not merely fail. On the first task it reported *"The file test_ctx.txt
has been created"* and the directory was empty. On the second it claimed a file
was empty while it held data, and executed nothing.

The usual explanation for this is a context window left at Ollama's low default.
That fix was applied here before the runs, at the recommended 16K, with MCP
servers removed so nothing bloated the schema. It failed anyway.

Full write-up, raw logs and the reproduction steps:
[`reports/harness-comparison/`](reports/harness-comparison/report.md).

### A fifth harness, measured separately

[Hermes Agent](https://github.com/NousResearch/hermes-agent) 0.20.0 is a much
larger and much more actively developed project than the three above, and it
supports any OpenAI-compatible endpoint including Ollama. It was measured on the
same two tasks and the same machine, but with `qwen3:4b-instruct-2507` rather
than `granite4:micro`, so it sits in its own table rather than in the one above.

| Configuration | create a file | list + run command + append |
|---|---|---|
| **Hermes, stock install** | **fail, 13 s, 0 tool calls** | **fail, 3 s, 0 tool calls** |
| Hermes, after reconfiguring the Ollama **server** | pass | not established |

Out of the box it never reached the model. Its tool schema measures **16,283
tokens** before your sentence is added, and Ollama's OpenAI-compatible endpoint
serves 4,096 by default. That endpoint **silently discards `options.num_ctx`**,
while the native one honours it, same server and same model:

| Endpoint | asked for | actually loaded |
|---|---|---|
| `/v1/chat/completions` (OpenAI-compat) | 32768 | **4096** |
| `/api/chat` (native, what isaacli uses) | 32768 | **32768** |

So the context has to be raised on the Ollama server itself. No Hermes setting
can do it, because Hermes speaks only the wire that drops the field. That is
decision 1 below, which was a design guess when it was written and is now a
measurement.

Worth saying plainly: **once the server is fixed, Hermes executes honestly.** It
called the right tool, wrote the right bytes, and told the truth about it. It is
not in the same category as the codex-cli run above. What it costs is time: the
`write_file` call itself took under a second, while more than three minutes went
to processing that fixed 16 K prompt, which is reprocessed on every turn. Those
timings were taken on a machine that turned out to be under load, so the report
records them as indicative rather than measured.

Full write-up, raw logs and an upstream bug found along the way:
[`reports/hermes-agent/`](reports/hermes-agent/report.md).

## Quickstart

Requires [Ollama](https://ollama.com), Python 3.10+, and `bwrap`
(`bubblewrap`) for the sandbox.

```bash
git clone https://github.com/owevertonguedes/isaacli.git
cd isaacli
ollama pull granite4:micro-h
./scripts/build-model.sh

./isaacli                                 # interactive REPL
./isaacli "run git status and tell me what is pending"
./isaacli --workspace /path/to/project
```

The base model is not baked in. `build-model.sh` reads `BASE_MODEL`,
`MODEL_NAME`, `NUM_CTX` and `TEMPERATURE` from the environment, because the right
base is a measurement result rather than a constant, and it changes as models
improve. It refuses a base that does not advertise the `tools` capability, since
that failure otherwise shows up as the agent narrating work instead of doing it.

```bash
BASE_MODEL=granite4:micro NUM_CTX=16384 ./scripts/build-model.sh
```

Inside the REPL, `/help` lists the commands, `/tools` shows which tools and which
binaries are allowed, and `/status` reports token usage for the session.

## Why it works

Nothing exotic. Four decisions, each of which is a failure mode avoided:

1. **Native `/api/chat`**, not an OpenAI-compat translation layer. Codex requires
   `wire_api = "responses"`, and `wire_api = "chat"` is refused outright by
   0.146.0. The compat layer also costs you the context window: Ollama drops
   `options.num_ctx` on `/v1` and honours it on `/api/chat`, measured both ways
   in [`reports/hermes-agent/`](reports/hermes-agent/report.md).
2. **A short tool schema.** Seven file and shell tools, so the list stays inside
   what a 2 GB model can hold and match against.
3. **A model with native tool calling**, picked by measurement rather than by
   parameter count. See the reasoning in
   [`Modelfile.isaac-granite.tmpl`](tool_harness/Modelfile.isaac-granite.tmpl).
4. **`num_ctx` and `temperature` set explicitly** in the model, so they travel
   with the model instead of depending on how the server was started. Ollama's
   default context truncates the tool schema silently, and a model that cannot see
   its tools invents plausible ones.

## The sandbox

Command execution is contained in three independent layers, in
[`tool_harness/execucao.py`](tool_harness/execucao.py):

- **Direct execve**, no shell, so there is no injection through `;`, `&&` or `$()`
- **A short allowlist** of binaries, widened by use rather than in anticipation
- **`bwrap`** with the whole disk read-only, networking closed, and only the
  working directory writable

File tools refuse to escape their root, including through absolute paths and
`..`. This is tested with bait planted outside the directory, in
[`testar_sandbox.py`](tool_harness/testar_sandbox.py).

This part is reusable on its own, in any project that executes model-generated
code, local or cloud.

## What else is in here

This started as an experiment on whether a small local model could be made into a
reliable agent, and the measurements from that are kept, including the ones that
failed.

| directory | what it holds |
|---|---|
| [`tool_harness/`](tool_harness/) | the agent: CLI, tools, sandbox, dataset validator, and a test per piece |
| [`bancada/`](bancada/) | a code bench whose validator also checks that the naive solution *fails*, because a ruler that passes the naive solution is not measuring anything |
| [`qwen_tools_lora/`](qwen_tools_lora/) | teaching tool calling to a model that could not do it: 0/8 to 6/8, and the `lm_head` fix that made it work |
| [`datasets/`](datasets/) | 30 curated examples, each with the criterion that admitted it |
| [`reports/`](reports/) | raw measurements from every run, including the rejected ones |

Two findings from that phase are worth pulling out, because they generalise past
this repo:

**The ruler lies before the model does.** Six measurement bugs appeared in a
single day. Five made results look worse than they were, one made them look
better. `skip_special_tokens=True` was deleting `<tool_call>` from the string
being graded. A reasoning budget was consuming the entire output allowance,
producing an empty response with HTTP 200 and no error. A bench answer key was
simply wrong. When a result looks bad, suspect the ruler first.

**Scaffolding beats a bigger model when the hardware is the ceiling.** `pass@1`
was 40% against `pass@8` at 75%. That gap is capability already sitting in the
weights that a single attempt does not reach. The ceiling per attempt is fixed;
the ceiling of the *system* is not.

## Honest limits

- Two tasks, one model, one machine. This shows that a small purpose-built harness
  executes reliably where general-purpose harnesses did not, on this hardware, on
  file and shell work. It does not show it is better at what Aider or Codex are
  built for: diff-based editing across a large repository, deep git integration,
  or driving frontier cloud models.
- A 2 GB model is a 2 GB model. Raw capability comes from pretraining and you
  download it finished. What this repo adds is reliability and specialisation,
  not intelligence.
- No LoRA adapter trained here was ever approved for use. The best one moved a
  workflow from 1/5 to 5/5 and left another at 4/6, still emitting tool names that
  did not exist. Nothing was merged into the base weights. The rejected runs are
  in [`reports/lora/`](reports/lora/), which is where the honest information is.

## Contributing

Issues and pull requests are welcome, particularly reproductions on other
hardware, and particularly another measurement bug that slipped through.

By submitting a pull request you agree to license your contribution under AGPLv3
and grant the maintainer the right to include it in commercial licenses of this
project. See [LICENSING.md](LICENSING.md).
