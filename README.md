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

## Quickstart

Requires [Ollama](https://ollama.com), Python 3.10+, and `bwrap`
(`bubblewrap`) for the sandbox.

```bash
git clone https://github.com/owevertonguedes/isaacli.git
cd isaacli
ollama pull granite4:micro
ollama create isaac-granite -f tool_harness/Modelfile.isaac-granite

./isaac                                   # interactive REPL
./isaac "run git status and tell me what is pending"
./isaac --workspace /path/to/project
```

Inside the REPL, `/help` lists the commands, `/tools` shows which tools and which
binaries are allowed, and `/status` reports token usage for the session.

## Why it works

Nothing exotic. Four decisions, each of which is a failure mode avoided:

1. **Native `/api/chat`**, not an OpenAI-compat translation layer. Codex requires
   `wire_api = "responses"`, and `wire_api = "chat"` is refused outright by
   0.146.0.
2. **A short tool schema.** Seven file and shell tools, so the list stays inside
   what a 2 GB model can hold and match against.
3. **A model with native tool calling**, picked by measurement rather than by
   parameter count. See the reasoning in
   [`Modelfile.isaac-granite`](tool_harness/Modelfile.isaac-granite).
4. **`num_ctx` and `temperature` set explicitly** in the Modelfile, so they travel
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
- Parts of the codebase are still in Portuguese, including several module names.
  That is being migrated.

## Contributing

Issues and pull requests are welcome, particularly reproductions on other
hardware, and particularly another measurement bug that slipped through.

By submitting a pull request you agree to license your contribution under AGPLv3
and grant the maintainer the right to include it in commercial licenses of this
project. See [LICENSING.md](LICENSING.md).
