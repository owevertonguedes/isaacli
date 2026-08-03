# Same model, four harnesses: which one actually executes?

**Date:** 2026-08-04
**Hardware:** GTX 1650, 4 GB VRAM, 15 GB RAM
**Ollama:** 0.30.10

## What this measures

Every harness below was given the **same model**, on the same machine, with the
same context window: `granite4:micro` (2.1 GB) served by Ollama with
`num_ctx=16384` and `temperature=0`.

That matters, because the usual explanation for a local model failing inside an
agent is "your model is too small" or "your context is too short". Holding both
fixed turns the question into something answerable: **does the harness get the
model to execute, or not?**

Two tasks, deliberately trivial:

1. **Create a file** with given content.
2. **Multi-step**: list the directory, run `wc -l` on a file, append a line to it.

Success is checked on disk, not in the transcript. A harness only passes if the
file exists with the expected bytes afterwards.

## Results

| Harness | Task 1 | Task 2 | Tokens (t1 / t2) |
|---|---|---|---|
| **isaac** (this repo) | pass | pass | **2,309 / 4,918** |
| codex-cli 0.146.0 | fail | fail | 34,752 / 52,193 |
| ollama run `--experimental` | fail | not run | n/a |
| aider 0.86.2 | pass (with caveat) | fail (out of scope) | n/a |

Isaac used **15x fewer tokens on task 1 and 10.6x fewer on task 2**, and finished
each in under 10 seconds, while being the only harness to pass both.

## Failure modes, verbatim

### codex-cli: reports success for a file it never created

Raw log: [`codex-create-file.log`](codex-create-file.log)

```
ERROR codex_core::tools::router: error=unsupported call:
ERROR codex_core::tools::router: error=unsupported call:
ERROR codex_core::tools::router: error=unsupported call:
codex
The file `test_ctx.txt` has been created in the current directory with
the content **"ola mundo"**.
```

The working directory was empty afterwards. The model emitted tool names that do
not exist in the harness (`write_file`, `create_local_file` in earlier runs),
each rejected as `unsupported call`, and then produced a confident completion
message describing work that never happened.

This is the worst failure shape available: not an error, a false pass. An
operator reading only the last line would record a success.

### codex-cli: hallucinates file contents on the multi-step task

Raw log: [`codex-multistep.log`](codex-multistep.log)

The file `alvo.txt` contained `ola mundo`. Codex printed shell commands as prose
instead of executing them, and asserted:

```
No entanto, como o arquivo está vazio, a contagem retornará `0`.
```

The file was not empty. Nothing was listed, nothing was run, nothing was
appended. Final content was unchanged.

### ollama run: describes the work instead of doing it

With `--experimental --experimental-yolo` (Ollama's own agent loop), asking for a
file produced a Python snippet explaining how to write one. No file was created.

With `--experimental-websearch`, asking for the current top BBC headline produced
a fabricated story, including a factual tell: it placed king penguins in the
Arctic. No `OLLAMA_API_KEY` was configured, and the missing-search condition
failed silently rather than raising an error.

### aider: passes task 1 after a config fix, task 2 is out of its scope

Aider wrote the file only after the provider prefix was changed from `ollama/` to
`ollama_chat/`. With the wrong prefix it looped on "The LLM did not conform to
the edit format" until it gave up. Even on the successful run the model named the
file `test_aider.txt` rather than the requested `teste_aider.txt`.

On task 2 it did nothing, and **this one is not a defect**: Aider deliberately
does not expose an arbitrary-command tool to the model. It is a git-paired code
editor, so shell work surfaces as a suggestion or a fixed lint/test command, not
as an autonomous action. Recording it as a loss for Aider would be measuring it
against a job it does not apply for.

## The configuration objection, closed in advance

The common answer to "local models hallucinate tool calls" is that Ollama
defaults the context window too low, and that raising `num_ctx` to 16K fixes it.
That is real advice and it is repeated in the harness ecosystem, including open
issues on other projects:

- [sst/opencode#1034 — Local Ollama tool calling either not calling or failing outright](https://github.com/sst/opencode/issues/1034)
- [sst/opencode#1068 — Tool use with Ollama models](https://github.com/sst/opencode/issues/1068)

**That fix was applied here before the runs above.** `num_ctx` was set to 16384,
the documented minimum, and the MCP servers were removed from the Codex
configuration so nothing bloated the tool schema. Codex still failed both tasks.

So the result is not "Codex was misconfigured". It is: with the recommended fix
in place and the model held constant, the difference that remained was the
harness.

## Why the small harness wins here

Nothing exotic, and none of it is a new idea. It is four boring decisions:

1. **Native `/api/chat`**, not an OpenAI-compat translation layer. Codex requires
   `wire_api = "responses"`; setting `wire_api = "chat"` is refused outright by
   0.146.0.
2. **A short tool schema**: six file and shell tools, so the list stays inside
   what a 2 GB model can hold and match.
3. **A model with native tool-calling**, chosen by measurement rather than by
   parameter count.
4. **Explicit `num_ctx` and `temperature=0`**, set in the Modelfile rather than
   inherited from a default.

The general lesson is the one this repository keeps arriving at from different
directions: when the hardware is the ceiling, the scaffolding is where the
remaining wins are.

## Scope and limits

Read this for what it is. Two tasks, one model, one machine, one session. It
shows that a small purpose-built harness executes reliably where general-purpose
harnesses did not, on this hardware, on file and shell work. It does not show
that this harness is better at what Aider or Codex are actually built for:
diff-based editing across a large repository, git integration, or working against
frontier cloud models.

Reproduce it before you trust it. The raw logs are in this directory.
