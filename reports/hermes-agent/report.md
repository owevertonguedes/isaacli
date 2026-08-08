# Hermes Agent on 4 GB: the context size it asks for never arrives

**Date:** 2026-08-05
**Hardware:** GTX 1650, 4 GB VRAM, 15 GB RAM
**Ollama:** 0.30.10
**Hermes Agent:** v0.20.0 (2026.8.3), installed from the official installer
**Model under test:** `qwen3:4b-instruct-2507-q4_K_M` (2.5 GB), native context 262,144

## Why I ran this

The earlier comparison in [`reports/harness-comparison/`](../harness-comparison/report.md)
put isaacli against codex-cli, aider and `ollama run`. Hermes Agent is a much
larger and much more actively developed harness than any of those, from Nous
Research, and it advertises support for any OpenAI-compatible endpoint including
Ollama. If a general-purpose harness of that quality drives a small local model
correctly on this hardware, the case for a purpose-built harness gets weaker.

So the question is narrow and falsifiable: **with nothing but a stock install and
the documented Ollama setup, does it execute the same two trivial tasks?**

## Result

Two tasks, same as the earlier comparison, verified on disk and not in the
transcript.

| Configuration | task 1 (create a file) | task 2 (list + run + append) |
|---|---|---|
| **Hermes, stock install** | **fail, 13 s, 0 tool calls** | **fail, 3 s, 0 tool calls** |
| Hermes, after reconfiguring the Ollama server | pass, 368 s and 671 s | not established, see limits |
| isaacli | pass, 9.3 s | pass, 7.8 s |

Out of the box it never reached the model. Both tasks died before a single tool
call, with this on screen:

```
Context length exceeded (34 tokens). Cannot compress further.
```

And this in `~/.hermes/logs/agent.log`:

```
request (16283 tokens) exceeds the available context size (4096 tokens)
```

## The actual cause, isolated

This is not the model being too small, and it is worth being precise about it
because the obvious reading is wrong.

`qwen3:4b-instruct-2507` advertises **262,144 tokens** of context. Hermes reads
that correctly from Ollama's `/api/show`. Hermes then asks for it on every
request. The request is what does not survive the trip.

Ollama's OpenAI-compatible endpoint **silently discards `options.num_ctx`**. The
native endpoint honours it. Same server, same model, same field, one curl each:

| Endpoint | `options.num_ctx` sent | CONTEXT the server loaded |
|---|---|---|
| `/v1/chat/completions` (OpenAI-compat) | 32768 | **4096** |
| `/api/chat` (native) | 32768 | **32768** |

Reproduce it:

```bash
curl -s http://localhost:11434/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:4b-instruct-2507-q4_K_M","messages":[{"role":"user","content":"oi"}],
       "max_tokens":5,"options":{"num_ctx":32768}}'
ollama ps    # CONTEXT 4096

ollama stop qwen3:4b-instruct-2507-q4_K_M
curl -s http://localhost:11434/api/chat -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:4b-instruct-2507-q4_K_M","messages":[{"role":"user","content":"oi"}],
       "stream":false,"options":{"num_ctx":32768,"num_predict":5}}'
ollama ps    # CONTEXT 32768
```

Hermes does the right thing on its side: its `custom` provider plugin maps
`ollama_num_ctx` to `extra_body.options.num_ctx`
(`plugins/model-providers/custom/__init__.py`). The field simply has no channel
on that wire. So `model.ollama_num_ctx` in `config.yaml`, which is the setting
Hermes itself points you to, **is a no-op against Ollama over `/v1`**.

That is the part that decides the question I asked, so it is worth ruling out the
obvious objection that I picked the wrong setup path. I did not, and the reason
is structural rather than empirical: **Hermes has no native Ollama transport in
its chat loop.** Grepping the source for `/api/chat` returns a Slack URL, a web
upload route, and a comment in the custom provider plugin noting that
`/api/chat` is the endpoint honouring a field the OpenAI wire drops
(`plugins/model-providers/custom/__init__.py:59`, citing ollama#14820). Every
chat request leaves over `/v1`.

Combine that with the table above and it closes: Hermes speaks only the wire that
discards `num_ctx`, so no value of any Hermes setting can raise the served
context. The fix is not a setting I overlooked, and there is no setting that
works. The context has to be changed outside the harness, either by restarting
the server with `OLLAMA_CONTEXT_LENGTH`, or by baking `PARAMETER num_ctx` into a
Modelfile.

Meanwhile the 4,096 default is fatal here for a reason that has nothing to do
with parameter count: Hermes ships **15 toolsets enabled by default**, and its
system prompt plus tool schema measures **16,283 tokens** before the user's
sentence is added. A 1 B model would need the same 16,283 tokens. isaacli exposes
**7 tools** and spent 2,249 prompt tokens on the same task, 7.2x less.

## A guard that checks the wrong number

Hermes has a purpose-built error for exactly this situation, in
`agent/conversation_loop.py`, and it is a good one: it names the model, the
base URL, and tells you to set `model.ollama_num_ctx: 65536` or
`PARAMETER num_ctx 65536` in a Modelfile.

It never fired. The guard runs only when `_ollama_num_ctx < MINIMUM_CONTEXT_LENGTH`
(64,000, in `agent/model_metadata.py:390`). Hermes had detected 262,144 from
`/api/show`, so by its own bookkeeping it had plenty. **The guard checks the
context Hermes requested, not the context the server actually loaded**, and the
two differ by 64x precisely in the case the guard exists to catch.

The information needed to correct that was on hand and went unread. Ollama puts
the real value in the body of the 400 it returns:

```
HTTP 400: {"error":{"code":400,"message":"request (16301 tokens) exceeds the
available context size (4096 tokens), try increasing it",
"type":"exceed_context_size_error","n_prompt_tokens":16301,"n_ctx":4096}}
```

and the next line of the same transcript reads:

```
Context length exceeded, but provider did not report a max context length;
keeping context_length at 262,144 tokens and compressing.
```

The provider did report it, as `n_ctx: 4096`, in the error Hermes had just
parsed. Hermes then tries to compress a 4,535-token conversation to fit a window
it believes is 262,144, concludes it cannot compress further, and prints
`Context length exceeded (34 tokens). Cannot compress further.`, which points at
nothing. Worth reporting upstream: it is a bug in the diagnosis path rather than
in the agent loop, and the fix is to read `n_ctx` off the error body.

## What happens once the server is fixed

With `OLLAMA_CONTEXT_LENGTH=64000`, `OLLAMA_KV_CACHE_TYPE=q8_0`, and
`model.context_length: 64000`, Hermes runs, and **it runs honestly**.

On task 1 it called `write_file` with the right arguments, the file appeared on
disk with the right bytes, and the summary it printed was true. No invented tool
names, no success claimed for work that did not happen. That is a materially
different harness from the codex-cli run in the earlier report, which reported
creating a file into an empty directory. Hermes did not do anything of the kind,
and the comparison should not be read as putting them in the same category.

The cost is wall time. Task 1 took 368 s on one run and 671 s on another, against
9.3 s for isaacli. The breakdown explains itself: **the `write_file` call itself
took 0.7 s**, and 202.9 s went to processing the 16,283-token fixed prompt, at
80 tok/s, before any work began. That prompt is reprocessed every turn, so a
task needing three tool calls pays it three times.

The 64,000-token floor is the second half of the cost. A 64 K KV cache does not
fit in 4 GB: the 2.5 GB model occupied **9.0 GB** and ran at **70% CPU / 30% GPU**.
The binding constraint on this machine is the KV cache, not the weights, which is
why picking an even smaller model does not help.

## What this does and does not show

- **Out of the box, against Ollama, on any hardware:** established. Two tasks,
  stock install, documented setup, zero tool calls. The transport experiment
  above shows the cause is independent of the model and of the GPU.
- **The wall-time numbers: treat as indicative, not measured.** Both timed runs
  were contended. A 9.9 GB model was downloading during the first, and a separate
  job repeatedly tried to load it onto the same 4 GB GPU during the second, which
  eventually killed the Ollama server. Clean numbers, taken on an idle machine
  over three passes, are still owed. Anyone reproducing this should check
  `ollama ps` is empty and that `Load failed` does not appear in the server log
  during the run.
- **Task 2 under a fixed server: not established.** One run hit a 900 s timeout
  after two real tool calls, under contention. The clean retry died on
  `APIConnectionError` when the server went down. No valid run exists, and I am
  not recording a verdict for it.
- **Hermes with a large or cloud model: not tested.** Nothing here says Hermes
  performs badly in the deployment it is built for, and the 64 K floor plus a
  16 K tool schema are cheap assumptions when the model is served by somebody
  else's GPU.
- **One machine, one model, one session.**

## The conclusion I draw

The failure that matters is not "a big harness cannot drive a small model".
Hermes drove a 4 B model correctly and truthfully once it could see its tools.
The failure is that the three assumptions it inherits from cloud deployment, an
OpenAI-compatible transport, a 64,000-token floor, and a 16 K tool schema, are
each individually fatal on a 4 GB box, and **none of the three is about how many
parameters the model has.**

Which is the same conclusion the earlier report reached from the other side, now
with a cleaner mechanism attached to it: on this hardware the scaffolding is
where the remaining wins are, and the specific win is refusing to pay a fixed
16 K tokens per turn for tools the task will never call.

The raw logs are in this directory. Reproduce it before you trust it.
