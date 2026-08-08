# isaacli: current architecture

This document is the technical entry point for people and for AI agents. It
describes the code that exists; it is not a promise that the current
organisation is the desired one.

## Read this first

Read in this order before changing the project:

1. `AGENTS.md`, when present in the development environment;
2. this document;
3. the current diff (`git diff`) and the Git state;
4. only the modules related to the task;
5. the matching tests in `tool_harness/check_*.py`.

Old logs and sessions are historical evidence, not current state. Re-verify
processes, configuration, installed models and test results live.

## Main flow

```text
isaacli (launcher)
  -> tool_harness/cli.py (arguments, REPL and session)
     -> setup_ollama.py (models, context, effort and API keys)
     -> agent.py (loop: messages -> model -> tool calls -> model)
        -> tools.py (schemas and tools)
           -> execution.py (no-shell command, allowlist, approval and bwrap)
     -> terminal_ui.py (alternate screen and selectors)
     -> i18n.py + locales/ (every string the user reads)
```

The root launcher only finds Python and hands execution to the harness. The
workspace chosen by the user becomes the boundary for the file tools and the
executor. Configuration, secrets and sessions live outside the workspace being
worked on.

## Module map

| File | Responsibility | Note |
| --- | --- | --- |
| `tool_harness/cli.py` | arguments, REPL, `/` commands, sessions, presentation, permissions and the Ollama lifecycle | The largest coupling point, and the first target for a responsibility inventory. |
| `tool_harness/terminal_ui.py` | alternate screen, menus, busy prompt | Must not enable mouse reporting: that breaks the terminal's native selection and copy. |
| `tool_harness/setup_ollama.py` | local setup, curated catalog, local models, context, reasoning and OpenAI-compatible API | Context is per-request configuration; it must not create `16k`/`32k` Ollama copies. |
| `tool_harness/agent.py` | Ollama/API calls, streaming, normalisation and the tool loop | Ollama uses `/api/chat`; remote APIs use `/chat/completions`. |
| `tool_harness/tools.py` | schemas and implementations of the agent's tools | `fetch_url` is the general web reader; terminal commands stay offline in the sandbox. |
| `tool_harness/execution.py` | classification, approval and confined execution of programs | Never add a shell, pipes or redirection as a UI shortcut. |
| `tool_harness/config.py` | public config and local secrets | API keys live in `secrets.json` with mode `0600`, outside Git. |
| `tool_harness/i18n.py`, `locales/` | every user-facing string, in English and Portuguese | A new key has to exist in both catalogs, with the same placeholders. |
| `tool_harness/model_catalog.json` | small curation of recommendations | It does not represent installed models; those come live from the local Ollama. |

## Language boundary

There are two audiences and they get different treatment:

- **Text the user reads** goes through `i18n.py` and lives in
  `locales/{en,pt-BR}.json`. Nothing user-facing is hardcoded in a module.
- **Text the model reads** (system prompts, tool names, tool descriptions,
  tool results and sandbox refusals) is always English, regardless of the
  interface language. It is a contract with the model, not a preference of the
  person at the keyboard. The system prompt separately instructs the model to
  answer in the user's language.

Identifiers, comments and docstrings are English throughout.

`cli.py` keeps one translator per process (`set_language`/`t`), because the CLI
is a single session and threading a `Translator` through every helper would be
ceremony without a second reader.

## Local state and data

- Configuration: `~/.config/isaacli/config.json` (or `XDG_CONFIG_HOME`).
- Secrets: `~/.config/isaacli/secrets.json`.
- Sessions: `tool_harness/cli_sessions/*.jsonl`.
- Feedback: `tool_harness/feedback/*.jsonl`.
- Managed-Ollama coordination: the user's runtime directory, or `/tmp`.

New sessions use UUIDv4. Older date-based IDs are still accepted for resuming,
and `_load_session` also reads logs written before the JSONL field names were
translated, so sessions recorded by earlier versions do not silently rebuild as
empty.

The resume command uses `isaacli` when this installation is on `PATH`;
otherwise it prints the absolute launcher that is actually executable.

## Terminal and shutdown

The REPL uses the alternate buffer so the conversation does not mix with the
shell history. `/history` prints normally, which keeps the transcript in the
terminal's native scrollback and leaves it selectable and copyable. ↑/↓ at the
prompt belong to the typed-message history.

Mouse reporting is never enabled. An earlier version drove a scrollable viewport
with the wheel; it was removed because DEC 1007 turns the wheel and ↑ into the
same sequence, and enabling reporting at all costs the terminal's native
selection. Do not reintroduce it.

Full-screen menus must always redraw the recent conversation when returning to
the REPL.

Keys received during startup are discarded before the first prompt. During
generation, input is not echoed and is flushed at the end. `Ctrl+C` at the
prompt ends the REPL; repeated signals during cleanup are ignored until the
Ollama coordination finishes.

An Ollama started by isaacli is shared between isaacli sessions. The last
registered session stops only the server isaacli itself started. A pre-existing
server belongs to the user and must not be terminated by the app.

## Model contracts

- Ollama: native chat API, tools required, `options.num_ctx` per call.
- Remote API: OpenAI-compatible Chat Completions with streaming and function
  calling. `reasoning_effort` is optional and can be turned off.
- Providers with their own incompatible native formats will need explicit
  adapters; do not scatter provider-name conditionals through the REPL.

The `/model` menu separates source, model, context and effort. Recommended is
not a synonym for installed: recommendations come from the curated JSON,
installed models come from querying the Ollama server.

When a provider rejects `reasoning_effort`, that rejection is the source of
truth: the agent retries without the parameter, stops sending it for the rest of
the turn, and signals the caller so `cli._persist_adjusted_thinking` can write
the correction to the profile. There is no per-model table.

Generation metrics are only comparable when context, prompt and tools are
equivalent too. A short benchmark at `num_ctx=4096` does not represent the cost
of an agent session at `num_ctx=32768` with history and tool schemas: beyond
generation, the model has to process that whole prefix, and a larger KV cache
may stop fitting entirely in the GPU.

## Safety invariants any reorganisation must preserve

- one program per execution, with no shell;
- the workspace as the boundary for files and commands;
- approval before unauthorised mutations, and an explicit warning when the
  command is destructive, so approval does not become a reflex;
- no exposure of API keys in config, logs or output;
- public URLs validated against local and private destinations;
- child processes tied to the right lifecycle, and idempotent cleanup;
- model text sanitised before it reaches the terminal.

## Verification

```bash
python3 tool_harness/check_cli.py
python3 tool_harness/check_agent_config.py
python3 tool_harness/check_setup.py
python3 tool_harness/check_tools.py
python3 tool_harness/check_sandbox.py
python3 tool_harness/check_execution.py
git diff --check
```

The checks are standalone scripts, not pytest modules: they run their assertions
at import time. That is why they are named `check_*` and not `test_*`: pytest
collection alone would execute them.

Run `check_execution.py` outside a nested sandbox: `bwrap` needs to create its
own loopback interface.

`check_cli.py` points `XDG_CONFIG_HOME` at a temporary directory before
importing the CLI, so the suite neither reads nor writes the real
`~/.config/isaacli/config.json`.

## Next step: organisation

The next session should start with an inventory, not by moving files blindly.
The visible debt is that `cli.py` concentrates presentation, application,
sessions, commands and lifecycle. A reorganisation should define testable
boundaries for at least:

- internal commands;
- sessions and persistence;
- providers/models;
- terminal presentation and input;
- process lifecycle;
- permission policy.

Before each move, record imports and consumers, keep a compatibility layer when
needed, and run the suite. Do not mix a mechanical reorganisation with a
behaviour change in the same step.
