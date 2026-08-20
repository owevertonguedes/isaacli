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
5. the matching tests in `tests/check_*.py`.

Old logs and sessions are historical evidence, not current state. Re-verify
processes, configuration, installed models and test results live.

Read [SECURITY.md](SECURITY.md) before changing persistence, logs or remote
providers. Read [INSTALLATION.md](INSTALLATION.md) before changing the launcher,
install or any removal path.

## Main flow

```text
isaacli (launcher)
  -> tool_harness/cli.py (arguments, REPL and session)
     -> setup_ollama.py (models, context, effort and API keys)
     -> cli_kaggle.py (Kaggle CLI lifecycle, explicit kernel push and URL discovery)
     -> agent.py (loop: messages -> model -> tool calls -> model)
        -> tools.py (schemas and tools)
           -> execution.py (no-shell command, allowlist, approval and bwrap)
              -> seccomp_filter.py (the BPF filter bwrap loads)
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
| `tool_harness/setup_ollama.py` | shared engine selection, local setup, local models, context, reasoning and OpenAI-compatible API | Kaggle entries delegate to `cli_kaggle.run_kaggle`; context must not create `16k`/`32k` Ollama copies. |
| `tool_harness/agent.py` | Ollama/API calls, streaming, normalisation and the tool loop | Ollama uses `/api/chat`; remote APIs use `/chat/completions`. |
| `tool_harness/tools.py` | schemas and implementations of the agent's tools | `fetch_url` is the general web reader; unapproved terminal commands stay offline in the sandbox. |
| `tool_harness/execution.py` | classification, approval and confined execution of programs | Never add a shell, pipes or redirection as a UI shortcut. Never add a veto the user cannot see: once a command is approved, only the kernel says no. |
| `tool_harness/seccomp_filter.py` | assembles the seccomp-BPF program `execution.py` hands to `bwrap` | Pure Python, no `libseccomp` dependency and no committed blob, so the deny-list stays reviewable. Syscall numbers are x86_64's; `build_filter()` returns `None` elsewhere instead of guessing. |
| `tool_harness/config.py` | public config and local secrets | API keys live in `secrets.json` with mode `0600`, outside Git. |
| `tool_harness/installation.py` | per-user launcher install, uninstall and explicitly confirmed purge | It never removes a launcher owned by another checkout or an unrecognised Ollama installation. |
| `tool_harness/cli_kaggle.py` | installs a private Kaggle CLI, authenticates, filters benchmark-backed models with `hardware.fits`, selects the exact accelerator, pushes a generated kernel, discovers the tunnel and saves a profile | Setup, `isaacli kaggle`, `/model` and `/kaggle` call this one flow. Kernel orchestration ends at a generic `openai_compatible` profile. |
| `tool_harness/i18n.py`, `locales/` | every user-facing string, in English and Portuguese | A new key has to exist in both catalogs, with the same placeholders. |
| `tool_harness/model_catalog.json` | curated Ollama and Kaggle candidates with public evidence and exact GGUF metadata | It does not represent installed models. Kaggle candidates are filtered by hardware at runtime. |

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

The inventory below describes current paths. Its privacy guarantees and known
gaps are documented in [SECURITY.md](SECURITY.md); installation ownership,
recovery and destructive validation live in
[INSTALLATION.md](INSTALLATION.md).

- Configuration: `~/.config/isaacli/config.json` (or `XDG_CONFIG_HOME`).
- Secrets: `~/.config/isaacli/secrets.json`.
- Sessions: `tool_harness/cli_sessions/*.jsonl`.
- Feedback: `tool_harness/feedback/*.jsonl`.
- Managed-Ollama coordination: the user's runtime directory, or `/tmp`.

New sessions use UUIDv4. Older date-based IDs are still accepted for resuming,
and `_load_session` also reads logs written before the JSONL field names were
translated, so sessions recorded by earlier versions do not silently rebuild as
empty.

`isaacli install` creates only a per-user symlink in `~/.local/bin`. A normal
`uninstall` removes only that symlink; `uninstall --purge` also removes the
configuration, secrets, sessions, feedback and stale runtime coordination after
an explicit confirmation. `--purge --ollama` is the deliberately stronger Linux
route for someone who installed Ollama solely for Isaac: it additionally removes
an installation recognised as coming from Ollama's official script, including
its service, models and user data. It refuses unrecognised layouts and never
touches the clone. The official script may choose either `/usr/local` or `/usr`;
the latter is accepted only with both official-layout artifacts and no RPM/DEB
ownership. A known custom `OLLAMA_MODELS` path in the environment or systemd
unit/drop-in blocks automatic removal instead of making a false full-cleanup
claim. Strong purge validates the Isaac ownership/session state and completes
the Ollama removal before deleting the launcher and Isaac data, so a refused or
failed system teardown leaves a usable recovery route. Purge also refuses to
start while a live Isaac session is
registered, so it cannot remove the engine underneath another process.

`uninstall --purge --kaggle` is the sibling strong purge. It removes Kaggle only when `kaggle-install.json` records that isaacli created the isolated per-user environment and launcher. A package-managed or changed executable is refused. Kaggle authentication files are removed only after the strong purge warning. Existing third-party Kaggle installations and remote kernels are preserved.

The normal `isaacli kaggle` path always renders the GPU template in `contrib/kaggle/`. It never starts automatically and asks for confirmation immediately before each push. The separate `--flow-validation-cpu` switch renders a short CPU-only OpenAI-compatible probe for validating push, tunnel discovery, reachability and profile persistence without spending GPU quota. It cannot run a model and is never the default.

The resume command uses `isaacli` when this installation is on `PATH`;
otherwise it prints the absolute launcher that is actually executable.

## Terminal and shutdown

The REPL runs on the main screen. The conversation is plain output in the
terminal's own scrollback: the wheel scrolls it from the first message to the
last, and the text stays selectable and copyable. ↑/↓ at the prompt belong to
the typed-message history and to nothing else.

The alternate buffer must not come back to the REPL. It has no scrollback, and
terminals translate the wheel into ↑/↓ while it is active, so the wheel took
over the prompt's message history. Only full-screen menus use it, and leaving it
restores the conversation on its own, with no redraw.

Mouse reporting is never enabled either. An earlier version drove a scrollable
viewport with the wheel; it was removed because DEC 1007 turns the wheel and ↑
into the same sequence, and enabling reporting at all costs the terminal's
native selection. Do not reintroduce it.

Returning from a full-screen menu announces the outcome (`redraw_session`) and
prints nothing else: reprinting the transcript would duplicate it in the
scrollback.

Keys received during startup are discarded before the first prompt. During
generation, input is not echoed and is flushed at the end. `Ctrl+C` at the
prompt ends the REPL; repeated signals during cleanup are ignored until the
Ollama coordination finishes.

An Ollama started by isaacli is shared between isaacli sessions. The last
registered session stops only the server isaacli itself started. A pre-existing
server belongs to the user and must not be terminated by the app.

The same lifecycle is available to any local OpenAI-compatible server. An
`openai_compatible` profile may carry `"autostart": {"cmd": [...], "health_url":
"..."}`; when present, isaacli starts that server on demand, shares it across
sessions and stops it with the last one, exactly as it does for Ollama. Each
server gets its own lock and state file, keyed by profile, so two managed
servers can never read each other's client list and stop a process that still
has users. The health probe for these servers treats an HTTP answer as
reachable, including a 4xx: they do not share Ollama's `/api/version` shape, and
`GET /models` answering at all is the proof that matters. The gateway codes are
the exception. A llama-server that is still reading its model file answers 503,
and counting that as ready hands the user's first turn to a server that then
refuses it, so 502, 503 and 504 mean not ready rather than up.

The startup budget does not carry over from Ollama either. Ollama's daemon
answers immediately and loads the model on demand, while these servers read the
whole model before answering anything; a 2 GB model measurably takes longer than
Ollama's ten seconds. The budget is `AUTOSTART_TIMEOUT`, and a profile can raise
it for a large model on slow storage.

A key is required for a remote endpoint and optional for a loopback one, in the
request path as well as in setup. Enforcing it only at setup left the local path
accepted at configuration time and refused at request time. With no key, no
`Authorization` header is sent at all: an empty bearer value is worse than none,
because some servers reject the malformed header outright.

The guided setup only offers autostart for a loopback endpoint, because there
is nothing to start on a machine that is not this one. For the same reason a
loopback endpoint does not require an API key: a server the user runs has no
key to demand, and requiring one closed the local-first path entirely. Anything
reachable over the network still requires one.

## Model contracts

- Ollama: native chat API, tools required, `options.num_ctx` per call.
- Local OpenAI-compatible server (llama-server and equivalents): same contract
  as a remote API, optionally with a lifecycle managed by isaacli. Ollama
  remains the recommended path, because it is a single installer with a model
  catalog; the alternative is offered, not advertised as faster. No throughput
  claim is made in the interface without a measurement on the user's hardware.
- Remote API: OpenAI-compatible Chat Completions with streaming and function
  calling. `reasoning_effort` is optional and can be turned off.
- Providers with their own incompatible native formats will need explicit
  adapters; do not scatter provider-name conditionals through the REPL.

The `/model` menu separates source, model, context and effort. Kaggle appears as
a provider even before configuration and delegates to the same explicit setup
used by the top-level command. Recommended is not a synonym for installed:
Ollama installations come from its live server, while Kaggle candidates come
from the curated JSON and only appear when `hardware.fits` accepts their exact
weights and 16K KV cache for the selected accelerator.

When a provider rejects `reasoning_effort`, that rejection is the source of
truth: the agent retries without the parameter, stops sending it for the rest of
the turn, and signals the caller so `cli._persist_adjusted_thinking` can write
the correction to the profile. There is no per-model table.

Compatible API rate limits follow the same direction: successful response
headers provide the remaining quota and reset window, and the adapter uses the
provider's reported usage to estimate whether the next tool-loop request fits.
It waits before crossing that boundary when possible; a 429 with `retry-after`
or an equivalent error body remains the fallback. Provider names, plans, models
and numeric limits do not belong in the adapter.

For an explicit mutation request, a textual answer is not evidence of a change.
Until a changing tool has been called, the first text-only answer is discarded
and the model gets one explicit corrective attempt. Read-only exploration does
not satisfy that contract, the correction never loops, and a second text-only
answer is surfaced honestly as a clarification or failure.

The explicit-request detector is only a fast path for common phrasing; it is
not treated as a complete natural-language parser. Every read-only tool result
also carries an English model-facing reminder that inspection changed nothing
and that a persistent outcome requires a changing tool. This catches uncommon
wording without growing parallel verb lists for every interface language or
adding a separate intent-classification inference to every turn. A changing
call counts as confirmed only when its result reports success: a denied command,
non-zero exit or file-tool error remains an attempted change and receives a
model-facing failure reminder instead.

Successful file mutations return bounded objective evidence to the normal
post-tool model step: created/updated state, UTF-8 byte counts and a unified
diff (or an explicit no-line-difference marker). The evidence covers `write_file`, `append_file`,
`replace_between` and `replace_text`, is explicitly marked when truncated, and
only describes the resulting bytes; it never claims that the user's broader
request is complete. `replace_text` is the exact, unambiguous option: it refuses
absent or repeated old text without modifying the file, while whole-file and
section replacement remain available for deliberately broad work. Tool
arguments are validated against the same schema sent to the model, so a missing
required field produces a precise corrective result before Python dispatch.

Mutation requests stream from the first inference even while a tentative text
answer is hidden pending a successful changing tool. A separate progress
callback observes reasoning, answer and tool-argument chunks without displaying
their content, so the transient indicator can show an approximate live rate
during tool selection too. Ollama's exact generated-token count remains a
request-final metric and is used for the final rate.

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
- approval is the decision, not a suggestion. Exactly one thing may still refuse
  after the user says yes: the kernel (`bwrap`, nothing writable outside the
  workspace). The allowlist, the `gh` routes, the force-push flags, the network
  and the no-shell parser are all defaults for what the user did not look at, and
  every one of them steps aside on approval, an approved line with operators
  running through `sh -c` inside the same jail. Adding a second veto is a
  regression: whoever forked this owns their machine, and a rule that says "no,
  not even if you insist" only pushes them to a plain terminal with no sandbox at
  all. `git clone`, `push --force` and chained commands all paid for this;
- a cgroup ceiling on every command, via `systemd-run --user --scope` wrapping
  the `bwrap` line: `MemoryMax` together with `MemorySwapMax=0` (the first
  alone is not a limit while swap exists), `TasksMax` against fork bombs, and
  `CPUQuota`. This does not step aside on approval, same as `bwrap`: it is the
  kernel's cgroup controller, not policy. When `systemd-run` is missing the
  command still runs, unlimited, with a `NOTE:` appended to the output saying
  so; the docs and the tests must never claim the ceiling exists on a machine
  where it silently does not (see `tasks/done/007-seccomp-and-cgroup-limits.md`
  for the measurements behind the exact values);
- a seccomp-BPF filter on every command, passed to `bwrap` through
  `--add-seccomp-fd` and assembled in `tool_harness/seccomp_filter.py`. It denies
  nested namespaces and mounts, module and `kexec` control, the kernel keyring,
  NUMA and page migration, `bpf`, `userfaultfd`, `ptrace` and
  `perf_event_open`. The namespace group is the one that earned the work:
  measured here, a process inside the jail could call `unshare(CLONE_NEWUSER)`
  successfully, and a new user namespace carries a full capability set inside
  itself. Like the cgroup ceiling, it does not step aside on approval. It is
  x86_64-only, because the syscall numbers are, and degrades to a `NOTE:` on
  any other architecture rather than applying the wrong number table. Checking
  the arch is not sufficient by itself and the filter does not pretend it is:
  x32 reports the same arch and sets a bit in the syscall number, so any number
  carrying that bit is killed before the deny-list is consulted. Without that
  guard the entire list is one bit away from being skipped. It knowingly leaves
  `clone3` alone: seccomp cannot read the flags behind its struct pointer, and
  denying it would break threads in `python3` and `pytest`, so "no nested user
  namespaces" is not a claim this project makes;
- no exposure of API keys in config, logs or output;
- public URLs validated against local and private destinations;
- child processes tied to the right lifecycle, and idempotent cleanup;
- model text sanitised before it reaches the terminal.

## Verification

```bash
python3 tests/check_cli.py
python3 tests/check_agent_config.py
python3 tests/check_setup.py
python3 tests/check_tools.py
python3 tests/check_sandbox.py
python3 tests/check_execution.py
python3 tests/check_hardware.py
git diff --check
```

Everything that is a test lives under `tests/`: the fast checks above at the top
level, and the ones that need a container or a real model under
`tests/integration/`. `scripts/` holds development utilities that are not tests.
The two checks outside the pass above are `tests/check_commit_workflow.py`,
which calls a real model through `isaacli`, and
`tests/integration/test-install-lifecycle.sh`, which builds a disposable systemd
container to exercise install and purge without touching the developer's HOME.

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
