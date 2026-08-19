# Using isaacli

Detail that would bloat the [README](../README.md): the setup flow, the REPL
commands, the permission model and how sessions are resumed.

## Installation and removal

From a fresh clone, install the per-user command and run setup:

```bash
./isaacli install
isaacli setup
```

Installation creates only `~/.local/bin/isaacli`, a symlink to the clone. It
does not need `sudo`, never overwrites another command and transparently runs on
the host when invoked from a Flatpak terminal such as VS Code from Flathub.

Removal has three deliberately separate levels:

| Command | Removed | Preserved |
| --- | --- | --- |
| `isaacli uninstall` | the per-user command | configuration, secrets, sessions, Ollama and the clone |
| `isaacli uninstall --purge` | command, configuration, API keys, permissions, sessions, feedback and runtime state | Ollama, its models and the clone |
| `isaacli uninstall --purge --ollama` | everything above plus a recognised official Linux Ollama installation, its service, models and user data | the clone |

The two purge forms require the exact confirmation displayed by the command.
Purge refuses to run while another isaacli session is active. Ollama removal
also refuses package-managed or otherwise unrecognised installations, which
must be removed through the package manager that owns them.

## Setup

`isaacli setup` walks through language, engine, model, context size and
reasoning effort. The first interactive run opens it automatically when no
profile exists.

### Local models through Ollama

Installed tags are read live from the Ollama API. The model menu has two
sections:

- a small curated recommendation list (Qwen3.6 35B-A3B UD-IQ1_M first), which is
  data in `tool_harness/model_catalog.json`, not model-specific branches in the
  code;
- everything the local server reports as installed.

Models already shown as recommendations are not repeated in the second section,
and legacy aliases known to be context-only copies are collapsed into their base
model. Long lists scroll inside the selector instead of overflowing the
terminal.

A recommended tag that is not registered in Ollama shows as not installed and is
downloaded only after confirmation. Capabilities and the context limit are
detected from Ollama rather than assigned from the catalog, so a model without
the `tools` capability is refused instead of being configured into a state where
it cannot work.

Context and reasoning are separate settings. Context is stored in the profile
and sent per request, so choosing 16K or 32K does not create duplicate Ollama
models. Manual context accepts friendly values such as `12K`; the safe minimum
is `8K`.

### OpenAI-compatible APIs

The same setup can create a generic API profile. It asks for a provider label,
base endpoint, exact model ID, API key and reasoning mode. No provider or model
is compiled into the adapter. Groq, for example, is just base endpoint
`https://api.groq.com/openai/v1` with model `openai/gpt-oss-20b`.

The key is stored in `~/.config/isaacli/secrets.json` with mode `0600`, never in
the workspace, the session log or the regular `config.json`.

Different providers accept different `reasoning_effort` values and none of them
declare that in a standard way. Rather than keep a per-model table, isaacli
treats the provider's own HTTP 400 rejection as the source of truth: it retries
without the parameter, tells you, and saves the correction to the profile so the
extra round trip does not repeat.

### Local servers you run yourself

A base endpoint on `localhost`, `127.0.0.1` or `::1` needs no API key: a server
running on your own machine has none to demand. Press Enter at the key prompt.

For those endpoints setup also offers to manage the server's lifecycle. Give it
the command that starts your server and isaacli starts it when you open a
session, shares one server across simultaneous sessions and stops it when the
last session closes, exactly as it already does for Ollama:

```
Command (Enter to skip): llama-server -m /models/qwen2.5-coder-7b-q4.gguf -c 8192
```

Leave it empty to keep starting the server yourself; isaacli will then only
check whether the endpoint is already up. A server that was already running when
isaacli started belongs to you and is never stopped by isaacli.

The stored profile carries the command as a plain list, so you can also write or
edit it directly in `~/.config/isaacli/config.json`:

```json
"autostart": {
  "cmd": ["llama-server", "-m", "/models/qwen2.5-coder-7b-q4.gguf", "-c", "8192"],
  "health_url": "http://127.0.0.1:8080/v1/models"
}
```

Ollama remains the recommended engine, because it is one installer with a model
catalog and it already manages its own lifecycle. A direct llama-server is
offered as an alternative, not as a promise of more speed: the two have not been
measured against each other on this project's hardware, and published
comparisons disagree with each other.

### Debugging

`isaacli --debug` prints the traceback for exceptions the normal flow absorbs on
purpose, such as probing a server that is not up yet or reading an error body
that turned out to be unreadable. It changes nothing about what runs or what is
returned; it only stops those causes from being invisible. `ISAACLI_DEBUG=1`
does the same for the paths that run before arguments are parsed. Each site
reports once per run.

### Advanced: build-model.sh

The `build-model.sh` flow remains available for automation. It reads
`BASE_MODEL`, `MODEL_NAME`, `NUM_CTX` and `TEMPERATURE` from the environment and
refuses a base that does not advertise the `tools` capability.

```bash
BASE_MODEL=granite4:micro NUM_CTX=16384 ./scripts/build-model.sh
```

## Commands

| Command | What it does |
| --- | --- |
| `/help` | list the commands |
| `/setup` | configure or repair the engine without closing isaacli |
| `/status` | session, workspace, model and token usage |
| `/tools` | which tools and terminal binaries are allowed |
| `/sessions` | saved CLI sessions |
| `/history` | this session's full conversation |
| `/show [n\|last]` | expand the full output of a collapsed command |
| `/log` | path of this session's JSONL file |
| `/feedback` | how to rate the session or task |
| `/good`, `/bad`, `/score 0-10` | rate the last task |
| `/workspace [path]` | show or change the working folder |
| `/model [profile\|name]` | select model, reasoning effort and context |
| `/permissions` | list or clear persistent authorizations |
| `/mode` | switch the permission mode |
| `/language` | change the interface language (saved) |
| `/clear` | clear the conversation context |
| `/new` | end the current session and start another |
| `/exit` | quit |

`/bom`, `/ruim` and `/nota` still work as aliases for `/good`, `/bad` and
`/score`.

Typing `/` on an empty prompt opens the command palette. Results filter while
you type; arrow keys select and `Tab` inserts. When `prompt_toolkit` is not
installed, GNU Readline `Tab` completion is the fallback. `Alt+Enter` inserts a
line break without sending the message.

## Permissions

Terminal commands are always shown before execution.

- **Safe mode** (default) runs read-only commands automatically. Anything that
  could change the workspace stops and offers four choices: allow once, always
  allow in this workspace, always allow globally, or deny.
- **Authorized-only mode** (`Shift+Tab`, or `/mode`) asks even for reads, so only
  rules you already saved run without a prompt.

When the command is destructive or hard to undo (`rm`, `git push`, `git reset`,
force flags and the like) the prompt says so explicitly. The point is to keep
approval from becoming a reflex: a prompt you always answer the same way stops
being a decision.

`/permissions` lists the saved rules and can clear the workspace or global set.

Approval is the decision, not a suggestion. The allowlist decides what runs
*without asking*; it is not a list of things you are forbidden to do. Say yes and
the command runs: `git clone`, `push --force`, `gh pr create`, `find -delete`, a
program that is not on the list at all. It is your machine and your repository.

A line with `|`, `&&`, `;` or `>` runs too once you approve it: it goes to
`sh -c` inside the same jail, with exactly the string you read on screen. It just
never runs on its own, since a shell is how a model would smuggle `curl | sh`
past a review that never happened.

One thing approval cannot change, because it is not a rule of ours: **writing
outside the working directory**. That is `bwrap`, the kernel. No answer at the
prompt widens it, with or without a shell.

Same for the resources a command may consume: every command runs under a
memory, process-count and CPU ceiling (`systemd-run --user --scope`), and
approval does not raise it either. A runaway command dies of its own weight
instead of eating the machine; if `systemd-run` is not installed, the output
says so with a `NOTE:` rather than pretending the ceiling is there.

Same for the syscalls a command may make: every command runs under a seccomp
filter that denies mounts, `unshare`/`setns`, module loading, `kexec`, the
kernel keyring, `bpf`, `userfaultfd` and `ptrace`. Approval does not lift it
either. It stops short of being airtight, and says so: `clone3` can still reach
a user namespace, because seccomp cannot inspect the arguments it passes behind
a pointer. The filter is x86_64-only; on another architecture the output says
so with a `NOTE:` instead of quietly leaving the layer out.

The approval prompt says when a command is destructive, which is the whole point
of it existing: the lever is yours, and you should see what you are pulling.

## Web access

Public web content is read through `fetch_url`, a general read-only tool for
pages, documentation, shared links and HTTP APIs. It accepts only HTTP(S),
rejects local, private and reserved destinations, ignores proxy configuration
and caps the download size.

Commands that run on their own are offline: nothing you were not shown reaches
the network. `curl` is off the default allowlist, so `fetch_url` remains the
normal way to read the web. Read-only `gh` views and searches are the separate
option for structured or authenticated GitHub access. `gh` mutations do not run
unasked, since they change a server the sandbox cannot roll back, but you can
approve them like anything else.

## Sessions

Every session is a JSONL log under `tool_harness/cli_sessions/`. On exit,
isaacli prints the exact `--resume` command on its own copyable line.

A resumed run restores the workspace, model, messages and tool results into a
*new* session log, and redraws recent messages, tool calls, results and
permission decisions in the terminal. The original JSONL is never modified.

New session IDs are full UUIDv4 values. Older date-based IDs, and logs written
before the internal field names were translated to English, remain resumable.

`/clear` resets only the conversation context. `/new` closes the current session
log and starts a fresh one with a new ID, recording the link between them.

## Terminal behaviour

The REPL runs on the main screen and clears it on start, so the shell that
launched isaacli is not left one wheel turn above the first message. From there
the conversation is plain output in the terminal's own scrollback: the wheel
scrolls it from the first message to the last, and the text stays selectable and
copyable. `/history` reprints the whole transcript the same way.

Arrow keys at the prompt navigate previously submitted messages, and only that.

Mouse reporting is never enabled: turning it on would break the terminal's own
text selection. The alternate screen is used by full-screen menus only, never by
the REPL: it has no scrollback, and terminals translate the wheel into arrow keys
while it is active.

While a response streams, one transient line reports approximate live
tokens/second, including hidden thinking emitted by Ollama. The reasoning text
itself stays hidden, unless a model puts its entire answer there and leaves the
visible response empty, in which case the text is recovered rather than lost.
After each response, Ollama's exact evaluation duration is used when available.

Keys received while the agent is busy are discarded instead of leaking escape
sequences into the output. Leaving a menu redraws the recent conversation
instead of returning to a cleared screen; leaving the CLI restores the shell
exactly as it was.

Assistant responses render common Markdown in the terminal (headings, emphasis,
inline and fenced code, lists, checkboxes, quotes and links), while session logs
keep the original text unchanged. Control sequences in model output are stripped
before they reach the terminal.

## Multiple sessions

Several isaacli sessions share one Ollama server started by the CLI. Closing one
session keeps that server available to the others; the last registered session
stops it. An Ollama server that was started outside isaacli belongs to you and
is never stopped automatically.
