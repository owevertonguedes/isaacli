# Installation and removal lifecycle

This document is the operational contract for installing, launching and
removing isaacli. It describes current behaviour; old validation logs are
evidence, not a substitute for checking the live machine.

## Per-user installation

`./isaacli install` creates one symbolic link:

```text
~/.local/bin/isaacli -> <this checkout>/isaacli
```

The command requires no root privileges, is idempotent and refuses to replace
an existing file or a link owned by another checkout. The clone remains the
application: moving or deleting it leaves a broken link. The current recovery
for that case is to remove the broken link manually and install again from the
new checkout.

The launcher resolves its own symlink before locating `tool_harness`. It
preserves the caller's working directory and arguments. When `FLATPAK_ID` is
present and `flatpak-spawn` is available, it executes the resolved launcher once
on the host with `flatpak-spawn --host`; the host invocation no longer has the
Flatpak environment, so it cannot loop.

## Removal levels

| Command | Removes | Preserves |
| --- | --- | --- |
| `isaacli uninstall` | This checkout's link in `~/.local/bin` | Config, secrets, sessions, feedback, runtime state, Ollama, models and clone |
| `isaacli uninstall --purge` | Link and isaacli-owned local data | Ollama, models and clone |
| `isaacli uninstall --purge --ollama` | Everything above plus a recognised official-script Ollama installation | Clone |
| `isaacli uninstall --purge --kaggle` | Everything in purge plus a Kaggle CLI installed by isaacli and known Kaggle authentication files | Existing third-party Kaggle installations, remote kernels and clone |
| `isaacli uninstall --purge --llamacpp` | Everything in purge plus a llama.cpp installed by isaacli | A llama.cpp the user installed themselves, model weights, Ollama and clone |

## llama.cpp lifecycle

`isaacli setup` offers the local engine only after showing which upstream build it would fetch: the exact asset name, its backend, its size and the directory. A `llama-server` already on `PATH` belongs to the user and is used as it is, with no ownership record written for it.

An install this program performs downloads the published archive for this platform and backend, checks it against the `sha256` digest the release declares for that asset, refuses any archive member that names a path outside the target, unpacks it into `~/.local/share/isaacli/llama.cpp`, and only then runs `llama-server --list-devices`. The ownership record at `~/.config/isaacli/llamacpp-install.json`, mode `0600`, is written last: an install that could not prove it works records nothing and removes the directory it created.

The backend follows the hardware, read from `/sys/class/drm` PCI vendor ids rather than from `nvidia-smi` alone, which by construction only ever reports NVIDIA. Upstream publishes CUDA binaries for Windows only, so on Linux an NVIDIA card is served by the Vulkan build; AMD is offered ROCm, Intel SYCL, and the CPU build is always the last resort. macOS has one build per architecture with Metal inside it and therefore no backend menu.

The strong purge requires that record, refuses a path outside the directory isaacli installs into, refuses an executable the package manager owns, and refuses anything whose removal would need administrator rights, because needing `sudo` proves isaacli did not put it there. Each refusal names what it refused.

## Model weights

Weights live in two directories that are never one:

```text
~/.local/share/isaacli/models/downloaded/   fetched by isaacli
~/.local/share/isaacli/models/from-ollama/  symbolic links into Ollama's blob store
```

Models Ollama already downloaded are reused where they lie, by link. Nothing is copied and nothing inside Ollama's store is ever written, so removing isaacli leaves every model Ollama downloaded exactly as it was.

`--purge` removes the links, which costs nothing because a link is not the data. It deliberately does **not** delete downloaded weights: those are gigabytes somebody waited for. Their folder, count and total size are printed instead, and deleting them stays the user's decision.

## Kaggle CLI lifecycle

`isaacli kaggle` first checks for an existing `kaggle` command. An existing command belongs to the user and is used without creating an ownership record. When the command is missing, isaacli shows one plan and, after confirmation, creates an isolated virtual environment at `~/.local/share/isaacli/kaggle-cli`, installs the Python package there and links `~/.local/bin/kaggle`. Only after `kaggle --version` succeeds does it write `~/.config/isaacli/kaggle-install.json` with mode `0600`.

The strong Kaggle purge requires that record and fixed paths under the user's home. It refuses a package-managed executable, an altered launcher or authentication data unless the caller has passed through the explicit warning. It invokes no administrator command. The removal path must be tested with temporary `HOME` and `XDG_CONFIG_HOME`, never with the developer's real Kaggle installation or credentials.

All purge forms require their exact confirmation phrase. A live registered
isaacli session blocks purge before Ollama is touched. Repeating a completed
removal is supported.

Strong purge validates the launcher ownership and live-session state, removes
Ollama, and only then removes isaacli data. If recognition, custom storage,
`sudo`, `systemctl`, `rm`, `userdel` or `groupdel` fails, the clone and isaacli
data remain available for inspection and recovery. A failure after some system
commands may still leave a partial Ollama layout; use the reported command and
inspect the fixed paths before retrying from the clone.

## Recognised Ollama layouts

The official Linux script selects the first supported bin directory present in
`PATH`. isaacli recognises these pairs:

```text
/usr/local/bin/ollama  +  /usr/local/lib/ollama
/usr/bin/ollama        +  /usr/lib/ollama
```

The `/usr` layout is accepted only when both the official library and
`/etc/systemd/system/ollama.service` exist and neither RPM nor dpkg reports
ownership of the executable. Package-managed and unknown layouts are refused;
they must be removed with their package manager.

Only fixed, recognised paths are passed to privileged removal commands:

```text
/usr/local/bin/ollama or /usr/bin/ollama
/usr/local/lib/ollama or /usr/lib/ollama
/etc/systemd/system/ollama.service
/usr/share/ollama
~/.ollama
```

An `OLLAMA_MODELS` path found in the environment, service unit or systemd
drop-in that falls outside the covered default directories blocks automatic
removal and is shown to the user. Unknown storage that appears nowhere in those
sources cannot be discovered, so the program must not broaden privileged
deletion to guessed or environment-derived paths.

## Isolated validation

Never exercise purge against the developer's real HOME, clone or Ollama. Run:

```bash
tests/integration/test-install-lifecycle.sh
```

The script builds a Fedora container from `git archive HEAD` plus the candidate
diff, mounts no host data, uses a non-root test user and real systemd, plants
fixed-path bait, injects partial failures and destroys the container/image in a
trap. It fingerprints the real host paths before and after the run. The regular
`tests/check_*.py` suite remains required; integration and unit checks cover
different failures.

See [SECURITY.md](SECURITY.md) for data handling and privacy boundaries, and
[ARCHITECTURE.md](ARCHITECTURE.md) for the module-level invariants.
