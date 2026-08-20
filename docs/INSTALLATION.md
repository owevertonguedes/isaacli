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

Both purge forms require their exact confirmation phrase. A live registered
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
