"""Confinement of _safe(): with the root selectable in the app, this is the ONLY protection.

Run it after ANY change to tools.py or to the app's folder selector:

    python3 check_sandbox.py

It tests ../, absolute paths and symlinks, including after switching the root
at runtime (which is what the app's folder selector does).
"""
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

import execution
import tools


def expect_block(description, fn):
    try:
        fn()
    except ValueError:
        print(f"[ok    ] {description}: blocked")
        return True
    print(f"[LEAK  ] {description}: ESCAPED THE SANDBOX")
    return False


def check_root(root: Path) -> int:
    tools.SANDBOX_ROOT = root
    root.mkdir(parents=True, exist_ok=True)
    leaks = 0

    outside = root.parent / "outside_the_sandbox.txt"
    outside.write_text("secret")

    if not expect_block(f"{root.name}: ../",
                        lambda: tools._safe("../outside_the_sandbox.txt")):
        leaks += 1
    # An absolute path does NOT raise: it is neutralised (lstrip('/') treats it
    # as relative). The contract is that the result always lands INSIDE the root.
    for absolute in (str(outside), "/etc/passwd"):
        p = tools._safe(absolute)
        if tools.SANDBOX_ROOT.resolve() not in p.parents:
            print(f"[LEAK  ] {root.name}: absolute {absolute} resolved outside: {p}")
            leaks += 1
        else:
            print(f"[ok    ] {root.name}: absolute {absolute} becomes an internal path")

    link = root / "shortcut"
    link.unlink(missing_ok=True)
    link.symlink_to(outside.parent)
    if not expect_block(f"{root.name}: symlink pointing out",
                        lambda: tools._safe("shortcut/outside_the_sandbox.txt")):
        leaks += 1
    link.unlink()
    outside.unlink()
    return leaks


def check_mount_policy() -> int:
    """What the jail agrees to mount read-only, decided against a fake home.

    `check_execution.py` proves the mounts by running commands inside the real
    jail, and that check cannot run on a hosted runner, where bwrap cannot map
    uids. This one needs no bwrap: it drives the PLANNER with a home built for
    the purpose, so the policy is still verified everywhere the suite runs. A
    directory that should never be mounted and is counts as a leak here, exactly
    like a path escaping the root above.

    The LOGIN SHELL is part of the fixture too (task 052), and not incidentally:
    since the planner reads it, leaving the real one in place would drive this
    check with whatever this machine's dotfiles happen to say, and the answer
    would differ per machine. Planting it also buys the case that matters most
    about the snapshot: the second PATH must obey exactly the same refusals as
    the first, or the fix for 052 would have widened the jail.
    """
    print("\n--- what the sandbox agrees to mount read-only ---")
    leaks = 0
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "fakehome"
        workspace = Path(tmp) / "workspace"
        workspace.mkdir(parents=True)
        (home / ".ssh").mkdir(parents=True)
        (home / ".ssh" / "id_rsa").write_text("PRIVATE KEY")
        (home / ".config" / "isaacli").mkdir(parents=True)
        (home / ".config" / "isaacli" / "secrets.json").write_text("{}")
        (home / ".local" / "share").mkdir(parents=True)
        prefix = home / ".local" / "share" / "toolchain"
        (prefix / "bin").mkdir(parents=True)
        (prefix / "lib").mkdir()
        (prefix / "lib" / "real-tool").write_text("#!/bin/sh\ntrue\n")
        (prefix / "lib" / "real-tool").chmod(0o755)
        (prefix / "bin" / "tool").symlink_to("../lib/real-tool")
        checkout = home / "project"
        (checkout / ".git").mkdir(parents=True)
        (checkout / "bin").mkdir()
        # An executable directly in the checkout, for the same reason the stray
        # script below sits in the home: without it the checkout is refused for
        # holding no executable, and the guard that this check exists to prove --
        # "a project is not a toolchain" -- is never the thing that answered.
        # Measured on 2026-08-23 by removing that guard for directories named by
        # the login shell: the check stayed green, because the wrong refusal
        # covered for the missing one.
        (checkout / "build.sh").write_text("#!/bin/sh\ntrue\n")
        (checkout / "build.sh").chmod(0o755)
        # An executable loose in the home, so the home and the symlink to it are
        # refused by the guard being tested and not by "it holds no executable",
        # which would make both of those cases pass for the wrong reason.
        (home / "stray-script").write_text("#!/bin/sh\ntrue\n")
        (home / "stray-script").chmod(0o755)
        # A PATH entry that is a SYMLINK to the home. Every comparison-based
        # refusal passes for it, because the link is not equal to the home, so
        # the only thing that stops it is resolving before deciding.
        disguised = home / "looks_like_a_bin_dir"
        disguised.symlink_to(home)

        # Only the login shell knows about this one, and it is a real toolchain
        # directory: it has to be mounted, or a user who started isaacli from a
        # launcher loses it.
        snapshot_prefix = home / ".local" / "share" / "snapshot-toolchain"
        (snapshot_prefix / "bin").mkdir(parents=True)
        (snapshot_prefix / "bin" / "snapshot-tool").write_text("#!/bin/sh\ntrue\n")
        (snapshot_prefix / "bin" / "snapshot-tool").chmod(0o755)
        # A checkout that ONLY the login shell names. It cannot be the one
        # already on the process PATH: identical entries are deduplicated, so
        # that one would never reach the policy through the snapshot at all, and
        # the case would pass without ever exercising what it claims to.
        snapshot_checkout = home / "snapshot-project"
        (snapshot_checkout / ".git").mkdir(parents=True)
        (snapshot_checkout / "build.sh").write_text("#!/bin/sh\ntrue\n")
        (snapshot_checkout / "build.sh").chmod(0o755)
        # And a symlink to the home that only the login shell names: the guard
        # that stops it is resolving before deciding, and `home` itself is
        # already on the process PATH, so reusing it would be deduplicated away.
        snapshot_disguised = home / "snapshot_looks_like_a_bin_dir"
        snapshot_disguised.symlink_to(home)
        # A directory that a RELATIVE PATH entry would reach from the working
        # directory this check runs in. It holds an executable, so nothing else
        # in the policy would refuse it: only "not an absolute path" can.
        relative_entry = Path(tmp) / "relative-bin"
        relative_entry.mkdir()
        (relative_entry / "relative-tool").write_text("#!/bin/sh\ntrue\n")
        (relative_entry / "relative-tool").chmod(0o755)
        login_shell = Path(tmp) / "login-shell"
        login_shell.write_text(
            "#!/bin/sh\n"
            "echo 'a banner some dotfile prints'\n"
            f"printf '\\n{execution.LOGIN_SHELL_MARKER}%s\\n' "
            f"'{os.pathsep.join([str(snapshot_prefix / 'bin'), str(snapshot_checkout), str(snapshot_disguised)])}'\n")
        login_shell.chmod(0o755)

        previous = {name: os.environ.get(name)
                    for name in ("HOME", "PATH", "SHELL", "XDG_CONFIG_HOME",
                                 "XDG_DATA_HOME", "XDG_STATE_HOME",
                                 "XDG_CACHE_HOME", "XDG_RUNTIME_DIR")}
        os.environ["HOME"] = str(home)
        os.environ["SHELL"] = str(login_shell)
        execution.reset_path_snapshot()
        for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
                     "XDG_CACHE_HOME", "XDG_RUNTIME_DIR"):
            os.environ.pop(name, None)
        os.environ["PATH"] = os.pathsep.join([
            str(prefix / "bin"),          # a real toolchain: must be mounted
            str(home),                    # the home itself: never
            str(home / ".ssh"),           # keys: never
            str(home / ".config"),        # an XDG base: never
            str(checkout),                # a project checkout: never
            str(disguised),               # a symlink to the home: never
            str(workspace),               # already mounted, and writable
            str(home / "does_not_exist"),
            relative_entry.name,          # a RELATIVE entry: never
        ])
        os.environ["ISAACLI_POLICY_FAKE_TOKEN"] = "sk-planted-token"
        os.environ["ISAACLI_POLICY_TOOL_HOME"] = str(prefix / "lib")
        # A relative PATH entry means a directory under whatever isaacli is
        # running in, and `os.path.realpath` would make it absolute before any
        # guard could object. Proved from a known working directory rather than
        # from the repository's, so the answer does not depend on where the
        # suite was started.
        previous_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            binds, path_dirs, forwarded = execution._toolchain_mounts(workspace)
        finally:
            os.chdir(previous_cwd)
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            os.environ.pop("ISAACLI_POLICY_FAKE_TOKEN", None)
            os.environ.pop("ISAACLI_POLICY_TOOL_HOME", None)
            # The snapshot is session-wide on purpose, so it has to be dropped
            # here or every later check in this process would keep deciding
            # against the fake home this block just took apart.
            execution.reset_path_snapshot()

        mounted = {str(path) for path in binds}
        must_not_mount = {
            "the home itself": home,
            "the ssh directory": home / ".ssh",
            "an XDG base directory": home / ".config",
            "a project checkout": checkout,
            "a symlink pointing at the home": disguised,
            "the workspace, which is writable": workspace,
            "a directory reached by a relative PATH entry": relative_entry,
        }
        for description, path in must_not_mount.items():
            if str(path) in mounted:
                print(f"[LEAK  ] {description} was mounted: {path}")
                leaks += 1
            else:
                print(f"[ok    ] not mounted: {description}")

        # The snapshot proposes these two as well, and the policy does not care
        # which PATH proposed a directory: a second source of directories that
        # skipped the refusals would be a wider jail, not a fixed one.
        for description, path in (("a project checkout, named ONLY by the login "
                                   "shell", snapshot_checkout),
                                  ("a symlink to the home, named ONLY by "
                                   "the login shell", snapshot_disguised)):
            if str(path) in mounted:
                print(f"[LEAK  ] {description} was mounted, so the login shell "
                      f"PATH skipped the mount policy: {path}")
                leaks += 1
            else:
                print(f"[ok    ] not mounted: {description}")

        for description, path in (("the toolchain bin on PATH", prefix / "bin"),
                                  ("the install tree behind its symlink",
                                   prefix / "lib"),
                                  ("a toolchain only the login shell knows about",
                                   snapshot_prefix / "bin")):
            if str(path) in mounted:
                print(f"[ok    ] mounted: {description}")
            else:
                print(f"[LEAK  ] {description} was NOT mounted, so the toolchain "
                      f"is unreachable: {path}")
                leaks += 1

        if str(prefix / "bin") not in path_dirs:
            print("[LEAK  ] the toolchain directory did not reach the jail's PATH")
            leaks += 1
        if "ISAACLI_POLICY_FAKE_TOKEN" in forwarded:
            print("[LEAK  ] a secret in the environment was forwarded into the jail")
            leaks += 1
        else:
            print("[ok    ] a secret in the environment is not forwarded")
        if forwarded.get("ISAACLI_POLICY_TOOL_HOME") != str(prefix / "lib"):
            print("[LEAK  ] a variable naming a mounted directory was dropped, so "
                  "the toolchain would not find its own files")
            leaks += 1
    return leaks


def main():
    leaks = 0
    leaks += check_mount_policy()
    with tempfile.TemporaryDirectory() as tmp:
        # the default root and a root swapped at runtime (what the app's selector does)
        leaks += check_root(Path(tmp) / "root_a" / "sandbox")
        leaks += check_root(Path(tmp) / "root_b" / "project")
    print(f"\n{'SANDBOX CONFINED' if leaks == 0 else f'{leaks} LEAK(S), do not enable the folder selector'}")
    sys.exit(0 if leaks == 0 else 1)


if __name__ == "__main__":
    main()
