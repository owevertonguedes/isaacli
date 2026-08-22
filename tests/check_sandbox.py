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

        previous = {name: os.environ.get(name)
                    for name in ("HOME", "PATH", "XDG_CONFIG_HOME",
                                 "XDG_DATA_HOME", "XDG_STATE_HOME",
                                 "XDG_CACHE_HOME", "XDG_RUNTIME_DIR")}
        os.environ["HOME"] = str(home)
        for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
                     "XDG_CACHE_HOME", "XDG_RUNTIME_DIR"):
            os.environ.pop(name, None)
        os.environ["PATH"] = os.pathsep.join([
            str(prefix / "bin"),          # a real toolchain: must be mounted
            str(home),                    # the home itself: never
            str(home / ".ssh"),           # keys: never
            str(home / ".config"),        # an XDG base: never
            str(checkout),                # a project checkout: never
            str(workspace),               # already mounted, and writable
            str(home / "does_not_exist"),
        ])
        os.environ["ISAACLI_POLICY_FAKE_TOKEN"] = "sk-planted-token"
        os.environ["ISAACLI_POLICY_TOOL_HOME"] = str(prefix / "lib")
        try:
            binds, path_dirs, forwarded = execution._toolchain_mounts(workspace)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            os.environ.pop("ISAACLI_POLICY_FAKE_TOKEN", None)
            os.environ.pop("ISAACLI_POLICY_TOOL_HOME", None)

        mounted = {str(path) for path in binds}
        must_not_mount = {
            "the home itself": home,
            "the ssh directory": home / ".ssh",
            "an XDG base directory": home / ".config",
            "a project checkout": checkout,
            "the workspace, which is writable": workspace,
        }
        for description, path in must_not_mount.items():
            if str(path) in mounted:
                print(f"[LEAK  ] {description} was mounted: {path}")
                leaks += 1
            else:
                print(f"[ok    ] not mounted: {description}")

        for description, path in (("the toolchain bin on PATH", prefix / "bin"),
                                  ("the install tree behind its symlink",
                                   prefix / "lib")):
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
