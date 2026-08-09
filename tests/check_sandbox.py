"""Confinement of _safe(): with the root selectable in the app, this is the ONLY protection.

Run it after ANY change to tools.py or to the app's folder selector:

    python3 check_sandbox.py

It tests ../, absolute paths and symlinks, including after switching the root
at runtime (which is what the app's folder selector does).
"""
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

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


def main():
    leaks = 0
    with tempfile.TemporaryDirectory() as tmp:
        # the default root and a root swapped at runtime (what the app's selector does)
        leaks += check_root(Path(tmp) / "root_a" / "sandbox")
        leaks += check_root(Path(tmp) / "root_b" / "project")
    print(f"\n{'SANDBOX CONFINED' if leaks == 0 else f'{leaks} LEAK(S), do not enable the folder selector'}")
    sys.exit(0 if leaks == 0 else 1)


if __name__ == "__main__":
    main()
