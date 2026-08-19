"""Surface the exceptions the normal flow deliberately absorbs.

isaacli absorbs some exceptions on purpose: probing a server that is not up
yet, reading an error body that may not be readable, asking git for a key the
host may not have configured. Absorbing them is correct. Losing them is not:
a layer that disappears in silence is worse than no layer, because we stop
looking for it.

Nothing here changes control flow. With debug off every call is a no-op; with
debug on the traceback goes to stderr and the caller still returns exactly
what it returned before. Each site reports once per run, so a probe inside a
retry loop cannot bury the terminal in forty identical tracebacks.
"""
import os
import sys
import traceback

ENABLED = False
_reported = set()


def enable(active=True):
    """Turn reporting on. Also resets the once-per-site memory, so a test or a
    second run inside the same process starts from a clean slate."""
    global ENABLED
    ENABLED = bool(active)
    _reported.clear()


def enabled_from_environment():
    """ISAACLI_DEBUG covers the paths that run before argument parsing."""
    return os.environ.get("ISAACLI_DEBUG", "").strip().lower() in ("1", "true", "yes")


def swallowed(site):
    """Call from inside an except block, naming the site that absorbs it."""
    if not ENABLED or site in _reported:
        return
    _reported.add(site)
    print(f"\n--- isaacli --debug: exception absorbed at {site} ---",
          file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
