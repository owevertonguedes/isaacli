"""Local and web-reading tools the model can call.

Files are confined to SANDBOX_ROOT. Web reading accepts only public HTTP(S),
with no cookies, credentials, proxies or access to the local network.
"""
import ipaddress
import difflib
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import debug
from itertools import islice
from html.parser import HTMLParser
from pathlib import Path

# The agent's working directory. Overridable by env so it can operate on a real
# project instead of the test sandbox. It stays confined, only the root changes.
SANDBOX_ROOT = Path(os.environ.get("ISAACLI_ROOT", Path(__file__).parent / "sandbox"))
MAX_WEB_BYTES = 80_000
MAX_MUTATION_DIFF_LINES = 80
MAX_MUTATION_DIFF_BYTES = 6_000
# The diff output is bounded, but bounding only the output still costs the
# memory of building it: two full line lists plus difflib's quadratic matcher.
# On a large file that peaks at several times the file size on the user's
# machine, to produce evidence that gets truncated anyway. So the *input* is
# bounded too, and past this size the byte counts alone are the evidence.
MAX_DIFF_INPUT_BYTES = 1_000_000
# read_file exists to put content in front of a model whose context is far
# smaller than this. Reading more only spends the user's memory and fills the
# session log; the cut is explicit so nobody mistakes a partial read for a
# whole file.
MAX_READ_BYTES = 200_000


def _safe(path: str) -> Path:
    p = (SANDBOX_ROOT / path.lstrip("/")).resolve()
    root = SANDBOX_ROOT.resolve()
    if not (p == root or root in p.parents):
        raise ValueError(f"path outside the sandbox: {path}")
    return p


def read_file(path: str) -> str:
    p = _safe(path)
    if not p.is_file():
        return f"ERROR: file does not exist: {path}"
    size = p.stat().st_size
    if size <= MAX_READ_BYTES:
        return p.read_text()
    with p.open("rb") as f:
        head = f.read(MAX_READ_BYTES)
    return (head.decode("utf-8", errors="ignore")
            + f"\n\n... FILE TRUNCATED BY ISAACLI LIMITS: showing the first "
              f"{MAX_READ_BYTES} of {size} bytes ...")


def write_file(path: str, content: str) -> str:
    p = _safe(path)
    existed = p.is_file()
    before_bytes = p.stat().st_size if existed else 0
    # Do not read the old content at all when it is too large to diff: reading
    # it would spend the memory this limit exists to protect. None means
    # "not read", which is not the same as a file that was empty or absent.
    before = _readable_before(p, existed, before_bytes)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = _unescape(content)
    p.write_text(text)
    return _mutation_result(
        f"OK: wrote {_byte_len(text)} bytes to {path}", path, before, text,
        existed_before=existed, before_bytes=before_bytes,
    )


def list_dir(path: str = ".") -> str:
    p = _safe(path)
    if not p.is_dir():
        return f"ERROR: not a directory: {path}"
    items = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
    return "\n".join(items) if items else "(empty)"


def append_file(path: str, content: str) -> str:
    """Append at the end without the model having to reproduce what is already there.

    A small model gets it wrong when rewriting a whole file (it guesses content,
    escapes \\n badly). Giving it a tool that does not require reproducing the
    content eliminates the entire class of error, instead of trying to fix the
    model.
    """
    p = _safe(path)
    existed = p.is_file()
    before_bytes = p.stat().st_size if existed else 0
    diffable = before_bytes <= MAX_DIFF_INPUT_BYTES
    before = _readable_before(p, existed, before_bytes)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = _unescape(content)
    if not text.endswith("\n"):
        text += "\n"
    with p.open("a") as f:
        f.write(text)
    # `before + text` is a whole extra copy of the file. Only pay for it when
    # the file is small enough for that copy to be worth a diff.
    after = (before or "") + text if diffable else None
    return _mutation_result(
        f"OK: appended {_byte_len(text)} bytes to the end of {path}",
        path, before, after, existed_before=existed,
        before_bytes=before_bytes, after_bytes=before_bytes + _byte_len(text),
    )


def replace_between(path: str, start_marker: str, end_marker: str, content: str) -> str:
    """Replace the section between two markers that already exist in the file.

    This is scaffolding for a small model: instead of asking for a whole file
    again on every requirement, the agent swaps only the part it is working on.
    """
    start_marker = _normalize_marker(start_marker)
    end_marker = _normalize_marker(end_marker)
    p = _safe(path)
    if not p.is_file():
        return f"ERROR: file does not exist: {path}"
    start_family = re.fullmatch(r"(.+)_START", start_marker)
    end_family = re.fullmatch(r"(.+)_END", end_marker)
    if start_family and end_family and start_family.group(1) != end_family.group(1):
        return (
            "ERROR: mismatched markers. "
            f"{start_marker} must close with {start_family.group(1)}_END, "
            f"not with {end_marker}. Do one swap at a time.")
    text = p.read_text()
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start < 0:
        return f"ERROR: start marker not found: {start_marker}"
    if end < 0:
        return f"ERROR: end marker not found: {end_marker}"
    if end <= start:
        return "ERROR: the end marker appears before the start marker"

    section = _unescape(content)
    # The markers sit on comment lines, for example:
    #   <!-- SECTION_START -->
    #   /* SECTION_START */
    # Replacing from the marker text itself put the code INSIDE the comment and
    # broke the file. The correct swap preserves the marker lines and changes
    # only the body between them.
    content_start = text.find("\n", start)
    if content_start < 0:
        content_start = start + len(start_marker)
    else:
        content_start += 1
    content_end = text.rfind("\n", 0, end)
    if content_end >= 0:
        content_end += 1
    if content_end < content_start:
        content_end = end
    # A model often echoes the marker lines back inside the content it sends.
    # Keeping them would duplicate the markers and break the next replacement.
    body_lines = [
        line for line in section.splitlines()
        if not (_normalize_marker(line.strip()) in (start_marker, end_marker))
    ]
    body = "\n".join(body_lines).strip()
    if body:
        body = body + "\n"
    updated = text[:content_start] + body + text[content_end:]
    p.write_text(updated)
    summary = (
        f"OK: replaced {_byte_len(section)} bytes between {start_marker} and "
        f"{end_marker} in {path}")
    return _mutation_result(summary, path, text, updated, existed_before=True)


def replace_text(path: str, old_text: str, new_text: str) -> str:
    """Replace one unambiguous exact string and preserve everything else."""
    p = _safe(path)
    if not p.is_file():
        return f"ERROR: file does not exist: {path}"
    old = _unescape(old_text)
    new = _unescape(new_text)
    if not old:
        return "ERROR: old_text must not be empty"
    text = p.read_text()
    count = text.count(old)
    if count == 0:
        return "ERROR: old_text was not found; the file was not modified"
    if count > 1:
        return (
            f"ERROR: old_text appears {count} times; the file was not modified. "
            "Provide a longer unique old_text"
        )
    if old == new:
        return "ERROR: old_text and new_text are identical; the file was not modified"
    updated = text.replace(old, new, 1)
    p.write_text(updated)
    return _mutation_result(
        f"OK: replaced one exact occurrence in {path}", path, text, updated,
        existed_before=True,
    )


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _readable_before(p: Path, existed: bool, before_bytes: int):
    """The previous content, or None when it is too large to hold in memory.

    An absent file is "" (there was nothing, and that is knowable), never None
    (there is something, and we chose not to read it)."""
    if not existed:
        return ""
    return p.read_text() if before_bytes <= MAX_DIFF_INPUT_BYTES else None


def _mutation_result(summary: str, path: str, before, after,
                     existed_before: bool, before_bytes=None,
                     after_bytes=None) -> str:
    """Attach bounded factual evidence for the model's post-tool review.

    `before`/`after` are None when the file was too large to hold in memory for
    a diff. The byte counts are still exact, so the evidence stays truthful; it
    just says less.
    """
    if before_bytes is None:
        before_bytes = _byte_len(before or "")
    if after_bytes is None:
        after_bytes = _byte_len(after or "")
    oversized = (before is None or after is None
                 or max(before_bytes, after_bytes) > MAX_DIFF_INPUT_BYTES)
    if oversized:
        diff = ("(file too large for a line-level diff; the byte counts above "
                "are the evidence for this change)")
        return _mutation_evidence(summary, diff, existed_before,
                                  before_bytes, after_bytes)
    # islice, not list: a bounded read of the generator never materialises the
    # whole diff. One extra line is enough to know it was cut.
    lines = list(islice(difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
    ), MAX_MUTATION_DIFF_LINES + 1))
    truncated = len(lines) > MAX_MUTATION_DIFF_LINES
    lines = lines[:MAX_MUTATION_DIFF_LINES]
    diff = "\n".join(lines)
    encoded = diff.encode("utf-8")
    if len(encoded) > MAX_MUTATION_DIFF_BYTES:
        truncated = True
    if truncated:
        marker = "\n... DIFF TRUNCATED BY ISAACLI LIMITS ..."
        budget = MAX_MUTATION_DIFF_BYTES - len(marker.encode("utf-8"))
        diff = encoded[:max(0, budget)].decode("utf-8", errors="ignore") + marker
    if not diff:
        diff = "(no line-level textual difference)"
    return _mutation_evidence(summary, diff, existed_before,
                              before_bytes, after_bytes)


def _mutation_evidence(summary, diff, existed_before, before_bytes,
                       after_bytes) -> str:
    state = "updated existing file" if existed_before else "created new file"
    return (
        f"{summary}\n\n"
        "CHANGE EVIDENCE (objective result; does not prove task completion):\n"
        f"State: {state}; before={before_bytes} bytes; "
        f"after={after_bytes} bytes\n"
        f"{diff}"
    )


def _normalize_marker(marker: str) -> str:
    """Accept a bare marker or one wrapped in the comment where it appears."""
    m = str(marker or "").strip()
    for prefix, suffix in (("<!--", "-->"), ("/*", "*/")):
        if m.startswith(prefix) and m.endswith(suffix):
            return m[len(prefix):-len(suffix)].strip()
    return m


def _unescape(s: str) -> str:
    r"""A small model emits a literal '\n' (2 chars) thinking it is a line break.

    It only converts when there is NO real line break in the text, so it
    does not damage content that legitimately contains a backslash.
    """
    if "\n" not in s and "\\n" in s:
        return s.replace("\\n", "\n").replace("\\t", "\t")
    return s


def _normalize_web_url(url: str) -> str:
    url = str(url or "").strip()
    if len(url) > 2048:
        raise ValueError("URL is too long")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("use a full URL starting with http:// or https://")
    if parts.username or parts.password:
        raise ValueError("a URL with a username or password is not allowed")
    try:
        _ = parts.port
    except ValueError as e:
        raise ValueError("invalid port in the URL") from e

    issue = re.fullmatch(r"/([^/]+)/([^/]+)/issues/(\d+)/?", parts.path)
    if parts.hostname.casefold() == "github.com" and issue:
        owner, repo, number = issue.groups()
        return f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
    return urllib.parse.urlunsplit(parts)


def _validate_web_target(url: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("redirect to a protocol that is not allowed")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        targets = socket.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"could not resolve {parts.hostname}: {e}") from e
    ips = {ipaddress.ip_address(item[4][0]) for item in targets}
    if not ips or any(not ip.is_global for ip in ips):
        raise ValueError("the web tool does not reach localhost or private/reserved networks")


class _SafeWebRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urljoin(req.full_url, newurl)
        _validate_web_target(target)
        return super().redirect_request(req, fp, code, msg, headers, target)


class _HTMLExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "svg", "noscript"):
            self.hidden += 1
        elif not self.hidden and tag in ("p", "div", "br", "li", "h1", "h2", "h3", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "svg", "noscript") and self.hidden:
            self.hidden -= 1
        elif not self.hidden and tag in ("p", "div", "li", "h1", "h2", "h3", "tr"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

    def text(self):
        lines = (re.sub(r"[ \t]+", " ", line).strip()
                 for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)


def fetch_url(url: str) -> str:
    """Read public content without granting network access to the confined terminal."""
    target = _normalize_web_url(url)
    _validate_web_target(target)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _SafeWebRedirect(),
    )
    request = urllib.request.Request(
        target,
        headers={
            "User-Agent": "IsaacCLI/0.1 (+read-only fetch_url)",
            "Accept": "text/html, application/json, text/plain, application/xml;q=0.9",
        },
    )
    try:
        with opener.open(request, timeout=20) as response:
            data = response.read(MAX_WEB_BYTES + 1)
            final = response.geturl()
            kind = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as e:
        return f"HTTP ERROR {e.code} while reading {target}: {e.reason}"
    except urllib.error.URLError as e:
        return f"NETWORK ERROR while reading {target}: {e.reason}"

    if not (kind.startswith("text/") or kind in (
            "application/json", "application/xml", "application/xhtml+xml")):
        return f"ERROR: non-textual content refused ({kind})"
    truncated = len(data) > MAX_WEB_BYTES
    text = data[:MAX_WEB_BYTES].decode(charset, errors="replace")
    if kind in ("text/html", "application/xhtml+xml"):
        parser = _HTMLExtractor()
        parser.feed(text)
        text = parser.text()
    header = f"Final URL: {final}\nType: {kind}\n"
    suffix = "\n… content truncated by the tool limit" if truncated else ""
    return header + text + suffix


def run_command(cmd: str) -> str:
    """Run a confined command. The whole containment lives in execution.py.

    Late import on purpose: `tools` is imported by everyone here (tests, batch
    scripts), and `execution` only makes sense when someone is actually going to
    run a command.
    """
    import execution
    return execution.run_command(cmd)


IMPLS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "append_file": append_file,
    "replace_between": replace_between,
    "replace_text": replace_text,
    "fetch_url": fetch_url,
    "run_command": run_command,
}

# OpenAI-shaped schema: this is what goes in the `tools` field of the API call.
SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file and return its content as a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "relative path of the file"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (or overwrite) a text file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "relative path of the file"},
                    "content": {"type": "string", "description": "full content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": (
                "Append text to the END of a file, preserving what is already there. "
                "Use this instead of write_file when you are only adding something."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "relative path of the file"},
                    "content": {"type": "string", "description": "text to append at the end"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_between",
            "description": (
                "Replace only the content between two textual markers that already "
                "exist in the file. Use it to change a small section without "
                "rewriting the whole file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "relative path of the file"},
                    "start_marker": {"type": "string", "description": "literal start marker"},
                    "end_marker": {"type": "string", "description": "literal end marker"},
                    "content": {"type": "string", "description": "new content for the section"},
                },
                "required": ["path", "start_marker", "end_marker", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_text",
            "description": (
                "Replace one exact, unique old string with new text while preserving "
                "everything else. It refuses to modify the file if the old text is "
                "absent or appears more than once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "relative path of the file"},
                    "old_text": {"type": "string", "description": "exact unique text to replace"},
                    "new_text": {"type": "string", "description": "replacement text"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "General tool for reading textual content from a public HTTP(S) URL: "
                "pages, documentation, shared links and APIs. Use it whenever you need "
                "to consult the public web; do not try curl in the terminal. It does "
                "not reach private networks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "full public URL"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the files and folders of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "relative path of the directory"}
                },
                "required": [],
            },
        },
    },
]


def _attach_command_schema():
    """Attach the run_command schema without duplicating its description here.

    The description lives next to the rules it describes (execution.py). Copying
    it here would create two versions of the truth, and the one the model READS
    would be the copy, the classic way for the allowlist to change while the
    model keeps believing the old one.
    """
    try:
        import execution
    except ImportError:
        return  # without the module, the tool simply does not exist for the model
    SCHEMA.append(execution.SCHEMA)


_attach_command_schema()


def filtered_schema(names):
    """Return only the tools needed for the current task."""
    allowed = set(names)
    return [s for s in SCHEMA if s["function"]["name"] in allowed]


def _function_schema(name):
    for item in SCHEMA:
        if item["function"]["name"] == name:
            return item["function"].get("parameters") or {}
    return {}


def _argument_error(name, args):
    """Return a precise schema error before Python dispatch obscures the fix."""
    schema = _function_schema(name)
    required = schema.get("required") or []
    missing = [key for key in required if key not in args]
    if missing:
        return (
            f"ERROR: wrong arguments for {name}: missing required argument(s): "
            f"{', '.join(missing)}. Required arguments: {', '.join(required)}"
        )
    allowed = set((schema.get("properties") or {}).keys())
    unexpected = [key for key in args if key not in allowed]
    if unexpected:
        return (
            f"ERROR: wrong arguments for {name}: unexpected argument(s): "
            f"{', '.join(unexpected)}. Allowed arguments: {', '.join(sorted(allowed))}"
        )
    return None


def execute(name: str, args_json: str) -> str:
    """Run the tool the model asked for and return the result as text."""
    if name not in IMPLS:
        return f"ERROR: unknown tool '{name}'. Available: {list(IMPLS)}"
    try:
        args = json.loads(args_json) if isinstance(args_json, str) else (args_json or {})
    except json.JSONDecodeError as e:
        return f"ERROR: arguments are not valid JSON ({e}): {args_json}"
    if not isinstance(args, dict):
        return f"ERROR: wrong arguments for {name}: expected a JSON object"
    error = _argument_error(name, args)
    if error:
        return error
    try:
        return IMPLS[name](**args)
    except TypeError as e:
        return f"ERROR: wrong arguments for {name}: {e}"
    except Exception as e:
        debug.swallowed(f"tools.execute({name})")
        return f"ERROR while running {name}: {e}"
