"""Read geometry out of a GGUF file header, without loading the weights.

A model served by llama.cpp is a file on disk, not an entry a running server
reports. Everything the fit calculation needs (layer count, KV head count, head
dimension, trained context) is declared in the header, so the answer to "does
this fit in my GPU" is one bounded read away and needs no network and no server.

Every read here is bounded by the size of the file being read. A truncated or
hostile file must fail with a reason, never make this allocate a length it just
read out of that same file.
"""
import struct
from pathlib import Path

MAGIC = b"GGUF"
# Versions 1 and 2 exist but differ in how counts are encoded, and no shipping
# quantization has produced them for years. Refusing by name beats silently
# misreading a header with the wrong struct layout.
SUPPORTED_VERSIONS = (3,)

# Fixed-width metadata value types, as the GGUF specification numbers them.
_SCALAR_TYPES = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_STRING_TYPE = 8
_ARRAY_TYPE = 9

# The tokenizer vocabulary is the one array in the header big enough to matter,
# it is hundreds of thousands of strings, and nothing here uses it. Skipping it
# by name turns a multi-second read into an instant one.
_SKIPPED_PREFIXES = ("tokenizer.",)
# The one tokenizer key that is not vocabulary and that decides whether a file
# can be served at all: llama-server --jinja renders the chat template out of
# the GGUF, and a file that carries none cannot format a conversation. Knowing
# that before offering the model is the difference between a screen that says
# so and a server that starts and answers nonsense.
_KEPT_KEYS = ("tokenizer.chat_template",)

# How much of the file to hold in memory while stepping over one of those
# arrays. This bounds nothing the user can observe: it is the read size that
# keeps the walk from costing a system call per token, and the walk itself is
# still bounded by the file size like every other read here.
_SKIP_BLOCK_BYTES = 1 << 20


class GGUFError(RuntimeError):
    """The file is not a GGUF header this program can read."""


class _Reader:
    """A file cursor that refuses any length the file itself cannot hold."""

    def __init__(self, handle, size):
        self.handle = handle
        self.size = size

    def take(self, count):
        if count < 0 or count > self.size:
            raise GGUFError(f"declared length {count} exceeds the file size")
        chunk = self.handle.read(count)
        if len(chunk) != count:
            raise GGUFError("header ends before the value it declared")
        return chunk

    def scalar(self, fmt, width):
        return struct.unpack(fmt, self.take(width))[0]

    def string(self):
        return self.take(self.scalar("<Q", 8)).decode("utf-8", "replace")

    def value(self, value_type):
        if value_type == _STRING_TYPE:
            return self.string()
        if value_type == _ARRAY_TYPE:
            element_type = self.scalar("<I", 4)
            count = self.scalar("<Q", 8)
            # One element occupies at least one byte, so a count past the file
            # size is a corrupt header rather than a very long array.
            if count > self.size:
                raise GGUFError(f"array of {count} elements exceeds the file size")
            return [self.value(element_type) for _ in range(count)]
        if value_type not in _SCALAR_TYPES:
            raise GGUFError(f"unknown metadata value type {value_type}")
        return self.scalar(*_SCALAR_TYPES[value_type])

    def jump(self, count):
        """Step over bytes without allocating them.

        The vocabulary is the bulk of a header, hundreds of thousands of
        strings, and materialising it to throw it away is what made reading ten
        files take seconds instead of milliseconds. The bound is still the file
        size, and the seek is checked against the end so a truncated array
        cannot pass as a skipped one.
        """
        if count < 0 or count > self.size:
            raise GGUFError(f"declared length {count} exceeds the file size")
        position = self.handle.seek(count, 1)
        if position > self.size:
            raise GGUFError("header ends before the value it declared")

    def skip_string_array(self, count):
        """Walk a string array in memory instead of one read call per element.

        A vocabulary is a length-prefixed string per token, so stepping over it
        means reading every one of those lengths. Doing that against the file
        object costs two calls per token and dominated the whole read; doing it
        against a block already in memory is arithmetic. The block is refilled
        from the file, so the walk is still bounded by the file itself.
        """
        block = b""
        offset = 0
        available = 0
        for _ in range(count):
            while offset + 8 > available:
                block = block[offset:]
                offset = 0
                more = self.handle.read(_SKIP_BLOCK_BYTES)
                if not more:
                    raise GGUFError("header ends before the value it declared")
                block += more
                available = len(block)
            length = int.from_bytes(block[offset:offset + 8], "little")
            if length > self.size:
                raise GGUFError(f"declared length {length} exceeds the file size")
            offset += 8
            if offset + length <= available:
                offset += length
                continue
            # The string runs past what is buffered, so give the remainder back
            # to the file cursor and let the next refill start after it.
            self.jump(length - (available - offset))
            block = b""
            offset = 0
            available = 0
        # Only the bytes actually walked were consumed; the rest of the last
        # block belongs to whatever key comes next.
        if available > offset:
            self.handle.seek(offset - available, 1)

    def skip_value(self, value_type):
        """Consume a value without building it, for keys nothing here reads."""
        if value_type == _STRING_TYPE:
            self.jump(self.scalar("<Q", 8))
            return
        if value_type == _ARRAY_TYPE:
            element_type = self.scalar("<I", 4)
            count = self.scalar("<Q", 8)
            if count > self.size:
                raise GGUFError(f"array of {count} elements exceeds the file size")
            if element_type in _SCALAR_TYPES:
                self.jump(_SCALAR_TYPES[element_type][1] * count)
                return
            if element_type == _STRING_TYPE:
                self.skip_string_array(count)
                return
            for _ in range(count):
                self.skip_value(element_type)
            return
        if value_type not in _SCALAR_TYPES:
            raise GGUFError(f"unknown metadata value type {value_type}")
        self.jump(_SCALAR_TYPES[value_type][1])


def read_metadata(path):
    """Return the header's metadata keys, minus the tokenizer vocabulary."""
    path = Path(path)
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            reader = _Reader(handle, size)
            if reader.take(4) != MAGIC:
                raise GGUFError("the file does not start with the GGUF magic")
            version = reader.scalar("<I", 4)
            if version not in SUPPORTED_VERSIONS:
                raise GGUFError(f"GGUF version {version} is not supported")
            tensor_count = reader.scalar("<Q", 8)
            kv_count = reader.scalar("<Q", 8)
            if kv_count > size:
                raise GGUFError(f"{kv_count} metadata keys exceed the file size")
            metadata = {}
            for _ in range(kv_count):
                key = reader.string()
                value_type = reader.scalar("<I", 4)
                if key.startswith(_SKIPPED_PREFIXES) and key not in _KEPT_KEYS:
                    reader.skip_value(value_type)
                    continue
                metadata[key] = reader.value(value_type)
    except GGUFError:
        raise
    except OSError as error:
        raise GGUFError(str(error)) from error
    except struct.error as error:
        raise GGUFError(f"truncated header: {error}") from error
    return {"version": version, "tensor_count": tensor_count, "metadata": metadata}


def _first_int(value):
    """Collapse a scalar-or-array metadata value to one positive integer."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, list):
        positive = [item for item in value
                    if isinstance(item, int) and not isinstance(item, bool)
                    and item > 0]
        # A hybrid model declares this per layer and writes 0 where the layer
        # keeps no cache. Every attention layer in such a file carries the same
        # head count, so the distinct positive value is the answer; more than
        # one would mean a geometry this cannot describe with a single number.
        if len(set(positive)) == 1:
            return positive[0]
    return None


def _attention_layers(value, block_count):
    """How many layers actually hold a KV cache.

    A dense model answers block_count. A hybrid model that alternates attention
    with convolution or recurrence declares head_count_kv per layer and writes 0
    for the layers that cache nothing; counting those as attention layers would
    overstate the cache, on this machine by three times for LFM2.
    """
    if isinstance(value, list):
        caching = sum(
            1 for item in value
            if isinstance(item, int) and not isinstance(item, bool) and item > 0
        )
        if caching:
            return caching
    return block_count


def geometry(path):
    """Describe one GGUF file the way the fit calculation needs it.

    Returns None for the keys it cannot establish rather than guessing: a model
    whose geometry is unreadable has to say so on screen, which is the honest
    thing to say about a file nobody can measure.
    """
    header = read_metadata(path)
    metadata = header["metadata"]
    architecture = metadata.get("general.architecture")
    if not isinstance(architecture, str) or not architecture:
        raise GGUFError("the header declares no architecture")

    def key(name):
        return metadata.get(f"{architecture}.{name}")

    block_count = _first_int(key("block_count"))
    head_count = _first_int(key("attention.head_count"))
    raw_kv_heads = key("attention.head_count_kv")
    n_kv_heads = _first_int(raw_kv_heads)
    if n_kv_heads is None and head_count is not None:
        # No separate KV head count means every head keeps its own cache.
        n_kv_heads = head_count
    embedding_length = _first_int(key("embedding_length"))

    # key_length is the dimension the cache actually stores. It is written only
    # when it differs from the even split, so the split is the fallback, not a
    # guess: a model with partial rope (phi3 here) still caches the full head.
    head_dim = _first_int(key("attention.key_length"))
    if head_dim is None and embedding_length and head_count:
        head_dim, remainder = divmod(embedding_length, head_count)
        if remainder:
            head_dim = None

    n_layers = (_attention_layers(raw_kv_heads, block_count)
                if block_count else None)

    return {
        "architecture": architecture,
        # The file name is what the user sees in their own folder, so it is the
        # name. general.name is kept beside it because it is not always a name:
        # LFM2.5 ships with a hex digest there, and showing that would make the
        # list unreadable for the one model the machine can actually run.
        "name": Path(path).stem,
        "declared_name": (metadata.get("general.name")
                          if isinstance(metadata.get("general.name"), str) else None),
        "n_layers": n_layers,
        "block_count": block_count,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "context_length": _first_int(key("context_length")),
        "expert_count": _first_int(key("expert_count")),
        "expert_used_count": _first_int(key("expert_used_count")),
        "chat_template": bool(metadata.get("tokenizer.chat_template")),
        "tensor_count": header["tensor_count"],
    }
