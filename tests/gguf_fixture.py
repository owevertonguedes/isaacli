"""A GGUF file written by the checks, so no test needs anybody's real weights.

Shared rather than copied: two checks need a readable weight on disk, and a
second hand-written header would agree with this one today and drift apart at
the first field the reader learns to want. Importing the check that used to own
it would have run that whole check as a side effect, including its exit code.
"""
import struct
from pathlib import Path

UINT32, UINT64, STRING, ARRAY = 4, 10, 8, 9


def _string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key, value_type, payload):
    return _string(key) + struct.pack("<I", value_type) + payload


def write_gguf(path, keys, version=3, tensor_count=0, body=b"", magic=b"GGUF"):
    """Write a GGUF header. `keys` maps a key to (type, packed payload)."""
    out = bytearray(magic)
    out += struct.pack("<I", version)
    out += struct.pack("<QQ", tensor_count, len(keys))
    for key, (value_type, payload) in keys.items():
        out += _kv(key, value_type, payload)
    out += body
    Path(path).write_bytes(bytes(out))
    return Path(path)


def dense_keys(architecture="llama", layers=28, kv_heads=8, heads=32,
               embedding=4096, context=131072, template=True, vocabulary=64):
    keys = {
        "general.architecture": (STRING, _string(architecture)),
        "general.name": (STRING, _string("Fixture Model")),
        f"{architecture}.block_count": (UINT32, struct.pack("<I", layers)),
        f"{architecture}.attention.head_count": (UINT32, struct.pack("<I", heads)),
        f"{architecture}.attention.head_count_kv": (UINT32, struct.pack("<I", kv_heads)),
        f"{architecture}.embedding_length": (UINT32, struct.pack("<I", embedding)),
        f"{architecture}.context_length": (UINT32, struct.pack("<I", context)),
    }
    if vocabulary:
        # A real vocabulary is the bulk of a header and the one thing the
        # reader steps over in blocks rather than building. Putting one in the
        # fixture is what exercises that walk.
        tokens = b"".join(_string(f"token{index}") for index in range(vocabulary))
        keys["tokenizer.ggml.tokens"] = (
            ARRAY, struct.pack("<I", STRING) + struct.pack("<Q", vocabulary) + tokens)
        keys["tokenizer.ggml.token_type"] = (
            ARRAY, struct.pack("<I", UINT32) + struct.pack("<Q", vocabulary)
            + struct.pack("<I", 1) * vocabulary)
    if template:
        keys["tokenizer.chat_template"] = (STRING, _string("{{ messages }}"))
    return keys
