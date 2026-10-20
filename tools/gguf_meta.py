#!/usr/bin/env python3
"""Dump a GGUF file's metadata keys without reading any weight bytes.

The header is all this needs: magic, version, tensor count and KV count, then
the key/value pairs (tensors come after and are skipped). That is why it answers
for a 78 GiB file in milliseconds, and why it is the first thing to run when a
loader refuses a file -- the refusal names a feature, the metadata names its
shape (`compress_ratios`, `indexer.top_k`, `hyper_connection.count`, ...).

    python3 tools/gguf_meta.py FILE [FILE...]
    python3 tools/gguf_meta.py --grep INDEXER FILE     # substring filter

Token-array keys (`tokenizer.ggml.tokens` and friends) are summarised rather
than printed: they are the only large values in the header and never the answer.
"""
import struct
import sys

# GGUF value type ids -> reader. Types 8 (string) and 9 (array) are special.
_READERS = {
    0: lambda f: struct.unpack("<B", _rd(f, 1))[0],   # UINT8
    1: lambda f: struct.unpack("<b", _rd(f, 1))[0],   # INT8
    2: lambda f: struct.unpack("<H", _rd(f, 2))[0],   # UINT16
    3: lambda f: struct.unpack("<h", _rd(f, 2))[0],   # INT16
    4: lambda f: struct.unpack("<I", _rd(f, 4))[0],   # UINT32
    5: lambda f: struct.unpack("<i", _rd(f, 4))[0],   # INT32
    6: lambda f: struct.unpack("<f", _rd(f, 4))[0],   # FLOAT32
    7: lambda f: _rd(f, 1)[0] != 0,                   # BOOL
    10: lambda f: struct.unpack("<Q", _rd(f, 8))[0],  # UINT64
    11: lambda f: struct.unpack("<q", _rd(f, 8))[0],  # INT64
    12: lambda f: struct.unpack("<d", _rd(f, 8))[0],  # FLOAT64
}

# Only large, never-informative header arrays.
_SUMMARISE = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges",
              "tokenizer.ggml.token_type", "tokenizer.ggml.scores")


def _rd(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError("truncated header (wanted %d bytes, got %d)" % (n, len(b)))
    return b


def _u32(f):
    return struct.unpack("<I", _rd(f, 4))[0]


def _u64(f):
    return struct.unpack("<Q", _rd(f, 8))[0]


def _str(f):
    n = _u64(f)
    return _rd(f, n).decode("utf-8", "replace")


def _value(f, t):
    if t == 8:
        return _str(f)
    if t == 9:
        et = _u32(f)
        n = _u64(f)
        return [_value(f, et) for _ in range(n)]
    if t not in _READERS:
        raise ValueError("unknown GGUF value type %d" % t)
    return _READERS[t](f)


def read_meta(path):
    """[(key, value)] in file order. Raises on a truncated or non-GGUF file."""
    with open(path, "rb") as f:
        if _rd(f, 4) != b"GGUF":
            raise ValueError("%s: not a GGUF file (bad magic)" % path)
        version = _u32(f)
        if version not in (2, 3):
            raise ValueError("%s: GGUF v%d is not supported" % (path, version))
        _u64(f)  # tensor count: not needed, the KV block comes first
        n_kv = _u64(f)
        out = []
        for _ in range(n_kv):
            key = _str(f)
            t = _u32(f)
            out.append((key, _value(f, t)))
    return out


def _show(key, value):
    if key in _SUMMARISE:
        return "<%d values>" % len(value)
    if isinstance(value, list):
        if len(value) > 12:
            return "<%d values: %s ...>" % (len(value), value[:8])
        return value
    return value


def main(argv):
    grep = None
    paths = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--grep":
            i += 1
            if i >= len(argv):
                print("gguf_meta: --grep needs a substring", file=sys.stderr)
                return 2
            grep = argv[i].lower()
        elif a.startswith("-"):
            print("gguf_meta: unknown flag %s" % a, file=sys.stderr)
            return 2
        else:
            paths.append(a)
        i += 1
    if not paths:
        print(__doc__.strip().splitlines()[-4], file=sys.stderr)
        return 2

    rc = 0
    for path in paths:
        print("== %s" % path)
        try:
            meta = read_meta(path)
        except (OSError, ValueError, EOFError) as e:
            print("   %s" % e, file=sys.stderr)
            rc = 1
            continue
        for key, value in meta:
            if grep and grep not in key.lower():
                continue
            # ascii, because a Windows console is not UTF-8 by default and a
            # model name with a fullwidth character must not end the dump.
            line = "%s = %s" % (key, _show(key, value))
            print("   " + line.encode("ascii", "replace").decode())
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
