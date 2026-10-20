#!/usr/bin/env python3
"""List GGUF tensor names/dims/types without reading weight bytes."""
import struct
import sys


def u32(f):
    return struct.unpack("<I", f.read(4))[0]


def u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def s(f):
    n = u64(f)
    return f.read(n).decode("utf-8", "replace")


def skip_value(f, t):
    if t == 8:
        s(f)
    elif t == 9:
        et = u32(f)
        n = u64(f)
        for _ in range(n):
            skip_value(f, et)
    else:
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        f.read(sizes[t])


def main(path, filt=None):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        _v = u32(f)
        n_tensors = u64(f)
        n_kv = u64(f)
        for _ in range(n_kv):
            s(f)
            skip_value(f, u32(f))
        for _ in range(n_tensors):
            name = s(f)
            n_dims = u32(f)
            dims = [u64(f) for _ in range(n_dims)]
            ttype = u32(f)
            _off = u64(f)
            if filt and filt not in name:
                continue
            print("%-46s %-22s type=%d" % (name, "x".join(str(d) for d in dims), ttype))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
