#!/usr/bin/env python3
"""Read named F32 tensors from a GGUF and report min/max/non-finite.

  python3 tools/gguf_read_f32.py FILE PATTERN [PATTERN...]

Read-only, no weight load: this answers "is the file's own bytes NaN" for the
small F32 tensors an engine reads late in a layer (norms, biases), where a
poisoned value surfaces as an all-NaN activation one stage later. A pattern that
matches nothing prints nothing.
"""
import struct
import sys

SZS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: None, 9: None, 10: 8, 11: 8, 12: 8}


def rd(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError("short read")
    return b


def value(f, t):
    if t == 9:
        et = struct.unpack("<I", rd(f, 4))[0]
        n = struct.unpack("<Q", rd(f, 8))[0]
        if et == 8:
            for _ in range(n):
                rd(f, struct.unpack("<Q", rd(f, 8))[0])
            return None
        rd(f, SZS[et] * n)
        return None
    if t == 8:
        return rd(f, struct.unpack("<Q", rd(f, 8))[0])
    fmt = {1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t]
    return struct.unpack("<" + fmt, rd(f, SZS[t]))[0]


def main():
    path = sys.argv[1]
    pats = sys.argv[2:]
    with open(path, "rb") as f:
        if rd(f, 4) != b"GGUF":
            raise SystemExit("not a GGUF")
        ver, n_tensor, n_kv = struct.unpack("<IQQ", rd(f, 20))
        alignment = 32
        for _ in range(n_kv):
            k = rd(f, struct.unpack("<Q", rd(f, 8))[0]).decode("utf-8", "replace")
            t = struct.unpack("<I", rd(f, 4))[0]
            v = value(f, t)
            if k == "general.alignment" and isinstance(v, int):
                alignment = v
        info = []
        for _ in range(n_tensor):
            name = rd(f, struct.unpack("<Q", rd(f, 8))[0]).decode("utf-8", "replace")
            nd = struct.unpack("<I", rd(f, 4))[0]
            dims = struct.unpack("<%dQ" % nd, rd(f, 8 * nd))
            ttype, off = struct.unpack("<IQ", rd(f, 12))
            info.append((name, dims, ttype, off))
        # The data section starts after the tensor directory, aligned.
        data_start = f.tell()
        data_start = (data_start + alignment - 1) // alignment * alignment
        for name, dims, ttype, off in info:
            if not any(p in name for p in pats):
                continue
            n = 1
            for d in dims:
                n *= d
            if ttype != 0:
                print("  %-42s type%d (not F32, skipped)" % (name, ttype))
                continue
            f.seek(data_start + off)
            vals = struct.unpack("<%df" % n, rd(f, 4 * n))
            nan = sum(1 for v in vals if v != v)
            inf = sum(1 for v in vals if v == float("inf") or v == float("-inf"))
            fin = [v for v in vals if v == v and abs(v) != float("inf")]
            print("  %-42s n=%-6d min=%-12.6g max=%-12.6g nan=%d inf=%d"
                  % (name, n, min(fin) if fin else 0.0, max(fin) if fin else 0.0, nan, inf))


main()
