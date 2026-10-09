#!/usr/bin/env python3
"""Print every tensor of a GGUF with its dimensions, plus the metadata keys a
drafter's contract depends on. Read-only, no weight load.

  python3 tools/gguf_tensor_shapes.py FILE [NAME_FILTER]

Why this exists: a DFlash head set sizes its fusion projection against the
*target's* embedding width (6 captured layers x target_embd), so the target it
was built for is a fact in the file rather than something to guess by trying
targets until one loads.
"""
import struct
import sys

SZS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: None, 9: None, 10: 8, 11: 8, 12: 8}


def rd(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError("short read")
    return b


def scal(f, t):
    if t == 8:                                   # string
        return rd(f, struct.unpack("<Q", rd(f, 8))[0]).decode("utf-8", "replace")
    sz = SZS[t]
    fmt = {1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t]
    return struct.unpack("<" + fmt, rd(f, sz))[0]


def value(f, t):
    if t == 9:                                   # array
        et = struct.unpack("<I", rd(f, 4))[0]
        n = struct.unpack("<Q", rd(f, 8))[0]
        if et == 8:                              # array of strings: skip payload
            for _ in range(n):
                rd(f, struct.unpack("<Q", rd(f, 8))[0])
            return "[%d strings]" % n
        sz = SZS[et]
        raw = rd(f, sz * n)
        # Small integer arrays carry meaning (target_layer_ids, ramps), so show
        # them: an id the target does not have is a refusal, not a detail.
        if et in (4, 5, 10, 11) and n <= 64:
            return list(struct.unpack("<%d%s" % (n, {4: "I", 5: "i", 10: "Q", 11: "q"}[et]), raw))
        return "[%d x type%d]" % (n, et)
    return scal(f, t)


def main():
    path = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    with open(path, "rb") as f:
        if rd(f, 4) != b"GGUF":
            raise SystemExit("not a GGUF")
        ver, n_tensor, n_kv = struct.unpack("<IQQ", rd(f, 20))
        print("version %d, %d tensors, %d kv" % (ver, n_tensor, n_kv))
        meta = {}
        for _ in range(n_kv):
            k = rd(f, struct.unpack("<Q", rd(f, 8))[0]).decode("utf-8", "replace")
            t = struct.unpack("<I", rd(f, 4))[0]
            meta[k] = value(f, t)
        for _ in range(n_tensor):
            name = rd(f, struct.unpack("<Q", rd(f, 8))[0]).decode("utf-8", "replace")
            nd = struct.unpack("<I", rd(f, 4))[0]
            dims = struct.unpack("<%dQ" % nd, rd(f, 8 * nd))
            ttype, off = struct.unpack("<IQ", rd(f, 12))
            if filt and filt.lower() not in name.lower():
                continue
            # GGUF stores dims fastest-first; print the shape the code sees.
            shape = "x".join(str(d) for d in reversed(dims))
            print("  %-44s %-14s %s" % (name, shape, "type%d" % ttype))
        print("-- metadata worth reading --")
        for k in sorted(meta):
            if filt and filt.lower() not in k.lower():
                continue
            v = meta[k]
            if isinstance(v, str) and len(v) > 60:
                v = v[:57] + "..."
            if isinstance(v, str) and v.startswith("[") and len(v) > 40:
                v = v[:37] + "..."
            print("  %-46s %s" % (k, v))


main()
