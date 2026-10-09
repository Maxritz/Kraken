#!/usr/bin/env python3
"""Report the fp16 block scales of one expert slice of an IQ3_XXS expert tensor.

  python3 tools/iq_scale_probe.py FILE TENSOR EXPERT [EXPERT...]

Why: an IQ3_XXS block is 98 bytes -- an fp16 scale, 64 grid-index bytes and 32
bytes of group scale/sign words. The value a block can produce is at most
d * (0.5 + 15) * 0.5 * 255, so the *scale* alone says whether a correct decode
can exceed the f16 range (65504). A device whose dequant stores _Float16 would
overflow to inf where a host f32 path would not, which is a device-only failure
with no counterpart in the file.

Read-only; no weight load, no grid table needed.
"""
import struct
import sys

QK = 256          # values per IQ3_XXS block
BLOCK = 98        # bytes per block


def scan(path):
    """Return (data_start, {name: (dims, ttype, offset)})."""
    with open(path, "rb") as f:
        def rd(n):
            b = f.read(n)
            if len(b) != n:
                raise EOFError("short read")
            return b

        if rd(4) != b"GGUF":
            raise SystemExit("not a GGUF")
        _, n_tensor, n_kv = struct.unpack("<IQQ", rd(20))
        alignment = 32
        for _ in range(n_kv):
            k = rd(struct.unpack("<Q", rd(8))[0]).decode("utf-8", "replace")
            t = struct.unpack("<I", rd(4))[0]
            if t == 8:
                v = rd(struct.unpack("<Q", rd(8))[0])
                if k == "general.alignment":
                    alignment = struct.unpack("<I", v)[0]
            elif t == 9:
                et = struct.unpack("<I", rd(4))[0]
                n = struct.unpack("<Q", rd(8))[0]
                for _ in range(n):
                    if et == 8:
                        rd(struct.unpack("<Q", rd(8))[0])
                    else:
                        rd({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[et])
            else:
                rd({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[t])
        tens = {}
        for _ in range(n_tensor):
            name = rd(struct.unpack("<Q", rd(8))[0]).decode("utf-8", "replace")
            nd = struct.unpack("<I", rd(4))[0]
            dims = struct.unpack("<%dQ" % nd, rd(8 * nd))
            ttype, off = struct.unpack("<IQ", rd(12))
            tens[name] = (dims, ttype, off)
        start = (f.tell() + alignment - 1) // alignment * alignment
        return start, tens


def h2f(u):
    return struct.unpack("<e", struct.pack("<H", u))[0]


def main():
    path, tname = sys.argv[1], sys.argv[2]
    experts = [int(x) for x in sys.argv[3:]]
    start, tens = scan(path)
    if tname not in tens:
        raise SystemExit("no tensor %s" % tname)
    dims, ttype, off = tens[tname]
    if ttype != 18:
        raise SystemExit("%s is type%d, not IQ3_XXS(18)" % (tname, ttype))
    row = dims[0]                     # fastest dim = row width in values
    n_rows = dims[1] if len(dims) > 1 else 1
    n_exp = dims[2] if len(dims) > 2 else 1
    blocks_per_expert = (row * n_rows) // QK
    print("%s: row=%d rows=%d experts=%d, %d blocks/expert" %
          (tname, row, n_rows, n_exp, blocks_per_expert))
    with open(path, "rb") as f:
        for e in experts:
            base = start + off + e * blocks_per_expert * BLOCK
            f.seek(base)
            raw = f.read(blocks_per_expert * BLOCK)
            ds = []
            for b in range(blocks_per_expert):
                ds.append(h2f(struct.unpack("<H", raw[b * BLOCK:b * BLOCK + 2])[0]))
            fin = [d for d in ds if d == d and abs(d) != float("inf")]
            worst = max((abs(d) for d in fin), default=0.0)
            # Upper bound on a decoded value: d * (0.5 + 15) * 0.5 * max grid byte.
            bound = worst * 7.75 * 255.0
            print("  expert %-4d blocks=%-6d max|d|=%-12.6g implied max|v|=%-12.6g "
                  "over_f16=%s" % (e, len(ds), worst, bound, bound > 65504.0))


main()
