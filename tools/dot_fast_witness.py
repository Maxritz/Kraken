#!/usr/bin/env python3
"""Accuracy witness for the opt-in block-format dot (KRK_CPU_DOT_FAST=1).

Reads real q8_0 / q4_0 expert rows out of a GGUF, computes the dot both ways
in strict f32 -- one sequential accumulator (the reference) and four lanes
(a0+a1)+(a2+a3) (the opt-in) -- and reports how far apart they are.

  python3 tools/dot_fast_witness.py FILE [rows]

The values are identical in both arms; only the summation order differs, so
what this measures is exactly the rounding the switch trades away. It is not a
model-quality claim: the weights are whatever the file holds.
"""
import struct
import sys

import numpy as np

SZS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: None, 9: None,
       10: 8, 11: 8, 12: 8}
FMT = {1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q",
       11: "q", 12: "d"}


def rd(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError("short read")
    return b


def scal(f, t):
    if t == 8:
        return rd(f, struct.unpack("<Q", rd(f, 8))[0]).decode("utf-8", "replace")
    return struct.unpack("<" + FMT[t], rd(f, SZS[t]))[0]


def value(f, t):
    if t == 9:  # array
        et = struct.unpack("<I", rd(f, 4))[0]
        n = struct.unpack("<Q", rd(f, 8))[0]
        return [value(f, et) for _ in range(n)]
    return scal(f, t)


def dequant_q8_0(raw):
    """ggml q8_0: f16 scale + 32 int8 per 34-byte block."""
    out = np.empty(len(raw) // 34 * 32, dtype=np.float32)
    o = 0
    for b in range(0, len(raw), 34):
        d = np.float32(struct.unpack("<e", raw[b:b + 2])[0])
        q = np.frombuffer(raw[b + 2:b + 34], dtype=np.int8).astype(np.float32)
        out[o:o + 32] = np.float32(d * q)
        o += 32
    return out


def dequant_q4_0(raw):
    """ggml q4_0: f16 scale + 16 packed byte pairs per 18-byte block."""
    out = np.empty(len(raw) // 18 * 32, dtype=np.float32)
    o = 0
    for b in range(0, len(raw), 18):
        d = np.float32(struct.unpack("<e", raw[b:b + 2])[0])
        for byte in raw[b + 2:b + 18]:
            out[o] = np.float32(d * np.float32((byte & 0x0F) - 8))
            out[o + 16] = np.float32(d * np.float32((byte >> 4) - 8))
            o += 1
        o += 16
    return out


def dot_exact(w, x):
    s = np.float32(0.0)
    for i in range(len(w)):
        s = np.float32(s + np.float32(w[i] * x[i]))
    return s


def dot_fast(w, x):
    a = [np.float32(0.0)] * 4
    for i in range(0, len(w) - 3, 4):
        for k in range(4):
            a[k] = np.float32(a[k] + np.float32(w[i + k] * x[i + k]))
    s = np.float32(np.float32(a[0] + a[1]) + np.float32(a[2] + a[3]))
    for i in range(len(w) & ~3, len(w)):  # the scalar tail in the C++ path
        s = np.float32(s + np.float32(w[i] * x[i]))
    return s


def main():
    path = sys.argv[1]
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    rng = np.random.default_rng(12345)

    with open(path, "rb") as f:
        if rd(f, 4) != b"GGUF":
            raise SystemExit("not a GGUF")
        ver, n_tensor, n_kv = struct.unpack("<IQQ", rd(f, 20))
        meta = {}
        for _ in range(n_kv):
            k = rd(f, struct.unpack("<Q", rd(f, 8))[0]).decode()
            t = struct.unpack("<I", rd(f, 4))[0]
            meta[k] = value(f, t)
        tensors = {}
        for _ in range(n_tensor):
            name = rd(f, struct.unpack("<Q", rd(f, 8))[0]).decode()
            nd = struct.unpack("<I", rd(f, 4))[0]
            dims = struct.unpack("<%dQ" % nd, rd(f, 8 * nd))
            ttype, off = struct.unpack("<IQ", rd(f, 12))
            tensors[name] = (dims, ttype, off)
        align = meta.get("general.alignment", 32)
        data0 = f.tell()
        data0 = (data0 + align - 1) // align * align

        name, (dims, ttype, off) = next(
            ((n, v) for n, v in tensors.items() if "gate_exps" in n))
        deq = {8: dequant_q8_0, 2: dequant_q4_0}.get(ttype)
        if deq is None:
            raise SystemExit("tensor type %d is not a block format this reads"
                             % ttype)
        n_in = dims[0]
        n_expert = dims[2] if len(dims) > 2 else 1
        blk = 34 if ttype == 8 else 18
        per_expert = (n_in * dims[1]) // 32 * blk
        row_bytes = n_in // 32 * blk
        f.seek(data0 + off)

        rels = []
        cond = []
        peak = 0.0
        for _ in range(rows):
            e = int(rng.integers(0, n_expert))
            r = int(rng.integers(0, dims[1]))
            f.seek(data0 + off + e * per_expert + r * row_bytes)
            w = deq(rd(f, row_bytes))
            x = (rng.standard_normal(n_in, dtype=np.float32) *
                 np.float32(0.5)).astype(np.float32)
            a = dot_exact(w, x)
            b = dot_fast(w, x)
            # Two denominators, because they answer different questions.
            # |dot| is what a user would divide by, and it explodes near a
            # cancellation -- a dot that should be 0.001 and lands on 0.0013 is
            # "30% wrong" and completely fine. sum|w*x| is the size of the terms
            # being added, which is the error the summation order can actually
            # introduce.
            d = abs(float(a) - float(b))
            rels.append(d / max(abs(float(a)), 1e-30))
            cond.append(d / max(float(np.abs(np.float32(w) * x).sum()), 1e-30))
            peak = max(peak, abs(float(a)))

    rels.sort()
    cond.sort()
    print("%s: %s %s  n_in=%d  rows=%d" %
          (path.split("/")[-1], name, "q8_0" if ttype == 8 else "q4_0",
           n_in, rows))
    print("  |dot| peak                    %.4f" % peak)
    print("  err / |dot|       median %.3e   p99 %.3e   max %.3e"
          % (rels[len(rels) // 2], rels[int(len(rels) * 0.99)], rels[-1]))
    print("  err / sum|w.x|    median %.3e   p99 %.3e   max %.3e"
          % (cond[len(cond) // 2], cond[int(len(cond) * 0.99)], cond[-1]))


main()
