#!/usr/bin/env python3
"""Reference MoE for ONE gpt-oss layer from the GGUF, compared against KRK_DUMP.

aim: what did kraken compute for moe.sum at layer L, against numpy reading the
same expert bank with the same router, the same clamps and the same biases.
    python3 tmp_moe_ref.py FILE LAYER DUMPFILE
"""
import struct
import sys

import numpy as np

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
# ggml type -> (block elements, bytes per block)
BLOCK = {0: (1, 4), 1: (1, 2), 8: (32, 34), 39: (32, 17)}


def u32(f):
    return struct.unpack("<I", f.read(4))[0]


def u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def st(f):
    n = u64(f)
    return f.read(n).decode("utf-8", "replace")


def skip(f, t):
    if t == 8:
        st(f)
    elif t == 9:
        et = u32(f)
        n = u64(f)
        for _ in range(n):
            skip(f, et)
    else:
        f.read(SIZES[t])


class Gguf:
    def __init__(self, path):
        self.f = open(path, "rb")
        f = self.f
        assert f.read(4) == b"GGUF"
        u32(f)
        n_tensors = u64(f)
        n_kv = u64(f)
        self.align = 32
        for _ in range(n_kv):
            k = st(f)
            t = u32(f)
            if k == "general.alignment" and t == 4:
                self.align = u32(f)
            else:
                skip(f, t)
        self.info = {}
        for _ in range(n_tensors):
            name = st(f)
            nd = u32(f)
            dims = [u64(f) for _ in range(nd)]
            ttype = u32(f)
            off = u64(f)
            self.info[name] = (dims, ttype, off)
        pos = f.tell()
        self.base = (pos + self.align - 1) // self.align * self.align

    def raw(self, name):
        dims, ttype, off = self.info[name]
        n = 1
        for d in dims:
            n *= d
        be, bb = BLOCK[ttype]
        nbytes = (n // be) * bb
        self.f.seek(self.base + off)
        return dims, ttype, self.f.read(nbytes)

    def f32(self, name):
        dims, ttype, buf = self.raw(name)
        assert ttype == 0, (name, ttype)
        return np.frombuffer(buf, dtype="<f4").copy()

    def matrix(self, name, experts=None):
        """[out, in] float32 (or [n_exp, out, in] slices)."""
        dims, ttype, buf = self.raw(name)
        # GGUF ne = (in, out): the flat rows are OUTPUT rows of `in` values.
        inn, out = dims[0], dims[1]
        be, bb = BLOCK[ttype]
        nrow_blocks = inn // be
        per = out * nrow_blocks * bb
        rows = []
        for e in (experts if experts is not None else [0]):
            start = e * per
            sub = np.frombuffer(buf[start:start + per], dtype=np.uint8)
            if ttype == 0:
                m = np.frombuffer(sub.tobytes(), dtype="<f4").reshape(out, inn)
            elif ttype == 8:  # Q8_0: f16 scale + 32 int8
                blocks = sub.reshape(out, nrow_blocks, bb)
                sc = blocks[:, :, :2].copy().view("<f2").astype(np.float32)
                q = blocks[:, :, 2:].copy().view(np.int8).astype(np.float32)
                m = (sc * q).reshape(out, inn)
            elif ttype == 39:  # MXFP4: 16 bytes of nibbles + E8M0 scale
                blocks = sub.reshape(out, nrow_blocks, bb)
                sc = np.ldexp(np.ones((out, nrow_blocks), np.float32),
                              blocks[:, :, 0].astype(np.int32) - 127)
                lo = (blocks[:, :, 1:17] & 0x0F).astype(np.int32)
                hi = (blocks[:, :, 1:17] >> 4).astype(np.int32)
                nib = np.empty((out, nrow_blocks, 32), np.int32)
                nib[:, :, 0::2] = lo
                nib[:, :, 1::2] = hi
                m = (_e2m1(nib) * sc[:, :, None]).reshape(out, inn)
            else:
                raise SystemExit("type %d not handled" % ttype)
            rows.append(m.astype(np.float32))
        return rows


def _e2m1(nib):
    """MXFP4's 4-bit e2m1 codebook."""
    tbl = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)
    return tbl[nib & 0x0F]


def dump_row(path, stage, layer):
    for line in open(path):
        p = line.split()
        if p[0] == stage and p[1] == "L%02d" % layer:
            return np.array([float(x) for x in p[3:]], np.float32)
    raise SystemExit("stage %s L%02d not in %s" % (stage, layer, path))


def main(path, layer, dump):
    g = Gguf(path)
    b = "blk.%d." % layer
    x = dump_row(dump, "attn.ffnn", layer)
    print("input rms %.6g (from the dump)" % np.sqrt(np.mean(x.astype(np.float64) ** 2)))

    wg = g.f32(b + "ffn_gate_inp.weight").reshape(128, 2880)  # [n_exp, in]
    bg = g.f32(b + "ffn_gate_inp.bias")
    logits = wg @ x + bg
    m = logits.max()
    p = np.exp(logits - m)
    p /= p.sum()
    top = np.argsort(-p)[:4]
    print("routed experts", top.tolist(), "probs", ["%.5f" % p[e] for e in top])

    bg_e = g.f32(b + "ffn_gate_exps.bias").reshape(128, 2880)
    bu_e = g.f32(b + "ffn_up_exps.bias").reshape(128, 2880)
    bd_e = g.f32(b + "ffn_down_exps.bias").reshape(128, 2880)

    acc = np.zeros(2880, np.float32)
    for e in top:
        wgate = g.matrix(b + "ffn_gate_exps.weight", [int(e)])[0]
        wup = g.matrix(b + "ffn_up_exps.weight", [int(e)])[0]
        wdown = g.matrix(b + "ffn_down_exps.weight", [int(e)])[0]
        gate = wgate @ x + bg_e[e]
        up = wup @ x + bu_e[e]
        act = np.minimum(gate, 7.0) / (1.0 + np.exp(-1.702 * np.minimum(gate, 7.0)))
        y = act * (np.clip(up, -7.0, 7.0) + 1.0)
        acc += np.float32(p[e]) * (wdown @ y + bd_e[e])
    k = dump_row(dump, "moe.sum", layer)
    rel = np.linalg.norm(acc - k) / max(np.linalg.norm(k), 1e-9)
    print("python rms %.6g   kraken rms %.6g   rel-l2 %.3g"
          % (np.sqrt(np.mean(acc.astype(np.float64) ** 2)),
             np.sqrt(np.mean(k.astype(np.float64) ** 2)), rel))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
