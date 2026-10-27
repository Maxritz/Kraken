#!/usr/bin/env python3
"""Write a llama-shaped GGUF fixture for benchmarking, at a chosen weight format.

Why this exists: a benchmark needs a model whose *shape* is known and whose
*bytes* are reproducible, and this repo's test fixtures live inside the C++ test
binary. This writes the same kind of file from the command line, in the quant
formats real checkpoints actually ship, so the virtual NPU's dequantizing weight
cache is exercised rather than assumed.

Everything is generated from a seeded PRNG, so two runs produce byte-identical
files and a benchmark number can be attributed to the code rather than to the
weights. The values are small (a mixture of +-0.04) so a 4-8 layer stack stays
in a sane activation range and the decoded text is coherent.

Pure Python on purpose: a fixture generator that needs numpy is one more thing
between a reader and the number.

    python3 tools/make_bench_gguf.py out.gguf --embd 512 --layers 4 --format q8_0

Formats: f32, f16, q8_0, q4_0. Block layouts are ggml's, verified against
src/quant.cpp (deq_q8_0: f16 scale + 32 int8 in 34 bytes; deq_q4_0: f16 scale +
16 packed nibble pairs in 18 bytes, value = d * (nibble - 8)). Norms stay f32,
as every checkpoint keeps them.

Tensor names are the llama schema the loader reads: token_embd, output,
output_norm, and blk.N.{attn_norm,ffn_norm,attn_q,attn_k,attn_v,attn_output,
ffn_gate,ffn_up,ffn_down}.
"""
import argparse
import random
import struct
import sys

# ggml tensor type ids (ggml.h); the four this tool writes.
T_F32, T_F16, T_Q4_0, T_Q8_0 = 0, 1, 2, 8
FORMATS = {
    "f32": (T_F32, 1, 0),
    "f16": (T_F16, 1, 0),
    "q8_0": (T_Q8_0, 34, 32),
    "q4_0": (T_Q4_0, 18, 32),
}

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGN = 32

# metadata value types
(M_U32, M_F32, M_BOOL, M_STR, M_ARR) = (4, 6, 7, 8, 9)


class Gguf:
    def __init__(self):
        self.meta = b""
        self.n_kv = 0
        self.tensors = []  # (name, ne0, ne1, dtype, data)

    # ---- metadata ----------------------------------------------------------
    def _key(self, k):
        self.n_kv += 1
        b = k.encode()
        return struct.pack("<Q", len(b)) + b

    def u32(self, k, v):
        self.meta += self._key(k) + struct.pack("<I", M_U32) + struct.pack("<I", v)

    def f32(self, k, v):
        self.meta += self._key(k) + struct.pack("<I", M_F32) + struct.pack("<f", v)

    def boolean(self, k, v):
        self.meta += self._key(k) + struct.pack("<I", M_BOOL) + struct.pack("<?", v)

    def string(self, k, v):
        b = v.encode()
        self.meta += (self._key(k) + struct.pack("<I", M_STR) +
                      struct.pack("<Q", len(b)) + b)

    def str_array(self, k, vals):
        out = self._key(k) + struct.pack("<I", M_ARR) + struct.pack("<I", M_STR)
        out += struct.pack("<Q", len(vals))
        for v in vals:
            b = v.encode()
            out += struct.pack("<Q", len(b)) + b
        self.meta += out

    def f32_array(self, k, vals):
        out = self._key(k) + struct.pack("<I", M_ARR) + struct.pack("<I", M_F32)
        out += struct.pack("<Q", len(vals))
        out += struct.pack("<%df" % len(vals), *vals)
        self.meta += out

    # ---- tensors -----------------------------------------------------------
    def tensor(self, name, ne0, ne1, data, dtype):
        self.tensors.append((name, ne0, ne1, dtype, data))

    def write(self, path):
        infos = b""
        blobs = []
        off = 0
        for (name, ne0, ne1, dtype, data) in self.tensors:
            infos += struct.pack("<Q", len(name)) + name.encode()
            infos += struct.pack("<I", 2)  # n_dims
            infos += struct.pack("<Q", ne0) + struct.pack("<Q", ne1)
            infos += struct.pack("<I", dtype) + struct.pack("<Q", off)
            blobs.append(data)
            off += len(data)
            off = (off + ALIGN - 1) // ALIGN * ALIGN

        head = struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", GGUF_VERSION)
        head += struct.pack("<Q", len(self.tensors)) + struct.pack("<Q", self.n_kv)
        head += self.meta + infos
        head += b"\0" * ((-len(head)) % ALIGN)

        with open(path, "wb") as f:
            f.write(head)
            for data in blobs:
                f.write(data)
                f.write(b"\0" * ((-len(data)) % ALIGN))
        return len(head) + sum(len(b) for b in blobs)


def f16(x):
    return struct.pack("<e", x)


def quantize_q8_0(vals):
    """ggml q8_0: one f16 scale per 32 values, then 32 int8."""
    out = bytearray()
    for b in range(0, len(vals), 32):
        blk = vals[b:b + 32]
        amax = max(abs(v) for v in blk) or 1e-8
        d = amax / 127.0
        out += f16(d)
        for v in blk:
            q = max(-127, min(127, int(round(v / d))))
            out += struct.pack("<b", q)
    return bytes(out)


def quantize_q4_0(vals):
    """ggml q4_0: one f16 scale per 32 values, 16 bytes of nibble pairs,
    value = d * (nibble - 8), low nibble = index k, high nibble = index k+16."""
    out = bytearray()
    for b in range(0, len(vals), 32):
        blk = vals[b:b + 32]
        amax = max((-min(blk), max(blk)))
        d = amax / -8.0 if amax else 0.0
        id_ = 1.0 / d if d else 0.0
        q = [max(0, min(15, int(round(v * id_ + 8)))) for v in blk]
        out += f16(d)
        for k in range(16):
            out += struct.pack("<B", (q[k] & 0xF) | ((q[k + 16] & 0xF) << 4))
    return bytes(out)


def make_weights(vals, fmt):
    dtype, _blk_bytes, blk = FORMATS[fmt]
    if blk:
        pad = (-len(vals)) % blk
        vals = vals + [0.0] * pad
    if dtype == T_F32:
        return struct.pack("<%df" % len(vals), *vals), dtype
    if dtype == T_F16:
        return b"".join(f16(v) for v in vals), dtype
    if dtype == T_Q8_0:
        return quantize_q8_0(vals), dtype
    if dtype == T_Q4_0:
        return quantize_q4_0(vals), dtype
    raise ValueError(fmt)


SPM_EXTRA = [("\u2581the", -1.0), ("\u2581quick", -1.0), ("\u2581brown", -1.0),
             ("\u2581fox", -1.0), ("\u2581", -2.0), ("jumps", -1.0),
             ("\u2581over", -1.0), ("\u2581lazy", -1.0), ("\u2581dog", -1.0),
             ("\u2581and", -1.0), ("\u2581is", -1.0), ("\u2581a", -1.0)]


def build(path, n_embd, n_layer, n_head, n_kv, hd, n_ff, vocab, ctx, fmt, seed):
    rnd = random.Random(seed)
    q_dim, kv_dim = n_head * hd, n_kv * hd

    tokens = ["<unk>", "<s>", "</s>"]
    scores = [-10.0, 0.0, 0.0]
    for i in range(256):
        tokens.append("<0x%02X>" % i)
        scores.append(-30.0)
    for t, s in SPM_EXTRA:
        tokens.append(t)
        scores.append(s)
    while len(tokens) < vocab:
        tokens.append("<extra%d>" % len(tokens))
        scores.append(-50.0)
    assert len(tokens) == vocab, (len(tokens), vocab)

    g = Gguf()
    g.string("general.architecture", "llama")
    g.string("general.name", "kraken-bench-%d-%s" % (n_embd, fmt))
    g.u32("general.alignment", ALIGN)
    g.u32("llama.block_count", n_layer)
    g.u32("llama.context_length", ctx)
    g.u32("llama.embedding_length", n_embd)
    g.u32("llama.feed_forward_length", n_ff)
    g.u32("llama.attention.head_count", n_head)
    g.u32("llama.attention.head_count_kv", n_kv)
    g.u32("llama.attention.key_length", hd)
    g.u32("llama.attention.value_length", hd)
    g.f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
    g.f32("llama.rope.freq_base", 10000.0)
    g.string("tokenizer.ggml.model", "llama")
    g.str_array("tokenizer.ggml.tokens", tokens)
    g.f32_array("tokenizer.ggml.scores", scores)
    g.u32("tokenizer.ggml.bos_token_id", 1)
    g.u32("tokenizer.ggml.eos_token_id", 2)
    g.boolean("tokenizer.ggml.add_bos_token", True)

    scale = 0.04

    def w(name, n_in, n_out):
        vals = [rnd.gauss(0.0, scale) for _ in range(n_in * n_out)]
        data, dtype = make_weights(vals, fmt)
        g.tensor(name, n_in, n_out, data, dtype)

    def ones(name, n):
        g.tensor(name, n, 1, struct.pack("<%df" % n, *([1.0] * n)), T_F32)

    w("token_embd.weight", n_embd, vocab)
    ones("output_norm.weight", n_embd)
    w("output.weight", n_embd, vocab)
    for l in range(n_layer):
        p = "blk.%d." % l
        ones(p + "attn_norm.weight", n_embd)
        ones(p + "ffn_norm.weight", n_embd)
        w(p + "attn_q.weight", n_embd, q_dim)
        w(p + "attn_k.weight", n_embd, kv_dim)
        w(p + "attn_v.weight", n_embd, kv_dim)
        w(p + "attn_output.weight", q_dim, n_embd)
        w(p + "ffn_gate.weight", n_embd, n_ff)
        w(p + "ffn_up.weight", n_embd, n_ff)
        w(p + "ffn_down.weight", n_ff, n_embd)

    return g.write(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--embd", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=0,
                    help="query heads (0 = embd/64)")
    ap.add_argument("--kv-heads", type=int, default=0,
                    help="kv heads (0 = same as --heads)")
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--ff", type=int, default=0,
                    help="feed-forward width (0 = embd * 2.6875, rounded to 32)")
    ap.add_argument("--vocab", type=int, default=1000)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--format", default="f16", choices=sorted(FORMATS))
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    a = ap.parse_args(argv)

    heads = a.heads or max(1, a.embd // 64)
    kv_heads = a.kv_heads or heads
    ff = a.ff or (int(a.embd * 2.6875) // 32 * 32 or 32)
    if a.head_dim * heads != a.embd and a.embd % heads:
        ap.error("--embd must be a multiple of --heads")
    for nm, val in (("embd", a.embd), ("ff", ff), ("head-dim", a.head_dim)):
        if val % 32:
            ap.error("--%s must be a multiple of 32 (quant block size)" % nm)

    size = build(a.out, a.embd, a.layers, heads, kv_heads, a.head_dim, ff,
                 a.vocab, a.ctx, a.format, a.seed)
    print("wrote %s: %d layers, embd %d, %d/%d heads, hd %d, ff %d, vocab %d, "
          "%s, %.1f MiB" % (a.out, a.layers, a.embd, heads, kv_heads,
                            a.head_dim, ff, a.vocab, a.format,
                            size / 1048576.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
