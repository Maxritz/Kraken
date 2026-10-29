#!/usr/bin/env python3
"""Write a BitNet-b1.58 GGUF whose matmul weights are ternary.

There is no public BitNet checkpoint in this repository (and no network path to
one), so the I2_S loader would otherwise only ever be exercised by in-process
fixtures. This script writes a *file*: a real GGUF v3 container, the metadata
keys a real `bitnet-b1.58` file carries, the tensor names llama.cpp's
`src/models/bitnet.cpp` uses (including the SubLN pair), and weights in the
exact layout microsoft/BitNet's `utils/convert-hf-to-gguf-bitnet.py` produces.

Everything is deterministic: the same arguments write the same bytes.

Two I2_S dialects exist in the wild, and they are NOT interchangeable:

  hf  -- tools/../convert-hf-to-gguf-bitnet.py `quantize_to_i2_s`:
         128 values / 32 bytes, planar (byte j -> elements j, j+32, j+64,
         j+96), the high 2-bit pair first, code = value + 1, and a 32-byte
         tail whose first four bytes are one f32 for the WHOLE tensor and
         whose remaining 28 bytes are zero.
  ms  -- utils/convert-ms-to-gguf-bitnet.py `transform_to_i2` +
         `preprocess_weights`: code = value + 2 (so 0 is an unused code) and
         the payload is permuted into the converter's tile layout using
         constants read from that build's `build/kcfg.ini`. It is only
         readable by BitNet's own kernel for the tile shape it was built for.
         Its tail is `np.tile(scale, 8)` -- EIGHT copies of the same f32.

`--dialect ms` writes the second kind so the loader's refusal can be tested
(decoding it as if it were `hf` yields a whole model of wrong weights and no
error at all).

usage:
  tools/make_bitnet_gguf.py out.gguf [--dialect hf|ms] [--scale 0.0625]
                                        [--storage i2s|f16] [--layers 4]
"""

import argparse
import struct
import sys

# ---------------------------------------------------------------------------
# GGUF v3 writer
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGN = 32

# value types
T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY = 4, 5, 6, 7, 8, 9
# element types used inside arrays
E_INT32, E_FLOAT32, E_STRING = 5, 6, 8
# tensor types
TENSOR_F32, TENSOR_F16, TENSOR_I2_S = 0, 1, 36


class Writer:
    def __init__(self):
        self.meta = bytearray()
        self.kv = 0
        self.infos = bytearray()
        self.n_tensors = 0
        self.data = bytearray()

    # -- KV ---------------------------------------------------------------
    def _key(self, k):
        b = k.encode()
        self.meta += struct.pack("<Q", len(b)) + b

    def str_(self, k, v):
        self._key(k)
        b = v.encode()
        self.meta += struct.pack("<I", T_STRING)
        self.meta += struct.pack("<Q", len(b)) + b
        self.kv += 1

    def u32(self, k, v):
        self._key(k)
        self.meta += struct.pack("<II", T_UINT32, v)
        self.kv += 1

    def f32(self, k, v):
        self._key(k)
        self.meta += struct.pack("<If", T_FLOAT32, v)
        self.kv += 1

    def bool_(self, k, v):
        self._key(k)
        self.meta += struct.pack("<I?", T_BOOL, v)
        self.kv += 1

    def str_array(self, k, vals):
        self._key(k)
        self.meta += struct.pack("<IIQ", T_ARRAY, E_STRING, len(vals))
        for v in vals:
            b = v.encode()
            self.meta += struct.pack("<Q", len(b)) + b
        self.kv += 1

    def f32_array(self, k, vals):
        self._key(k)
        self.meta += struct.pack("<IIQ", T_ARRAY, E_FLOAT32, len(vals))
        self.meta += struct.pack("<%df" % len(vals), *vals)
        self.kv += 1

    def i32_array(self, k, vals):
        self._key(k)
        self.meta += struct.pack("<IIQ", T_ARRAY, E_INT32, len(vals))
        self.meta += struct.pack("<%di" % len(vals), *vals)
        self.kv += 1

    # -- tensors ----------------------------------------------------------
    def tensor(self, name, ttype, ne, payload):
        # offsets are relative to the start of the data section and aligned
        while len(self.data) % ALIGN:
            self.data += b"\0"
        off = len(self.data)
        self.data += payload
        b = name.encode()
        self.infos += struct.pack("<Q", len(b)) + b
        self.infos += struct.pack("<I", len(ne))
        for d in ne:
            self.infos += struct.pack("<Q", d)
        self.infos += struct.pack("<IQ", ttype, off)
        self.n_tensors += 1

    def write(self, path):
        with open(path, "wb") as f:
            f.write(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION,
                                self.n_tensors, self.kv))
            f.write(self.meta)
            f.write(self.infos)
            # GGUF v3 header is magic u32 + version u32 + n_tensors u64 +
            # n_kv u64 = 24 bytes. Using 8 here lands the data section 16
            # bytes off and every tensor reads back short -- which is exactly
            # how the official `gguf` reader reported it.
            pad = (-(24 + len(self.meta) + len(self.infos))) % ALIGN
            f.write(b"\0" * pad)
            f.write(self.data)


# ---------------------------------------------------------------------------
# ternary weights
# ---------------------------------------------------------------------------

def tern(name, i):
    """Deterministic ternary code in {-1, 0, +1} for (tensor, element)."""
    h = (hash(name) & 0xFFFFFFFF) if False else 0
    # FNV-1a over the name, mixed with the element index: no Python hash()
    # (it is salted per process, and the files must be reproducible).
    h = 2166136261
    for c in name.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    x = (h ^ ((i * 2654435761) & 0xFFFFFFFF)) & 0xFFFFFFFF
    x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
    x ^= x >> 15
    return (x % 3) - 1


def i2s_pack(mat, n_out, n_in, scale, dialect):
    """(n_out, n_in) ternary matrix -> I2_S bytes + 32-byte tail."""
    n = n_out * n_in
    flat = [tern(mat, i) for i in range(n)]
    assert n % 128 == 0, "an I2_S tensor is padded as a whole; use n %% 128 == 0"
    out = bytearray(n // 4)
    for blk in range(n // 128):
        for j in range(32):
            b = 0
            for l in range(4):
                v = flat[blk * 128 + j + l * 32]
                # hf: code = value + 1 (0..2, all three used)
                # ms: code = value + 2 (1..3, 0 unused)
                code = (v + (2 if dialect == "ms" else 1)) & 3
                b |= code << (6 - 2 * l)
            out[blk * 32 + j] = b
    tail = bytearray(32)
    if dialect == "ms":
        # np.tile(scale, 8): eight copies of the same f32
        tail[0:32] = struct.pack("<8f", *([scale] * 8))
    else:
        tail[0:4] = struct.pack("<f", scale)
    return bytes(out) + bytes(tail)


def f16_weights(mat, n_out, n_in, scale):
    vals = [tern(mat, i) * scale for i in range(n_out * n_in)]
    return struct.pack("<%de" % len(vals), *vals)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

def build(path, n_embd=256, n_layer=4, n_head=8, n_kv=8, hd=32,
          n_ff=512, vocab=512, ctx=256, scale=0.0625,
          storage="i2s", dialect="hf"):
    q_dim = n_head * hd
    kv_dim = n_kv * hd
    arch = "bitnet-b1.58"
    w = Writer()

    w.str_("general.architecture", arch)
    w.str_("general.name", "kraken-bitnet-%s-%s" % (dialect, storage))
    w.u32("general.alignment", ALIGN)
    w.u32(arch + ".block_count", n_layer)
    w.u32(arch + ".context_length", ctx)
    w.u32(arch + ".embedding_length", n_embd)
    w.u32(arch + ".feed_forward_length", n_ff)
    w.u32(arch + ".attention.head_count", n_head)
    w.u32(arch + ".attention.head_count_kv", n_kv)
    w.u32(arch + ".attention.key_length", hd)
    w.f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5)
    w.f32(arch + ".rope.freq_base", 10000.0)

    # tokenizer: a llama-style SPM vocab, the same shape a real BitNet file
    # carries. Real text does not matter here -- the point is a file that
    # loads, tokenizes and generates.
    tokens = ["<unk>", "<s>", "</s>"]
    scores = [-10.0, 0.0, 0.0]
    for i in range(256):
        tokens.append("<0x%02X>" % i)
        scores.append(-30.0)
    tokens.append("\u2581the")
    scores.append(-1.0)
    tokens.append("\u2581cat")
    scores.append(-1.5)
    tokens.append("\u2581sat")
    scores.append(-1.5)
    while len(tokens) < vocab:
        tokens.append("<extra%d>" % len(tokens))
        scores.append(-50.0)
    w.str_("tokenizer.ggml.model", "llama")
    w.str_array("tokenizer.ggml.tokens", tokens)
    w.f32_array("tokenizer.ggml.scores", scores)
    w.i32_array("tokenizer.ggml.token_type", [3, 3, 3] + [1] * 256 +
                [1, 1, 1] + [2] * (vocab - 262))
    w.u32("tokenizer.ggml.bos_token_id", 1)
    w.u32("tokenizer.ggml.eos_token_id", 2)
    w.bool_("tokenizer.ggml.add_bos_token", True)

    def matmul(name, n_out, n_in):
        if storage == "f16":
            w.tensor(name, TENSOR_F16, [n_in, n_out],
                     f16_weights(name, n_out, n_in, scale))
        else:
            w.tensor(name, TENSOR_I2_S, [n_in, n_out],
                     i2s_pack(name, n_out, n_in, scale, dialect))

    # token_embd and output_norm stay F16/F32 in a real BitNet file: only the
    # matmul weights are ternary.
    emb = []
    for i in range(n_embd * vocab):
        h = (i * 1664525 + 1013904223) & 0xFFFFFFFF
        emb.append(((h >> 16) % 9 - 4) * 0.02)
    w.tensor("token_embd.weight", TENSOR_F16, [n_embd, vocab],
             struct.pack("<%de" % len(emb), *emb))
    w.tensor("output_norm.weight", TENSOR_F32, [n_embd],
             struct.pack("<%df" % n_embd, *([1.0] * n_embd)))

    for l in range(n_layer):
        p = "blk.%d." % l
        w.tensor(p + "attn_norm.weight", TENSOR_F32, [n_embd],
                 struct.pack("<%df" % n_embd, *([1.0] * n_embd)))
        w.tensor(p + "ffn_norm.weight", TENSOR_F32, [n_embd],
                 struct.pack("<%df" % n_embd, *([1.0] * n_embd)))
        # SubLN: the pair that makes this graph BitNet's and not llama's.
        w.tensor(p + "attn_sub_norm.weight", TENSOR_F32, [n_embd],
                 struct.pack("<%df" % n_embd,
                             *[0.5 + 0.05 * (i % 5) for i in range(n_embd)]))
        w.tensor(p + "ffn_sub_norm.weight", TENSOR_F32, [n_ff],
                 struct.pack("<%df" % n_ff,
                             *[0.5 + 0.05 * (i % 7) for i in range(n_ff)]))
        matmul(p + "attn_q.weight", q_dim, n_embd)
        matmul(p + "attn_k.weight", kv_dim, n_embd)
        matmul(p + "attn_v.weight", kv_dim, n_embd)
        matmul(p + "attn_output.weight", n_embd, q_dim)
        matmul(p + "ffn_gate.weight", n_ff, n_embd)
        matmul(p + "ffn_up.weight", n_ff, n_embd)
        matmul(p + "ffn_down.weight", n_embd, n_ff)

    w.write(path)
    return {
        "path": path, "n_embd": n_embd, "n_layer": n_layer, "n_ff": n_ff,
        "vocab": vocab, "scale": scale, "storage": storage, "dialect": dialect,
    }


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--dialect", choices=["hf", "ms"], default="hf")
    ap.add_argument("--storage", choices=["i2s", "f16"], default="i2s")
    ap.add_argument("--scale", type=float, default=0.0625)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-kv", type=int, default=8)
    ap.add_argument("--hd", type=int, default=32)
    ap.add_argument("--n-ff", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=512)
    a = ap.parse_args(argv)
    info = build(a.out, n_embd=a.n_embd, n_layer=a.layers, n_head=a.n_head,
                 n_kv=a.n_kv, hd=a.hd, n_ff=a.n_ff, vocab=a.vocab,
                 scale=a.scale, storage=a.storage, dialect=a.dialect)
    print("wrote %(path)s  %(n_layer)dx%(n_embd)d ff=%(n_ff)d "
          "vocab=%(vocab)d  %(storage)s/%(dialect)s  scale=%(scale)g" % info)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
