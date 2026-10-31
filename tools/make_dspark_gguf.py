#!/usr/bin/env python3
"""Write a DSpark head-set GGUF fixture.

DSpark is a parallel block drafter: a small `blk.N` stack with non-causal
attention over a draft block, a fusion projection over the target's captured
hidden states, a Markov bigram head and a confidence head. There is no public
checkpoint of this family that this repo has validated, so the loader's schema
is exercised against a synthetic file whose *shape* is known and whose bytes are
reproducible -- the same rule the rest of the fixtures follow.

    python3 tools/make_dspark_gguf.py /tmp/fix/dspark.gguf \\
        --embd 1024 --layers 2 --heads 8 --kv-heads 4 --ff 2048 \\
        --vocab 16000 --target-layers 1,3 --rank 256 --block-size 8

What it does NOT do: pretend to be a real drafter. The weights are seeded noise,
so the block will propose noise tokens. What the fixture proves is the part that
can be proved without a checkpoint: the file is read, the geometry is checked
against the target it is handed (a mismatch is a refusal, not a resize), the
block runs, and the Markov head conditions row k on the token chosen at row k-1
-- which `tests/test_kraken.cpp` checks by turning the head off and watching the
chain change.

Pure Python, no numpy, same as make_bench_gguf.py.
"""
import argparse
import math
import struct
import sys

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGN = 32
(M_U32, M_I32, M_F32, M_BOOL, M_STR, M_ARR) = (4, 5, 6, 7, 8, 9)
(DT_F32, DT_F16) = (0, 1)


class Gguf:
    def __init__(self):
        self.meta = b""
        self.n_kv = 0
        self.tensors = []

    def _key(self, k):
        self.n_kv += 1
        b = k.encode()
        return struct.pack("<Q", len(b)) + b

    def u32(self, k, v):
        self.meta += self._key(k) + struct.pack("<I", M_U32) + struct.pack("<I", v)

    def i32(self, k, v):
        self.meta += self._key(k) + struct.pack("<I", M_I32) + struct.pack("<i", v)

    def f32(self, k, v):
        self.meta += self._key(k) + struct.pack("<I", M_F32) + struct.pack("<f", v)

    def boolean(self, k, v):
        self.meta += self._key(k) + struct.pack("<I", M_BOOL) + struct.pack("<?", v)

    def string(self, k, v):
        b = v.encode()
        self.meta += (self._key(k) + struct.pack("<I", M_STR) +
                      struct.pack("<Q", len(b)) + b)

    def i32_array(self, k, vals):
        out = self._key(k) + struct.pack("<I", M_ARR) + struct.pack("<I", M_I32)
        out += struct.pack("<Q", len(vals))
        out += struct.pack("<%di" % len(vals), *vals)
        self.meta += out

    def tensor(self, name, dims, dtype, data):
        self.tensors.append((name, tuple(dims), dtype, data))

    def f32_matrix(self, name, rows, cols, rng_scale, seed):
        vals = prng(rows * cols, seed, rng_scale)
        self.tensor(name, (cols, rows), DT_F32,
                    struct.pack("<%df" % len(vals), *vals))

    def f32_vector(self, name, n, rng_scale, seed):
        vals = prng(n, seed, rng_scale)
        self.tensor(name, (n,), DT_F32, struct.pack("<%df" % len(vals), *vals))

    def write(self, path):
        infos = b""
        off = 0
        sizes = []
        for (name, dims, dtype, data) in self.tensors:
            infos += struct.pack("<Q", len(name)) + name.encode()
            infos += struct.pack("<I", len(dims))
            for d in dims:
                infos += struct.pack("<Q", d)
            infos += struct.pack("<I", dtype) + struct.pack("<Q", off)
            sizes.append(data)
            off += len(data)
            off = (off + ALIGN - 1) // ALIGN * ALIGN
        head = struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", GGUF_VERSION)
        head += struct.pack("<Q", len(self.tensors)) + struct.pack("<Q", self.n_kv)
        head += self.meta + infos
        head += b"\0" * ((-len(head)) % ALIGN)
        total = 0
        with open(path, "wb") as f:
            f.write(head)
            for data in sizes:
                f.write(data)
                f.write(b"\0" * ((-len(data)) % ALIGN))
                total += len(data)
        return len(head) + total


def prng(n, seed, scale):
    """Deterministic uniform in [-scale, +scale]. xorshift32, no imports."""
    out = []
    st = seed & 0xFFFFFFFF or 1
    for _ in range(n):
        st ^= (st << 13) & 0xFFFFFFFF
        st ^= st >> 17
        st ^= (st << 5) & 0xFFFFFFFF
        st &= 0xFFFFFFFF
        out.append((st / 0xFFFFFFFF * 2.0 - 1.0) * scale)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--embd", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=0)
    ap.add_argument("--ff", type=int, default=2048)
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--target-layers", default="1,3")
    ap.add_argument("--target-embd", type=int, default=0,
                    help="width of the target's residual stream (default: --embd)")
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--markov-scale", type=float, default=0.0,
                    help="write markov_w2.scale when > 0")
    ap.add_argument("--no-markov", action="store_true")
    ap.add_argument("--no-confidence", action="store_true")
    ap.add_argument("--mask-id", type=int, default=12)
    ap.add_argument("--arch", default="dflash",
                    help="general.architecture to write (a real DSpark file says dflash)")
    ap.add_argument("--seed", type=int, default=0xD5A2C)
    ap.add_argument("--scale", type=float, default=0.02)
    a = ap.parse_args()

    target_embd = a.target_embd or a.embd
    hd = a.head_dim or a.embd // a.heads
    tl = [int(x) for x in a.target_layers.split(",") if x.strip()]
    n_enc = target_embd * len(tl)
    q_dim = a.heads * hd
    kv_dim = a.kv_heads * hd

    g = Gguf()
    P = a.arch + "."   # metadata travels under the arch string, as in every GGUF
    # A DSpark file is a DFlash file: the reference loads it with the DFlash
    # loader and adds the head when `markov_w1` is present. `--arch dspark` also
    # works; `dflash` is what a real checkpoint carries.
    g.string("general.architecture", a.arch)
    g.string("general.name", "kraken-dspark-fixture")
    g.u32(P + "block_count", a.layers)
    g.u32(P + "embedding_length", a.embd)
    g.u32(P + "feed_forward_length", a.ff)
    g.u32(P + "attention.head_count", a.heads)
    g.u32(P + "attention.head_count_kv", a.kv_heads)
    g.u32(P + "attention.key_length", hd)
    g.f32(P + "attention.layer_norm_rms_epsilon", 1e-6)
    g.u32(P + "vocab_size", a.vocab)
    g.u32(P + "context_length", 4096)
    g.u32(P + "block_size", a.block_size)
    g.i32(P + "mask_token_id", a.mask_id)
    g.boolean(P + "attention.causal", False)
    g.f32(P + "rope.freq_base", 10000.0)
    g.u32(P + "rope.dimension_count", hd)
    g.i32_array(P + "target_layers", tl)
    g.u32(P + "encoder.embd", n_enc)
    g.boolean(P + "markov.enabled", not a.no_markov)
    g.u32(P + "markov.rank", a.rank)
    g.u32("tokenizer.ggml.mask_token_id", a.mask_id)

    seed = a.seed
    sd = seed

    def mat(name, rows, cols, seed=None):
        nonlocal sd
        sd = sd if seed is None else seed
        sd += 7919
        g.f32_matrix(name, rows, cols, a.scale, sd)

    # Every matrix carries the `.weight` suffix a real GGUF uses -- the DFlash
    # loader reads that spelling, and a fixture without it would test a file no
    # converter produces.
    def mat(name, rows, cols, sv=None):
        nonlocal sd
        sd = sd if sv is None else sv
        sd += 7919
        g.f32_matrix(name + ".weight", rows, cols, a.scale, sd)

    def gain(name, n):
        # Gain vectors are 1-D and near 1.0: a norm gain of 0 would make the
        # fixture's residuals vanish and hide a broken block path.
        g.f32_vector(name, n, 0.05, hash(name) & 0xFFFF)

    mat("fc", a.embd, n_enc)
    gain("output_norm.weight", a.embd)
    gain("enc.output_norm.weight", a.embd)
    # One gain per capture slot, stacked: [n_embd, n_aux] in ggml order.
    g.f32_matrix("enc.aux_norm.weight", len(tl), a.embd, 0.05, 0x51A7)
    for l in range(a.layers):
        p = "blk.%d." % l
        gain(p + "attn_norm.weight", a.embd)
        gain(p + "ffn_norm.weight", a.embd)
        mat(p + "attn_q", q_dim, a.embd)
        mat(p + "attn_k", kv_dim, a.embd)
        mat(p + "attn_v", kv_dim, a.embd)
        mat(p + "attn_output", a.embd, q_dim)
        gain(p + "attn_q_norm.weight", hd)
        gain(p + "attn_k_norm.weight", hd)
        mat(p + "ffn_gate", a.ff, a.embd)
        mat(p + "ffn_up", a.ff, a.embd)
        mat(p + "ffn_down", a.embd, a.ff)
    # The Markov head, in the reference's own spelling: w1 and w2 are both
    # [rank, n_vocab] tables (ne[0] = rank = the contraction, ne[1] the
    # vocabulary), and w1 is read with a row gather indexed by the token.
    if not a.no_markov:
        mat("markov_w1", a.vocab, a.rank, seed + 1)
        mat("markov_w2", a.vocab, a.rank, seed + 2)
        if a.markov_scale > 0.0:
            g.f32_vector("markov_w2.scale", a.vocab, a.markov_scale, seed + 3)
    if not a.no_confidence:
        # conf = sigmoid(proj . [hidden ; w1_prev] + bias): the contraction is
        # n_embd + rank wide, as the reference builds it.
        mat("dspark_conf_proj", 1, a.embd + a.rank, seed + 4)
        g.f32_vector("dspark_conf_proj.bias", 1, 0.0, seed + 5)

    n = g.write(a.out)
    print("wrote %s: %.2f MiB, %d layers, embd %d, %d/%d heads, ff %d, "
          "block %d, rank %d, %d target layers"
          % (a.out, n / (1024.0 * 1024.0), a.layers, a.embd, a.heads,
             a.kv_heads, a.ff, a.block_size, a.rank, len(tl)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
