#!/usr/bin/env python3
"""make_minimax_gguf.py -- write a small, self-consistent MiniMax-M3 GGUF.

The point of this tool is the SCHEMA, not the size: every tensor name and every
metadata key below is what llama.cpp's `src/models/minimax-m3.cpp` binds (read
2026-10-31), so a file written here is read by the loader through exactly the
path a 428B checkpoint would take. A fixture that spelled things differently
would prove the loader reads *a* minimax-m3, not *the* one.

What it writes, and why each piece is there:

  * `blk.N.attn_{q,k,v,output,norm,q_norm,k_norm}` -- ordinary GQA with a
    per-head QK-norm. The norm gains are head-dim long here, not n_embd: the
    reference creates them at `n_embd_head_k`.
  * the first `leading_dense_block_count` layers get `ffn_{gate,down,up}` and
    nothing else; every later layer gets a router (`ffn_gate_inp`), a router
    bias (`ffn_exp_probs_b.bias`), the routed banks (`ffn_{gate,up,down}_exps`),
    a wide shared expert (`ffn_{gate,up,down}_shexp`), and the four indexer
    tensors (`index_{q,k}_proj`, `index_{q,k}_norm`).
  * metadata: `expert_gating_func = 2` (sigmoid), `expert_weights_scale`,
    `expert_weights_norm`, `expert_shared_count`, and the five
    `attention.indexer.*` keys the block selector reads. `rope.dimension_count`
    is deliberately HALF the head dim -- MiniMax-M3 uses partial rotary, and a
    fixture that rotated the whole head would not exercise that path.

  python3 tools/make_minimax_gguf.py out.gguf --layers 3 --embd 128 ...

The file is a fixture: weights are a deterministic RNG walk, not a model. It
exists so the loader, the MSA block selector and the refusal paths can be run
against a real byte layout without a 428B download.
"""
import argparse
import struct
import sys

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32
M_U32, M_I32, M_F32, M_BOOL, M_STR = 4, 5, 6, 7, 8
T_F32 = 0


def prng(n, seed, scale):
    """A deterministic, well-spread walk: xorshift64* reduced to [-1, 1]."""
    s = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    out = []
    for _ in range(n):
        s ^= (s >> 12)
        s ^= (s << 25) & 0xFFFFFFFFFFFFFFFF
        s ^= (s >> 27)
        v = ((s * 2685821657736338717) & 0xFFFFFFFFFFFFFFFF) / 2.0 ** 63 - 1.0
        out.append(v * scale)
    return out


class Gguf:
    def __init__(self):
        self.meta = bytearray()
        self.n_kv = 0
        self.tensors = []

    def _key(self, k):
        b = k.encode()
        self.meta += struct.pack("<Q", len(b)) + b
        self.n_kv += 1

    def u32(self, k, v):
        self._key(k)
        self.meta += struct.pack("<II", M_U32, v)

    def i32(self, k, v):
        self._key(k)
        self.meta += struct.pack("<Ii", M_I32, v)

    def f32(self, k, v):
        self._key(k)
        self.meta += struct.pack("<If", M_F32, v)

    def boolean(self, k, v):
        self._key(k)
        self.meta += struct.pack("<IB", M_BOOL, 1 if v else 0)

    def string(self, k, v):
        self._key(k)
        b = v.encode()
        self.meta += struct.pack("<I", M_STR) + struct.pack("<Q", len(b)) + b

    def str_array(self, k, vals):
        self._key(k)
        self.meta += struct.pack("<II", 9, M_STR)          # array of strings
        self.meta += struct.pack("<Q", len(vals))
        for v in vals:
            b = v.encode()
            self.meta += struct.pack("<Q", len(b)) + b

    def f32_array(self, k, vals):
        self._key(k)
        self.meta += struct.pack("<II", 9, M_F32)
        self.meta += struct.pack("<Q", len(vals))
        self.meta += struct.pack("<%df" % len(vals), *vals)

    def tensor(self, name, dims, vals):
        self.tensors.append((name, tuple(dims),
                             struct.pack("<%df" % len(vals), *vals)))

    def write(self, path):
        infos = b""
        off = 0
        for (name, dims, data) in self.tensors:
            infos += struct.pack("<Q", len(name)) + name.encode()
            infos += struct.pack("<I", len(dims))
            for d in dims:
                infos += struct.pack("<Q", d)
            infos += struct.pack("<I", T_F32) + struct.pack("<Q", off)
            off += len(data)
            off = (off + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        head = GGUF_MAGIC + struct.pack("<I", GGUF_VERSION)
        head += struct.pack("<Q", len(self.tensors)) + struct.pack("<Q", self.n_kv)
        head += bytes(self.meta) + infos
        head += b"\0" * ((-len(head)) % ALIGNMENT)
        with open(path, "wb") as f:
            f.write(head)
            pos = 0
            for (_, _, data) in self.tensors:
                pad = (pos + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT - pos
                if pad:
                    f.write(b"\0" * pad)
                    pos += pad
                f.write(data)
                pos += len(data)
        return len(head) + off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dense-lead", type=int, default=1)
    ap.add_argument("--embd", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--rot-dim", type=int, default=16,
                    help="partial rotary: fewer than head_dim on purpose")
    ap.add_argument("--ff", type=int, default=256)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--experts-used", type=int, default=2)
    ap.add_argument("--ff-exp", type=int, default=64)
    ap.add_argument("--shared-experts", type=int, default=1)
    ap.add_argument("--idx-heads", type=int, default=2)
    ap.add_argument("--idx-dim", type=int, default=16)
    ap.add_argument("--idx-top-k", type=int, default=2)
    ap.add_argument("--idx-block", type=int, default=4)
    ap.add_argument("--idx-local", type=int, default=1)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--scale", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0x3A3)
    ap.add_argument("--no-indexer", action="store_true",
                    help="omit the indexer tensors: the loader must refuse")
    a = ap.parse_args()

    A = "minimax-m3."
    q_dim = a.heads * a.head_dim
    kv_dim = a.kv_heads * a.head_dim
    n_enc = a.ff_exp * a.shared_experts

    g = Gguf()
    g.string("general.architecture", "minimax-m3")
    g.string("general.name", "kraken-minimax-m3-fixture")
    g.u32("general.alignment", ALIGNMENT)
    g.u32(A + "block_count", a.layers)
    g.u32(A + "context_length", a.ctx)
    g.u32(A + "embedding_length", a.embd)
    g.u32(A + "feed_forward_length", a.ff)
    g.u32(A + "attention.head_count", a.heads)
    g.u32(A + "attention.head_count_kv", a.kv_heads)
    g.u32(A + "attention.key_length", a.head_dim)
    g.u32(A + "attention.value_length", a.head_dim)
    g.f32(A + "attention.layer_norm_rms_epsilon", 1e-6)
    g.u32(A + "rope.dimension_count", a.rot_dim)
    g.f32(A + "rope.freq_base", 10000.0)
    g.u32(A + "vocab_size", a.vocab)
    g.u32(A + "leading_dense_block_count", a.dense_lead)
    g.u32(A + "expert_count", a.experts)
    g.u32(A + "expert_used_count", a.experts_used)
    g.u32(A + "expert_feed_forward_length", a.ff_exp)
    g.u32(A + "expert_shared_count", a.shared_experts)
    g.f32(A + "expert_weights_scale", 2.5)
    g.boolean(A + "expert_weights_norm", True)
    g.i32(A + "expert_gating_func", 2)          # 2 = sigmoid, as the reference
    # The block selector's five keys. Without them a file cannot say how it
    # selects, and the loader refuses instead of running dense.
    g.u32(A + "attention.indexer.head_count", a.idx_heads)
    g.u32(A + "attention.indexer.key_length", a.idx_dim)
    g.u32(A + "attention.indexer.top_k", a.idx_top_k)
    g.u32(A + "attention.indexer.block_size", a.idx_block)
    g.u32(A + "attention.indexer.local_blocks", a.idx_local)

    # A tokenizer, because a model file that cannot tokenize a prompt is not a
    # model file. Byte-fallback style: the fixture only needs ids that resolve.
    toks = ["<unk>", "<s>", "</s>"] + ["tok%d" % i for i in range(a.vocab - 3)]
    g.string("tokenizer.ggml.model", "llama")
    g.str_array("tokenizer.ggml.tokens", toks)
    g.f32_array("tokenizer.ggml.scores", [0.0] * a.vocab)
    g.u32("tokenizer.ggml.bos_token_id", 1)
    g.u32("tokenizer.ggml.eos_token_id", 2)
    g.boolean("tokenizer.ggml.add_bos_token", True)

    seed = a.seed

    def mat(name, rows, cols, scale=None):
        nonlocal seed
        seed += 7919
        vals = prng(rows * cols, seed, a.scale if scale is None else scale)
        # ggml order: ne[0] is the contraction (columns), ne[1] the rows.
        g.tensor(name, (cols, rows), vals)

    def mat3(name, e0, e1, e2):
        """A routed expert bank: [contract, per-expert rows, expert] in ggml
        order, which is the 3-D layout the loader pages per expert."""
        nonlocal seed
        seed += 7919
        vals = prng(e0 * e1 * e2, seed, a.scale)
        g.tensor(name, (e0, e1, e2), vals)

    def gain(name, n):
        nonlocal seed
        seed += 7919
        g.tensor(name, (n,), [1.0 + v for v in prng(n, seed, 0.02)])

    mat("token_embd.weight", a.vocab, a.embd)
    gain("output_norm.weight", a.embd)
    mat("output.weight", a.vocab, a.embd)

    for l in range(a.layers):
        p = "blk.%d." % l
        gain(p + "attn_norm.weight", a.embd)
        gain(p + "ffn_norm.weight", a.embd)
        mat(p + "attn_q.weight", q_dim, a.embd)
        mat(p + "attn_k.weight", kv_dim, a.embd)
        mat(p + "attn_v.weight", kv_dim, a.embd)
        mat(p + "attn_output.weight", a.embd, q_dim)
        # Per-HEAD QK-norm: one head-dim gain applied to every head, which is
        # the shape llama.cpp creates for this model.
        gain(p + "attn_q_norm.weight", a.head_dim)
        gain(p + "attn_k_norm.weight", a.head_dim)
        if l < a.dense_lead:
            mat(p + "ffn_gate.weight", a.ff, a.embd)
            mat(p + "ffn_up.weight", a.ff, a.embd)
            mat(p + "ffn_down.weight", a.embd, a.ff)
            continue
        mat(p + "ffn_gate_inp.weight", a.experts, a.embd)
        gain(p + "ffn_exp_probs_b.bias", a.experts)
        # Routed banks are 3-D: one plane per expert, so a hot-expert cache can
        # page them independently. A 2-D bank is a different schema and the
        # loader refuses it by name.
        mat3(p + "ffn_gate_exps.weight", a.embd, a.ff_exp, a.experts)
        mat3(p + "ffn_up_exps.weight", a.embd, a.ff_exp, a.experts)
        mat3(p + "ffn_down_exps.weight", a.ff_exp, a.embd, a.experts)
        mat(p + "ffn_gate_shexp.weight", n_enc, a.embd)
        mat(p + "ffn_up_shexp.weight", n_enc, a.embd)
        mat(p + "ffn_down_shexp.weight", a.embd, n_enc)
        if a.no_indexer:
            continue
        mat(p + "index_q_proj.weight", a.idx_heads * a.idx_dim, a.embd)
        mat(p + "index_k_proj.weight", a.idx_dim, a.embd)
        gain(p + "index_q_norm.weight", a.idx_dim)
        gain(p + "index_k_norm.weight", a.idx_dim)

    n = g.write(a.out)
    print("wrote %s: %.1f MiB, %d layers (%d dense lead), embd %d, %d/%d heads, "
          "partial rope %d/%d, %d experts (%d used, %d shared), MSA top_k %d "
          "blocks of %d, local %d"
          % (a.out, n / 1048576.0, a.layers, a.dense_lead, a.embd, a.heads,
             a.kv_heads, a.rot_dim, a.head_dim, a.experts, a.experts_used,
             a.shared_experts, a.idx_top_k, a.idx_block, a.idx_local))
    return 0


if __name__ == "__main__":
    sys.exit(main())
