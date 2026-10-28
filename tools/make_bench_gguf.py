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

MoE (`--experts N --experts-used K --expert-ff F`) writes the qwen3moe schema
instead: `blk.N.ffn_gate_inp.weight` [embd, N] and three stacked expert tensors
`ffn_{gate,up,down}_exps.weight` with ne = [in, out, N]. An all-MoE checkpoint
has no dense ffn_* tensors; the loader takes n_ff from expert_feed_forward_length.

Sizes: one expert costs 3 * embd * ff_exp * 2 bytes at f16, so a 2 GB
expert file is e.g. `--embd 768 --ff 768 --expert-ff 768 --layers 20
--experts 32 --experts-used 4 --vocab 32000` (2.3 GiB). At that size the
stack cannot be held in RAM, so expert tensors are streamed to the file as
they are generated and numpy (when importable) does the sampling:

    python3 tools/make_bench_gguf.py /tmp/fix/moe-2g-f16.gguf --embd 768 \
        --layers 20 --heads 12 --kv-heads 4 --ff 768 --expert-ff 768 \
        --experts 32 --experts-used 4 --vocab 32000 --ctx 2048

The numpy path is reproducible from `--seed` but is not byte-identical to the
pure-Python one; the dense llama path is unchanged and still pure Python. It
covers block formats too: `--format q4_0` on a 2 GiB expert stack is ~90 s with
numpy against ~30 min in pure Python, which is the difference between a usable
fixture and a coffee break.
"""
import argparse
import random
import struct
import sys

# ggml tensor type ids (ggml.h); the four this tool writes.
T_F32, T_F16, T_Q4_0, T_Q8_0 = 0, 1, 2, 8
# Ternary ids: TQ1_0 is ggml's BitNet format, PTQ1_0 is the Prism/Bonsai
# group-128 one.
T_TQ1_0, T_PTQ1_0 = 34, 143
# K-quants: 256-value blocks. Q2_K is 84 B (d, dmin, 16 packed scales,
# 64 B of 2-bit quants); Q3_K is 110 B (32 B hmask, 64 B quants, 12 B of
# scrambled 6-bit scales, d).
T_Q2_K, T_Q3_K = 10, 11
FORMATS = {
    "f32": (T_F32, 1, 0),
    "f16": (T_F16, 1, 0),
    "q8_0": (T_Q8_0, 34, 32),
    "q4_0": (T_Q4_0, 18, 32),
    # Ternary, for the TQ1_0 / PTQ1_0 dot kernels. The payload is written as
    # RAW BYTES rather than from the f32 stream, on purpose: a ternary block is
    # {-1,0,+1} behind one scale, and every byte pattern decodes to a legal
    # block. That makes this fixture a reference test rather than a plausible
    # one -- the dequantising path and the int8 kernel read the SAME bytes, so
    # any disagreement between them is a kernel bug and nothing else.
    "ptq1_0": (T_PTQ1_0, 28, 128),
    "tq1_0": (T_TQ1_0, 54, 256),
    # Raw-byte reference fixtures, same reasoning as the ternary ones: every
    # byte pattern is a legal K-quant block once the dtype slots are set to a
    # real f16, so the dequantising path and the int8 kernel read identical
    # bytes and any disagreement is a kernel bug.
    "q2_K": (T_Q2_K, 84, 256),
    "q3_K": (T_Q3_K, 110, 256),
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
        self.tensors.append((name, (ne0, ne1), dtype, data))

    def tensor_stream(self, name, dims, dtype, emit, nbytes):
        """A tensor emitted lazily.

        `emit(f)` writes exactly `nbytes` to the file handle. A routed-expert
        stack is [n_in, n_ff, n_expert] and one expert's slice is contiguous,
        so a caller writes the same bytes it would for a 2-D matrix, once per
        expert, without ever holding the whole stack in RAM -- which is the
        difference between generating a 2 GB expert file and OOM-ing on a 3 GB
        box. The same applies to a 32000 x 768 embedding: as a Python list of
        24.6 M floats it is ~800 MB of boxed doubles before it becomes bytes."""
        self.tensors.append((name, tuple(dims), dtype, (nbytes, emit)))

    def tensor3(self, name, ne0, ne1, ne2, dtype, emit, nbytes):
        self.tensor_stream(name, (ne0, ne1, ne2), dtype, emit, nbytes)

    def write(self, path):
        infos = b""
        sizes = []
        off = 0
        for (name, dims, dtype, data) in self.tensors:
            nbytes = len(data) if isinstance(data, bytes) else data[0]
            infos += struct.pack("<Q", len(name)) + name.encode()
            infos += struct.pack("<I", len(dims))  # n_dims
            for d in dims:
                infos += struct.pack("<Q", d)
            infos += struct.pack("<I", dtype) + struct.pack("<Q", off)
            sizes.append((nbytes, data))
            off += nbytes
            off = (off + ALIGN - 1) // ALIGN * ALIGN

        head = struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", GGUF_VERSION)
        head += struct.pack("<Q", len(self.tensors)) + struct.pack("<Q", self.n_kv)
        head += self.meta + infos
        head += b"\0" * ((-len(head)) % ALIGN)

        total = 0
        with open(path, "wb") as f:
            f.write(head)
            for (nbytes, data) in sizes:
                if isinstance(data, bytes):
                    f.write(data)
                    want = len(data)
                else:
                    want = data[0]
                    data[1](f)
                f.write(b"\0" * ((-want) % ALIGN))
                total += want
        return len(head) + total


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


_rnd = random.Random(0xC0FFEE)


def rnd_byte():
    return _rnd.getrandbits(8)


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
    if dtype in (T_Q2_K, T_Q3_K):
        nblk = len(vals) // blk
        raw = bytearray()
        for _ in range(nblk):
            b2 = bytearray(rnd_byte() for _ in range(_blk_bytes))
            if dtype == T_Q2_K:
                b2[80:82], b2[82:84] = f16(0.02), f16(0.01)
            else:
                b2[108:110] = f16(0.02)
            raw += b2
        return bytes(raw), dtype
    if dtype in (T_PTQ1_0, T_TQ1_0):
        # Random trit payload; the scale is a real f16 so the block is legal.
        nblk = len(vals) // blk
        raw = bytearray()
        for _ in range(nblk):
            raw += bytes(rnd_byte() for _ in range(_blk_bytes - 2))
            raw += f16(0.02)
        return bytes(raw), dtype
    raise ValueError(fmt)


SPM_EXTRA = [("\u2581the", -1.0), ("\u2581quick", -1.0), ("\u2581brown", -1.0),
             ("\u2581fox", -1.0), ("\u2581", -2.0), ("jumps", -1.0),
             ("\u2581over", -1.0), ("\u2581lazy", -1.0), ("\u2581dog", -1.0),
             ("\u2581and", -1.0), ("\u2581is", -1.0), ("\u2581a", -1.0)]

# numpy is optional. The pure-Python path stays the default and is the one that
# writes every fixture in the repo's docs; this only exists because a routed
# expert stack wide enough to spill out of RAM is ~10^9 weights, which pure
# Python would spend an hour producing. Same distribution (standard normal at
# the caller's scale), different generator, so a numpy fixture is reproducible
# from its seed but is NOT byte-identical to the pure-Python one.
try:
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only without numpy
    _np = None


def stream_matrix(n_in, n_out, fmt, seed, scale, n_expert=1):
    """Bytes for an [n_in, n_out, n_expert] stack, written without holding it.

    Returns (emit, nbytes), where `emit(f)` writes exactly nbytes. Every expert
    slice is contiguous and rows are whole blocks, so splitting at any
    multiple of the block size is bit-identical to quantising in one pass."""
    dtype, blk_bytes, blk = FORMATS[fmt]
    per = n_in * n_out
    total = per * n_expert
    if blk:
        assert per % blk == 0 and n_in % blk == 0, (n_in, n_out, blk)
    # A block format stores one block per `blk` elements, so the byte count is
    # blocks * block_bytes -- not elements * block_bytes. Getting that wrong
    # writes a tensor header larger than the data and the loader refuses the
    # file with "overruns", which is how this was found.
    unit = blk_bytes if blk else (4 if dtype == T_F32 else 2)
    nbytes = (total // blk) * unit if blk else total * unit

    if _np is not None and blk:
        chunk_np = min(per, 1 << 15) if per % 32 == 0 else 0
        chunk_np = max(32, (chunk_np // 32) * 32)

        def emit_q(f):
            gen = _np.random.default_rng(seed & 0xFFFFFFFF)
            left = total
            while left:
                take = min(left, chunk_np)
                v = (gen.standard_normal(take, dtype=_np.float32) * scale)
                f.write(_quant_np(v, fmt))
                left -= take
        return emit_q, nbytes

    if _np is None or blk:
        # Pure Python: exact and slow. Correct for any size, practical for the
        # small fixtures.
        rnd = random.Random(seed)

        def emit_py(f):
            left = total
            while left:
                take = min(left, 1 << 16)
                vals = [rnd.gauss(0.0, scale) for _ in range(take)]
                if blk:
                    vals = vals[:take - (take % blk)]
                data, _ = make_weights(vals, fmt)
                f.write(data)
                left -= len(vals)
            assert left == 0
        return emit_py, nbytes

    chunk = min(per, 1 << 18)  # elements per RNG draw
    np_dtype = _np.float32 if dtype == T_F32 else _np.float16

    def emit_np(f):
        gen = _np.random.default_rng(seed & 0xFFFFFFFF)
        left = total
        while left:
            take = min(left, chunk)
            v = (gen.standard_normal(take, dtype=_np.float32) * scale)
            f.write(v.astype(np_dtype).tobytes())
            left -= take
    return emit_np, nbytes


def _quant_np(v, fmt):
    """One numpy block-format pass, block-aligned. Matches the pure-Python
    quantisers in value terms (same amax/scale/rounding rule); the bytes are
    reproducible from the seed but are not expected to equal a pure-Python
    file, because the underlying RNG differs."""
    blk = 32
    assert v.size % blk == 0
    x = v.reshape(-1, blk)
    if fmt == "q8_0":
        amax = _np.max(_np.abs(x), axis=1)
        d = (_np.maximum(amax, 1e-8) / 127.0).astype(_np.float16)
        q = _np.clip(_np.round(x / d.astype(_np.float32)[:, None]),
                     -127, 127).astype(_np.int8)
        out = _np.empty((x.shape[0], 34), dtype=_np.uint8)
        out[:, :2] = d.view(_np.uint8).reshape(-1, 2)
        out[:, 2:] = q.view(_np.uint8)
        return out.tobytes()
    if fmt == "q4_0":
        amax = _np.maximum(-x.min(axis=1), x.max(axis=1))
        d = _np.where(amax > 0, amax / -8.0, 0.0)
        safe = _np.where(d != 0, d, 1.0)
        q = _np.clip(_np.round(x * (1.0 / safe)[:, None] + 8.0), 0,
                     15).astype(_np.uint8)
        packed = (q[:, :16] & 0x0F) | ((q[:, 16:] & 0x0F) << 4)
        out = _np.empty((x.shape[0], 18), dtype=_np.uint8)
        out[:, :2] = d.astype(_np.float16).view(_np.uint8).reshape(-1, 2)
        out[:, 2:] = packed
        return out.tobytes()
    raise ValueError(fmt)


def build(path, n_embd, n_layer, n_head, n_kv, hd, n_ff, vocab, ctx, fmt, seed,
          arch="llama", n_expert=0, n_expert_used=0, n_ff_exp=0, scale=0.04,
          fused=False, fused_ffn=None):
    # `fused` is the Q|K|V projection, `fused_ffn` the gate|up one; a Phi file
    # has both, and the two are separable so a test can fuse one at a time and
    # find which of them changed the logits.
    if fused_ffn is None:
        fused_ffn = fused
    rnd = random.Random(seed)
    q_dim, kv_dim = n_head * hd, n_kv * hd
    moe = n_expert > 0
    assert arch in ("llama", "phi3") or arch == "qwen3moe", arch
    assert not moe or (n_ff_exp > 0 and 0 < n_expert_used <= n_expert)

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
    g.string("general.architecture", arch)
    g.string("general.name", "kraken-bench-%d-%s" % (n_embd, fmt))
    g.u32("general.alignment", ALIGN)
    g.u32(arch + ".block_count", n_layer)
    g.u32(arch + ".context_length", ctx)
    g.u32(arch + ".embedding_length", n_embd)
    g.u32(arch + ".feed_forward_length", n_ff)
    g.u32(arch + ".attention.head_count", n_head)
    g.u32(arch + ".attention.head_count_kv", n_kv)
    g.u32(arch + ".attention.key_length", hd)
    g.u32(arch + ".attention.value_length", hd)
    g.f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5)
    g.f32(arch + ".rope.freq_base", 10000.0)
    if moe:
        g.u32(arch + ".expert_count", n_expert)
        g.u32(arch + ".expert_used_count", n_expert_used)
        g.u32(arch + ".expert_feed_forward_length", n_ff_exp)
    g.string("tokenizer.ggml.model", "llama")
    g.str_array("tokenizer.ggml.tokens", tokens)
    g.f32_array("tokenizer.ggml.scores", scores)
    g.u32("tokenizer.ggml.bos_token_id", 1)
    g.u32("tokenizer.ggml.eos_token_id", 2)
    g.boolean("tokenizer.ggml.add_bos_token", True)

    def w(name, n_in, n_out):
        # Ternary is emitted straight from a byte source: the f32 value list
        # below exists only to be quantised, and for a ternary block there is
        # nothing to quantise -- every byte pattern is a legal block. Skipping
        # it is what makes a model big enough to measure against (a 1024x4096
        # tensor is 4M Python floats otherwise, for bytes that are then thrown
        # away one line later).
        if FORMATS[fmt][0] in (T_PTQ1_0, T_TQ1_0, T_Q2_K, T_Q3_K):
            _t, _bb, _blk = FORMATS[fmt]
            assert (n_in * n_out) % _blk == 0, (name, n_in, n_out, _blk)
            nblk = (n_in * n_out) // _blk
            raw = bytearray()
            for _ in range(nblk):
                if _t == T_Q2_K:
                    b2 = bytearray(rnd_byte() for _ in range(_bb))
                    b2[80:82], b2[82:84] = f16(0.02), f16(0.01)
                    raw += b2
                elif _t == T_Q3_K:
                    b2 = bytearray(rnd_byte() for _ in range(_bb))
                    b2[108:110] = f16(0.02)
                    raw += b2
                else:
                    raw += bytes(rnd_byte() for _ in range(_bb - 2))
                    raw += f16(0.02)
            g.tensor(name, n_in, n_out, bytes(raw), _t)
            return
        # Streaming threshold: below it the pure-Python boxed-float list is
        # cheap and gives the fixtures their byte-for-byte reproducibility;
        # above it (an embedding table, a wide projection) the list is
        # hundreds of MB of Python floats and the numpy streamer writes it a
        # chunk at a time instead.
        if _np is not None and FORMATS[fmt][1] == 1 and n_in * n_out > (1 << 20):
            emit, nbytes = stream_matrix(n_in, n_out, fmt, rnd.getrandbits(32),
                                         scale)
            g.tensor_stream(name, (n_in, n_out), FORMATS[fmt][0], emit, nbytes)
            return
        vals = [rnd.gauss(0.0, scale) for _ in range(n_in * n_out)]
        data, dtype = make_weights(vals, fmt)
        g.tensor(name, n_in, n_out, data, dtype)

    def ws(name, n_in, n_out, n_exp):
        """A routed stack: one contiguous [n_in, n_out] slice per expert."""
        emit, nbytes = stream_matrix(n_in, n_out, fmt, rnd.getrandbits(32),
                                     scale, n_exp)
        g.tensor3(name, n_in, n_out, n_exp, FORMATS[fmt][0], emit, nbytes)

    def w_cat(name, parts):
        """One tensor whose row is the concatenation of `parts`, all n_in wide.

        Every row is a whole number of quantisation blocks, so quantising the
        concatenated stream produces byte-for-byte the same bytes the separate
        tensors would have carried. That is what lets a fused fixture and a
        split fixture hold the SAME weights, which is the only way to test a
        fused load: compare the two files' logits, not the layout."""
        assert all(p[0] == parts[0][0] for p in parts), "parts must share n_in"
        vals = []
        for (ni, no) in parts:
            vals.extend(rnd.gauss(0.0, scale) for _ in range(ni * no))
        data, dtype = make_weights(vals, fmt)
        g.tensor(name, parts[0][0], sum(p[1] for p in parts), data, dtype)

    def ones(name, n):
        g.tensor(name, n, 1, struct.pack("<%df" % n, *([1.0] * n)), T_F32)

    w("token_embd.weight", n_embd, vocab)
    ones("output_norm.weight", n_embd)
    w("output.weight", n_embd, vocab)
    for l in range(n_layer):
        p = "blk.%d." % l
        ones(p + "attn_norm.weight", n_embd)
        ones(p + "ffn_norm.weight", n_embd)
        if fused:
            # Phi-3: Q | K | V in one projection, and under GQA the K and V
            # blocks are kv_dim wide, not n_embd -- drawn q, k, v in that order
            # so the stream matches the split file's.
            w_cat(p + "attn_qkv.weight",
                  [(n_embd, q_dim), (n_embd, kv_dim), (n_embd, kv_dim)])
        else:
            w(p + "attn_q.weight", n_embd, q_dim)
            w(p + "attn_k.weight", n_embd, kv_dim)
            w(p + "attn_v.weight", n_embd, kv_dim)
        w(p + "attn_output.weight", q_dim, n_embd)
        if moe:
            # Router projection, then the three stacked expert tensors. An
            # all-MoE checkpoint legitimately has no dense ffn_* tensors: the
            # loader falls back to expert_feed_forward_length for n_ff.
            w(p + "ffn_gate_inp.weight", n_embd, n_expert)
            ws(p + "ffn_gate_exps.weight", n_embd, n_ff_exp, n_expert)
            ws(p + "ffn_up_exps.weight", n_embd, n_ff_exp, n_expert)
            ws(p + "ffn_down_exps.weight", n_ff_exp, n_embd, n_expert)
        else:
            if fused_ffn:
                # gate | up in one row, gate first (drawn gate-then-up).
                w_cat(p + "ffn_up.weight", [(n_embd, n_ff), (n_embd, n_ff)])
            else:
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
    ap.add_argument("--experts", type=int, default=0,
                    help="routed experts per MoE layer (0 = dense llama)")
    ap.add_argument("--experts-used", type=int, default=0,
                    help="experts routed per token (0 = 2)")
    ap.add_argument("--expert-ff", type=int, default=0,
                    help="expert hidden width (0 = --ff)")
    ap.add_argument("--phi3-qkv-only", action="store_true",
                    help="phi3 arch with only attn_qkv fused (bisecting aid)")
    ap.add_argument("--phi3-ffn-only", action="store_true",
                    help="phi3 arch with only gate|up fused (bisecting aid)")
    ap.add_argument("--phi3", action="store_true",
                    help="write the Phi-3 layout: arch string phi3, one fused "
                         "attn_qkv (Q|K|V) and one fused ffn_up (gate|up), no "
                         "ffn_gate tensor")
    ap.add_argument("--scale", type=float, default=0.04,
                    help="std-dev of the weights; a deep stack needs a "
                         "smaller value to stay in an f16 activation range")
    a = ap.parse_args(argv)

    heads = a.heads or max(1, a.embd // 64)
    kv_heads = a.kv_heads or heads
    ff = a.ff or (int(a.embd * 2.6875) // 32 * 32 or 32)
    ff_exp = a.expert_ff or ff
    used = a.experts_used or (2 if a.experts else 0)
    if a.head_dim * heads != a.embd and a.embd % heads:
        ap.error("--embd must be a multiple of --heads")
    for nm, val in (("embd", a.embd), ("ff", ff), ("head-dim", a.head_dim),
                    ("expert-ff", ff_exp)):
        if val % 32:
            ap.error("--%s must be a multiple of 32 (quant block size)" % nm)
    if a.experts and used > a.experts:
        ap.error("--experts-used cannot exceed --experts")

    size = build(a.out, a.embd, a.layers, heads, kv_heads, a.head_dim, ff,
                 a.vocab, a.ctx, a.format, a.seed,
                 arch="qwen3moe" if a.experts
                 else ("phi3" if (a.phi3 or a.phi3_qkv_only or a.phi3_ffn_only)
                       else "llama"),
                 n_expert=a.experts, n_expert_used=used, n_ff_exp=ff_exp,
                 scale=a.scale, fused=a.phi3 or a.phi3_qkv_only,
                 fused_ffn=a.phi3 or a.phi3_ffn_only)
    moe = ("%d experts top-%d, expert ff %d, " % (a.experts, used, ff_exp)
           if a.experts else "")
    print("wrote %s: %d layers, embd %d, %d/%d heads, hd %d, ff %d, %s"
          "vocab %d, %s, %.1f MiB" % (a.out, a.layers, a.embd, heads, kv_heads,
                                      a.head_dim, ff, moe, a.vocab, a.format,
                                      size / 1048576.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
