#!/usr/bin/env python3
"""Convert a published EAGLE-3 draft head (safetensors) into a kraken GGUF.

A draft head is not a model. It is one or two blocks that take the TARGET's own
hidden state plus the last token's embedding and propose k tokens, which the
target then verifies. The published artifacts (`yuhuili/EAGLE3-*`, the Arctic
speculators, the gpt-oss heads) ship as a single safetensors file of
projections, so this writes the shape `krk::DraftHead` loads: F16 weights,
`krk.draft.*` metadata, and a `d2t` map when the head's vocabulary is a subset
of the target's.

    python3 tools/eagle_to_gguf.py head.safetensors -o head.gguf \\
        --target-vocab 201088 --target-arch gpt-oss

    python3 tools/eagle_to_gguf.py head.safetensors -o head.gguf --inspect

The target vocabulary is REQUIRED (not inferred): a head whose `d2t` maps into a
vocabulary the target does not have would propose ids the verifier can never
accept, and Kraken refuses that pair at load rather than drafting badly. Read it
off the target with `python3 tools/gguf_meta.py MODEL | grep vocab`.

Geometry (hidden, heads, ff, ...) is inferred from tensor shapes where it can be
and can be overridden with the flags `--hidden`, `--target-hidden`, `--n-head`,
`--n-kv`, `--head-dim`, `--n-ff`, `--draft-vocab`, `--rope-theta`, `--rms-eps`.

The mapping is by tensor name with the usual prefixes stripped (`model.`,
`eagle.`, `midlayer.`) and `--map OLD=NEW` for anything else; `--inspect` prints
what the file holds and what would be written, and writes nothing.
"""
import argparse
import json
import math
import struct
import sys

# ---- GGUF tensor types this tool writes -----------------------------------
GGML_F32 = 0
GGML_F16 = 1

# safetensors dtypes -> (bytes per element, kind)
_ST_DTYPE = {
    "F32": (4, "f32"),
    "F16": (2, "f16"),
    "BF16": (2, "bf16"),
    "I32": (4, "i32"),
    "I64": (8, "i64"),
    "F64": (8, "f64"),
}

# Header value types
_V_U32, _V_I32, _V_F32, _V_STR, _V_U64 = 4, 5, 6, 8, 10

_STRIP = ("model.", "eagle.", "midlayer.", "speculator.", "draft_head.")


def _fail(msg):
    print("eagle_to_gguf: " + msg, file=sys.stderr)
    raise SystemExit(2)


# ---------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------
def read_safetensors(path):
    """{name: (dtype, shape, bytes)} plus the file's metadata block."""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 8:
        _fail("%s: too short to be safetensors" % path)
    (n,) = struct.unpack("<Q", raw[:8])
    if 8 + n > len(raw):
        _fail("%s: header length %d is past the end of the file" % (path, n))
    try:
        head = json.loads(raw[8:8 + n].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        _fail("%s: header is not JSON: %s" % (path, e))
    base = 8 + n
    meta = head.pop("__metadata__", {}) or {}
    out = {}
    for name, t in head.items():
        dt = t.get("dtype")
        if dt not in _ST_DTYPE:
            _fail("tensor %s has dtype %s, which this converter does not read"
                  % (name, dt))
        shape = list(t.get("shape") or [])
        b0, b1 = t.get("data_offsets") or (0, 0)
        if b1 < b0 or base + b1 > len(raw):
            _fail("tensor %s has data offsets past the end of the file" % name)
        out[name] = (dt, shape, raw[base + b0:base + b1])
    return out, meta


def _to_float(values, kind):
    """Decode a numeric payload to a list of floats (f64, python)."""
    if kind == "f32":
        return list(struct.unpack("<%df" % (len(values) // 4), values))
    if kind == "f16":
        return [struct.unpack("<e", values[i:i + 2])[0]
                for i in range(0, len(values), 2)]
    if kind == "bf16":
        out = []
        for i in range(0, len(values), 2):
            hi = struct.unpack("<H", values[i:i + 2])[0]
            out.append(struct.unpack("<f", struct.pack("<I", hi << 16))[0])
        return out
    if kind == "f64":
        return list(struct.unpack("<%dd" % (len(values) // 8), values))
    _fail("cannot decode %s as floats" % kind)


def _f16_bytes(floats, name):
    out = bytearray(len(floats) * 2)
    for i, v in enumerate(floats):
        if not math.isfinite(v):
            # A non-finite weight is a wrong file (a truncated or half-converted
            # checkpoint), not something to clamp: f16 would store inf and the
            # head would propose from garbage.
            _fail("tensor %s has a non-finite value at index %d" % (name, i))
        try:
            struct.pack_into("<e", out, i * 2, v)
        except OverflowError:
            _fail("tensor %s value %g at index %d is outside f16 range"
                  % (name, v, i))
    return bytes(out)


# ---------------------------------------------------------------------------
# name mapping
# ---------------------------------------------------------------------------
def canon(name):
    for p in _STRIP:
        if name.startswith(p):
            return name[len(p):]
    return name


# canonical -> candidate source names, in preference order
_SLOTS = {
    "fc.weight": ("fc.weight", "fc.linear.weight", "fc"),
    "norm.weight": ("norm.weight", "final_norm.weight", "norm"),
    "layers.0.input_layernorm.weight": (
        "layers.0.input_layernorm.weight", "layers.0.norm1.weight",
        "layers.0.attn_norm.weight"),
    "layers.0.post_attention_layernorm.weight": (
        "layers.0.post_attention_layernorm.weight", "layers.0.norm2.weight",
        "layers.0.ffn_norm.weight"),
    "layers.0.self_attn.q_proj.weight": (
        "layers.0.self_attn.q_proj.weight", "layers.0.q_proj.weight",
        "layers.0.attn.q_proj.weight"),
    "layers.0.self_attn.k_proj.weight": (
        "layers.0.self_attn.k_proj.weight", "layers.0.k_proj.weight",
        "layers.0.attn.k_proj.weight"),
    "layers.0.self_attn.v_proj.weight": (
        "layers.0.self_attn.v_proj.weight", "layers.0.v_proj.weight",
        "layers.0.attn.v_proj.weight"),
    "layers.0.self_attn.o_proj.weight": (
        "layers.0.self_attn.o_proj.weight", "layers.0.o_proj.weight",
        "layers.0.attn.o_proj.weight"),
    "layers.0.mlp.gate_proj.weight": (
        "layers.0.mlp.gate_proj.weight", "layers.0.gate_proj.weight"),
    "layers.0.mlp.up_proj.weight": (
        "layers.0.mlp.up_proj.weight", "layers.0.up_proj.weight"),
    "layers.0.mlp.down_proj.weight": (
        "layers.0.mlp.down_proj.weight", "layers.0.down_proj.weight"),
    "lm_head.weight": ("lm_head.weight", "head.weight", "lm_head", "head"),
    "d2t": ("d2t", "draft_to_target", "t2d_inv"),
}


def resolve(tensors, extra_map):
    """canonical name -> source tensor name, and the list of unread tensors."""
    by_canon = {}
    for name in tensors:
        by_canon.setdefault(canon(name), name)
    used = set()
    picked = {}
    for slot, candidates in _SLOTS.items():
        src = extra_map.get(slot)
        if src and src in tensors:
            picked[slot] = src
            used.add(src)
            continue
        for cand in candidates:
            if cand in by_canon:
                picked[slot] = by_canon[cand]
                used.add(by_canon[cand])
                break
            if cand in tensors:
                picked[slot] = cand
                used.add(cand)
                break
    # An explicit --map slot that names nothing is a mistake worth refusing.
    for slot, src in extra_map.items():
        if slot in _SLOTS and slot not in picked:
            _fail("--map %s=%s: no such tensor in the file" % (slot, src))
        elif slot not in _SLOTS:
            _fail("--map %s=...: %s is not a slot this converter writes"
                  % (slot, slot))
    return picked, [n for n in tensors if n not in used]


# ---------------------------------------------------------------------------
# GGUF writer
# ---------------------------------------------------------------------------
def _kv_u32(key, v):
    return _gguf_str(key) + struct.pack("<II", _V_U32, v)


def _kv_f32(key, v):
    return _gguf_str(key) + struct.pack("<If", _V_F32, v)


def _kv_str(key, v):
    return _gguf_str(key) + struct.pack("<I", _V_STR) + _gguf_str(v)


def _gguf_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, meta_kvs, tensors):
    """tensors: [(name, ggml_type, dims, payload)]; dims fastest-first."""
    align = 32
    head = bytearray()
    head += b"GGUF"
    head += struct.pack("<I", 3)
    head += struct.pack("<Q", len(tensors))
    head += struct.pack("<Q", len(meta_kvs))
    for kv in meta_kvs:
        head += kv
    offsets = []
    blob = bytearray()
    for name, ttype, dims, payload in tensors:
        off = len(blob)
        offsets.append(off)
        blob += payload
        pad = (-len(blob)) % align
        blob += b"\x00" * pad
        head += _gguf_str(name)
        head += struct.pack("<I", len(dims))
        for d in dims:
            head += struct.pack("<Q", d)
        head += struct.pack("<I", ttype)
        head += struct.pack("<Q", off)
    pad = (-len(head)) % align
    head += b"\x00" * pad
    with open(path, "wb") as f:
        f.write(head)
        f.write(blob)
    return len(head) + len(blob)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _shape(t):
    return list(t[1])


def main(argv):
    ap = argparse.ArgumentParser(add_help=True, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--target-vocab", type=int, default=0)
    ap.add_argument("--target-arch", default="")
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--target-hidden", type=int, default=0)
    ap.add_argument("--n-layer", type=int, default=1)
    ap.add_argument("--n-head", type=int, default=0)
    ap.add_argument("--n-kv", type=int, default=0)
    ap.add_argument("--head-dim", type=int, default=0)
    ap.add_argument("--n-ff", type=int, default=0)
    ap.add_argument("--n-in", type=int, default=1)
    ap.add_argument("--draft-vocab", type=int, default=0)
    ap.add_argument("--rope-theta", type=float, default=10000.0)
    ap.add_argument("--rms-eps", type=float, default=1e-5)
    ap.add_argument("--map", action="append", default=[], metavar="OLD=NEW",
                    help="remap one slot's source tensor (repeatable)")
    a = ap.parse_args(argv)

    tensors, st_meta = read_safetensors(a.input)
    extra = {}
    for m in a.map:
        if "=" not in m:
            _fail("--map needs SLOT=SOURCE, got %r" % m)
        k, v = m.split("=", 1)
        extra[k] = v

    picked, leftovers = resolve(tensors, extra)

    def shape_of(slot):
        src = picked.get(slot)
        return _shape(tensors[src]) if src else []

    fc = shape_of("fc.weight")
    if not fc:
        _fail("no fc (fusion) weight in %s -- not an EAGLE-3 head; --inspect "
              "lists the tensors it holds" % a.input)
    q = shape_of("layers.0.self_attn.q_proj.weight")
    k = shape_of("layers.0.self_attn.k_proj.weight")
    lm = shape_of("lm_head.weight")
    gate = shape_of("layers.0.mlp.gate_proj.weight")
    d2t_src = picked.get("d2t")

    # safetensors linear weights are [out, in]: fc is
    # [hidden, target_hidden * (n_in + 1)] because it fuses the captured target
    # slots and the last token's embedding into one row.
    hidden = a.hidden or (fc[0] if len(fc) == 2 else 0)
    target_hidden = a.target_hidden or (
        fc[1] // (a.n_in + 1) if len(fc) == 2 else 0)
    n_head = a.n_head or (q[0] // (a.head_dim or 0) if q and a.head_dim else 0)
    head_dim = a.head_dim or 0
    n_kv = a.n_kv or 0
    if k and head_dim:
        if not n_kv:
            n_kv = k[0] // head_dim
    if q and not head_dim:
        # Without --head-dim, the head count is the only free variable; a square
        # guess is not good enough, so require both.
        _fail("pass --n-head and --head-dim (q_proj is %d wide)" % q[0])
    n_ff = a.n_ff or (gate[0] if gate else 0)
    draft_vocab = a.draft_vocab or (lm[0] if lm else 0)
    target_vocab = a.target_vocab
    if a.inspect:
        print("== %s" % a.input)
        for name in sorted(tensors):
            dt, shape, payload = tensors[name]
            print("   %-58s %-5s %s" % (name, dt, "x".join(str(d) for d in shape)))
        print("   metadata: %s" % (st_meta or "{}"))
        print("-- resolved slots")
        for slot in _SLOTS:
            print("   %-46s %s" % (slot, picked.get(slot, "-")))
        if leftovers:
            print("-- not read (kept for a look, not written): %s" % ", ".join(leftovers))
        print("-- geometry")
        for label, v in (("hidden", hidden), ("target_hidden", target_hidden),
                         ("n_head", n_head), ("n_kv", n_kv), ("head_dim", head_dim),
                         ("n_ff", n_ff), ("draft_vocab", draft_vocab),
                         ("target_vocab", target_vocab),
                         ("has d2t", bool(d2t_src))):
            print("   %-16s %s" % (label, v))
        return 0

    if not a.output:
        _fail("either --inspect or -o FILE is required")
    if target_vocab <= 0:
        _fail("--target-vocab is required: a head whose ids were mapped into "
              "the wrong vocabulary drafts tokens the verifier can never accept")
    for label, v in (("hidden", hidden), ("target_hidden", target_hidden),
                     ("n_head", n_head), ("n_kv", n_kv), ("head_dim", head_dim),
                     ("n_ff", n_ff), ("draft_vocab", draft_vocab)):
        if v <= 0:
            _fail("%s could not be inferred (%d); pass it explicitly" % (label, v))
    if n_head % n_kv:
        _fail("n_head %d is not a multiple of n_kv %d" % (n_head, n_kv))

    # ---- payloads ---------------------------------------------------------
    out = []
    for slot, ttype, dims in (
            ("fc.weight", GGML_F16,
             (target_hidden * (a.n_in + 1), hidden)),
            ("norm.weight", GGML_F16, (hidden,)),
            ("layers.0.input_layernorm.weight", GGML_F16, (hidden,)),
            ("layers.0.post_attention_layernorm.weight", GGML_F16, (hidden,)),
            ("layers.0.self_attn.q_proj.weight", GGML_F16,
             (hidden, n_head * head_dim)),
            ("layers.0.self_attn.k_proj.weight", GGML_F16,
             (hidden, n_kv * head_dim)),
            ("layers.0.self_attn.v_proj.weight", GGML_F16,
             (hidden, n_kv * head_dim)),
            ("layers.0.self_attn.o_proj.weight", GGML_F16,
             (n_head * head_dim, hidden)),
            ("layers.0.mlp.gate_proj.weight", GGML_F16, (hidden, n_ff)),
            ("layers.0.mlp.up_proj.weight", GGML_F16, (hidden, n_ff)),
            ("layers.0.mlp.down_proj.weight", GGML_F16, (n_ff, hidden)),
            ("lm_head.weight", GGML_F16, (hidden, draft_vocab))):
        src = picked.get(slot)
        if not src:
            _fail("missing tensor for %s (--inspect lists what the file holds)"
                  % slot)
        dt, shape, payload = tensors[src]
        want = dims[0] * dims[1] if len(dims) == 2 else dims[0]
        if len(shape) == 2:
            got = shape[0] * shape[1]
        else:
            got = shape[0] if shape else 0
        if got != want:
            _fail("%s: shape %s has %d values, geometry needs %d -- pass the "
                  "right --hidden/--n-head/--head-dim" %
                  (slot, "x".join(str(d) for d in shape), got, want))
        kind = _ST_DTYPE[dt][1]
        if kind == "i32" or kind == "i64":
            _fail("%s is %s where a float weight was expected" % (slot, dt))
        out.append((slot, GGML_F16, dims,
                    _f16_bytes(_to_float(payload, kind), slot)))

    if d2t_src:
        dt, shape, payload = tensors[d2t_src]
        kind = _ST_DTYPE[dt][1]
        n = 1
        for d in shape:
            n *= d
        if kind == "i32":
            vals = list(struct.unpack("<%di" % n, payload))
        elif kind == "i64":
            vals = list(struct.unpack("<%dq" % n, payload))
        elif kind == "f32":
            vals = [int(round(v)) for v in _to_float(payload, "f32")]
        else:
            _fail("d2t has dtype %s; I32 is what the loader reads" % dt)
        if n != draft_vocab:
            _fail("d2t holds %d ids but the head's vocabulary is %d"
                  % (n, draft_vocab))
        for i, v in enumerate(vals):
            if v < 0 or v >= target_vocab:
                _fail("d2t[%d] = %d is outside the target vocabulary (0..%d)"
                      % (i, v, target_vocab - 1))
        # F32 and not GGML_TYPE_I32: the container Kraken reads describes
        # float and quantized tensor types only, so an integer entry is
        # reported as unknown and its payload is never reachable. 4 bytes an id
        # is 800 KiB at a 200k vocabulary -- the price of a readable map.
        out.append(("d2t", GGML_F32, (n,),
                    struct.pack("<%df" % n, *[float(v) for v in vals])))

    kvs = [
        _kv_str("general.architecture", "draft-head"),
        _kv_str("general.name", "eagle3 draft head"),
        _kv_str("krk.draft.kind", "eagle3"),
        _kv_str("krk.draft.source", a.input.replace("\\", "/").rsplit("/", 1)[-1]),
        _kv_u32("krk.draft.hidden", hidden),
        _kv_u32("krk.draft.target_hidden", target_hidden),
        _kv_u32("krk.draft.n_layer", a.n_layer),
        _kv_u32("krk.draft.n_head", n_head),
        _kv_u32("krk.draft.n_kv", n_kv),
        _kv_u32("krk.draft.head_dim", head_dim),
        _kv_u32("krk.draft.n_ff", n_ff),
        _kv_u32("krk.draft.draft_vocab", draft_vocab),
        _kv_u32("krk.draft.target_vocab", target_vocab),
        _kv_u32("krk.draft.n_in", a.n_in),
        _kv_f32("krk.draft.rope_theta", a.rope_theta),
        _kv_f32("krk.draft.rms_eps", a.rms_eps),
    ]
    if a.target_arch:
        kvs.append(_kv_str("krk.draft.target_arch", a.target_arch))

    n_bytes = write_gguf(a.output, kvs, out)
    print("wrote %s: %d tensor(s), %.1f MiB, hidden %d, %d head(s) x %d, "
          "%d KV head(s), ff %d, draft vocab %d%s -> target vocab %d"
          % (a.output, len(out), n_bytes / (1024.0 * 1024.0), hidden, n_head,
             head_dim, n_kv, n_ff, draft_vocab,
             " (d2t mapped)" if d2t_src else "", target_vocab))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
