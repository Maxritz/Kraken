"""Full numpy reference for LFM2.5-24B-A2B (`lfm2moe`), one forward at a time.

Built to answer two questions the engine cannot answer by itself:
  1. does a faithful reading of the reference graph reproduce llama.cpp's
     greedy text for this file? (if yes, the reading is right and the engine is
     wrong; if no, the reading is wrong and chasing the engine is wasted)
  2. where does the engine's row first leave that reading? `--trace STAGE`
     prints the same stage names kraken's KRK_DUMP prints, for one position.

Only the experts a token routes to are dequantised -- a 3-D GGUF tensor is
contiguous per expert (ne0 is fastest), so a slice is a byte range, not a copy
of the whole bank.
"""
import struct, sys, os
import numpy as np

PATH = sys.argv[1]
N_GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 12
STAGE = sys.argv[3] if len(sys.argv) > 3 else None
NORM_MOE = os.environ.get('NORM_MOE', '1') != '0'

f = open(PATH, 'rb')

def rd(n):
    b = f.read(n)
    assert len(b) == n, 'short read'
    return b

def u32(): return struct.unpack('<I', rd(4))[0]
def u64(): return struct.unpack('<Q', rd(8))[0]
def s():
    n = u64(); return rd(n).decode('utf-8', 'replace')

def val(t):
    if t == 0: return struct.unpack('<B', rd(1))[0]
    if t == 1: return struct.unpack('<b', rd(1))[0]
    if t == 2: return struct.unpack('<H', rd(2))[0]
    if t == 3: return struct.unpack('<h', rd(2))[0]
    if t == 4: return u32()
    if t == 5: return struct.unpack('<i', rd(4))[0]
    if t == 6: return struct.unpack('<f', rd(4))[0]
    if t == 7: return struct.unpack('<B', rd(1))[0]
    if t == 8: return s()
    if t == 9:
        at = u32(); n = u64(); return [val(at) for _ in range(n)]
    if t == 10: return u64()
    if t == 11: return struct.unpack('<q', rd(8))[0]
    if t == 12: return struct.unpack('<d', rd(8))[0]
    raise Exception('type %d' % t)

assert rd(4) == b'GGUF'
ver = u32()
nt = u64(); nkv = u64()
M = {}
for _ in range(nkv):
    k = s(); t = u32(); M[k] = val(t)
T = {}
for _ in range(nt):
    nm = s(); nd = u32(); dims = [u64() for _ in range(nd)]
    ty = u32(); off = u64()
    T[nm] = (dims, ty, off)
align = M.get('general.alignment', 32)
data0 = (f.tell() + align - 1) // align * align

F32, F16, Q4_0, Q6_K = 0, 1, 2, 14
BLK = {F32: (1, 4), F16: (1, 2), Q4_0: (32, 18), Q6_K: (256, 210)}

def _q4(a):
    d = a[:, :2].copy().view('<f2').astype(np.float32).reshape(-1, 1)
    q = a[:, 2:].astype(np.int32)
    w = np.empty((a.shape[0], 32), dtype=np.float32)
    w[:, :16] = (q & 0x0F)
    w[:, 16:] = (q >> 4)
    return ((w - 8.0) * d).reshape(-1)

def _q6(a):
    d = a[:, 208:210].copy().view('<f2').astype(np.float32).reshape(-1, 1)
    ql = a[:, :128].astype(np.int32)
    qh = a[:, 128:192].astype(np.int32)
    sc = a[:, 192:208].astype(np.int8).astype(np.float32)
    out = np.empty((a.shape[0], 256), dtype=np.float32)
    for n in (0, 128):
        qlo = ql[:, n // 2:n // 2 + 64]
        qho = qh[:, n // 4:n // 4 + 32]
        sco = sc[:, n // 16:n // 16 + 8]
        q1 = (qlo[:, :32] & 0xF) | (((qho >> 0) & 3) << 4)
        q2 = (qlo[:, 32:] & 0xF) | (((qho >> 2) & 3) << 4)
        q3 = (qlo[:, :32] >> 4) | (((qho >> 4) & 3) << 4)
        q4 = (qlo[:, 32:] >> 4) | (((qho >> 6) & 3) << 4)
        for half in (0, 1):
            l = slice(half * 16, half * 16 + 16)
            b = n + half * 16
            out[:, b:b + 16] = d * sco[:, 0 + half:1 + half] * (q1[:, l] - 32)
            out[:, b + 32:b + 48] = d * sco[:, 2 + half:3 + half] * (q2[:, l] - 32)
            out[:, b + 64:b + 80] = d * sco[:, 4 + half:5 + half] * (q3[:, l] - 32)
            out[:, b + 96:b + 112] = d * sco[:, 6 + half:7 + half] * (q4[:, l] - 32)
    return out.reshape(-1)

def _decode(ty, raw):
    if ty == F32:
        return np.frombuffer(raw, dtype='<f4').astype(np.float32)
    if ty == F16:
        return np.frombuffer(raw, dtype='<f2').astype(np.float32)
    if ty == Q4_0:
        return _q4(np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18))
    if ty == Q6_K:
        return _q6(np.frombuffer(raw, dtype=np.uint8).reshape(-1, 210))
    raise Exception('type %d' % ty)

_CACHE = {}

def tensor(nm):
    """Whole tensor, shaped [outer..][n_in] (GGUF ne0 is fastest)."""
    if nm in _CACHE:
        return _CACHE[nm]
    dims, ty, off = T[nm]
    blk, size = BLK[ty]
    count = 1
    for d in dims: count *= d
    assert count % blk == 0, (nm, count, ty)
    f.seek(data0 + off)
    v = _decode(ty, rd((count // blk) * size))
    v = v.reshape(list(reversed(dims)))
    _CACHE[nm] = v
    return v

def expert(nm, e):
    """One expert of a 3-D bank, as [n_out][n_in] -- its own byte range."""
    dims, ty, off = T[nm]
    blk, size = BLK[ty]
    per = dims[0] * dims[1]
    assert per % blk == 0
    f.seek(data0 + off + (e * per // blk) * size)
    v = _decode(ty, rd((per // blk) * size))
    return v.reshape(dims[1], dims[0])

def mm(x, nm):
    W = tensor(nm)
    return W @ x

N_EMBD = M['lfm2moe.embedding_length']
N_LAYER = M['lfm2moe.block_count']
N_HEAD = M['lfm2moe.attention.head_count']
HD = M['lfm2moe.attention.key_length'] if 'lfm2moe.attention.key_length' in M else 64
N_KV = M['lfm2moe.attention.head_count_kv']
N_USED = M['lfm2moe.expert_used_count']
L_CACHE = M['lfm2moe.shortconv.l_cache']
EPS = M['lfm2moe.attention.layer_norm_rms_epsilon']
THETA = M['lfm2moe.rope.freq_base']

def rms(x, w):
    return (x / np.sqrt((x * x).mean() + EPS) * w).astype(np.float32)

def silu(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)

ROPE = os.environ.get('ROPE', 'half')      # 'half' = neox pairing, 'int' = pairs
TAPS = os.environ.get('TAPS', 'fwd')        # 'fwd' = w[0] oldest, 'rev' = w[0] newest

def rope1(v, pos):
    """Half-split pairing, the form GGUF's converter permutes for."""
    half = HD // 2
    inv = 1.0 / (THETA ** (np.arange(half, dtype=np.float32) / half))
    ang = pos * inv
    c, s_ = np.cos(ang), np.sin(ang)
    if ROPE == 'int':
        a, b = v[0::2].copy(), v[1::2].copy()
        v[0::2] = a * c - b * s_
        v[1::2] = a * s_ + b * c
        return v
    a, b = v[:half].copy(), v[half:].copy()
    v[:half] = a * c - b * s_
    v[half:] = a * s_ + b * c
    return v

TRACE = None
LAYER = None
def trace(stage, row):
    if TRACE == stage:
        tag = '%s L%02d' % (stage, LAYER) if LAYER is not None else stage
        print('  %-14s %s' % (tag, ' '.join('%.7f' % v for v in row[:8])))

KV = {}
HID = {}   # HID[l] = the residual after layer l, for ad-hoc stage checks


def forward(tokens, pos0=0, want_stage=None):
    global TRACE, KV, LAYER
    if pos0 == 0:
        KV = {}
    TRACE = want_stage
    kv = KV
    x = tensor('token_embd.weight')[tokens[-1]].astype(np.float32)
    trace('embed', x)
    for l in range(N_LAYER):
        LAYER = l
        p = f'blk.{l}.'
        xn = rms(x, tensor(p + 'attn_norm.weight'))
        if p + 'shortconv.in_proj.weight' in T:
            proj = mm(xn, p + 'shortconv.in_proj.weight')
            B, C, X = proj[:N_EMBD], proj[N_EMBD:2 * N_EMBD], proj[2 * N_EMBD:]
            bx = B * X
            hist = kv.get(('conv', l))
            if hist is None:
                hist = np.zeros((L_CACHE - 1, N_EMBD), dtype=np.float32)
            buf = np.concatenate([hist, bx[None, :]], axis=0)
            convw = tensor(p + 'shortconv.conv.weight')      # [ch, kern]
            y = np.zeros(N_EMBD, dtype=np.float32)
            for k in range(L_CACHE):
                tap = k if TAPS == 'fwd' else (L_CACHE - 1 - k)
                y += buf[k] * convw[:, tap]
            kv[('conv', l)] = buf[1:]
            y = C * y
            trace('lfm.sc', y)
            x = x + mm(y, p + 'shortconv.out_proj.weight')
        else:
            nkv = N_KV[l]
            qn, kn = tensor(p + 'attn_q_norm.weight'), tensor(p + 'attn_k_norm.weight')
            q = mm(xn, p + 'attn_q.weight').reshape(N_HEAD, HD)
            q = np.stack([rms(q[h], qn) for h in range(N_HEAD)])
            trace('attn.qk', q.reshape(-1))
            q = np.stack([rope1(q[h], pos0) for h in range(N_HEAD)])
            k = mm(xn, p + 'attn_k.weight').reshape(nkv, HD)
            k = np.stack([rope1(rms(k[h], kn), pos0) for h in range(nkv)])
            trace('attn.q', q.reshape(-1))
            trace('attn.k', k.reshape(-1))
            v = mm(xn, p + 'attn_v.weight').reshape(nkv, HD)
            kh, vh = kv.get(('k', l)), kv.get(('v', l))
            kh = k[None] if kh is None else np.concatenate([kh, k[None]], 0)
            vh = v[None] if vh is None else np.concatenate([vh, v[None]], 0)
            kv[('k', l)], kv[('v', l)] = kh, vh
            rep = N_HEAD // nkv
            kk = np.repeat(kh, rep, axis=1)
            vv = np.repeat(vh, rep, axis=1)
            sc = np.einsum('hd,khd->hk', q, kk) * (HD ** -0.5)
            sc = sc - sc.max(axis=-1, keepdims=True)
            pr = np.exp(sc)
            pr /= pr.sum(axis=-1, keepdims=True)
            out = np.einsum('hk,khd->hd', pr, vv).reshape(-1)
            trace('attn', out)
            x = x + mm(out, p + 'attn_output.weight')
        xn = rms(x, tensor(p + 'ffn_norm.weight'))
        trace('ffn.pre', xn)
        if p + 'ffn_gate_inp.weight' in T:
            logits = mm(xn, p + 'ffn_gate_inp.weight')
            # The selection bias shifts WHICH experts are picked; the weight an
            # expert is mixed with is its own sigmoid, un-biased (llama.cpp's
            # ffn_exp_probs_b is added to the top-k logits only).
            sel = logits + tensor(p + 'exp_probs_b.bias') if p + 'exp_probs_b.bias' in T else logits
            prob_sel = 1.0 / (1.0 + np.exp(-sel))
            prob_w = 1.0 / (1.0 + np.exp(-logits))
            idx = np.argsort(-prob_sel)[:N_USED]
            w = prob_w[idx].copy()
            if NORM_MOE:
                w = w / w.sum()
            trace('ffn.router', logits)
            acc = np.zeros(N_EMBD, dtype=np.float32)
            for e, we in zip(idx, w):
                g = expert(p + 'ffn_gate_exps.weight', int(e))
                u = expert(p + 'ffn_up_exps.weight', int(e))
                d = expert(p + 'ffn_down_exps.weight', int(e))
                acc += we * (d @ (silu(g @ xn) * (u @ xn)))
            trace('moe.sum', acc)
            trace('moe.idx', np.array([float(i) for i in idx]))
            x = x + acc
        else:
            g = mm(xn, p + 'ffn_gate.weight')
            u = mm(xn, p + 'ffn_up.weight')
            trace('attn.ffng', g)
            x = x + mm(silu(g) * u, p + 'ffn_down.weight')
        trace('rec.post', x)
        HID[l] = x
    xf = rms(x, tensor('token_embd_norm.weight'))
    head = tensor('output.weight') if 'output.weight' in T else tensor('token_embd.weight')
    return head @ xf

PROMPT = [597, 5205, 302, 3980, 355]   # "The capital of France is"

if STAGE:
    for t in range(len(PROMPT) - 1):
        forward(PROMPT[:t + 1], t)
    forward(PROMPT, len(PROMPT) - 1, STAGE)
    sys.exit(0)

cur = list(PROMPT)
for t in range(len(PROMPT) - 1):
    forward(cur[:t + 1], t)
pos = len(PROMPT) - 1
out = []
for _ in range(N_GEN):
    nxt = int(np.argmax(forward(cur, pos)))
    out.append(nxt)
    cur.append(nxt)
    pos += 1
print('norm_moe', NORM_MOE, 'rope', ROPE, 'taps', TAPS)
print('tokens', out)
toks = M['tokenizer.ggml.tokens']
print('text  ', repr(''.join(toks[i] for i in out)))
