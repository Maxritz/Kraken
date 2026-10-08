#!/usr/bin/env python3
# gguf_scan.py — read a GGUF header and print what kraken needs to decide
# whether it can load the file: architecture, geometry, quantization types
# in use, MoE expert counts, mtp tensors, vocab size, file size.
#
# Usage: python3 gguf_scan.py file.gguf [file.gguf ...]
# Output: one JSON object per line (a JSON Lines report).
import struct, sys, os, json

GT = {0:'U8',1:'I8',2:'U16',3:'I16',4:'U32',5:'I32',6:'F32',7:'Bool',
      8:'String',9:'Array',10:'U64',11:'I64',12:'F64'}

# GGML type ids (on-disk GGUF quant ids). Ids kraken's DType enum defines
# are the ones it can ever load; anything else is reported raw.
DT = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',
      10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',
      16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',
      21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',24:'I8',25:'I16',26:'I32',27:'I64',
      28:'F64',29:'IQ1_M',30:'BF16',34:'TQ1_0',35:'TQ2_0',39:'MXFP4',40:'NVFP4',
      41:'Q1_0',42:'Q2_0',48:'Q2_0_64',50:'F8_E4M3FN',
      100:'Q4_0_ROCMFP4',101:'Q4_0_ROCMFP4_FAST',
      102:'Q6_0_ROCMFPX',104:'Q3_0_ROCMFPX',107:'Q2_0_ROCMFPX',
      142:'PQ2_0',143:'PTQ1_0'}

# Type ids with a known block geometry, measured by --audit. 42 is the one
# whose meaning is not fixed by the id alone: upstream llama.cpp's Q2_0 is a
# 64-value block (18 bytes, 2.25 bpw) while the llama-dx fork's is a 128-value
# block (34 bytes, 2.125 bpw). Both spell the id 42, so only the file's own
# tensor offsets say which one a file carries; the --audit mode below measures
# that and names the variant. 142 (PQ2_0) and 143 (PTQ1_0) are listed only so
# the audit prints their geometry string -- their ids are unambiguous.
BLOCK_GEOMETRY = {   # type id -> (block_bytes, block_values), id-only cases
    42: [(34, 128), (18, 64)],
    102: [(26, 32)],
    104: [(14, 32)],
    107: [(10, 32)],
    142: [(34, 128)],
    143: [(28, 128)],
}
# Keyed by (id, geometry): only id 42 needs an override, because its id alone
# cannot say which layout a file carries. 142/143 keep their DT names, so a
# (34, 128) geometry under id 142 stays PQ2_0 and is never renamed Q2_0.
NAME_BY_GEOMETRY = {(42, (34, 128)): 'Q2_0', (42, (18, 64)): 'Q2_0_64'}

def rd_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')

# How many elements of an array to keep in the report. Every element is still
# read — GGUF packs the values with no length prefix, so skipping any desyncs
# the rest of the file. (A tokenizer array is a quarter million strings; the
# report only needs to know it is there.)
KEEP = 64

def rd_val(f, t, keep=KEEP):
    if t == 0: return struct.unpack('<B', f.read(1))[0]
    if t == 1: return struct.unpack('<b', f.read(1))[0]
    if t == 2: return struct.unpack('<H', f.read(2))[0]
    if t == 3: return struct.unpack('<h', f.read(2))[0]
    if t == 4: return struct.unpack('<I', f.read(4))[0]
    if t == 5: return struct.unpack('<i', f.read(4))[0]
    if t == 6: return struct.unpack('<f', f.read(4))[0]
    if t == 7: return f.read(1) != b'\x00'
    if t == 8: return rd_str(f)
    if t == 10: return struct.unpack('<Q', f.read(8))[0]
    if t == 11: return struct.unpack('<q', f.read(8))[0]
    if t == 12: return struct.unpack('<d', f.read(8))[0]
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        if et == 9: raise ValueError('nested array')
        out = []
        for i in range(n):
            v = rd_val(f, et, 0)
            if i < keep: out.append(v)
        if n > keep: out.append('...%d more' % (n - keep))
        return out
    raise ValueError('bad gguf type %d' % t)

def scan(path):
    out = {'file': os.path.basename(path), 'size_mb': round(os.path.getsize(path)/1e6, 1)}
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                out['error'] = 'not a GGUF file (magic %r)' % magic
                return out
            ver, tc, mc = struct.unpack('<IQQ', f.read(20))
            out['gguf_version'] = ver
            meta = {}
            for _ in range(mc):
                k = rd_str(f)
                t = struct.unpack('<I', f.read(4))[0]
                try:
                    meta[k] = rd_val(f, t)
                except Exception:
                    meta[k] = '<unparsed>'
            arch = meta.get('general.architecture', '?')
            out['arch'] = arch
            out['name'] = meta.get('general.name', '')
            p = arch + '.'
            def gi(suffix, default=None):
                v = meta.get(p + suffix, None)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return int(v)
                return default
            out['n_layer'] = gi('block_count')
            out['n_embd'] = gi('embedding_length')
            out['n_head'] = gi('attention.head_count')
            out['n_head_kv'] = gi('attention.head_count_kv')
            out['head_dim'] = gi('attention.key_length')
            out['n_ff'] = gi('feed_forward_length')
            out['ctx'] = gi('context_length')
            out['n_expert'] = gi('expert_count', 0)
            out['n_expert_used'] = gi('expert_used_count')
            out['n_ff_exp'] = gi('expert_feed_forward_length')
            out['n_ff_shexp'] = gi('expert_shared_feed_forward_length')
            out['n_expert_shared'] = gi('expert_shared_count')
            out['rope_base'] = meta.get(p + 'rope.freq_base')
            tensors = []
            types = {}
            mtp = 0
            for _ in range(tc):
                name = rd_str(f)
                nd = struct.unpack('<I', f.read(4))[0]
                dims = struct.unpack('<' + 'Q' * nd, f.read(8 * nd))
                tt = struct.unpack('<I', f.read(4))[0]
                f.read(8)  # offset
                tensors.append((name, tt, dims))
                types[tt] = types.get(tt, 0) + 1
                if name.startswith('mtp.'):
                    mtp += 1
            out['quant_types'] = {DT.get(t, 'T%d' % t): n for t, n in sorted(types.items())}
            out['mtp_tensors'] = mtp
            for want in ('output.weight', 'token_embd.weight'):
                for name, tt, dims in tensors:
                    if name == want or (want == 'token_embd.weight' and name == 'tok_embeddings.weight'):
                        out['vocab'] = dims[0]
                        if want == 'output.weight':
                            out['out_type'] = DT.get(tt, 'T%d' % tt)
                        break
            out['n_tensors'] = tc

            # KV geometry PER LAYER, which the scalars above cannot express. A
            # hybrid keeps KV on a subset of its blocks (nemotron_h_moe: 6 of
            # 52) and a per-layer head count is legal (gemma4 declares
            # attention.head_count_kv as an array), so neither the charge nor
            # the number of layers a KV tier must hold is predictable from
            # `n_layer x n_head_kv`. Two independent statements of it come out
            # of the file itself:
            #
            #   kv_layers_by_tensor  blocks carrying an attn_q tensor, which is
            #                        the same predicate kraken's layer_has_kv
            #                        uses. null when the naming convention does
            #                        not apply to this architecture (a fused
            #                        attn_qkv, for instance), so a caller can
            #                        tell "absent" from "zero".
            #   n_head_kv_by_layer   the array verbatim, when it is an array.
            hkv = meta.get(p + 'attention.head_count_kv')
            if isinstance(hkv, list):
                out['n_head_kv_by_layer'] = [int(v) for v in hkv
                                             if isinstance(v, (int, float))]
            out['key_length'] = gi('attention.key_length')
            out['value_length'] = gi('attention.value_length')
            # The multi-token-prediction heads are the LAST
            # `<arch>.nextn_predict_layers` blocks, and the engine loads
            # block_count minus those (verdict.cpp: mtp_block_count, model.cpp
            # subtracts it and warns). A KV tier's working set is the LOADED
            # stack, so an attn_q in the skipped tail is not a layer the engine
            # will ever charge, tier or attend with. Measured: a qwen35 file
            # with 33 blocks declares nextn_predict_layers 1, so its ninth
            # attn_q block (blk.32) is a draft head -- the engine reports 8 KV
            # layers and this scanner, before it read the key, insisted on 9.
            blk = out.get('n_layer') or 0
            nextn = meta.get(p + 'nextn_predict_layers')
            if isinstance(nextn, list) and nextn:
                nextn = nextn[0]
            nextn = int(nextn) if isinstance(nextn, (int, float)) else 0
            if nextn < 0 or nextn > blk:
                nextn = 0
            out['nextn_predict_layers'] = nextn
            out['n_layer_loaded'] = (blk - nextn) if blk else None
            kvq = set()
            for name, tt, dims in tensors:
                if not name.endswith('attn_q.weight'):
                    continue
                parts = name.split('.')
                if len(parts) >= 2 and parts[0] in ('blk', 'block') \
                        and parts[1].isdigit():
                    kvq.add(int(parts[1]))
            if blk:
                kvq = {i for i in kvq if i < blk - nextn}
            out['kv_layers_at'] = sorted(kvq)
            out['kv_layers_by_tensor'] = len(kvq) if kvq else None
    except Exception as e:
        out['error'] = '%s: %s' % (type(e).__name__, e)
    return out

def audit(path, verbose=False):
    """Measure the file's own tensor spans and report bits per element.

    This is the diagnostic that identifies a mis-sized block: read the tensor
    infos, sort them by their own offset, and compare `next_offset - offset`
    against the computed span for that tensor's type. It is the only way to
    tell a 128-value Q2_0 from a 64-value one, because they share an id, and it
    also catches a tensor table that simply does not tile the file. Row padding
    shows up as a single alignment mismatch at the end of a shard, so the report
    counts them instead of failing on them.
    """
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            return {'file': os.path.basename(path), 'error': 'not a GGUF file'}
        struct.unpack('<IQQ', f.read(20))
        out = {'file': os.path.basename(path)}
        # Re-walk only far enough to reach the tensor table.
    import struct as _s
    with open(path, 'rb') as f:
        f.read(4)
        _, tc, mc = _s.unpack('<IQQ', f.read(20))
        for _ in range(mc):
            rd_str(f)
            t = _s.unpack('<I', f.read(4))[0]
            _skip_val(f, t)
        tensors = []
        for _ in range(tc):
            name = rd_str(f)
            nd = _s.unpack('<I', f.read(4))[0]
            dims = _s.unpack('<' + 'Q' * nd, f.read(8 * nd))
            tt = _s.unpack('<I', f.read(4))[0]
            off = _s.unpack('<Q', f.read(8))[0]
            tensors.append((name, tt, dims, off))
        end = _s.unpack('<Q', __import__('os').path.getsize(path))[0] if False else None
    ordered = sorted(tensors, key=lambda t: t[3])
    counts = {}
    for i, (name, tt, dims, off) in enumerate(ordered):
        n = 1
        for d in dims:
            n *= d
        if n == 0 or i + 1 >= len(ordered):
            continue
        span = ordered[i + 1][3] - off
        bpw = round(span * 8.0 / n, 5)
        key = (tt, bpw)
        counts.setdefault(key, []).append(name)
    out['geometry'] = []
    for (tt, bpw), names in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        entry = {'type': tt, 'name': DT.get(tt, 'T%d' % tt), 'bits_per_element': bpw,
                 'tensors': len(names)}
        for cand in BLOCK_GEOMETRY.get(tt, []):
            bb, bv = cand
            if abs(bpw - bb * 8.0 / bv) < 1e-4:
                entry['geometry'] = '%d bytes / %d values' % (bb, bv)
                entry['name'] = NAME_BY_GEOMETRY.get((tt, cand), entry['name'])
                break
        if verbose:
            entry['sample'] = names[:4]
        out['geometry'].append(entry)
    return out


def _skip_val(f, t, keep=0):
    if t == 8:
        rd_str(f)
        return None
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            _skip_val(f, et)
        return None
    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[t]
    f.read(size)
    return None


if __name__ == '__main__':
    args = sys.argv[1:]
    mode = 'scan'
    if args and args[0] in ('--audit', '--audit-verbose'):
        mode = args.pop(0)
    for path in args:
        if mode == 'scan':
            print(json.dumps(scan(path)))
        else:
            print(json.dumps(audit(path, verbose=(mode == '--audit-verbose'))))
