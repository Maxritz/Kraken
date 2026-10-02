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
      21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',24:'IQ1_M',25:'BF16',30:'BF16',
      40:'NVFP4'}

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
    except Exception as e:
        out['error'] = '%s: %s' % (type(e).__name__, e)
    return out

if __name__ == '__main__':
    for path in sys.argv[1:]:
        print(json.dumps(scan(path)))
