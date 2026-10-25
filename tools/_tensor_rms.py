import math
import struct
import sys

path = sys.argv[1]
f = open(path, 'rb')
magic = f.read(4)
ver, n_t, n_k = struct.unpack('<IQQ', f.read(20))


def readstr(fh):
    n = struct.unpack('<Q', fh.read(8))[0]
    return fh.read(n).decode('utf-8')


MUL = {0: 4, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

# GGUF v3 order: header, then the KV block, then the tensor infos.
for _ in range(n_k):
    readstr(f)
    t = struct.unpack('<I', f.read(4))[0]
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        if et == 8:
            for _ in range(n):
                readstr(f)
        else:
            f.read(MUL[et] * n)
    elif t == 8:
        readstr(f)
    else:
        f.read(MUL[t])

info = {}
for _ in range(n_t):
    name = readstr(f)
    nd = struct.unpack('<I', f.read(4))[0]
    ne = struct.unpack('<%dQ' % nd, f.read(8 * nd))
    t = struct.unpack('<I', f.read(4))[0]
    off = struct.unpack('<Q', f.read(8))[0]
    info[name] = (t, ne, off)
align = 32
f.seek((f.tell() + align - 1) // align * align)
data_start = f.tell()

for nm in sys.argv[2:]:
    hits = [k for k in info if nm in k]
    if not hits:
        print(nm, 'absent')
        continue
    for k in hits:
        t, ne, off = info[k]
        n = 1
        for d in ne:
            n *= d
        if t != 0:
            print('%-42s type=%d shape=%s' % (k, t, ne))
            continue
        f.seek(data_start + off)
        v = struct.unpack('<%df' % n, f.read(4 * n))
        rms = math.sqrt(sum(x * x for x in v) / len(v))
        print('%-42s n=%d rms=%.4f min=%.4f max=%.4f' % (k, n, rms, min(v), max(v)))
