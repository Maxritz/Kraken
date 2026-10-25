"""Print a GGUF's tensor names (and count them), filtered by a regex.

The engine's --info prints the model it BUILT, so it cannot answer "is there an
MTP head in this file that the loader ignored". This reads the tensor table
directly: names only, no weight data, so it is fast on a 22 GiB file.
"""
import re
import struct
import sys

GGUF_MAGIC = 0x46554747


def rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def rstr(f):
    n = rd(f, "<Q")
    return f.read(n).decode("utf-8", "replace")


def read_value(f, t):
    """A value as a printable scalar/array, for the metadata dump."""
    if t == 8:
        return rstr(f)
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
           6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}.get(t)
    if fmt:
        return rd(f, fmt)
    if t == 9:
        at = rd(f, "<I")
        n = rd(f, "<Q")
        out = [read_value(f, at) for _ in range(n)]
        return out[:6] + (["...%d more"] if len(out) > 6 else [])
    raise ValueError("bad gguf value type %d" % t)


def skip_value(f, t):
    # gguf value types
    if t in (0, 1, 7):      # u8 i8 bool
        f.read(1)
    elif t in (2, 3):       # u16 i16
        f.read(2)
    elif t in (4, 5, 6):    # u32 i32 f32
        f.read(4)
    elif t in (10, 11, 12):  # u64 i64 f64
        f.read(8)
    elif t == 8:            # string
        f.read(rd(f, "<Q"))
    elif t == 9:            # array
        at = rd(f, "<I")
        n = rd(f, "<Q")
        for _ in range(n):
            skip_value(f, at)
    else:
        raise ValueError("bad gguf value type %d" % t)


def main():
    path = sys.argv[1]
    pattern = re.compile(sys.argv[2] if len(sys.argv) > 2 else ".*", re.I)
    with open(path, "rb") as f:
        if rd(f, "<I") != GGUF_MAGIC:
            raise SystemExit("not a gguf")
        version = rd(f, "<I")
        n_tensors = rd(f, "<Q")
        n_kv = rd(f, "<Q")
        kv = []
        for _ in range(n_kv):
            k = rstr(f)
            t = rd(f, "<I")
            if "--kv" in sys.argv:
                kv.append((k, t, read_value(f, t)))
            else:
                skip_value(f, t)
        names = []
        for _ in range(n_tensors):
            name = rstr(f)
            n_dims = rd(f, "<I")
            f.read(8 * n_dims)   # dims
            rd(f, "<I")          # type
            rd(f, "<Q")          # offset
            names.append(name)
    print("gguf v%d, %d tensors" % (version, len(names)))
    if "--kv" in sys.argv:
        for k, t, v in kv:
            if pattern.search(k):
                print("  kv %s = %r" % (k, v))
    hits = [n for n in names if pattern.search(n)]
    for n in hits:
        print("  " + n)
    print("%d name(s) matched %r" % (len(hits), pattern.pattern))


main()
