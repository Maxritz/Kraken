#!/usr/bin/env python3
"""Debug GGUF layout: locate the tensor info section and time the metadata parse."""
import struct
import sys
import time

TYPES = {
    0: ("UINT8", 1), 1: ("INT8", 1), 2: ("UINT16", 2), 3: ("INT16", 2),
    4: ("UINT32", 4), 5: ("INT32", 4), 6: ("FLOAT32", 4), 7: ("BOOL", 1),
    8: ("STRING", None), 9: ("ARRAY", None), 10: ("UINT64", 8), 11: ("INT64", 8),
}


def main(path):
    t0 = time.time()
    with open(path, "rb") as f:
        magic = f.read(4)
        u32 = lambda: struct.unpack("<I", f.read(4))[0]
        u64 = lambda: struct.unpack("<Q", f.read(8))[0]
        version = u32()
        n_tensors = u64()
        n_kv = u64()
        print(f"version={version} tensors={n_tensors} kv={n_kv}")

        def sval():
            n = u64()
            return f.read(n).decode("utf-8", "replace")

        def value(t, depth=0):
            name, size = TYPES[t]
            if t == 8:
                return sval()
            if t == 9:
                et = u32()
                n = u64()
                if depth > 2:
                    if TYPES[et][1]:
                        f.seek(n * TYPES[et][1], 1)
                    return ["..."]
                return [value(et, depth + 1) for _ in range(n)]
            if size is None:
                raise SystemExit(f"unhandled type {t}")
            return f.read(size)

        for i in range(n_kv):
            key = sval()
            t = u32()
            value(t)
            if i % 8 == 0:
                print(f"  kv[{i}] {key} @ {time.time()-t0:.2f}s", file=sys.stderr)
        pos = f.tell()
        print(f"metadata done @ {time.time()-t0:.2f}s, pos={pos}, pad={(-pos)%32}")

        # Candidate tensor-info start after 32-byte alignment.
        cand = pos + ((-pos) % 32)
        print(f"candidate tensor-info offset: {cand}")

        # Ground truth: scan the first 4 MiB for known tensor names.
        f.seek(0)
        head = f.read(4 * 1024 * 1024)
        for pat in (b"token_embd.weight", b"output.weight", b"blk.0.", b"blk.3."):
            needle = struct.pack("<Q", len(pat)) + pat
            idx = head.find(needle)
            print(f"  {pat.decode()}: name-field at file offset {idx}")

        # Show what the candidate offset actually contains.
        f.seek(cand)
        n = u64()
        print(f"at candidate: u64={n}")
        if 0 < n < 4096:
            print(f"  as string: {f.read(n)!r}")


if __name__ == "__main__":
    main(sys.argv[1])
