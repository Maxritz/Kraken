#!/usr/bin/env python3
"""Dump GGUF metadata keys (and value types) without touching tensor data.

Usage: gguf_keys.py model.gguf [filter_substring]
"""
import struct
import sys

TYPES = {
    0: ("UINT8", 1), 1: ("INT8", 1), 2: ("UINT16", 2), 3: ("INT16", 2),
    4: ("UINT32", 4), 5: ("INT32", 4), 6: ("FLOAT32", 4), 7: ("BOOL", 1),
    8: ("STRING", None), 9: ("ARRAY", None), 10: ("UINT64", 8), 11: ("INT64", 8),
}


def main(path, filt=None):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic not in (b"GGUF",):
            raise SystemExit(f"not a GGUF file: {magic!r}")
        def u32():
            return struct.unpack("<I", f.read(4))[0]

        def u64():
            return struct.unpack("<Q", f.read(8))[0]

        version = u32()
        # GGUF v3: both counts are u64 (the v2 u32 counts were never
        # shipped in the wild; every producer writes u64 today).
        n_tensors = u64()
        n_kv = u64()
        print(f"# version={version} tensors={n_tensors} metadata={n_kv}")



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
                    f.seek(n, 1) if TYPES[et][1] else None
                    return ["..."]
                return [value(et, depth + 1) for _ in range(n)]
            if size is None:
                raise SystemExit(f"unhandled type {t}")
            return f.read(size)

        # Metadata KVs are packed with no padding between them; the
        # section as a whole is what gets aligned before tensor info.
        keys = []
        for _ in range(n_kv):
            key = sval()
            t = u32()
            v = value(t)
            keys.append((key, t, v))
        # Tensor info follows the metadata directly. (The `alignment` key
        # applies to where tensor *data* starts, not to this section; seeking
        # 32 here desyncs the whole tensor list.)
        tensors = []
        for _ in range(n_tensors):
            name = sval()
            n_dims = u32()
            dims = [u64() for _ in range(n_dims)]
            ttype = u32()
            offset = u64()
            tensors.append((name, dims, ttype, offset))
        shown = 0
        for key, t, v in keys:
            if filt and filt not in key:
                continue
            if shown > 400:
                print("# ... truncated")
                break
            shown += 1
            if TYPES[t][0] == "STRING" and isinstance(v, str):
                print(f"{key} = {v}")
            elif TYPES[t][0] == "ARRAY":
                print(f"{key} = [{len(v)} items]")
            else:
                print(f"{key} = {v!r} ({TYPES[t][0]})")
        if tensors:
            print(f"# --- tensor info ({len(tensors)} tensors) ---")
            tshown = 0
            for name, dims, ttype, offset in tensors:
                if filt and filt not in name:
                    continue
                if tshown > 400:
                    print("# ... truncated")
                    break
                tshown += 1
                print(f"{name}  dims={dims} type={TYPES.get(ttype, ('?',))[0]} off={offset}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
