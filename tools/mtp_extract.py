#!/usr/bin/env python3
"""mtp_extract.py -- repack an MTP (multi-token-prediction) head into a GGUF.

An MTP head is the speculative drafter that a checkpoint sometimes ships beside
its own weights: `mtp.*` tensors -- a norm pair, two input projections, and one
transformer block with its own attention and MoE. Strata keeps one as a 889 MB
`mtp-q2_0.gguf`; this tool builds the same kind of artifact from the raw BF16
tensors a checkpoint extraction leaves behind.

Two things it deliberately does NOT do:

  * No requantisation. The tensors are copied verbatim, so every tensor's sha256
    in the output equals the source manifest's. A quantiser is a second stage
    with its own parity gate; a drafter that silently reads the wrong bytes
    proposes wrong tokens, and the accept rate would fall quietly rather than
    loudly.
  * No guessing. A tensor whose dtype the manifest does not name, whose byte
    count disagrees, or whose hash does not match stops the run before a byte is
    written.

The output is a *head set*, not a model: no token embedding, no output head.
Its `general.architecture` is its own string (`qwen4-mtp` by default), which is
what lets the loader tell it from a draft model and refuse it with a reason
rather than reading it against llama defaults.

  python3 tools/mtp_extract.py --manifest DIR/mtp-manifest.json --out mtp.gguf

Source layout, as the reference extraction left it:

  C:/Strata-HIP-data/mtp/mtp-manifest.json     name, dtype, shape, bytes, sha256
  C:/Strata-HIP-data/mtp/tensors/*.bin         the raw tensor payloads

Exit status is 0 only when the written file re-reads with every tensor intact.
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32

# GGML type ids, as include/krk/quant.hpp spells them. Only the two a BF16
# checkpoint extraction produces: F32 for norms, BF16 for everything else.
TYPE = {"F32": 0, "BF16": 30}
TYPE_NAME = {v: k for k, v in TYPE.items()}

# gguf metadata value types
UINT32, STRING, UINT64 = 4, 8, 10


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def align_up(n, a):
    return (n + a - 1) // a * a


def put_string(buf, s):
    b = s.encode("utf-8")
    buf += struct.pack("<Q", len(b)) + b


class Kv:
    """The metadata block, counted as it is built.

    A wrong kv count makes the file unreadable, and a bytearray's length is no
    help, so the count belongs to the writer that appends the entries rather
    than to a second pass over the serialized bytes.
    """

    def __init__(self):
        self.buf = bytearray()
        self.n = 0

    def _key(self, key):
        put_string(self.buf, key)
        self.n += 1

    def u32(self, key, val):
        self._key(key)
        self.buf += struct.pack("<II", UINT32, val)

    def u64(self, key, val):
        self._key(key)
        self.buf += struct.pack("<IQ", UINT64, val)

    def string(self, key, val):
        self._key(key)
        self.buf += struct.pack("<I", STRING)
        put_string(self.buf, val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    help="the extraction manifest: name/dtype/shape/bytes/sha256/file")
    ap.add_argument("--out", help="the .gguf to write")
    ap.add_argument("--arch", default="qwen4-mtp",
                    help="the head set's own general.architecture string")
    ap.add_argument("--root", default=None,
                    help="prefix for the manifest's relative `file` paths "
                         "(default: the manifest's own directory)")
    ap.add_argument("--check-only", action="store_true",
                    help="verify the source hashes and stop before writing")
    ap.add_argument("--verify-only", action="store_true",
                    help="re-read an already written artifact against the manifest")
    ap.add_argument("--selftest", action="store_true",
                    help="hermetic proof that this tool fails when it should")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.manifest or not a.out:
        ap.error("--manifest and --out are required (or --selftest)")

    root = a.root or os.path.dirname(os.path.abspath(a.manifest))
    with open(a.manifest) as f:
        manifest = json.load(f)
    if not isinstance(manifest, list) or not manifest:
        print("mtp_extract: the manifest is not a non-empty list", file=sys.stderr)
        return 1

    if a.verify_only:
        return verify(a.out, manifest, a.arch)

    # ---- 1. the source, checked before anything is written -----------------
    tensors = []
    total = 0
    print("source tensors (%d)" % len(manifest))
    for t in manifest:
        name = t["name"]
        dtype = t["dtype"]
        if dtype not in TYPE:
            print("mtp_extract: %s has dtype %r, which this tool does not write"
                  % (name, dtype), file=sys.stderr)
            return 1
        path = os.path.join(root, t["file"])
        if not os.path.exists(path):
            print("mtp_extract: missing payload %s" % path, file=sys.stderr)
            return 1
        size = os.path.getsize(path)
        if size != t["bytes"]:
            print("mtp_extract: %s is %d bytes, the manifest says %d"
                  % (name, size, t["bytes"]), file=sys.stderr)
            return 1
        # The manifest's shapes are PyTorch order (outer first); GGUF stores
        # ne[] with the contiguous dimension first, so the list is reversed.
        ne = list(reversed(t["shape"]))
        digest = sha256_file(path)
        if digest != t["sha256"]:
            print("mtp_extract: %s hash %s != manifest %s"
                  % (name, digest[:16], t["sha256"][:16]), file=sys.stderr)
            return 1
        tensors.append({"name": name, "ne": ne, "type": TYPE[dtype], "path": path,
                        "bytes": size, "sha256": digest})
        total += size
        print("  ok  %-58s %-4s %-18s %10d B"
              % (name, dtype, "x".join(str(d) for d in t["shape"]), size))
    print("  %d tensors, %.1f MiB, every source hash matches the manifest"
          % (len(tensors), total / 1048576.0))
    if a.check_only:
        return 0

    # ---- 2. header, metadata, tensor table ---------------------------------
    kv = Kv()
    kv.string("general.architecture", a.arch)
    kv.string("general.name", "MTP head (%s)" % a.arch)
    kv.u32("general.alignment", ALIGNMENT)
    kv.u32("%s.block_count" % a.arch, 1)
    kv.string("%s.role" % a.arch,
              "speculative head set: no token embedding, no output head")
    kv.string("%s.extracted_by" % a.arch, "tools/mtp_extract.py")
    kv.string("%s.source_manifest_sha256" % a.arch, sha256_file(a.manifest))

    # A head's own geometry, read off the tensors rather than assumed: the width
    # from its input projection, the expert count from the expert axis of its
    # MoE, the hyper-connection width from its mixer.
    for t in tensors:
        if t["name"] == "mtp.fc_embedding.weight":
            kv.u32("%s.embedding_length" % a.arch, t["ne"][0])
        if t["name"].endswith("mlp.experts.gate_up_proj"):
            kv.u32("%s.expert_count" % a.arch, t["ne"][-1])
        # ne is in GGUF order (contiguous first), so the bottleneck width of a
        # [low_rank, width] down-projection is the last entry, not the first.
        if t["name"].endswith("hyper_connection_mixer.input_mix_weight_down.weight"):
            kv.u32("%s.hyper_connection.low_rank" % a.arch, t["ne"][-1])

    head = bytearray()
    head += GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, len(tensors), kv.n)
    head += kv.buf

    # ---- 3. write the data section, hashing what lands ---------------------
    off = 0
    infos = bytearray()
    for t in tensors:
        put_string(infos, t["name"])
        infos += struct.pack("<I", len(t["ne"]))
        for d in t["ne"]:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<IQ", t["type"], off)
        off = align_up(off + t["bytes"], ALIGNMENT)
    head += infos
    head += b"\0" * (align_up(len(head), ALIGNMENT) - len(head))

    tmp = a.out + ".part"
    written = {}
    try:
        with open(tmp, "wb") as f:
            f.write(head)
            pos = 0
            for t in tensors:
                pad = align_up(pos, ALIGNMENT) - pos
                if pad:
                    f.write(b"\0" * pad)
                    pos += pad
                h = hashlib.sha256()
                got = 0
                with open(t["path"], "rb") as src:
                    for chunk in iter(lambda: src.read(1 << 22), b""):
                        f.write(chunk)
                        h.update(chunk)
                        got += len(chunk)
                written[t["name"]] = h.hexdigest()
                pos += got
        os.replace(tmp, a.out)
    except OSError as e:
        if os.path.exists(tmp):
            os.unlink(tmp)
        print("mtp_extract: writing %s failed: %s" % (a.out, e), file=sys.stderr)
        return 1

    # ---- 4. re-read the artifact: header, table, and every payload hash ----
    if verify(a.out, manifest, a.arch) != 0:
        return 1
    print("\nwrote %s" % a.out)
    print("  %d tensors, %.1f MiB, every tensor's bytes hash to its manifest "
          "sha256 on the way out and back in"
          % (len(tensors), os.path.getsize(a.out) / 1048576.0))
    return 0


def run(args):
    """The tool as a caller meets it: a subprocess of this same file."""
    p = subprocess.run([sys.executable, os.path.abspath(__file__)] + args,
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def selftest():
    """Seven constructed cases, five of them failures.

    The artifact this tool writes is verified against a 4.9 GiB extraction that
    is not in the repo, so the verification that matters -- "does this tool
    notice a wrong byte?" -- has to be re-runnable without it. A checker that
    cannot fail proves nothing; these cases build their own manifest and then
    break it in the four ways that would otherwise reach a drafter.
    """
    import tempfile
    cases, ok = 0, 0
    with tempfile.TemporaryDirectory() as d:

        def write_manifest(payloads, path=os.path.join(d, "m.json")):
            entries = []
            for name, dtype, shape, blob in payloads:
                fn = os.path.join(d, name + ".bin")
                with open(fn, "wb") as f:
                    f.write(blob)
                entries.append({"name": name, "dtype": dtype, "shape": shape,
                                "bytes": len(blob), "sha256": sha256_bytes(blob),
                                "file": os.path.basename(fn)})
            with open(path, "w") as f:
                json.dump(entries, f)
            return path

        good = [("mtp.fc_embedding.weight", "BF16", [4, 8], bytes(range(64))),
                ("mtp.pre_fc_norm_embedding.weight", "F32", [4], bytes(16)),
                ("mtp.layers.0.mlp.experts.gate_up_proj", "BF16", [2, 2, 4],
                 bytes(range(32)))]
        m = write_manifest(good)
        out = os.path.join(d, "head.gguf")

        rc, log = run(["--manifest", m, "--out", out])
        cases += 1
        if rc == 0 and os.path.exists(out):
            ok += 1
        else:
            print("  a clean extraction failed: rc=%d\n%s" % (rc, log))

        rc, log = run(["--manifest", m, "--out", out, "--verify-only"])
        cases += 1
        if rc == 0:
            ok += 1
        else:
            print("  --verify-only rejected its own clean artifact: %s" % log)

        # 3. a payload byte that no longer matches the manifest hash
        with open(os.path.join(d, "mtp.fc_embedding.weight.bin"), "r+b") as f:
            f.seek(8)
            f.write(b"\xff")
        rc, log = run(["--manifest", m, "--out", os.path.join(d, "corrupt.gguf")])
        cases += 1
        if rc == 1 and not os.path.exists(os.path.join(d, "corrupt.gguf")):
            ok += 1
        else:
            print("  a corrupted source was not refused: rc=%d" % rc)

        # 4. a byte count the payload does not have
        m2 = write_manifest(good, os.path.join(d, "m2.json"))
        e = json.load(open(m2))
        e[0]["bytes"] += 2
        json.dump(e, open(m2, "w"))
        rc, log = run(["--manifest", m2, "--out", os.path.join(d, "short.gguf")])
        cases += 1
        if rc == 1:
            ok += 1
        else:
            print("  a wrong byte count was not refused: rc=%d" % rc)

        # 5. a dtype this tool does not write
        bad = list(good)
        bad[0] = ("mtp.fc_embedding.weight", "Q4_K", [4, 8], bytes(range(64)))
        m3 = write_manifest(bad, os.path.join(d, "m3.json"))
        rc, log = run(["--manifest", m3, "--out", os.path.join(d, "q.gguf")])
        cases += 1
        if rc == 1:
            ok += 1
        else:
            print("  an unwritable dtype was not refused: rc=%d" % rc)

        # 6. a byte flipped inside a *payload* of the written artifact. The aim
        #    matters: the writer pads each payload to the alignment, that padding
        #    is not data, and the payload hashes deliberately do not cover it --
        #    which is why this case aims at the last payload, whose bytes run to
        #    EOF (the fixture's last tensor is the 32-byte gate_up_proj).
        size = os.path.getsize(out)
        assert size > 8
        flip = os.path.join(d, "flipped.gguf")
        with open(out, "rb") as src, open(flip, "wb") as f:
            f.write(src.read())
        with open(flip, "r+b") as f:
            f.seek(size - 8)
            b = f.read(1)
            f.seek(size - 8)
            f.write(bytes([b[0] ^ 0xFF]))
        rc, log = run(["--manifest", m, "--out", flip, "--verify-only"])
        cases += 1
        if rc == 1:
            ok += 1
        else:
            print("  a flipped payload byte passed verification: rc=%d" % rc)

        # 7. a truncated artifact: the last payload is short of the size the
        #    table declares, which is the corruption a half-finished copy leaves.
        cut = os.path.join(d, "cut.gguf")
        with open(out, "rb") as src, open(cut, "wb") as f:
            f.write(src.read(size - 16))
        rc, log = run(["--manifest", m, "--out", cut, "--verify-only"])
        cases += 1
        if rc == 1:
            ok += 1
        else:
            print("  a truncated artifact passed verification: rc=%d" % rc)

    print("selftest: %d/%d cases behaved as required" % (ok, cases))
    return 0 if ok == cases else 1


def sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def verify(out, manifest, arch):
    """Prove the artifact says what it should: the arch string is its own, the
    tensor set is the manifest's, and every payload hashes to the manifest's
    sha256. Runs on the writer's own output and as `--verify-only`."""
    read = read_back(out)
    if read is None:
        print("mtp_extract: %s does not re-read as a GGUF" % out, file=sys.stderr)
        return 1
    meta, table = read
    if meta.get("general.architecture") != arch:
        print("mtp_extract: the artifact's architecture is %r, not %r"
              % (meta.get("general.architecture"), arch), file=sys.stderr)
        return 1
    if len(table) != len(manifest):
        print("mtp_extract: %d tensors in the artifact, %d in the manifest"
              % (len(table), len(manifest)), file=sys.stderr)
        return 1
    bad = 0
    for t in manifest:
        got = table.get(t["name"])
        if got is None:
            print("mtp_extract: %s is missing from the artifact" % t["name"],
                  file=sys.stderr)
            bad += 1
        elif got != t["sha256"]:
            print("mtp_extract: %s reads back as %s, the manifest says %s"
                  % (t["name"], got[:16], t["sha256"][:16]), file=sys.stderr)
            bad += 1
    if bad:
        print("mtp_extract: %d tensor(s) did not round-trip" % bad, file=sys.stderr)
        return 1
    print("verify %s" % out)
    print("  arch %r, 1 block, %d tensors, %d metadata keys, %d bytes"
          % (meta.get("general.architecture"), len(table), len(meta),
             os.path.getsize(out)))
    for key in ("%s.embedding_length", "%s.expert_count",
                "%s.hyper_connection.low_rank"):
        k = key % arch
        if k in meta:
            print("  %-34s %s" % (k, meta[k]))
    print("  all %d payloads hash to the manifest" % len(table))
    return 0


def read_back(path):
    """Minimal GGUF v3 reader, used only to prove the file we just wrote is the
    file we meant to write: metadata strings, tensor table, payload hashes."""
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            return None
        version, count, nkv = struct.unpack("<IQQ", f.read(20))
        if version != GGUF_VERSION:
            return None
        meta = {}
        for _ in range(nkv):
            n = struct.unpack("<Q", f.read(8))[0]
            key = f.read(n).decode("utf-8")
            t = struct.unpack("<I", f.read(4))[0]
            if t == STRING:
                n = struct.unpack("<Q", f.read(8))[0]
                meta[key] = f.read(n).decode("utf-8")
            elif t == UINT32:
                meta[key] = struct.unpack("<I", f.read(4))[0]
            elif t == UINT64:
                meta[key] = struct.unpack("<Q", f.read(8))[0]
            else:
                return None
        table = {}
        for _ in range(count):
            n = struct.unpack("<Q", f.read(8))[0]
            name = f.read(n).decode("utf-8")
            nd = struct.unpack("<I", f.read(4))[0]
            ne = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            t, off = struct.unpack("<IQ", f.read(12))
            size = 1
            for d in ne:
                size *= d
            nbytes = size * (2 if t in (TYPE["BF16"],) else 4)
            table[name] = (t, off, nbytes)
        align = meta.get("general.alignment", 32)
        data_start = align_up(f.tell(), align)
        out = {}
        for name, (t, off, nbytes) in table.items():
            f.seek(data_start + off)
            h = hashlib.sha256()
            left = nbytes
            while left:
                chunk = f.read(min(left, 1 << 22))
                if not chunk:
                    return None
                h.update(chunk)
                left -= len(chunk)
            out[name] = h.hexdigest()
    return meta, out


if __name__ == "__main__":
    sys.exit(main())
