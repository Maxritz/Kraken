#!/usr/bin/env python3
"""Assemble an interleaved-A/B trace file from the harness's own run logs.

scripts/ab_perf.sh and scripts/determinism.sh keep every run under $OUT as
r<rep>-a<arm>.log/.err. This turns those into one readable artifact: the
decode line and the text hash per run, then the per-arm totals. It reads the
logs the harness actually wrote, so the trace cannot disagree with the run.
"""
import hashlib
import os
import re
import sys

out_dir = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "arm"
reps = max(int(n[1:].split("-")[0]) for n in os.listdir(out_dir)
           if n.endswith(".err"))

decode_re = re.compile(r"decode\s+(\d+) tok in ([\d.]+) ms = ([\d.]+) tok/s "
                       r"\(([\d.]+) ms/tok\)")
expert_re = re.compile(r"\[stats \] expert-path\s+(.*)")


def text_hash(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read().replace(b"\r", b"")).hexdigest()[:8]


print("# interleaved A/B, %d repetitions, arms alternate run by run" % reps)
print("# each arm's text hash is shown, so a speed win that changed the output")
print("# is visible in the same table")
print()
print("%-9s %-38s %-9s %-12s %s" % ("run", label, "text", "ms/tok", "tok/s"))
rows = {}
for a in range(2):
    for r in range(1, reps + 1):
        tag = "r%d-a%d" % (r, a)
        err = os.path.join(out_dir, tag + ".err")
        log = os.path.join(out_dir, tag + ".log")
        if not (os.path.exists(err) and os.path.exists(log)):
            continue
        with open(err, errors="replace") as fh:
            body = fh.read()
        m = decode_re.search(body)
        ms = m.group(4) if m else "-"
        tps = m.group(3) if m else "-"
        rows.setdefault(a, []).append((float(ms) if m else 0.0, text_hash(log)))
        print("%-9s a%-37s %-9s %-12s %s" % (tag, "", text_hash(log), ms, tps))

print()
for a in sorted(rows):
    ms = sorted(x[0] for x in rows[a])
    med = ms[len(ms) // 2]
    hashes = sorted(set(x[1] for x in rows[a]))
    print("a%d median %.2f ms/tok (%.2f tok/s), %d distinct output(s): %s" %
          (a, med, 1000.0 / med if med else 0.0, len(hashes),
           ", ".join(hashes)))

for tag in sorted(os.listdir(out_dir)):
    if not tag.endswith(".err"):
        continue
    with open(os.path.join(out_dir, tag), errors="replace") as fh:
        for line in fh:
            m = expert_re.search(line)
            if m:
                print("%-9s expert-path %s" % (tag[:-4], m.group(1).strip()))