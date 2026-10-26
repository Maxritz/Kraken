#!/usr/bin/env bash
# Three-length fit of the prefill attention cost: `a ns per (row,key) pair +
# b us per block`.
#
# The point of the fit is that a change to the kernel's *per-block* fixed work
# (the Q stage, the row epilogue) and a change to its per-pair work (the KV
# stage, the dot, the softmax chain) land in different coefficients, and only
# three prompt lengths tell them apart. Interleaved run by run for the same
# reason every other probe here is: a sequential A-then-B reads drift.
#
# Usage:
#   tools/attn_fit.sh <reps> "<env arm A>" "<env arm B>" ...
#
# An arm is an environment string, and `KR_BIN=<path>` in it compares two
# BINARIES rather than two env knobs -- that is the form an arm uses when the
# change under test is not switchable at runtime.
set -uo pipefail

REPS=${1:?repetitions}
shift
ARMS=("$@")
[ ${#ARMS[@]} -ge 1 ] || { echo "need at least one arm" >&2; exit 2; }

ROOT=$(cd "$(dirname "$0")/.." && pwd)
KR=${KR:-$ROOT/build-hip/kraken.exe}
OUT=${OUT:-/tmp/krk-attn-fit}
MODELS=${MODELS:-/g/More-models/Qwen3-8B-Q4_K_M.gguf}
# The shape the pair/block counts come from: Qwen3-8B is 36 layers x 32 query
# heads at hd 128, so QT is the kernel's default 8.
LAYERS=${LAYERS:-36}
HEADS=${HEADS:-32}
QT=${QT:-8}
# Prompt lengths, as repetitions of one phrase; the engine reports the token
# count it actually built, and the fit reads that rather than assuming.
LENS=${LENS:-"25 50 100"}

rm -rf "$OUT"
mkdir -p "$OUT"
: > "$OUT/points.tsv"

cleanup() { taskkill //F //IM kraken.exe //T >/dev/null 2>&1 || true; }
trap cleanup EXIT

for ((r = 1; r <= REPS; r++)); do
  for a in "${!ARMS[@]}"; do
    arm=${ARMS[$a]}
    K=$(printf '%s' "$arm" | tr ' ' '\n' | sed -n 's/^KR_BIN=//p')
    K=${K:-$KR}
    for m in $MODELS; do
      for N in $LENS; do
        P=$(python3 -c "print('The capital of France is '*$N, end='')")
        tag="r${r}-a${a}-n${N}"
        # shellcheck disable=SC2086
        env $arm KRK_TIME=attention timeout 900 "$K" -m "$m" \
          -p "$P" --max-tokens 1 --temp 0 --greedy \
          > "$OUT/$tag.log" 2>&1
        rc=$?
        T=$(sed -n 's/.*prefill *\([0-9]*\) tok in .*/\1/p' "$OUT/$tag.log" | tail -1)
        pre=$(sed -n 's/.*prefill .* = \([0-9.]*\) tok\/s.*/\1/p' "$OUT/$tag.log" | tail -1)
        # The per-op table's own row: `attention <calls> <raw> <floor> <NET> ...`
        # (9 fields). The component table has a row starting with `attention`
        # too, but its second field is `prefill`, so the field count separates
        # them -- matching on $1 alone reads the component row and reports a
        # host-side number as if it were the kernel's.
        att=$(awk 'NF==9 && $1=="attention" {print $5}' "$OUT/$tag.log" | tail -1)
        printf '%-16s rc=%-3s tok=%-5s attn_net=%-9s prefill=%s tok/s\n' \
               "$tag" "$rc" "${T:--}" "${att:--}" "${pre:--}"
        printf '%s\t%s\t%s\t%s\t%s\n' "a$a" "$r" "${T:-0}" "${att:-0}" \
               "${pre:-0}" >> "$OUT/points.tsv"
        taskkill //F //IM kraken.exe //T >/dev/null 2>&1
      done
    done
  done
done

echo
python3 - "$OUT/points.tsv" "$LAYERS" "$HEADS" "$QT" "${#ARMS[@]}" <<'PY'
import sys, statistics as st
path, layers, heads, qt, narm = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
pts = {}
for line in open(path):
    arm, rep, tok, att, pre = line.split()
    if float(att) <= 0 or int(tok) <= 0:
        continue
    pts.setdefault((arm, int(tok)), []).append(float(att))

def fit(x1, y1, x2, y2):
    # Two points, two unknowns: ms = a*pairs + b*blocks.
    b = (y2 - y1 * x2[0] / x1[0]) / (x2[1] - x1[1] * x2[0] / x1[0])
    a = (y1 - b * x1[1]) / x1[0]
    return a, b

print(f"{'arm':<6} {'tokens':>7} {'pairs':>12} {'blocks':>9} {'attn ms':>9} {'med of':>7}")
for arm in sorted({k[0] for k in pts}):
    rows = []
    for (aa, tok), v in sorted(pts.items(), key=lambda kv: kv[0][1]):
        if aa != arm:
            continue
        pairs = layers * heads * tok * (tok + 1) / 2
        blocks = layers * heads * ((tok + qt - 1) // qt)
        med = st.median(v)
        rows.append((tok, pairs, blocks, med, len(v)))
        print(f"{arm:<6} {tok:>7} {pairs:>12.3e} {blocks:>9} {med:>9.2f} {len(v):>7}")
    if len(rows) < 3:
        print(f"       {arm}: need three lengths for a fit\n")
        continue
    lo, mid, hi = rows[0], len(rows) // 2, rows[-1]
    mid = rows[mid]
    a, b = fit((lo[1], lo[2]), lo[3], (hi[1], hi[2]), hi[3])
    pred = a * mid[1] + b * mid[2]
    err = 100.0 * (pred - mid[3]) / mid[3] if mid[3] else 0.0
    print(f"       {arm}: {a * 1e6:.3f} ns/pair + {b * 1e3:.3f} us/block"
          f"  (middle {mid[0]} tok: predicted {pred:.2f} vs {mid[3]:.2f} ms,"
          f" {err:+.1f}%)")
    print(f"       {arm}: block term at {hi[0]} tok = {b * hi[2]:.2f} ms of"
          f" {hi[3]:.2f} ms ({100 * b * hi[2] / hi[3]:.1f}%)\n")
PY
echo "wrote $OUT/points.tsv"
