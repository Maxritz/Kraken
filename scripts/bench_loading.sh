#!/usr/bin/env bash
# Loading-mode benchmark: memory-resident vs auto vs re-read-from-disk weights.
#
# Every arm is the same binary, the same --bench prompt, the same ctx and chunk;
# the only difference is where the dense weights live between steps:
#
#   resident  --weight-vram-mb -1       upload once, keep it (the old path)
#   auto      (no flag)                 keep what fits, stream the tail, promote
#                                       a streamed layer resident on first use
#   disk      --weight-vram-mb N with   a fixed N MiB prefix stays resident and
#             KRK_WEIGHT_PROMOTE=0      every other layer is re-read from the
#                                       file before each use, every step
#
# The disk arm needs KRK_WEIGHT_PROMOTE=0 to be a disk arm at all: a "streamed"
# layer is promoted into a permanent allocation the first time it runs, so a
# model that fits is fully resident within the first token without it.
#
# Arms run interleaved and order-balanced (reversed on even reps) because page
# cache and driver state warm between them; a sequential probe reports the
# ordering as the arm. The summary is the warmest rep's numbers, and every
# recorded run is printed so a cold one is visible rather than averaged away.
#
# Usage:
#   sh scripts/bench_loading.sh [MODEL.gguf ...]
#
# Env:
#   KRK_BIN         kraken binary          (default build-hip/kraken.exe)
#   KRK_BENCH_REPS  reps per arm           (default 2)
#   KRK_BENCH_N     generated tokens       (default 32; --bench's prompt is fixed)
#   KRK_BENCH_CTX   context                (default 512)
#   KRK_BENCH_DISK_MB  disk-arm resident MiB (default 2048)
#   KRK_BENCH_OUT   log/tsv directory      (default /tmp)
#   KRK_BENCH_TIMEOUT  per-run seconds     (default 420)
#
# Model arguments may carry a per-model token count as PATH:n, e.g.
#   sh scripts/bench_loading.sh models/Qwen3-8B-Q4_K_M.gguf:64
#
# Numbers and the caveats behind them: docs/BENCH-LOADING.md.

set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 1
K=${KRK_BIN:-$ROOT/build-hip/kraken.exe}
REPS=${KRK_BENCH_REPS:-2}
N=${KRK_BENCH_N:-32}
CTX=${KRK_BENCH_CTX:-512}
DISK_MB=${KRK_BENCH_DISK_MB:-2048}
OUT=${KRK_BENCH_OUT:-/tmp}
TMO=${KRK_BENCH_TIMEOUT:-420}

if [ ! -x "$K" ]; then
    echo "bench_loading: no kraken binary at $K (build it, or set KRK_BIN)" >&2
    exit 2
fi
mkdir -p "$OUT" || exit 2

tsv="$OUT/bench_loading.tsv"
printf 'model\tarm\trep\trc\tsecs\tload_ms\tprefill_tok\tprefill_tps\tdecode_tps\tresidency\n' > "$tsv"

# One arm of one model. Everything the report needs is read back out of the
# run's own log, so a field that silently stops being printed reads as 0 here
# instead of as a plausible stale number.
run_arm() { # tag path arm n rep
    local tag=$1 path=$2 arm=$3 n=$4 rep=$5
    local envs="" args=""
    case "$arm" in
        resident) args="--weight-vram-mb -1" ;;
        auto)     args="" ;;
        disk)     args="--weight-vram-mb $DISK_MB"
                  envs="KRK_WEIGHT_PROMOTE=0" ;;
        *)        echo "bench_loading: unknown arm '$arm'" >&2; return 2 ;;
    esac

    local slug log rc t0 load ptok ptps dtps res
    slug=$(printf '%s' "$tag" | tr -c 'a-zA-Z0-9\n' '_')
    log="$OUT/bl_${slug}_${arm}_${rep}.log"
    t0=$(date +%s)
    # shellcheck disable=SC2086 # envs/args are deliberately word-split here
    env $envs timeout "$TMO" "$K" -m "$path" --bench \
        --ctx "$CTX" --chunk 256 -n "$n" --greedy $args > "$log" 2>&1
    rc=$?
    local secs=$(( $(date +%s) - t0 ))

    load=$(sed -n 's/^load *\([0-9][0-9.]*\) *ms.*/\1/p' "$log" | head -1)
    ptok=$(sed -n 's/^prefill *\([0-9][0-9]*\) tokens.*/\1/p' "$log" | head -1)
    ptps=$(sed -n 's/^prefill .*(\([0-9.]*\) tok\/s).*/\1/p' "$log" | head -1)
    dtps=$(sed -n 's/^decode .*(\([0-9.]*\) tok\/s).*/\1/p' "$log" | head -1)
    res=$(sed -n 's/^\[info \] weight residency: //p' "$log" | head -1 | cut -d';' -f1)
    : "${load:=0}" "${ptok:=0}" "${ptps:=0}" "${dtps:=0}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$tag" "$arm" "$rep" "$rc" "$secs" "$load" "$ptok" "$ptps" "$dtps" \
        "${res:-<none>}" >> "$tsv"
    printf '%-22s %-8s rep%s rc=%s %4ss  load %6s ms  prefill %7s tok (%7s tok/s)  decode %6s tok/s\n' \
        "$tag" "$arm" "$rep" "$rc" "$secs" "$load" "$ptok" "$ptps" "$dtps"
    [ -n "$res" ] && printf '  %s\n' "$res"
    return 0
}

bench_model() { # tag path n
    local tag=$1 path=$2 n=$3 r a i
    local arms=(resident auto disk)
    for ((r = 1; r <= REPS; r++)); do
        if ((r % 2 == 1)); then
            for a in "${arms[@]}"; do run_arm "$tag" "$path" "$a" "$n" "$r" || return 2; done
        else
            for ((i = ${#arms[@]} - 1; i >= 0; i--)); do
                run_arm "$tag" "$path" "${arms[$i]}" "$n" "$r" || return 2
            done
        fi
    done
}

echo "loading-mode benchmark, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "binary $K  ctx $CTX  chunk 256  n $N  reps $REPS  disk ${DISK_MB} MiB"

# No arguments: the four models docs/BENCH-LOADING.md quotes, out of $KRK_MODELS
# (default /g/More-models). A missing one is skipped rather than fetched.
if [ $# -eq 0 ]; then
    MODELS=${KRK_MODELS:-/g/More-models}
    for spec in \
        "$MODELS/Phi-3.5-mini-instruct-python_coding_assistanQ4_K_M.gguf:32" \
        "$MODELS/Qwen3-8B-Q4_K_M.gguf:32" \
        "$MODELS/Qwen3-30B-A3B-abliterated-erotic.i1-Q2_K.gguf:16" \
        "$MODELS/Qwen3.8-27B-WebGGUF-Q4_0.gguf:8"; do
        [ -f "${spec%:*}" ] && set -- "$@" "$spec"
    done
    if [ $# -eq 0 ]; then
        echo "bench_loading: no model in $MODELS; pass MODEL.gguf[:tokens]" >&2
        exit 2
    fi
fi

fail=0
for spec in "$@"; do
    path=${spec%:*}
    n=$N
    case "$spec" in *:*) n=${spec##*:} ;; esac
    if [ ! -f "$path" ]; then
        echo "bench_loading: no such model: $path" >&2
        fail=2
        continue
    fi
    bench_model "$(basename "$path" .gguf)" "$path" "$n" || fail=2
done

# One row per (model, arm): the warmest rep (the one with the smallest load,
# which is the page-cache-warm read) rather than a mean over a cold first rep.
echo
echo "model                  arm       load ms  prefill tok/s  decode tok/s  streamed"
awk -F'\t' 'NR > 1 && $4 == 0 {
        k = $1 SUBSEP $2
        if (!(k in best) || $6 + 0 < best[k] + 0) {
            best[k] = $6; row[k] = $1 "\t" $2 "\t" $6 "\t" $8 "\t" $9 "\t" $10
        }
    }
    END {
        for (k in row) print row[k]
    }' "$tsv" | sort |
    awk -F'\t' '{ printf "%-22s %-8s %7s  %13s  %12s  %s\n", $1, $2, $3, $4, $5, $6 }'

# An arm that did not print its residency line is not evidence about loading,
# and a non-zero rc is a failed run, not a slow one.
if awk -F'\t' 'NR > 1 && ($4 != 0 || $10 == "<none>") { bad = 1 } END { exit bad ? 0 : 1 }' "$tsv"; then
    echo "bench_loading: some runs failed or printed no residency line -- see $tsv" >&2
    fail=1
fi
echo "runs and logs: $tsv, $OUT/bl_*.log"
exit $fail
