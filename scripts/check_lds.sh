#!/bin/sh
# ============================================================================
#  check_lds.sh — the LDS bank-conflict regression guard.
#
#  WHAT IT PROTECTS. A tile kernel whose LDS row stride aliases the banks keeps
#  working and keeps agreeing with its reference; it just runs 2-6x slow. That
#  is how the gfx10 prefill GEMM sat at 2.9 TFLOP/s (fp16) and the int8 tile at
#  6.2 TOP/s without any test noticing. The counters do notice:
#  SQC_LDS_BANK_CONFLICT / SQ_INSTS_LDS was 16.4 and 19.4 there, against 0.0-2.5
#  for the same kernels once their fetch was rotated off the aliased banks.
#
#  So this runs a real workload under rocprofv3 and hands the counters to
#  tools/prof_agg.py --guard, which exits non-zero when any kernel exceeds the
#  conflict budget. Wire it into whatever runs before a release; it needs ROCm
#  and a GPU, so it is not part of the unit-test suite.
#
#    sh ./scripts/check_lds.sh                          # default model, fp16 path
#    KRK_GEMM_INT8=1 sh ./scripts/check_lds.sh          # also the int8 path
#    KRK_GUARD_MODEL=/tmp/q9b.gguf sh ./scripts/check_lds.sh
#    KRK_BIN=build-hip/kraken sh ./scripts/check_lds.sh
#
#  The one optional argument is an existing rocprofv3 counter CSV (or the
#  directory holding one), which is analysed instead of taking a fresh profile
#  — so a saved CI artifact can be re-judged without a GPU. Anything else is
#  rejected rather than ignored.
#
#  Exit codes: 0 inside budget, 1 over budget, 2 could not run (no model, no
#  rocprofv3, no counters, bad usage) — 2 is a failure too, because a guard that
#  cannot measure is not a guard.
# ============================================================================
set -e

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
bin=${KRK_BIN:-"$here/build-hip/kraken"}
model=${KRK_GUARD_MODEL:-$(ls "$here"/models/*.gguf 2>/dev/null | head -1 || true)}
prof=${KRK_PROF:-rocprofv3}
saved=
case $# in
    0) ;;
    1) saved=$1 ;;
    *)
        echo "check_lds: at most one argument (an existing counter CSV), got $#" >&2
        echo "usage: sh scripts/check_lds.sh [counter_collection.csv | dir]" >&2
        exit 2
        ;;
esac
if [ -n "$saved" ]; then
    if [ -d "$saved" ]; then
        saved=$(find "$saved" -name '*_counter_collection.csv' | head -1)
    fi
    if [ -z "$saved" ] || [ ! -f "$saved" ]; then
        echo "check_lds: no counter CSV at '$1'" >&2
        exit 2
    fi
fi
out=$(mktemp -d "${TMPDIR:-/tmp}/krk-ldsguard.XXXXXX")

if [ -n "$saved" ]; then
    echo "check_lds: judging saved counters in $saved"
    if python3 "$here/tools/prof_agg.py" --guard "$saved"; then exit 0; fi
    echo "check_lds: FAILED — see docs/TODO.md P1b for the fetch rotation this" \
         "guard exists to protect" >&2
    exit 1
fi

if [ ! -x "$bin" ]; then
    echo "check_lds: no kraken binary at $bin (set KRK_BIN)" >&2
    exit 2
fi
if [ -z "$model" ] || [ ! -f "$model" ]; then
    echo "check_lds: no model to run; set KRK_GUARD_MODEL or put a .gguf in" \
         "models/ (scripts/fetch_moe_model.sh fetches a small MoE one)" >&2
    exit 2
fi
if ! command -v "$prof" >/dev/null 2>&1; then
    echo "check_lds: $prof not found; this guard needs ROCm (set KRK_PROF)" >&2
    exit 2
fi

echo "check_lds: $bin on $model"
echo "check_lds: KRK_GEMM_INT8=${KRK_GEMM_INT8:-<unset>}"
"$prof" --pmc SQ_INSTS_LDS SQC_LDS_BANK_CONFLICT -f csv -d "$out" -- \
    "$bin" --model "$model" --bench --chunk 64 >"$out/run.log" 2>&1 || {
    echo "check_lds: the workload failed; see $out/run.log" >&2
    exit 2
}
csv=$(find "$out" -name '*_counter_collection.csv' | head -1)
if [ -z "$csv" ]; then
    echo "check_lds: $prof produced no counter CSV; see $out/run.log" >&2
    exit 2
fi

echo "check_lds: counters in $csv"
# The failure branch must print its diagnostic, so the python call is guarded
# rather than left to `set -e` (which would abort before the message ran).
if python3 "$here/tools/prof_agg.py" --guard "$csv"; then
    exit 0
fi
echo "check_lds: FAILED — see docs/TODO.md P1b for the fetch rotation this" \
     "guard exists to protect" >&2
exit 1
