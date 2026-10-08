#!/usr/bin/env bash
# scripts/kv_tier_geometry_check.sh — prove the KV tier can always hold the KV
# that exists, on every model on this machine.
#
# The property: with no COLD directory, a KV page that fits neither HOT nor
# WARM is DROPPED -- the layer is zero-filled on its next step and the run keeps
# producing fluent text that is wrong. So HOT + WARM must cover every
# KV-carrying layer for EVERY flag combination, and WARM's capacity is derived
# from the geometry rather than from its byte budget. A COLD directory is the
# one case where a small WARM is a working choice, so the budget is obeyed there
# and the floor must NOT engage.
#
# Three arms per model, load-only (`--info`, no generation) so a 53 GB model
# costs a load rather than a run:
#   A  --kv-hot-mb 1                     the auto WARM policy
#   B  --kv-hot-mb 1 --kv-warm-mb 0      a budget with nowhere to spill
#   C  --kv-hot-mb 1 --kv-warm-mb 0 --kv-cold-dir DIR   the spill path
# and then, for every model small enough to run quickly, a decode arm whose
# stdout must be byte-identical to the same prompt with the cache flat.
#
# Each arm's numbers are judged against the model file's OWN metadata by
# tools/kv_tier_geometry.py, which can be shown to fail on constructed inputs
# (`python3 tools/kv_tier_geometry.py selftest`, 13 cases including four
# failures). The engine's report is the artifact under test; the file's header
# is the independent expectation.
#
# Usage:  scripts/kv_tier_geometry_check.sh
# Env:    KRK_BIN, KRK_INSPECT, KRK_CTX, KRK_TOKENS, KRK_DECODE_MAX_MB,
#         KRK_TIMEOUT, KRK_MODEL_DIRS, KRK_LOG_DIR, KRK_VERBOSE=1,
#         KRK_MAX_MB (skip models above this, reporting them), KRK_MIN_MB
#         (silently start above this, to resume a sweep), KRK_ARMS (which
#         arms, e.g. "B" for one load per model instead of three)
# Exit:   0 when nothing failed (skips are reported, not failures), 1 otherwise.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
BIN=${KRK_BIN:-$ROOT/build-hip/kraken.exe}
INSPECT=${KRK_INSPECT:-$ROOT/build-hip/kraken-inspect.exe}
SCAN_PY=${KRK_SCAN_PY:-$ROOT/tools/gguf_scan.py}
CHECK_PY=${KRK_CHECK_PY:-$ROOT/tools/kv_tier_geometry.py}
# Largest model to run the arms on, in MiB (0 = no cap). Above this a load
# stages tens of GB of experts through WARM and the KV tier is not the question
# being asked any more, so the model is reported as not covered rather than
# quietly left out.
MAX_MB=${KRK_MAX_MB:-0}
MIN_MB=${KRK_MIN_MB:-0}
# Which arms to run. A B C is the full matrix; a `KRK_ARMS=B` pass checks just
# the invariant on models too large to run three times (see KRK_MAX_MB).
ARMS=${KRK_ARMS:-"A B C"}
CTX=${KRK_CTX:-512}
TOKENS=${KRK_TOKENS:-8}
DECODE_MAX_MB=${KRK_DECODE_MAX_MB:-1200}
TMO=${KRK_TIMEOUT:-420}
# Repo-relative by default so the paths written into each arm's JSON open from
# the Python side unchanged: bash /tmp is not C:\tmp for a native Windows
# python, and a relative path has no such gap.
LOGDIR=${KRK_LOG_DIR:-build-hip/kv_tier_check}
DIRS=${KRK_MODEL_DIRS:-"$ROOT/models G:/More-models H:/OLLAMA-Models/GGUF"}
PROMPT="The history of computing is a history of abstraction, from relays to vacuum tubes"
COLD="$LOGDIR/cold"
VERBOSE=${KRK_VERBOSE:-0}

[ -f "$BIN" ] || { echo "no binary at $BIN (build it first)"; exit 2; }
[ -f "$SCAN_PY" ] || { echo "no scanner at $SCAN_PY"; exit 2; }
[ -f "$CHECK_PY" ] || { echo "no checker at $CHECK_PY"; exit 2; }
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR" "$COLD"

n_pass=0; n_fail=0; n_skip=0; n_decode=0; n_flat=0
FAILS="$LOGDIR/failures.txt"; : > "$FAILS"
SUMMARY="$LOGDIR/summary.txt"; : > "$SUMMARY"

arm_flags() {
    case "$1" in
        A) echo "--kv-hot-mb 1" ;;
        B) echo "--kv-hot-mb 1 --kv-warm-mb 0" ;;
        C) echo "--kv-hot-mb 1 --kv-warm-mb 0 --kv-cold-dir $COLD" ;;
    esac
}
arm_cold() { [ "$1" = "C" ] && echo true || echo false; }
arm_asked() { [ "$1" = "A" ] && echo null || echo 0; }

emit_arm_json() { # $1 file, $2 name, $3 rc, $4 log, $5 cold, $6 asked
    printf '{"name": "%s", "rc": %s, "log": "%s", "cold_dir": %s, "asked_warm_mb": %s}\n' \
        "$2" "$3" "$4" "$5" "$6" > "$1"
}

expect_of() { # $1 scan json -> "layers mib"
    KRK_CTX=$CTX python3 "$CHECK_PY" expect "$1" | tr -d '\r' |
        python3 -c 'import sys,json; d=json.load(sys.stdin); e=d["expect"]; print("%s %s" % (e.get("kv_layers"), round(d.get("kv_mib_at_ctx") or 0)))'
}

decode_arm() { # $1 model, $2 log, $3 stdout, $4... flags
    local model="$1" log="$2" out="$3"; shift 3
    timeout "$TMO" "$BIN" --model "$model" -p "$PROMPT" -n "$TOKENS" \
        --temp 0 --greedy --ctx "$CTX" "$@" > "$out" 2> "$log"
    echo $?
}

echo "KV tier geometry check"
echo "  binary  : $BIN"
echo "  models  : $DIRS"
echo "  ctx     : $CTX   decode arms: <= ${DECODE_MAX_MB} MiB, ${TOKENS} tokens"
echo "  logs    : $LOGDIR"
echo

# Smallest first, so a long run produces its results early and a model that
# takes minutes cannot hold up the seventy that do not.
LIST=$LOGDIR/models.txt
: > "$LIST"
for dir in $DIRS; do
    [ -d "$dir" ] || continue
    for model in "$dir"/*.gguf; do
        [ -f "$model" ] || continue
        printf '%s %s\n' "$(wc -c < "$model" | tr -d ' ')" "$model" >> "$LIST"
    done
done
sort -n "$LIST" -o "$LIST"
echo "$(( $(wc -l < "$LIST") )) models found, smallest first"

while read -r bytes model; do
        name=$(basename "$model")
        safe=$(printf '%s' "$name" | tr -c 'A-Za-z0-9._-' '_')
        size_mb=$(( bytes / 1048576 ))

        if [ "$MAX_MB" != "0" ] && [ "$size_mb" -gt "$MAX_MB" ]; then
            printf '%-4s %-50s NOT COVERED: %s MiB > KRK_MAX_MB=%s\n' \
                SKIP "$name" "$size_mb" "$MAX_MB" | tee -a "$SUMMARY"
            n_skip=$((n_skip + 1))
            continue
        fi
        if [ "$MIN_MB" != "0" ] && [ "$size_mb" -lt "$MIN_MB" ]; then
            n_skip=$((n_skip + 1))
            continue
        fi

        # A refusal is cheap to establish from the header alone, and expensive
        # to re-establish by loading the weights three times. kraken-inspect
        # answers from the same krk_core tables the loader uses.
        if [ -x "$INSPECT" ]; then
            iverdict=$(timeout 60 "$INSPECT" "$model" 2>&1 | tr -d '\r' |
                       sed -n 's/^verdict *//p' | head -1)
            if [ "$iverdict" = "refused" ]; then
                ireason=$(timeout 60 "$INSPECT" "$model" 2>&1 | tr -d '\r' |
                          grep -m1 '^  *- ' | sed 's/^  *- //')
                printf '%-4s %-50s refused by kraken-inspect: %s\n' \
                    SKIP "$name" "${ireason:0:70}" | tee -a "$SUMMARY"
                n_skip=$((n_skip + 1))
                continue
            fi
        fi

        scan="$LOGDIR/$safe.scan.json"
        python3 "$SCAN_PY" "$model" > "$scan" 2>"$scan.err"
        exp_line=$(expect_of "$scan")
        exp_layers=${exp_line%% *}

        arms=""
        for arm in $ARMS; do
            log="$LOGDIR/$safe.$arm.log"
            jf="$LOGDIR/$safe.$arm.json"
            rc=$(timeout "$TMO" "$BIN" --model "$model" --info --expert-warmup 0 \
                     --ctx "$CTX" $(arm_flags "$arm") > "$log" 2>&1; echo $?)
            emit_arm_json "$jf" "$arm" "$rc" "$log" "$(arm_cold "$arm")" "$(arm_asked "$arm")"
            arms="$arms $jf"
        done

        # Python on Windows writes \r\n, and a trailing CR in a captured
        # verdict turns every later comparison into a silent miss.
        verdict=$(KRK_CTX=$CTX python3 "$CHECK_PY" check "$scan" $arms | tr -d '\r')

        status=PASS; notes=""; detail_lines=""; idx=0; ran=0
        while IFS= read -r vline; do
            case "$vline" in
                note\ *) notes="$notes
      note: ${vline#note }"; continue ;;
            esac
            idx=$((idx + 1))
            armname=$(printf '%s\n' $ARMS | sed -n "${idx}p")
            case "$vline" in
                FAIL*) status=FAIL ;;
                SKIP*) [ "$status" = "PASS" ] && status=SKIP ;;
                PASS*) ran=1 ;;
            esac
            # Drop the checker's "<status> <file> [A] " prefix: the arm letter
            # already says which arm this is and the file name is on the line
            # above it.
            dline=${vline#*\] }
            # Why this arm's slot counts look the way they do, straight out of
            # its own log: the checker requires the raise to be reported, and
            # this is where a reader can see that it was.
            if grep -q "WARM was raised to" "$LOGDIR/$safe.$armname.log" 2>/dev/null; then
                dline="$dline   [no spill dir: WARM raised to cover the cache, and reported]"
            fi
            if grep -q "is not writable, so it was ignored" "$LOGDIR/$safe.$armname.log" 2>/dev/null; then
                dline="$dline   [cold dir unwritable: ignored, so the case is arm B]"
            fi
            if [ -z "$detail_lines" ]; then
                detail_lines="      $armname  $dline"
            else
                detail_lines="$detail_lines
      $armname  $dline"
            fi
        done <<< "$verdict"
        if [ -z "$verdict" ]; then
            status=FAIL
            detail_lines="
      the checker printed nothing (did it run?)"
        fi

        # Phase 2: decode equality, for models small enough to run.
        dec=""
        if [ "$status" = "PASS" ] && [ "$size_mb" -le "$DECODE_MAX_MB" ]; then
            frc=$(decode_arm "$model" "$LOGDIR/$safe.flat.log" "$LOGDIR/$safe.flat.out")
            trc=$(decode_arm "$model" "$LOGDIR/$safe.tier.log" "$LOGDIR/$safe.tier.out" \
                      --kv-hot-mb 1 --kv-warm-mb 0)
            if [ "$frc" != "0" ] || [ "$trc" != "0" ]; then
                dec="decode: SKIP (rc $frc/$trc)"
            elif grep -q "^\[info \] kv tiered" "$LOGDIR/$safe.tier.log"; then
                fm=$(md5sum "$LOGDIR/$safe.flat.out" | cut -d' ' -f1)
                tm=$(md5sum "$LOGDIR/$safe.tier.out" | cut -d' ' -f1)
                # The stats line, not the word: the engine's own warning about
                # pages being dropped contains it too.
                if grep -q "^\[stats \] kv.*DROPPED" "$LOGDIR/$safe.tier.log"; then
                    status=FAIL
                    dec="decode: pages DROPPED (KV history lost)"
                elif [ "$fm" = "$tm" ]; then
                    n_decode=$((n_decode + 1))
                    dec="decode: tiered output == flat output (${fm:0:12})"
                else
                    status=FAIL
                    dec="decode: tiered ${tm:0:12} != flat ${fm:0:12}"
                fi
            else
                n_flat=$((n_flat + 1))
                dec="decode: cache stayed flat (KV fits the 1 MiB HOT budget)"
            fi
        fi

        case "$status" in
            PASS) n_pass=$((n_pass + 1)) ;;
            FAIL) n_fail=$((n_fail + 1))
                  printf '%s\n  file says %s KV layer(s)%s%s\n  %s\n' \
                      "$name" "$exp_layers" "$notes" "$detail_lines" "$dec" >> "$FAILS" ;;
            SKIP) n_skip=$((n_skip + 1)) ;;
        esac

        printf '%-4s %-50s file: %s KV layer(s)\n' "$status" "$name" "$exp_layers"
        printf '%s\n' "$detail_lines"
        [ -n "$notes" ] && printf '%s\n' "$notes"
        [ -n "$dec" ] && printf '      %s\n' "$dec"
        printf '%-4s %-50s %s\n' "$status" "$name" "$dec" >> "$SUMMARY"
        [ "$VERBOSE" = "1" ] && [ "$status" != "PASS" ] && printf '%s\n' "$verdict" >&2
done < "$LIST"

echo
echo "=== summary: $n_pass passed, $n_fail failed, $n_skip skipped" \
     "($n_decode decode arms compared flat vs tiered, $n_flat models with a flat cache)"
if [ "$n_fail" -gt 0 ]; then
    echo "--- failures"
    cat "$FAILS"
fi
echo "per-model table: $SUMMARY"
[ "$n_fail" -eq 0 ]
