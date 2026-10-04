#!/usr/bin/env bash
# Coherence check: greedy-generate the same prompt on the scalar CPU reference
# and on the device, then compare. Inference output is text a human reads, so
# the bar is readable words, not "close enough logits": a backend that prints
# numbers, never stops, or answers with something unrelated is failing.
#
# Verdicts:
#   PASS          identical text
#   PASS (tie)    text differs, but the first differing step was a coin flip at
#                 activation precision: both backends rank the same two tokens
#                 within `tol` of each other, so either choice is rounding
#   FAIL (drift)  the first differing step is a real gap
#   FAIL (text)   the generated text is not text (>25% digits, or no letters)
#   FAIL (empty)  a backend produced no text at all, or died, or did not finish
#                 inside KRK_TIMEOUT. An empty run is a failure, not a skip: the
#                 acceptance criterion for this script is "a human can read the
#                 answer", and no answer is not an answer.
#
# Exit status is non-zero when anything failed *or* when a model the caller
# named was not tested because it is over the cap — an untested model must not
# read as a passing one. KRK_ALLOW_BIG=1 runs those anyway; see the warning
# below for why the cap exists.
#
#   scripts/coherence_check.sh                      # every runnable model <= 8 GiB
#   scripts/coherence_check.sh models/foo.gguf ...  # just these
#
# Environment: KRK_BIN (default build-hip/kraken.exe), KRK_PROMPT, KRK_N (tokens
# to generate), KRK_CTX, KRK_CHUNK, KRK_MAX_MB (default 8192), KRK_TOL,
# KRK_TIMEOUT (seconds per engine run, default 120), KRK_ALLOW_BIG (0/1).
#
# The cap is not a preference. The CPU side of this check is the *scalar*
# reference, which dequantizes in place: on the 19 GiB laguna-xs2 it needs
# about 7 s per token, so the default 32-token run pins every core for ~4
# minutes and, alongside any other work on the machine, that reads as a hung
# system. Raise KRK_MAX_MB or set KRK_ALLOW_BIG=1 deliberately, never by
# accident, and keep KRK_TIMEOUT bounded when you do.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${KRK_BIN:-$ROOT/build-hip/kraken.exe}
PROMPT=${KRK_PROMPT:-"What is the capital of France?"}
N=${KRK_N:-32}
CTX=${KRK_CTX:-512}
CHUNK=${KRK_CHUNK:-32}
MAX_MB=${KRK_MAX_MB:-8192}
TOL=${KRK_TOL:-0.05}
TIMEOUT=${KRK_TIMEOUT:-120}
ALLOW_BIG=${KRK_ALLOW_BIG:-0}
# Instruct models fed a raw sentence do the one thing their training never
# asked of them: they continue the sentence, so the "answer" is the question
# again (Qwen3.5-0.8B loops "The capital of France is the capital of France"
# until the token budget runs out). That is the model behaving as trained, not
# a backend defect — both backends produce it identically — and it is easy to
# mistake for one. The gate therefore drives the models the way they were
# trained, through the ChatML turn markers, and KRK_CHAT=0 turns that off for
# base models.
CHAT=${KRK_CHAT:-1}

[ -x "$BIN" ] || { echo "no $BIN — build kraken first"; exit 2; }

# Generated text only: the banner and the [info ]/[warn ] chatter are not part
# of the answer. stdout carries exactly the continuation.
# Runs the engine once and echoes the generated text. Returns the engine's own
# status (124 when `timeout` killed it), which the caller checks: a pass that
# never finished has not passed, and stdout is captured through a file so the
# status survives the pipeline that strips the banner.
gen() {
    local f rc
    f=$(mktemp) || return 1
    timeout "$TIMEOUT" "$BIN" -m "$1" ${2:-} $CHAT_ARG -p "$PROMPT" \
        -n "$N" --greedy --ctx "$CTX" --chunk "$CHUNK" >"$f" 2>/dev/null
    rc=$?
    grep -v '^KRAKEN 0.1.0' "$f" | sed 's/[[:space:]]*$//'
    rm -f "$f"
    return $rc
}

# "id=value id=value" for the top two at each step, one line per step. The
# debug dump is on stderr, so stdout goes to /dev/null: the streamed text would
# otherwise interleave with it and an anchored match would drop a line, which
# reads as a divergence one step early.
steps() {
    timeout "$TIMEOUT" "$BIN" -m "$1" ${2:-} $CHAT_ARG -p "$PROMPT" \
        -n "$N" --greedy --ctx "$CTX" --chunk "$CHUNK" --debug-topk 2 \
        2>&1 1>/dev/null |
        grep '^step ' |
        sed -e 's/^step [0-9]* pos [0-9]* nan=[0-9]* inf=[0-9]* top2: //' \
            -e 's/([^)]*)//g'
}

# Why the two step streams first disagree: a tie, or a real gap.
first_diff() {
    # Tab is the field separator: a step line has spaces of its own.
    paste <(steps "$1" "") <(steps "$1" "--cpu") | awk -F '\t' -v tol="$TOL" '
        function parse(s, ids, vals,   n, i, kv) {
            n = split(s, kv, " ")
            for (i = 1; i <= n; i++) { split(kv[i], p, "="); ids[i] = p[1]; vals[i] = p[2] }
            return n
        }
        {
            parse($1, li, lv)
            parse($2, ri, rv)
            if (li[1] == ri[1]) next
            tie = (li[1] == ri[2] && ri[1] == li[2] &&
                   lv[1] - lv[2] < tol && rv[1] - rv[2] < tol)
            res = sprintf("%s step %d (%s vs %s)", tie ? "tie" : "drift",
                          NR - 1, li[1], ri[1])
            mismatch = 1
            exit
        }
        END { print mismatch ? res : "same picks" }'
}

text_ok() {
    awk '{ t += length($0); d += gsub(/[0-9]/, ""); a += gsub(/[A-Za-z]/, "") }
         END { exit !(t > 0 && a > 0 && d * 4 < t) }' <<<"$1"
}

models=("$@")
if [ ${#models[@]} -eq 0 ]; then
    while IFS='|' read -r f0 f1 status size path rest; do
        [ "$(echo "$status" | tr -d ' ')" = "runnable" ] || continue
        mb=$(echo "$size" | tr -cd '0-9')
        [ -n "$mb" ] && [ "$mb" -le "$MAX_MB" ] || continue
        p=$(echo "$path" | tr -d ' `')
        if [ -f "$ROOT/$p" ]; then models+=("$p"); elif [ -f "$p" ]; then models+=("$p"); fi
    done < <(grep '| runnable |' "$ROOT/docs/model-inventory.md")
fi

# A model named on the command line gets the same cap as the scan: an explicit
# argument is not a reason to start a multi-minute all-core scalar run by
# accident. Skipped models are reported and counted, and the script exits
# non-zero while any remain, because "not tested" is not "passes".
excluded=0
if [ ${#models[@]} -gt 0 ] && [ "$ALLOW_BIG" != "1" ]; then
    kept=()
    for m in "${models[@]}"; do
        bytes=$(stat -c%s "$m" 2>/dev/null || echo 0)
        mb=$(( (bytes + 1048575) / 1048576 ))
        if [ "$mb" -gt "$MAX_MB" ]; then
            echo "SKIP  $(basename "$m") (${mb} MiB > KRK_MAX_MB=${MAX_MB}): not tested"
            echo "      set KRK_ALLOW_BIG=1 to run it (~$((mb / 3))s per CPU token)"
            excluded=$((excluded + 1))
        else
            kept+=("$m")
        fi
    done
    models=(${kept[@]+"${kept[@]}"})
fi

CHAT_ARG=""
[ "$CHAT" = "1" ] && CHAT_ARG="--chat"

echo "prompt: $PROMPT   n=$N   ctx=$CTX   chunk=$CHUNK   chat=$CHAT   tie tol=$TOL"
echo "per-run timeout ${TIMEOUT}s"
pass=0
fail=0
for m in "${models[@]}"; do
    name=$(basename "$m")
    gpu=$(gen "$m" ""); grc=$?
    cpu=$(gen "$m" "--cpu"); crc=$?
    if [ "$grc" -ne 0 ] || [ "$crc" -ne 0 ]; then
        why="device rc=$grc, cpu rc=$crc"
        [ "$grc" -eq 124 ] || [ "$crc" -eq 124 ] &&
            why="did not finish in ${TIMEOUT}s (device rc=$grc, cpu rc=$crc)"
        fail=$((fail + 1))
        printf '%-22s %-42s %s\n' "FAIL" "$name" "$why"
        continue
    fi
    if [ -z "$gpu" ] || [ -z "$cpu" ]; then
        which="$([ -z "$gpu" ] && echo device || echo cpu)"
        [ -z "$gpu" ] && [ -z "$cpu" ] && which="both"
        fail=$((fail + 1))
        printf '%-22s %-42s %s\n' "FAIL" "$name" "$which produced no text"
        continue
    fi
    if [ "$gpu" == "$cpu" ]; then
        verdict="PASS"
    else
        why=$(first_diff "$m")
        case "$why" in
            tie*) verdict="PASS ($why)" ;;
            *)     verdict="FAIL ($why)" ;;
        esac
    fi
    if ! text_ok "$gpu"; then verdict="FAIL (text: $(echo "$gpu" | tr '\n' ' ' | cut -c1-70))"; fi
    case "$verdict" in
        PASS*) pass=$((pass + 1)) ;;
        *)     fail=$((fail + 1)) ;;
    esac
    printf '%-22s %-42s %s\n' "$verdict" "$name" \
        "$(echo "$gpu" | tr '\n' ' ' | cut -c1-88)"
done
echo "$pass coherent, $fail not"
if [ "$excluded" -gt 0 ]; then
    echo "$excluded not tested (over KRK_MAX_MB=$MAX_MB): the run does not pass"
fi
[ "$fail" -eq 0 ] && [ "$excluded" -eq 0 ]
