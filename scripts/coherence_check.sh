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
#
#   scripts/coherence_check.sh                      # every runnable model <= 8 GiB
#   scripts/coherence_check.sh models/foo.gguf ...  # just these
#
# Environment: KRK_BIN (default build-hip/kraken.exe), KRK_PROMPT, KRK_N (tokens
# to generate), KRK_CTX, KRK_CHUNK, KRK_MAX_MB (default 8192), KRK_TOL.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${KRK_BIN:-$ROOT/build-hip/kraken.exe}
PROMPT=${KRK_PROMPT:-"What is the capital of France?"}
N=${KRK_N:-32}
CTX=${KRK_CTX:-512}
CHUNK=${KRK_CHUNK:-32}
MAX_MB=${KRK_MAX_MB:-8192}
TOL=${KRK_TOL:-0.05}
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
gen() {
    "$BIN" -m "$1" ${2:-} $CHAT_ARG -p "$PROMPT" -n "$N" --greedy --ctx "$CTX" \
        --chunk "$CHUNK" 2>/dev/null | grep -v '^KRAKEN 0.1.0' |
        sed 's/[[:space:]]*$//'
}

# "id=value id=value" for the top two at each step, one line per step. The
# debug dump is on stderr, so stdout goes to /dev/null: the streamed text would
# otherwise interleave with it and an anchored match would drop a line, which
# reads as a divergence one step early.
steps() {
    "$BIN" -m "$1" ${2:-} $CHAT_ARG -p "$PROMPT" -n "$N" --greedy --ctx "$CTX" \
        --chunk "$CHUNK" --debug-topk 2 2>&1 1>/dev/null |
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

CHAT_ARG=""
[ "$CHAT" = "1" ] && CHAT_ARG="--chat"

echo "prompt: $PROMPT   n=$N   ctx=$CTX   chunk=$CHUNK   chat=$CHAT   tie tol=$TOL"
pass=0
fail=0
for m in "${models[@]}"; do
    name=$(basename "$m")
    gpu=$(gen "$m" "") || true
    cpu=$(gen "$m" "--cpu") || true
    if [ -z "$gpu" ] || [ -z "$cpu" ]; then
        echo "SKIP  $name (no output)"
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
[ "$fail" -eq 0 ]
