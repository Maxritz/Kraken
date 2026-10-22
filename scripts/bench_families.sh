#!/bin/bash
# Per-model benchmark + coherence sweep, one line per model, markdown table.
#
# For every model it reports what the engine actually did, not what the file
# claims: the arch the loader chose, the prefill and decode rates from the
# engine's own `--bench`, whether the device arm's text is identical to the
# `--cpu` arm's (that is what "coherent" means here -- one engine, two
# backends), and the expert tier's behaviour (hot-hit share, bytes read per
# token, promotions) plus whether <model>.krakenexperts.json was found.
#
# Usage:
#   bash scripts/bench_families.sh [-o out.md] MODEL [MODEL ...]
#   KRK_MODELS="a.gguf:b.gguf" bash scripts/bench_families.sh
#
# Environment: KRK_BIN (default build-hip/kraken.exe), KRK_CTX (512),
# KRK_N (decode tokens for the bench, default the engine's own), KRK_TOKENS
# (16, the coherence continuation), KRK_TIMEOUT (600 s per run),
# KRK_SCAN=1 (run --expert-scan first when a MoE has no index yet),
# KRK_SKIP_BENCH=1 (coherence only), KRK_ALLOW_BIG=1 is passed through to
# nothing here: this script benches whatever it is given.
#
# Exit status: 0 when every model loaded and its two arms agreed; 1 otherwise.
# A model that fails to load is a row with `refused`, not a crash.

set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${KRK_BIN:-$ROOT/build-hip/kraken.exe}
CTX=${KRK_CTX:-512}
TOKENS=${KRK_TOKENS:-16}
TIMEOUT=${KRK_TIMEOUT:-600}
OUT=""
MODELS=()

while [ $# -gt 0 ]; do
    case "$1" in
        -o) OUT=$2; shift 2 ;;
        -o*) OUT=${1#-o}; shift ;;
        *) MODELS+=("$1"); shift ;;
    esac
done
if [ ${#MODELS[@]} -eq 0 ] && [ -n "${KRK_MODELS:-}" ]; then
    IFS=':' read -r -a MODELS <<< "$KRK_MODELS"
fi
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "usage: bash scripts/bench_families.sh [-o out.md] MODEL [MODEL ...]" >&2
    exit 2
fi
if [ ! -x "$BIN" ]; then
    echo "FAIL: no kraken binary at $BIN (build it first)" >&2
    exit 2
fi

# The coherence prompt is the one scripts/coherence_check.sh uses, so a row here
# and a PASS there mean the same thing.
PROMPT=${KRK_PROMPT:-"What is the capital of France?"}

emit() { # emit <line>  -- stdout always, the file too when -o was given
    printf '%s\n' "$1"
    if [ -n "$OUT" ]; then printf '%s\n' "$1" >> "$OUT"; fi
}

say() { printf '%s\n' "$1" >&2; }

# Run one arm and echo its stdout (banner stripped). stderr lands in $ERRF so a
# refusal can be reported as the reason instead of "produced nothing".
ERRF=$(mktemp 2>/dev/null || echo /tmp/krk-arm-err.$$)
arm() {
    local model=$1; shift
    timeout "$TIMEOUT" "$BIN" -m "$model" "$@" 2>"$ERRF" | tail -n +2
    return ${PIPESTATUS[0]}
}

strip() { tr -d '\r' < /dev/stdin | sed -e 's/[[:space:]]*$//' -e '/^$/d'; }

if [ -n "$OUT" ]; then : > "$OUT"; fi
emit "| model | family | prefill tok/s | decode tok/s | device==cpu | hot | read MiB/tok | index | note |"
emit "|---|---|---|---|---|---|---|---|---|"

bad=0
for model in "${MODELS[@]}"; do
    name=$(basename "$model")
    if [ ! -f "$model" ]; then
        emit "| $name | - | - | - | - | - | - | - | missing |"
        bad=1
        continue
    fi
    say "== $name"

    # One load names the arch, the shape and the layers.
    hdr=$(timeout "$TIMEOUT" "$BIN" -m "$model" -p "hi" -n 1 --ctx 64 --greedy 2>&1 >/dev/null |
          grep -E "loaded |architecture .* not in|cannot run" | head -1)
    family=$(printf '%s' "$hdr" | sed -n 's/.*: \([0-9]*\) layers.*/\1 layers/p')
    arch=$(printf '%s' "$hdr" | sed -n 's/.*loaded .*(\([a-z0-9_.-]*\)):.*/\1/p')
    if [ -z "$arch" ]; then
        arch=$(printf '%s' "$hdr" | sed -n "s/.*architecture '\([a-z0-9_.-]*\)'.*/\1/p")
    fi
    [ -z "$arch" ] && arch="-"
    if printf '%s' "$hdr" | grep -q "cannot run"; then
        note=$(printf '%s' "$hdr" | sed 's/.*cannot run this file[^:]*: *//' | cut -c1-60)
        emit "| $name | $arch | - | - | - | - | - | - | refused: $note |"
        bad=1
        continue
    fi

    # An expert index is path-derived: <model>.krakenexperts.json.
    idx="no"
    if [ -f "$model.krakenexperts.json" ]; then idx="yes"; fi

    pre=""; dec=""; hot=""; rd=""; note=""
    warm=()
    # KRK_WARMUP=1 ranks the experts by the model's own index before the run
    # (--expert-warmup 1). On a MoE that fits its budget this is the difference
    # between reading the routed bank inside the prefill and reading it once at
    # load; KRK_SCAN=1 writes the index first when the model has none.
    [ "${KRK_WARMUP:-0}" = "1" ] && warm=(--expert-warmup 1)
    if [ "${KRK_SCAN:-0}" = "1" ] && [ "$idx" = "no" ] && [ ${#warm[@]} -gt 0 ]; then
        say "   expert-scan"
        timeout "$TIMEOUT" "$BIN" -m "$model" -p "$PROMPT" -n 24 --ctx "$CTX"             --greedy --expert-scan >/dev/null 2>&1
        [ -f "$model.krakenexperts.json" ] && idx="yes(scanned)"
    fi
    if [ "${KRK_SKIP_BENCH:-0}" != "1" ]; then
        say "   bench${warm:+ (expert-warmup 1)}"
        out=$(timeout "$TIMEOUT" "$BIN" -m "$model" --bench --ctx "$CTX" "${warm[@]}" 2>&1)
        pre=$(printf '%s\n' "$out" | sed -n 's/^prefill .*(\([0-9.]*\) tok\/s).*/\1/p' | head -1)
        dec=$(printf '%s\n' "$out" | sed -n 's/^decode .*(\([0-9.]*\) tok\/s).*/\1/p' | head -1)
        hot=$(printf '%s\n' "$out" | sed -n 's/.*HOT hits [0-9]* (\([0-9.]*\)%).*/\1/p' | head -1)
        rd=$(printf '%s\n' "$out" | sed -n 's/.*(\([0-9.]*\) MiB\/tok).*/\1/p' | head -1)
        [ -z "$pre" ] && note="bench: no rate printed"
    fi
    [ -z "$hot" ] && hot="-"
    [ -z "$rd" ] && rd="-"
    [ -z "$pre" ] && pre="-"
    [ -z "$dec" ] && dec="-"

    # Two arms, same prompt, same greedy rule: identical text is the pass.
    say "   coherence (device vs --cpu)"
    dev=$(arm "$model" -p "$PROMPT" -n "$TOKENS" --ctx "$CTX" --greedy | strip)
    cpu=$(arm "$model" -p "$PROMPT" -n "$TOKENS" --ctx "$CTX" --greedy --cpu | strip)
    same="FAIL"
    if [ -n "$dev" ] && [ "$dev" = "$cpu" ]; then same="yes"; else
        bad=1
        if [ -z "$dev" ]; then
            note="device: $(grep -h '^kraken:' "$ERRF" | head -1 | cut -c1-70)"
            [ "$note" = "device: " ] && note="device arm produced nothing"
        elif [ -z "$cpu" ]; then note="cpu arm produced nothing"
        else
            step=0
            while IFS= read -r a && IFS= read -r b; do
                step=$((step + 1))
                [ "$a" != "$b" ] && break
            done < <(printf '%s\n%s\n' "$dev" "$cpu" | paste - - 2>/dev/null)
            [ -z "$note" ] && note="text differs at line $step (device: $(printf '%s' "$dev" | cut -c1-32))"
        fi
    fi
    emit "| $name | $arch${family:+ }${family:+/ $family} | $pre | $dec | $same | $hot% | $rd | $idx | $note |"
done

if [ "$bad" -ne 0 ]; then
    say "some rows failed: see the device==cpu column and the note"
    exit 1
fi
say "every model loaded and both arms produced identical text"
exit 0
