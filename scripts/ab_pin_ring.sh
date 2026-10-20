#!/usr/bin/env bash
# Interleaved, order-balanced A/B for the expert cache's hot ring.
#
#   arm A: KRK_EXPERT_PIN=0  -- every pin spends the pool's budget (previous)
#   arm B: ring on           -- the index's ranked prefix is pinned in a budget
#                              the LRU window does not see (new); RING_MB=<n>
#                              sizes it instead of 1/6 of the usable card
#
# Arms alternate slots every pair, so the slot a run occupies inside a pair
# cannot be read as the arm's effect (AGENTS.md: the slot is worth several
# percent). Throughput comes from --bench, which prints p refill and decode
# rates; text identity is a separate pair of greedy runs, because a speed win
# with different tokens is a different answer rather than a faster one.
#
#   scripts/ab_pin_ring.sh <model.gguf> [pairs] [tokens]
#   RING_MB=2048 scripts/ab_pin_ring.sh <model.gguf> 5 64
set -u
KRAKEN=${KRAKEN:-./build-hip/kraken.exe}
MODEL=${1:?usage: ab_pin_ring.sh <model.gguf> [pairs] [tokens]}
PAIRS=${2:-3}
TOKENS=${3:-64}
PROMPT=${PROMPT:-"The capital of France is"}
CTX=${CTX:-512}

arm_env() { # prints the VAR=value list for an arm
    case "$1" in
        A) printf 'KRK_EXPERT_PIN=0' ;;
        B) [ -n "${RING_MB:-}" ] && printf 'KRK_EXPERT_PIN_MB=%s' "$RING_MB" ;;
    esac
}

rate() { # $1 = arm, $2 = prefill|decode
    grep -oE "^$2 +[0-9]+ tokens in [0-9.]+ ms +\([0-9.]+ tok/s\)" \
        "/tmp/ab_pin_$1.out" | tail -1 | sed -E 's/.*\(([0-9.]+) tok.*/\1/'
}

run_arm() { # $1 = arm label
    local envs
    envs=$(arm_env "$1")
    # shellcheck disable=SC2086
    env $envs timeout 1800 "$KRAKEN" -m "$MODEL" --bench --ctx "$CTX" \
        -n "$TOKENS" > "/tmp/ab_pin_$1.out" 2> "/tmp/ab_pin_$1.err"
    local rc=$?
    printf '%s\t%s\t%s\t%s\n' "$1" "$rc" "$(rate "$1" prefill)" \
        "$(rate "$1" decode)"
}

printf 'arm\trc\tprefill_tok_s\tdecode_tok_s\n'
for p in $(seq 1 "$PAIRS"); do
    if [ $(( p % 2 )) -eq 1 ]; then order="A B"; else order="B A"; fi
    for a in $order; do run_arm "$a"; done
    grep -h "hot ring admitted" "/tmp/ab_pin_B.err" | tail -1 >&2
    grep -h "HOT hits" "/tmp/ab_pin_A.err" | tail -1 >&2
    grep -h "HOT hits" "/tmp/ab_pin_B.err" | tail -1 >&2
done

strip() { sed -e '/^\[/d' -e '/^$/d' -e '/^KRAKEN /d' -e '/^kraken /d'; }
for a in A B; do
    # shellcheck disable=SC2046,SC2086
    env $(arm_env "$a") timeout 1800 "$KRAKEN" -m "$MODEL" --ctx "$CTX" \
        --greedy -n "$TOKENS" -p "$PROMPT" > "/tmp/ab_text_$a.out" 2>/dev/null
done
if cmp -s <(strip < /tmp/ab_text_A.out) <(strip < /tmp/ab_text_B.out); then
    echo "text: identical ($(strip < /tmp/ab_text_A.out | md5sum | cut -c1-12))"
else
    echo "text: DIFFERS" >&2
    diff <(strip < /tmp/ab_text_A.out) <(strip < /tmp/ab_text_B.out) | head >&2
fi
