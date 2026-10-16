#!/usr/bin/env bash
# The refused-startup path, end to end, through both CLIs -- it has no other
# automated proof. Worth its own check because both CLIs used to answer a failed
# Engine::init with `delete be; return 1;`: the refusal line printed and the
# process then died with SIGSEGV (rc=139), unwinding past the engine, whose
# destructor frees buffers through the backend the caller had already released.
# `std::exit(1)` is the fix and this is what pins it.
#
# No device is needed: `KRK_VRAM_TOTAL_MB` makes the CPU backend report a card of
# that size (include/krk/vram_guard.hpp), so a budget too large for it is refused
# exactly as on real hardware -- and the check runs beside a device run instead of
# fighting it for the card. The same variable is what the suite uses for its
# Engine::init arm (test_expert_cache_byte_budget).
#
# Verdicts, one line per arm, and the arms include the script's own red
# construction: the same budget on a card large enough must NOT be refused, so an
# unconditional refusal cannot read as a pass.
#
#   scripts/vram_refusal_check.sh
#
# Environment: KRK_BIN, KRK_SERVER_BIN, KRK_MODEL (an MoE GGUF), KRK_CARD_MB
# (the simulated card, default 2048), KRK_BUDGET_MB (default 2048 -- over the room
# the 1024 MiB reserve leaves), KRK_TIMEOUT (seconds per run, default 120).
#
# A crash is the failure this exists to catch, so the exit status is read as a
# signal number as well: 139 (SIGSEGV) or a 0xC0000005 Windows code is reported
# as CRASHED, never as "the refusal was rejected".
set -u

BIN=${KRK_BIN:-build-hip/kraken.exe}
SERVER_BIN=${KRK_SERVER_BIN:-build-hip/kraken-server.exe}
MODEL=${KRK_MODEL:-models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf}
CARD_MB=${KRK_CARD_MB:-2048}
BUDGET_MB=${KRK_BUDGET_MB:-2048}
TIMEOUT=${KRK_TIMEOUT:-120}
TMP=$(mktemp -d)

fails=0
pass() { printf 'PASS   %-46s %s\n' "$1" "$2"; }
bad()  { printf 'FAIL   %-46s %s\n' "$1" "$2"; fails=$((fails + 1)); }

for f in "$BIN" "$SERVER_BIN" "$MODEL"; do
    if [ ! -e "$f" ]; then
        printf 'SKIP   %-46s %s does not exist\n' "$(basename "$f")" "$f"
        rm -rf "$TMP"
        exit 1
    fi
done

# One CLI run against a simulated card. Echoes the exit status.
run_cli() { # <binary> <log-prefix> <extra args...>
    local bin=$1 prefix=$2; shift 2
    ( KRK_VRAM_TOTAL_MB="$CARD_MB" timeout "$TIMEOUT" "$bin" --cpu --model "$MODEL" \
        -p "The capital of France is" -n 1 --greedy --ctx 64 -v "$@" \
        >"$TMP/$prefix.out" 2>"$TMP/$prefix.err" ); echo $?
}

# The server takes a narrower flag set than the CLI (no -p/-n/--greedy), so its
# invocation is its own: if it does NOT refuse it binds a port and listens, and
# the timeout is what turns that into a verdict instead of a hung check.
run_server() { # <log-prefix> <expert-cache-mb>
    ( KRK_VRAM_TOTAL_MB="$CARD_MB" timeout "$TIMEOUT" "$SERVER_BIN" --cpu \
        --model "$MODEL" --ctx 64 --expert-cache-mb "$2" -v \
        >"$TMP/$1.out" 2>"$TMP/$1.err" ); echo $?
}

verdict() { # <rc> <prefix> -> CRASHED / the rc
    local rc=$1 prefix=$2
    if [ "$rc" -eq 139 ] || [ "$rc" -gt 128 ]; then
        echo "CRASHED (rc=$rc)"
    elif grep -qi "0xc0000005\|access violation" "$TMP/$prefix.err" 2>/dev/null; then
        echo "CRASHED (access violation)"
    else
        echo "rc=$rc"
    fi
}

printf 'simulated card %s MiB, requested budget %s MiB, model %s\n' \
    "$CARD_MB" "$BUDGET_MB" "$MODEL"
printf '\n'

# 1. The refusal: this is the path that segfaulted, so the exit status is the
#    check and the message is the evidence.
rc=$(run_cli "$BIN" refuse --expert-cache-mb "$BUDGET_MB")
v=$(verdict "$rc" refuse)
if [ "$rc" -eq 1 ] && grep -q "does not fit" "$TMP/refuse.err"; then
    pass "refused (kraken)" "$v, names the largest that fits"
else
    bad "refused (kraken)" "$v, expected rc=1 with a 'does not fit' refusal"
    sed 's/^/       | /' "$TMP/refuse.err" | tail -3
fi

# 2. The advice in the refusal must be a working command: take the budget it
#    names and run it. A refusal that cannot be acted on is a worse failure than
#    a trim, because the user has no way back to a working run.
largest=$(sed -n 's/.*--expert-cache-mb \([0-9][0-9]*\).*/\1/p' "$TMP/refuse.err" | tail -1)
if [ -z "$largest" ]; then
    bad "the refusal names a usable budget" "no --expert-cache-mb <MiB> in the message"
elif [ "$largest" -ge "$BUDGET_MB" ] || [ "$largest" -le 0 ]; then
    bad "the refusal names a usable budget" "named $largest against a rejected $BUDGET_MB"
else
    pass "the refusal names a usable budget" "$largest MiB"
    rc=$(run_cli "$BIN" advice --expert-cache-mb "$largest")
    v=$(verdict "$rc" advice)
    if [ "$rc" -eq 0 ] && ! grep -q "does not fit" "$TMP/advice.err"; then
        pass "and that budget runs" "$v"
    else
        bad "and that budget runs" "$v (the refusal's own advice does not work)"
        sed 's/^/       | /' "$TMP/advice.err" | tail -3
    fi
fi

# 3. The script's own red construction: the same budget on a card that fits must
#    produce a working run, so arm 1 cannot be passing because the CLI refuses
#    everything it is given.
card=$CARD_MB
CARD_MB=$((BUDGET_MB * 4))
rc=$(run_cli "$BIN" generous --expert-cache-mb "$BUDGET_MB")
v=$(verdict "$rc" generous)
CARD_MB=$card
if [ "$rc" -eq 0 ] && ! grep -q "does not fit" "$TMP/generous.err"; then
    pass "the same budget on a large card runs" "$v"
else
    bad "the same budget on a large card runs" "$v (the refusal is unconditional)"
    sed 's/^/       | /' "$TMP/generous.err" | tail -3
fi

# 4. The server refuses the same way and does not start listening. A server that
#    ignored it would hang here, which is what the timeout is for; rc=124 is
#    reported as such rather than as a refusal.
rc=$(run_server server "$BUDGET_MB")
v=$(verdict "$rc" server)
if [ "$rc" -eq 1 ] && grep -q "does not fit" "$TMP/server.err"; then
    pass "refused (kraken-server)" "$v, names the largest that fits"
elif [ "$rc" -eq 124 ]; then
    bad "refused (kraken-server)" "still running after ${TIMEOUT}s (it did not refuse)"
else
    bad "refused (kraken-server)" "$v, expected rc=1 with a 'does not fit' refusal"
    sed 's/^/       | /' "$TMP/server.err" | tail -3
fi

printf '\n'
if [ "$fails" -eq 0 ]; then
    printf '4 arms, 0 failed\n'
    rm -rf "$TMP"
else
    printf '4 arms, %d failed -- logs kept in %s\n' "$fails" "$TMP"
fi
[ "$fails" -eq 0 ]
