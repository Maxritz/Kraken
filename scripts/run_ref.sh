#!/bin/bash
# run_ref.sh MODEL LOGFILE [extra llama-cli args...]
# Runs the ROCm reference llama-cli, records peak process working set,
# and ALWAYS kills the process when finished.
MODEL="$1"; shift
LOG="$1"; shift

ROCM="H:/LLAMA-bins/rocm10/llama-cli.exe"
rm -f "$LOG" "$LOG.mem"

# Poll the llama process working set while it runs.
(
  peak=0
  while true; do
    ws=$(powershell -NoProfile -Command "\$p=Get-Process -Name 'llama-cli' -ErrorAction SilentlyContinue | Sort-Object WS -Descending | Select-Object -First 1; if(\$p){[math]::Round(\$p.WS/1GB,2)}else{0}" 2>/dev/null | tr -d '\r' | grep -E '^[0-9.]+$')
    [ -z "$ws" ] && ws=0
    echo "$ws" > "$LOG.mem.now"
    peak=$(awk -v a="$peak" -v b="$ws" 'BEGIN{print (b>a)?b:a}')
    echo "$peak" > "$LOG.mem"
    sleep 1
  done
) &
POLLER=$!

timeout "${REF_TIMEOUT:-300}" "$ROCM" -m "$MODEL" "$@" </dev/null > "$LOG" 2>&1
rc=$?

# Kill the reference binary and the poller. Nothing of ours survives this run.
taskkill //F //IM llama-cli.exe //T >/dev/null 2>&1
kill $POLLER 2>/dev/null
wait $POLLER 2>/dev/null

echo "rc=$rc"
echo "peak_RAM_GiB=$(cat "$LOG.mem" 2>/dev/null || echo '?')"
echo "=== timings ==="
grep -aE "load time|prompt eval time|eval time|total time|sampling time|Prompt:" "$LOG"