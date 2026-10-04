#!/bin/bash
# Interleaved A/B of --expert-warm-prefetch.
#
# The corrected measurement says 26.6 ms of a 94.8 ms decode step is synchronous
# expert staging (upload_paged 721 us/call host, warm_read 192 us/call host),
# and the host floor is 0.63 us/op, so those are real waits rather than
# recorder cost. gemv is NOT the problem: probe_gemv measures it at 2.78 us.
#
# The flag takes its value as the NEXT argv (main_cli calls next()), so it is
# `--expert-warm-prefetch 1`, not `=1`. It fills WARM at load rather than on
# demand, so the hypothesis is that the 26.6 ms moves into load time and decode
# gets faster. If decode does not move, the staging cost is per-token work that
# caching cannot remove.
#
# Interleaved because sequential arms drift more than the effect under test.
M=G:/More-models/laguna-xs2-Q4_K_M.gguf
P="what is the capital of france?"

run() {
  tag="$1"; shift
  timeout 300 ./build-hip/kraken.exe -m "$M" \
      --prompt "$P" --max-tokens 24 --temp 0 --greedy --chat "$@" \
      > "pf_$tag.out" 2>&1
  taskkill //F //IM kraken.exe //T >/dev/null 2>&1
  local d t h
  d=$(grep -aoE "decode +[0-9]+ tok in [0-9.]+ ms = [0-9.]+ tok/s" "pf_$tag.out" | head -1)
  t=$(grep -aoE "[0-9]+ promotions, [0-9]+ MiB in [0-9.]+ ms" "pf_$tag.out" | head -1)
  # Hash the generated text, not the stats line: the stats line carries a
  # variable millisecond count, so hashing it reports timing as nondeterminism.
  h=$(sed 's/\x1b\[[0-9;]*m//g' "pf_$tag.out" | tr -d '\r' \
      | grep -aE "^(Okay|The|Paris|I|It|There|Yes|No|Sorry)" | head -1 \
      | md5sum | cut -c1-8)
  echo "$tag: ${d:-FAILED} | ${t:-no-transfer} | text=$h"
}

for rep in 1 2; do
  run "A$rep"
  run "B$rep" --expert-warm-prefetch 1
done