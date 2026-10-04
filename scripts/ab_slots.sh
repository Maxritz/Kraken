#!/bin/bash
# Interleaved A/B: raise the HOT slot cap past the 4590-expert working set.
#
# The trace shows all 4590 COLD misses are FIRST TOUCHES, so the only way to
# remove NVMe traffic is to hold the whole working set resident at once. Working
# set is 4590 keys; the default cap leaves 4340 resident. This asks whether 250
# more slots is worth the VRAM.
#
# Interleaved on purpose: sequential arms measured minutes apart differ by more
# than the effect under test (page cache warms between runs -- one pair of
# back-to-back default runs measured 10.6 and 6.6 tok/s). kraken is killed
# after every arm.
M=G:/More-models/laguna-xs2-Q4_K_M.gguf
P="what is the capital of france?"

run() {  # $1 = tag, rest = extra args
  tag="$1"; shift
  timeout 300 ./build-hip/kraken.exe -m "$M" \
      --prompt "$P" --max-tokens 24 --temp 0 --greedy --chat "$@" \
      > "ab_$tag.out" 2>&1
  taskkill //F //IM kraken.exe //T >/dev/null 2>&1
  local d s c
  d=$(grep -aoE "decode +[0-9]+ tok in [0-9.]+ ms = [0-9.]+ tok/s \([0-9.]+ ms/tok\)" "ab_$tag.out" | head -1)
  s=$(grep -aoE "[0-9]+/[0-9]+ slots resident" "ab_$tag.out" | head -1)
  c=$(grep -aoE "COLD misses [0-9]+ \([0-9.]+%\)" "ab_$tag.out" | head -1)
  echo "$tag: ${d:-no-decode-line} | ${s:-no-slots} | ${c:-no-cold}"
}

for rep in 1 2; do
  run "A$rep"
  run "B$rep" --expert-cache-slots 4700
done