#!/usr/bin/env python3
import io

p = 'docs/traces/launch-floor-and-drains-2026-10-11.txt'
raw = io.open(p, 'r', encoding='utf-8', newline='').read()
crlf = raw.count('\r\n') > 0
t = raw.replace('\r\n', '\n').rstrip('\n')

t += '''
================================================================================
6. the two guards, and the paths they are on
================================================================================
# A. a normal run must not trigger the reserve, and must not move a byte
$ ./build-hip/kraken.exe --model models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf \\
    -p "The history of computing is a history of abstraction" -n 32 --temp 0 \\
    --greedy --ctx 512 --expert-warmup 1
  rc=0   stdout md5 = 28fa19c3733a5cab9dc53611de71600c    (the recorded hash)
  [info ] vram policy: 6.0 GiB cap (0.6 GiB already in use, 1.2 GiB demand,
          15.3 GiB free of 15.9 GiB, 1.0 GiB reserved) -> expert budget 610.3 MiB
  [info ] expert warmup: even order, 2 rounds, 4 experts/layer deep; WARM 610 MiB
          of 610 MiB in 112 slots (610 MiB read), VRAM +112 experts (610 MiB, 112
          slots); 481 ms                     <- no reserve stop, warm-up complete

# B. the guard, forced (a reserve above what is free)
$ KRK_VRAM_RESERVE_MB=16000 ./build-hip/kraken.exe ... --expert-warmup 1
  rc=0   stdout md5 = 28fa19c3733a5cab9dc53611de71600c    (text unchanged)
  [info ] expert warmup: ... VRAM +0 experts (0 MiB, 0 slots); 155 ms
          (stopped at the VRAM reserve: 16000 MiB kept free of 16 MiB total)
  [stats ] device    0.77 GiB in use of 15.92 GiB (15.16 GiB free)

# C. the refusal, an explicit budget larger than the card
$ ./build-hip/kraken.exe --model models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf \\
    -p x -n 4 --greedy --ctx 512 --expert-cache-mb 20000
  rc=1
  kraken: expert budget 20000 MiB does not fit: 16304 MiB of VRAM, 599 MiB
  already in use, 1024 MiB reserved -> the largest budget this card can hold is
  14681 MiB (--expert-cache-mb 14681)
$ ./build-hip/kraken-server.exe --model ... --expert-cache-mb 20000
  rc=1   kraken-server: expert budget 20000 MiB does not fit: ... 14681 MiB

# Three defects surfaced while building these guards. Each one was a path that
# had never been executed before this session.
#   1. configure_expert_cache went from void to bool to carry the refusal, and
#      its fall-through exit kept no `return` -- UB. The compiler reported 0
#      errors, build_check.sh stayed green, dense models were FINE (they return
#      early), every MoE run died with rc=3 and EMPTY stdout, and kraken-tests
#      went 2927/2927 -> rc=1 with four groups dead. Fixed with the explicit
#      `return true;` at the function's end (engine.cpp).
#   2. the refusal path itself: `delete be; return 1;` freed the backend and then
#      unwound past the engine, whose destructor releases buffers through that
#      pointer -- SIGSEGV, rc=139, for a run whose whole job is to print one line.
#      Both CLIs are `std::exit(1)` now.
#   3. the intended one: the warm-up filled the tier to a budget chosen before the
#      workspaces and the KV tier existed, ending at 0.00 GiB free. The reserve
#      bounds it at promotion time, by asking the driver.

# Final gates on the fixed binary: ninja no work to do; kraken-tests 2927/2927
# rc=0; the tests-baseline check "no regression"; kraken-bench --gate rc=0 with 0
# FAIL; coherence 3 coherent / 0 not; build_check.sh rc=0 (1 standing WARN).
'''

io.open(p, 'w', encoding='utf-8', newline='').write(
    (t + '\n').replace('\n', '\r\n') if crlf else t + '\n')
print('ok', p)
