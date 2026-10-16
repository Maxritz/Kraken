#!/usr/bin/env python3
import io, sys

def insert_after(path, needle, lines):
    raw = io.open(path, 'r', encoding='utf-8', newline='').read()
    crlf = raw.count('\r\n') > 0
    body = raw.replace('\r\n', '\n').split('\n')
    hits = [i for i, l in enumerate(body) if needle in l]
    if len(hits) != 1:
        print('ANCHOR FAIL %s: %d hits for %r' % (path, len(hits), needle))
        sys.exit(1)
    body[hits[0] + 1:hits[0] + 1] = lines
    out = '\n'.join(body)
    io.open(path, 'w', encoding='utf-8', newline='').write(
        out.replace('\n', '\r\n') if crlf else out)
    print('ok %s @ %d' % (path, hits[0] + 1))

insert_after('AGENTS.md',
             '-> the largest budget this card can hold is 14681 MiB`.',
[
 '  **The host twin, the headroom line, and the test.** The same rule now covers',
 '  host RAM: `krk::host_reserve_bytes` is `max(2 GiB, installed/8)` (or',
 '  `KRK_RAM_RESERVE_MB`), and the WARM staging loop re-checks it every round',
 '  against `host_available_bytes()`. The ram policy already plans a 20% headroom,',
 '  but it plans it once, before the KV tier and the workspaces exist, and staging',
 '  is what spends it. Forced with `KRK_RAM_RESERVE_MB=200000` on',
 '  Qwen3-MoE-4x0.6B: `0 rounds, 4 experts/layer deep; WARM 0 MiB ... 0 MiB read,',
 '  VRAM +0 experts; 0 ms (stopped at the host RAM reserve: 200000 MiB kept free)`;',
 '  unforced, the same command completes (`2 rounds, WARM 610 MiB, 112 experts,',
 '  649 ms`). Every run now prints its headroom in its first lines, dense models',
 '  included -- `[info ] vram headroom: 15.57 GiB free of 15.92 GiB total, 1.00 GiB',
 '  reserved` -- because the fault this guards against reached `0.00 GiB free` and',
 '  the only line that showed it printed at the END of the run.',
 '  The decision itself is `include/krk/vram_guard.hpp`: pure arithmetic, no',
 '  device, no state, so `test_expert_cache_byte_budget` drives both halves of it',
 '  (13 checks: the exact-fit boundary, the refusal naming the largest budget in',
 '  MiB, a card with only the reserve left, and the backend that cannot report',
 '  capacity). **The first version of those checks was written in bytes against a',
 '  message formatted in MiB and the suite failed it** -- which is what they are',
 '  for. 2940/2940.',
])

p = 'docs/traces/launch-floor-and-drains-2026-10-11.txt'
raw = io.open(p, 'r', encoding='utf-8', newline='').read()
crlf = raw.count('\r\n') > 0
t = raw.replace('\r\n', '\n').rstrip('\n')
t += '''

================================================================================
7. the host-RAM twin, the headroom line, and the guard's own test
================================================================================
# The headroom line, on a DENSE model -- which sizes no expert tier at all, so
# before this change it printed nothing about VRAM:
$ ./build-hip/kraken.exe --model models/SmolLM2-135M-Instruct.Q4_K_M.gguf \\
    -p "The history of computing is a history of abstraction" -n 4 --greedy --ctx 256
  rc=0
  [info ] vram headroom: 15.57 GiB free of 15.92 GiB total, 1.00 GiB reserved
          (KRK_VRAM_RESERVE_MB overrides)

# The host-RAM staging guard, forced:
$ KRK_RAM_RESERVE_MB=200000 ./build-hip/kraken.exe --model \\
    models/Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf -p "..." -n 8 --greedy --ctx 512 \\
    --expert-warmup 1
  rc=0
  [info ] expert warmup: even order, 0 rounds, 4 experts/layer deep; WARM 0 MiB of
          610 MiB in 0 slots (0 MiB read), VRAM +0 experts (0 MiB, 0 slots);
          0 ms (stopped at the host RAM reserve: 200000 MiB kept free)

# ... and unforced, the same command must still fill the tier:
  [info ] expert warmup: even order, 2 rounds, 4 experts/layer deep; WARM 610 MiB
          of 610 MiB in 112 slots (610 MiB read), VRAM +112 experts (610 MiB, 112
          slots); 649 ms

# The guard decision is now testable without a device: include/krk/vram_guard.hpp
# holds the reserves and the fit decision as pure arithmetic, and
# test_expert_cache_byte_budget drives it.
  [39/54] test_expert_cache_byte_budget                 33/33      48ms
  2940/2940 checks passed        (was 2927/2927; the group went 20 -> 33 checks)
# Its first version checked a message formatted in MiB with byte-scale inputs and
# the suite failed it -- 32/33, the group FAILED -- which is the check earning its
# keep rather than being decorative.

# The busy-queue arm's fullness bound was also too tight: measured 583/177 = 3.3x
# and 660/177 = 3.7x on a quiet machine, but 518/178 = 2.9x on a loaded one
# against a 3.0x bound, so the gate flapped once. Now 2.0x with the ratio printed
# (a queue that is not fuller reports 1.0x), and the red construction still fails
# exactly the two fullness checks.
'''
io.open(p, 'w', encoding='utf-8', newline='').write(
    (t + '\n').replace('\n', '\r\n') if crlf else t + '\n')
print('ok trace')
