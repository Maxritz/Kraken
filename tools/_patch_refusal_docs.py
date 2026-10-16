#!/usr/bin/env python3
import io, sys

def load(path):
    raw = io.open(path, 'r', encoding='utf-8', newline='').read()
    crlf = raw.count('\r\n') > 0
    return raw.replace('\r\n', '\n').split('\n'), crlf

def save(path, lines, crlf):
    out = '\n'.join(lines)
    io.open(path, 'w', encoding='utf-8', newline='').write(
        out.replace('\n', '\r\n') if crlf else out)

def find_one(lines, needle):
    hits = [i for i, l in enumerate(lines) if needle in l]
    if len(hits) != 1:
        print('ANCHOR FAIL: %d hits for %r' % (len(hits), needle))
        sys.exit(1)
    return hits[0]

def insert_before(path, needle, new):
    lines, crlf = load(path)
    i = find_one(lines, needle)
    lines[i:i] = new
    save(path, lines, crlf)
    print('ok insert_before %s @ %d' % (path, i + 1))

def replace_line(path, needle, new):
    lines, crlf = load(path)
    i = find_one(lines, needle)
    lines[i:i + 1] = new
    save(path, lines, crlf)
    print('ok replace_line %s @ %d' % (path, i + 1))

# 1. The missing-return class is now a build error, not a warning.
insert_before('AGENTS.md',
              '- **Changing a `void` helper to `bool` to add a refusal is a two-part edit, and',
[
 '- **A missing `return` cannot land any more: `-Werror=return-type` is fatal for every',
 '  translation unit.** That defect class is compiler-visible -- `-Wall` already',
 '  contained `-Wreturn-type`, so the diagnostic was there to read every time and',
 '  nobody read it -- and `CMakeLists.txt` now makes it an error (skipped on MSVC),',
 '  which covers all 90 host TUs and the HIP ones rather than the one function that',
 '  failed. Proved by red construction: a `static bool` that falls off its end appended',
 '  to a clean `src/json.cpp` fails the build with',
 '  `error: non-void function does not return a value in all control paths',
 '  [-Werror,-Wreturn-type]` and ninja rc=1; reverting the file is green again.',
])

# 2. The refused startup is driven end to end, and the check is proved able to fail.
replace_line('AGENTS.md',
             '  for. 2940/2940.',
[
 '  for. **The refused startup is driven END TO END too.** The suite now builds an',
 '  MoE fixture, tells the CPU backend to report a 4 GiB card (`KRK_VRAM_TOTAL_MB`),',
 '  asks for a 4096 MiB budget against 3072 MiB of room, requires `Engine::init` to',
 '  RETURN false with a refusal naming 3072, and then destroys the engine -- the',
 '  unwinding that used to walk a freed backend. Its first check is a CONTROL: the',
 '  same fixture at 1024 MiB must initialize, or a fixture that cannot init at all',
 '  would make the refusal pass for the wrong reason.',
 '  `scripts/vram_refusal_check.sh` is that path through the real CLIs, device-free,',
 '  and it is the only proof of the `std::exit(1)` fix: put `delete be; return 1;`',
 '  back in `main_cli.cpp` and it prints `CRASHED (rc=139)` and exits 1. Its other',
 '  red construction is `KRK_CARD_MB=65536`, where the refusal arm correctly FAILs',
 '  (rc=0, no message) while the generous-card arm still passes -- that pair is what',
 '  stops an unconditional refusal from reading as a pass. It also takes the budget',
 '  the refusal names and RUNS it: a refusal whose advice is not a working command is',
 '  worse than a trim, because the user has no way back. 2944/2944.',
 '  `KRK_VRAM_TOTAL_MB` is read by the CPU backend alone and only while it has no',
 '  capacity of its own to report, so no GPU run can be affected by it; everything',
 '  else in the run (the headroom line, the warm-up reserve, `[stats ] device`) then',
 '  reports the simulated card, because that is the number every decision used.',
])

# 3. The trace: the raw transcripts.
p = 'docs/traces/launch-floor-and-drains-2026-10-11.txt'
lines, crlf = load(p)
t = '\n'.join(lines).rstrip('\n')
t += '''

================================================================================
8. the refused startup, end to end -- and the check proved able to fail
================================================================================
# The class, made unlandable. A `static bool` falling off its end appended to a
# clean src/json.cpp:
$ ninja -C build-hip krk_core ; echo rc=$?
  ...hipcc.exe ... -DNDEBUG -Werror=return-type -O3 -Wall -Wextra -Wshadow ...
  C:/.../src/json.cpp:407:1: error: non-void function does not return a value in all
  control paths [-Werror,-Wreturn-type]
  rc=1
# (file reverted; the next build is rc=0 with 0 error lines)

# The suite's own arm, through the real Engine::init on the CPU backend, with the
# card simulated -- the control first, so the refusal cannot pass vacuously:
$ ./build-hip/kraken-tests --run-group 38
  [info ] vram headroom: 4.00 GiB free of 4.00 GiB total, 1.00 GiB reserved
  [info ] vram policy: 3.5 GiB cap (... 4.0 GiB free of 4.0 GiB, 1.0 GiB reserved)
          -> expert budget 1024.0 MiB          <- the CONTROL initialized
  (refused: expert budget 4096 MiB does not fit: 4096 MiB of VRAM, 0 MiB already in
   use, 1024 MiB reserved -> the largest budget this card can hold is 3072 MiB
   (--expert-cache-mb 3072))
  [group test_expert_cache_byte_budget] 37/37 checks
  rc=0

# The same path through the CLIs, device-free (--cpu + the simulated card):
$ bash scripts/vram_refusal_check.sh ; echo rc=$?
  simulated card 2048 MiB, requested budget 2048 MiB
  PASS   refused (kraken)                               rc=1, names the largest that fits
  PASS   the refusal names a usable budget              1024 MiB
  PASS   and that budget runs                           rc=0
  PASS   the same budget on a large card runs           rc=0
  PASS   refused (kraken-server)                        rc=1, names the largest that fits

  4 arms, 0 failed
  rc=0

# Red construction 1: a card the budget fits. The refusal arm must fail, and the
# generous-card arm must still pass -- otherwise the arm is just "rc=1 anywhere".
$ KRK_CARD_MB=65536 bash scripts/vram_refusal_check.sh ; echo rc=$?
  FAIL   refused (kraken)                               rc=0, expected rc=1 with a 'does not fit' refusal
  FAIL   the refusal names a usable budget              no --expert-cache-mb <MiB> in the message
  PASS   the same budget on a large card runs           rc=0
  rc=1

# Red construction 2: the segfault this check exists for. `std::exit(1)` in
# main_cli.cpp replaced by the old `delete be; return 1;`:
$ bash scripts/vram_refusal_check.sh ; echo rc=$?
  FAIL   refused (kraken)                               CRASHED (rc=139), expected rc=1 with a 'does not fit' refusal
       | kraken: expert budget 2048 MiB does not fit: 2048 MiB of VRAM, 0 MiB already
       | in use, 1024 MiB reserved -> the largest budget this card can hold is 1024 MiB
  rc=1
# The refusal line printed and then the process died -- exactly the shape that made
# this path invisible before, since a correct refusal and a fatal one print the same
# thing. The fix was restored and the check is green again.

# Note for the next patch: main_cli.cpp is CRLF (1187 lines), so a multi-line
# anchor written with \\n matches nothing and the patch no-ops silently. The red
# construction above failed to apply the first time for exactly that reason, and the
# "RED 2" run that followed was of an unpatched binary -- which read as a pass.
'''
save(p, (t + '\n').split('\n'), crlf)
print('ok trace section 8')
