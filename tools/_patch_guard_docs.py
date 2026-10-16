#!/usr/bin/env python3
# Structural line anchors: find the sentence by a distinctive single line, then
# splice. No multi-line matching, so line wrapping cannot break the patch.
import io, sys

def patch(path, needle, insert_after_line, extra_lines):
    raw = io.open(path, 'r', encoding='utf-8', newline='').read()
    crlf = raw.count('\r\n') > 0
    lines = raw.replace('\r\n', '\n').split('\n')
    hits = [i for i, l in enumerate(lines) if needle in l]
    if len(hits) != 1:
        print('ANCHOR FAIL %s: %d hits for %r' % (path, len(hits), needle))
        sys.exit(1)
    i = hits[0]
    if insert_after_line and i + 1 < len(lines) and lines[i + 1].startswith(needle[:0] or ''):
        pass
    lines[i + 1:i + 1] = extra_lines
    out = '\n'.join(lines)
    io.open(path, 'w', encoding='utf-8', newline='').write(
        out.replace('\n', '\r\n') if crlf else out)
    print('ok %s @ line %d' % (path, i + 1))

guards = [
    '  **The engine enforces it now.** `Engine::configure_expert_cache` holds back',
    '  a VRAM reserve (`max(1 GiB, total/16)`, or `KRK_VRAM_RESERVE_MB`), clamps the',
    '  *auto* budget to what is left of it, and **refuses** an explicit',
    '  `--expert-cache-mb` that cannot fit, naming the largest budget the card can',
    '  hold. The warm-up then re-checks that reserve every round against',
    '  `device_free_bytes()` -- the plan-time arithmetic cannot see the workspaces',
    '  and the KV tier, which are allocated after the budget is chosen -- and stops',
    '  with `stopped at the VRAM reserve: N MiB kept free of M MiB total`. Measured',
    '  on Qwen3-MoE-4x0.6B: a normal run does not trigger it (text byte-identical),',
    '  `KRK_VRAM_RESERVE_MB=16000` promotes **0** experts and prints the report, and',
    '  `--expert-cache-mb 20000` on this 15.9 GiB card exits **rc=1** with',
    '  `does not fit: 16304 MiB of VRAM, 599 MiB already in use, 1024 MiB reserved',
    '  -> the largest budget this card can hold is 14681 MiB`.',
]

refusal = [
    '  The same edit\'s *refusal* path had a second defect of the same family -- it',
    '  had never been reachable, so nobody had run it: both CLIs answered a failed',
    '  `init` with `delete be; return 1;`, which frees the backend and then unwinds',
    '  past the engine, whose destructor releases buffers through that pointer. The',
    '  refusal printed its line and then died with **SIGSEGV (rc=139)**. It is',
    '  `std::exit(1)` now, in `main_cli.cpp` and `main_server.cpp` both. A new',
    '  early-return path is a new path: run it.',
]

patch('AGENTS.md', 'keep headroom.** A 12.2 tok/s', True, guards)
patch('AGENTS.md', 'missing `return`.', True, refusal)
