#!/usr/bin/env python3
# The fullness checks compared a 512-call pre-issue against an empty queue with a
# 3.0x bound. Measured across runs: 583/177 = 3.3x, 660/177 = 3.7x, and on a
# busier machine 518/178 = 2.9x -- so a 3.0 bound flaps. A queue that is not
# fuller reports 1.0x, so 2.0 discriminates and does not flap.
import io, sys

def load(p):
    raw = io.open(p, 'r', encoding='utf-8', newline='').read()
    return raw.replace('\r\n', '\n'), raw.count('\r\n') > 0

def save(p, t, crlf):
    io.open(p, 'w', encoding='utf-8', newline='').write(
        t.replace('\n', '\r\n') if crlf else t)

def sub(t, old, new, count=1, tag=''):
    if t.count(old) != count:
        print('ANCHOR FAIL %s: %d != %d' % (tag, t.count(old), count))
        sys.exit(1)
    return t.replace(old, new)

B, crlf = load('tools/bench_ops.cpp')

B = sub(B, """    check(bcopy.tail_us > 3.0 * ecopy.tail_us,
          "a queue holding `depth` calls is measurably fuller than an empty one "
          "(device left to drain: launch arm)");
    check(bcopy.device_us > 3.0 * ecopy.device_us,
          "and the same difference is visible per call");""",
"""    // 2.0x, not 3.0x: measured ratios run 2.9x on a busy machine to 3.7x on a
    // quiet one, while a queue that is NOT fuller reports 1.0x, so 2.0
    // discriminates without flapping with the machine's load. The ratios are
    // printed so the margin is visible in the gate's own output.
    const double tail_ratio = ecopy.tail_us > 0 ? bcopy.tail_us / ecopy.tail_us : 0.0;
    const double dev_ratio = ecopy.device_us > 0 ? bcopy.device_us / ecopy.device_us : 0.0;
    check(tail_ratio > 2.0,
          "a queue holding `depth` calls is measurably fuller than an empty one "
          "(device left to drain: launch arm)");
    check(dev_ratio > 2.0, "and the same difference is visible per call");""",
tag='thresholds')

B = sub(B, """    std::printf("    device left to drain: %.0f us deep vs %.0f us empty "
                "(launch %.0f vs %.0f us)\\n",
                brec.tail_us, erec.tail_us, bcopy.tail_us, ecopy.tail_us);""",
"""    std::printf("    device left to drain: %.0f us deep vs %.0f us empty "
                "(launch %.0f vs %.0f us = %.2fx, needs >2.0x)\\n",
                brec.tail_us, erec.tail_us, bcopy.tail_us, ecopy.tail_us,
                ecopy.tail_us > 0 ? bcopy.tail_us / ecopy.tail_us : 0.0);""",
tag='print ratio')
save('tools/bench_ops.cpp', B, crlf)

# keep the prose in step with the bound
for path, old, new in (
    ('AGENTS.md',
     'it asserts the queue\n  really is fuller (launch-arm tail, >=3x)',
     'it asserts the queue\n  really is fuller (launch-arm tail >2x, measured 2.9-3.7x)'),
):
    t, c = load(path)
    t = sub(t, old, new, tag=path)
    save(path, t, c)

print('patched: tools/bench_ops.cpp (2x), AGENTS.md')
