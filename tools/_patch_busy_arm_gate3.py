#!/usr/bin/env python3
# Replace the two checks whose premise this run disproved, and add the one that
# is actually true: the queue changes what the device has left, not what the
# enqueue costs.
import io, sys

def load(path):
    raw = io.open(path, 'r', encoding='utf-8', newline='').read()
    return raw.replace('\r\n', '\n'), raw.count('\r\n') > 0

def save(path, txt, crlf):
    io.open(path, 'w', encoding='utf-8', newline='').write(
        txt.replace('\n', '\r\n') if crlf else txt)

def sub(txt, old, new, count=1, tag=''):
    n = txt.count(old)
    if n != count:
        print('ANCHOR FAIL %s: found %d, expected %d' % (tag, n, count))
        print(old[:500]); sys.exit(1)
    return txt.replace(old, new)

B, crlf = load('tools/bench_ops.cpp')

old = """    // The defect the arm was built to stop: quoting one half of a call as the
    // call. A legacy-stream record is ~0.3 us of host enqueue and ~24 us of
    // device marker; a table that measures 28 us by T(N)-T(1) has measured the
    // device half and called it the record.
    check(brec.device_us > 4.0 * brec.host_us,
          "a record's device half is >=4x its host half (so the two figures a "
          "table might quote are not the same number)");
    check(bcopy.device_us < 4.0 * bcopy.host_us,
          "a small launch's halves are the SAME order, which is why one number "
          "was good enough for it and was not for the record");
    return bad;"""
new = """    // The defect the arm was built to stop: quoting one half of a call as the
    // call. A legacy-stream record is ~0.3 us of host enqueue and ~28 us of
    // device marker; a table that measures 28 us by T(N)-T(1) has measured the
    // device half and called it the record.
    check(brec.device_us > 4.0 * brec.host_us,
          "a record's device half is >=4x its host half (so the two figures a "
          "table might quote are not the same number)");
    // And the claim this run DISPROVED, inverted into the one it supports: the
    // queue state does not move the host enqueue. The arm used to print a
    // "busy" record of 0.32 us beside a "quiet" 28.2 us from a different
    // method and read the ratio as queue state; both states enqueue in 0.28 us,
    // and the 28.2 was the device half. A future change that really does make
    // the enqueue queue-dependent owes this check an edit.
    check(brec.host_us < 2.0 * erec.host_us && erec.host_us < 2.0 * brec.host_us,
          "the host enqueue is the same in both queue states (the queue changes "
          "what the device has left, not what the enqueue costs)");
    check(ecopy.device_us * 4.0 < erec.device_us,
          "the instrument resolves the device cost per op (an empty queue: a "
          "launch vs a record, >=4x apart)");
    return bad;"""
B = sub(B, old, new, tag='checks')

save('tools/bench_ops.cpp', B, crlf)
print('patched: gate_busy_queue checks (fullness, halves, state-independent enqueue, per-op resolution)')
