#!/usr/bin/env python3
# The refused-startup path freed the backend and then unwound past the engine,
# whose destructor releases buffers through that pointer (SIGSEGV, exit 139).
# Anchors are deliberately backslash-free: no string literals are matched.
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

OLD = """        delete be;
        return 1;"""

NEW = """        // A refused startup must not free the backend: the engine holds
        // pointers into the backend's buffers and its own destructor releases
        // them through that pointer, so deleting the backend here freed it and
        // then unwound past the engine -- SIGSEGV (exit 139) on a path whose
        // whole job is to print one line and stop. `exit` skips that unwind.
        std::exit(1);"""

for path in ('src/main_cli.cpp', 'src/main_server.cpp'):
    t, crlf = load(path)
    # `delete be;` appears seven times in main_cli; the refusal path is the one
    # directly followed by `return 1;` at the same indent, which sub() asserts.
    t = sub(t, OLD, NEW, tag=path)
    save(path, t, crlf)
    print('ok', path)
