#!/usr/bin/env python3
# main_server.cpp has four `delete be;` sites; only the engine.init failure is
# the refusal path. Find it structurally: the first `delete be;` after
# `if (!engine.init(`.
import io, sys

path = 'src/main_server.cpp'
raw = io.open(path, 'r', encoding='utf-8', newline='').read()
crlf = raw.count('\r\n') > 0
text = raw.replace('\r\n', '\n')
lines = text.split('\n')

i = None
for k, l in enumerate(lines):
    if l.strip().startswith('if (!engine.init('):
        i = k
        break
if i is None:
    print('ANCHOR FAIL: no engine.init check'); sys.exit(1)

j = None
for k in range(i, len(lines)):
    if lines[k].strip() == 'delete be;':
        j = k
        break
if j is None or lines[j + 1].strip() != 'return 1;':
    print('ANCHOR FAIL: no delete-be/return-1 pair after init'); sys.exit(1)

indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
repl = [
    indent + '// A refused startup must not free the backend: the engine holds',
    indent + '// pointers into the backend\'s buffers and its own destructor',
    indent + '// releases them through that pointer, so deleting the backend here',
    indent + '// freed it and then unwound past the engine -- SIGSEGV on a path',
    indent + '// whose whole job is to print one line and stop. `exit` skips it.',
    indent + 'std::exit(1);',
]
print('patching lines %d..%d (1-indexed) of %s' % (j + 1, j + 2, path))
lines[j:j + 2] = repl
out = '\n'.join(lines)
io.open(path, 'w', encoding='utf-8', newline='').write(
    out.replace('\n', '\r\n') if crlf else out)
print('ok', path)
