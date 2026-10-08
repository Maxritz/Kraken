---
name: "ponytail-diag"
description: "Structured debugging from code-as-data, one-line verdict by default. Trigger: what should vs did happen, /ponytail-diag."
---
# Ponytail Diag

Diagnose failures by building an internal model of the code (call graph,
control-flow branches, data-state transitions) and emitting only the verdict.
All heavy analysis happens in reasoning — **zero output tokens** until the
final verdict line.

## Default (minimal output)

Emit exactly one line:

`ROOT CAUSE: <file>:<line> — <one-line cause> | FIX: <one-line direction> | PROBE: <line> <expr>`

Example: `ROOT CAUSE: parser.py:42 — len(fields)<3 not guarded before unpack | FIX: add guard or use unpack-safe pattern | PROBE: 42 assert len(fields)>=3`

No prose. No tables. No templates.

## When to expand

User says "more", "deep dive", "show work", "expanded", "EXPANDED MODE",
"SHOW WORK", or explicitly requests fishbone/truth tables/flowcharts → emit all
sections below. Emit sections in order, labeled headers, no prose between.

## FLOW

Trace entry → exit. Branches annotated `[br:F]` (false) or `[br:T]` (true).
First unexpected value = probe target. Value materializing from nowhere =
skipped branch or stale alias.

## DATA

Variable lifecycle: `[init → assign → mutate → use]`. nil/None appearing at
branch with no prior assignment = flag.

## BRANCHES

For every if/switch/match on failing path, record concrete value causing it.

## TRUTH TABLES

When ≥2 flags combine, enumerate all combinations. Mark the actual failing row
with `← (failing)` and mark `untested` / `unreachable` paths.

```
flA=F flB=T → path-D  ← (failing)
flA=T flB=F → path-E  <untested>
flA=T flB=T → path-F  <unreachable>
```

## HYPOTHESES

≤4 live hypotheses. Each:

```
H<n>: <claim>
For: <evidence>  Against: <evidence>
Test: <cheapest experiment>
Cost: low|med|hi  Status: confirmed|refuted|unknown
```

## FISHBONE / ISHIKAWA

When ≥3 subsystems at fault, show bottom-up test for each category:
Method, Machine, Material, Man.

## 5 WHYS

For opaque single-cause failures, show 5-question drill-down ending in
a requirement / dependency / config choice.

## BARRIER ANALYSIS

What should have stopped this? Existence + bypass = logic error.
Absent = design gap → TODO.

## FTA (FAULT TREE)

Top event → AND/OR gates → basic events.
AND = all parents must occur. OR = any suffices.

## CHANGE ANALYSIS

"Worked last week" — smallest code/env diff correlating with onset.

## CONCURRENCY

Enumerate interleavings. No happens-before = defect.

## TIMING/DRIFT

Sample clocks + RSS + disk + CPU. Spike >2σ at bad-state boundary = suspect.

## TODO

```
TODO <file>:<line>: [sev:P0|P1|P2] <issue> | fix: <candidate> | test: <repro>
```

P0 — root defect / blocks repro. P1 — doesn't block repro. P2 — hardening.

## Handoff

When a hypothesis reaches `confirmed`:
emit `HANDOFF: debug-fix ready — one repro, one patch, one verify.`
Do not apply fixes in diag mode.

## Boundaries

- Zero guessing. Every annotation traces to real input, call site, or evidence.
  If static-only, say `PROBE NEEDED`.
- If no known-good reference exists, validate by invariants alone.
  Do not fabricate a reference.
- Probe placement: branch decision → function entry → invariant assert before
  nil propagates. Suggest exact line + expression.
- Does not apply fixes. Does not write study docs unless asked.
- `stop ponytail-diag` or `normal mode`: revert.