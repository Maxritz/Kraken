---
name: "sherlock-it"
description: "Measurement-driven performance investigation and optimisation. Profiles every execution layer, creates instrumentation traps to expose hidden bottlenecks, identifies root causes, implements verified optimisations, and retests against measured baselines."
---
**Objective:** Find, prove, fix and verify every significant performance bottleneck. Never optimise by guesswork.

## Modes

| Mode | Purpose |
|---|---|
| `scan` | Rapid profiling, bottleneck identification and optimisation opportunities. |
| `trace` | Per-operation timing, data-flow analysis, launch overhead and resource tracking. |
| `deep` | Full kernel, shader, memory, hardware and execution-path investigation. |
| `autopsy` | Exhaustive investigation with additional instrumentation, diagnostic traps, competing hypotheses and repeated validation. |

Default: `trace`. Escalate only when evidence is insufficient.

## Mandatory workflow

**1. Baseline**
- Build and run the unmodified target.
- Record hardware, driver, compiler, build flags, workload, input sizes and runtime configuration.
- Capture throughput, latency, resource utilisation and correctness.
- Preserve reproducible baseline results.

**2. Decompose**
- Map the complete execution and data flow.
- Enumerate every kernel, shader, operator, dispatch, transfer, synchronisation point and fallback path.
- Record invocation counts, tensor dimensions, formats, bytes moved and dependencies.
- Distinguish host time, device execution time, queue wait, launch overhead and end-to-end wall time.

**3. Instrument**
- Add targeted diagnostic traps wherever visibility is insufficient.
- Capture per-call latency, launch dimensions, memory transactions, cache behaviour, occupancy, register pressure, stalls and synchronisation.
- Add execution markers, timestamp queries, counters, trace events and selective debug logging.
- Use sampling or selective instrumentation when tracing overhead distorts results.
- Compare instrumented and uninstrumented runs.

**4. Investigate**
- Rank bottlenecks by cumulative cost and impact on end-to-end latency.
- Compare actual throughput against workload-specific hardware ceilings, not theoretical peak numbers alone.
- Investigate inefficient algorithms, poor instruction selection, scalar fallbacks, memory-bound execution, redundant transfers, serial dependencies, excessive dispatches and underutilisation.
- Form explicit hypotheses and design measurements that can disprove them.

**5. Optimise**
- Prioritise changes by measured expected impact, implementation complexity and correctness risk.
- Research hardware-specific instructions, compiler behaviour, alternative algorithms and established implementations.
- Change one meaningful variable at a time.
- Preserve a known-good implementation and isolate experimental variants.
- Avoid speculative rewrites and unrelated refactoring.

**6. Verify**
- Rebuild and rerun identical workloads.
- Compare raw per-kernel performance, aggregate operator costs and end-to-end throughput.
- Verify numerical correctness, edge cases, memory safety and concurrency.
- Reject regressions and improvements that cannot be reproduced.
- Retain results, measurements and the exact configuration for every accepted change.

## Diagnostic traps

Create additional instrumentation dynamically when unexplained costs remain.

| Trap | Captures |
|---|---|
| Dispatch trap | Launch count, grid size, workgroup dimensions, launch latency |
| Memory trap | Bytes read/written, alignment, bandwidth, cache misses |
| Instruction trap | Generated ISA, vectorisation, instruction mix, fallback paths |
| Synchronisation trap | Barriers, fences, queue waits, idle gaps |
| Allocation trap | Allocation frequency, memory pressure, temporary buffers |
| Transfer trap | Host-device copies, staging, transfer latency |
| Dependency trap | Serial execution, critical path, hidden synchronisation |
| Accuracy trap | Output differences, numerical error, precision-related performance |
| Regression trap | Before/after timings, workload sensitivity and performance variance |

Traps must have a clear hypothesis, measurable output and a removal or disable mechanism.

## Data flow

`Workload → Baseline → Execution map → Instrumentation → Measurements → Bottleneck ranking → Hypothesis → Targeted experiment → Optimisation → Retest → Evidence`

At each stage, preserve the evidence needed by the next stage. If measurements are inconclusive, return to instrumentation rather than guessing.

## Output contract

Produce a concise, evidence-backed report containing:

- **Findings:** Ranked bottlenecks with raw measurements and invocation counts.
- **Root causes:** Proven causes, suspected causes and unresolved questions, explicitly distinguished.
- **Hardware analysis:** Relevant architectural limits and observed utilisation.
- **Optimisation plan:** Specific changes, expected impact and validation criteria.
- **Results:** Before/after measurements, percentage change, correctness and reproducibility.
- **Next actions:** The highest-impact unresolved investigation.

Use tables for per-kernel and per-shader measurements. Include raw timings and workload dimensions, not just percentages or aggregate scores.

## Enforcement rules

- Never assume a kernel is optimal because it uses a specialised instruction.
- Never equate aggregate device time with individual invocation latency.
- Never confuse theoretical peak performance with achievable workload performance.
- Never accept an optimisation based on a single noisy measurement.
- Never claim a root cause without evidence.
- Never stop at the first bottleneck if another significant bottleneck remains.
- Never discard the original baseline or correctness tests.
- Always redirect the investigation towards the highest-impact measurable performance gap.

**Success criterion:** Reproducible performance improvement with verified correctness, explained by evidence at the relevant execution layer.


example of a sherlock trace:
stdout:
     component           ops   %dev      dev us     idle us     host us
     attention            24  95.4%   535880.80      377.62       49.20
     projections         187   3.2%    17723.32     2026.19      460.80
     recurrent            90   1.1%     6127.59     1170.02      143.80
     attention-mix        72   0.2%     1335.28     1102.27      111.60
     head+sample           1   0.0%       98.20       91.80       56.90
     bias                 18   0.0%       90.03      265.52       66.60
     transfer              1   0.0%       79.80     1087.10     1084.30
     norms                67   0.0%       72.98      784.24      139.40
     residual             48   0.0%       23.40      198.44       73.30
     ffn-activate         42   0.0%       21.52      372.84       67.70
     embed                 1   0.0%        0.48       25.44        2.40
   instrumentation floor, 256 empty ops timed through the same begin/end path: 0.70 us host, 19.54 us device each.

---