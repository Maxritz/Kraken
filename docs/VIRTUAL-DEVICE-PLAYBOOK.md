# The virtual device: every idea we have, and how to use it

Scope, stated up front so nothing below drifts:

* **CPU only.** Every idea in this file is buildable and measurable on an
  ordinary CPU today, on this box, with no accelerator present.
* **One binary, many machines.** You may be on Meteor Lake, Zen 3, a Zen 5
  laptop, an ARM server, or a VM whose topology table lies (this one does).
  Nothing here may assume a lane count, an opcode, a core count or a cache size
  that has not been checked at runtime.
* **The device is general; Kraken is its first client, not its definition.**
  It must serve any engine that submits work — a tokeniser, a DSP-style filter,
  an image pipeline, someone's weekend project — with Kraken using exactly the
  same surface as everyone else. §9 is the structural consequence of this and it
  moves work out of Kraken's driver and into the device.
* **The backend that runs today is the host CPU**; a second is the virtual NPU,
  which is also CPU-backed. Two implementations of one contract is what proves
  the contract, and it is what lets another engine be bolted on later without
  touching the graph.
* **A unit is something that registers and reports.** Nothing in the contract
  names a particular kind of compute engine; a unit declares what it has
  verified and inherits the whole fallback machinery. Adding one later must be
  a plugin, not a refactor.

Everything is tagged with its evidence:

| tag | meaning |
|---|---|
| **M** | measured on this box (2-thread Xeon VM, 2.60 GHz, AVX-512 F/BW/DQ/VL/VNNI/IFMA, 48 KiB L1D, 1.25 MiB L2, 54 MiB L3) |
| **D** | derived from arithmetic that is written out here, so it can be checked |
| **R** | reported by a third party (device docs, patch stack, paper), not re-measured here |

---

## 1. What "a solid virtual device" actually means

A device is solid when five contracts hold and each one is *testable*. Every
architecture idea in §3 belongs to exactly one of them, which is how the list
stays a design instead of a collage.

| # | contract | question it answers | failure mode when it is missing |
|---|---|---|---|
| C1 | **Capability** | what can this device do, on *this* machine, right now? | an instruction that does not exist; a kernel silently narrower than claimed |
| C2 | **Data** | where do bytes live, in what form, who moves them? | the 6.1× staging blowup: 552 MiB read from a 90 MB model |
| C3 | **Execution** | how is work issued, batched, fused, tailed? | dispatch-bound kernels; a 1.74× loss to a ragged prompt |
| C4 | **Scheduling** | which core runs what, and how many threads is a core worth? | the pool spreads over the slow cluster; SMT is counted as capacity |
| C5 | **Observation** | how do we know any of this is true? | "it got faster" with no row in the trace that says where |

Kraken already obeys parts of all five. The rest of this file is: what exists,
what the outside world offers, and the smallest build that closes each gap.

---

## 2. Fallbacks are a taxonomy, not a behaviour

"Have fallbacks" is not a feature you add. It is seven distinct mechanisms, each
with its own **trigger**, **cost**, **report** and **test**. Getting this wrong
is how you end up with a device that is 30% slower on Zen 3 and nobody notices
for three months — which is exactly what happened to us (see the VNNI bug below).

| class | when it fires | what it costs | how it must be reported | how it is tested |
|---|---|---|---|---|
| **F1 ISA** | the host lacks a vector width or extension | proportional to lane ratio: 512→256 is ≈2× on the f32 chains, ≈1.4× on int8 **M** | the tier name in the banner and in the trace's `%peak` | `isa_parity`: byte-identical outputs across every tier **R** |
| **F2 Opcode** | the device has no op for this operation | host round trip; measured 83% device idle before the fix **M** | a counter in the run summary: `0 declined`, `q8_0 in place N` | a test that asserts the counter moves when it should |
| **F3 Format** | weights are in a format the unit cannot read | **6.1×** read amplification (552.1 MiB from 90 MB) **M** | bytes staged per token, and the packed MiB, in the summary | staging-byte budget test per format |
| **F4 Residency** | the working set does not fit the tier | COLD reads; 25.7% miss → warm-up fixes it **R** | tier hit rates per run (HOT/WARM/COLD) | the acceptance suite's existing counter assertions |
| **F5 Topology** | no cpufreq, no hybrid info, no NUMA | nothing — placement is simply not imposed | `2 cores, unpinned` vs `8 cores, 4P+8E, pinned` **M** | the banner; plus a forced `KRK_AFFINITY=1` arm |
| **F6 Scheduler** | SMT / hybrid / cgroup quota makes "N threads" ambiguous | up to 2× if you assume siblings are cores | thread count actually used, and the measured calibration | the probe in §6.6, run at startup |
| **F7 Numeric** | no bf16/f16/int8 op, or an accuracy trade is taken | f32 upcast: 2× bytes, same MACs | the accuracy witness number, in the PR, not the log | logit-distance gate against the exact arm |

Two rules make the taxonomy enforceable:

1. **A fallback is never silent.** If the code took a slower path, a counter
   moved and a number printed. Kraken already does this for `declined` and for
   `w8a8 deferred`; the other five classes need the same treatment.
2. **A fallback is never a surprise in the numbers.** Each tier declares its
   numeric relationship to the reference — *bit-exact* (ISA narrowing only),
   *tolerance-declared* (reordering), or *accuracy-witnessed* (quantisation).
   The `--prefill-hash` gate already encodes this; it needs a per-tier table.

### The bug that proves the point

`npuxl-0003` in this repo exists because `VNPU_FORCE_ISA=avx512` **silently
cleared the VNNI feature bit**, routing the int8 GEMM to the AVX2 kernel. Every
"AVX-512" comparison in that tree was measuring AVX2, and nothing in the output
said so. Measured: 20,304 MAC/s forced-avx512 against 82,395 with VNNI — a
**4.1×** error, invisible for two documents **R**.

That is C1 failing, and it is why §4 starts with a receipt rather than a kernel.

---

## 3. What the outside lineages give us

Each row: the idea in its own vocabulary → the contract it serves → what it is
worth *here* → the Kraken hook → status. Honest status, not aspiration.

### 3.1 DB2 / AIX — buffer management for data bigger than memory

The contribution is not instructions, it is 30 years of paging policy.

| idea | contract | worth here | Kraken hook | status |
|---|---|---|---|---|
| buffer pool + page **fixing** | C2 | stops re-deciding a hot page under pressure | `ExpertCache` pin budget, `KRK_EXPERT_PIN` | **have** |
| sequential detection + **list prefetch** | C3 | turns first-touch misses into a ranked read | `moe_predictor_` warm-up pass | **have** |
| **compressed buffer pools** — data stays compressed in RAM, decoded on use | C2 | this *is* "never dequant": the 6.1× staging disappears if the unit reads blocks | no opcode in either device tree **M** | **not built — top of the list** |
| large pages / page colouring | C2 | removes TLB misses on a big mmap | GGUF mapping alignment | not built; ≈0 on a 92 MB file **D** |
| WLM resource classes, `smtctl` | C4 | per-workload thread/SMT policy | `Pool::set_workers`, `KRK_THREADS` | partial (count only, no SMT policy) |
| SMIT-style bounded tuning knobs | C5 | every knob has a documented range and a default | env vars — but undocumented ranges | partial |

### 3.2 The HPC/vector lineages

| lineage | idea | contract | worth here | Kraken hook | status |
|---|---|---|---|---|---|
| **ARM SVE / RISC-V V** | **predication**: the tail never leaves the vector path | C3 | a ragged prompt costs **1.74×** (58 tok vs 64, tile-aligned) **M** | `gemm_*` tails, `attn_*` remainders | **not built — highest value** |
| **Cell SPE / Hopper TMA / `cp.async`** | async double-buffered staging | C3 | stage weight N+1 while computing N | `Offload` submit path is synchronous | partial (disk read overlaps; device stage does not) |
| **Blue Gene "torrent"** | work queues instead of barriers | C3 | 1360 tiny GEMM calls per 82-token run **M** | one batched expert GEMM + `tile_mask` | not built |
| **Fugaku A64FX** | L2 sector cache as a managed resource | C2 | nothing: no CAT/QoS in a VM, and streaming hints measured ≈0 **M** | — | **do not build** |
| **Xeon Phi MCDRAM** | explicit near/far placement | C2 | already the residency policy | HOT/WARM/COLD + `admit_hot` + decay | **have** |
| **Apple unified memory** | delete the copy step | C2 | same conclusion as DB2 compressed pools, from the other direction | — | not built (same item) |
| **IBM z16/Telum** | put the matmul where the data is | C3 | fused epilogue instead of a separate `silu_mul` pass: **1.0–1.4%** of the run **M** | `silu_mul` is its own row in the trace | not built, and last |
| **Tilera / Kalray / Epiphany** | expert-per-core, static, no scheduler | C4 | real at ≥8 cores; meaningless on 2 | `Pool` static partition | **not built (gated on cores)** |
| **UltraSPARC T / POWER SMT4–8** | SMT is a *policy*, not a capacity | C4 | see §6.6: the topology table here lies | none | not built |
| **Earth Simulator** | batching beats micro-tuning | C3 | 8-row kernel beat per-row by 15× **R** | `gemm_q6k_rN` | **have** |
| **Cray T3D SHMEM** | one-sided put/get | C3 | only pays with a real interconnect; device cluster demo: 0.86× wall clock **R** | — | **do not build** |

### 3.3 The instruction-set lineages: Itanium, P6, MMX, P54C

Four parts, four different lessons — and two of them contradict each other,
which is the reason to read them together.

#### Itanium / IA-64 — the most useful failure in the field

| idea | contract | worth here | Kraken / device hook | status |
|---|---|---|---|---|
| **Predication on every instruction** (64 predicates, no branches for small conditionals) | C3 | merges three separate mechanisms we have: ragged-M tails, MoE "only these tokens", attention masks | every op takes an **optional mask**; §5.1 becomes the general form, not an attention special case | **build with §5.1** |
| **Explicit bundles + stop bits** (the bundle declares what may dual-issue) | C3, C5 | batching becomes a *declared* grouping rather than a runtime guess — `push_batch_named` already is one | the tracer reports **bundle occupancy** (slots used / available), not just µs per batch | build into the trace |
| **Rotating registers / modulo scheduling** (the compiler pipelines the loop, hardware does not) | C3 | the K-loop over 144-byte blocks with double-buffered register staging | caller supplies prologue / steady / epilogue; one steady-state body | build for the quantised dot |
| **Register stack engine** (automatic spill/fill of register frames on call) | C4 | worker-local scratch that survives task boundaries — we already hit a null-deref from `thread_local` inside `par_for` **M** | a managed per-worker arena, owned by the pool | build (small) |
| Speculative load + NaT (deferred exception) | C1 | issue a load for a row that *might* be out of range, report later instead of branching | `support()` could carry "may-fault" | **do not build** — complexity above the payoff |

And the failure itself, which is the part worth more than the instruction set:
**the hardware executed a fixed schedule, so it could not hide anything the
compiler had not predicted** — a cache miss, a frequency drop, a thermal
throttle, a co-tenant on the same L3. IA-64 asked the compiler to know the
latency of the whole machine and it never does.

> **Rule taken from it:** a static schedule is authoritative only on a machine
> that behaves. A laptop does not. So bundles and tile plans are **hints**,
> retuned at run boundaries and cached — never frozen at build time.

#### Pentium Pro (P6) — the counter-argument that won

| idea | contract | worth here | hook | status |
|---|---|---|---|---|
| **Dynamic out-of-order + in-order retirement** | C3, C5 | the balance to Itanium: keep the *interface* in-order and simple, put adaptivity inside the unit, retune per shape | design rule; no code | **rule** |
| **Speculation with exact architectural state** | C1, C3 | speculative *work* (stage or compute the predicted expert) with a guarantee that results are unchanged — `moe_predictor_` says "routes unchanged" today as a comment | make it a checked contract, and count mispredictions | strengthen |
| **Micro-op translation + decoded-µop cache** | C3 | **plan cache**: compile a shape into a kernel plan once, key it by `(op, shape, tier)`, keep it | we cache *weights* today; we should cache *plans* | **build** |
| **Register renaming** (kills false dependencies) | C3 | the accumulator-over-`j` mistake in the first attention draft was exactly a false dependency **M** | the tracer should flag dependent-chain depth per kernel | build into trace |
| **Dual independent bus / many outstanding transactions** | C2 | per-thread memory-level parallelism: 9.9 GB/s at 1 thread vs 20.9 at 2 **M** — so prefetch *distance* is a property of the loop, not a constant | prefetch depth as a tunable, per tier | build |
| **MESI ownership (M/E/S/I)** | C2 | **a buffer ownership state machine**: each device buffer is exactly one of host-owned / device-owned / shared-clean / stale, and every transition is counted | replaces ad-hoc "did someone flush this?" reasoning; §7's staging accounting hangs off it | **build — structural** |

#### P55C (Pentium MMX) — narrow SIMD that paid anyway

| idea | contract | worth here | hook | status |
|---|---|---|---|---|
| 57 packed-integer ops on 64-bit registers | C1 | proof that a **narrow** integer SIMD path beats scalar when the data type fits — this is precisely `patches/npuxl-0002` (AVX2 int8) at 15.7× over scalar on a Zen 3-class part **R** | it means a Zen 3 laptop is not a second-class machine; §4's receipt must show `T2` as a first-rate tier | **have** |
| **Register-file aliasing with x87 → the `emms` stall** | C3 | the real lesson: **switching a buffer or a thread between two representations costs a stall.** Co-locate work by representation (all int8 stages together, all f32 together); make representation changes explicit, batched and *counted* | today we switch per GEMM — `QUANTIZE_I8` then `MATMUL_I8` as two commands **M**. That is the `emms`, and it should be printed as a switch count | **build — accounting first** |
| Widening on accumulation (int8×int8 → int16 → int32) | C2 | a block format must widen **exactly once** per block dot, never twice — already the rule in the q4/q6 kernels **M** | write it down as a format rule, not folklore | have |
| MMX shipped without a 32-bit path; SSE existed a year later | C1 | a narrow tier always omits something, which is what the receipt in §4 is for | — | rule |

#### P54C (Pentium classic) — superscalar with pairing rules

| idea | contract | worth here | hook | status |
|---|---|---|---|---|
| **Asymmetric U/V pipes** (the V pipe can only do a subset) | C5 | issue slots are asymmetric and partly wasted; a kernel should say whether it is **port-bound, latency-bound or bandwidth-bound** | I diagnosed load-port saturation in the attention kernel *by hand* **M** — the tracer should classify it | build (trace) |
| In-order dual issue, simple decoder | C3 | keep the contract in-order: it is the only thing a cheap compiler can target | design rule | rule |

### 3.4 Dead ends, recorded so they are not re-proposed

z/Arch decimal arithmetic (no target), MAJC VLIW (no compiler, no future),
Kryo/DSP (no access), PA-RISC MAX-2, 3DNow!, VIS, AltiVec (all superseded by
wider SIMD), streaming/NT stores (≈0 measured on both sides **M/R**),
M-blocking at −30% (laptop-specific; needs a >L3 working set **R**),
speculative decoding (every decoder tested lost throughput **R**), 2:4 sparsity
(no consumer), sector-cache quality-of-service (no OS control in a VM).

---

## 4. C1 — capability: a receipt, not a datasheet

**The rule: an ISA claim needs a kernel behind it, and the kernel needs a
known-answer test behind *that*.** Not `__builtin_cpu_supports`, which tells you
what the CPUID says, and not a datasheet, which tells you what the marketing
department says.

### The tiers

| tier | requires | who has it | what it buys |
|---|---|---|---|
| `T0 scalar` | x86-64 baseline | everything | the reference; always available |
| `T1 sse2` | SSE2 | every x86-64 | baseline vector, 4 f32 |
| `T2 avx2` | AVX2 + FMA | Haswell+, Zen 1+ | 8 f32 lanes; **the Zen 3 ceiling** |
| `T3 avx2-vnni` | AVX-VNNI (256-bit) | Alder Lake+, **Meteor Lake**, Zen 4/5 | int8 dot without AVX-512 |
| `T4 avx512` | F + BW + VL + DQ | Skylake-SP, Ice Lake, Zen 4/5 | 16 f32 lanes |
| `T5 avx512-vnni` | + VNNI | Cascade Lake+, Zen 4/5 | 64 int8 MAC/instr |
| `T6 avx512-bf16/fp16/ifma` | + BF16, FP16, IFMA | Cooper Lake+, SPR, Zen 5 | mixed precision, fast integer mul-add |
| `A1 neon` / `A2 dotprod` / `A3 sve` | ARMv8 / v8.2 / SVE | Apple (A1), Graviton, A64FX (A3) | the non-x86 ladder |

Two notes that matter for a laptop fleet:

* **Intel client disabled AVX-512 from Alder Lake on**, so a Meteor Lake part
  lands at `T3`, not `T4`. A binary built with `-march=native` on a Zen 5 desktop
  faults there. This is the whole argument for a multiversioned build.
* **Skylake-SP downclocks** under heavy AVX-512. So `T4` is not automatically
  faster than `T2` for narrow or memory-bound work; the receipt has to *measure*
  both on the actual part.

### How to build it

```cpp
enum class Tier { T0, T1, T2, T3, T4, T5, T6, A1, A2, A3 };

// One row per (op, tier). Declared, then verified at startup.
struct Receipt {
    Opcode op;
    Tier   tier;
    bool   verified;       // the known-answer test passed on THIS machine
    double gmacs;          // measured, not quoted
    const char *note;      // why it was refused, if it was
};
```

Startup, once, single-threaded, ~1 ms:

1. For each tier the CPUID claims, run the microkernel on a fixed input and
   compare against the `T0` result under the tier's declared numeric rule
   (bit-exact for ISA narrowing; tolerance for reordering).
2. Measure its rate with the same 16-independent-chains trick already used by
   `isa_peak_gmacs` — but **per tier, per op**, not one number for the box.
3. Publish the survivors as the dispatch table. A tier that claims support and
   fails its test is **refused by name**, exactly as `0003` now refuses an
   `avx512` request on an AVX2 host **R**.
4. Print it:

```
capability receipt (2 threads, x86-64)
  tier            verified   f32 GMAC/s   i8 GMAC/s   note
  T0 scalar          yes          1.9          3.8    reference
  T2 avx2+fma        yes         52.3            -    Zen 3 ceiling
  T3 avx2-vnni       yes            -        255.8    Meteor Lake lands here
  T4 avx512          yes        103.1            -    16 lanes
  T5 avx512-vnni     yes            -        651.0    +24% over T3 per lane-pair
  T6 bf16/ifma       no             -            -    IFMA present, bf16 absent
```

That table is the single most valuable artefact in this document, because it
replaces four assumptions we have been making in prose.

**Why this is first:** it costs a day, it touches no kernel, and every later
item's measurement is meaningless until we know which tier produced it.

---

## 5. C3 — execution: the three changes that are worth real time

### 5.1 Predicated tails (SVE's idea, done in AVX-512/AVX2)

Today every kernel is `for (; i + V <= n; i += V) vector; for (; i < n; i++)
scalar;`. That costs a scalar loop *per call*, and on shapes that are not lane
multiples the whole tail serialises. Measured: a 58-token prompt runs **1.74×**
slower than 64 tokens **M** — 9.5% more tokens, 74% more time. That is not
arithmetic, that is a tail.

The SVE answer is one loop with a mask:

```cpp
// AVX-512: the tail is the same instruction with fewer active lanes.
for (i64 i = 0; i < n; i += 16) {
    const __mmask16 m = i + 16 <= n ? (__mmask16)0xFFFF
                                    : (__mmask16)((1u << (n - i)) - 1);
    __m512 a = _mm512_maskz_loadu_ps(m, src + i);
    __m512 r = fn(a);                       // full-width, inactive lanes are 0
    _mm512_mask_storeu_ps(dst + i, m, r);
}
```

AVX2 has no per-lane masks for f32, so its version is a **static mask table**
indexed by `n - i`, applied with `_mm256_maskload_ps` / `_mm256_maskstore_ps`,
plus a `blend` for arithmetic that must not touch inactive lanes. Both are
loop-free at the tail.

Where it pays, in order: GEMM K and N remainders (every MoE expert GEMM has a
non-multiple N), `hd` = 80/96/256 attention rows, the quantised dot's K tail,
and any op with a ragged M. Where it does not: shapes that are already lane
multiples — do not restructure those.

**Test:** the existing tail tests at `n = 3, 16, 58` already exist in
`test_npu_backend`; extend the same shapes to every tier and to `hd` 80/96.

### 5.2 One dispatch for many experts (Blue Gene's work queue)

Measured: an 82-token dense run issues **1360 GEMMs**; the MoE path issues one
per (expert, token) on decode. Every one is a command, a flush decision, a
potential host round trip. Blue Gene's lesson is that a queue of independent
work items beats a barrier-per-item, and the device already has the mechanism:
`push_batch_named` fuses commands into one batch, and `tile_mask` +
`WILDCAT_CULL` can skip tiles belonging to experts no token routed to.

Two steps, in this order:

1. **Fuse** — the batched decode path already exists (`moe batch: N layer(s) ran
   the batched decode path`); the work is to make the *device* see one batched
   expert GEMM instead of K calls. Purely a scheduling change, no numerics.
2. **Cull** — pass a one-byte-per-tile mask for the experts that are actually
   routed. On a model with 8 experts used = 2, that is up to 4× fewer tiles in
   the ideal case. **Do not quote the ideal case as a result** — measure it with
   the real routing distribution, because MoE routing is not uniform.

### 5.3 Async double-buffered staging (Cell / TMA)

The device staging path is submit-then-run. Cell's split between the DMA engine
and the SPU is the reason to restructure it: while the unit executes tile `i`,
the host stages tile `i+1`, and neither waits. The host already does this for
disk→RAM on the expert path (read overlaps compute, `active-stage issued /
consumed / demand-wait` counters exist); the device path does not reuse it.

The honest caveat: on the CPU-as-device there is no separate DMA engine, so this
buys **prefetch into cache**, not overlap — measurably less. It becomes real
when the unit is something other than the cores you are already using. Build it
when a second unit exists; design for it now by keeping staging addressable as a
double buffer rather than a copy.

---

## 6. C4 — scheduling: measure the machine, do not read it

### 6.1 Placement (already built, and honest about it)

`par_pool.hpp` reads `cpufreq` per CPU, stable-sorts fastest-first, and gives the
caller the best core and worker *i* the (i+1)-th best; no `cpufreq` → no
ordering imposed. Measured **1.00×** on this uniform box — the no-op case, by
design **M**. On a hybrid part it is worth a lot; that claim waits for a hybrid
box, and the banner reports which branch it took.

### 6.2 Never read `thread_siblings_list`

This box reports `1 socket, 1 core, 2 threads`, `thread_siblings_list = 0-1`.
It is **two real cores**, and the topology table is synthetic **M**:

| probe | 1 thread | 2 threads | ratio |
|---|---:|---:|---:|
| f32 register-only chains | 48.5–52.3 GMAC/s | 100.6–103.1 | **2.00×** |
| int8 VNNI chains | 255.8–378.0 | 600.7–651.0 | **1.94×** |
| 768 MiB streamed read | 9.90 GB/s | 20.93 GB/s | **2.11×** |

SMT siblings share the vector ports and the uncore path; they cannot double
every one of those. So the package table lies, and a scheduler that obeyed it —
for example by packing threads onto "siblings" — would have halved throughput.

**Consequence:** the pool must calibrate, not introspect. One extra probe at
startup (the same two kernels, 1 thread vs 2, ~2 ms) gives the real
core-equivalence factor, and the thread default becomes
`threads = physical_estimate × factor` rather than `hardware_concurrency()`.

### 6.3 Expert-per-core, when there are cores

Tilera/Kalray's model — put each expert on a fixed core, never migrate — is the
right answer for a MoE on a wide machine and a meaningless one on 2 threads. The
precondition already holds: `Pool` partitions statically. The gate: enable only
at ≥8 participants, and only when the expert bytes per core fit that core's L2.
Measure as HOT-hit rate and bytes-per-token, not as tok/s alone.

---

## 7. C2 — data: the format question, stated exactly

One number drives this section. A 90 MB q8_0 model makes the device read
**552.1 MiB** of expanded f32 per step — 6.1× amplification **M**. Nothing about
tiling, units or threads fixes that; it is a *format* fallback (F3) being paid on
every token.

Three states, and we are stuck in the middle:

| state | weight traffic per token | accuracy cost | available today |
|---|---:|---|---|
| expand to f32 | 6.1× | none | yes (the current path) |
| quantise **both** sides (`MATMUL_I8`) | 1.0× int8 | activation quantisation; ranks drift on near-ties **M** | yes, opt-in `KRK_NPU_W8A8` |
| read blocks where they lie | 1.0× native | none | **no opcode in either device tree** **M** |

The third row is DB2's compressed buffer pool and Apple's no-copy, arriving at
the same design from opposite ends: **the unit decodes the storage format in
registers, and the format never changes shape in memory.** That is the single
highest-value item on the C2 list, it needs one opcode and one kernel, and the
host side already proves it works (the host reads the 144-byte blocks in place).

Until it exists, the honest move is to make F3 **visible and budgeted**: print
bytes staged per format per run, and fail a test if a format silently starts
expanding. Today that number exists (`552.1 MiB`, `336.0 MiB staged`) but nothing
asserts on its trend.

---

## 8. The contract, as code

Small enough to read, and it is the thing another implementation (or another
engine) has to satisfy. It is written against behaviour we already have — no new
capability is being invented here.

```cpp
// ---- what a unit is -------------------------------------------------------
struct UnitCaps {
    Tier   tier;            // the highest VERIFIED tier (see §4)
    u32    lanes_f32;       // 16 for avx512, 8 for avx2, 4 for sse2
    u32    lanes_i8;        // 64 for vnni, else 0
    bool   f16, bf16, ifma; // verified, not CPUID-claimed
    i64    l1d, l2, l3;     // bytes, 0 = unknown
    i64    threads;         // calibrated equivalents, not hardware_concurrency
    double gmacs_f32, gmacs_i8;   // measured on this machine
};

// ---- buffer ownership, so "did someone flush this" is never a guess ------
enum class Owner { Host, Device, Shared, Stale };
struct Buffer {
    Owner owner;      // exactly one state at every point (MESI, §3.3 P6)
    u64   bytes;
    bool  quantised;  // representation, so a switch is counted (P55C §3.3)
};

// ---- what happens to one operation ---------------------------------------
enum class Level { Native, Emulated, Refused };

struct Support {
    Level  level;
    Tier   tier;            // the tier that will actually run it
    u64    extra_bytes;     // F3: staging amplification, 0 if none
    double est_scale;       // estimated cost vs the fastest verified path
    const char *why;        // never empty when level != Native
};

// ---- one op, with the mask as a first-class argument ----------------------
// Itanium's predication, applied at the API instead of the ISA: a ragged tail,
// an inactive token, and a masked attention row are the same mechanism, so
// there is one code path to test instead of three (§3.3, §5.1).
struct OpDesc {
    Opcode op;
    const void *a; const void *b; void *c;
    i64 m, k, n;
    const u8 *mask;   // optional, 1 bit or 1 byte per row; null = all active
    Tier  want;       // Tier::T0 means "fastest verified"
};

// ---- the device ----------------------------------------------------------
class Unit {
public:
    virtual UnitCaps caps() const = 0;
    // Ask BEFORE issuing. A refused op is a graph error, not a silent host call.
    virtual Support  support(const OpDesc &) const = 0;
    virtual void     exec(const OpDesc &, void **args) = 0;
    virtual std::string fallback_log() const = 0;   // F1..F7 counters
    virtual void     trace_dump(const char *path) const = 0;
};
```

Four rules that make it solid rather than decorative:

1. **`support()` is answerable without running the op.** It is what lets a
   planner choose M-blocks, chunk sizes and expert order before touching data,
   instead of discovering a fallback mid-graph.
2. **`Refused` exists.** A device may say no. Today every op has some path, but
   the API must not encode that assumption, or the first unit that cannot do
   something will silently get a host call it did not ask for.
3. **`fallback_log()` is part of the contract, not a debug print.** If the answer
   to "did this run take a slow path" is not a counter, the run is not
   reproducible in the sense that matters.
4. **Determinism is declared per tier, not asserted globally.** Same tier +
   same shape + same thread count ⇒ same bytes. A different tier may differ, and
   must publish its `--prefill-hash` alongside the reference's.

---

## 9. Who owns the policy: the device, not the client

This is the structural consequence of the device being general, and it changes
where code lives.

The device is **already broader than its only client**: it exposes 17 opcodes
including MTP head, attention, softmax, GELU, fused epilogues and a cluster API,
and Kraken calls six of them **M**. An API larger than its user means the client
is not the definition — it is the first consumer.

Three pieces of policy currently live in Kraken's `src/npu/npu_offload.cpp` and
each one is a bug we already paid for. Every future client would re-implement
them and re-discover the same bugs:

| policy | bug it fixed, and what it cost when it was absent |
|---|---|
| staging budget derived from real RAM | flat 1024 MiB default declined 393 of 1087 GEMMs; **83% device idle** **M** |
| pack-on-reuse instead of pack-on-first-touch | **+1.0 s of pure loss** on a prefill that never reused a weight **M** |
| the one-time pack must not sit in the measured step | produced a *wrong verdict on the whole backend*: "int8 loses" instead of 2.5× **M** |

So the move-down plan is not tidiness, it is bug prevention:

```
   client (Kraken, or anything else)      <- graph, tensors, routing
   -----------------------------------
   unit shim: policy, plans, accounting   <- receipt, budgets, plan cache,
                                             ownership states, fallback log
   -----------------------------------
   engine (vnpu / host CPU)               <- kernels, dispatch, ISA tiers
```

The shim owns: **capability receipt** (§4), **residency budget**, **plan cache**
(§3.3 P6), **buffer ownership states** (§3.3 P6/MESI), **format accounting**
(§7) and the **fallback log** (§2). Kraken then consumes those like any other
client, and loses the code that only it has.

Two tests that keep the contract honest, both cheap:

1. **Second client.** Build one of the device's own examples/benchmarks through
   the shim with no Kraken code in the link. If it needs anything from
   `src/npu/npu_offload.cpp`, the contract is still client-shaped.
2. **Portability of the surface.** No op in the contract may require a
   Kraken-specific layout. A block format enters through `support()` as
   `Native` / `Emulated` / `Refused`, so a unit that cannot read a format **says
   so** instead of expanding it silently — the defect in §7.

## 10. Build order

CPU only, each item independently mergeable, each with a number attached. This
is the whole plan; anything not here is not planned.

| # | item | scope | contract | effort | how it is judged |
|---|---|---|---|---|---|
| **1** | **Capability receipt** (§4) | shim | C1 | S | the printed table matches reality on ≥2 machines; a raised tier is refused by name when the kernel behind it is missing |
| **2** | **Masks as an op argument** — one mechanism for ragged M, MoE-active-tokens and attention masks (§3.3 Itanium + §5.1) | shim + kernels | C3 | M | 58-token prompt closes most of its 1.74× gap **M**; `hd` 80/96 improve; bit-identical on existing fixtures |
| **3** | **Fallback log** — F1…F7 counters in the run summary (§2) | shim | C1–C5 | S | every arm of an A/B shows which paths it took; no counter left at zero-but-claimed |
| **4** | **Calibrated thread count** (§6.2) | shim | C4 | S | the probe reports 2 on this box; the default follows the probe, not `hardware_concurrency()` |
| **5** | **Buffer ownership states + format accounting** (MESI, §3.3 P6 / §7) | shim | C2 | M | every buffer has exactly one owner at every point; bytes staged per format asserted on trend instead of drifting 6× **M** |
| **6** | **Plan cache** keyed by `(op, shape, tier)` (§3.3 P6) | shim | C3 | M | first call compiles, later calls hit; hit rate printed per run |
| **7** | **Second client** through the shim — no Kraken code in the link (§9) | — | C1–C5 | S | it builds and runs; if it needs anything from `npu_offload.cpp`, the contract is not done |
| **8** | **Batched MoE dispatch** (§5.2 step 1), then **cull** (step 2) | engine | C3 | M/L | device commands per token falls; tiles culled measured against the real routing distribution |
| **9** | **Block-format opcode** (§7) | engine | C2 | L | 6.1× staging → 1.0×, with an accuracy witness of zero change |
| **10** | **Trace depth** — bundle occupancy, dependent-chain depth, port/latency/bandwidth verdict (Itanium, P6, P54C) | shim | C5 | M | the attention kernel's load-port saturation is reported by the tool instead of by hand |
| **11** | **Fused epilogue** (`bias + SiLU`) (§3.2 z16) | engine | C3 | M | `silu_mul` row shrinks; it is 1.0–1.4% of the run, so last by construction **M** |
| **12** | **Expert-per-core** (§6.3) | shim | C4 | M | gated at ≥8 participants; judged on HOT-hit rate and bytes/token |
| — | **Async double-buffer** (§5.3) | engine | C3 | L | deferred until a second unit exists; design for it now |
| — | **Prefetch depth as a tunable** (§3.3 P6) | kernels | C2 | S | single-thread streamed read is 9.9 GB/s; the tunable must move it **M** |
| — | **A64FX sector cache, SHMEM cluster, NT stores, IA-64 NaT** | — | — | — | not built, for the reasons in §3.4 |

Items 1, 3, 4 and 7 are a day each, and items 1–7 are what turn "a device" into
"a device someone else can build against": they live in the shim, so the second
client gets them for free. Item 2 is the only one in the top half that will show
up in tok/s this week. The engine items (8–11) sit behind the shim on purpose —
every one of them is easier to judge once the accounting exists.

---

## 11. The gate every item must pass

Non-negotiable, and mostly already in place:

1. **Correctness**: the full suite, plus the tier's declared numeric rule
   (bit-exact or tolerance) against the `T0` reference on the fixed fixtures.
2. **Coherence**: real text out of a real model, not just a hash.
3. **Determinism**: `--prefill-hash` invariant across `--chunk` and thread count
   at a fixed tier; the per-tier table updated when a tier changes it.
4. **Honest numbers**: interleaved arms, medians of ≥3, **ratios reported,
   absolute GMAC/s marked indicative** — this box is shared and has no turbo
   control.
5. **No silent fallback**: if the change can take a slower path, the counter is
   in the summary and a test asserts it moves.
6. **Every claim traceable**: a row in a trace, or arithmetic written down, or a
   citation. No third option.

---

## 12. Caveats that belong in the file, not the footnotes

* The fixture model has random weights. It is a **speed** fixture, not an
  accuracy one; accuracy claims need a real checkpoint.
* This box's numbers move ±10% between runs and its topology table is wrong.
  Treat every ratio here as a direction and re-measure on the target laptop
  before quoting it anywhere else.
* Half of what is called "unbuilt" above is unbuilt *in Kraken's driver*, not in
  the world: the device tree already ships units, autotune, cull masks and
  cluster, and `patches/npuxl-0002` already contains a 15× AVX2 int8 kernel that
  exists in exactly one of the two forks. Consolidation before construction.
* Nothing in this file needs an accelerator. Every item is measurable with
  `--cpu` on the machine in front of you, which is the only way a fallback story
  can be tested at all.
