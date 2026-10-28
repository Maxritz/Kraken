# Kraken — technical readme

This is the document to read before the others. It states what the engine does,
how a token actually flows through it, which model families run and which do
not, what is measured, and what is not. Where a claim is a measurement it says
so and points at the record; where a family is unsupported it says which tensors
or kernels are missing rather than "not supported".

Detailed records live beside this file and are linked per section. The short
version of the honest position: **the CPU reference path is fast and exact, the
NPU path is measured and opt-in, and four of the newest model families are
partly or wholly unimplemented, with the reason named.**

---

## 1. What this engine is

Kraken is a GGUF inference engine with three backends and one shared
implementation of the math:

| backend | state | role |
|---|---|---|
| **CPU** (`src/backend_cpu.cpp`) | always available | the **scalar reference**. Every other backend is validated against it. Being exact matters more than being fast — but see §5, it is also the only backend this box can run, and it is now 4-6x faster than it was |
| **HIP / ROCm** (`src/hip/`) | needs a gfx1031/gfx1201 device | the throughput path: WMMA tiles, DP4A int8, fused epilogues, tiered expert cache |
| **Virtual NPU** (`src/npu/`) | opt-in at build time (`-DKRK_ENABLE_NPU=ON`) | rides the **CPU** path, not a GPU; a software "device" over AVX2/VVVNI. Rode the CPU path by design — see §7 |

The design constraint throughout: **a path that cannot compute something
correctly refuses by name.** There is no silent fallback that produces
plausible-looking wrong output, and no "supported" flag that is not backed by a
test that runs the graph.

## 2. How a token flows

```
GGUF ──(mmap, no copy)──► Model::load ──► LayerWeights[L] (dense tensors)
                                      └─► ExpertCache (routed banks, tiered)

prefill:  tokens ─► forward(chunk) ─► per layer:
                        rmsnorm ─► attn(QKV proj ─► rope ─► scores/softmax ─► out proj)
                        ─► residual ─► rmsnorm ─► FFN
                                              ├─ dense:   gate/up ─► silu ─► down
                                              └─ MoE:     router ─► top-k ids+weights
                                                          ─► group rows by expert
                                                          ─► per expert: acquire, gather rows,
                                                             gate/up ─► silu ─► down, scatter-add
                        ─► final norm ─► head (n_vocab) ─► sample
decode:   the same graph at rows = 1
```

Two things about that diagram are load-bearing and easy to get wrong:

- **The routed experts are grouped by expert, not iterated per token.** For a
  chunk of `n` tokens the layer builds, per expert, the list of rows that
  selected it, gathers those rows, and runs one GEMM per expert over the whole
  group. This was already true; it is why MoE prefill gained 6x from a dot
  product fix instead of needing a restructuring.
- **Rows are independent.** Every output element is one dot product, so a
  change of loop order or thread partition does not change the answer. That is
  what lets the CPU backend be optimised without invalidating a single golden
  hash (§5).

## 3. Expert residency: HOT / WARM / COLD

Routed experts are the largest thing in a modern checkpoint and the only part
that is accessed sparsely, so they get their own cache:

| tier | lives in | filled by |
|---|---|---|
| **HOT** | resident device memory (or host RAM on the CPU backend) | promotion from WARM/COLD, bounded by `--expert-cache-mb` / `--expert-cache-slots` |
| **WARM** | host RAM, optionally pinned | read from the mapping; can be prefetched |
| **COLD** | the GGUF file itself, via the mmap | read on demand, in place when the slot allows |

"Keep only the active expert in RAM and leave the rest on disk" is the COLD
path, and it is what the CPU backend does by construction: the weights are
`mmap`ed, the expert cache holds a budgeted resident set, and everything else
is a page fault away. Measured on a 2.3 GiB / 32-expert model on a 3.8 GiB box
with a 512 MiB budget — see [BENCH-CPU-DOT.md](BENCH-CPU-DOT.md) §6.

**The asymmetry you should expect, and why it is not a bug:** decode reads the
same `top-k` experts every step (12.5% of the pool for 4-of-32), so a warm cache
serves it; prefill's per-layer **union** of selected experts saturates very
fast (35% of the pool at 9 tokens, 43% at 144). Decode is a working-set problem,
prefill is a union problem, and `--expert-cache-mb` is one knob for both. Size
it for prefill to avoid churn; size it for decode to save memory and accept
re-reads during prefill.

### The prefill defect, and the number behind it

Prefill was not batching. `gemm()` called `vec_dot()` once per (row, output)
element and `vec_dot()` decodes the weight as it goes, so every weight row was
decoded once per ROW of the batch; weight traffic was `rows x` the matrix. The
tell was that cost per prefilled token was FLAT -- 34.1, 29.1, 27.8, 28.3 ms at
58, 149, 331 and 695 tokens -- where a GEMM must fall steeply.

`vec_dot_tiled()` decodes each block once into registers and FMAs it against a
tile of eight rows: hand-written for q8_0/q4_0, an F16C tile for f16/f32, and a
generic tile for every other block format (Q4_K, Q5_K, Q6_K, iq-quants). Same
flags, parent commit against this one: **2.28-3.35x** on the q8_0 MoE and
**3.83-4.29x** on a 1 GiB f16 dense file.

Results are bit-identical, not close: each kernel performs the same operations
in the same order as the path it replaces. `test_gemm_batched_bit_identical`
(100 checks: 5 formats x 4 arms x 5 tile widths, `memcmp` against `vec_dot()`)
and an unchanged `--prefill-hash` end to end on both a q8_0 MoE and an f16 dense
file. The first attempt -- widening each weight row to f32 and reusing it -- was
1.9x SLOWER and was reverted; it is written up in
[MOE-RESIDENCY.md §7](MOE-RESIDENCY.md) because it is the obvious first idea.

**This also changed the residency conclusion.** Prefill used to look immune to
expert residency (13.68 s against 13.95 s). It looked that way because compute
was dominating for the wrong reason: with the GEMM fixed, the expert read is 61%
of a streamed prefill at a 64 MiB budget, so `--expert-warm-mb` and the device
budget now matter for prefill as well. The ladder below is the pre-fix record
and is kept for its decode column, which is still current.

### The measured ladder (2026-10-27)

Full tables in [MOE-RESIDENCY.md](MOE-RESIDENCY.md); the shape of the result, on
a 1.2 GiB q8_0 MoE (20 layers, 32 experts, top-4) on 2 vCPU:

| | all resident | 64 MiB | 16 MiB |
|---|---:|---:|---:|
| prefill, exact | 13.95 s | 13.68 s | 14.02 s |
| prefill, AVX2 dot | 3.94 s | 3.82 s | 3.95 s |
| decode, AVX2 dot | 41.1 ms | 73.6 ms | 74.6 ms |
| read | 369 MiB | 2519 MiB | 2519 MiB |

**Prefill is flat across residency and decode is not.** The expert path's read is
533 ms of a 3858 ms prefill at 64 MiB (14%): the gathers are real but they are
not what prefill is spending its time on. Decode pays 1.79× for the same
residency because it has one row of arithmetic to hide 1405 expert loads behind.
Two corollaries that change how the knobs should be set: `--chunk` below ~128
multiplies the gathers (chunk 8 reads 2.8× what chunk 128 does for the same
prompt), and `--expert-warm-mb` is worth more than the device budget (a 512 MiB
WARM tier at a 64 MiB budget cuts the read to 246 MiB, 0.10×).

On the virtual NPU the asymmetry becomes a cliff rather than a slope: with
`KRK_NPU_W8A8=1` decode is 32.8 ms/token resident against 416–719 ms/token when
experts stream, because the int8 pack is memoized per weight address and a
streaming slot changes address on every visit. The mechanism, the rejected fix
and the numbers are in [MOE-RESIDENCY.md §5](MOE-RESIDENCY.md).

Prefetch and prediction, all opt-in and all measured before being believed:

- `--expert-warm-prefetch` fills WARM at load.
- `KRK_MOE_PREDICT=1` runs the route-hint predictor (`src/moe_predictor.cpp`,
  `include/krk/moe_predictor.hpp`). It never alters a selection; it only
  submits pinned-WARM hints. Historical next-layer prefetch was measured
  **9% slower** on REAP-320, which is why nothing here is on by default.
- `--hybrid-experts` / `--hybrid-frac` move the overflow of the device tier to
  host compute at a fixed cache size, which is how the two arms are swept
  against each other. With no device present there is nothing to overflow.

## 4. Model families

The table is `src/arch.cpp`'s `kTable`, and `kraken-inspect file.gguf` prints
the verdict for a specific file — including, for a file that cannot run, which
tensor classes it could not take. A family verdict is about the *architecture*;
individual files are still checked at load, because several families ship in
more than one layout.

### Runs today

| family | arch string | shape | notes |
|---|---|---|---|
| Llama / Mistral-style | `llama` | dense | the baseline schema |
| Qwen2 / Qwen3 | `qwen2`, `qwen3` | dense | QK-norm, per-head rope conventions |
| Qwen3-MoE | `qwen3moe` | moe | the MoE schema this repo benchmarks against |
| OLMoE | `olmoe` | moe | wide per-head QK-norm |
| Qwen3.5 / 3.5-MoE | `qwen35`, `qwen35moe` | recurrent / recurrent-moe | Gated DeltaNet hybrid |
| LFM2-MoE | `lfm2moe` | recurrent-moe | |
| DeepSeek-V2 / V2-Lite / Coder-V2-Lite | `deepseek2` | moe | MLA, lite layout |
| Nemotron-H / -MoE | `nemotron_h`, `nemotron_h_moe` | recurrent(-moe) | Mamba-2 SSD blocks |
| Laguna | `laguna` | moe | sliding-window pattern |
| gpt-oss | `gpt-oss` | moe | partial: see the gap list below |
| Qwen3.8 Flash-Next class (`qwen4exp`) | `qwen4exp` | recurrent-moe | HC / PLE / QSA / GDN implemented; a file is refused at load if it carries a piece that is not, and whole-graph coherence against a real checkpoint is **not** established |
| Draft/MTP heads | `dflash`, `qwen35-dflash-draft`, `dflash-draft` | dense | speculative heads extracted by `tools/mtp_extract.py` |

### The four families asked about, with the actual blocker

These are real models (August–October 2026) and none of them can be downloaded
into this sandbox: HuggingFace is not reachable from it, and the smallest is
122 B parameters. Nothing below claims to have *run* one of them.

| model | arch string | state | what stands in the way |
|---|---|---|---|
| **DeepSeek-V4.1-Flash-1017** (552 B backbone / 763 B weights, 40 layers, hidden 5120, 384 routed experts top-6 + 1 shared) | `deepseek41` | **recognised, Partial** — added this pass | The graph is `deepseek4`'s and maps onto it: shared-latent MLA, the grouped low-rank output projection and the `tid2eid` hash route run (`test_dsv4_whole_graph` pins the dense-core case). Missing: the **CSA/HCA compressors** (`attn_compressor_*`), the **sparse indexer**, `attn_sinks`, **Engram** conditional memory (196 B params, a token-hash lookup), and the 4-wide **hyper-connections**. A real V4.1 checkpoint carries a non-zero `compress_ratios` and will be refused at `Engine::init` **by name** rather than mis-loaded |
| **Qwen3.8-Flash-Next** (125 B MoE + 51 B n-gram head, 512 routed experts top-10 + 1 shared, 48 layers, Gated DeltaNet + QSA hybrid) | `qwen4exp` | **implemented, gated** | Every component has a kernel (HC, PLE, QSA, GDN, experts) and component tests pass. What is missing is a **whole-graph coherence gate** against an independently verified reference on the real GGUF: one recorded REAP-288 prompt matched 24/24 and another coding prompt historically diverged. Serving fluent wrong output is the failure this engine exists to avoid, so the family stays gated until that gate is built |
| **MiniMax-M3.1-Flash** (428 B MoE, 23 B active, 1M context, MSA sparse attention throughout, native multimodal, DSpark draft head) | `minimax-m3` (registered 2026-10-27) | **not supported, refused by name** | No GGUF tensor schema has been validated for it, and the graph needs pieces this engine does not have: **MSA** (a MinMax-specific sparse attention), multimodal input towers (text+image+video), and DSpark. Mapping a schema from the safetensors naming alone would be a guess, and a guessed mapping is what produces silently wrong output |
| **Ornith-1.5** (122 B, released 2026-10-20) | not mapped | **not supported** | Same position: no schema validated, no architecture reference to implement against |

**What would move these:** a `kraken-inspect` run on a real file (it prints the
tensor set and the gap map), then the missing kernels in the order that
`ARCHITECTURE.md` describes. For V4.1 the compressors and the indexer are the
whole task — the surrounding graph is already here.

## 5. Performance: what changed, and the number behind it

The headline of this pass is one finding: **the CPU dot product was calling an
out-of-line `fp16_to_fp32` once per weight element**, which put the reference
backend at 0.8 GMAC/s. Replacing it with a four-lane F16C + FMA form that keeps
the *same* four accumulators in the *same* order made it bit-identical and 4-6x
faster.

| measurement | before | after | |
|---|---:|---:|---:|
| dense 1 GiB prefill | 1.7 tok/s | 10.0 tok/s | 5.9x |
| MoE 2 GiB prefill (T=32) | 4.3 tok/s | 22.9 tok/s | 5.3x |
| MoE 2 GiB prefill (T=128) | 4.7 tok/s | 28.0 tok/s | 6.0x |
| MoE 2 GiB decode | ~260 ms/step | 58.9 ms/step | 4.3x |

Exactness witness: **4576/4576 checks unchanged** across the change (4621/4621
with the opt-in block-format test added), golden hashes included. Full record, method, block-format limits and remaining
headroom: [BENCH-CPU-DOT.md](BENCH-CPU-DOT.md).

Known limits, stated because they are the next person's starting point:

- The **block formats (q8_0/q4_0) do not get this win by default** and cannot,
  while staying exact: their reduction is a single dependent chain, and q4_0 is
  *half the bytes of q8_0 and 30% slower* for that reason. The opt-in is
  `KRK_CPU_DOT_FAST=1` — four accumulators instead of one, **3.07x / 2.71x** on
  a q8_0 MoE (prefill / decode) and 1.98x / 2.02x on q4_0, at a cost of up to
  1.4e-07 of the term magnitude. While it is on, the CPU arm is **no longer a
  bit-exact oracle**, so turn it off before comparing against `--device` or
  `--npu`. Witness and numbers: `tools/dot_fast_witness.py`,
  [BENCH-CPU-DOT.md](BENCH-CPU-DOT.md) §5.
- Dense weights are re-read once per row of `x`. Inverting the loop is not
  enough (`vec_dot` re-widens each call); it needs a widened row reused across
  a row block, with output blocking to keep the activation block in L2.
- `--expert-cache-mb` serves a decode working set and a prefill union with one
  number (§3).

Dense models split across RAM and VRAM through the **tiering** machinery
(`--vram-cap-mb`, `--kv-hot-mb`, KV HOT/WARM/COLD in `src/kv_tier.cpp`) plus the
HIP backend's weight residency. There is **no dense layer-streaming path** that
pages dense weights from disk on demand — the expert cache is the only
streaming tier, and that is a genuine gap for dense models larger than RAM.

### The prefill GEMM (2026-10-27)

`gemm()` on the CPU backend was not batching: it called `vec_dot()` per
`(row, output)` element, and `vec_dot()` decodes the weight row as it goes, so
every weight row was decoded once per row of the batch. `vec_dot_tiled()` decodes
each block once into registers and reuses it across eight rows; 2.28-3.35x on
the q8_0 MoE and 3.83-4.29x on f16 dense, bit-identical, decode untouched. Full
write-up and the rejected first attempt: [MOE-RESIDENCY.md §7](MOE-RESIDENCY.md).

## 6. The NPU path

Build with `-DKRK_ENABLE_NPU=ON -DKRK_VNPU_ROOT=<npuxl tree>`. It rides the CPU
path (there is no NPU+GPU wiring, deliberately: no untested HIP+NPU composition
ships). It is separable — the whole backend is `src/npu/` plus two call sites.

Two optimisations, both opt-in and both measured:

| switch | what it does | measured |
|---|---|---|
| `KRK_NPU_W8A8=1` | int8 weight staging for the device | 1.12-1.14x prefill on small stacks, up to 3.6x on medium ones |
| `KRK_NPU_ATTN=1` | runs attention on the device when the call is expressible | +10.3% prefill at 257 tokens, +25.6% at 513, decode unchanged |

And on the CPU side, independent of the NPU:

| switch | what it does | measured |
|---|---|---|
| `KRK_CPU_DOT_FAST=1` | four accumulators in the block-format dot (q4_0/q8_0/q5/q6) | 3.07x prefill / 2.71x decode on q8_0, ~2x on q4_0; up to 1.4e-07 of the term magnitude different, so the CPU arm stops being the bit-exact oracle while it is set |

### The Zen 3 arm (`VNPU_FORCE_ISA=avx2`)

Most machines running a quantised model have no AVX-512. Zen 3 does not; its
per-core ceiling is two 256-bit FMA3 pipes, and AVX-512 only arrives with Zen 4
(where it is double-pumped and buys roughly the same throughput in fewer
instructions). So the 256-bit kernel is the *fast path* on that class of
hardware, not a fallback -- which is why it was worth fixing rather than
inheriting.

The engine's AVX2 int8 GEMM was widening 8 int8 to int32, multiplying with
`mullo_epi32`, and gathering B one byte at a time: 8 MACs per iteration, and
measured at **4.89 GFLOP/s against scalar's 3.80**. On the packed
`[n_block][k4][16][4]` layout, `maddubs` + `madd_epi16` does 32 MACs in two
instructions. The `+128` offset the VNNI path relies on cannot be reused --
`maddubs` produces int16 and `255*127*2` overflows it -- so the kernel uses
`|a| * (b*sign(a))` instead, which peaks at 32512 and needs no correction term.
`quantize_rows`, 25.8% of device busy in a W8A8 trace, was scalar and is now
vectorised to be bit-identical (half-away-from-zero is emulated, because
`vcvtps2dq` rounds half to even).

| one core, 768x768 expert | scalar | old avx2 | new avx2 | gain |
|---|---:|---:|---:|---:|
| 1 row (decode) | 0.358 ms | 0.326 ms | **0.075 ms** | **4.3x** |
| 32 rows | 9.456 ms | 8.321 ms | **0.548 ms** | **15.2x** |
| 128 rows | 39.735 ms | 30.869 ms | **1.960 ms** | **15.7x** |

End to end on the 2 GiB q8_0 MoE fixture with `KRK_NPU_W8A8=1`, forcing the
Zen 3 arm (best of 3):

| arm | prefill before | prefill after | decode before | after |
|---|---:|---:|---:|---:|
| `VNPU_FORCE_ISA=avx2` | 13271.4 ms | **3446.9 ms (3.85x)** | 233.4 ms | **42.8 ms (5.5x)** |
| default (AVX-512) | 2974.5 ms | 2940.5 ms | 29.6 ms | 24.9 ms |

The prefill hash is unchanged (`fc8a5598db7ec9cd` on both arms before and
after), so the 3.85x costs no accuracy. The baseline AVX2 number was within 1%
of scalar: before this, a Zen 3 user turning on int8 weights got scalar speed
and no reason to know it. The AVX-512 VNNI path was not touched and did not
regress.

Without `KRK_NPU_W8A8=1` the improvement is **not** distinguishable from noise
(run-to-run spread is 6-8% on this box, and the difference is inside it), which
is the expected result: that arm's device work is `MATMUL_F32`.

#### What the engine's own benchmark flag was hiding

The AVX-512 path is faster than the AVX2 one by a wide margin, and until
recently it was not possible to see that through the engine's test flag.
`VNPU_FORCE_ISA=avx512` cleared the VNNI feature bit, so forcing "avx512" fell
straight through to the AVX2 int8 kernel: **20,304 MAC/s against 82,395**, 4.1x,
silently. Since that flag is the documented way to make cross-machine
comparisons, it meant anyone comparing arms was measuring AVX2 twice.

That is recorded here because it produced a wrong claim in this document's own
history: an earlier revision said the 256-bit kernel "outruns the AVX-512 VNNI
path". It does not. Single-threaded and kernel-only, VNNI is the faster path by
about 4x at 128x768x768; it always was. The comparison was an artefact of the
flag, not a property of the kernels.

Both are fixed in `patches/npuxl-0003`: `avx512` now keeps VNNI (with
`avx512-novnni` as the explicit opt-out), forcing an ISA the host cannot run is
refused by name instead of faulting, and the VNNI kernel's scale multiplication
was reordered to match the scalar oracle, which made it bit-identical on every
shape for the first time.

A second lesson worth keeping: the public benchmark could not answer "is the
kernel slow or is everything around it slow". `tests/bench_kernel.cpp` calls the
microkernel directly, one tile, one worker -- that separation is what turned a
plausible theory about the epilogue into a measurement, and the measurement
said the theory was wrong.

Verification is exact rather than tolerance-based. `tests/isa_parity.cpp` in the
engine hashes the raw output bytes per op, so the three ISA arms can be
compared directly: the int8 GEMM and the quantiser are **bit-identical** across
scalar, AVX2 and AVX-512 on every shape tested, including K not a multiple of 4
and N not a multiple of 16. The vectorised f32 reductions (RMSNorm, softmax,
f32 GEMM) still differ across ISAs in the last bits -- that predates this work
and is worth knowing before pinning results across a mixed fleet.

Attention is **off by default because of chunk invariance, not accuracy**: the
device kernel is invariant on its own, but the host/device *switch* is a
per-call gate, so two runs differing only in `--chunk` can land on different
paths and hash differently. Default runs stay byte-identical to before.

## 7. Testing and determinism

- `kraken-tests` — 4617 checks on the CPU build (4608 + NPU checks with
  `-DKRK_ENABLE_NPU=ON`). Golden hashes of prefill and decode output are part
  of it, which is what makes "bit-identical" a testable claim rather than an
  adjective.
- `--prefill-hash` — hashes a prompt's prefill; two runs differing only in
  `--chunk` must agree. This is the invariant that keeps a prompt and its
  continuation on the same arithmetic, and the reason several optimisations in
  this tree are opt-in.
- `tools/ref_llama.py --check` — an independent NumPy reference for the dense
  graph (5/5 match, 15/15 across sizes).

### An unlisted architecture is refused, and the verdict says so

`assess_model` and the loader used to disagree about files whose
`general.architecture` is not in the table. The loader refused them
(`GGUF metadata is missing core geometry`) because `<arch>.block_count` is read
from the arch prefix and an unlisted prefix has no such key. The verdict, having
no rule that looked at it, called them `runnable`. The tool that a user reaches
for first was the one giving the wrong answer.

The rule now asks the file rather than the name: an unlisted arch with no
`<arch>.block_count` is a blocker naming the architecture, the missing key, and
the architectures that *are* known; an unlisted label whose file still carries
usable geometry stays runnable with a note, because that file can load. Pinned by
`test_load_verdict`, with the control file declaring geometry so the test is
about the shape rule rather than about this one.

## 8. Building

```bash
# CPU only (always works; this is the reference)
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DKRK_ENABLE_HIP=OFF
ninja -C build-cpu            # build everything: kraken-tests needs kraken-server

# with the virtual NPU
cmake -S . -B build-npu -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DKRK_ENABLE_HIP=OFF -DKRK_ENABLE_NPU=ON \
      -DKRK_VNPU_ROOT=/path/to/npuxl-improved \
      -DKRK_VNPU_LIB=/path/to/npuxl-improved/lib/libvnpu.a
ninja -C build-npu
```

`-DKRK_ENABLE_HIP=OFF` is required on a machine with no ROCm — the default is
ON and configuration fails without it.

## 9. Where to read next

| document | covers |
|---|---|
| [MOE-RESIDENCY.md](MOE-RESIDENCY.md) | expert residency against prefill and decode, the WARM tier, chunk size, the CPU/NPU arms, the int8+streaming cliff |
| [OPTIONS-MATRIX.md](OPTIONS-MATRIX.md) | every flag and `KRK_*` variable, the ones that decide throughput first, the full list by owning file |
| [BENCH-CPU-DOT.md](BENCH-CPU-DOT.md) | the dot-product finding, the 2 GiB MoE experiment, block-format limits, headroom |
| [CPU-VS-NPU-BENCHMARK.md](CPU-VS-NPU-BENCHMARK.md) | the NPU path end to end: w8a8, device attention, refusal rules, repro commands |
| [ARCHITECTURE.md](ARCHITECTURE.md), [ARCHITECTURE-TIERED.md](ARCHITECTURE-TIERED.md) | module layout and the residency tiers |
| [DEEP-DIVE-MOE-PREFILL-DECODE-2026-10-22.md](DEEP-DIVE-MOE-PREFILL-DECODE-2026-10-22.md) | the MoE prefill/decode investigation on real checkpoints |
| [AUDIT-pending-index.md](AUDIT-pending-index.md) | what is settled, what is open, and what was tried and lost |
