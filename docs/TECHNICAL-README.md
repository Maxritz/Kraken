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
| **MiniMax-M3.1-Flash** (428 B MoE, 23 B active, 1M context, MSA sparse attention throughout, native multimodal, DSpark draft head) | not mapped | **not supported** | No GGUF tensor schema has been validated for it, and the graph needs pieces this engine does not have: **MSA** (a MinMax-specific sparse attention), multimodal input towers (text+image+video), and DSpark. Mapping a schema from the safetensors naming alone would be a guess, and a guessed mapping is what produces silently wrong output |
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

Exactness witness: **4576/4576 checks unchanged** across the change, golden
hashes included. Full record, method, block-format limits and remaining
headroom: [BENCH-CPU-DOT.md](BENCH-CPU-DOT.md).

Known limits, stated because they are the next person's starting point:

- The **block formats (q8_0/q4_0) do not get this win** and cannot, while
  staying exact: their reduction is a single dependent chain. q4_0 is *half the
  bytes of q8_0 and 30% slower* for that reason. Fixing it means changing the
  numbers, so it has to be an opt-in with an accuracy witness.
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

## 6. The NPU path

Build with `-DKRK_ENABLE_NPU=ON -DKRK_VNPU_ROOT=<npuxl tree>`. It rides the CPU
path (there is no NPU+GPU wiring, deliberately: no untested HIP+NPU composition
ships). It is separable — the whole backend is `src/npu/` plus two call sites.

Two optimisations, both opt-in and both measured:

| switch | what it does | measured |
|---|---|---|
| `KRK_NPU_W8A8=1` | int8 weight staging for the device | 1.12-1.14x prefill on small stacks, up to 3.6x on medium ones |
| `KRK_NPU_ATTN=1` | runs attention on the device when the call is expressible | +10.3% prefill at 257 tokens, +25.6% at 513, decode unchanged |

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
| [BENCH-CPU-DOT.md](BENCH-CPU-DOT.md) | the dot-product finding, the 2 GiB MoE experiment, block-format limits, headroom |
| [CPU-VS-NPU-BENCHMARK.md](CPU-VS-NPU-BENCHMARK.md) | the NPU path end to end: w8a8, device attention, refusal rules, repro commands |
| [ARCHITECTURE.md](ARCHITECTURE.md), [ARCHITECTURE-TIERED.md](ARCHITECTURE-TIERED.md) | module layout and the residency tiers |
| [DEEP-DIVE-MOE-PREFILL-DECODE-2026-10-22.md](DEEP-DIVE-MOE-PREFILL-DECODE-2026-10-22.md) | the MoE prefill/decode investigation on real checkpoints |
| [AUDIT-pending-index.md](AUDIT-pending-index.md) | what is settled, what is open, and what was tried and lost |
