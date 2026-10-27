<img width="768" height="512" alt="Octopus AI Workshop in a Vintage Study" src="https://github.com/user-attachments/assets/7cff7936-e8fe-4da3-a3c8-6be9c04c5b06" />

# KRAKEN

Kraken is a GGUF inference engine written in C++17 for AMD GPUs. It talks to the
card through the ROCm HIP SDK directly. There is no PyTorch, no Python in the hot
path, and no hipBLAS, MIOpen or rocBLAS. Every kernel in this repository is
written against the instruction set of the card: WMMA on RDNA3 and RDNA4, and
packed `v_dot2_f32_f16` math on RDNA2.

It is built and tested on the **Radeon RX 6700 XT** (`gfx1031`, RDNA2) and the
**Radeon RX 9070 XT** (`gfx1201`, RDNA4), on Windows 11 and on Linux, and it
should run on the rest of the RDNA family.

A 7B model in Q4_K_M occupies 4.4 GB of VRAM and not 14 GB, because the weights
stay quantized for the whole run. A 235B Mixture-of-Experts model runs on a 12 GB
card, because only the experts the router selects have to be resident at any
moment. Those two numbers are what the design is for.

## The idea

Most AMD inference stacks are a GEMM library with a graph on top. This engine is
the other way round. The kernels are the product and the graph is thin.

**The weights stay quantized in VRAM.** A Q4_K row is read as Q4_K. Decode
dequantizes 32-value chunks into registers, and prefill stages dequantized f16
tiles through shared memory. Neither path keeps a full f16 copy of the model.

**One chunking rule for every format.** `dequant_chunk<T>(block, chunk, out)`
materializes exactly 32 values. Every GGUF format is a whole number of 32-value
chunks: one for Q4_0 and Q8_0, eight for the 256-value K-quants, one for raw
f16, bf16 or f32. The GEMM, the GEMV and the embedding gather therefore share one
inner loop and one tiling rule, and K tiles are multiples of 32.

**Three compute paths, chosen per dtype pair and per row count:**

| path | when it runs | primitive |
|---|---|---|
| `gemv_kernel` | decode, one row | fused dequant plus dot into registers, one block per channel |
| `gemm_wmma` | prefill, gfx11 and gfx12 | `v_wmma_f32_16x16x16_f16`, 32x64x128 block tile |
| `gemm_simt` | gfx10, or any non-WMMA target | 64x64 tile, `v_dot2_f32_f16` inner product |

**Prefill skips the output head** except on the final row, so a 4000 token prompt
does not pay for 4000 vocabulary projections.

**Attention is an online-softmax kernel.** The KV cache is walked in tiles, the
running maximum and normalizer are kept per tile, and the output accumulator is
rescaled. The score row is never materialized. Shared memory stays under about
33 KiB at any context length.

**Experts are paged in, and the cache is frequency-aware.** A `qwen3moe` or
`qwen2moe` file carries one FFN per expert per layer, and a 235B-A22B model
touches about 22B of them per token. Kraken maps the file and pages in only the
experts the router selects, into a byte-budgeted residency set. The policy is LFU
with aging rather than LRU, because expert popularity is sticky: every hit earns
a counter, a hot expert is pinned against eviction, and periodic decay lets a pin
lapse when the traffic moves on. `--expert-warm-mb` adds a second tier in
pageable host RAM, and the mapped file is the cold source below it.

## Build

### Windows 11, native

Install the HIP SDK for Windows from AMD, then:

```powershell
.\scripts\build_windows.ps1 -Targets "gfx1031;gfx1201"
```

A single card is `-Targets gfx1201`. The script finds `hipcc.bat` under
`%HIP_PATH%` or the standard install directories, and falls back to a CPU-only
build with a clear message when the SDK is absent.

### Windows 11, with CMake

```powershell
cmake -B build -DKRK_GPU_TARGETS="gfx1031;gfx1201"
cmake --build build --config Release
```

### Linux

```sh
sh ./scripts/build_linux.sh
KRK_TARGETS="gfx1031 gfx1201" sh ./scripts/build_linux.sh
cmake -B build -DKRK_GPU_TARGETS="gfx1201" && cmake --build build -j
```

### CPU only

```sh
sh ./scripts/build_linux.sh --cpu-only
cmake -B build-cpu -DKRK_ENABLE_HIP=OFF && cmake --build build-cpu -j
```

The binaries land in `build-hip/`, or in `build/` for a CPU-only configure. Keep
them current: `kraken-inspect` answers from the same tables the loader uses, so a
stale copy reports `refused` for formats the engine already runs. The CPU backend
is a scalar f32 reference, and it is what the device path is checked against.

### On a virtual NPU

The compute backend is selectable: HIP, the scalar CPU reference, or the virtual
NPU from [npuxl-improved](https://github.com/riteshnair/npuxl-improved) — a
dependency-free software device that runs on its own SIMD kernels (AVX2 /
AVX-512 / NEON / scalar). Build the device first, then the engine against it:

```sh
git clone https://github.com/riteshnair/npuxl-improved
cmake -S npuxl-improved -B npuxl-improved/build -DVNPU_BUILD_TOOLS=OFF \
      -DVNPU_BUILD_TESTS=OFF -DVNPU_BUILD_SHARED=OFF
cmake --build npuxl-improved/build -j

cmake -B build-npu -DKRK_ENABLE_HIP=OFF -DKRK_ENABLE_NPU=ON \
      -DKRK_VNPU_ROOT=$PWD/npuxl-improved
cmake --build build-npu -j

build-npu/kraken -m model.gguf -p "The capital of France is" -n 64 --npu
```

Both arms are per run, so the same binary still runs the reference arm (`--cpu`)
for comparison:

| arm | what runs on the device | what stays on the reference |
| --- | --- | --- |
| `--npu` | the GEMM: all of the model's weight math | everything else |
| `--npu-cpu` | the GEMM plus RMSNorm (and `qk_norm`), SwiGLU, add, and mul | rope, attention, the KV cache, embedding, MoE routing and top-k, the SSM / MLA / hyper-connection blocks |

`--npu` is the conservative arm: the GEMM is the one VNPU op whose numerics
agree with the reference to f32 rounding (measured 2.7e-07 relative), so the run
differs from `--cpu` in the last bits and nowhere else. `--npu-cpu` adds the ops
whose results are defined by a nonlinearity — and a nonlinearity is where a
device stops agreeing to the last bit. Its SwiGLU runs VNPU's own `fast_expf`
(a degree-5 polynomial for 2^x) rather than libm: measured **8.4e-05 relative**
against the exact silu, which is still better than an f16 activation's 5e-04 and
still a different rounding of the same model, not a different model. `add`,
`axpy` and `mul` are bit-identical to the reference; RMSNorm is f32 rounding.
VNPU's `gelu` is *not* used at all — it measures 5.5e-03 absolute and 0.74
relative on the tails — so a GeGLU model keeps the reference activation on both
arms. `src/npu/npu_offload.hpp` has the full argument, op by op.

Measured on this box (2 cores, AVX-512 VNNI) against a 4-layer 512-dim llama
fixture, 88-token prefill then 64-token greedy decode, min of 5:

| arm | prefill | decode |
| --- | --- | --- |
| `--cpu` | 1393 ms (63 tok/s) | 63.5 tok/s |
| `--npu` | 128 ms (689 tok/s) | 159.7 tok/s |
| `--npu-cpu` | 125 ms (706 tok/s) | 153.1 tok/s |

So the GEMM is the whole story at this size: both NPU arms are ~11x the
reference on prefill and ~2.5x on decode, and `--npu-cpu` is within noise of
`--npu`, not faster than it — the extra ops are 512-1376 elements per call,
which the reference body does in a few microseconds from cache. Routing them
is not free but it is close to it, and it is the arm to read the whole graph
through. `KRK_NPU_MIN_ELEMS=N` sends only ops of N or more elements to the
device, which is how to skip the hand-off for tiny calls (0, the default, sends
everything).

What the two arms do *not* change:

- **Chunk invariance still holds.** Each arm prints the same `--prefill-hash`
  for `--chunk 8`, `128` and `512` (and each arm's hash differs from the
  others', which is the point of having them).
- **Batching loses nothing.** `KRK_NPU_STRICT_QUEUE=1` submits and waits on
  every single call instead of one batch per host boundary, and produces
  byte-identical output — that is the control that proves no host kernel reads
  a row the device has not written.
- **The greedy trajectory survives the rounding.** On that fixture `--cpu`,
  `--npu` and `--npu-cpu` all produce the same 128-token greedy text. That is a
  coherence check, not an accuracy claim; the accuracy claim is the 8.4e-05
  above, measured directly against the reference activation.

Every GEMM call is pinned to one compute unit by default (`KRK_NPU_UNIT=voodoo`;
`cell` and `wildcat` are the other fixed choices), because VNPU's `auto` picks a
different kernel for a one-row call than for a prefill-sized one and the
chunk-invariance the engine is built around would stop holding as soon as a
prompt had more than four tokens. `KRK_NPU_UNIT=auto` restores VNPU's own
choice. The run summary carries the split, per op and with the host ops named:

```
[stats ] npu       gemm 3712, rmsnorm 1152, swiglu 512, add 1024, mul 0, 0 declined, 640 batch(es); 4.7 GFLOP, 50.2 MiB staged; host: rope, attention, kv cache, embedding, MoE routing, top-k, ssm/MLA/HC blocks (no VNPU opcode)
```

Other knobs: `KRK_NPU_THREADS` (device workers, 0 = hardware concurrency),
`KRK_NPU_WEIGHT_MB` (dequantized weight staging budget, default 1024),
`KRK_NPU_MIN_ELEMS` (small-op floor, see above), `KRK_NPU_PIN_THREADS=1` and
`KRK_NPU_TRACE=file.json` (VNPU's own Chrome trace).
A weight tensor that does not fit the staging budget stays on the reference
path, counted in `declined`, and says so once.

`KRK_NPU_W8A8=1` switches the GEMM path to the device's int8 kernel: weights
are packed to int8 once (per output row, so each row keeps its own scale) and
the activations are quantized to int8 on the device, per row, immediately
before each multiply. That quarters the weight traffic, which is what decode is
bound by, and it is **off by default because it is lossy** -- it quantizes the
activations as well as the weights, there is no int8-weights-by-f32-activations
opcode to fall back on. Measured on the synthetic fixtures, weights land 4x
smaller (247.4 -> 61.8 MiB on the 1024-dim one) and decode runs 1.3x-3.6x
faster, but a 32-token greedy continuation is byte-identical to the f32 arm on
6 of 9 fixtures and differs on the other 3; the per-position flip rate is 3-15%.
Use it when a changed tail is acceptable and the tokens/s is not.
`docs/CPU-VS-NPU-BENCHMARK.md` section 10 has the full table, the one-time cost
and the accuracy witness.

`KRK_NPU_ATTN=1` offers ATTENTION to the device as well. It is the quadratic
term in prefill and the reference implementation is a scalar loop, so on a
4-layer fixture prefill runs 8% faster at a 257-token prompt and 28% faster at
513 -- the gain grows with prompt length exactly as the arithmetic predicts.
Decode is unchanged, because the device loses below about four query rows and
those calls stay on the reference (measured 0.21x at 1x129).

It is **off by default for a different reason than int8's**: not accuracy, but
the chunk-invariance gate. The device's own attention kernel is chunk-invariant
-- a run with every chunk on the device hashes identically at `--chunk 8`, `128`
and `512` -- but the row-count threshold means a small chunk takes the reference
and a large one takes the device, and those differ at 1e-07. So the *switch*,
not the kernel, is what costs the invariant. With the switch off, this tree
still prints the same `--prefill-hash` at every chunk size, byte for byte, which
is what the shipped default does. `KRK_NPU_ATTN_MIN_TOK` (default 16) moves the
crossover. Section 11 has the numbers and the refusal table.

## Run

```sh
kraken-inspect model.gguf                        # what is in this file
kraken -m model.gguf --info                      # the device, and the VRAM ledger
kraken -m model.gguf -p "The capital of France is" -n 64
kraken -m model.gguf --chat --system "You are terse." -p "Hi" --greedy -v
kraken -m model.gguf --bench --greedy            # prefill and decode throughput
kraken -m model.gguf --chat                      # interactive
kraken -m big.gguf --draft small.gguf --draft-tokens 4 -p "The capital of France is"
```

A short run first is the habit worth keeping, because a model that is going to
produce nonsense has already said so by the third token. To check correctness as
well, `scripts/coherence_check.sh model.gguf` runs one prompt on the device and
on the scalar CPU reference and requires the two texts to be identical.

Exit status: `0` ran, `1` the run failed and the message names the phase, `2` the
command line was wrong. A refused GGUF is a load-time verdict and not a crash.

The test suite needs no GPU:

```sh
build-hip/kraken-tests.exe     # 4796/4796 checks on the CPU-only build
```

There is also an OpenAI-compatible HTTP server, `kraken-server`, which the long
form describes. It keeps the KV cache between requests: a prompt that repeats or
extends what the previous request prefilled is only prefilled from the point
where it stops matching, and `usage.prompt_tokens_details.cached_tokens` says how
many tokens that was. Measured on a 21-token prompt, the second identical request
went from 135 ms to 27 ms with byte-identical text. A request that fails clears
the cache entry rather than leaving rows behind that a later prompt would assume
were its own.

### Schemas, not flags

A schema is a separate reader, not a set of flags on llama: a file read against
the wrong schema does not fail, it emits fluent nonsense. So
`general.architecture` selects the reader, and an architecture this build does
not know is refused by name with the list of ones it does.

**GPT-2 is the newest one, and it is the first that is not a llama variant.**

| | GPT-2 | llama |
|---|---|---|
| norm | LayerNorm, carries a bias | RMSNorm, no bias |
| positions | learned table, added into the embedding | RoPE |
| q / k / v | one fused `attn_qkv` and one bias | three tensors |
| MLP | GELU, ungated -- the file has no `ffn_gate` at all | SwiGLU, gated |
| output | untied `output.weight` with a bias | often tied |

`attn_qkv.weight` is the tensor every other family is *refused* for; it is
served for gpt2 only, by a flag on the arch table rather than by a name test at
the call site. LayerNorm has no backend op, so it runs host-side -- that is a
kernel still to write, not something to fake with an RMSNorm.

```sh
build-npu/kraken -m Tinystories-gpt-0.1-3m.gguf --cpu -p "Once upon a time" -n 48 --greedy
```

```
, there was a little boy named Tim. Tim was a big, happy dog named Tim. He
lived in a big house, there lived a big house with a big house. One day, Tim
was a little boy named Tim was
```

That is a 3M-parameter model with 8 layers, 64-wide embeddings, 4-wide heads and
four quant formats in one file (F32, Q8_0, Q3_K, IQ4_NL), so the loop, not the
prose, is what it demonstrates: `--cpu` and `--npu-cpu` emit byte-identical
text, and `--chunk 1 / 2 / 3 / 512` emit identical text, on a 50257-token
vocabulary.

On a real Q3_K_M llama (`tinyllama-15M-stories`, 6 layers, 288 embd, hd 48,
`vocab 32000`) the same gates hold. The device is worth nothing there, and the
comparison is only honest because the reference arm now runs the kernels its
host can actually execute:

| arm | prefill | decode |
|---|---|---|
| `--cpu` | 655 tok/s | 209 tok/s |
| `--npu-cpu` | 602 tok/s | 119 tok/s |

An earlier version of this table read 337 -> 870 tok/s, a 2.58x win for the
device. That was measured against a reference arm that was running the scalar
dot product on a host with AVX2, i.e. against a handicap this file set up
itself. Fixing the handicap (`KRK_CPU_DOT_AVX2` now defaults to whatever the
CPU reports, see below) removed the win. On the Q4_K MoE the device does still
win prefill, 49 -> 132 tok/s, and loses decode, 21 -> 17.

## Prefill

Prefill was running at GEMV speed. `gemm()` split its work over the flat
`(row, output)` index space and called `vec_dot()` per element, and `vec_dot()`
decodes a weight row as it goes -- so every weight row was decoded once **per
row of the batch** instead of once per chunk. The signature was that cost per
prefilled token did not fall with batch size: flat at 34.1, 29.1, 27.8, 28.3 ms
across prompts of 58, 149, 331 and 695 tokens.

`vec_dot_tiled()` decodes each weight block once into registers and reuses it
across a tile of eight rows. Same flags, parent commit's binary against this one:

| model | prompt tok | before | after | speedup |
|---|---:|---:|---:|---:|
| 2 GiB q8_0 MoE, `KRK_CPU_DOT_AVX2=1` | 331 | 9300 ms | 2778 ms | **3.35x** |
| | 695 | 19575 ms | 6760 ms | **2.90x** |
| 1 GiB f16 dense | 149 | 14348 ms | 3348 ms | **4.29x** |

It is a speed change and nothing else: every kernel does the same operations in
the same order as the path it replaces, so results are **bit-identical**, not
close. `test_gemm_batched_bit_identical` compares 5 formats against `vec_dot()`
with `memcmp` across 4 arms and 5 tile widths (100 checks), and `--prefill-hash`
is unchanged end to end on both a q8_0 MoE and an f16 dense file. Decode is
untouched -- at `rows == 1` there is nothing to amortize, so it keeps the old
path.

Covered formats: `q8_0`, `q4_0` (hand-written AVX2 kernels), `f16`, `f32` (F16C),
and every other block format through a generic tile -- Q4_K, Q5_K, Q6_K,
iq-quants, which is most of a real collection. Unlisted formats and hosts
without the ISA fall back to the per-element path rather than refusing.

**These kernels are now selected by the CPU, not by the environment.** They used
to need `KRK_CPU_DOT_AVX2=1`, which meant the default run -- the one people
measure -- was the slow one, and every device comparison was drawn against it.
They are bit-identical to the four-lane path they accelerate
(`test_vec_dot_avx2_matches_fast` asserts that with `memcmp`), so an AVX2 host
should simply run them. `KRK_CPU_DOT_AVX2=0` still forces the scalar reference
for a bit-exactness run.

VNNI is chosen at run time, not at build time: `cpu_has_vnni()` tests
`avx512vnni`/`avxvnni` with `__builtin_cpu_supports`, so one binary runs
`vpdpbusd` kernels on a Zen 4 / Sapphire Rapids host and the `maddubs + madd`
pair on the Zen 3 target, with no rebuild and no flag.
`KRK_CPU_NO_VNNI=1` forces the AVX2 kernels on a VNNI host -- that is how the
AVX2 twins are measured here, and it is the escape hatch for a hypervisor that
reports the feature wrongly.

#### The min term, and what it was costing

Q4_K stores `w = d*sc*(q) - dmin*m` per 32 values, so every dot product also
needs `S_s = sum(x_k)` for that sub-block. The kernels were rebuilding `S_s`
**per weight row** -- two `maddubs`, one `madd`, a `cvt` and a `fnmadd` for every
32 values -- which is as many instructions as the dot product itself. That is
the whole reason the prefill kernel sat at 3.5 MAC/instruction when its own
scale structure allows about 6.4.

`quantize_q8_row_sx()` computes the sub-block sums once per *activation row*,
next to the quantisation it already does. Two kernels then use them:

* `dot_q4_K_q8_sx_x8()` -- eight activation rows against one weight row. The
  block's eight `(sc, m)` pairs are decoded once into an eight-lane f32 vector,
  and the min term becomes **one `cvt` + one `fnmadd` per row per block**, where
  the old kernel spent thirty-two instructions per row per block on it. The
  scale sweep is shared by all eight rows instead of repeated 64 times.
* `dot_q4_K_q8_sx()` -- the single-row (decode) shape, same substitution.

Nothing is dequantised: the block is read where it is stored, and the scale
decoding produces registers, not a tensor. `KRK_CPU_DOT_X8SX=0` routes the
eight-row shape back through the per-row min term; that is the A/B below.

Measured on Tiny-Moe Q4_K_M, greedy, 98-token prompt, best-of-5 interleaved arms,
same binary, both arms on the int8 path:

| arm | prefill | decode |
|---|---:|---:|
| per-row min term (before) | 177.8 tok/s | 83.5 tok/s |
| `sx` + batched min term | **265.4 tok/s** | **86.8 tok/s** |
| | **1.49x** | 1.04x |

On a 46-token prompt the same change measures 1.37x, because the fixed cost of
the prompt-encoding tail is a larger share of a shorter prefill. Decode gains
4-11% depending on prompt length, from the single-row kernel alone.

#### Q6_K: the bias was being recomputed per row, twice over

Q6_K stores `w = d·sc·(q − 32)` with `sc` every **16** values. Three things were
wrong with the kernel:

1. `S = Σx` was computed **per weight row**, for every one of the `n_out` rows.
2. The correction was applied as `t = I − 32·S` in int32 — a `mullo` and a `sub`
   per 16 values, on top of the dot.
3. There was no eight-row kernel at all, so Q6_K prefill ran at GEMV speed while
   Q4_K had been batched for months. That is why Q6_K was 18% of the run at 6% of
   peak.

The fix is to notice that `vpdpbusd` takes an **accumulator**: Q6_K's nibbles are
0..63 (the unsigned operand), the activation is int8 (the signed one), and
`Σ(q−32)·x` is exactly what you get if the `−32·Σx` is already sitting in the
lanes when the dot lands. So `quantize_q8_row_sq4()` stores
`−32·Σx` per **four** values — four, not sixteen, because the lanes of
`vpdpbusd` each own four values, and a 16-value sum written into all four lanes
subtracts it four times (measured 3.8e+02 relative error; that is how the wrong
granularity announces itself). With that, the correction costs the kernel
nothing — no subtract, no multiply, no permute — and the whole 32-value group is
`load · vpdpbusd · cvt · fmadd`.

`dot_q6_K_q8_sq4_x8()` adds the eight-row prefill shape, which decodes the
210-byte weight block and its nibble unpack **once** and uses it eight times.

| Tiny-Moe Q4_K_M, 98-token prompt, 10 runs interleaved both ways | prefill | decode |
|---|---:|---:|
| per-row bias, no x8 (before) | 266.4 tok/s | 113.1 tok/s |
| `sq4` + x8 (after) | **309.1 tok/s** | **124.0 tok/s** |
| | 1.18x | 1.10x |

Per component, which is where the change actually lands:

| row | before | after | |
|---|---:|---:|---|
| `gemm_q6k_rN` (prefill) | 4226.9 us/op, 20.3 GMAC/s | **1697.1 us/op, 50.6 GMAC/s** | **2.49x** |
| `gemm_q6k_r1` (decode) | 84.1 us/op, 16.0 GMAC/s | **62.1 us/op, 21.7 GMAC/s** | 1.35x |

End-to-end moves less than the kernel because Q6_K is 12-15% of the run and the
Q4_K rows are larger. `KRK_CPU_DOT_Q6SQ4=0` restores the old kernel; that is the
A/B.

**Accuracy is the *same* kernel, not a near miss.** `dot_q6_K_q8_sq4()` and the
eight-row form agree with the dequantised block to 1.50e-06 relative — identical
to the kernel they replace, because the integer dot is exact and only the order
of the f32 scale application moves. The gate is logits, not text: on four
prompts, against the exact non-int8 path, the new arm sits **closer** to the
reference than the old one and agrees on the top-1 token more often.

| arm | mean \|Δlogit\| on the exact top-1 | top-1 agreement |
|---|---:|---:|
| int8, per-row bias (before) | 0.0346 | 3/4 |
| int8, `sq4` (after) | **0.0320** | **4/4** |

Greedy text still differs between arms on one prompt — the exact model's own
top-2 are 0.017 apart there, and int8 quantisation is worth ~0.03, so that is a
coin flip in the model and not a kernel defect. Judging a kernel by greedy text
on a 4096-piece vocabulary is how you chase noise.

#### The same table on the NPU

`--kprof` was CPU-only, which made it useless for the backend that needed it
most. The vNPU executes a *batch*, so the only honest way to put device time on
a component is to remember which component filled the batch, measure the batch
once at `sync`, and split it across those components by MAC count. That is what
`src/npu/npu_offload.cpp` now does, into the same `KernelTime` table — so a run
reads the same on every backend, and a mixed run (`--npu-cpu`, where the ops the
device has no opcode for stay on the host) shows device rows and host rows
side by side, which is the question "what is the NPU not doing" answered
directly.

Tiny-Moe Q4_K_M, `--npu --kprof`, 46-token prompt, 32 tokens:

```
component                       ops         dev_us      us/op    %run    GMAC/s    %peak
npu_gemm_f32_r1                4124      1169020.0      283.5   79.4%       3.8    13.1%
npu_gemm_f32_rN                 132       253280.1     1918.8   17.2%      25.7    87.6%
attention                       384        31054.0       80.9    2.1%       2.4     3.5%
silu_mul                        768         9795.0       12.8    0.7%       0.0        -
rope                            384         3641.0        9.5    0.2%       0.0        -
kv_append                       384         3413.0        8.9    0.2%       0.0        -
rmsnorm                         800         2580.0        3.2    0.2%       0.0        -
```

`%peak` here is the **device's own** ceiling, measured on the device at first
use: one 64×1024×1024 matmul through the same stream the GEMMs use, three
passes, best kept. Against the host ISA it would have been a nonsense number,
and against a datasheet it would have been a lie.

The table answers the question in one line, and the answer is not the one the
earlier write-up assumed:

* **`npu_gemm_f32_rN` runs at 87.6% of the device's peak.** On batched work the
  device kernel is essentially done. There is nothing to tune there.
* **`npu_gemm_f32_r1` runs at 13.1%** — and not because of the round trip. A
  bare stream probe (no Kraken, no engine) times a single 1×1024×1024 matmul at
  **107 µs**, which is 4 MiB of weights at ~39 GB/s: the call is **memory-bound
  on the f32 staging copy**, not launch-bound. Submit over a whole run is
  26.6 ms against 1434.6 ms of sync, so dispatch is 1.8% of the wait.

The cause is structural and now measurable: **this vnpu.h has no
block-quantized matmul opcode**, so every weight the device touches is first
expanded from Q4_K to f32 — 552.1 MiB staged from a 90 MB model, 6.1×. The host
path never does that; it reads the 144-byte blocks where they lie.

The device *does* have a packed int8 matmul, and turning it on
(`KRK_NPU_W8A8=1`) moves decode 17.4 → **21.6 tok/s** (+24%), which is the
direction the arithmetic predicts — but it quantizes activations as well, and it
still loses to the host's 124. On a shared host the NPU cannot win decode by
running the same CPU better; it wins only if the weight format stops being
expanded. That needs an opcode; it is not something the engine can paper over.

The HIP backend already had a richer recorder of its own (`hipEvents`, a
per-op device span plus host submit time and a device-idle gap), but it printed
in its own format. It now also feeds the shared table — device span net of the
recorder's measured per-op floor, MACs from the FLOPs the op reports — so
`--kprof` is one tool with one heading on all three backends. It passes no
roofline there: a `%peak` against a number this process cannot measure would be
a guess, so those rows print `-`.

This is also why `KRK_ENABLE_NPU` now runs a **capability probe** at configure
time rather than assuming: `VNPU_OP_MATMUL_Q8_0` exists in some builds of
npuxl-improved and not in v1.0.0, and a link error is a bad way to discover
that. The probe reports which way the binary was built, and the Q8_0-in-place
path compiles out to the f32 staging path — a working path, not a stub.

#### What the trace says now

`--kprof` prints a `GMAC/s` and a `%peak` column per component, and the peak is
**measured in the same process** on the first `--kprof` run: sixteen independent
`vpdpbusd` (or `fma`) chains, register-only operands, counted against the exact
number of MACs those instructions retire, times the cores in the pool. A
datasheet number would have been wrong here -- this box quotes 512-bit VNNI while
the kernels are 256-bit.

Tiny-Moe, `--cpu --kprof`, one quality run:

```
component                       ops         dev_us      us/op    %run    GMAC/s    %peak
gemm_q4k_i8_rN                  102       119371.0     1170.3   37.2%      45.9    13.3%
gemm_q4k_i8_r1                 1122       113818.0      101.4   35.4%      11.5     3.3%
gemm_q6k_rN                      18        48111.0     2672.8   15.0%      21.4     6.2%
gemm_q6k_r1                     210        17964.0       85.5    5.6%      15.8     4.6%
attention                       144         9559.0       66.4    3.0%       5.2     4.8%
silu_mul                        288         5866.0       20.4    1.8%       0.0        -
kv_append                       144         2481.0       17.2    0.8%       0.0        -
rope                            144         2180.0       15.1    0.7%       0.0        -
rmsnorm                         300         1341.0        4.5    0.4%       0.0        -
tracked total: 281.7 ms across 12 component(s) -- model load, sampling and tokenisation are outside every row
roofline probe: int8 365 GMAC/s, f32 115 GMAC/s (this host, all cores)
```

`13.3%` is the number to look at. It is not a scheduling artefact: the kernel is
instruction-bound, and the instruction budget per 64 values is dominated by the
per-32-value scale application, not by the dot product. Closing that gap needs a
different *layout* -- a permuted or unpack-free weight representation so the
scale can be folded once -- not another multiply. Unpack-free 4-bit and ternary
paths exist (BitNet's ternary dot, NVFP4/MXFP4 micro-scaling, table-lookup
NPU datapaths); they are on the list, and the current `x8` kernel is the best
this layout can do at 45.9 GMAC/s. The Q6_K rows (18% of the run, at 6% of peak)
are next, and they have no `sc`-per-32 structure problem -- the same
`sx` substitution applies.

Measured on three real files, greedy, 24 tokens, `--cpu`, same output to the
byte on every row:

| model | quant | prefill scalar -> auto | decode scalar -> auto |
|---|---|---|---|
| Tinystories-gpt-0.1-3m | Q3_K/Q8_0/IQ4_NL | 2198 -> **3328** tok/s | 523 -> **895** tok/s |
| tinyllama-15M-stories | Q3_K_M | 345 -> **653** tok/s | 121 -> **206** tok/s |
| Tiny-Moe | Q4_K_M | 22.5 -> **49.0** tok/s | 13.8 -> **22.0** tok/s |

### The NPU arm of the same benchmark

The virtual NPU does not win by tiling. A device trace of a 58-token prefill
puts **864 ms of device work inside a 2464 ms run** -- 67% of the wall clock the
device is idle, and the remaining 1600 ms is host work (rope, attention, KV,
routing), sync waits, and one-time weight expansion. Its GEMMs run at 24.8
GFLOPS in-graph and 25.7 GFLOPS standalone at the same shape, so the kernel is
*not* being starved; it is shape-limited, and it is already the fastest thing in
the run.

Two defects were worth more than any kernel change, both fixed:

| defect | effect | fix | measured |
|---|---|---|---|
| staging budget was a flat 1024 MiB, silently declining 393 of 1087 GEMMs to the host | 83% device idle | derive the default from available RAM | 3042 -> 2357 ms (1.29x) |
| `KRK_NPU_W8A8` packed every weight on first use, including prefills that never reuse one | +1.0 s of pure loss | pack only when a weight is reused (decode) | 3395 -> 2381 ms (1.43x) |

Both show up in the run summary (`0 declined`, and a w8a8 line that reports how
many weights were deferred), so neither is a silent trade. Against the CPU arm on
the same box the NPU prefill goes from 1.34x to **1.73x**, and W8A8 decode stays
at 72.84 ms/token against 131.37 ms without it.

The arithmetic ceiling for this build, with the host cost driven to zero and the
f32 kernels kept, is 4.7x -- an 8x target additionally needs the int8 kernels, a
packed-weight cache, and the host gap cut by roughly 9x. Full evidence, including
the M-scaling table that shows tile-aligned M is 1.74x faster than the M=58 a
58-token prompt produces, is in [docs/NPU-PREFILL.md](docs/NPU-PREFILL.md).

Method, the rejected first attempt, and the arm table: [docs/MOE-RESIDENCY.md](docs/MOE-RESIDENCY.md) §7.

## MoE under a RAM limit

Can you run a big MoE with a few experts in RAM and the rest on disk? Full
tables in [docs/MOE-RESIDENCY.md](docs/MOE-RESIDENCY.md).

| | all experts resident | 64 MiB, rest on disk |
|---|---:|---:|
| prefill, AVX2 dot | 1020 ms | 857 ms |
| decode, AVX2 dot | 61.1 ms/token | 75.3 ms/token |
| expert read | 307 ms | 521 ms |
| read share of prefill | 30% | **61%** |

Before the prefill fix this table said residency did not matter for prefill at
all (13.95 s against 13.68 s). It read that way because prefill was not batching:
once compute stopped dominating, the expert read became the majority of a
streamed prefill. **So `--expert-warm-mb` and the device budget now matter for
prefill too**, which they did not before.

Three things worth knowing before tuning:

- `--expert-warm-mb` is the lever, not the device budget. A 512 MiB host WARM
  tier at a 64 MiB device budget cut the read from 2519 to 246 MiB (**0.10x**).
- `--chunk` multiplies fetches. Chunk 8 re-fetches per 8 rows: 3907 MiB read
  against 1372 MiB at chunk 128.
- `--threads` halves prefill 1.80x and moves decode not at all. Decode is bytes;
  prefill is arithmetic and, now, some bytes too.

The arms disagree, and the fix changed which one wins: **AVX2 CPU prefills 3.2x
faster than the virtual NPU** (1071 ms against 3386 ms) and the **NPU keeps the
decode win** (33.4 ms against 58.2 ms). With experts streaming the NPU's decode
collapses 12.7x (32.8 -> 416 ms) where the CPU's costs 1.79x, because the int8
pack is memoized per weight address and a streaming slot changes address every
visit -- **do not combine `KRK_NPU_W8A8=1` with forced streaming** (149.0
ms/token against 718.7). That defect and the fix that was tried and rejected on
measurement are in [docs/MOE-RESIDENCY.md §5](docs/MOE-RESIDENCY.md).

Every switch, the arm it belongs to and what it is worth:
[docs/OPTIONS-MATRIX.md](docs/OPTIONS-MATRIX.md).

```sh
# keep 64 MiB of experts on the device and a 512 MiB warm tier in host RAM
KRK_CPU_DOT_AVX2=1 kraken -m moe.gguf --cpu --expert-cache-mb 64 \
    --expert-warm-mb 512 -p "..." -n 64 --greedy

# fastest decode measured here: whole graph on the virtual NPU, int8 + attention
KRK_NPU_W8A8=1 KRK_NPU_ATTN=1 kraken -m moe.gguf --npu-cpu -p "..." -n 64
```

`kraken-inspect` is the first thing to run on an unfamiliar file: it gives the
same verdict the loader will, without loading the weights, and an architecture it
does not know is now **refused by name** instead of being read against llama
defaults.

## Scores

Measured on 2026-10-26 with `kraken --model <MODEL> --bench --greedy
--max-tokens 64`, one run per model, device only, greedy sampling, `--ctx 4096`
and `--chunk 512`. Radeon RX 9070 XT, `gfx1201`, 64 CU, 15.9 GiB VRAM, HIP 7.12
runtime, Release build. The sweep was re-run the same day after the prefill
staging change, so the prefill column is one build newer than the rest of the
record; `docs/AUDIT-prefill-gemm.md` has that change and its numbers.

| Model | Size MiB | Arch | Load ms | Prefill tok/s | Decode tok/s | Note |
|---|---:|---|---:|---:|---:|---|
| Laguna-S-2.1-PS-IQ2_XXS | 32093 | laguna | 23214 | 5.5 | 4.5 | range guard fired |
| K2-Horizon-MoVA-36B-A4B-Q4_K_M | 21331 | not loaded | - | - | - | refused at load |
| laguna-xs2-Q4_K_M | 19335 | laguna | 22089 | 98.1 | 33.2 | |
| LFM2.5-24B-A2B-QAD-Q4_0 | 12949 | lfm2moe | 6293 | 424.3 | 80.7 | |
| Mellum2.1-12B-A2.5B-Thinking-Q8_0 | 12326 | mellum | 9134 | 460.9 | 60.7 | range guard fired |
| Underdog-Saluki-27B-1.0-IQ2-mix-MTP | 7962 | qwen35 | 6102 | 224.9 | 30.8 | |
| omnicoder-9b-q6_k | 7018 | qwen35 | 8763 | 271.2 | 63.1 | |
| Qwen2.5-Coder-7B-Instruct-Q5_K_M | 5192 | qwen2 | 4309 | 561.2 | 72.5 | |
| qwen35-9b-instruct-nvfp4 | 5067 | qwen35 | 4860 | 798.5 | 79.0 | |
| lacuna-v1-Q4_K_M | 4836 | llama | 3956 | 655.3 | 91.7 | |
| Qwen3-8B-Q4_K_M | 4794 | qwen3 | 4279 | 639.5 | 80.8 | |
| mythos-9b-unhinged-heretic.i1-Q4_K_M | 4794 | qwen3 | 3721 | 654.3 | 80.8 | |
| Qwen3.5-9B-DeepSeek-V4-Flash-MTP-Q3_K_M | 4518 | qwen35 | 2632 | 778.7 | 68.8 | |
| Opus4.7-GODs.Ghost.Codex.Distill.4B-Q8_0 | 4274 | qwen35 | 3870 | 1328.8 | 92.8 | |
| Qwen3-4B-Q8_0 | 4082 | qwen3 | 2407 | 1333.5 | 96.2 | |
| VibeThinker-3B.Q8_0 | 3133 | qwen2 | 3181 | 1630.8 | 0.0 | emits EOS at once |
| Samastam-2.5B-Q8_0 | 2560 | llama | 2248 | 3705.3 | 148.8 | |
| Phi-3.5-mini-instruct-python_coding_assistanQ4_K_M | 2211 | llama | 1739 | 1851.7 | 94.2 | |
| smolcode-coder-cpp-3b-q4_k_m | 2007 | qwen2 | 2103 | 1293.5 | 138.8 | |
| smolcode-coder-cpp-1.5b-q4_k_m | 1065 | qwen2 | 1377 | 2521.6 | 196.9 | |
| bitnet-2b-tq1_0 | 1054 | bitnet | 953 | 940.1 | 66.2 | |

Three things the table says:

* **Prefill is still the weak number on the larger dense files, but it moved.**
  Underdog-Saluki-27B reads a prompt at 224.9 tok/s against 30.8 tok/s of decode,
  omnicoder-9b at 271.2 against 63.1, and Qwen2.5-Coder-7B at 561.2 against 72.5,
  against 172.9, 221.5 and 494.3 in the previous sweep. The small models are
  quick at both ends (smolcode 1.5B: 2521.6 and 196.9), so what these rows
  measure is the weight read rather than the kernel shape.
* **What moved is the prefill GEMM's staging chain, and it is measured.** The
  activation tile was the one operand staged with no prefetch, so its loads sat
  in front of the barrier with nothing to overlap them; holding them in
  registers for one round is worth 22% of prefill on Qwen3-8B (830.9 -> 1013.7
  tok/s at a 501-token prompt, interleaved medians) and 31% of the GEMM itself
  (375.1 -> 258.9 ms). The decomposition, the LDS wall that stops a full double
  buffer from fitting, and what is still open are in
  [docs/AUDIT-prefill-gemm.md](docs/AUDIT-prefill-gemm.md): prefill is the GEMM
  first (59 to 69% of it) and attention second (24 to 33%), the GEMM now runs at
  about a quarter of the card's peak against a fifth before, and `--prefill
  exact`, the rung that can produce a hash, measures 8 to 12 times slower than
  the default.
* **MoE decode is bound by expert bring-in, and those rows are not stable.**
  laguna-xs2 holds 74.3% of its expert acquires in VRAM and decodes at 33.2
  tok/s, while LFM2.5-24B, whose routed set fits its tier at 98.9% hits, decodes
  at 80.7. Their prefill column follows the tier's state rather than the kernel:
  laguna-xs2 read 92.8, 82.0 and 98.1 tok/s across three sweeps the same day,
  and Laguna-S-2.1 is 32 GiB against a 15.9 GiB card at 4.5 tok/s.
* **Two MoEs do not produce a trustworthy answer at these defaults.** Mellum2.1
  and Laguna-S-2.1 both leave the f16 activation range, the range guard reports
  the clipped stores, and the run exits 2. Their rates are what the run reached
  before it stopped being trustworthy.

The full record, including the guard details, the files that have left this box,
and the refusal list, is in [docs/MODEL-STATUS.md](docs/MODEL-STATUS.md) and
[docs/traces/results.tsv](docs/traces/results.tsv).

## Where the rest of it is

* [docs/TECHNICAL-README.md](docs/TECHNICAL-README.md) is the technical readme:
  how a token flows, the expert residency tiers, the full model-family table
  with the reason each unsupported family is unsupported, what is measured and
  what is not.
* [more-info.md](more-info.md) is the long form: every switch with worked
  examples, the full benchmark record with context scaling, KV quantization and
  speculative decode, the HTTP server, the supported model table, and the list of
  what is refused and why.
* [AGENTS.md](AGENTS.md) is the working note: what each gate in the repository
  is for, the trap behind each default, and the measurement each default rests
  on.
* [docs/MOE-RESIDENCY.md](docs/MOE-RESIDENCY.md) is the expert-residency study:
  prefill and decode against every expert budget, the WARM tier, chunk size, and
  the device arms side by side.
* [docs/OPTIONS-MATRIX.md](docs/OPTIONS-MATRIX.md) is every switch with its
  measured worth.
* `docs/` holds the design and audit documents, one per subsystem.
* `scripts/` holds the gates (`build_check.sh`, `kraken_tests_baseline_check.sh`,
  `coherence_check.sh`) and the benchmark harnesses.
