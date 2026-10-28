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
(`KRK_NPU_W8A8=1`) is the only configuration measured in this project where the
virtual NPU **beats the host CPU at decode**. That claim was missed for months
because it was measured at the wrong generation length. The pack is one-time and
it lands inside whichever step first re-requests a weight, so a 16-token decode
reports it as a regression:

| n | f32 ms/step | i8 ms/step | i8 / f32 |
|---:|---:|---:|---:|
| 16 | 43.2 | 58.4 | 0.74x |
| 32 | 35.6 | 32.6 | 1.09x |
| 64 | 36.2 | 20.5 | 1.76x |
| 128 | 35.7 | **14.2** | **2.51x** |

At n=128 on a dense q8_0 fixture (12 layers x 768, ff 2048, 2 cores): host CPU
58.0 tok/s, NPU f32 28.0, **NPU int8 70.4, NPU+CPU hybrid int8 72.7**. The
summary line now prints the pack time (`84.0 MiB packed in 524 ms one-time`) so
the one-time cost cannot be read as steady state — in a 32-token run it is 48%
of the wall clock, in a 128-token run it is 4.1 ms of 14.2.

The format problem is still the ceiling: `MATMUL_I8` gets the weight traffic
from 370 MiB to 92 MiB per token but pays for it by quantising activations, and
there is still no opcode that reads a block format where it lies. The
consolidated inventory — every idea from the porting table, the patch stack, the
device's own docs and the traces, with the duplicates collapsed — is in
[docs/NPU-CONSOLIDATED.md](docs/NPU-CONSOLIDATED.md).

#### Two changes that made the device the faster arm at decode

Both are routing, not kernels. Same fixture, `--ctx 512`, greedy, medians of 3:

| arm | n=16 | n=64 | n=128 | prefill ms |
|---|---:|---:|---:|---:|
| host CPU | 52.4 | 49.4 | 50.9 | 328.8 |
| NPU f32 | 20.4 | 24.6 | 26.1 | 473.0 |
| NPU int8, one-time cost left in the decode steps (previous) | 15.6 | 41.6 | 54.7 | 316.7 |
| **NPU int8, both (default)** | **83.5** | **76.3** | **77.4** | **314.3** |

1. **The pack moved to load.** The device converts each weight into its int8
   layout once, 2.3 ms each. Left in the decode path it was 48% of a 16-token
   run, which made the int8 arm read as a 0.74x regression while its steady
   state was 2.5x — the earlier "the NPU cannot win decode" verdict came from
   the short end of that curve. The engine now runs one 1-token prepare pass at
   load (`prepare pass: 537.0 ms (one-time weight conversion, outside the timed
   run)`), bracketed by `Backend::set_prepare` and skipped entirely by backends
   with no such cost. It is numerically neutral: it produces identical tokens to
   packing lazily on first use, and differs from *never* packing eagerly only
   because the first use of a weight would otherwise run f32.
   `KRK_NPU_NO_PREPARE=1` opts out.

2. **The device declines shapes it does not win.** Sweeping prompt length, the
   host is faster for every GEMM with more than one row — 2.2x at 5 rows, parity
   at 339 — because the host reads block formats where they lie and the device
   must expand them. So `npu::gemm` now refuses a GEMM above `KRK_NPU_MAX_ROWS`
   (default 1 with int8 on) **before queueing anything**, the caller runs its
   own body, and the summary counts it: `shape refused 84 (rows > 1: the host is
   the faster arm for that shape)`. A refusal is a routing decision, printed, not
   a silent fallback.

Net: decode **1.5x the host CPU, flat from 16 to 128 tokens**, and the prefill
penalty for using the device at all drops from 1.5x to nothing. Details, the
crossover table and the reproduce commands:
[docs/NPU-ROUTING.md](docs/NPU-ROUTING.md).

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

#### Attention, which owns the other half of a long prompt

The `--kprof` table above is a short prompt, where attention is 3% of the run.
Give the same model a real context and the composition inverts: at 2048 tokens
attention is **two thirds** of the wall clock, and it was running at 3% of the
f32 roofline. That is not a rounding error in a kernel nobody needs — it is the
reason long prompts stall while short ones look fine.

The kernel was the textbook form: for every (token, head), walk the visible keys,
dot `q` against each one, softmax, then walk them again for `A·V`. Every key dot
was a 64-deep dependent FMA chain with `q` re-loaded from L1 for each key. Two
changes, both in `CpuBackend::attention`:

* `q` lives in eight `__m256` registers for the whole scan, and the dot reduces
  over 4 independent accumulators instead of one — the chain depth per key drops
  from 64 to 8, which is the difference between waiting on latency and waiting
  on ports;
* the mask moves to a second pass, so the key loop is branch-free, and `A·V`
  accumulates *across dimensions* (the j order of the output sum is untouched,
  so the output half of the kernel is order-exact).

Same run, both arms, 2048-token prefill, `--kprof`:

| arm | prefill | `attention` µs/op | share of run | GMAC/s | %peak |
|---|---:|---:|---:|---:|---:|
| scalar reference (`KRK_CPU_ATTN_SCALAR=1`) | 92.7 tok/s | 310,955 | 67.9% | 3.2 | 3.0% |
| AVX2 + FMA | **162.7 tok/s** | 110,906 | 42.6% | 9.1 | 9.0% |

**1.75x end to end, 2.80x on the op.** On a short prompt attention is 3% of the
run and the same change is 1.08x, which is the honest size of it.

Accuracy, on a 10-shape fixture (decode row at pos 1400, 64-token prefill, GQA
8/2, `hd` 80/96/256, window 64, `skip_from`, sinks, softcap) against the scalar
reference: 18.5% of the 88,320 output floats bit-identical, global relative
error **1.82e-07**, max absolute **9.5e-07** — the signature of a reordered f32
reduction, nothing else. `--prefill-hash` moves (it must: the QK dot order
changed) but stays invariant across `--chunk 8/32/128`, and the suite is
4796/4796.

#### The layout was the kernel: one gather, 2.2x

`9%` of peak was still the number to argue with, so the next step was the obvious
one: four keys per horizontal reduce. It bought **nothing** — and the reason is
the whole of the next change. Same arithmetic, same flags, 4800 key-rows:

| variant | real cache (`pos_stride` 3072 B) | contiguous tile (256 B) |
|---|---:|---:|
| one key per reduce (as shipped) | 16.14 ns/key | 2.99 |
| four keys per reduce | 14.47 | 2.44 |
| + prefetch 8 ahead | 13.77 | 2.29 |

The KV cache is `[position][head][hd]`, so one head's slice of one key is **256 B
out of a 3072 B stride** — the loop reads four cache lines and uses one. Against
a contiguous tile the identical kernel is **5.4x** faster; every optimisation
aimed at the loop is capped at ~1.2x.

So the strided pass is **amortised, not removed**: one gather per head serves
every query row in the chunk, instead of one pass per (token, head). It is gated
on `n_tok > 1` — a decode step has one row, nothing to amortise — and the loop
is now head-major, because the tile belongs to a head and every token reuses it.
`KRK_CPU_ATTN_GATHER=0` keeps the in-place path for comparison.

Full metrics, 3 interleaved reps, `median [min, max]`, attention row from
`--kprof`:

| prompt | arm | prefill tok/s | attention ms/op | %run | GMAC/s | %peak |
|---|---|---:|---:|---:|---:|---:|
| ~46 | scalar | 329.9 [273.9, 352.3] | 0.2 | 4.7% | 8.1 | 6.8% |
| ~46 | in-place | 372.2 [335.7, 418.6] | 0.1 | 2.0% | 20.0 | 16.2% |
| ~46 | **gather** | 370.2 [353.4, 385.2] | 0.1 | 1.6% | 25.5 | 19.0% |
| ~900 | scalar | 117.6 [114.3, 119.4] | 54.3 | 51.0% | 4.1 | 3.6% |
| ~900 | in-place | 166.3 [156.5, 172.6] | 21.7 | 29.1% | 10.1 | 8.7% |
| ~900 | **gather** | **212.5 [193.5, 214.1]** | 7.9 | 13.4% | 28.0 | 24.7% |
| ~2040 | scalar | 89.9 [88.6, 92.5] | 293.6 | 61.9% | 3.4 | 3.0% |
| ~2040 | in-place | 148.0 [146.7, 150.2] | 105.4 | 37.0% | 9.5 | 8.5% |
| ~2040 | **gather** | **198.1 [197.0, 202.7]** | **35.6** | **16.9%** | 28.3 | **25.3%** |

| prompt | in-place / scalar | gather / scalar | gather / in-place |
|---|---:|---:|---:|
| ~46 | 1.13x | 1.12x | 0.99x |
| ~900 | 1.41x | **1.81x** | **1.28x** |
| ~2040 | 1.65x | **2.20x** | **1.34x** |

Attention on its own op at ~2040 tokens: **293.6 -> 35.6 ms/op, 8.25x** against
the scalar reference and 2.96x against the previous kernel; its share of the run
61.9% -> **16.9%**, its rate 3.4 -> **28.3 GMAC/s** (25.3% of the measured f32
peak). Decode is untouched by design (60.0 vs 60.8 tok/s at ~46 tokens).

**The gather adds exactly zero error**: in-place vs gather is **100.00%
bit-identical** on all ten accuracy shapes, and both sit at rel **7.7e-08** to
the scalar reference. The fixture earned its keep twice here — it caught two
bugs written in this change, an assigning 64-dim chunk (loses all but the last at
`hd` 128/256, rel 7e-02) and an assigning remainder (loses the first at
`hd` 80/96), neither of which is visible by reading the code.

Details, the crossover table and the reproduce commands:
[docs/ATTN-GATHER.md](docs/ATTN-GATHER.md). INT8 kernel tiers by ISA, and why
ggml's block formats need none of the bias correction Intel's VNNI sample
describes: [docs/INT8-ISA-TIERS.md](docs/INT8-ISA-TIERS.md).

#### Q8_0 was the last format that dequantised in the hot loop

With attention down, the trace shows what it was hiding: **`gemm_q8_0_rN`, 81.3%
of the run at 19.3 GMAC/s**. Q8_0 was one of the two documented exceptions to
"never dequant" — `dot_q8_0_tiled()` widened every weight byte to f32 and
multiplied by the block scale.

`VPDPBUSD` is unsigned × signed and Q8_0 is signed × signed, so the tutorial fix
is to bias the weight by 128 and subtract `128*sum(x)`. **On AVX2 that silently
breaks**: `VPMADDUBSW` accumulates into int16 and `255*127*2` overflows it, so
the instruction *saturates* rather than wraps. The kernel instead splits the
weight into unsigned magnitudes and subtracts the products — `128*127*2 = 32512`
fits, for any legal byte pair, including the `-128` the format allows (which a
saturating `abs_epi8` would have mangled to 127). No bias, no `sx[]` array.

Two 32-byte blocks are exactly one 512-bit register and consecutive blocks'
activations are contiguous, so the 512-bit kernel does 64 MACs per maddubs pair.

| ~2040 tokens | prefill tok/s | GEMM ms/op | GEMM %run | GEMM GMAC/s |
|---|---:|---:|---:|---:|
| dequant | 180.9 [179.0, 189.8] | 9689.5 | 82.0% | 18.0 |
| int8 256-bit | 256.8 [245.6, 274.2] | 5986.7 | 73.7% | 29.1 |
| **int8 512-bit** | **272.1 [244.5, 277.8]** | **5048.0** | **70.4%** | **34.5** |

**The GEMM op is 1.92x faster; the run is 1.50x** (1.49x at ~900 tokens, 1.39x
at ~46), because attention becomes 17.0% -> 25.7% of what remains. Decode
improves too without any decode kernel changing — rows == 1 lands on the
single-row int8 kernel instead of the f32 widen: 33.1 -> 44.4 tok/s at ~900.

**Accuracy, asked as two separate questions.** The kernel computes the dot it
claims to: worst error vs an f64 reference **2.329e-08**, and the 256-bit
single-row and 8-row kernels are bit-identical. The 512-bit one is *not*, and
must not be required to be — it accumulates 16 i32 lanes per two blocks where
the 256-bit kernel accumulates 8 per one, so the reduction tree differs while
the integers summed do not. End to end, quantising activations (which the Q4_K
and Q6_K paths have always done) moves the worst top-8 log-prob by 1.7e-02 with
**identical greedy text over 16 tokens**; `KRK_CPU_DOT_INT8=0` restores the exact
f32 path.

The 512-bit widening bought only 1.07x, not the ~2x the instruction count
predicts, so this kernel is not instruction-bound on this host. That gap is the
honest state of it and the next thing to measure.
[docs/Q80-INT8-GEMM.md](docs/Q80-INT8-GEMM.md).

#### Chasing that gap: two hypotheses, both partly wrong

The GEMM is 83.7% of the run and 13% of the int8 ceiling, so the missing
performance is there. Two explanations were each tested and each explained far
less than it predicted.

**The ceiling had never been measured** — `%peak` divided by `f32 x 4`, a guess.
Probing it properly took two attempts: the first probe used loop-invariant
`maddubs` operands, the compiler hoisted them, and the loop degenerated to eight
integer adds reported as 301 GMAC/s. With the operands bumped so the dot cannot
hoist, the measured ceiling is **int8 301 / f32 102 GMAC/s**, consistent with
2 cores at 3.2 GHz.

| | GMAC/s | vs before |
|---|---:|---:|
| GEMM, before | 29.3 | — |
| + column blocking (1.04x) | 30.5 | the loop streamed the whole activation matrix once per output column — 805 MB per call |
| + sign flip (1.26x) | **39.1** | `sum(w*x) = sum(|w|*(x*sign(w)))`: one `maddubs` instead of two, 5 instructions per 64 MACs instead of 8 |
| prefill tok/s | 299 -> **384** | 1.28x |

Blocking is correct as well as faster — bit-identical logits, which is what a
loop reorder should give. The sign flip is exact for `-128` (a masked subtract,
where `abs_epi8` saturates to 127) and its accuracy is unchanged at 2.329e-08.

##### The ceiling above was wrong, and the kernel is not near it

The 301 GMAC/s figure came from a register-only probe, and it is not repeatable
against the same loop measured standalone: reproducing that exact loop gives
73.7–99.1 GMAC/s per core, so two threads reporting 338–430 GMAC/s are at least
1.7x too high. The host is also not the 3.2 GHz the figure implied — a
dependent FP add with 4-cycle latency takes 2.14 ns here, about 1.87 GHz
effective, with `cpu/wall` at 1.00 and steal time of 7 ticks, so it is not
being descheduled.

The measurement that does hold up puts the same instruction mix, with loads,
in one process beside the shipped kernel (`tools/bench_isa_ceiling.cpp`):

| variant | GMAC/s | of C |
|---|---:|---:|
| A registers only, integer accumulate | 45–70 | 115–133% |
| C A plus the real loads and float tail | 35–58 | 100% |
| F shipped kernel body, scale chain removed | 19–45 | 55–70% |
| E **the shipped kernel** | 13–33 | **29–53%** |

So the GEMM runs at about 41% of what this machine does with the same
instructions, and removing the per-block scale dependency chain alone is worth
1.4x. Two attempts to shorten that chain — pair-expanded scales and a 64-bit
broadcast plus permute — both measured neutral (1.04x and 1.01x), which says
the cost is the chain's latency and not its instruction count: shortening it
cannot help, only overlapping it can. `docs/GEMM-PERF-WALL.md` records both
dead ends, the two throughput modes this host flips between, and why every
number here is quoted as a ratio.

##### Hoisting the chain into L1 also loses

The obvious third attempt — build the scale products for a chunk of block pairs
into a 4 KiB stack buffer that stays in L1, so the loop reads them as plain
loads — was built and measured. It is correct (accuracy unchanged at
2.329e-08) and **slower in four of four interleaved pairs**, 313/334/320/339
tok/s against 335/354/351/343 for the shipped kernel. The store-then-load
traffic and the extra addressing cost more than the dependency it removes.
Three attempts at the chain, three dead ends: it survives shortening *and*
hoisting, so the remaining win has to come from doing fewer operations per MAC
or fewer bytes per MAC, not from rearranging this one.

##### Ablation finds where the cycles actually go

`tools/bench_q80_ablate.cpp` changes one component of the production kernel at a
time, all rungs in one process. Two results worth having:

| rung | GMAC/s | of production |
|---|---:|---:|
| L0 production | 24.54 | 100% |
| L2 constant weight scale | 33.01 | 135% |
| L7 integer accumulate | 30.54 | 124% |
| L1 **constant activation scale** | 24.66 | **100%** |

The activation scale chain costs **nothing** — deleting it outright changes the
kernel by 0.5%. That is the post-hoc explanation for all three neutral rewrites:
they were optimising a dependency that was never on the critical path. What does
cost is the scalar `fp16_to_fp32` weight-scale decode (**34.5%**) — a branchy
routine with a denormal loop, called twice per block pair where one
`_mm_cvtph_ps` does both. Replacing it won 4 of 4 interleaved pairs, **1.17x**
on the GEMM (median 38.7 vs 33.0 GMAC/s), bit-identical across all 65536 half
values. The int-to-float tail is the next **44%** and is untouched; 16 rows in flight
measures 96% of 8, so the inner loop is not short of independent work and that
lever is closed.

##### Batching several sequences: the enabling change is in

`AttnDesc` now carries `row_pos` and `row_kv_base` (both null by default, i.e.
no behaviour change for any existing caller). Every other kernel is already
row-independent — the GEMM is 85% of prefill — so this is the only piece that
had to learn about sequences. `kraken-attn-multiseq` gates it: four rows from
four sequences at positions 3/17/40/63 must equal the same rows computed one at
a time, **bit for bit**. They do, in both the causal and prefix cases, and the
legacy contiguous path is unchanged.

##### Decode had none of it, and now does

Every kernel win of the previous rounds went into the *eight-row prefill*
kernel. Decode calls the single-row one, which was still the original 256-bit
form: two `maddubs` per block, a scalar fp16 decode, no sign flip. It now has all
three ideas, and decode went **66.6 -> 78.2 tok/s median** (3 interleaved reps).
VNNI specifically buys nothing at one row (latency-bound, one accumulator
chain), so `bw` and `vnni` are equal there; the win is width + sign flip + f16c.

Prefill, same three reps: `avx2` 345.4, `bw` 395.2, **`vnni` 433.5 tok/s**
median — 1.26x across the ladder. Full tables, component breakdown and the
historical progression are in `docs/SCORES.md`.

##### The fill is 1% and the allocation behind it was the bug

`docs/FILL-CODE-ANALYSIS.md` measures it: quantising 512x768 activations costs
0.145 ms, and 84 GEMM calls per prefill make that **12.1 ms, 1.0% of the run**.
Parallelising it would save half a percent, inside this host's noise — so the
serial fill loop is not a lever and was left alone. What *was* wrong: the two
auxiliary arrays (`xs` for Q4_K, `xs4` for Q6_K) were allocated and zero-filled
**for every format**, so Q8_0 paid 442 KiB of pointless memset per GEMM call.
They are now sized to zero where unused.

##### Against the engine spec

`docs/EXPERT-PATH-BUDGET.md` measures what expert management is actually worth.
On a 111.8 MiB `qwen3moe` fixture, decoding goes 110.1 -> 140.3 tok/s as the
cache improves from thrashing to perfect — **1.27x** — because expert reads are
39.5% of decode at 0% hit. At a 512 MB budget the cache already sits on the
compulsory-miss floor (98 misses, 98 distinct experts routed), which no
predictor can beat. The lever only becomes dominant when weights are on NVMe
rather than in page cache, i.e. exactly the small-VRAM case.

`docs/ENGINE-SPEC-GAP.md` maps `ENGINE_MASTER_SPEC.md` onto the code that is
actually on disk. Its headline: the spec's MoE gap list is out of date — the
hot-expert predictor (`moe_predictor.hpp`) and the byte-budgeted expert cache
(`expert_cache.cpp`, LFU with aging) both exist. The real gaps are paged KV +
continuous batching (§1.3), the two determinism-contract items in §1.2, and
Linux read-ahead, which is the cheapest unclaimed item in the document.

**What is actually left**: `objdump` shows the hot loop still spilling its eight
zmm accumulators to the stack — `vmovaps 0xc0(%rsp),%zmm13` … `vfmadd231ps` …
`vmovaps %zmm13,0x440(%rsp)` — 60 stack moves in a ~50-instruction body. At 5.6
instructions per 64 MACs and 2.2 ops/cycle it should reach ~80 GMAC/s per core
and reaches 24.4. Fewer rows in flight does not fix it (2 rows is 14.3 vs 24.4),
so the fix is to stop the allocator spilling, not to use fewer registers: build
the scale as two 256-bit `vbroadcastss`+`fmadd` instead of a `vinserti64x4` per
row, and hoist the eight row pointers. That is the next change, and it is
measured to be the right one rather than assumed.
[docs/GEMM-PERF-WALL.md](docs/GEMM-PERF-WALL.md).

`25%` of peak is still the number to argue with, and the arithmetic is no longer
what stops it. At 2048 tokens this kernel *asks for* roughly 155 GB of K/V loads
— every query row re-streams the keys before it — and this host reads 22.75 GB/s,
so the same work done from DRAM would cost 6.8 s against the 5.3 s measured. It
is a **load** problem now: the fix is to block query rows so one K row feeds
several rows of output, exactly the `rN` trick that took `gemm_q6k` from 4227 to
1697 µs/op. That is the next kernel, not another shuffle of this one.

#### Where the threads land

Every thread here is an OS thread, and until this change nothing told the OS
which core to put it on: through a whole GEMM a worker could sit on a slow core
or share one with another worker. On a hybrid part (Zen 4/5, Alder/Raptor Lake)
it is worse, because the fast cores are not enumerated first — a pool of
`hardware_concurrency()` threads spreads over the *slow* cluster and the whole
pool runs at the E-core clock.

`par_pool.hpp` now reads `cpufreq` for every CPU, orders them fastest-first, and
gives the calling thread the best core and worker *i* the (i+1)-th best, so a
small pool lands entirely on the performance cluster. The ranking is a stable
sort, so a uniform part gets the identity permutation and nothing moves. No
`cpufreq` (VM, container, no driver) means no order to impose and the previous
behaviour — `KRK_AFFINITY=1` forces placement anyway, `=0` disables it. The
backend prints what it did: `2 cores, unpinned` here, `8 cores, top 3.40 GHz,
4P+8E, pinned` on a hybrid laptop.

Measured on this box, which is 2 uniform cores and therefore the *no-op* case:
**1.00x** on decode over 20 interleaved runs. The win is on parts that have two
core classes; it cannot be shown here, and it is not claimed here.

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

### Nine families, one alias, and a latent-attention bug

A second pass added `glm4moe` (GLM-4.5 / GLM-4.7-Flash: glm4's attention with
routed experts, a sigmoid router carrying a per-expert selection bias, and a
shared expert of `n_ff_exp * n_expert_shared` columns), `phi3moe` (literally
phi3's graph with a `build_moe_ffn` where the dense MLP was) and `minicpm3`.

Only two of the three needed code, and both are *defaults for a file that says
nothing*: glm4-moe's files default to a **sigmoid** router where this engine's
generic fallback is softmax — on 128 experts that is a different selection, not
a different weighting — and its reference leaves `expert_weights_norm` false
where the engine's historical default renormalizes. `minicpm3` needed its depth
scale: every layer multiplies the ATTENTION branch's output by
`1.4 / sqrt(n_layer)` before the residual add, a constant in the reference with
no tensor and no metadata key behind it.

Building the MiniCPM3 fixture found a real defect in code that already shipped.
The lite latent-attention forward projected the query **only** through
`attn_q`; a file with a `q_lora_rank` stores `attn_q_a → attn_q_a_norm →
attn_q_b` instead. The loader read and shape-checked that trunk, the forward
never asked for it, and the projection ran against an absent tensor to produce
queries at 2.8e22 that turned into NaN one layer later — with nothing reported
in between. That was not only MiniCPM3: every V3-class DeepSeek file carries a
trunk, so the `deepseek2` row's "runs" claim covered a shape nothing had
executed end to end. Both halves are fixed now, including the workspace that was
allocated only for the shared layout — which turned the first fix into a
segfault until it was noticed.

`test_minicpm3_whole_graph` pins all three facts: the file loads, the resolved
scale is `1.4/sqrt(n_layer)`, and the same bytes under a label without the scale
produce different logits. `kraken-tests` 5676/5676.

### llama's schema with a piece swapped: six families, and one alias

Six architectures that were refused are llama's own tensor set with one or two
parts exchanged, which is why they cost a flag each rather than a graph:
`chatglm` (one fused `ffn_up` whose row is gate | up), `glm4` (that MLP plus a
norm on each branch's output before the residual), `mellum` (qwen3-MoE with a
4-layer sliding-window pattern and softmax gating), `phimoe` (Phi-3's block with
routed experts and `output.bias` on the logits), `minicpm` (partial rotary) and
`qwen2vl` (the text tower of the MiniCPM-V family, with q/k/v biases).

Two of them needed something real. `glm4` reuses the gemma post-branch norm,
which is a different function from norming the residual sum. `phimoe` adds a
per-vocabulary bias to the logits, and the device top-k reads the logits buffer
without ever touching the host copy — so the bias has to land in the buffer.
That is now `Engine::logit_head()`, the single place the vocabulary head is
formed; six call sites used to build it inline, and a bias added at one would
have been missing at the other five.

The seventh, **`qwen38`, is an alias and not a schema.** llama.cpp has no such
architecture; `Qwen3.8-Flash-Next` converts as **`qwen4exp`**
(`conversion/qwen4exp.py` registers `Qwen4ExpForCausalLM` with that exact
`.example()`). But the name cannot pick the schema either, because the family
has a sparse Flash form — hyper-connections, a 4-lane residual, a PLE n-gram
table, a QSA indexer, a 288-expert MoE — and a dense one with none of it. So the
alias probes the file: any `hc_*_inject` / `per_layer_token_embd` / `ssm_conv1d` /
`indexer.q_norm` tensor and it is read as `qwen4exp`; otherwise as qwen3
geometry. The verdict prints which way it went.

Writing the test for that found a live bug. `Engine::init` sized the query
projection with `arch_spec(mc.arch).q_output_gate` — asking the table **by
name** — while the loader had already bound the packed row from the resolved
spec. For a file whose name is an alias the two disagree, and the engine decoded
a different model with no error anywhere. The flag now travels on `ModelConfig`,
copied from the resolved spec. The general rule: **the engine must never
re-derive geometry from `general.architecture`**, because after alias resolution
or a format fallback the name is no longer the authority.

Seven fixtures, all seven run; `minicpm` reproduces `chatglm`'s logits with no
fusing at all, and `--split-ffn` reproduces them from split tensors, which is a
third independent check on the same equivalence. `--out-bias` was added to the
fixture generator to prove the Phi-MoE bias is read — and on the first attempt it
proved it was **not**, because the load sat inside `if (cfg_.ln_bias)` and Phi-MoE
is an RMSNorm family that never enters that branch. `kraken-tests` 5500/5500.
See [docs/LLAMA-SCHEMA-VARIANTS.md](docs/LLAMA-SCHEMA-VARIANTS.md).

### Fused projections: Phi-3 / Phi-4

`phi3` is now a supported architecture. Phi-3 and Phi-4 store two fused
projections — `attn_qkv` (`Q | K | V`, with K/V `n_head_kv*head_dim` wide under
GQA) and `ffn_up` (`gate | up`, no `ffn_gate` tensor) — and the engine reads both
as **row views** of one weight, so each runs the same gemm the split tensors
would. `fused_qkv` and `fused_gate_up` are new `ArchSpec` flags, separate from
`gpt2_style` because Phi needs the fused shapes with RMSNorm, partial rotary and
SwiGLU rather than GPT-2's bundle.

Verified bit-exact: a split-tensor model and the same weights laid out fused give
identical logits (`58=1.20879912` both), for QKV alone, gate|up alone, and both.
See [docs/PHI3-FUSED-PROJECTIONS.md](docs/PHI3-FUSED-PROJECTIONS.md), which also
records the two gating bugs the equality test caught — attention and FFN both
skipped silently on a fused file.

### Ternary int8 dot kernels (BitNet, Ternary-Bonsai)

`TQ1_0` (BitNet, 256 values / 54 B) and `PTQ1_0` (Ternary-Bonsai, 128 values /
28 B) now have int8 dot kernels instead of dequantising. Values are
{-1, 0, +1} behind one f16 scale, so the dot has no bias term and no multiply in
the inner loop at all — `_mm256_sign_epi8(x, t)` *is* `x*t` for a trit, and
`maddubs`+`madd` reduce 32 products to eight i32 lanes. The trit expansion calls
the same `tq1_trit_at`/`ptq1_trit_at` the dequantiser uses, so the kernel cannot
drift from the reference.

Measured on a 23.4 MiB PTQ1_0 model, three interleaved reps: **26.8 vs 22.4
tok/s prefill (1.20×)**. Verified against the dequantising path on identical
bytes — same ranking, values agreeing to the fourth decimal.
See [docs/TERNARY-KERNELS.md](docs/TERNARY-KERNELS.md), which also lists the
formats still missing kernels and why each is the same job again.

### Q2_K and Q3_K int8 dot kernels

Six of the thirty support-list files are Q2_K or Q3_K; neither had a dot kernel.
Both are Q6_K's block shape (256 values, sixteen 16-value sub-blocks, one scale
each), so the kernels are ports — Q3_K's signed `[-4,3]` values go through
`abs`/`sign_epi8` because `sign_epi8` is only exact for `-1/0/+1`, and Q2_K
takes its two per-sub-block sums inline instead of reading a sidecar.

Two measured mistakes are recorded in [docs/KQUANT-KERNELS.md](docs/KQUANT-KERNELS.md):
reducing vector lanes through memory made the kernel **slower than
dequantising** (24.1 vs 36.1 tok/s), and scalar unpacking left Q3_K a non-win
(23.8 vs 28.0). Vectorised bit-plane extraction — legal because the hmask bit
index is uniform per 16-value group — turned Q3_K into the biggest ratio here.

Verified against the reference decoder on the *same* activation by
`kraken-kquant-dot`: worst error 4.1e-08 of gross magnitude, `failure: none`.
Measured over six alternating reps, Q2_K **2.1×** and Q3_K **4.5×** median
(6/6 reps favour the kernel for both).

### The plain pre-norm family: gptneox, gptj, phi2

`gpt2_style` bundled four things as one architecture — LayerNorm-with-bias,
learned positions, a fused QKV and an ungated GELU MLP — because on GPT-2 they
*are* one architecture. They are not one architecture across the family, which
is why `gptneox` had nowhere to be added. Four flags now name the pieces
(`ln_bias`, `learned_pos`, `parallel_residual`, `no_gate_ffn`), `gpt2_spec()`
sets all four so GPT-2 is unchanged, and `Engine::plain_forward` covers all four
architectures.

`gptneox` reads the fused QKV **as row views** of one weight, so fused and split
layouts are bit-identical on the same weights — verified for all three, with
`parallel_residual` separately checked to change the graph rather than being
ignored. See [docs/PLAIN-PRENORM-FAMILY.md](docs/PLAIN-PRENORM-FAMILY.md), which
also records two loader bugs this found (`output_norm.bias` wrongly *required*,
so complete GPT-J files were refused; and `layer_norm_epsilon` never read).

### ALiBi, and the mpt and bloom families

`mpt` and `bloom` are the same pre-norm stack again with a different position
signal: **ALiBi**, a per-head multiple of the query-to-key distance subtracted
from the attention score. No rotary, no learned position. The slope anchor is
`floor(log2(n_head))` and the second group of heads uses *half* the max bias as
its base — two details that silently re-scale every penalty if guessed at, so
both are written out in `alibi_slope_impl()` and the slope is hoisted per head
rather than recomputed per key.

MPT states its scale in the file; BLOOM hardcodes 8.0 in its loader and reads no
key. Reading only the key leaves BLOOM with ALiBi **off**, which still decodes —
the failure mode this change exists to avoid. BLOOM additionally normalizes the
embedding before block 0. With ALiBi forced to 0, MPT reproduces `phi2`'s logits
digit for digit; with it on, they move — and BLOOM's graph provably equals MPT's
plus that one norm. See [docs/ALIBI-AND-MPT-BLOOM.md](docs/ALIBI-AND-MPT-BLOOM.md),
which also records three loader bugs this found (`position_embd.weight` wrongly
*required*, so complete MPT files were refused; per-head QK-norm tensors loaded
but never applied in the plain path, decoding silently wrong; and
`token_embd_norm` confused for a per-layer norm).

`falcon` joined them on the same change, and is *not* ALiBi: its loader reads
only the norm epsilon, so its position signal is the NEOX rotary. Its parallel
branch uses **two** norms — `x + attn(ln1(x)) + mlp(ln2(x))` — where GPT-NeoX
reuses one, so `parallel_two_norm` takes a second row of the *pre-attention*
residual. Proved by a file whose `attn_norm_2` carries `attn_norm`'s own
weights: the result must be, and is, identical to the last digit.

ALiBi is now covered by tests that recover the **slope ladder out of the softmax
probabilities** and compare it against the published formula
(`test_alibi_slope_ladder`, 12 heads so the anchor is 8 — not 12, not 16). That
test found a real bug in the change above: the bias had been added to the int8-KV
attention body only, so the f32 body every checkpoint uses ignored it, and the
model still decoded. The bias now lives in all three bodies, `Backend` has an
`attention_alibi_supported()` gate so a backend without it refuses the family by
name instead of decoding something else, and the virtual-NPU attention layer
declines such a call under its own `alibi` counter rather than dropping the term.
