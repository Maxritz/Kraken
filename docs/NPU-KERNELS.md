# NPU kernels, ISA dispatch and quantised weights

What the NPU can be made to do with its own kernels, what "use more CPU
extensions" is actually worth on this hardware, and what is left before a
27.5x is even arithmetically available.

Everything below is measured on the same box: 2 vCPU Xeon @2.60 GHz
(AVX512F/BW/DQ/VL/VNNI, 512-bit units), 3.9 GiB RAM, gcc 12.2, `npuxl` as the
virtual device, 1.2 GiB q8_0 MoE fixture, 58-token prompt.

---

## 1. Headline

| | before | after | |
|---|---|---|---|
| prefill, NPU arm | 2357 ms | **1828 ms** | 1.29x |
| prefill, NPU vs CPU arm | 1.73x | **2.85x** | |
| host weight memory staged | 2745 MiB | **0.0 MiB** | - |
| dequantization passes | whole model, once | **none** | - |
| GEMMs declined to the host | 200 | **0** | - |
| default-build q8_0 kernel | 7 GFLOPS (scalar) | **57 GFLOPS (AVX512)** | 8.1x |
| same binary, simulated Zen 3 | 7 GFLOPS (scalar) | **47 GFLOPS (AVX2)** | 6.7x |

Decode is unchanged at 164.5 ms/step against the CPU arm's 166.0 -- both are
bound by reading 1.1 GiB of experts per token, which is a storage problem, not
a kernel one.

Correctness: `kraken-tests` **4796/4796**, with a new gate that compares the
in-place q8_0 kernel against the CPU reference at rows 1, 3, 16 and 58. Greedy
continuations are byte-identical between `--cpu` and `--npu-cpu`.

---

## 2. Finding 1: the engine's AVX512 kernels were not in the binary at all

`src/kernels/kernels.cpp` opens with a comment claiming "the translation unit is
compiled with conservative baseline flags; GCC/Clang target attributes enable
AVX2/AVX512 code only inside the marked functions". That intent is right. The
implementation contradicted it:

```c
#if defined(__AVX512F__)          // <-- false in a baseline-flag build
VNPU_TARGET_AVX512
static void gemm_f32_avx512_nt_tile(...)
```

`__AVX512F__` is defined only when the *whole translation unit* is compiled for
AVX512. A default npuxl build passes no `-mavx512f`, so every AVX512 kernel was
deleted by the preprocessor, and `dispatch_tile`'s
`if (p.isa == ISA_AVX512)` branch had nothing left to call -- it fell through to
the AVX2 tile. The runtime CPU probe was correct all along
(`__builtin_cpu_supports`, requiring f+bw+dq+vl, so Zen 3 correctly reports
AVX2); there was simply no AVX512 code to dispatch *to*.

Verified by symbol table:

```
$ nm -C <as-shipped build>/kernels.cpp.o | grep avx512
      (nothing)
$ nm -C <build after the fix>/kernels.cpp.o | grep avx512
gemm_f32_avx512_nn_tile
gemm_f32_avx512_nt_tile
gemm_q80_avx512_tile
```

The fix replaces the compile-time question with the right one:

```c
#if VNPU_X86 && (defined(__GNUC__) || defined(__clang__))
#  define VNPU_HAVE_X86_TARGETS 1     /* the compiler CAN emit these under an
                                       * attribute -- unrelated to what this TU
                                       * was compiled for */
```

The guards now ask "can the compiler emit this ISA here", never "was this file
compiled for it". Which kernel runs stays a runtime decision in `dispatch_tile`.

### What that bought

| f32 GEMM | AVX2 build (as shipped) | AVX512 build | |
|---|---|---|---|
| 58x768x768 | 41.5 GFLOPS | 39.5 | -5% |
| 128x768x768 | 83.9 | 82.0 | -2% |
| 256x768x768 | 90.1 | 70.1 | **-22%** |
| 512x3072x768 | 46.7 | 49.0 | +5% |

**On the existing f32 kernels, 512-bit SIMD is worth nothing** -- and at
M=256 it is 22% *worse*. Those kernels are already bandwidth-bound (a 128x768x768
f32 GEMM moves 3.0 MB and does 150 MFLOP, i.e. 50 flop/byte, and the machine
sustains ~3 GB/s per core in this pattern), so widening the registers does not
move the bottleneck. This is the concrete answer to "make the NPU use more CPU
extensions": for f32 weights the extensions are not the limit.

Where ISA width *does* pay is a kernel that reads fewer bytes per flop. That is
the q8_0 kernel below, and there the 512-bit path is 1.2x the 256-bit one.

---

## 3. Finding 2: a quantised checkpoint was being expanded before the first token

The staging cache dequantized every non-f32 weight to f32 on first touch and
kept it. For a 1.2 GiB q8_0 model that is 4.4 GiB resident on a 3.9 GiB box, so
the budget bound, and the GEMMs that no longer fit fell back to the host: 200
declines in a 58-token prefill, device idle while the CPU ran expert matmuls.

### The new opcode: `VNPU_OP_MATMUL_Q8_0`

GGUF q8_0 is N rows of K weights, quantised 32 at a time into 34-byte blocks
(`f16 d; int8 qs[32]`). The new kernel consumes that array where it lies:

```
C[m,n] = sum_b d_b * ( sum_{i<32} A[m,32b+i] * q_b[i] )
```

Per block it widens 32 signed bytes into two 16-lane f32 vectors, does the FMAs,
and folds the block scale in once per block instead of once per weight. Rows are
blocked 4 at a time so four output rows share one reading of B.

Two things this buys: **4x less memory traffic** per weight byte (1 byte/weight
instead of 4), and **no conversion pass** -- the model is used exactly as it sits
in the mapped file, so RAM holds the checkpoint plus activations and nothing else.

| q8_0 GEMM | GFLOPS | notes |
|---|---|---|
| scalar fallback | 7 | the path a default build had before |
| AVX512 kernel | 48-57 | this machine |
| AVX2 kernel (simulated Zen 3) | 33-47 | `VNPU_FORCE_ISA=avx2` |
| f32 GEMM, same shapes | 25-90 | 4x the bytes |

Against the CPU's own dequantizing path on the same blocks, the device is 2.85x
on prefill and identical on decode (see headline table).

### The first version was 8x too slow, and the reason matters

The initial kernel reduced per block -- four horizontal sums per 32 weights.
A horizontal reduction is a serial chain of shuffles, so it dominated:

```
v1 (reduce per block):   M=58  9.45 ms   7.2 GFLOPS
v2 (running vector acc): M=58  1.84 ms  37.2 GFLOPS     5.1x
```

The fix is to scale the 32 weight lanes by `d_b` (two multiplies, shared by all
four rows) and keep a running vector accumulator, reducing to a scalar once per
output element. Same arithmetic -- scaling before the FMA is the same product as
scaling the block sum after it -- and it is what makes the kernel worth having.

---

## 4. Finding 3: the M tail is a scalar cliff, and it is in the prompt path

`gemm_f32_avx512_nn_tile` blocks 4 rows of M and sends anything left over to
`gemm_f32_scalar_tile`. Measured (`vnpu-cli bench`, f32, 768x768):

| M | ms | GFLOPS | us/row |
|---|---|---|---|
| 12 (aligned) | 0.4709 | 30.06 | 39.2 |
| 13 | 1.0503 | 14.60 | 80.8 |
| 14 | 1.2758 | 12.95 | 91.1 |
| 15 | 1.5944 | 11.10 | 106.3 |
| 16 (aligned) | 0.4236 | 44.56 | 26.5 |
| 58 | 2.6615 | 25.71 | 44.9 |
| 64 (aligned) | 1.5256 | 49.49 | 23.8 |

**M=14 costs 3.0x what M=16 costs while doing 12% less work.** Only `M % 4 == 0`
escapes the cliff. Prefill M is the token count of the chunk, and the routed
expert gather produces per-expert M in the 9-32 range, so this is the common
case, not an edge case.

A blocked-4 `nt` tile (which is the path Kraken actually uses, `trans_b=1`) was
written and measured. It is bit-identical by construction -- same k order, same
lane accumulation, same horizontal sum -- and FNV hashes match on every shape
tested. Interleaved A/B, 4 rounds, medians:

| M | old nt | new nt (blocked 4) | |
|---|---|---|---|
| 14 | 0.7487 | 0.4628 | 1.62x |
| 16 | 0.6285 | 0.5228 | 1.20x |
| 20 | 0.7116 | 0.6162 | 1.15x |
| 58 | 1.9823 | 1.4986 | 1.32x |
| 64 | 2.0132 | 1.6990 | 1.18x |

It is faster everywhere, but the run-to-run spread is 40-90% on this box and the
gain is inside a single sample of it. **It is not in the shipped patch**: the
numbers do not clear the bar the rest of this work is held to, and a kernel
change that cannot be measured reliably is not a kernel change yet. It stays on
the list with its measurements attached.

---

## 5. "More NPUs, one per core, synced as one larger device"

This is exactly the architecture npuxl already models: `vnpu_config_t` has
`num_threads` and `num_compute_units`, tiles are distributed over a thread pool,
and there is a real `vnpu_cluster` API with `vnpu_cluster_add_device`,
`vnpu_cluster_add_remote`, weight caching per worker and row-split matmul -- a
chiplet fabric, including remote nodes.

Measured scaling on this box (`vnpu-cli bench --threads N`, f32):

| threads | 16x768x768 | 128x768x768 | 256x768x768 |
|---|---|---|---|
| 1 | 31.2 GFLOPS | 44.5 | 51.4 |
| 2 | 42.0 | 70.8 | 85.7 |
| 4 | 30.2 | 81.2 | 94.1 |

Two threads give 1.35-1.67x, not 2x. Four threads on two physical cores add
about 10% over two. The limiting factor is shared -- one memory bus, one L3 --
and a GEMM at these sizes is a streaming operation, so adding compute units adds
contention rather than throughput.

The honest conclusion: **more NPUs help in proportion to how much memory
bandwidth they bring with them.** On one socket with one bus they buy 1.4-1.7x
for 2x the units. A real chiplet split pays when each chiplet has its own
memory controller and the working set is sharded to match -- that is a topology
question, and `vnpu_cluster` is the right vehicle to test it on, but nothing in
these measurements supports 27.5x from unit count alone.

---

## 6. Where 27.5x could come from, and where it cannot

The claim was 27.5x over direct CPU use. Measured today: **2.85x** prefill,
1.00x decode, and 8.1x for the best single kernel against the scalar path it
replaced. A 27.5x needs several independent factors multiplying:

| lever | measured here | to reach 27.5x total |
|---|---|---|
| quantised weights in place (4x less traffic) | 2.85x | required |
| int8 act x weight (VNNI) | 151.9 GFLOPS vs 57 f32 | would help, needs a paid-off pack |
| fewer host round trips | sync wait is 36% of prefill | overlap, not speed |
| more units | 1.35-1.67x for 2 | not 27.5x |
| sparse / expert-skipping | not measured | unknown |

27.5x = 2.85 x 9.6. The remaining 9.6 has to come from somewhere the device
currently spends time on a critical path -- and today the device is idle 67% of
a prefill, so the next factor is more likely to come from **removing host work
and sync waits** than from any kernel. The int8 path is the one lever with
measured headroom (385 ms of device work against 829 ms for f32 at the same
shapes), and it needs the 2.4 s weight pack amortised across runs to be viable
at all.

That is the honest accounting: 27.5x is not refuted, but it is not a kernel
tuning exercise and it is not reachable by ISA width. It needs a different
memory hierarchy, or a workload where the weights stream once and the flops
amortise -- which is exactly what a long prefill is, and exactly what these
measurements should be repeated on with a real checkpoint rather than a 1.2 GiB
fixture.

---

## 7. Quantisation roadmap: what is native, what falls back

| format | device path today | note |
|---|---|---|
| q8_0 | **native, in place** | `VNPU_OP_MATMUL_Q8_0`; no dequant, no staging |
| f32 | native | direct |
| f16 | dequantized into staging cache | one f32 copy per weight, bounded by budget |
| q4_0, q5_0, q5_1, q4_k, q5_k, q6_k, iq*, nvfp4, mxfp4 | dequantized into staging cache | same path as f16; `dtype_row_bytes`/`dequant_row` cover them |

The natural next step by the same argument as q8_0 is **q4_0 / MXFP4 in place**:
4-bit weights halve the traffic again, and the block-widening trick is the same
(`iq4_nl` needs a 16-entry lookup table, MXFP4 an E8M0 scale -- both are
per-block constants, so both fit the same kernel skeleton). That is where the
next measured factor has to come from.

On KV reduction the repo already carries an int8 KV path with its own tests
(`test_kv_quant_int8`) and a tiered KV store with per-plane cold files. Turbo
quant and LMCache are not in this tree; both would land as another `DType` on the
same two seams -- `dtype_row_bytes`/`dequant_row` for storage, and a device
kernel for the read path. The q8_0 work above is the template: a block format is
a constant plus a payload, and a kernel that understands both reads it in place.

**Coherence note.** The 1.2 GiB fixture is seeded PRNG weights, so its decoded
text is gibberish *by construction* -- that is not a property of any backend.
Coherence is therefore enforced the only way it can be enforced without a trained
checkpoint: every quant path is compared against the CPU reference on the same
bytes (new gate: q8_0 in place at rows 1/3/16/58), and greedy continuations are
compared byte for byte between `--cpu` and `--npu-cpu`.

---

## 8. Reproducing

```sh
# the ISA-dispatch fix, shown by symbol table
nm -C <npuxl build>/CMakeFiles/vnpu_shared.dir/src/kernels/kernels.cpp.o | grep avx512

# one binary, both paths
VNPU_FORCE_ISA=avx2 ./bench    # Zen 3 fallback
./bench                        # native

# prefill, in place, no staging
./build-npu/kraken -m fix/moe-2g-q8_0.gguf --npu-cpu -p "<58 tok>" -n 0 \
    --greedy --ctx 1024 --chunk 512

# coherence: must be empty output
diff <(./build-npu/kraken ... --cpu ...) <(./build-npu/kraken ... --npu-cpu ...)
```

Engine patch: `patches/npuxl-0004-q8-0-in-place-and-isa-dispatch.patch`.
