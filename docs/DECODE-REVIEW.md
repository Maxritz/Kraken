# Decode and token generation: where the time actually goes

Measured 2026-10-28 on the sandbox box: 2 hardware threads, AVX2 (no AVX-512
build here), release build, CPU backend, three real trained checkpoints.
Everything below is a measurement unless it is explicitly marked *code read*.

The question this answers: **decode is slow on every backend, including the
GPU. Is it the device, or is it the code around the device?**

## Verdict

On the CPU, decode is **kernel-bound and nothing else**. The effective FLOP
rate is the same for a dense model and a 2-expert MoE (6.5 vs 5.1 GFLOPS),
which means routing, sampling, logits fetch and dispatch are all noise. The
K-quant dot product is the entire story, and it is the same kernel that makes
prefill slow.

On the GPU, nothing here was measured -- this sandbox has no AMD device. The
structural evidence points at the *number of operations per token* rather than
at any single expensive call, and the two specific suspects are named at the
end with the check that settles each.

## The baseline

3 real models, greedy, 24 tokens, `--cpu`:

| model | params | file | prefill | decode | decode GFLOPS | decode GB/s |
|---|---:|---:|---:|---:|---:|---:|
| Tinystories-gpt-0.1-3m | 3 M | 7.7 MB | 3328 tok/s | 895 tok/s | 5.4 | 6.9 |
| tinyllama-15M-stories Q3_K_M | 15 M | 14 MB | 653 tok/s | 206 tok/s | 6.5 | 2.9 |
| Tiny-Moe Q4_K_M | 141 M | 90 MB | 49 tok/s | 22 tok/s | 5.1 | 2.0 |

A 2-thread AVX2 f32 host peaks near 90 GFLOPS. These runs are at 6-7% of it.

## Where the time is NOT

Each of these was measured, not assumed, on Tiny-Moe (12 layers, 2 experts,
top-2) with `--profile`:

| what | measured | share of a 55 ms step |
|---|---|---|
| `decode.sample` (softmax, top-k, draw) | 0.010 ms/token | 0.02% |
| `decode.fetch` (logits to host) | 0.005 ms/token | 0.01% |
| `decode.logits.dev` (device argmax) | 0.004 ms/token | 0.01% |
| `moe.route.gemm` (the routing matrix) | 0.001 ms/layer | 0.02% |
| `tokenizer.decode` | 0.001 ms/token | 0.002% |

The token loop itself is clean: sampling, detokenisation and the logits
round-trip together are under 0.1% of a step. The default path already takes
the argmax on the device (`--sample-host` is the switch that costs a 1 MB
download per token), so that trap is not being fallen into.

The MoE dispatch is also not the problem, which is worth stating because it
looks like it should be: `moe.route.rest` self-time is 36 ms/step, and the
expert GEMMs alone are 282 MFLOP, so that span is running at 7.8 GFLOPS --
*above* the dense model's 6.5. There is no meaningful dispatch overhead hiding
in there.

Two `be_->sync()` calls appear in the layer body region. Both are in
`mtp_ffn()` and the MTP propose pass, not in the main forward. *Code read.*

## Where the time IS

Both prefill and decode run the same `vec_dot()` family, and only two formats
have a hand-written SIMD kernel:

```
vec_dot_tiled()                     src/quant.cpp:1628
  Q8_0, Q4_0  -> dot_q8_0_tiled / dot_q4_0_tiled   (AVX2, fused)
  everything else -> dot_generic_tiled()            src/quant.cpp:1543
```

`dot_generic_tiled` does this per block:

```cpp
dequant_block(t, p, buf);                       // widen 256 weights to f32
acc[j] = _mm_fmadd_ps(_mm_loadu_ps(buf + k),    // 128-bit: 4 f32 lanes
                      _mm_loadu_ps(xr + k), acc[j]);
```

That is dequantise-then-multiply at 128-bit width. A 4-lane FMA chain has about
a 4-cycle latency, so ~2 flops/cycle/core, ~2 cores at ~3 GHz = **~12 GFLOPS**.
Measured: 13.8-20 GFLOPS prefill. The arithmetic predicts the measurement, so
there is no second effect hiding behind it.

Every real model here is a K-quant. The formats that have kernels are the ones
real models do not use.

| format | kernel | measured |
|---|---|---|
| q8_0 | hand-written AVX2/AVX-512 | 47-57 GFLOPS |
| Q3_K, Q4_K | generic dequant + 4-lane f32 | 13.8-20 GFLOPS |

Decode is the same kernel at `rows == 1`, which is why fixing prefill fixes
decode: there is no separate decode defect to chase.

## The GPU question

Not measured here. Two structural facts, both from the code:

1. **Operation count per token.** A dense layer body issues roughly 15-20
   backend calls (rmsnorm, gemm, rope, kv_append, attention, silu_mul,
   add_residual, copy_act, ...). At 12 layers that is on the order of 200
   backend operations per generated token. On a CPU each is a virtual call and
   costs nothing; on the GPU each is a kernel launch and a queue entry. At 5-10
   us of launch cost that is **1-2.5 ms per token of pure issue**, which caps a
   fast device at a few hundred tok/s no matter how fast its GEMMs are. This is
   the most likely reason a fast GPU feels slow, and it is a property of the
   processing code, not the device.
2. **The device is idle between those launches.** Everything except the GEMM,
   RMSNorm, SwiGLU, add and mul runs as a separate small op (the `--npu` vs
   `--npu-cpu` split in the README is exactly this boundary). A 64-wide norm or
   a 4096-element add is launch-bound, not bandwidth-bound.

What has *not* been shown: that any single call on the GPU is slow. The claim
here is aggregate -- too many small ops, not one bad one.

To settle it on a real device, in this order:

1. `--trace` a 32-token decode on the RX 9070 XT and read the per-op table:
   if `dev ms` is small but the sum of gaps between ops is large, it is issue
   cost and (1) is the answer.
2. `KRK_NPU_MIN_ELEMS=N` already exists to keep small ops on the host. Raising
   it until throughput improves is a direct measurement of how much the small
   ops cost.
3. If issue cost dominates, the fix is fewer, wider ops: fuse
   rmsnorm+gemm, gemm+silu_mul+gemm, and kv_append+attention into single
   submissions. That is the "grouped instructions" idea from the earlier
   design discussion, and it applies to HIP exactly as it does to the VNPU.

## Fix order

1. **AVX2/AVX-512 K-quant dots** (`Q4_K`, `Q6_K`, `Q3_K`, `Q2_K`): quantise the
   activation row to int8 once, integer dot (`_mm256_maddubs_epi16` +
   `_mm256_madd_epi16`), apply `d`/`dmin` at the end. Weights stay in their
   stored blocks, which is the standing rule. Expected 4-8x on both prefill and
   decode; the 47-57 GFLOPS q8_0 kernel is the local proof of the ceiling.
   Gate: same tokens as `dot_generic_tiled` on all three real models.
2. **Reduce op count per token** for the GPU, with the trace above as the
   instrument rather than a guess.
3. Only after those: tiling/fusion on the VNPU, which was 2.85x on a q8_0 MoE
   where the kernels already existed.
