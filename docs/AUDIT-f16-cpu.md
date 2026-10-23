# f16-storage CPU backend: survey and plan

The CPU backend is f32 throughout (`act_size() == 4`, `src/backend_cpu.cpp:87`)
while the device stores every activation in f16 (`act_size() == 2`). Coherence
today is therefore behavioral (same text), never equality (same bytes). The
`vec_dot` decision stands — the CPU dot order is its own — but a CPU backend
that rounds every intermediate to f16 exactly as the device does would turn
coherence into an equality proof: same prefill hash, same bytes, no tolerance.

Precedent: `src/expert_cpu.cpp:62` already has the discipline in miniature.
`r16(v) = fp16_to_fp32(fp32_to_fp16(v))` — "the device stores the gate as f16"
— applied to the gate, the up projection, their silu product, and the
accumulator (`expert_cpu.cpp:220-236`). The plan below copies that outward.

## Survey: where each op stands relative to the device rounding point

"Order" = accumulation/association order. "Store" = what lands in the buffer.
Device stores always go through `d_sat_f16` / `d_sat_f16_count`
(`src/hip/krk_hip.hpp:155-193`): round-to-nearest-even, NaN → 0, |v| > 65504 →
±65504, with a saturation counter.

| Op | CPU today | Device | Gap to equality |
|---|---|---|---|
| `hc_mix` | `krk::hc_mix`, f32, lanes 0..hc-1, one divide at end | same order, f32 accum, saturating f16 store | **store rounding only** — closest op in the engine |
| `hc_combine` | `krk::hc_combine`, f32, per (c,e) `res += b*w` | one thread per (t,e), loop over c, saturating store | store rounding only (same c order) |
| `silu_mul` / fused gate-up | `x/(1+exp(-x))*u`, f32 | same formula, saturating f16 store | store rounding only |
| residual `dst += Wx`, scale, bias, `silu_act` | f32 elementwise | same order, saturating store | store rounding only |
| `gemm` / `gemv` | `vec_dot`, f32, serial i order, f32 store | dequant→f16, pairwise `d_dot2` f32 accum, f16 store | **order + store**; the vec_dot order is load-bearing, keep it and pin the delta |
| `attention` (decode) | two-pass (max, then sum), serial j order, f32 store | 8-warp online max per key shard + merge (split) or per-key online (`bexact`), f16 store | **order + store**; the bexact kernel is decode's arithmetic, so the CPU twin targets bexact's order, not the split's |
| `rmsnorm` / `hc_norm` | f32 (hc_norm uses f64 `ss`) | f32 accum, saturating store | accum precision + store |
| `rope` | `rope_apply`, f32 libm sin/cos | device sin/cos intrinsics | **libm identity risk** — host and device transcendentals are not guaranteed bit-equal; survey first, equality may stop here |
| `dequant` (weights→act) | to f32 inside `vec_dot` | to f16 (`dequant_chunk`) | value rounding; feeds the gemm order gap |
| `topk` / sample | f32 logits | f16 logits | input rounding; equality needs the same input bytes first |
| `fp32_to_fp16` vs `d_sat_f16` | NaN→canonical NaN, Inf→Inf (`src/common.cpp:167`) | NaN→0, Inf→±65504 | **specified mismatch** — the CPU twin must replicate `d_sat`, not `fp32_to_fp16` |

Two facts make this tractable: (1) the elementwise half of the engine
(hc, silu, residual, scale, bias) already shares the device's association
order and needs only the store rounded; (2) `fp32_to_fp16` is
round-to-nearest-even with ties-to-even, which is what the hardware `cvt`
does, so finite values already agree and only NaN/Inf handling needs a twin.

## Plan

- **Stage 0 — `d_sat` twin.** Host `f16_sat_store(f32) -> u16`: RNE + NaN→0 +
  saturate ±65504. Unit-test against `fp32_to_fp16` on finite values (must
  agree bit-for-bit) plus the specified NaN/Inf cases (must differ exactly as
  specified). No backend changes.
- **Stage 1 — f16 CPU backend behind `KRK_CPU_F16=1`.** `act_size() == 2`,
  u16 buffers, every store through the Stage-0 twin, compute in f32. Do not
  retrofit the existing backend in place: a second class reusing the `krk::`
  host maths keeps the default path untouched. Order: elementwise ops first
  (equality expected immediately), then gemm with vec_dot order kept (delta
  pinned, not closed), then attention in bexact order, rope last with a
  decision point if libm will not agree.
- **Stage 2 — per-op device-vs-twin arms.** The suite cannot open the device,
  so each op gets a `kraken-bench` comparison arm (the `--hc` 10/10 pattern:
  BIT-identical where the survey says order matches, tolerance where it says
  it does not). Each arm needs its negative control: a deliberately
  mis-ordered twin that must FAIL, or the arm proves nothing.
- **Stage 3 — coherence as equality.** `coherence_check.sh` gains a byte
  arm: device prefill hash vs `KRK_CPU_F16=1` prefill hash on the fixture,
  then full chunk-invariance + prefill-then-decode text equality (the gate
  in `test_prefill_chunk_bitexact`) with the CPU backend as one more arm.
  Text equality today, hash equality when the last order gap closes.

Non-goal: speed. The f16 backend is a reference, not a rung; it may be slower
than the f32 one (every store pays a convert). Its job is to make the next
fast kernel prove invariance instead of asserting it.
