// ============================================================================
//  krk_hip.hpp — HIP glue for the RDNA kernels.
//
//  Architecture gating is driven by the clang-defined __gfx*__ macros so a
//  single source tree can be offloaded to several targets
//  (--offload-arch=gfx1031 --offload-arch=gfx1201 ...). amdclang defines
//  those macros only in the *device* pass of an offload compile, so the
//  build also mirrors every KRK_GPU_TARGETS entry into a
//  -DKRK_OFFLOAD_GFX* definition that both passes see: the host pass has
//  to compile the same kernel launch sites, because a kernel that is
//  #if'ed out of the host pass never enters the fat binary and its launch
//  silently does nothing. Each pass sees exactly one of
//  KRK_GFX10 / KRK_GFX11 / KRK_GFX12.
//
//  Wave size: every RDNA target here is wave32. WMMA builtins on gfx11/gfx12
//  are wave32-only, so the kernels never assume wave64 and never use
//  __shfl_*_sync masks that would break under wave32.
// ============================================================================
#ifndef KRK_HIP_HPP
#define KRK_HIP_HPP

#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "krk/backend.hpp"
#include "krk/common.hpp"

// ---------------------------------------------------------------------------
// gfx family detection
// ---------------------------------------------------------------------------

// The __gfx*__ macros exist only in the device pass; the KRK_OFFLOAD_GFX*
// macros (one -D per KRK_GPU_TARGETS entry, emitted by CMake for the whole
// compile command) exist in both. Fold the two sources together so the host
// and device passes select the same family.
#if defined(__gfx1200__) || defined(__gfx1201__) || \
    defined(KRK_OFFLOAD_GFX1200) || defined(KRK_OFFLOAD_GFX1201)
#define KRK_GFX12 1
#elif (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) ||  \
       defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) ||  \
       defined(__gfx1152__) || defined(__gfx1153__)) ||                          \
      (defined(KRK_OFFLOAD_GFX1100) || defined(KRK_OFFLOAD_GFX1101) ||          \
       defined(KRK_OFFLOAD_GFX1102) || defined(KRK_OFFLOAD_GFX1103) ||          \
       defined(KRK_OFFLOAD_GFX1150) || defined(KRK_OFFLOAD_GFX1151) ||          \
       defined(KRK_OFFLOAD_GFX1152) || defined(KRK_OFFLOAD_GFX1153))
#define KRK_GFX11 1
#elif (defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) ||  \
       defined(__gfx1010__) || defined(__gfx1012__)) ||                          \
      (defined(KRK_OFFLOAD_GFX1030) || defined(KRK_OFFLOAD_GFX1031) ||          \
       defined(KRK_OFFLOAD_GFX1032) || defined(KRK_OFFLOAD_GFX1010) ||          \
       defined(KRK_OFFLOAD_GFX1012))
#define KRK_GFX10 1
#else
// An unsupported arch. The serviceable fallback is the portable SIMT kernel.
#define KRK_GFX_FALLBACK 1
#endif

// v_dot2_f32_f16 exists on gfx10.3+, gfx11 and gfx12. The RDNA4 ISA reference
// lists V_DOT2_F32_F16 (and V_DOT2_F32_BF16 / V_DOT4_* / V_DOT8_*) in the VOP3P
// packed-math table, and probe_min measures it computing the expected
// 1.5*2.0 + 2.5*4.0 = 13.0 on gfx1201 — so the *instruction* is available on
// every target this tree builds; WMMA is an additional path, not a replacement.
//
// Where the packed form is *used* is decided by measurement, per above:
// availability on gfx12, selection only where it beats the scalar pair.
#if defined(KRK_GFX10) || defined(KRK_GFX11) || defined(KRK_GFX12)
#define KRK_HAS_DOT2 1
#endif

// Whether a WMMA GEMM kernel exists in this binary at all. The family macros
// above answer "which device pass is this", which is the wrong question for a
// dispatch decision: a fat binary may hold several targets, and the device that
// actually runs is chosen at runtime. The offload macros say what code objects
// the binary carries, so they are what decides this, and the *runtime* device
// then decides whether to use the rung.
#if defined(KRK_GFX11) || defined(KRK_GFX12)
#define KRK_WMMA_COMPILED 1
#endif

#if defined(KRK_GFX10) || defined(KRK_GFX11)
#define KRK_SIMT_USE_DOT2 1
#endif

namespace krk {

// ---------------------------------------------------------------------------
// error handling
// ---------------------------------------------------------------------------

const char *hip_err_name(hipError_t e);
void hip_check(hipError_t e, const char *what);

#define KRK_HIP(expr) ::krk::hip_check((expr), #expr)
#define KRK_HIP_TRY(expr, okvar)                                                   \
    do {                                                                           \
        const hipError_t _e = (expr);                                              \
        if (_e != hipSuccess) {                                                    \
            (okvar) = false;                                                       \
            KRK_ERROR("%s failed: %s", #expr, ::krk::hip_err_name(_e));            \
        }                                                                          \
    } while (0)

// ---------------------------------------------------------------------------
// device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ f32 d_h2f(u16 h) {
    return __half2float(*reinterpret_cast<const __half *>(&h));
}

__device__ __forceinline__ u16 d_f2h(f32 f) {
    const __half h = __float2half(f);
    return *reinterpret_cast<const u16 *>(&h);
}

__device__ __forceinline__ f32 d_silu(f32 x) { return x / (1.0f + __expf(-x)); }

__device__ __forceinline__ f32 d_rsqrt(f32 x) { return rsqrtf(x); }

// ---------------------------------------------------------------------------
// packed-math primitive: v_dot2_f32_f16, d = a0*b0 + a1*b1 + c
//
// Two fp16 MACs into an fp32 accumulator at the packed rate. RDNA2 added the
// dot-product ALU ops specifically "to accelerate inferencing", the RDNA4 ISA
// reference still lists V_DOT2_F32_F16 in the VOP3P packed-math table, and
// probe_min measures it computing 1.5*2.0 + 2.5*4.0 = 13.0 on gfx1201 — so
// every target in this tree may use it (see KRK_HAS_DOT2 above).
//
// Availability is not the same as a win, though: in the compute-dense SIMT
// tile loop the instruction measured 2.8x *slower* than the scalar pair on
// gfx1201 (gemm_simt rows=51: 68 us scalar vs 195 us packed, probe_decode
// §2b), because that loop issues 16 dot2 per k-pair per thread and RDNA4 does
// not retire it at the packed rate RDNA2 does. KRK_SIMT_USE_DOT2 is therefore
// gfx10/gfx11 only, while the latency-bound decode attention kernel — one
// packed pair per lane per key, no throughput cliff — keeps the packed form.
//
// The u32 form takes the half pair already in a register, which is what the
// decode attention kernel needs: pointing at a register-resident _Float16
// array would force it to local memory (the generic-pointer reinterpret has to
// be addressable). The pointer form is for operands that really live in LDS or
// global memory, as in the SIMT tile GEMM.
// ---------------------------------------------------------------------------
#if defined(KRK_HAS_DOT2)
__device__ __forceinline__ f32 d_dot2_pk(u32 ha, u32 hb, f32 c) {
    f32 d;
    __asm__ volatile("v_dot2_f32_f16 %0, %1, %2, %3"
                     : "=v"(d)
                     : "v"(ha), "v"(hb), "v"(c));
    return d;
}
#else
__device__ __forceinline__ f32 d_dot2_pk(u32 ha, u32 hb, f32 c) {
    const _Float16 *ap = reinterpret_cast<const _Float16 *>(&ha);
    const _Float16 *bp = reinterpret_cast<const _Float16 *>(&hb);
    return c + static_cast<f32>(ap[0]) * static_cast<f32>(bp[0]) +
           static_cast<f32>(ap[1]) * static_cast<f32>(bp[1]);
}
#endif

__device__ __forceinline__ f32 d_dot2(const _Float16 *a, const _Float16 *b, f32 c) {
    return d_dot2_pk(*reinterpret_cast<const u32 *>(a),
                     *reinterpret_cast<const u32 *>(b), c);
}

// ---------------------------------------------------------------------------
// int8 packed dot: v_dot4c_i32_i8, d = a0*b0 + a1*b1 + a2*b2 + a3*b3 + c
//
// Four int8 MACs per lane into an int32 accumulator, and LANE-LOCAL: section 1
// of tools/probe_gemm_tile.hip settles that by running the instruction rather
// than assuming it (all 32 lanes agree when they are given the same operands,
// and a per-lane operand ramp returns each lane's own four bytes). That is what
// makes a plain row-major int8 tile usable: four consecutive k values per 32-bit
// register, each thread owning a disjoint output tile, no shuffles.
//
// The clamp flag is 0 (exact, no saturation) — the int8 GEMM harness verified
// its results against a CPU reference with this form. RDNA2 measures the
// register-only rate at 52.08 TOP/s, 93% of the 256 MAC/clk/CU issue rate.
//
// The intrinsic is only *used* by the opt-in int8 path (kernels/gemm_dp4a.hpp);
// the scalar form keeps this header compilable on a target without the
// instruction.
//
// It is gfx10/gfx11 only, and that is a compiler gate rather than a hardware
// one: clang rejects the builtin for gfx1201 with "needs target feature
// dot1-insts", which selecting the arch does not enable. Writing the intrinsic
// behind the wider guard made the whole HIP target fail to compile on the
// gfx1201 machine after the int8 port landed -- nothing here had compiled that
// path for gfx12, so the break went unnoticed until it was built there.
#if defined(KRK_GFX10) || defined(KRK_GFX11)
#define KRK_HAS_DOT4 1
#endif

#if defined(KRK_HAS_DOT4)
__device__ __forceinline__ i32 d_dot4(i32 a, i32 b, i32 c) {
    return __builtin_amdgcn_sdot4(a, b, c, 0);
}
#else
// Four signed byte products, exact, with the bytes taken by shifting rather
// than through a byte pointer: the pointer form took the address of its
// parameters, which forces them into scratch on a target that cannot issue the
// instruction. Arithmetic right shift of a signed value is what sign-extends
// each byte here.
__device__ __forceinline__ i32 d_dot4(i32 a, i32 b, i32 c) {
    i32 r = c;
#pragma unroll
    for (int s = 0; s < 4; s++) {
        const i32 av = (a << (24 - 8 * s)) >> 24;
        const i32 bv = (b << (24 - 8 * s)) >> 24;
        r += av * bv;
    }
    return r;
}
#endif

// Wave primitives. RDNA1/2/3/4 are all wave32; the width is fixed here rather
// than probed so the reductions unroll to exactly five shuffle steps.
constexpr int kWaveSize = 32;

__device__ __forceinline__ f32 d_wave_reduce_sum(f32 v) {
#pragma unroll
    for (int off = kWaveSize / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
    return v;
}

__device__ __forceinline__ f32 d_wave_reduce_max(f32 v) {
#pragma unroll
    for (int off = kWaveSize / 2; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor(v, off));
    return v;
}

// WMMA fragment vector types. These are the exact types the AMDGPU builtins
// expect; the GPUOpen WMMA guides use the same aliases.
#if defined(__HIP__) && (defined(KRK_GFX11) || defined(KRK_GFX12))
#define KRK_HAVE_WMMA_TYPES 1
using v16half = _Float16 __attribute__((ext_vector_type(16)));
using v8half = _Float16 __attribute__((ext_vector_type(8)));
using v8float = f32 __attribute__((ext_vector_type(8)));
#endif

} // namespace krk

#endif // KRK_HIP_HPP
