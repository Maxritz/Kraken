// ============================================================================
//  npu_offload.hpp — the seam between the reference backend and a virtual NPU.
//
//  The virtual NPU (github.com/riteshnair/npuxl-improved) is a software device
//  that executes a small op set — matmul (f32 / packed i8), rmsnorm, softmax,
//  rope, swiglu, gelu, add, mul, attention, an MTP head — on its own SIMD
//  kernels, addressed with plain host pointers. Kraken's Backend surface is
//  much wider than that (MoE routing and dispatch, MLA, a quantized KV cache,
//  gated delta net, ssd_scan, shortconv, hyper-connections, PLE...), and every
//  one of those has a scalar reference implementation already: the CPU backend.
//
//  So this layer never reimplements the graph. It offers each op to the device
//  and the caller keeps its own body as the fallback, which makes the reference
//  implementation the answer to every op the device declines or does not have.
//
//  TWO MODES
//
//  Mode::Gemm -- the GEMM path only, which is >95% of a transformer's FLOPs and
//  the one VNPU op whose numerics agree with the reference to f32 rounding
//  (measured: 2.7e-07 relative worst case). Everything else stays on the
//  reference, so this arm differs from the CPU arm only where two GEMM kernels
//  would differ anyway: in the last bits.
//
//  Mode::Graph -- `--npu-cpu`: the whole inference runs through the device.
//  Every op VNPU has is issued to it (GEMM, RMSNorm, SwiGLU, add, mul); the ops
//  it has no opcode for keep the reference body and are named in the run
//  summary. This arm is NOT bit-identical to the CPU one,
//  because the device's nonlinearities run its own `fast_expf` (a degree-5
//  polynomial 2^x) rather than libm: measured, silu lands within 8.4e-05
//  relative of the exact function, against an f16 activation's 5e-04 — so the
//  arm is still more accurate than the GPU's f16 path, and still a different
//  rounding of the same model.
//
//  WHAT IS DELIBERATELY NOT OFFERED
//
//  vnpu_gelu_f32 is not used at all: measured against the exact tanh-GELU it is
//  5.5e-03 absolute and 0.74 relative on the tails, three orders of magnitude
//  worse than the same polynomial's swiglu. gemma's GeGLU stays on the
//  reference. rope and attention stay there too, and not for accuracy reasons:
//  the device's rope takes one theta and one scaling where Kraken carries an
//  arbitrary per-layer inv_freq table (YaRN, per-layer windows), and its
//  attention takes contiguous [batch, heads, seq, dim] K/V where Kraken has a
//  grouped-query cache with a row stride, a window, sinks and an int8 layout.
//  Both would change the model, not just its rounding.
//
//  ORDERING
//
//  In Graph mode the ops are QUEUED and submitted as one batch per sync, which
//  is what makes routing a 4096-element RMSNorm through a device worth doing:
//  the queue turns ten small ops per layer into zero extra round trips. Every
//  op that is NOT queued flushes the queue before it runs (see
//  CpuBackend::npu_flush_), so a host kernel never reads a row the device has
//  not written yet. KRK_NPU_STRICT_QUEUE=1 submits and waits on every single
//  call instead, which is the control arm that proves the batching loses
//  nothing: the same input must produce the same numbers either way.
//
//  COST
//
//  A queued small op still costs the device worker a dispatch, and a 512-wide
//  RMSNorm or a 1376-wide SwiGLU is a few microseconds of work that the
//  reference body already has in its own cache. Measured on a 4-layer 512-dim
//  fixture, handing those over is within noise of not doing it (decode 153.1
//  against 159.7 tok/s, prefill identical) -- the GEMM is the whole story at
//  that size, and it is on the device either way. KRK_NPU_MIN_ELEMS raises the
//  floor so only calls of N or more elements go to the device; 0, the default,
//  offers everything, because "the whole graph on the device" is what this arm
//  is for.
// ============================================================================
#ifndef KRK_NPU_OFFLOAD_HPP
#define KRK_NPU_OFFLOAD_HPP

#include <string>
#include "krk/common.hpp"
#include "krk/quant.hpp"

namespace krk {
namespace npu {

// What the device is asked to do. See the two modes above.
enum class Mode {
    Gemm,  // the GEMM path only
    Graph, // the whole graph: every op the device has
};

// An open VNPU device plus the command queue this layer keeps for it. Opaque on
// purpose: vnpu.h must not leak into the core headers.
class Offload;

// True when this build was configured with -DKRK_ENABLE_NPU=ON. A build without
// support still links these symbols (npu_offload.cpp compiles a stub body), so
// the backend-selection code needs no #ifdef and `--npu` on such a build fails
// with a sentence instead of a link error.
bool available();

// Opens the device and its stream. Returns nullptr and fills *why on failure
// (no support compiled in, device creation failed, ...). The handle is owned by
// the caller and released with destroy().
Offload *create(int device_index, Mode mode, std::string *why);
void destroy(Offload *o);

// One line naming the device and what it reports about itself, for the banner.
std::string describe(const Offload *o);

// Counters for the run: per-op call counts, how many calls the device declined,
// the batch count, the dequantized weight staging, and which ops stayed on the
// host. Empty for a null handle.
std::string stats(const Offload *o);

// Submits and waits for everything queued. Returns false if the device reported
// an error; the queue is empty afterwards either way. Cheap when empty: a Gemm
// arm that has already synced costs one vector-size check.
bool flush(Offload *o);

// ---- the ops ---------------------------------------------------------------
//
// Each returns true when the device took the call and false when the caller
// must run its own body, which is the only fallback there is. In Gemm mode the
// activation ops return false immediately; gemm() is the one op both modes
// offer.

// out[rows, n_out] = x[rows, n_in] * w[n_out, n_in]^T, the layout every Kraken
// weight already carries. w is in its GGUF block format; a weight that is not
// f32 is dequantized once into the staging cache (budget: KRK_NPU_WEIGHT_MB,
// default 1024).
bool gemm(Offload *o, void *out, const void *x, const void *w, DType wt,
          i64 n_out, i64 n_in, i64 rows);

// out[rows, n] = rmsnorm(x, w, eps), rows contiguous n-wide rows.
bool rmsnorm(Offload *o, void *out, const void *x, const f32 *w, i64 rows, i64 n,
             f32 eps);

// out[i] = silu(gate[i]) * up[i].
bool swiglu(Offload *o, void *out, const void *gate, const void *up, i64 n);

// a[i] += b[i].
bool add(Offload *o, void *a, const void *b, i64 n);

// dst[i] += alpha * src[i].
bool add_scaled(Offload *o, void *dst, const void *src, f32 alpha, i64 n);

// a[i] *= b[i].
bool mul(Offload *o, void *a, const void *b, i64 n);

} // namespace npu
} // namespace krk

#endif // KRK_NPU_OFFLOAD_HPP
