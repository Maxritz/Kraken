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
//  So the NPU backend is the CPU backend with ONE op handed over: the GEMM.
//  That is not a compromise of convenience, it is the honest split. The GEMM
//  path is >95% of the FLOPs of a transformer and it is the one VNPU op whose
//  numerics agree with the reference to f32 rounding (measured: 2.7e-7
//  relative worst case on the host that runs these tests — an FMA-order
//  difference, the same class of difference two BLAS libraries show). The ops
//  that would NOT agree are the transcendental ones: VNPU's swiglu uses a
//  polynomial sigmoid and measured 1.8e-05 absolute on small inputs, which is
//  two orders of magnitude worse than the reference and compounds over layers.
//  Those stay on the reference implementation, deliberately, and the file-level
//  comment in npu_offload.cpp says so where the next reader will look.
//
//  Everything here returns false when the op is not applicable (wrong shape,
//  unsupported weight type, no budget left, or no NPU built in); the caller
//  then runs its own body. That makes the CPU body the fallback for every
//  single call rather than a parallel implementation to keep in sync.
// ============================================================================
#ifndef KRK_NPU_OFFLOAD_HPP
#define KRK_NPU_OFFLOAD_HPP

#include <string>
#include "krk/common.hpp"
#include "krk/quant.hpp"

namespace krk {
namespace npu {

// An open VNPU device plus the GEMM offload's own bookkeeping. Opaque on
// purpose: vnpu.h must not leak into the core headers.
class Offload;

// True when this build was configured with -DKRK_ENABLE_NPU=ON. A build without
// support still links these symbols (npu_offload.cpp compiles a stub body), so
// the backend-selection code needs no #ifdef and a `--npu` flag on such a build
// fails with a sentence instead of a link error.
bool available();

// Opens the device and its stream. Returns nullptr and fills *why on failure
// (no support compiled in, device creation failed, ...). The returned handle is
// owned by the caller and released with destroy().
Offload *create(int device_index, std::string *why);
void destroy(Offload *o);

// One line naming the device and what it reports about itself, for the banner.
std::string describe(const Offload *o);

// Counters for the run: gemm calls offered, calls served by the device, calls
// that fell back, GFLOPs handed over, and the bytes of dequantized weights held
// in the staging cache. Empty for a null handle.
std::string stats(const Offload *o);

// out[rows, n_out] = x[rows, n_in] * w[n_out, n_in]^T, the layout every Kraken
// weight already carries. w is in its GGUF block format; a weight that is not
// f32 is dequantized once into the staging cache (budget: KRK_NPU_WEIGHT_MB,
// default 1024). Returns true when the device computed the result.
bool gemm(Offload *o, f32 *out, const f32 *x, const void *w, DType wt,
          i64 n_out, i64 n_in, i64 rows);

} // namespace npu
} // namespace krk

#endif // KRK_NPU_OFFLOAD_HPP
