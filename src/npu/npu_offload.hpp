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
//  reference. rope stays there too: the device's rope takes one theta and one
//  scaling where Kraken carries an arbitrary per-layer inv_freq table (YaRN,
//  per-layer windows), so handing it over would change the model rather than
//  its rounding.
//
//  ATTENTION is offered, but only for the calls the device can express, and
//  that subset is checked field by field (see AttnCall). The device wants
//  contiguous head-major [batch, heads, seq, dim] where Kraken's cache is
//  token-major with a row stride, so this layer gathers. That is worth doing
//  because the reference attention is a scalar loop and attention is the
//  quadratic term in prefill: measured at the shapes prefill actually runs, the
//  gather + device kernel is 2.4x the reference at 257x257 and 3.4x at 513x513,
//  and the end-to-end difference is 1.2e-07 -- the f32 rounding of a different
//  summation order. At DECODE shapes the gather is O(cache) for one query row
//  and the device LOSES below about four rows (measured 0.21x at 1x129), so
//  `attn_min_tok` keeps decode on the reference. A call carrying an 8-bit
//  cache, a sliding window, a masked block, a logit softcap or a sink is not
//  representable at all and is refused by name, counted, and named in the run
//  summary -- it is never silently approximated.
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

// ---- the load-time prepare pass -------------------------------------------
//
// Packing a weight into the device's int8 layout costs 2.3 ms, and the cost is
// paid once per weight. Where it lands is not cosmetic: leaving it in the first
// decode step makes a 16-token run look like an int8 REGRESSION (0.74x) while
// the steady state is a 2.5x WIN (measured, both arms, this box). A one-time
// cost inside a benchmark is a lie about the thing being benchmarked.
//
// So the engine asks, at load, whether the backend wants a pass whose only job
// is to touch every weight, and runs one 1-token forward with `preparing` set.
// In that mode a single-row GEMM packs instead of deferring, so every weight is
// converted before the first prompt and the decode path starts warm.
//
// wants_prepare() is false unless the int8 path is on: with W8A8 off there is
// no pack to move, and a prepare pass would be a wasted forward.
bool wants_prepare(const Offload *o);
void set_preparing(Offload *o, bool on);
bool preparing(const Offload *o);

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

// One attention call, as the backend's AttnDesc describes it. Every field that
// can make a call unrepresentable on the device is carried here so the decision
// and the counting live in ONE place instead of being spread over the caller:
// a call is either taken by the device or refused with a name.
struct AttnCall {
    void *out = nullptr;
    const void *q = nullptr;
    const void *kcache = nullptr;
    const void *vcache = nullptr;
    i64 n_head = 0;
    i64 n_kv = 0;
    i64 hd = 0;
    i64 n_tok = 0;
    i64 pos0 = 0;
    i64 layer = 0;
    i64 layer_stride = 0; // elements
    i64 pos_stride = 0;   // elements (the f32 cache layout)
    f32 scale = 1.0f;
    bool causal = true;
    // The features that make a call unrepresentable. All default to "absent".
    bool kv_int8 = false;
    bool has_sinks = false;
    i64 window = 0;
    i64 skip_from = -1;
    f32 softcap = 0.0f;
    // ALiBi's scale. The device body has no per-head distance bias, so a call
    // carrying one is refused below rather than run WITHOUT it: the bias is the
    // family's whole position signal, so dropping it would decode fluent text
    // from a different attention distribution.
    f32 alibi_max_bias = 0.0f;
};

// Run one attention call on the device. Returns false -- having written
// nothing -- when the call is not representable, is below `attn_min_tok`, or a
// staging buffer did not fit; the caller then runs its own body. Every refusal
// is counted under the reason that caused it.
bool attention(Offload *o, const AttnCall &c);

// Whether attention is offered to the device at all (KRK_NPU_ATTN). The caller
// uses this to word its run summary honestly: with the opt-in off, attention is
// one of the ops that stayed on the host, and the summary says so.
bool attention_offered(const Offload *o);

} // namespace npu
} // namespace krk

#endif // KRK_NPU_OFFLOAD_HPP
