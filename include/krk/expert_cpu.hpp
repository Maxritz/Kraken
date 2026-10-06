// ============================================================================
//  expert_cpu.hpp — compute a routed expert on the CPU, beside the GPU.
//
//  The MoE expert path is the one place this engine is memory-bound rather than
//  compute-bound. A decode token routes k experts per layer, and any expert not
//  resident in VRAM has to be moved: measured on this host (NVMe, 1.82 MiB
//  reads) 593 MiB/s at one outstanding read, 1446 MiB/s at four, and the
//  promotion copy on top of that. The activation an expert needs is m x n_embd
//  floats -- 8 KiB at m = 1 -- against a ~1.95 MiB weight, so moving the WEIGHT
//  costs ~250x more than moving the OPERAND. That is the whole argument: do not
//  move a cold expert, COMPUTE it where its bytes already are.
//
//  Two things make that pay off here rather than in theory:
//
//    * the routed weight slice is already mapped (the GGUF mapping is the
//      canonical COLD tier), so a CPU expert costs no residency at all -- no
//      VRAM slot, and no WARM buffer either. That is what lets the VRAM expert
//      budget and the host WARM budget both shrink, which is the point: the
//      memory the expert cache gives up is what the KV cache and the rest of
//      the run get back;
//    * the GPU kernels for the resident experts are already queued on the
//      default stream, so CPU work issued after them overlaps them instead of
//      following them.
//
//  The arithmetic mirrors the four GPU ops the same expert would take, at the
//  same rounding points (see the .cpp): gate/up gemv, silu(g)*u rounded to f16,
//  down gemv, f16 rounding on the accumulation. The CPU result is a close
//  approximation of the GPU one, not a bit-identical replacement -- which is
//  why this sits behind a switch and why the gate suite runs with it off.
//
//  Row-wise, exactly as the device gemv kernel does it: one row dequantized
//  into a scratch buffer, dotted, discarded. Nothing is ever materialized in
//  f32, so one expert needs O(n_in + n_ff) of scratch and not n_ff * n_in.
// ============================================================================
#ifndef KRK_EXPERT_CPU_HPP
#define KRK_EXPERT_CPU_HPP

#include "krk/common.hpp"
#include "krk/expert_cache.hpp" // ExpertSource, QuantTensor, GgufTensor
#include "krk/quant.hpp"

namespace krk {

// A pool of persistent host threads that compute routed experts out of the
// weight mapping. One instance per process; the engine owns the only caller.
//
// Not thread-safe: run() must not be called concurrently with itself.
class CpuExpertPool {
public:
    // One expert's contribution to one layer: `m` rows of xn, each weighted by
    // its softmax gate, accumulated into the caller's output.
    struct Job {
        i32 expert = -1;
        const i32 *rows = nullptr;   // [m] row ids into xn/out
        const f32 *alpha = nullptr;  // [m] mixing weights
        i32 m = 0;
    };

    static CpuExpertPool &get();

    // Worker count. Clamped to [1, 63] and to hardware_concurrency(). Changing
    // it retires the current team; call it before the first forward pass.
    void set_threads(i32 n);
    i32 threads() const;

    // Computes every job and accumulates the results into `out`:
    //
    //     out[row][j] += alpha[r] * down( silu(gate . x) * (up . x) )
    //
    // `xn` and `out` are [rows][n_embd] f32, `xn` dense, `out` zeroed by the
    // caller. Blocks until every job is done. Jobs may share rows, so each
    // worker accumulates into its own private buffer and the buffers are summed
    // at the end; no job may touch another job's buffer. Returns the number of
    // experts computed, i.e. jobs with m > 0 and a slice that resolved.
    i32 run(const ExpertSource &src, i32 layer, const Job *jobs, i32 n_job,
            const f32 *xn, i32 rows, i64 n_embd, f32 *out);

    // Wall time of the last run() call, and lifetime counters. These are what
    // --profile prints for the CPU side of the split and what the expert-cache
    // sweep is read against.
    f64 last_ms() const;
    u64 experts_done() const;
    u64 rows_done() const;
    void reset_counters();

    CpuExpertPool(const CpuExpertPool &) = delete;
    CpuExpertPool &operator=(const CpuExpertPool &) = delete;

private:
    CpuExpertPool();
    ~CpuExpertPool();

    struct Impl;
    Impl *impl_ = nullptr;
};

} // namespace krk

#endif // KRK_EXPERT_CPU_HPP
