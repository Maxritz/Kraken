// expert_cpu.cpp — the CPU half of the hybrid MoE path. See expert_cpu.hpp for
// why this exists at all.
//
// THREE THINGS THE DEVICE DOES THAT THIS HAS TO COPY
// --------------------------------------------------
// The point of this path is to compute the same expert the GPU would have
// computed, just somewhere else. Anywhere the two disagree about *rounding*
// they disagree about the model, and the divergence is silent: the text stays
// fluent. The device keeps activations in f16 (Backend::act_type), so:
//
//   1. the gathered activation x is f16 — the engine hands over
//      `download_f32()` of the device row, which is the f16 values widened
//      exactly, so `xn` here already carries that rounding;
//   2. `gemm` writes g and u as f16, and `silu_mul` writes silu(g)*u as f16;
//      so this rounds g, u and the product at the same three points (r16);
//   3. `scatter_axpy_rows` writes `dst[rows[i]*n] += alpha[i]*src[i*n]` into an
//      f16 accumulator, so each expert's contribution is added and rounded, not
//      accumulated in f32. r16() on the accumulated value is that.
//
// The one deliberate difference is the ORDER of the per-row additions when two
// engineers land on the same row; the device order is the expert-id order of
// its own loop, this one is whichever worker claimed the job first. Both are
// valid f16 accumulations of the same terms. It is why the hybrid path is
// switchable and why the gate suite runs with it off.
//
// NO RESIDENCY, WHICH IS THE WHOLE POINT
// --------------------------------------
// This reads the GGUF mapping directly — no ExpertCache acquire, no WARM
// buffer, no VRAM slot, no promotion. The bytes are already mapped and the
// page cache owns them, so an expert computed here costs the run *nothing* to
// keep. That is what buys back the memory: a smaller `--expert-cache-mb` and a
// disabled WARM tier stop being a throughput disaster, because the experts
// they no longer hold are no longer moved at all.
//
// Row-wise, exactly as the device gemv kernel does it: one row dequantized into
// a reusable scratch buffer, dotted, discarded. One expert needs
// O(n_in + n_ff) of scratch, never n_ff * n_in.
#include "krk/expert_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace krk {

namespace {

constexpr i32 kMaxThreads = 63;
using Clock = std::chrono::steady_clock;

inline f64 ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// The device's rounding point: round through f16 and come back. fp16_to_fp32 of
// an f16 is exact, so this is exactly "store it in an activation buffer".
inline f32 r16(f32 v) { return fp16_to_fp32(fp32_to_fp16(v)); }

// silu, matching the device silu_mul kernel's formulation
// (x / (1 + exp(-x)) and never exp(x)/(1+exp(x)), which overflows at x ~ 89).
inline f32 silu(f32 v) { return v / (1.0f + std::exp(-v)); }

// The expert bank's geometry, resolved once per task instead of once per job:
// three divisions and two row-stride computations are not free at 40 layers a
// token, and a malformed bank must be rejected before any worker sees it.
struct Shape {
    bool ok = false;
    DType type = DType::Unknown;
    i64 n_in = 0;    // inner dim of gate/up
    i64 n_ff = 0;    // rows of gate/up, inner dim of down
    i64 n_expert = 0;
    size_t rs_in = 0;   // gate/up row stride
    size_t rs_ff = 0;   // down row stride
    size_t gate_per = 0; // one expert's bytes in each tensor
    size_t up_per = 0;
    size_t down_per = 0;
};

// SHAPE NOTES, because getting these wrong is silent (see cpu_expert_ceiling):
// the tensors are 3D [n_in, n_ff, n_expert] for gate/up and [n_ff, n_in,
// n_expert] for down. Reading n_expert off the wrong axis gets n_ff, and the
// whole bank reads garbage.
Shape shape_of(const ExpertSource &src) {
    Shape s;
    const GgufTensor *g = src.gate;
    const GgufTensor *u = src.up;
    const GgufTensor *d = src.down;
    if (!g || !u || !d || !g->data || !u->data || !d->data) return s;
    if (g->n_bytes == 0 || u->n_bytes == 0 || d->n_bytes == 0) return s;
    if (g->type != u->type || g->type != d->type) return s;
    if (!dtype_supported(g->type)) return s;
    const i64 n_expert = static_cast<i64>(g->ne[2]);
    const i64 n_in = static_cast<i64>(g->ne[0]);
    const i64 n_ff = static_cast<i64>(g->ne[1]);
    if (n_expert <= 0 || n_in <= 0 || n_ff <= 0) return s;
    if (static_cast<i64>(u->ne[0]) != n_in || static_cast<i64>(u->ne[1]) != n_ff ||
        static_cast<i64>(u->ne[2]) != n_expert)
        return s;
    if (static_cast<i64>(d->ne[0]) != n_ff || static_cast<i64>(d->ne[1]) != n_in ||
        static_cast<i64>(d->ne[2]) != n_expert)
        return s;
    // Rows must be whole quantisation blocks, or the row stride is not a
    // stride and every row after the first is misaligned. The device gemv has
    // the same requirement, so a model that gets here has already met it --
    // checked anyway because here the failure is a wrong number, not a crash.
    if (!dtype_row_aligned(g->type, n_in) || !dtype_row_aligned(g->type, n_ff)) return s;

    s.type = g->type;
    s.n_in = n_in;
    s.n_ff = n_ff;
    s.n_expert = n_expert;
    s.rs_in = dtype_row_bytes(g->type, n_in);
    s.rs_ff = dtype_row_bytes(g->type, n_ff);
    s.gate_per = g->n_bytes / static_cast<size_t>(n_expert);
    s.up_per = u->n_bytes / static_cast<size_t>(n_expert);
    s.down_per = d->n_bytes / static_cast<size_t>(n_expert);
    // The slices have to be at least what the row walk is going to read, or a
    // short tensor would walk off the end of its own expert into the next one.
    if (s.gate_per < s.rs_in * static_cast<size_t>(n_ff)) return s;
    if (s.up_per < s.rs_in * static_cast<size_t>(n_ff)) return s;
    if (s.down_per < s.rs_ff * static_cast<size_t>(n_in)) return s;
    s.ok = true;
    return s;
}

// Per-worker scratch. `acc` is the private output accumulator: two jobs may
// name the same row (a token's k experts), and two workers may run those two
// jobs at the same time, so a single shared output would be a race. Each worker
// accumulates into its own buffer and run() sums them.
//
// `epoch` is what keeps that from costing a full clear per task: a row is
// zeroed the first time THIS worker touches it in THIS task and never again, so
// the clearing is proportional to work done rather than to the layer's row
// count. run() reads the same stamps to know which rows to fold in, which is
// also what stops it summing stale rows a worker never touched.
struct Scratch {
    std::vector<f32> h;     // [n_ff]      silu(gate . x) * (up . x)
    std::vector<f32> row;   // [max(n_in, n_ff)] one dequantized weight row
    std::vector<f32> acc;   // [rows * n_embd] this worker's partial output
    std::vector<u64> epoch; // [rows] last task id that cleared this row here
};

struct Task {
    Shape sh;
    const ExpertSource *src = nullptr;
    // Qualified: `Task` lives in an anonymous namespace at global scope, where
    // the nested `Job` of CpuExpertPool is not reachable unqualified.
    const CpuExpertPool::Job *jobs = nullptr;
    i32 n_job = 0;
    const f32 *xn = nullptr;
    i32 rows = 0;
    i64 n_embd = 0;
    u64 epoch = 0;
};

} // namespace

struct CpuExpertPool::Impl {
    i32 n_threads = 0;
    std::vector<std::thread> team;
    std::vector<Scratch> scratch;

    std::mutex mu;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    bool shutdown = false;
    u64 task_id = 0;
    size_t finished = 0;

    Task task;
    u64 epoch = 0;
    std::atomic<int> next{0};
    std::atomic<int> task_experts{0};
    std::atomic<i64> task_rows{0};

    f64 last_ms = 0.0;
    u64 experts_done = 0;
    u64 rows_done = 0;
    bool warned_shape = false;

    // One expert for one job. Everything the worker needs arrives by const
    // reference: the shape and the task pointers are immutable while a task is
    // in flight, which is why the same Task is shared by every worker without a
    // lock.
    void compute(const Task &t, const Job &job, Scratch &s) {
        const Shape &sh = t.sh;
        const i64 nin = sh.n_in;
        const i64 nff = sh.n_ff;
        const size_t e = static_cast<size_t>(job.expert);
        const u8 *gb = t.src->gate->data + e * sh.gate_per;
        const u8 *ub = t.src->up->data + e * sh.up_per;
        const u8 *db = t.src->down->data + e * sh.down_per;

        for (i32 r = 0; r < job.m; r++) {
            const i32 row = job.rows[r];
            if (row < 0 || row >= t.rows) continue;
            const size_t ro = static_cast<size_t>(row);
            if (s.epoch[ro] != t.epoch) {
                s.epoch[ro] = t.epoch;
                std::fill(s.acc.begin() + static_cast<ptrdiff_t>(ro * t.n_embd),
                          s.acc.begin() + static_cast<ptrdiff_t>((ro + 1) * t.n_embd),
                          0.0f);
            }
            f32 *acc = s.acc.data() + ro * static_cast<size_t>(t.n_embd);
            const f32 *x = t.xn + ro * static_cast<size_t>(t.n_embd);
            const f32 alpha = job.alpha[r];

            // gate/up: one dequantized row at a time, dequantize -> dot ->
            // discard, the shape the device gemv uses.
            for (i64 j = 0; j < nff; j++) {
                dequant_row(sh.type, gb + static_cast<size_t>(j) * sh.rs_in,
                            s.row.data(), nin);
                f32 g = 0.0f;
                for (i64 i = 0; i < nin; i++) g += s.row[static_cast<size_t>(i)] * x[i];
                const f32 g16 = r16(g); // the device stores the gate as f16
                dequant_row(sh.type, ub + static_cast<size_t>(j) * sh.rs_in,
                            s.row.data(), nin);
                f32 u = 0.0f;
                for (i64 i = 0; i < nin; i++) u += s.row[static_cast<size_t>(i)] * x[i];
                const f32 u16 = r16(u); // ... and the up projection too
                s.h[static_cast<size_t>(j)] = r16(silu(g16) * u16);
            }
            // down, then the weighted accumulate the device's
            // scatter_axpy_rows does into an f16 accumulator.
            for (i64 i = 0; i < nin; i++) {
                dequant_row(sh.type, db + static_cast<size_t>(i) * sh.rs_ff,
                            s.row.data(), nff);
                f32 y = 0.0f;
                for (i64 j = 0; j < nff; j++)
                    y += s.row[static_cast<size_t>(j)] * s.h[static_cast<size_t>(j)];
                acc[i] = r16(acc[i] + alpha * r16(y));
            }
        }
    }

    void work(const Task &t, Scratch &s) {
        for (;;) {
            const int j = next.fetch_add(1, std::memory_order_relaxed);
            if (j >= t.n_job) break;
            const Job &job = t.jobs[j];
            if (job.m <= 0 || !job.rows || !job.alpha) continue;
            if (job.expert < 0 || job.expert >= t.sh.n_expert) continue;
            compute(t, job, s);
            task_experts.fetch_add(1, std::memory_order_relaxed);
            task_rows.fetch_add(job.m, std::memory_order_relaxed);
        }
    }

    void worker_loop(size_t idx) {
        u64 seen = 0;
        for (;;) {
            Task t;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv_work.wait(lk, [&] { return shutdown || task_id != seen; });
                if (shutdown) return;
                seen = task_id;
                // Snapshot under the lock: the task fields are published here
                // and must not be read while a later task overwrites them.
                t = task;
            }
            work(t, scratch[idx]);
            {
                std::lock_guard<std::mutex> lk(mu);
                if (++finished == team.size()) cv_done.notify_all();
            }
        }
    }

    void retire() {
        {
            std::lock_guard<std::mutex> lk(mu);
            if (team.empty()) return;
            shutdown = true;
            cv_work.notify_all();
        }
        for (std::thread &t : team)
            if (t.joinable()) t.join();
        team.clear();
        scratch.clear();
        {
            std::lock_guard<std::mutex> lk(mu);
            shutdown = false;
            finished = 0;
        }
    }

    void ensure_team() {
        if (!team.empty()) return;
        i32 n = n_threads > 0 ? n_threads : default_threads();
        n = std::min<i32>(n, kMaxThreads);
        if (n < 1) n = 1;
        n_threads = n;
        scratch.assign(static_cast<size_t>(n), Scratch{});
        for (size_t i = 0; i < static_cast<size_t>(n); i++)
            team.emplace_back([this, i] { worker_loop(i); });
    }

    static i32 default_threads() {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 1;
        return static_cast<i32>(std::min<unsigned>(hw, 8u));
    }
};

CpuExpertPool::CpuExpertPool() : impl_(new Impl()) {}

CpuExpertPool::~CpuExpertPool() {
    if (impl_) {
        impl_->retire();
        delete impl_;
        impl_ = nullptr;
    }
}

CpuExpertPool &CpuExpertPool::get() {
    static CpuExpertPool pool;
    return pool;
}

void CpuExpertPool::set_threads(i32 n) {
    if (!impl_) return;
    if (n <= 0) n = Impl::default_threads();
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    n = std::min<i32>(n, kMaxThreads);
    n = std::min<i32>(n, static_cast<i32>(hw));
    if (n < 1) n = 1;
    if (n == impl_->n_threads) return; // the live team is already the right size
    impl_->n_threads = n;
    // Retire rather than resize: the team is respawned lazily by the next
    // run(), which keeps this callable at init time when nothing is in flight.
    impl_->retire();
}

i32 CpuExpertPool::threads() const { return impl_ ? impl_->n_threads : 0; }

i32 CpuExpertPool::run(const ExpertSource &src, i32 layer, const Job *jobs, i32 n_job,
                       const f32 *xn, i32 rows, i64 n_embd, f32 *out) {
    (void)layer;
    if (!impl_ || !jobs || !xn || !out || n_job <= 0 || rows <= 0 || n_embd <= 0) return 0;
    Impl &im = *impl_;

    const Shape sh = shape_of(src);
    if (!sh.ok || sh.n_in != n_embd) {
        if (!im.warned_shape) {
            im.warned_shape = true;
            KRK_WARN("cpu experts: expert bank is not usable (n_in %lld vs n_embd %lld, "
                     "type %s); those experts fall back to the device path",
                     static_cast<long long>(sh.n_in), static_cast<long long>(n_embd),
                     dtype_name(sh.type));
        }
        return 0;
    }

    im.ensure_team();
    const size_t need = static_cast<size_t>(rows) * static_cast<size_t>(n_embd);
    const size_t buf = static_cast<size_t>(std::max<i64>(sh.n_in, sh.n_ff));
    for (Scratch &s : im.scratch) {
        if (s.acc.size() < need) s.acc.resize(need);
        if (s.epoch.size() < static_cast<size_t>(rows)) s.epoch.resize(static_cast<size_t>(rows), 0);
        if (s.h.size() < static_cast<size_t>(sh.n_ff)) s.h.resize(static_cast<size_t>(sh.n_ff));
        if (s.row.size() < buf) s.row.resize(buf);
    }

    const u64 epoch = ++im.epoch;
    const Clock::time_point t0 = Clock::now();
    {
        std::unique_lock<std::mutex> lk(im.mu);
        Task &t = im.task;
        t.sh = sh;
        t.src = &src;
        t.jobs = jobs;
        t.n_job = n_job;
        t.xn = xn;
        t.rows = rows;
        t.n_embd = n_embd;
        t.epoch = epoch;
        im.next.store(0, std::memory_order_relaxed);
        im.task_experts.store(0, std::memory_order_relaxed);
        im.task_rows.store(0, std::memory_order_relaxed);
        im.finished = 0;
        im.task_id++;
        im.cv_work.notify_all();
        im.cv_done.wait(lk, [&] { return im.finished == im.team.size(); });
    }
    im.last_ms = ms_since(t0);

    // Fold the workers' private accumulators in. Only rows this worker actually
    // cleared-and-wrote are read (its epoch stamp says which), so a worker that
    // claimed no job contributes nothing instead of contributing stale rows.
    for (const Scratch &s : im.scratch) {
        for (i32 r = 0; r < rows; r++) {
            if (s.epoch[static_cast<size_t>(r)] != epoch) continue;
            const f32 *a = s.acc.data() + static_cast<size_t>(r) * static_cast<size_t>(n_embd);
            f32 *o = out + static_cast<size_t>(r) * static_cast<size_t>(n_embd);
            for (i64 i = 0; i < n_embd; i++) o[i] += a[i];
        }
    }

    const i32 done = im.task_experts.load(std::memory_order_relaxed);
    im.experts_done += static_cast<u64>(done);
    im.rows_done += static_cast<u64>(im.task_rows.load(std::memory_order_relaxed));
    return done;
}

f64 CpuExpertPool::last_ms() const { return impl_ ? impl_->last_ms : 0.0; }
u64 CpuExpertPool::experts_done() const { return impl_ ? impl_->experts_done : 0; }
u64 CpuExpertPool::rows_done() const { return impl_ ? impl_->rows_done : 0; }

void CpuExpertPool::reset_counters() {
    if (!impl_) return;
    impl_->experts_done = 0;
    impl_->rows_done = 0;
    impl_->last_ms = 0.0;
}

} // namespace krk
