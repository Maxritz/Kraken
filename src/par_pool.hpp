// ============================================================================
//  par_pool.hpp — persistent worker pool for the CPU backend.
//
//  The reference backend was single-threaded by design; on a many-core
//  host that leaves the machine idle. This pool parallelizes the
//  BLAS-shaped ops across INDEPENDENT units of work (output columns,
//  attention heads, token rows). No reduction is ever split across
//  blocks, so every floating-point accumulation keeps its serial order
//  and results are bit-identical to the single-threaded path — the
//  backend stays a valid oracle for the GPU implementation.
//
//  Sizing: one worker per hardware thread minus the calling thread, or
//  KRK_THREADS in the environment. EngineConfig::threads can pin it via
//  set_workers() before the first forward pass.
//
//  Scheduling. Every block in one par_for does identical work (same
//  grain, same op), so ranges are partitioned STATICALLY: participant
//  p owns blocks [p*stride, (p+1)*stride). There is no shared block
//  counter on the hot path — a decode step issues ~200 par_for calls
//  back to back, and making 32 threads fight over one mutex (or worse,
//  a condition variable) per op costs more than the work itself; that
//  contention is why decode degraded at high thread counts before.
//  Each participant reports completion with a single lock per task.
//
//  Wake policy — spin, then park. Ops arrive in bursts, so workers
//  watch the atomic task id for a couple of milliseconds before
//  parking: back-to-back ops find a hot team with zero syscalls,
//  while long idle periods still sleep.
// ============================================================================
#ifndef KRK_PAR_POOL_HPP
#define KRK_PAR_POOL_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#include "krk/common.hpp"

namespace krk {

#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
inline void cpu_relax() { __builtin_ia32_pause(); }
#elif defined(_MSC_VER)
inline void cpu_relax() { _mm_pause(); }
#else
inline void cpu_relax() {}
#endif
#else
inline void cpu_relax() {}
#endif

// ============================================================================
//  Core placement.
//
//  Every thread the pool spawns is an OS thread, and the OS is free to move it
//  at any time: through a whole GEMM a worker can land on a slow core, share a
//  core with another worker, or follow a migrating interrupt. On a hybrid part
//  (Zen 4/5, Alder/Raptor Lake, Snapdragon X) it is worse than that — the fast
//  cores are not enumerated first, so `hardware_concurrency()` threads spread
//  over the slow cores and the *whole* pool runs at the E-core clock. That is
//  the single cheapest throughput a CPU backend can leave on the table, and it
//  costs one syscall per thread at startup.
//
//  Policy, fastest core first:
//    * the calling thread (which also runs a slice) gets the best core;
//    * worker i gets the (i+1)-th best core, so a small pool lands entirely on
//      the performance cluster instead of straddling both;
//    * a pool larger than the machine wraps — sharing is better than parking.
//
//  The ordering comes from cpufreq, which is the only portable way to tell the
//  clusters apart without a vendor SDK (Apple exposes nothing at all; Windows
//  would need GetSystemCpuSetInformation). No cpufreq — a VM, a container, an
//  old kernel, a CPU without the driver — means no order to impose, and the
//  pool keeps its previous unpinned behaviour. KRK_AFFINITY=1 forces pinning on
//  such a box anyway (useful to measure placement alone), KRK_AFFINITY=0 always
//  disables it.
// ============================================================================
struct CpuTopology {
    std::vector<i64> order;  // CPU ids, fastest first
    i64 n_fast = 0;          // cores within 15% of the fastest clock
    i64 n_slow = 0;          // the rest, when the part is hybrid
    i64 max_khz = 0;
    bool hybrid = false;
    bool pinned = false;

    std::string summary() const {
        if (order.empty()) return "cpu topology unknown";
        std::string s = format("%lld cores", static_cast<long long>(order.size()));
        if (max_khz > 0) {
            s += format(", top %.2f GHz", max_khz / 1e6);
            if (hybrid)
                s += format(", %lldP+%lldE", static_cast<long long>(n_fast),
                                static_cast<long long>(n_slow));
            else
                s += " (uniform)";
        }
        s += pinned ? ", pinned" : ", unpinned";
        return s;
    }
};

inline i64 cpu_max_khz(i64 cpu) {
#if defined(__linux__)
    static const char *const sinks[] = {
        "/sys/devices/system/cpu/cpu%lld/cpufreq/cpuinfo_max_freq",
        "/sys/devices/system/cpu/cpu%lld/cpufreq/scaling_max_freq",
    };
    for (const char *pat : sinks) {
        char path[192];
        std::snprintf(path, sizeof path, pat, static_cast<long long>(cpu));
        if (FILE *f = std::fopen(path, "r")) {
            long long v = 0;
            if (std::fscanf(f, "%lld", &v) == 1) {
                std::fclose(f);
                if (v > 0) return v;
                continue;
            }
            std::fclose(f);
        }
    }
#else
    (void)cpu;
#endif
    return 0;
}

inline CpuTopology detect_cpu_topology(i64 hc) {
    CpuTopology t;
    t.order.reserve(static_cast<size_t>(hc));
    std::vector<std::pair<i64, i64>> f;  // (khz, cpu)
    f.reserve(static_cast<size_t>(hc));
    for (i64 i = 0; i < hc; i++) {
        const i64 k = cpu_max_khz(i);
        if (k > 0) {
            f.emplace_back(k, i);
            if (k > t.max_khz) t.max_khz = k;
        }
    }
    if (static_cast<i64>(f.size()) == hc && hc > 1) {
        // Stable: equal clocks keep their CPU order, so a uniform part gets the
        // identity permutation and nothing moves.
        std::stable_sort(f.begin(), f.end(),
                         [](const std::pair<i64, i64> &a, const std::pair<i64, i64> &b) {
                             return a.first > b.first;
                         });
        for (const auto &kv : f) {
            t.order.push_back(kv.second);
            // 15%: absorbs DVFS rounding and SMT siblings, still separates a
            // 3.0 GHz P core from a 2.0 GHz E core.
            if (kv.first * 100 >= t.max_khz * 85) t.n_fast++;
            else t.n_slow++;
        }
        t.hybrid = t.n_slow > 0;
    } else {
        for (i64 i = 0; i < hc; i++) t.order.push_back(i);
        t.n_fast = hc;
    }
    return t;
}

inline bool pin_this_thread(i64 cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<int>(cpu), &set);
    return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
#elif defined(_WIN32)
    if (cpu < 64) {
        SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << cpu);
        return true;
    }
    return false;
#else
    (void)cpu;
    return false;
#endif
}

class Pool {
public:
    static Pool &get() {
        static Pool pool;
        return pool;
    }

    // Total participants (workers + the calling thread).
    i64 workers() const {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<i64>(workers_.size()) + 1;
    }

    // Which core each participant landed on. Settled at construction and
    // re-settled by set_workers(), so it is safe to read any time.
    const CpuTopology &topology() const { return topo_; }

    // Pin the total participant count (workers = n - 1). Retires the
    // current worker generation and spawns a new one; a no-op while a
    // task is in flight. The engine calls this during init, before the
    // first forward pass.
    void set_workers(i64 n) {
        std::unique_lock<std::mutex> lk(mu_);
        if (n < 1 || body_fn_ != nullptr) return;
        const i64 want = std::min<i64>(n - 1, 63);
        if (want == static_cast<i64>(workers_.size())) return;
        // Retire the current generation: each worker exits its loop as
        // soon as its generation no longer matches, then is joined.
        gen_.fetch_add(1, std::memory_order_release);
        cv_start_.notify_all();
        for (std::thread &t : workers_)
            if (t.joinable()) t.join();
        workers_.clear();
        // No task can be in flight here (body_ == nullptr), so the new
        // generation starts at the CURRENT id: seeding it with 0 would send
        // every fresh worker chasing a task that was drained long ago.
        const i64 start = task_id_.load(std::memory_order_acquire);
        for (i64 i = 0; i < want; i++)
            workers_.emplace_back(
                [this, g = gen_.load(std::memory_order_acquire), idx = i, start] {
                    worker_loop(g, idx, start);
                });
    }

    // Run body(b, e) over blocks of at most `grain` indices covering
    // [0, n). body must touch only data disjoint from other blocks and
    // must not call back into the pool.
    //
    // `split == false` forces the serial path: the flop floor in par_for is a
    // throughput policy, not a semantic one. Both branches below call the SAME
    // instantiation of `invoke`, and that is load-bearing rather than tidy.
    //
    // Calling the lambda directly at the call site (what this did for small
    // ops) and calling it through std::function inside the pool (what it did
    // for large ones) compiles the same arithmetic TWICE, and GCC contracts
    // `a * c - b * s` into an FMA in one copy and not in the other. The
    // rounding difference is ~1e-7 relative, which is enough to make the CPU
    // reference backend's answer depend on how many rows the caller passed in
    // one call: measured on RoPE, row 65 of a 66-row call differed from the
    // same row of a 1-row call in 200 of 512 lanes, and that is exactly the
    // chunk-invariance property test_prefill_chunk_bitexact gates on. One
    // trampoline for both paths makes the serial case and the pooled case the
    // same machine code, so the floor can be tuned without moving a number.
    template <class F>
    void run(i64 n, i64 grain, bool split, F &&body) {
        if (n <= 0) return;
        using G = typename std::remove_reference<F>::type;
        void (*const invoke)(void *, i64, i64) = [](void *ctx, i64 b, i64 e) {
            (*static_cast<G *>(ctx))(b, e);
        };
        if (grain < 1) grain = n < 1 ? 1 : n;
        const i64 blocks = (n + grain - 1) / grain;
        const i64 participants = static_cast<i64>(workers_.size()) + 1;
        if (!split || blocks <= 1 || workers_.empty()) {
            invoke(&body, 0, n);
            return;
        }
        // Static partition of the (uniform) blocks. stride >= 1 even
        // when blocks < participants; participants whose range is empty
        // simply do nothing.
        const i64 stride = (blocks + participants - 1) / participants;
        // The calling thread is the last participant. It publishes and
        // claims its own slice in one critical section, so no worker can
        // observe a half-written task.
        Slice s;
        {
            std::unique_lock<std::mutex> lk(mu_);
            n_ = n;
            grain_ = grain;
            blocks_ = blocks;
            stride_ = stride;
            body_ctx_ = &body;
            body_fn_ = invoke;
            done_ = 0;
            // Release-order bump: workers acquire-load the id, so every
            // field written above is visible the moment they observe it.
            task_id_.fetch_add(1, std::memory_order_release);
            s = snapshot();
            cv_start_.notify_all();
        }
        run_slice(s, participants - 1);
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_done_.wait(lk, [&] { return done_ >= blocks_; });
            body_ctx_ = nullptr;
            body_fn_ = nullptr;
        }
    }

private:
    // How long a worker spins (watching the atomic task id, burning no
    // syscalls) before parking on the condition variable. Decode's op
    // gaps are microseconds; a couple of milliseconds of spin covers a
    // whole forward pass.
    static constexpr int kSpinMs = 2;

    // One participant's slice of the current task, plus the geometry it was
    // computed from. The caller rewrites n_/grain_/blocks_/stride_ for the
    // NEXT task the moment every participant has reported done, so these
    // fields may only be read while mu_ is held. Reading them unlocked let a
    // participant run a slice built from one task's geometry against another
    // task's body, covering some indices twice and others not at all
    // (test_par_pool_covers_every_block failed ~1 run in 3).
    struct Slice {
        i64 n = 0, grain = 0, blocks = 0, stride = 0;
        void *ctx = nullptr;
        void (*invoke)(void *, i64, i64) = nullptr;
    };

    // Snapshot the published geometry. Called under mu_.
    Slice snapshot() const {
        return Slice{n_, grain_, blocks_, stride_, body_ctx_, body_fn_};
    }

    // Run blocks [me*stride, min((me+1)*stride, blocks)) of the task
    // described by `s`, then report completion. `s` is a private copy taken
    // under the lock, so a concurrent publish of the next task cannot change
    // the range out from under the body while it runs.
    void run_slice(const Slice &s, i64 me) {
        const i64 b = me * s.stride;
        if (b >= s.blocks) return;
        const i64 e = std::min(b + s.stride, s.blocks);
        for (i64 i = b; i < e; i++)
            s.invoke(s.ctx, i * s.grain, std::min((i + 1) * s.grain, s.n));
        // One completion report per participant per task — staggered by
        // work completion, so the mutex sees almost no contention.
        {
            std::lock_guard<std::mutex> lk(mu_);
            done_ += e - b;
            if (done_ >= blocks_) cv_done_.notify_one();
        }
    }

    Pool() {
        i64 hc = static_cast<i64>(std::thread::hardware_concurrency());
        if (const char *e = std::getenv("KRK_THREADS")) {
            const i64 v = std::atoll(e);
            if (v > 0) hc = v;
        }
        if (hc < 1) hc = 1;
        setup_placement(hc);
        const i64 nw = std::min<i64>(hc - 1, 63);
        const i64 start = task_id_.load(std::memory_order_acquire);
        for (i64 i = 0; i < nw; i++)
            workers_.emplace_back([this, g = gen_.load(), idx = i, start] {
                if (pinned_) pin_this_thread(worker_cpu(idx));
                worker_loop(g, idx, start);
            });
    }

    // Decide the placement once, then apply it: the caller keeps the fastest
    // core (it also runs a slice, and on the MoE path it does the routing, the
    // sampling and the device submissions between slices) and the workers take
    // the cores that follow, in the same fastest-first order.
    void setup_placement(i64 hc) {
        topo_ = detect_cpu_topology(hc);
        bool want = topo_.hybrid;
        if (const char *e = std::getenv("KRK_AFFINITY")) want = std::atoll(e) != 0;
        if (!want) return;
        pinned_ = topo_.order.size() > 1 && pin_this_thread(topo_.order[0]);
        topo_.pinned = pinned_;
    }

    i64 worker_cpu(i64 idx) const {
        const i64 n = static_cast<i64>(topo_.order.size());
        if (n == 0) return -1;
        return topo_.order[static_cast<size_t>((idx + 1) % n)];
    }

    ~Pool() {
        quit_.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mu_);
            cv_start_.notify_all();
        }
        for (std::thread &t : workers_)
            if (t.joinable()) t.join();
    }

    Pool(const Pool &) = delete;
    Pool &operator=(const Pool &) = delete;

    void worker_loop(i64 my_gen, i64 my_idx, i64 seen) {
        // Id of the task this worker has already drained. Waiting for a
        // NEW id (not for body_ != nullptr, which stays set until the
        // caller finishes) is what keeps idle workers asleep instead of
        // spinning on the mutex.
        for (;;) {
            // ---- spin phase ----
            // Watch the atomic task id without touching the mutex: a
            // back-to-back op finds every worker already hot, so the
            // pool costs zero syscalls on the decode hot path.
            bool have = false;
            i64 run_id = 0;
            Slice s;
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(kSpinMs);
            for (;;) {
                if (quit_.load(std::memory_order_acquire) ||
                    my_gen != gen_.load(std::memory_order_acquire))
                    return;
                const i64 t = task_id_.load(std::memory_order_acquire);
                if (t > seen) {
                    // Claim the geometry for THIS task id under the lock.
                    // Reading it separately from the id let the two
                    // disagree: a worker could observe id N, then snapshot
                    // the geometry of N+1 after the caller had already
                    // retired it — running a slice twice, or with a body
                    // that had been reset to nullptr.
                    std::lock_guard<std::mutex> lk(mu_);
                    if (task_id_.load(std::memory_order_acquire) > seen) {
                        run_id = task_id_.load(std::memory_order_acquire);
                        s = snapshot();
                        have = s.invoke != nullptr;
                    }
                    if (have) break;
                    // The task was retired before we could claim it (the
                    // caller's fast path can clear body_ without us).
                    // Re-read the id and try again.
                    seen = task_id_.load(std::memory_order_acquire);
                    continue;
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    // ---- park phase ----
                    std::unique_lock<std::mutex> lk(mu_);
                    cv_start_.wait(lk, [&] {
                        return quit_.load(std::memory_order_acquire) ||
                               my_gen != gen_.load(std::memory_order_acquire) ||
                               task_id_.load(std::memory_order_acquire) > seen;
                    });
                    if (quit_.load(std::memory_order_acquire) ||
                        my_gen != gen_.load(std::memory_order_acquire))
                        return;
                    // Claim it here too, still under the lock, so the id
                    // and the geometry always come from the same publish.
                    run_id = task_id_.load(std::memory_order_acquire);
                    s = snapshot();
                    have = run_id > seen && s.invoke != nullptr;
                    if (!have) {
                        seen = run_id;
                        lk.unlock();
                        continue;
                    }
                    lk.unlock();
                    break;
                }
                cpu_relax();
            }
            run_slice(s, my_idx);
            // Mark the task just drained — NOT whatever task_id_ says now.
            // The caller may publish the next op while we report, and
            // re-reading the id here would swallow that task: its blocks
            // would go uncovered and the caller would wait forever.
            seen = run_id;
        }
    }

    mutable std::mutex mu_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    std::vector<std::thread> workers_;
    CpuTopology topo_;
    bool pinned_ = false;
    void *body_ctx_ = nullptr;
    void (*body_fn_)(void *, i64, i64) = nullptr;
    i64 n_ = 0;
    i64 grain_ = 0;
    i64 blocks_ = 0;
    i64 stride_ = 1;
    i64 done_ = 0;
    std::atomic<i64> gen_{0};
    std::atomic<i64> task_id_{0};
    std::atomic<bool> quit_{false};
};

} // namespace krk

#endif // KRK_PAR_POOL_HPP
