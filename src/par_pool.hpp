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
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

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

    // Pin the total participant count (workers = n - 1). Retires the
    // current worker generation and spawns a new one; a no-op while a
    // task is in flight. The engine calls this during init, before the
    // first forward pass.
    void set_workers(i64 n) {
        std::unique_lock<std::mutex> lk(mu_);
        if (n < 1 || body_ != nullptr) return;
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
    void run(i64 n, i64 grain, const std::function<void(i64, i64)> &body) {
        if (n <= 0 || !body) return;
        const i64 blocks = (n + grain - 1) / grain;
        const i64 participants = static_cast<i64>(workers_.size()) + 1;
        if (blocks <= 1 || workers_.empty()) {
            body(0, n);
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
            body_ = &body;
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
            body_ = nullptr;
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
        const std::function<void(i64, i64)> *body = nullptr;
    };

    // Snapshot the published geometry. Called under mu_.
    Slice snapshot() const {
        return Slice{n_, grain_, blocks_, stride_, body_};
    }

    // Run blocks [me*stride, min((me+1)*stride, blocks)) of the task
    // described by `s`, then report completion. `s` is a private copy taken
    // under the lock, so a concurrent publish of the next task cannot change
    // the range out from under the body while it runs.
    void run_slice(const Slice &s, i64 me) {
        const i64 b = me * s.stride;
        if (b >= s.blocks) return;
        const i64 e = std::min(b + s.stride, s.blocks);
        const std::function<void(i64, i64)> &fn = *s.body;
        for (i64 i = b; i < e; i++)
            fn(i * s.grain, std::min((i + 1) * s.grain, s.n));
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
        const i64 nw = std::min<i64>(hc - 1, 63);
        const i64 start = task_id_.load(std::memory_order_acquire);
        for (i64 i = 0; i < nw; i++)
            workers_.emplace_back([this, g = gen_.load(), idx = i, start] {
                worker_loop(g, idx, start);
            });
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
                        have = s.body != nullptr;
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
                    have = run_id > seen && s.body != nullptr;
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
    const std::function<void(i64, i64)> *body_ = nullptr;
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
