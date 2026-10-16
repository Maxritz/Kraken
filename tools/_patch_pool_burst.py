#!/usr/bin/env python3
"""One release mark per burst: the shared, refcounted pool mark.

Every anchor is asserted before it is used, and each file's own line endings are
preserved (backend.hpp / expert_cache.cpp / main_cli.cpp are CRLF in git,
backend_hip.hip is LF).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def patch(rel, pairs):
    p = ROOT / rel
    raw = p.read_text(newline="")
    eol = "\r\n" if "\r\n" in raw else "\n"
    s = raw.replace("\r\n", "\n")
    for old, new in pairs:
        old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
        n = s.count(old)
        if n != 1:
            sys.exit(f"{rel}: anchor found {n} times: {old[:70]!r}")
        s = s.replace(old, new)
    p.write_text(s.replace("\n", eol), newline="")
    print(f"{rel}: {len(pairs)} replacement(s), eol={'crlf' if eol == chr(13) + chr(10) else 'lf'}")


# ---------------------------------------------------------------- backend.hpp
patch("include/krk/backend.hpp", [
    ("""        u64 order_device = 0, order_host = 0, order_implicit = 0;
    };""",
     """        u64 order_device = 0, order_host = 0, order_implicit = 0;
        // Release marks: how many hipEventRecords the pool issued, and how many
        // slices those records covered. A burst release is ONE mark for its
        // whole burst, so release_slices / release_marks is the sharing factor
        // and a ratio of 1.0 means the burst path was not taken.
        u64 release_marks = 0, release_slices = 0;
    };"""),
    ("""    virtual void release_pooled(void *p, size_t bytes) {
        (void)bytes;
        if (p) release(p);
    }""",
     """    virtual void release_pooled(void *p, size_t bytes) {
        (void)bytes;
        if (p) release(p);
    }
    // Release several cells that are going back at the SAME stream point --
    // an expert's three slices, which is what an eviction returns. The mark a
    // reused cell needs is "the default-stream position its readers are
    // ordered before", and cells released together share that position, so a
    // backend that can express it records one mark for the burst instead of one
    // per cell. It is not an approximation: nothing is submitted between the
    // releases in a burst, so their positions are the same point.
    //
    // Sharing must not go further than a burst. A backend that re-records a
    // mark while a wait on it is still in flight (a rotating pool of marks,
    // say) can put the recorded position after the default stream's
    // `hipStreamWaitEvent(0, paged_ev)` -- the default stream waits on the copy
    // stream there while the copy stream waits on these marks -- and a mark
    // recorded after such a wait, then handed to a copy-stream writer, is a
    // dependency cycle. See HipBackend::PoolMark.
    virtual void release_pooled_batch(void *const ps[], const size_t bytes[], int n) {
        for (int i = 0; i < n; i++) release_pooled(ps[i], bytes[i]);
    }"""),
])

# ------------------------------------------------------------ backend_hip.hip
patch("src/hip/backend_hip.hip", [
    # 1. the mark itself, and the two owners that carry it
    ("""    struct PoolBlock {
        void *p = nullptr;
        hipEvent_t ev = nullptr; // owned by the cell, re-recorded per release
    };
    struct LiveCell {
        size_t rounded = 0;
        hipEvent_t ev = nullptr; // null until the cell's first release
    };""",
     """    // A release mark: "the default-stream position this cell's readers are
    // ordered before". It is JOINTLY owned -- every cell released under it holds
    // a reference, and so does a hand-out whose writer has not consumed the
    // dependency yet -- and the object and its event are recycled when the last
    // holder lets go. That refcount is what makes one record per BURST exact
    // instead of approximate: the slices in a burst are released consecutively
    // with nothing submitted between them, so their stream positions are the
    // same point, and the mark cannot be re-recorded while any wait on it is
    // still outstanding.
    struct PoolMark {
        hipEvent_t ev = nullptr;
        int refs = 0;
    };
    struct PoolBlock {
        void *p = nullptr;
        PoolMark *m = nullptr; // the cell's mark, null before its first release
    };
    struct LiveCell {
        size_t rounded = 0;
        PoolMark *m = nullptr; // null until the cell's first release
    };
    static constexpr int kPoolBurstMax = 8;"""),

    # 2. the pending map and the mark bookkeeping
    ("""    // A reused cell whose previous reader may still be queued. Keyed by cell
    // address, consumed by the next write into that cell.
    std::unordered_map<void *, hipEvent_t> pool_pending_;
    // Takes the dependency, so a caller with several streams to order can issue
    // the wait on each of them.
    hipEvent_t pool_take_dep_(void *dst) {
        const auto it = pool_pending_.find(dst);
        if (it == pool_pending_.end()) return nullptr;
        const hipEvent_t ev = it->second;
        pool_pending_.erase(it);
        return ev;
    }""",
     """    // Marks with no holders, kept so the next burst reuses the event:
    // create+destroy is 0.13 us against a 23 us record, so the pool exists to
    // keep the event object, not to avoid a syscall.
    std::vector<PoolMark *> pool_free_marks_;
    PoolMark *pool_mark_open_() {
        while (!pool_free_marks_.empty()) {
            PoolMark *m = pool_free_marks_.back();
            pool_free_marks_.pop_back();
            if (m->ev) return m; // refs is 0 by construction
            delete m;
        }
        auto *m = new PoolMark;
        if (hipEventCreateWithFlags(&m->ev, hipEventDisableTiming) != hipSuccess) {
            delete m;
            return nullptr;
        }
        return m;
    }
    // A mark whose record failed is unusable and is not recycled.
    static void pool_mark_close_(PoolMark *m) {
        if (!m) return;
        if (m->ev) (void)hipEventDestroy(m->ev);
        delete m;
    }
    static void pool_mark_ref_(PoolMark *m) {
        if (m) ++m->refs;
    }
    void pool_mark_unref_(PoolMark *m) {
        if (!m || --m->refs > 0) return;
        pool_free_marks_.push_back(m);
    }
    // A reused cell whose previous reader may still be queued. Keyed by cell
    // address, consumed by the next write into that cell. The entry holds a
    // reference, so the mark survives until the writer has waited on it.
    std::unordered_map<void *, PoolMark *> pool_pending_;
    // Takes the dependency, so a caller with several streams to order can issue
    // the wait on each of them. The entry's reference moves to the caller, which
    // must hand it back with pool_dep_done_() once the wait is enqueued.
    PoolMark *pool_take_dep_(void *dst) {
        const auto it = pool_pending_.find(dst);
        if (it == pool_pending_.end()) return nullptr;
        PoolMark *m = it->second;
        pool_pending_.erase(it);
        return m;
    }
    void pool_dep_done_(PoolMark *m) { pool_mark_unref_(m); }
    // Drops an outstanding hand-out for a cell that is going back to the free
    // list without ever having been written. The mark is superseded by the one
    // the new release records, which is at a later point and therefore covers
    // everything this one did.
    void pool_drop_pending_(void *dst) {
        const auto it = pool_pending_.find(dst);
        if (it == pool_pending_.end()) return;
        PoolMark *m = it->second;
        pool_pending_.erase(it);
        pool_mark_unref_(m);
    }"""),

    # 3. the three honor paths
    ("""    void pool_honor_(void *dst, hipStream_t s) {
        const hipEvent_t ev = pool_take_dep_(dst);
        if (!ev) return;
        KRK_HIP(hipStreamWaitEvent(s, ev, 0));
        ++expert_pool_.order_device;
    }""",
     """    void pool_honor_(void *dst, hipStream_t s) {
        PoolMark *m = pool_take_dep_(dst);
        if (!m) return;
        if (m->ev) KRK_HIP(hipStreamWaitEvent(s, m->ev, 0));
        pool_dep_done_(m);
        ++expert_pool_.order_device;
    }"""),
    ("""    void pool_honor_host_(void *dst) {
        const hipEvent_t ev = pool_take_dep_(dst);
        if (!ev) return;
        KRK_HIP(hipEventSynchronize(ev));
        ++expert_pool_.order_host;
    }""",
     """    void pool_honor_host_(void *dst) {
        PoolMark *m = pool_take_dep_(dst);
        if (!m) return;
        if (m->ev) KRK_HIP(hipEventSynchronize(m->ev));
        pool_dep_done_(m);
        ++expert_pool_.order_host;
    }"""),
    ("""    void pool_honor_implicit_(void *dst) {
        if (pool_take_dep_(dst)) ++expert_pool_.order_implicit;
    }""",
     """    void pool_honor_implicit_(void *dst) {
        PoolMark *m = pool_take_dep_(dst);
        if (!m) return;
        pool_dep_done_(m);
        ++expert_pool_.order_implicit;
    }"""),

    # 4. killing a cell for good
    ("""    void pool_kill_(PoolBlock &b) {
        pool_pending_.erase(b.p);
        if (b.ev) {
            KRK_HIP(hipEventSynchronize(b.ev));
            KRK_HIP(hipEventDestroy(b.ev));
            b.ev = nullptr;
        }
        KRK_HIP(hipFree(b.p));
    }""",
     """    void pool_kill_(PoolBlock &b) {
        pool_drop_pending_(b.p);
        if (b.m) {
            // The memory is about to be freed: whatever the mark covers has to
            // be done, or a queued reader is reading freed memory.
            if (b.m->ev) KRK_HIP(hipEventSynchronize(b.m->ev));
            pool_mark_unref_(b.m);
            b.m = nullptr;
        }
        KRK_HIP(hipFree(b.p));
    }"""),

    # 5. the reuse hand-out
    ("""    static bool pool_host_wait_() {""",
     """    // KRK_POOL_BURST=0 restores the release path this replaces: one
    // hipEventRecord per slice instead of one per burst. It is the A/B arm, so
    // the comparison is one process apart.
    static bool pool_burst_off_() {
        static const bool v = [] {
            const char *e = std::getenv("KRK_POOL_BURST");
            return e && e[0] == '0';
        }();
        return v;
    }
    static bool pool_host_wait_() {"""),
    ("""            if (pool_host_wait_()) {
                for (size_t i = 0; i < cells.size(); ++i) {
                    const hipError_t rc =
                        cells[i].ev ? hipEventQuery(cells[i].ev) : hipSuccess;
                    if (rc == hipSuccess) { pick = i; ready = true; break; }
                    if (rc != hipErrorNotReady) KRK_HIP(rc);
                }
            }
            PoolBlock b = cells[pick];
            cells.erase(cells.begin() + pick);
            if (b.ev) {
                if (pool_host_wait_()) {
                    if (!ready) {
                        KRK_HIP(hipEventSynchronize(b.ev));
                        ++expert_pool_.order_host;
                    }
                } else {""",
     """            if (pool_host_wait_()) {
                for (size_t i = 0; i < cells.size(); ++i) {
                    const hipError_t rc = cells[i].m
                                              ? hipEventQuery(cells[i].m->ev)
                                              : hipSuccess;
                    if (rc == hipSuccess) { pick = i; ready = true; break; }
                    if (rc != hipErrorNotReady) KRK_HIP(rc);
                }
            }
            PoolBlock b = cells[pick];
            cells.erase(cells.begin() + pick);
            if (b.m) {
                if (pool_host_wait_()) {
                    if (!ready) {
                        KRK_HIP(hipEventSynchronize(b.m->ev));
                        ++expert_pool_.order_host;
                    }
                } else {"""),
    ("""                    pool_pending_[b.p] = b.ev;
                }
            }
            expert_pool_.reusable -= rounded;
            expert_pool_.live += rounded;
            ++expert_pool_.reuses;
            pool_live_.emplace(b.p, LiveCell{rounded, b.ev});
            return b.p;""",
     """                    // The hand-out takes its own reference: the mark must
                    // outlive the cell's own reference if the cell is released
                    // again before a writer consumes this dependency.
                    pool_mark_ref_(b.m);
                    pool_pending_[b.p] = b.m;
                }
            }
            expert_pool_.reusable -= rounded;
            expert_pool_.live += rounded;
            ++expert_pool_.reuses;
            // The cell's own reference moves from the free list to the cell.
            pool_live_.emplace(b.p, LiveCell{rounded, b.m});
            return b.p;"""),

    # 6. the release path
    ("""    void release_pooled(void *p, size_t bytes) override {
        if (!p) return;
        const size_t rounded = pooled_size(bytes);
        const auto live = pool_live_.find(p);
        if (live == pool_live_.end() || live->second.rounded != rounded) {
            KRK_ERROR("invalid expert cell release (pointer, size or ownership)");
            std::abort();
        }
        hipEvent_t ev = live->second.ev; // the cell's own guard, if it has one
        pool_live_.erase(live);
        expert_pool_.live -= rounded;
        const char *inject = std::getenv("KRK_TEST_POOL_NO_EVENT");
        hipError_t rc = inject && inject[0] == '1' ? hipErrorOutOfMemory : hipSuccess;
        if (rc == hipSuccess && !ev)
            rc = hipEventCreateWithFlags(&ev, hipEventDisableTiming);
        // The mark is the cell's release point on the default stream: every
        // reader queued for it so far is ordered before this record.
        if (rc == hipSuccess) rc = hipEventRecord(ev, 0);
        if (rc != hipSuccess) {
            // Recover lifetime, not ownership: only base allocations live here.
            if (ev) KRK_HIP(hipEventDestroy(ev));
            ev = nullptr;
            pool_pending_.erase(p);
            KRK_HIP(hipDeviceSynchronize());
            ++expert_pool_.guard_fallbacks;
            if (!expert_arena()) {
                KRK_HIP(hipFree(p));
                expert_pool_.committed -= rounded;
                return;
            }
        }
        if (!expert_arena() || expert_pool_.committed > expert_pool_.budget) {
            PoolBlock b{p, ev};
            pool_kill_(b);
            expert_pool_.committed -= rounded;
        } else {
            pool_[rounded].push_back(PoolBlock{p, ev});
            expert_pool_.reusable += rounded;
        }
    }""",
     """    void release_pooled(void *p, size_t bytes) override {
        void *ps[1] = {p};
        const size_t bs[1] = {bytes};
        release_batch_(ps, bs, 1);
    }
    // An expert is THREE slices and an eviction returns all three at once, so
    // this is the shape the decode loop pays at a constrained budget. One
    // hipEventRecord per slice was 23 us x 5560 cells ~= 128 ms; one per burst
    // is a third of that. `KRK_POOL_BURST=0` restores the per-slice path.
    void release_pooled_batch(void *const ps[], const size_t bytes[],
                              int n) override {
        if (n <= 1 || n > kPoolBurstMax || pool_burst_off_()) {
            for (int i = 0; i < n; i++) {
                void *q = ps[i];
                const size_t b = bytes[i];
                release_batch_(&q, &b, 1);
            }
            return;
        }
        release_batch_(ps, bytes, n);
    }
    // The one release path, for a single cell and for a burst. All `n` cells are
    // released under ONE mark recorded at one default-stream position.
    void release_batch_(void *const ps[], const size_t bytes[], int n) {
        if (n <= 0) return;
        size_t rounded[kPoolBurstMax] = {};
        for (int i = 0; i < n; i++) {
            if (!ps[i]) continue;
            const size_t r = pooled_size(bytes[i]);
            const auto it = pool_live_.find(ps[i]);
            if (it == pool_live_.end() || it->second.rounded != r) {
                KRK_ERROR("invalid expert cell release (pointer, size or ownership)");
                std::abort();
            }
            rounded[i] = r;
        }
        // The mark is the release point on the default stream: every reader
        // queued for any cell in this burst so far is ordered before this
        // record, which is exactly what one record per slice expressed.
        PoolMark *m = pool_mark_open_();
        if (m) {
            const char *inject = std::getenv("KRK_TEST_POOL_NO_EVENT");
            hipError_t rc = inject && inject[0] == '1' ? hipErrorOutOfMemory
                                                       : hipSuccess;
            if (rc == hipSuccess) rc = hipEventRecord(m->ev, 0);
            if (rc != hipSuccess) {
                pool_mark_close_(m);
                m = nullptr;
            }
        }
        if (m) {
            ++expert_pool_.release_marks;
        } else {
            // Recover lifetime, not ownership: only base allocations live here.
            // The sync stands in for the mark -- after it no reader of any cell
            // in the burst is outstanding, so the cells may go back unmarked.
            KRK_HIP(hipDeviceSynchronize());
            ++expert_pool_.guard_fallbacks;
        }
        for (int i = 0; i < n; i++) {
            if (!ps[i]) continue;
            const size_t r = rounded[i];
            const auto it = pool_live_.find(ps[i]);
            pool_mark_unref_(it->second.m); // the cell's previous mark
            pool_live_.erase(it);
            expert_pool_.live -= r;
            pool_drop_pending_(ps[i]);
            if (!expert_arena() || expert_pool_.committed > expert_pool_.budget) {
                if (m && m->ev) KRK_HIP(hipEventSynchronize(m->ev));
                KRK_HIP(hipFree(ps[i]));
                expert_pool_.committed -= r;
            } else {
                pool_mark_ref_(m); // the cell's own reference to the new mark
                pool_[r].push_back(PoolBlock{ps[i], m});
                expert_pool_.reusable += r;
            }
            ++expert_pool_.release_slices;
        }
        pool_mark_unref_(m); // the burst's reference
    }"""),

    # 7. the two direct take_dep_ sites in the paged ring
    ("""        const hipEvent_t dep = pool_take_dep_(d);
        for (size_t off = 0; off < bytes; off += paged_chunk_) {""",
     """        PoolMark *dep = pool_take_dep_(d);
        for (size_t off = 0; off < bytes; off += paged_chunk_) {"""),
    ("""            if (dep) {
                KRK_HIP(hipStreamWaitEvent(paged_stream_[slot], dep, 0));
                ++expert_pool_.order_device;
            }
            std::memcpy(paged_buf_[slot], s + off, len);
            KRK_HIP(hipMemcpyAsync(d + off, paged_buf_[slot], len,
                                   hipMemcpyHostToDevice, paged_stream_[slot]));
            KRK_HIP(hipEventRecord(paged_ev_[slot], paged_stream_[slot]));
            paged_used_[slot] = true;
            if (nused < kPagedBufs) used[nused++] = slot;
        }
        paged_publish_(used, nused);""",
     """            if (dep) {
                KRK_HIP(hipStreamWaitEvent(paged_stream_[slot], dep->ev, 0));
                ++expert_pool_.order_device;
            }
            std::memcpy(paged_buf_[slot], s + off, len);
            KRK_HIP(hipMemcpyAsync(d + off, paged_buf_[slot], len,
                                   hipMemcpyHostToDevice, paged_stream_[slot]));
            KRK_HIP(hipEventRecord(paged_ev_[slot], paged_stream_[slot]));
            paged_used_[slot] = true;
            if (nused < kPagedBufs) used[nused++] = slot;
        }
        pool_dep_done_(dep);
        paged_publish_(used, nused);"""),
    ("""            const hipEvent_t dep = pool_take_dep_(d);
            for (size_t off = 0; off < bytes[i]; off += paged_chunk_) {""",
     """            PoolMark *dep = pool_take_dep_(d);
            for (size_t off = 0; off < bytes[i]; off += paged_chunk_) {"""),
    ("""                if (dep) {
                    KRK_HIP(hipStreamWaitEvent(paged_stream_[slot], dep, 0));
                    ++expert_pool_.order_device;
                }""",
     """                if (dep) {
                    KRK_HIP(hipStreamWaitEvent(paged_stream_[slot], dep->ev, 0));
                    ++expert_pool_.order_device;
                }"""),
    ("""                batch_bytes += static_cast<i64>(len);
                paged_used_[slot] = true;
                if (nused < kPagedBufs) used[nused++] = slot;
            }
        }
        krk_op_scope.bytes(batch_bytes);
        paged_publish_(used, nused);""",
     """                batch_bytes += static_cast<i64>(len);
                paged_used_[slot] = true;
                if (nused < kPagedBufs) used[nused++] = slot;
            }
            pool_dep_done_(dep);
        }
        krk_op_scope.bytes(batch_bytes);
        paged_publish_(used, nused);"""),

    # 8. teardown
    ("""        // Free cells release their guard event with the cell; live cells are
        // released (with their events) by their owner's clear() before this.
        for (auto &entry : pool_) for (PoolBlock &b : entry.second) {
            if (b.ev) {
                (void)hipEventSynchronize(b.ev);
                (void)hipEventDestroy(b.ev);
                b.ev = nullptr;
            }
            KRK_HIP(hipFree(b.p));
        }
        for (const auto &entry : pool_live_) KRK_HIP(hipFree(entry.first));
        pool_.clear();
        pool_live_.clear();""",
     """        // Free cells release their mark reference with the cell; live cells are
        // released (with theirs) by their owner's clear() before this. The marks
        // that come back land on pool_free_marks_, which is drained below.
        for (auto &entry : pool_) for (PoolBlock &b : entry.second) {
            if (b.m) {
                if (b.m->ev) (void)hipEventSynchronize(b.m->ev);
                pool_mark_unref_(b.m);
                b.m = nullptr;
            }
            KRK_HIP(hipFree(b.p));
        }
        for (const auto &entry : pool_live_) {
            pool_mark_unref_(entry.second.m);
            KRK_HIP(hipFree(entry.first));
        }
        for (const auto &entry : pool_pending_) pool_mark_unref_(entry.second);
        pool_pending_.clear();
        for (PoolMark *m : pool_free_marks_) {
            if (m->ev) (void)hipEventDestroy(m->ev);
            delete m;
        }
        pool_free_marks_.clear();
        pool_.clear();
        pool_live_.clear();"""),
])

# ------------------------------------------------------------ expert_cache.cpp
patch("src/expert_cache.cpp", [
    ("""void ExpertCache::drop(Slot &s) {
    if (!be_) return;
    finish_stage(s);
    QuantTensor *q[3] = {&s.m.gate, &s.m.up, &s.m.down};
    for (int i = 0; i < 3; i++) {
        if (q[i]->present() && q[i]->data)
            be_->release_pooled(q[i]->data, s.slice[i]);
        *q[i] = {};
    }
    s.m = {};
}""",
     """void ExpertCache::drop(Slot &s) {
    if (!be_) return;
    finish_stage(s);
    // An expert's three slices go back in ONE call so the pool issues one
    // release mark for the three instead of one per slice. An eviction is the
    // one place the decode loop pays this, once per promotion at a constrained
    // budget, and the three releases are the same default-stream point.
    void *cells[3] = {};
    size_t lens[3] = {};
    QuantTensor *q[3] = {&s.m.gate, &s.m.up, &s.m.down};
    int n = 0;
    for (int i = 0; i < 3; i++) {
        if (q[i]->present() && q[i]->data) {
            cells[n] = q[i]->data;
            lens[n] = s.slice[i];
            n++;
        }
        *q[i] = {};
    }
    s.m = {};
    if (n) be_->release_pooled_batch(cells, lens, n);
}"""),
    ("""        if (!p) {
            for (int j = 0; j < 3; j++)
                if (to[j]->present() && to[j]->data)
                    be_->release_pooled(to[j]->data, s.slice[j]);
            return false;
        }""",
     """        if (!p) {
            // Same burst shape as drop(): the half-promoted slices go back at
            // one stream point, so they share one release mark.
            void *cells[3] = {};
            size_t lens[3] = {};
            int n = 0;
            for (int j = 0; j < 3; j++)
                if (to[j]->present() && to[j]->data) {
                    cells[n] = to[j]->data;
                    lens[n] = s.slice[j];
                    n++;
                }
            if (n) be_->release_pooled_batch(cells, lens, n);
            return false;
        }"""),
])

# ---------------------------------------------------------------- main_cli.cpp
patch("src/main_cli.cpp", [
    ("""    std::fprintf(out, "[stats ] expert-pool budget %zu committed %zu live %zu reusable %zu bytes; "
                     "alloc %llu reuse %llu trim %llu guard-fallback %llu\\n",
                 pool.budget, pool.committed, pool.live, pool.reusable,
                 static_cast<unsigned long long>(pool.allocations),
                 static_cast<unsigned long long>(pool.reuses),
                 static_cast<unsigned long long>(pool.trims),
                 static_cast<unsigned long long>(pool.guard_fallbacks));""",
     """    std::fprintf(out, "[stats ] expert-pool budget %zu committed %zu live %zu reusable %zu bytes; "
                     "alloc %llu reuse %llu trim %llu guard-fallback %llu | "
                     "release %llu mark(s) for %llu slice(s)\\n",
                 pool.budget, pool.committed, pool.live, pool.reusable,
                 static_cast<unsigned long long>(pool.allocations),
                 static_cast<unsigned long long>(pool.reuses),
                 static_cast<unsigned long long>(pool.trims),
                 static_cast<unsigned long long>(pool.guard_fallbacks),
                 static_cast<unsigned long long>(pool.release_marks),
                 static_cast<unsigned long long>(pool.release_slices));"""),
])
