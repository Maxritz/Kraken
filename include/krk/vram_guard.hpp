#pragma once
// The one place that decides how much VRAM (and host RAM) a run refuses to
// spend, and whether an explicitly requested expert budget can fit on the card.
//
// Why this is its own file. On 2026-10-11 `--expert-warmup` filled a 15.92 GiB
// card to `0.00 GiB free` and took the machine's display down, twice. The budget
// it filled to had been chosen at plan time, before the workspaces and the KV
// tier existed, so nothing at the moment of spending checked anything. The guard
// that closes it has two halves that must agree -- the plan-time refusal/clamp,
// and the live per-round check the warm-up makes -- and both are pure
// arithmetic, and their shared pageable-memory admission is process-local state
// protected by a mutex. That is what this header is: no device dependency, so
// tests can drive the policy without opening one.
//
// The reserves are floors, not measurements of what any particular desktop
// needs. `KRK_VRAM_RESERVE_MB` / `KRK_RAM_RESERVE_MB` override them; setting
// either to `0` is the A/B arm that restores the pre-guard behaviour.
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

namespace krk {

// -1 when the variable is unset or empty, so `>= 0` means "the caller said so".
inline long long guard_env_mib(const char *name) {
    const char *v = std::getenv(name);
    return (v && *v) ? std::atoll(v) : -1;
}

inline size_t guard_mib_to_bytes(long long mib) {
    if (mib <= 0) return 0;
    const size_t unit = 1024u * 1024u;
    const size_t max_value = static_cast<size_t>(-1);
    if (static_cast<unsigned long long>(mib) > max_value / unit)
        return max_value;
    return static_cast<size_t>(mib) * unit;
}

// 1 GiB or a sixteenth of the card, whichever is larger. The floor is what
// matters on a small card: 15.92 GiB yields 1.0 GiB, so a card this size keeps
// the desktop alive instead of holding exactly the reserve's fraction.
inline size_t vram_reserve_bytes(size_t total_vram) {
    const long long env = guard_env_mib("KRK_VRAM_RESERVE_MB");
    if (env >= 0) return guard_mib_to_bytes(env);
    const size_t gib = static_cast<size_t>(1) << 30;
    const size_t share = total_vram / 16u;
    return share > gib ? share : gib;
}

// The floor a BUDGET must leave free, as opposed to the reserve above, which is
// what the driver and the desktop may take. The extra gigabyte is the
// allocator's own working room, and it is the difference between a full card
// and a pathological one. Measured on a 15.9 GiB card with a 22.1 GiB routed
// expert set (Tiel-Coder-35B-A3B Q5, 256 experts, top-8, ctx 4096): with 0.00
// GiB free a 2.2 MiB pool cell takes 30 ms to allocate and decode is 3.1 tok/s;
// with 2.27 GiB free the same cell takes 0.29 ms and decode is 15.8 tok/s --
// same weights, same text. The allocator, not the model.
// KRK_VRAM_FLOOR_MB overrides both the reserve and the extra gigabyte.
inline size_t vram_floor_bytes(size_t total_vram) {
    const long long env = guard_env_mib("KRK_VRAM_FLOOR_MB");
    if (env >= 0) return guard_mib_to_bytes(env);
    const size_t gib = static_cast<size_t>(1) << 30;
    // A fifth of the card, or the reserve plus a gigabyte on a small one. The
    // reserve alone is ~1 GiB here, and a card with 1 GiB free is one expert
    // admission away from 0 -- which does not just slow this run down, it
    // stalls the desktop while Windows pages the compositor out. A fifth is
    // 3.2 GiB on this 15.9 GiB card.
    const size_t share = total_vram / 5u;
    const size_t base = vram_reserve_bytes(total_vram) + gib;
    return share > base ? share : base;
}

// The fraction of installed memory a run may plan for -- VRAM and host RAM
// alike. 0.80 by default. This is a ceiling, not a measurement: two sessions
// took a desktop down by planning WARM against *free* RAM, and a 79 GB corpus
// on a 96 GiB box then filled the file cache until the machine paged itself to
// a stop. The same rule has to hold at the small end, because the deployment
// targets are 6/8/12 GiB cards with 16/24/32 GiB of RAM: 20% left to the OS is
// what keeps a 16 GiB machine responsive, and it is 20% here too.
// KRK_MEM_FRACTION overrides it (0.20..0.98; anything else is ignored).
inline double mem_fraction() {
    const char *v = std::getenv("KRK_MEM_FRACTION");
    if (!v || !*v) return 0.80;
    const double f = std::atof(v);
    return (f >= 0.20 && f <= 0.98) ? f : 0.80;
}

// Bytes this process may plan for, out of `installed`.
inline size_t mem_ceiling_bytes(size_t installed) {
    if (installed == 0) return 0;
    return static_cast<size_t>(static_cast<double>(installed) *
                               mem_fraction());
}

// The host twin: 2 GiB or a fifth of installed RAM. The ram policy already
// plans a 20% headroom for the WARM tier, but that plan is made once, before the
// KV tier, the workspaces and the model's own buffers exist -- and staging is
// what spends it, so the staging loop re-checks this floor per round. The
// SharedHostRamBudget ledger below ensures independent host tiers do not each
// claim the same free-memory sample.
inline size_t host_reserve_bytes(size_t installed) {
    const long long env = guard_env_mib("KRK_RAM_RESERVE_MB");
    if (env >= 0) return guard_mib_to_bytes(env);
    const size_t gib = static_cast<size_t>(1) << 30;
    // 20% at the default fraction, expressed as installed minus ceiling so
    // the two stay one number. On 16/24/32 GiB that is 3.2/4.8/6.4 GiB.
    const size_t ceiling = mem_ceiling_bytes(installed);
    const size_t share = installed > ceiling ? installed - ceiling : installed / 8u;
    return share > 2 * gib ? share : 2 * gib;
}

// Does staging another `need` bytes leave the reserve untouched? `pending` is
// the set of buffers already allocated for a batch read but not yet reflected
// in the OS's available-memory measurement (pageable allocations can commit on
// first write). Accounting for both makes a batched prefetch no different from
// a sequence of single admissions. If the OS cannot report available memory
// (available == 0), fail closed: an unmeasurable host is not permission to
// consume it.
inline bool host_staging_fits(size_t need, size_t available, size_t reserve,
                              size_t pending = 0) {
    if (available <= reserve) return false;
    const size_t room = available - reserve;
    return pending <= room && need <= room - pending;
}

// The automatic pinned WARM prefix, in bytes: the part of the host expert tier
// that is page-locked, so a promotion out of it is one DMA instead of a
// bounce-buffer copy through a 256 MiB BAR aperture. Measured on this box (RX
// 9070 XT, 256 MiB aperture, 15.9 GiB): REAP-320 42 GiB routed / 24 GiB WARM
// 101.7 -> 88.2 ms/step, a 35B-A3B at 9.4 GiB WARM 35.4 -> 30.1 ms/step -- the
// xfer rate per expert halves, 386 -> 193 us.
//
// The rule used to keep HALF the tier pageable ("a small box still has a
// pageable half") and pin at most an eighth of installed RAM. The half-tier
// clamp was measured wrong at the shape a promotion actually has, and it is the
// clamp that mattered: on the 256 MiB expert arm (Qwen3.8-Distill-35B-A3B Q2KXL,
// 40 layers x 256 experts, WARM 9600 MiB, 24 greedy tokens, identical stdout in
// every arm) the tier auto-pinned 4800 MiB and decode ran 139.8 ms/step with
// xfer 2815 ms of 3216 -- 469 us per promotion at 2.0 GB/s. Pinning the whole
// 9600 MiB took it to 65.3 ms/step, 15.3 tok/s, xfer 27 ms of host enqueue, 10 us
// per promotion. The pageable half was not a spare copy of anything: it is the
// half every miss lands on.
//
// So the bound is now a share of installed RAM only -- pinned pages cannot be
// swapped, and this runs on laptops with 16 GiB -- and the tier is pinned WHOLE
// whenever it fits inside that share. A quarter (24 GiB on a 96 GiB box), capped
// at 16 GiB so a workstation does not page-lock a third of itself for one model;
// 0 means pin nothing (no tier, an unknown host, or a tier smaller than a page).
//
// What this does NOT claim: that the H2D probe's ratio is the reason. The probe
// times 16-64 MiB copies, and at the ~1 MiB shape of one expert's promotion the
// pageable path measured 2.0 GB/s against the 9.9-10.6 GB/s the probe reports --
// so the probe is a floor on that path's cost, not a model of it.
inline size_t pinned_warm_prefix_bytes(size_t warm_bytes, size_t ram_installed) {
    if (warm_bytes == 0 || ram_installed == 0) return 0;
    if (warm_bytes < 65536) return 0;  // a tier smaller than a page is not an arena
    const size_t gib = static_cast<size_t>(1) << 30;
    size_t want = ram_installed / 4u;  // 24 GiB on a 96 GiB box
    const size_t cap = 16 * gib;       // no run claims more than this
    if (want > cap) want = cap;
    if (want > warm_bytes) want = warm_bytes;  // the whole tier when it fits
    return want;
}

// Is a pinned prefix worth its non-swappable RAM? Both numbers are MEASURED at
// init by timing the two promotion paths at the shape a promotion uses:
// `upload_paged_batch` (stage into the bounded pinned ring, then DMA) against
// `upload_pinned_batch` (DMA straight out of the caller's buffer). A ratio, not
// an absolute rate, because the question is only whether the pageable path pays
// a copy that the pin would remove.
//
// Measured on this box: 7.6 against 22.7 GB/s on a 120B MoE's promotion shape,
// 4.6 against ~9 GB/s on the REAP-320 file -- 2-3x. A card with a large
// aperture (Resizable BAR, i.e. a FIRMWARE setting rather than a property of
// the model) can do the pageable DMA without the ring's memcpy, and then the
// ratio falls towards 1 and the pin buys nothing but unswappable pages.
//
// 1.20x, and the asymmetry is deliberate: over-pinning costs RAM that cannot be
// swapped and nothing else (the measured pin curve is monotone and flat at the
// top), while under-pinning leaves 10-18% of decode on the table. The line is
// below every ratio measured here -- 1.60x at 16 MiB on the REAP-320 file,
// 1.84x on a 35B-A3B, 2-3x on a 120B MoE's promotion shape -- and above what an
// aperture with no host copy in the pageable path would give (~1.0-1.1x).
inline bool warm_pin_worthwhile(double pageable_gbs, double pinned_gbs) {
    if (!(pageable_gbs > 0.0) || !(pinned_gbs > 0.0)) return false;
    return pinned_gbs >= pageable_gbs * 1.20;
}

// Does the driver-staged pageable DMA beat the staging ring? Both move a
// pageable source; they differ in WHOSE copy stages it. The ring does a host
// memcpy into a bounded pinned buffer per chunk and hands that to the DMA, which
// is the only way through a narrow aperture. The direct path hands the pageable
// pointer to the driver, which stages it in a buffer of its own: on a narrow
// aperture that is one windowed bounce copy and it loses; on a wide one
// (Resizable BAR -- a FIRMWARE setting) the staging is free, and the caller's
// memcpy plus the per-chunk bookkeeping are gone.
//
// 1.20x, and that line is above a band that was MEASURED to flap. On this box
// (256 MiB aperture) the same two paths read 9.3 against 10.5 GB/s at 16 MiB
// (+13%) and 10.88 against 11.00 at 12.6 MB (+1%) -- the size alone moved the
// answer across a 1.10 line, and a path chosen by that would be chosen by the
// probe size rather than by the card. A wide aperture does not land in that
// band: it removes the host copy entirely, which is a 2x-class gap. Below the
// line the ring stays, because it owns an explicitly bounded buffer and a known
// fallback while the direct path depends on driver staging this source tree
// cannot see. Above it, `warm_pin_worthwhile` then declines the pin -- there is
// no copy left to buy back.
inline bool pageable_direct_wins(double ring_gbs, double direct_gbs) {
    if (!(ring_gbs > 0.0) || !(direct_gbs > 0.0)) return false;
    return direct_gbs >= ring_gbs * 1.20;
}

// Process-wide pageable-memory admission shared by the expert WARM tier and
// KV WARM/COLD staging paths. Per-cache checks against the OS alone race: both
// tiers can observe the same free headroom and independently spend it. The
// ledger tracks allocations and reservations; live OS samples also catch
// pressure from other processes. Its byte ceiling shrinks under external
// pressure and resets after all tracked allocations have been freed.
class SharedHostRamBudget {
public:
    static bool can_admit(size_t need, size_t available, size_t installed) {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        refresh(s, available, installed);
        return fits(s, need, available, installed);
    }

    static bool reserve(size_t need, size_t available, size_t installed) {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        refresh(s, available, installed);
        if (!fits(s, need, available, installed)) return false;
        if (need > static_cast<size_t>(-1) - s.pending) return false;
        s.pending += need;
        return true;
    }

    static bool commit(void *ptr, size_t bytes) {
        if (!ptr) return false;
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const auto existing = s.allocations.find(ptr);
        if (existing != s.allocations.end()) {
            s.pending = bytes > s.pending ? 0 : s.pending - bytes;
            return true;
        }
        try {
            s.allocations.emplace(ptr, bytes);
        } catch (...) {
            return false;
        }
        s.pending = bytes > s.pending ? 0 : s.pending - bytes;
        const size_t max_value = static_cast<size_t>(-1);
        s.used = bytes > max_value - s.used ? max_value : s.used + bytes;
        return true;
    }

    static void cancel(size_t bytes) {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        s.pending = bytes > s.pending ? 0 : s.pending - bytes;
    }

    // Called only once the pageable allocation has actually been freed. A HIP
    // deferred free retains its ledger entry until its copy-completion event.
    template <typename FreeFn>
    static void release_and_free(void *ptr, FreeFn free_fn) {
        if (!ptr) return;
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const auto it = s.allocations.find(ptr);
        if (it != s.allocations.end()) {
            s.used = it->second > s.used ? 0 : s.used - it->second;
            s.allocations.erase(it);
        }
        // Keep the lock through free(): an allocator may immediately reuse
        // this address, and a concurrent commit must not alias the old ledger
        // entry before it has been removed.
        free_fn(ptr);
    }

    static size_t used_bytes() {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        return s.used;
    }

private:
    struct State {
        std::mutex mutex;
        std::unordered_map<void *, size_t> allocations;
        size_t used = 0;
        size_t pending = 0;
        size_t limit = 0;
    };

    static State &state() {
        static State s;
        return s;
    }

    static size_t saturating_add(size_t a, size_t b) {
        const size_t max_value = static_cast<size_t>(-1);
        return b > max_value - a ? max_value : a + b;
    }

    static void refresh(State &s, size_t available, size_t installed) {
        const size_t reserve = host_reserve_bytes(installed);
        // Add both committed and pending reservations back to available RAM:
        // an allocation may or may not have reached the OS's live sample yet.
        // This estimates external pressure without counting our own bytes twice.
        const size_t live_total = saturating_add(
            saturating_add(available, s.used), s.pending);
        const size_t observed_limit = live_total > reserve ? live_total - reserve : 0;
        if (s.used == 0 && s.pending == 0) {
            s.limit = observed_limit;
        } else if (observed_limit < s.limit) {
            s.limit = observed_limit;
        }
    }

    static bool fits(const State &s, size_t need, size_t available, size_t installed) {
        if (available == 0 || available <= host_reserve_bytes(installed)) return false;
        if (s.used > s.limit || s.pending > s.limit - s.used) return false;
        const size_t room = s.limit - s.used - s.pending;
        const size_t live_room = available - host_reserve_bytes(installed);
        return need <= (room < live_room ? room : live_room);
    }
};

// A run with no device can still be asked to behave as if it had one, which is
// the only way the memory policy can be driven END TO END from the test suite:
// the refusal below needs a card that is too small for the budget the caller
// asked for. Read by the CPU backend, and only there, and only while it has no
// capacity of its own to report -- a real backend's own answer always wins, so
// this cannot make a GPU run refuse anything. The simulated figure is what the
// run reports as its card, in the `[stats ] device` line included, because it is
// the number every decision in the run was made against.
inline size_t simulated_total_bytes() {
    const long long mib = guard_env_mib("KRK_VRAM_TOTAL_MB");
    return guard_mib_to_bytes(mib);
}

// The largest budget this card can hold, with `reserve` left over.
inline size_t expert_budget_room(size_t total_vram, size_t used, size_t reserve) {
    if (used >= total_vram) return 0;
    const size_t available = total_vram - used;
    return available > reserve ? available - reserve : 0;
}

// Can the explicit budget be held? A budget the card cannot hold is REFUSED and
// not trimmed: silently keeping less than the user asked for turns their
// experiment into a different one, and the run would then report a residency
// they did not choose. `why` gets a message naming the budget asked for, what
// the card has, and the largest that fits -- one flag away from a working run.
//
// `total_vram == 0` means the backend cannot say (the CPU backend), and an
// unanswerable question is not a refusal: it fits, as it always did.
inline bool expert_budget_fits(size_t budget, size_t total_vram, size_t used,
                               size_t reserve, size_t *largest_fits,
                               std::string *why) {
    const size_t room = expert_budget_room(total_vram, used, reserve);
    if (largest_fits) *largest_fits = room;
    if (total_vram == 0 || budget <= room) return true;
    if (why) {
        char b[512];
        std::snprintf(
            b, sizeof(b),
            "expert budget %.0f MiB does not fit: %.0f MiB of VRAM, %.0f MiB "
            "already in use, %.0f MiB reserved -> the largest budget this card "
            "can hold is %.0f MiB (--expert-cache-mb %.0f)",
            static_cast<double>(budget) / 1048576.0,
            static_cast<double>(total_vram) / 1048576.0,
            static_cast<double>(used) / 1048576.0,
            static_cast<double>(reserve) / 1048576.0,
            static_cast<double>(room) / 1048576.0,
            static_cast<double>(room) / 1048576.0);
        *why = b;
    }
    return false;
}

} // namespace krk
