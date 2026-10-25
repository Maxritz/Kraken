#pragma once
// The host->device promotion probe: which PAGEABLE path to run, and whether a
// pinned WARM prefix buys anything over it.
//
// Why this is its own header. Both answers are measurements of the card, and a
// measurement is only worth having if it can be shown able to answer the other
// way. `Engine::init` used to time the arms inline in a TU the suite cannot
// reach, so the branch where a wide aperture wins had no witness at all: this
// machine reports a 256 MiB aperture, the probe keeps the staging ring every
// time, and the only way to see the other branch was to force `KRK_WARM_DMA=2`
// -- which bypasses the decision entirely. The probe is plain portable C++ over
// the `Backend` virtuals, so moving it here lets a test drive it with a backend
// whose two pageable paths really do cost different amounts of time.
//
// Three arms, and the shape of the comparison is the lesson:
//   ring   -- `upload_paged_batch` staging a pageable source through the bounded
//             pinned ring: a host memcpy per chunk, then a DMA. The only way to
//             reach the bus through a narrow aperture, because the driver's own
//             staging then serialises through that window.
//   direct -- the same call with the direct arm on: the driver stages the
//             pageable source itself, on one side stream. A narrow aperture
//             makes that one windowed bounce copy and it loses; a wide one
//             (Resizable BAR -- a FIRMWARE setting, not a card constant) makes
//             it free, and the caller's memcpy plus the ring's bookkeeping go.
//   pinned -- `upload_pinned_batch`, the DMA straight out of the caller's
//             buffer, i.e. what the pinned WARM prefix buys.
//
// INTERLEAVED AND ORDER-BALANCED, with a MEDIAN per arm. Three sequential arms
// read the order as the path: on this box the same pair measured 13.9 GB/s
// (ring) then 10.5 (direct) at 12.6 MB, and 10.9 then 12.8 at 16 MiB in the
// other order, which flipped the decision between two runs of one binary. Each
// iteration now runs both pageable paths and swaps which goes first, and pinned
// alternates between the head and the tail of the triple.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <direct.h> // _mkdir, for the default cache directory
#else
#include <sys/stat.h> // mkdir
#endif

#include "krk/backend.hpp"
#include "krk/vram_guard.hpp" // pageable_direct_wins / warm_pin_worthwhile: the listing quotes the engine's own predicates

namespace krk {

// The probe's copy size, and why it is a BUCKET rather than the model's own
// expert footprint. It is clamped to 16-64 MiB -- below the floor the launch
// and the sync are a large part of each arm and compress the ratio the decision
// is made on (measured on the REAP-320 file: 1.60x at 16 MiB against 1.41-1.69x
// at 4 MiB), and past the ceiling init would pay for copies that say nothing
// new -- and rounded UP to the next power of two. That keeps the set of sizes a
// machine can hold a verdict for at {16, 32, 64} MiB, which is what lets
// `kraken-bench --h2d` fill the cache for every model a normal run can ask
// about: with the model's exact footprint in the key, a 20 MiB expert would look
// up an entry no tool ever writes.
inline constexpr size_t kH2dProbeMinBytes = static_cast<size_t>(16) << 20;
inline constexpr size_t kH2dProbeMaxBytes = static_cast<size_t>(64) << 20;

inline size_t h2d_probe_size_bytes(size_t max_expert_bytes) {
    size_t bytes = kH2dProbeMinBytes;
    while (bytes < max_expert_bytes && bytes < kH2dProbeMaxBytes) bytes <<= 1;
    return bytes;
}

// The raw per-iteration rates, one vector per arm. Returned so the stopping rule
// can be fitted to measured noise instead of a guessed constant -- `kraken-bench
// --h2d-noise` prints exactly these -- and so a test can drive the rule with
// synthetic samples instead of this box's scheduler.
struct H2dSamples {
    std::vector<double> ring;
    std::vector<double> direct;
    std::vector<double> pinned;
    std::vector<double> blocking;
};

struct H2dProbe {
    bool valid = false;
    double pageable_gbs = 0.0;  // the staging ring
    double direct_gbs = 0.0;    // the driver-staged pageable DMA
    double blocking_gbs = 0.0;  // the plain blocking hipMemcpy
    // Measured, not asked: the best pageable arm reached at least half the
    // pinned rate, which only happens when the driver can write straight into
    // the aperture. See `aperture_is_wide` for the two populations.
    bool aperture_wide = false;
    double pinned_gbs = 0.0;
    // How many iterations it actually ran, and the worst relative MAD among the
    // three arms at the point it stopped. Published because the probe's own cost
    // is the thing the early stop buys, and "5 iterations" is only true when it
    // never settled.
    int iterations = 0;
    double spread = 0.0;
};

// The chosen path (0 ring, 1 direct, 2 blocking, -1 unmeasured) and the rate it
// will actually pay. ONE definition, because the engine's init, the pin
// decision, the log line, `kraken-bench --h2d` and the cache listing all have to
// name the same mechanism -- a listing that printed "ring" where init would run
// the blocking copy would be worse than printing nothing.
inline int h2d_path(const H2dProbe &p) {
    return pageable_path_choice(p.pageable_gbs, p.direct_gbs, p.blocking_gbs);
}

inline double h2d_path_gbs(const H2dProbe &p) {
    switch (h2d_path(p)) {
        case 0: return p.pageable_gbs;
        case 1: return p.direct_gbs;
        case 2: return p.blocking_gbs;
        default: return 0.0;
    }
}

// Named in prose, because the log line is read by someone deciding whether the
// number describes their machine: "ring 8.4" and "the blocking copy 22.9" are
// different answers about the same card.
inline const char *h2d_path_name(int path) {
    switch (path) {
        case 0: return "the staging ring";
        case 1: return "the driver-staged DMA";
        case 2: return "the driver's blocking copy";
        default: return "no measured path";
    }
}

inline double h2d_median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Median absolute deviation over the median: a scale estimate that one stalled
// iteration cannot inflate, which a standard deviation would.
inline double h2d_relative_mad(const std::vector<double> &v) {
    if (v.size() < 2) return 1.0;  // one sample says nothing about its own noise
    const double m = h2d_median(v);
    if (!(m > 0.0)) return 1.0;
    std::vector<double> dev;
    dev.reserve(v.size());
    for (size_t i = 0; i < v.size(); i++) dev.push_back(std::fabs(v[i] - m));
    return h2d_median(dev) / m;
}

// --- the stop rule ---------------------------------------------------------
// The probe paid for 5 iterations at every size. The cost of one iteration is
// proportional to the copy, so the 64 MiB bucket paid 4x the 16 MiB bucket for
// samples it did not need -- and it is the size that needs the FEWEST, because
// the fixed per-call overhead that makes small copies noisy is amortised there.
//
// Measured, not assumed (`kraken-bench --h2d-noise <MiB>`, one size per fresh
// process so the cold driver state is charged to every size equally): relative
// MAD at n=3 was 1.8/0.4/3.9% at 64 MiB, 1.1/3.2/12.3% at 32 MiB (pinned still
// ramping), and 6.0/0.4/6.4% at 16 MiB -- where the medians keep moving to the
// cap, so 16 MiB keeps its five. A 10% spread therefore buys 3 iterations at
// 64 MiB, 4 at 32 and 5 at 16, which is the ordering the cost wants.
//
// Two conditions, because a tight-but-ramping sample is the failure mode: the
// MAD says the samples are close together, and the last two agreeing with the
// whole set says they have stopped moving. The 16 MiB ring at n=3 is exactly
// that case -- 6% MAD, but its last two read 12% above the median.
inline constexpr int kH2dProbeMinIters = 3;
// The cap, and it only binds where the samples refuse to settle: 16 MiB here.
// Five is what the probe always paid, so the change can only make it cheaper.
inline constexpr int kH2dProbeMaxIters = 5;
inline constexpr double kH2dProbeSettleSpread = 0.10;

inline bool h2d_samples_settled(const std::vector<double> &v, double spread) {
    if (v.size() < static_cast<size_t>(kH2dProbeMinIters)) return false;
    if (!(h2d_median(v) > 0.0)) return false;
    if (h2d_relative_mad(v) > spread) return false;
    // The median of the two newest samples, against the median of all of them.
    // Order matters here and a sorted copy would erase it.
    const double recent = 0.5 * (v[v.size() - 2] + v[v.size() - 1]);
    const double all = h2d_median(v);
    return std::fabs(recent - all) <= spread * all;
}

inline bool h2d_probe_settled(const H2dSamples &s, double spread) {
    return h2d_samples_settled(s.ring, spread) &&
           h2d_samples_settled(s.direct, spread) &&
           h2d_samples_settled(s.blocking, spread) &&
           h2d_samples_settled(s.pinned, spread);
}

// `collect_all` runs the full cap even when the medians settle: it is for the
// arm that FITS the rule (`--h2d-noise`), which needs the whole series the rule
// was chosen from. Asking for the samples is not enough on its own -- a caller
// may want the few that decided the verdict.
inline H2dProbe probe_h2d_path(Backend *be, size_t bytes, int iters,
                               H2dSamples *samples = nullptr,
                               bool collect_all = false) {
    H2dProbe r;
    // A backend with one pageable path would report the same number twice and
    // turn the ratio into noise; ask rather than assume.
    if (!be || bytes == 0 || iters <= 0 || !be->pageable_paths_selectable())
        return r;
    void *dev = be->alloc(bytes);
    void *pin = be->alloc_host(bytes);
    void *pag = be->alloc_host_pageable(bytes);
    if (!dev || !pin || !pag) {
        if (dev) be->release(dev);
        if (pin) be->release_host(pin);
        if (pag) be->release_host_pageable(pag);
        return r;
    }
    std::memset(pin, 1, bytes);
    std::memset(pag, 1, bytes);
    // One copy of one path. 0 = ring, 1 = direct, 2 = pinned, 3 = blocking.
    // GB/s, or 0 when the call could not be timed.
    auto one = [&](int mode) {
        void *dsts[1] = {dev};
        const void *srcs[1] = {mode == 2 ? pin : pag};
        const size_t lens[1] = {bytes};
        if (mode != 2)
            be->set_pageable_path(
                static_cast<Backend::PageablePath>(mode == 3 ? 2 : mode));
        const std::chrono::steady_clock::time_point t0 =
            std::chrono::steady_clock::now();
        if (mode == 2)
            be->upload_pinned_batch(dsts, srcs, lens, 1);
        else
            be->upload_paged_batch(dsts, srcs, lens, 1);
        be->sync();  // every arm returns before the copy lands
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        return ms > 0.0 ? static_cast<double>(bytes) / (ms * 1e6) : 0.0;
    };
    // One untimed copy of each first: the first pageable copy builds the staging
    // ring (or the driver's own staging), and charging that allocation to the
    // rate is a measurement of startup, not of the path.
    one(0);
    one(1);
    one(2);
    one(3);
    // The A/B arm: `KRK_H2D_PROBE_ITERS=<n>` runs EXACTLY n iterations, which is
    // the fixed quota the early stop replaced -- forcing the cap alone would still
    // stop early and measure the change against itself. Below the floor is
    // refused rather than honoured: a median over two samples is not a
    // measurement, and an arm that runs less than the rule allows would be a
    // different experiment.
    bool no_early_stop = false;
    if (const char *forced = std::getenv("KRK_H2D_PROBE_ITERS")) {
        const int n = std::atoi(forced);
        if (n >= kH2dProbeMinIters) {
            iters = n;
            no_early_stop = true;
        }
    }
    std::vector<double> ring_gbs_v, direct_gbs_v, pinned_gbs_v, block_gbs_v;
    ring_gbs_v.reserve(static_cast<size_t>(iters));
    direct_gbs_v.reserve(static_cast<size_t>(iters));
    block_gbs_v.reserve(static_cast<size_t>(iters));
    pinned_gbs_v.reserve(static_cast<size_t>(iters));
    for (int iter = 0; iter < iters; iter++) {
        // Reversed order on alternate iterations, so an arm cannot benefit from
        // running while the bus is still settling from its predecessor.
        if (iter & 1) {
            pinned_gbs_v.push_back(one(2));
            block_gbs_v.push_back(one(3));
            direct_gbs_v.push_back(one(1));
            ring_gbs_v.push_back(one(0));
        } else {
            ring_gbs_v.push_back(one(0));
            direct_gbs_v.push_back(one(1));
            block_gbs_v.push_back(one(3));
            pinned_gbs_v.push_back(one(2));
        }
        // `iters` is a CAP, not a quota: stop once the medians have settled, with
        // kH2dProbeMinIters as the floor a median needs to mean anything.
        if (!collect_all && !no_early_stop && iter + 1 >= kH2dProbeMinIters) {
            H2dSamples so_far;
            so_far.ring = ring_gbs_v;
            so_far.direct = direct_gbs_v;
            so_far.blocking = block_gbs_v;
            so_far.pinned = pinned_gbs_v;
            if (h2d_probe_settled(so_far, kH2dProbeSettleSpread)) break;
        }
    }
    const double ring_gbs = h2d_median(ring_gbs_v);
    const double direct_gbs = h2d_median(direct_gbs_v);
    const double block_gbs = h2d_median(block_gbs_v);
    const double pin_gbs = h2d_median(pinned_gbs_v);
    r.iterations = static_cast<int>(ring_gbs_v.size());
    // Not std::max: <windows.h> defines `max` as a macro, and this header is
    // included by TUs that have it.
    r.spread = h2d_relative_mad(ring_gbs_v);
    const double spread_direct = h2d_relative_mad(direct_gbs_v);
    const double spread_block = h2d_relative_mad(block_gbs_v);
    const double spread_pin = h2d_relative_mad(pinned_gbs_v);
    if (spread_direct > r.spread) r.spread = spread_direct;
    if (spread_block > r.spread) r.spread = spread_block;
    if (spread_pin > r.spread) r.spread = spread_pin;
    if (samples) {
        samples->ring = ring_gbs_v;
        samples->direct = direct_gbs_v;
        samples->blocking = block_gbs_v;
        samples->pinned = pinned_gbs_v;
    }
    be->release(dev);
    be->release_host(pin);
    be->release_host_pageable(pag);
    if (ring_gbs > 0.0 && pin_gbs > 0.0) {
        r.pageable_gbs = ring_gbs;
        r.direct_gbs = direct_gbs;
        r.blocking_gbs = block_gbs;
        r.pinned_gbs = pin_gbs;
        r.valid = true;
        r.aperture_wide = aperture_is_wide(h2d_path_gbs(r), pin_gbs);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Caching the verdict between runs
// ---------------------------------------------------------------------------
// The probe measures the CARD, not the model: its answer is stable across runs
// of the same binary, yet init paid for it every time (three arms, five
// iterations, ~20 ms) on a model that had not changed. The verdict is now
// cached per (card, probe size, aperture hint), so a machine measures once and
// then reuses it.
//
// Every part of the key is there because leaving it out was a real way to be
// wrong:
//   name + gfx  another card is another answer;
//   vram_total  a different card of the same family -- or a driver reporting a
//               different one -- was not the card that was measured;
//   large_bar   the runtime's aperture HINT. A firmware change (Above 4G
//               decoding + Resizable BAR) flips it, and that is exactly the
//               event that makes an old verdict wrong: the same card reporting
//               a different window is 0.4 GB/s through the ring against 20
//               direct;
//   bytes       the ratio is size-dependent here (1.01x at 12.6 MB against
//               1.17x at 16 MiB), so a verdict measured with one probe size is
//               not quoted for another;
//   version     the probe's own algorithm. Sequential arms measured the ORDER
//               as the path, so an entry written by that code must not outlive
//               it.
// Age is the last guard. Entries expire, `KRK_H2D_PROBE_RERUN=1` re-measures
// without disturbing the rest of the file, `KRK_H2D_CACHE=<path>` points the
// cache somewhere else and `KRK_H2D_CACHE=0` turns it off -- the suite forces
// the last one so a test run never reads or writes the user's measurement.
// Bumped when the LINE FORMAT or the probe's own algorithm changes. v2 added the
// driver version, so a v1 line is a miss -- a verdict whose driver is unknown is
// exactly the one an update invalidates -- and the next write replaces it.
// v3 added the blocking arm, because a v2 verdict cannot say whether the fastest
// pageable mechanism on this card is the one it never measured: reusing a v2
// line would pin the run to the ring/direct choice the old code could make.
inline constexpr int kH2dCacheVersion = 3;
// Fields per line: version, card, vram, BAR hint, driver, bytes, ring, direct,
// blocking, pinned, stamp.
inline constexpr size_t kH2dCacheFields = 11;
// 30 days: long enough that a workstation measures once a month, short enough
// that a BIOS or driver change is re-checked without anyone remembering to.
inline constexpr long long kH2dCacheMaxAgeSec = 30LL * 24 * 60 * 60;
inline constexpr size_t kH2dCacheMaxEntries = 32;
// Another DRIVER's verdict is kept only while a rollback is still a plausible
// next act -- you update, something regresses, you go back, within days. Past
// that window it is one of the three lines every update leaves behind, and it
// is dead weight in a file with a cap. A rollback after the window costs one
// re-measure (~30 ms), not correctness.
inline constexpr long long kH2dCacheOtherDriverMaxAgeSec = 7LL * 24 * 60 * 60;

struct H2dCacheKey {
    std::string name;  // caps().name + " " + caps().gfx
    size_t vram_total = 0;
    int large_bar = -1;
    // caps().driver_version, 0 when the backend cannot say. The driver is what
    // stages a pageable source, so a driver update is the second event -- after
    // the firmware -- that makes an old verdict wrong.
    int driver_version = 0;
    size_t bytes = 0;  // the probe's copy size
};

struct H2dCacheLookup {
    bool hit = false;
    H2dProbe probe;
    long long age_sec = 0;
};

inline long long h2d_now_sec() {
    const std::time_t t = std::time(nullptr);
    return static_cast<long long>(t);
}

// "" when caching is off: an empty value and an explicit 0/off both mean no
// cache, so a harness can disable it without removing the variable.
inline std::string h2d_probe_cache_path() {
    const char *env = std::getenv("KRK_H2D_CACHE");
    if (env) {
        if (!*env || std::strcmp(env, "0") == 0 || std::strcmp(env, "off") == 0)
            return std::string();
        return std::string(env);
    }
#if defined(_WIN32)
    const char *base = std::getenv("LOCALAPPDATA");
    if (base && *base) return std::string(base) + "\\kraken\\h2d-probe.cache";
    const char *tmp = std::getenv("TEMP");
    if (!tmp || !*tmp) tmp = std::getenv("TMP");
    return std::string(tmp && *tmp ? tmp : ".") + "\\krk-h2d-probe.cache";
#else
    const char *xdg = std::getenv("XDG_CACHE_HOME");
    if (xdg && *xdg) return std::string(xdg) + "/kraken/h2d-probe.cache";
    const char *home = std::getenv("HOME");
    if (home && *home) return std::string(home) + "/.cache/kraken/h2d-probe.cache";
    const char *tmp = std::getenv("TMPDIR");
    return std::string(tmp && *tmp ? tmp : "/tmp") + "/krk-h2d-probe.cache";
#endif
}

inline void h2d_make_parent_dir(const std::string &path) {
    const size_t cut = path.find_last_of("/\\");
    if (cut == std::string::npos || cut == 0) return;
    const std::string dir = path.substr(0, cut);
#if defined(_WIN32)
    _mkdir(dir.c_str());
#else
    mkdir(dir.c_str(), 0700);
#endif
}

inline std::vector<std::string> h2d_split_tabs(const std::string &line) {
    std::vector<std::string> out;
    size_t start = 0;
    for (;;) {
        const size_t tab = line.find('\t', start);
        out.push_back(line.substr(start, tab == std::string::npos ? std::string::npos
                                                                : tab - start));
        if (tab == std::string::npos) break;
        start = tab + 1;
    }
    return out;
}

inline bool h2d_parse_ll(const std::string &s, long long *out) {
    if (s.empty() || !out) return false;
    char *end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (!end || *end != '\0') return false;
    *out = v;
    return true;
}

inline bool h2d_parse_gbs(const std::string &s, double *out) {
    if (s.empty() || !out) return false;
    char *end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    // A negative or non-finite rate is a corrupt entry, not a slow path.
    if (!end || *end != '\0' || !(v >= 0.0) || !(v < 1e9)) return false;
    *out = v;
    return true;
}

// Fields 1-5 of a line (card, vram, BAR hint, driver, probe size) are the key.
// Shared by the reader and the writer's prune, so the two cannot disagree about
// which line a write replaces.
inline bool h2d_line_matches_key(const std::vector<std::string> &f,
                                 const H2dCacheKey &key) {
    if (f.size() != kH2dCacheFields) return false;
    long long vram = 0, bar = 0, driver = 0, bytes = 0;
    if (f[1] != key.name) return false;
    if (!h2d_parse_ll(f[2], &vram) ||
        static_cast<size_t>(vram < 0 ? 0 : vram) != key.vram_total)
        return false;
    if (!h2d_parse_ll(f[3], &bar) ||
        static_cast<int>(bar < -1 ? -1 : (bar > 2147483647LL ? 2147483647LL : bar)) !=
            key.large_bar)
        return false;
    if (!h2d_parse_ll(f[4], &driver) ||
        static_cast<int>(driver < 0
                             ? 0
                             : (driver > 2147483647LL ? 2147483647LL : driver)) !=
            key.driver_version)
        return false;
    if (!h2d_parse_ll(f[5], &bytes) ||
        static_cast<size_t>(bytes < 0 ? 0 : bytes) != key.bytes)
        return false;
    return true;
}

// Reads one entry matching `key`. A miss is the normal answer: a corrupt line,
// an entry from another card or size, an expired stamp and a clock that moved
// backwards all read as "measure it again" rather than as an error.
inline H2dCacheLookup h2d_probe_cache_read(const std::string &path,
                                           const H2dCacheKey &key, long long now) {
    H2dCacheLookup r;
    if (path.empty()) return r;
    std::ifstream in(path.c_str());
    if (!in) return r;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> f = h2d_split_tabs(line);
        if (f.size() != kH2dCacheFields) continue;
        long long ver = 0, stamp = 0;
        double ring = 0.0, direct = 0.0, block = 0.0, pin = 0.0;
        if (!h2d_parse_ll(f[0], &ver) || ver != kH2dCacheVersion) continue;
        if (!h2d_line_matches_key(f, key)) continue;
        if (!h2d_parse_gbs(f[6], &ring) || !h2d_parse_gbs(f[7], &direct) ||
            !h2d_parse_gbs(f[8], &block) || !h2d_parse_gbs(f[9], &pin))
            continue;
        if (!h2d_parse_ll(f[10], &stamp)) continue;
        if (stamp > now + 60) continue;  // clock moved backwards: do not trust it
        const long long age = now > stamp ? now - stamp : 0;
        if (age > kH2dCacheMaxAgeSec) continue;
        if (!(ring > 0.0) || !(pin > 0.0)) continue;
        r.hit = true;
        r.age_sec = age;
        r.probe.valid = true;
        r.probe.pageable_gbs = ring;
        r.probe.direct_gbs = direct;
        r.probe.blocking_gbs = block;
        r.probe.pinned_gbs = pin;
        r.probe.aperture_wide = aperture_is_wide(h2d_path_gbs(r.probe), pin);
        return r;
    }
    return r;
}

// What a rewrite did, so the pruning is visible instead of silent: dropping a
// verdict is a fact about the file a user may want to know, and the alternative
// -- a cache that quietly shrinks -- is indistinguishable from corruption.
struct H2dCachePrune {
    int expired = 0;       // past kH2dCacheMaxAgeSec
    int other_driver = 0;  // past the rollback window
    int unreadable = 0;    // wrong shape or version, or a stamp that cannot be trusted
    int evicted = 0;       // pushed out by the cap
    int replaced = 0;      // this key's previous line (maintenance, not a drop)
    int kept = 0;          // lines in the file after the write
    int dropped() const { return expired + other_driver + unreadable + evicted; }
};

// "" when nothing was dropped, so a caller prints this only when it happened: a
// rewrite happens on the first run after an update or a month of idleness, and
// logging "dropped 0" on every clean run is noise.
inline std::string h2d_prune_text(const H2dCachePrune &p) {
    if (p.dropped() == 0) return std::string();
    char b[256];
    std::snprintf(b, sizeof b,
                  "dropped %d dead verdict(s) (%d expired, %d from another driver, "
                  "%d unreadable, %d over the cap); %d kept",
                  p.dropped(), p.expired, p.other_driver, p.unreadable, p.evicted,
                  p.kept);
    return b;
}

// Rewrites the file: this key's line is replaced, and the rewrite is where
// dead verdicts are PRUNED -- expired lines, lines this build cannot read, and
// another driver's verdict older than the rollback window. A reader already
// refuses a dead line, so leaving it in place only denies the slot to a live
// one; with a cap of 32 and three lines per driver update, that fills up.
// Past the cap the NEWEST entries win, because the oldest is the one least
// likely to be asked for again. `prune` is filled in when the caller wants to
// report what happened. Best effort: a cache that cannot be written is a
// measurement next run, not a failure.
inline bool h2d_probe_cache_write(const std::string &path, const H2dCacheKey &key,
                                  const H2dProbe &probe, long long now,
                                  H2dCachePrune *prune = nullptr) {
    if (path.empty() || !probe.valid) return false;
    h2d_make_parent_dir(path);
    struct Kept {
        std::string line;
        long long stamp = 0;
    };
    std::vector<Kept> keep;
    {
        std::ifstream in(path.c_str());
        std::string line;
        size_t read = 0;
        // Bounded even for a pathological file: this is a cache, not a log.
        while (read++ < kH2dCacheMaxEntries * 4 && std::getline(in, line)) {
            if (line.empty()) continue;
            const std::vector<std::string> f = h2d_split_tabs(line);
            if (f.size() != kH2dCacheFields) {
                if (prune) ++prune->unreadable;
                continue;
            }
            long long ver = 0, stamp = 0, driver = 0;
            if (!h2d_parse_ll(f[0], &ver) || ver != kH2dCacheVersion) {
                if (prune) ++prune->unreadable;
                continue;
            }
            if (!h2d_parse_ll(f[10], &stamp)) {
                if (prune) ++prune->unreadable;
                continue;
            }
            if (stamp > now + 60) {  // clock moved backwards: not trusted
                if (prune) ++prune->unreadable;
                continue;
            }
            const long long age = now > stamp ? now - stamp : 0;
            if (age > kH2dCacheMaxAgeSec) {  // dead: the reader refuses it
                if (prune) ++prune->expired;
                continue;
            }
            if (h2d_parse_ll(f[4], &driver)) {
                const int line_driver = static_cast<int>(
                    driver < 0 ? 0
                               : (driver > 2147483647LL ? 2147483647LL : driver));
                if (line_driver != key.driver_version &&
                    age > kH2dCacheOtherDriverMaxAgeSec) {
                    if (prune) ++prune->other_driver;  // a rollback window passed
                    continue;
                }
            }
            if (h2d_line_matches_key(f, key)) {
                if (prune) ++prune->replaced;  // this write replaces it
                continue;
            }
            keep.push_back(Kept{line, stamp});
        }
    }
    while (keep.size() + 1 > kH2dCacheMaxEntries) {
        size_t oldest = 0;
        for (size_t i = 1; i < keep.size(); i++)
            if (keep[i].stamp < keep[oldest].stamp) oldest = i;
        keep.erase(keep.begin() + static_cast<long>(oldest));
        if (prune) ++prune->evicted;
    }
    std::string mine = key.name;
    for (size_t i = 0; i < mine.size(); i++)
        if (mine[i] == '\t' || mine[i] == '\n' || mine[i] == '\r') mine[i] = ' ';
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "%d\t%s\t%llu\t%d\t%d\t%llu\t%.4f\t%.4f\t%.4f\t%.4f\t%lld",
                  kH2dCacheVersion, mine.c_str(),
                  static_cast<unsigned long long>(key.vram_total), key.large_bar,
                  key.driver_version,
                  static_cast<unsigned long long>(key.bytes), probe.pageable_gbs,
                  probe.direct_gbs, probe.blocking_gbs, probe.pinned_gbs, now);
    keep.push_back(Kept{buf, now});
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp.c_str(), std::ios::binary | std::ios::trunc);
        if (!out) return false;
        for (size_t i = 0; i < keep.size(); i++) out << keep[i].line << "\n";
        if (!out.good()) {
            out.close();
            std::remove(tmp.c_str());
            return false;
        }
    }
    // Windows' rename will not replace an existing target, and this file is
    // disposable: losing it to a crash between the two calls costs one re-measure.
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    if (prune) prune->kept = static_cast<int>(keep.size());
    return true;
}

// Provenance for the log line: how old the verdict is, and which driver measured
// it. One function rather than an snprintf in the engine, because the wording is
// the thing a reader acts on -- "from the cache, measured 8 s ago" says nothing
// about whether the numbers still describe this machine, and "on driver 71626331"
// is how a cached verdict is traced back without opening the file. "driver
// unknown" is spelled out rather than omitted: a missing clause reads as a
// driver-less measurement instead of as a backend that could not answer (the CPU
// backend reports 0).
inline std::string h2d_probe_origin(bool cached, long long age_sec,
                                    int driver_version) {
    char drv[64];
    if (driver_version > 0)
        std::snprintf(drv, sizeof drv, " on driver %d", driver_version);
    else
        std::snprintf(drv, sizeof drv, " (driver unknown)");
    if (!cached) return std::string("measured now") + drv;
    char age[64];
    if (age_sec < 90)
        std::snprintf(age, sizeof age, "from the cache, measured %lld s ago", age_sec);
    else if (age_sec < 5400)
        std::snprintf(age, sizeof age, "from the cache, measured %lld min ago",
                      age_sec / 60);
    else
        std::snprintf(age, sizeof age, "from the cache, measured %lld h ago",
                      age_sec / 3600);
    return std::string(age) + drv;
}

// ---------------------------------------------------------------------------
// Listing the cache
// ---------------------------------------------------------------------------
// The other half of `kraken-bench --h2d`, which fills it: what a run on THIS
// machine will START FROM. Same facts, opposite direction, and it needs no
// model -- an entry describes a card and a probe size, never weights.
//
// This is deliberately not a lookup. `h2d_probe_cache_read` answers "what do I
// use" for one key and returns nothing else; a listing has to answer "why is my
// run measuring again", and that question is answered by the DEAD lines. So
// every line is reported with the reason a run would skip it, on the same rules
// the reader applies, and the entry a matching key would actually consume is
// marked. One rule escapes the file: another driver's verdict is dead only
// relative to the driver asking, so that reason needs `current` and is not
// invented without it.
struct H2dCacheEntry {
    std::string raw;               // the line as stored, quoted when nothing parses
    std::string name;              // card + gfx, empty when the line did not parse
    size_t vram_total = 0;
    int large_bar = -1;
    int driver_version = 0;
    size_t bytes = 0;
    double pageable_gbs = 0.0;
    double direct_gbs = 0.0;
    double blocking_gbs = 0.0;
    double pinned_gbs = 0.0;
    long long age_sec = 0;
    std::string dead;              // why a run would skip it; empty = live
    bool matches = false;          // this card, ignoring liveness
    bool used = false;             // the line a run with a FULL key would consume
    bool live() const { return dead.empty(); }
};

// A missing file, an unreadable one and a cache that is off all read as "no
// entries": an empty listing is the honest answer to "what will a run start
// from" and must never make the caller fail. `current` is optional because the
// caps it comes from need the device, and a listing that failed BECAUSE the card
// could not be opened would be useless exactly when it is being consulted.
inline std::vector<H2dCacheEntry> h2d_cache_scan(const std::string &path,
                                                 long long now,
                                                 const H2dCacheKey *current = nullptr) {
    std::vector<H2dCacheEntry> out;
    if (path.empty()) return out;
    std::ifstream in(path.c_str());
    if (!in) return out;
    std::string line;
    size_t read = 0;
    bool took_used = false;
    // Bounded like the writer, so a corrupt file cannot make a listing iterate
    // a million lines: this is a cache, and the cap is what bounds it.
    while (read++ < kH2dCacheMaxEntries * 4 && std::getline(in, line)) {
        if (line.empty()) continue;
        H2dCacheEntry e;
        e.raw = line;
        const std::vector<std::string> f = h2d_split_tabs(line);
        long long vram = 0, bar = 0, driver = 0, bytes = 0, stamp = 0, ver = 0;
        bool rates_ok = false;
        // The key fields and the rates are read before the verdict, so an
        // EXPIRED line still says whose verdict it is and what it measured --
        // "an expired 64 MiB direct verdict" is actionable, "an expired entry"
        // is not. The verdict below only decides which reason wins.
        if (f.size() == kH2dCacheFields && h2d_parse_ll(f[0], &ver) &&
            ver == kH2dCacheVersion && h2d_parse_ll(f[2], &vram) &&
            h2d_parse_ll(f[3], &bar) && h2d_parse_ll(f[4], &driver) &&
            h2d_parse_ll(f[5], &bytes) && h2d_parse_ll(f[10], &stamp)) {
            e.name = f[1];
            e.vram_total = static_cast<size_t>(vram < 0 ? 0 : vram);
            e.large_bar = static_cast<int>(
                bar < -1 ? -1 : (bar > 2147483647LL ? 2147483647LL : bar));
            e.driver_version = static_cast<int>(
                driver < 0 ? 0 : (driver > 2147483647LL ? 2147483647LL : driver));
            e.bytes = static_cast<size_t>(bytes < 0 ? 0 : bytes);
            e.age_sec = stamp > now ? 0 : now - stamp;  // a future stamp is 0 s old
            rates_ok = h2d_parse_gbs(f[6], &e.pageable_gbs) &&
                       h2d_parse_gbs(f[7], &e.direct_gbs) &&
                       h2d_parse_gbs(f[8], &e.blocking_gbs) &&
                       h2d_parse_gbs(f[9], &e.pinned_gbs);
        }
        // Same order of refusals as the writer's prune, so the listing and the
        // rewrite cannot disagree about which line is dead and why.
        if (f.size() != kH2dCacheFields) {
            e.dead = "unreadable";
        } else if (!h2d_parse_ll(f[0], &ver) || ver != kH2dCacheVersion) {
            e.dead = "from an older cache format";
        } else if (e.name.empty()) {
            e.dead = "unreadable";
        } else if (stamp > now + 60) {
            e.dead = "stamped in the future";
        } else if (now - stamp > kH2dCacheMaxAgeSec) {
            e.dead = "expired";
        } else if (!rates_ok) {
            e.dead = "rates unreadable";
        } else if (!(e.pageable_gbs > 0.0) || !(e.pinned_gbs > 0.0)) {
            // The reader refuses this line and the writer keeps it, so such an
            // entry can occupy a slot for 30 days while never being used. That
            // is worth naming rather than showing as a live verdict.
            e.dead = "a failed measurement";
        }
        // `bytes == 0` in `current` means "any probe size", and a listing wants
        // exactly that: which size a run asks for depends on a model it never
        // opened, so the honest question is whether this CARD has a verdict to
        // start from. A full key answers the sharper question -- which line the
        // reader returns -- and only then is one line marked as used. The three
        // bucket sizes are the only ones a run can ask about, so a line written
        // at any other size is not a candidate.
        if (current && f.size() == kH2dCacheFields) {
            if (current->bytes != 0) {
                if (h2d_line_matches_key(f, *current)) {
                    e.matches = true;
                    if (e.live() && !took_used) {
                        e.used = true;  // the first live match: what the reader returns
                        took_used = true;
                    }
                }
            } else {
                H2dCacheKey any = *current;
                for (size_t b = kH2dProbeMinBytes; b <= kH2dProbeMaxBytes; b <<= 1) {
                    any.bytes = b;
                    if (h2d_line_matches_key(f, any)) {
                        e.matches = true;
                        break;
                    }
                }
            }
        }
        out.push_back(e);
    }
    return out;
}

// One line, in the order the reader asks: whose verdict, at which probe size,
// how fast, what a run does with it, and whether it is still usable. The path
// and the pin come from the SAME predicates the engine applies -- a listing that
// printed "direct" at a ratio init would not act on would be worse than none.
//
// A verdict from ANOTHER driver is live (the writer keeps it for the rollback
// window) but is not this run's, and showing it exactly like the one that is
// would make the listing useless for the question it exists to answer -- so the
// caller's key, when there is one, is what makes that visible.
inline std::string h2d_cache_entry_text(const H2dCacheEntry &e,
                                        bool other_driver = false) {
    char b[512];
    if (e.name.empty()) {  // nothing parsed: quote the line so it can be found
        std::string raw = e.raw.substr(0, 140);
        for (size_t i = 0; i < raw.size(); i++)
            if (raw[i] == '\t' || raw[i] == '\r' || raw[i] == '\n') raw[i] = ' ';
        std::snprintf(b, sizeof b, "%.140s  |  %s", raw.c_str(),
                      e.dead.empty() ? "unreadable" : e.dead.c_str());
        return std::string(b);
    }
    const int path = pageable_path_choice(e.pageable_gbs, e.direct_gbs,
                                          e.blocking_gbs);
    const double pageable = path == 1   ? e.direct_gbs
                            : path == 2 ? e.blocking_gbs
                                        : e.pageable_gbs;
    const bool pin = warm_pin_worthwhile(pageable, e.pinned_gbs);
    char use[160];
    if (e.live()) {
        // The aperture is stated here too: it is the fact the three rates are
        // evidence FOR, and a listing that showed the numbers without it would
        // leave the reader to redo the comparison.
        std::snprintf(use, sizeof use, "%s-> %s (%.2f GB/s), %s",
                      e.used ? "" : "   ",
                      path == 1   ? "direct"
                      : path == 2 ? "blocking"
                                  : "ring",
                      pageable, pin ? "pin a WARM prefix" : "no pin");
    } else {
        std::snprintf(use, sizeof use, "   [%s]", e.dead.c_str());
    }
    const std::string origin = h2d_probe_origin(true, e.age_sec, e.driver_version);
    std::snprintf(b, sizeof b,
                  "%-44.44s %3zu MiB  ring %5.2f direct %5.2f blocking %5.2f "
                  "pinned %5.2f GB/s  %s%s%s  |  %s",
                  e.name.c_str(), e.bytes >> 20, e.pageable_gbs, e.direct_gbs,
                  e.blocking_gbs, e.pinned_gbs, use,
                  aperture_is_wide(pageable, e.pinned_gbs) ? " [wide aperture]"
                                                          : "",
                  (e.live() && other_driver) ? "  [another driver]" : "",
                  origin.c_str());
    return std::string(b);
}

// Why a lookup missed, in the file's own terms. Called only when
// h2d_probe_cache_read returned nothing, so the FACT of the miss is settled and
// what is wanted is the reason: a run that measures the card says the same thing
// whether the cache was empty, held this card's verdict for another probe size,
// or held this card's verdict and it had expired -- and those are three
// different fixes (run `--h2d`, run a model that buckets there, or nothing at
// all, because the next measurement rewrites it).
inline std::string h2d_cache_miss_text(const std::string &path, long long now,
                                       const H2dCacheKey &key) {
    if (path.empty()) return "the cache is off (KRK_H2D_CACHE=0)";
    H2dCacheKey card = key;
    card.bytes = 0;  // any size: the other sizes are part of the answer
    const std::vector<H2dCacheEntry> entries = h2d_cache_scan(path, now, &card);
    if (entries.empty()) return "there is no cache file yet, or it is empty";
    int exact_dead = 0, other_size = 0, other_card = 0, other_driver = 0, dead = 0;
    std::string exact_reason;
    for (size_t i = 0; i < entries.size(); i++) {
        const H2dCacheEntry &e = entries[i];
        const bool same_card = e.name == key.name && e.vram_total == key.vram_total &&
                               e.large_bar == key.large_bar;
        if (!e.live()) {
            dead++;
            if (same_card && e.bytes == key.bytes &&
                e.driver_version == key.driver_version) {
                exact_dead++;
                exact_reason = e.dead;
            }
            continue;
        }
        if (!same_card) other_card++;
        else if (e.driver_version != key.driver_version) other_driver++;
        else if (e.bytes != key.bytes) other_size++;
    }
    char b[256];
    std::string why;
    if (exact_dead) {
        std::snprintf(b, sizeof b, "this card's %zu MiB entry is not usable (%s)",
                      key.bytes >> 20, exact_reason.c_str());
        why = b;
    } else {
        std::snprintf(b, sizeof b,
                      "the cache holds no %zu MiB verdict for this card",
                      key.bytes >> 20);
        why = b;
    }
    auto add = [&why](const std::string &clause) {
        if (why.empty()) why = clause;
        else why += ", " + clause;
    };
    if (other_size) add(std::to_string(other_size) + " live at other probe sizes");
    if (other_driver) add(std::to_string(other_driver) + " live from another driver");
    if (other_card) add(std::to_string(other_card) + " line(s) for another card");
    if (dead)
        add(std::to_string(dead) + " dead of " + std::to_string(entries.size()) +
            " line(s)");
    return why;
}

// ---------------------------------------------------------------------------
// The tally: is this cache earning its keep?
// ---------------------------------------------------------------------------
// One run's verdict is "reused" or "measured", a hit rate of 100% or 0% that
// says nothing. The question the cache exists to answer spans runs, so the answer
// has to outlive the process: a small file beside the cache counts inits, hits
// and the wall time spent measuring, and the end of a run reports the rate and
// what the cache has saved. Best effort like the cache itself -- a tally that
// cannot be written costs the next report its accuracy, not the run.
//
// Saved = hits x the average measurement this machine has actually paid. With no
// measurement in the tally there is no price, and inventing one would report a
// number nobody can check -- so it says so instead.
//
// It is counted PER PROBE SIZE, because the sizes are not interchangeable: the
// cache key carries the size, a 64 MiB verdict is not the verdict a 16 MiB model
// asks for, and the measurement it replaces costs several times as much. One
// blended rate over all three describes no run in particular -- a machine whose
// only warmed model buckets to 64 MiB read as a mediocre cache instead of as a
// cache that had never been asked the question.
struct H2dTallyBucket {
    size_t bytes = 0;        // the probe size this bucket counts
    long long runs = 0;      // inits that got a verdict (probe on, rates usable)
    long long cached = 0;    // ... from the cache
    long long measured = 0;  // ... by measuring the card
    long long probe_ms = 0;  // total wall time those measurements took
    double hit_rate() const {
        return runs > 0 ? 100.0 * static_cast<double>(cached) / static_cast<double>(runs)
                        : 0.0;
    }
    double avg_probe_ms() const {
        return measured > 0 ? static_cast<double>(probe_ms) / measured : 0.0;
    }
};

// The buckets plus their sum. The flat fields are the TOTAL, so a caller that
// wants the whole machine's rate does not add anything up, and the bucket line
// is what says which size paid for it.
struct H2dProbeTally {
    std::vector<H2dTallyBucket> buckets;  // ascending by size
    long long runs = 0;
    long long cached = 0;
    long long measured = 0;
    long long probe_ms = 0;
    double hit_rate() const {
        return runs > 0 ? 100.0 * static_cast<double>(cached) / static_cast<double>(runs)
                        : 0.0;
    }
    double avg_probe_ms() const {
        return measured > 0 ? static_cast<double>(probe_ms) / measured : 0.0;
    }
    const H2dTallyBucket *find(size_t bytes) const {
        for (size_t i = 0; i < buckets.size(); i++)
            if (buckets[i].bytes == bytes) return &buckets[i];
        return nullptr;
    }
    void sum_buckets() {
        runs = cached = measured = probe_ms = 0;
        for (size_t i = 0; i < buckets.size(); i++) {
            runs += buckets[i].runs;
            cached += buckets[i].cached;
            measured += buckets[i].measured;
            probe_ms += buckets[i].probe_ms;
        }
    }
};

// Beside the cache, so moving or disabling the cache moves its ledger with it.
inline std::string h2d_tally_path(const std::string &cache_path) {
    return cache_path.empty() ? std::string() : cache_path + ".stats";
}

// A corrupt tally reads as no tally rather than as a partial one: the numbers are
// cheap to lose and worth less than a hit rate that cannot be true. v1 (one
// blended line) is not a tally any more for the same reason the cache refuses a
// v1 verdict -- it cannot answer the question the buckets exist to answer.
inline H2dProbeTally h2d_tally_load(const std::string &path) {
    H2dProbeTally t;
    if (path.empty()) return t;
    std::ifstream in(path.c_str());
    if (!in) return t;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> f = h2d_split_tabs(line);
        long long ver = 0, bytes = 0, runs = 0, cached = 0, measured = 0, ms = 0;
        if (f.size() != 6 || !h2d_parse_ll(f[0], &ver) || ver != 2) return H2dProbeTally();
        if (!h2d_parse_ll(f[1], &bytes) || !h2d_parse_ll(f[2], &runs) ||
            !h2d_parse_ll(f[3], &cached) || !h2d_parse_ll(f[4], &measured) ||
            !h2d_parse_ll(f[5], &ms))
            return H2dProbeTally();
        // cached + measured <= runs: an init whose probe failed got neither.
        if (bytes <= 0 || runs < 0 || cached < 0 || measured < 0 || ms < 0 ||
            cached + measured > runs)
            return H2dProbeTally();
        H2dTallyBucket *b = nullptr;
        for (size_t i = 0; i < t.buckets.size(); i++)
            if (t.buckets[i].bytes == static_cast<size_t>(bytes)) b = &t.buckets[i];
        if (!b) {
            H2dTallyBucket fresh;
            fresh.bytes = static_cast<size_t>(bytes);
            t.buckets.push_back(fresh);
            b = &t.buckets.back();
        }
        b->runs += runs;
        b->cached += cached;
        b->measured += measured;
        b->probe_ms += ms;
    }
    std::sort(t.buckets.begin(), t.buckets.end(),
              [](const H2dTallyBucket &a, const H2dTallyBucket &b) {
                  return a.bytes < b.bytes;
              });
    t.sum_buckets();
    return t;
}

// One line per bucket, newest totals last: the format a reader can diff and a
// human can read, and it keeps the ROWS it counts next to the numbers.
inline bool h2d_tally_store(const std::string &path, const H2dProbeTally &t) {
    if (path.empty()) return false;
    h2d_make_parent_dir(path);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp.c_str(), std::ios::binary | std::ios::trunc);
        if (!out) return false;
        for (size_t i = 0; i < t.buckets.size(); i++) {
            const H2dTallyBucket &b = t.buckets[i];
            out << "2\t" << b.bytes << "\t" << b.runs << "\t" << b.cached << "\t"
                << b.measured << "\t" << b.probe_ms << "\n";
        }
        if (!out.good()) {
            out.close();
            std::remove(tmp.c_str());
            return false;
        }
    }
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

// Records one run's verdict at the probe size it asked about, and hands back the
// tally it contributed to. `wrote` is false when the file could not be updated,
// so a caller can tell a fresh tally from one it merely read.
inline H2dProbeTally h2d_tally_note(const std::string &cache_path, size_t bytes,
                                   bool cached, long long probe_ms,
                                   bool *wrote = nullptr) {
    const std::string path = h2d_tally_path(cache_path);
    H2dProbeTally t = h2d_tally_load(path);
    if (path.empty()) {
        if (wrote) *wrote = false;
        return t;
    }
    H2dTallyBucket *b = nullptr;
    for (size_t i = 0; i < t.buckets.size(); i++)
        if (t.buckets[i].bytes == bytes) b = &t.buckets[i];
    if (!b) {
        H2dTallyBucket fresh;
        fresh.bytes = bytes;
        t.buckets.push_back(fresh);
        b = &t.buckets.back();
    }
    b->runs++;
    if (cached) {
        b->cached++;
    } else {
        b->measured++;
        if (probe_ms > 0) b->probe_ms += probe_ms;
    }
    std::sort(t.buckets.begin(), t.buckets.end(),
              [](const H2dTallyBucket &x, const H2dTallyBucket &y) {
                  return x.bytes < y.bytes;
              });
    t.sum_buckets();
    // Stored even when the caller does not ask whether it worked: `wrote` is a
    // report, not a permission. Gating the store on it made every engine (which
    // passes no pointer) count in memory and write nothing at all.
    const bool ok = h2d_tally_store(path, t);
    if (wrote) *wrote = ok;
    return t;
}

// "" when nothing has been counted, so a run with no promotion path says
// nothing rather than printing zeros on every dense model.
inline std::string h2d_tally_text(const H2dProbeTally &t) {
    if (t.runs <= 0) return std::string();
    char b[512];
    if (t.measured > 0) {
        const double avg = t.avg_probe_ms();
        std::snprintf(b, sizeof b,
                      "%lld of %lld probed run(s) reused a cached verdict (%.0f%%), "
                      "%lld measured the card at %.1f ms on average, so the cache "
                      "has avoided about %.0f ms of probing",
                      t.cached, t.runs, t.hit_rate(), t.measured, avg,
                      static_cast<double>(t.cached) * avg);
    } else {
        std::snprintf(b, sizeof b,
                      "%lld of %lld probed run(s) reused a cached verdict (%.0f%%); "
                      "no measurement is recorded in this tally, so the time it "
                      "saved is not priced",
                      t.cached, t.runs, t.hit_rate());
    }
    return b;
}

// One bucket, in its own words. The size leads, because that is the fact a
// reader is comparing: with the total beside it, "16 MiB: 2 of 3" says whether
// the cache is being asked about the size the run uses at all.
inline std::string h2d_tally_bucket_text(const H2dTallyBucket &b) {
    if (b.runs <= 0) return std::string();
    char buf[384];
    std::string s;
    std::snprintf(buf, sizeof buf, "%3zu MiB: %lld of %lld reused (%.0f%%)",
                  b.bytes >> 20, b.cached, b.runs, b.hit_rate());
    s = buf;
    if (b.measured > 0) {
        std::snprintf(buf, sizeof buf, ", %lld measured at %.1f ms", b.measured,
                      b.avg_probe_ms());
        s += buf;
    }
    if (b.cached > 0) {
        if (b.measured > 0) {
            std::snprintf(buf, sizeof buf, " -> about %.0f ms avoided",
                          static_cast<double>(b.cached) * b.avg_probe_ms());
        } else {
            std::snprintf(buf, sizeof buf,
                          ", no measurement at this size, so not priced");
        }
        s += buf;
    }
    return s;
}

// The init line: reused or measured, and -- when measured -- why nothing was
// there to reuse. Wording lives here so the suite can pin it, the same reason
// h2d_prune_text and h2d_probe_origin do: a log line nobody tests drifts back to
// a bare "measured now", which is exactly the state this line exists to end.

// `iterations` is what the probe actually ran, not its cap: the count is how much
// measurement the reader is trusting, and printing the cap would claim five runs
// on a size that settled at three.
inline std::string h2d_verdict_text(bool cached, long long age_sec,
                                    int driver_version, size_t bytes,
                                    const std::string &miss, int iterations = 0) {
    char b[640];
    if (cached) {
        std::snprintf(b, sizeof b, "H2D verdict: reused the cached %zu MiB one (%s)",
                      bytes >> 20,
                      h2d_probe_origin(true, age_sec, driver_version).c_str());
    } else if (iterations > 0) {
        std::snprintf(b, sizeof b,
                      "H2D verdict: measured the card now (%zu MiB, 4 arms x %d "
                      "iteration(s)) -- %s",
                      bytes >> 20, iterations,
                      miss.empty() ? "nothing cached for this card" : miss.c_str());
    } else {
        std::snprintf(b, sizeof b,
                      "H2D verdict: measured the card now (%zu MiB, 4 arms) -- %s",
                      bytes >> 20,
                      miss.empty() ? "nothing cached for this card" : miss.c_str());
    }
    return std::string(b);
}

}  // namespace krk
