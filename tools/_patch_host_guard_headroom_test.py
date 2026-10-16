#!/usr/bin/env python3
# 1. engine.cpp uses krk::vram_guard.hpp instead of its own local reserve.
# 2. the WARM staging loop re-checks a host-RAM reserve every round.
# 3. every run prints its VRAM headroom before it starts.
# 4. test_expert_cache_byte_budget drives the guard decision.
import io, sys

def load(p):
    raw = io.open(p, 'r', encoding='utf-8', newline='').read()
    return raw.replace('\r\n', '\n'), raw.count('\r\n') > 0

def save(p, t, crlf):
    io.open(p, 'w', encoding='utf-8', newline='').write(
        t.replace('\n', '\r\n') if crlf else t)

def sub(t, old, new, count=1, tag=''):
    if t.count(old) != count:
        print('ANCHOR FAIL %s: %d != %d' % (tag, t.count(old), count))
        print(old[:300]); sys.exit(1)
    return t.replace(old, new)

E, crlf_e = load('src/engine.cpp')

# ---- 1. the header, and no local copy of the rule -------------------------
E = sub(E, '#include "krk/host_time.hpp"',
        '#include "krk/host_time.hpp"\n#include "krk/vram_guard.hpp"',
        tag='include')

E = sub(E, """// The VRAM this engine will not spend. The expert tier's budget is decided at
// load time from the memory that exists then, but the workspaces, the KV tier
// and the staging buffers are allocated after it -- and the warm-up then fills
// the tier to that budget. Measured 2026-10-11 on Qwen3.8-Distill-35B-A3B Q2KXL
// (256 experts, --expert-warmup 1): the card ended at 15.92 GiB in use of
// 15.92 GiB, 0.00 free, and took the machine's display down twice. The KV tier
// next door reserves max(256 MiB, free/8) for the same reason; the expert side
// had a 512 MiB constant used to cap the *plan*, and nothing that stopped a
// promotion. This is the number the warm-up enforces, at promotion time, by
// asking the driver instead of trusting a measurement taken before the
// allocations that caused the gap. KRK_VRAM_RESERVE_MB overrides it.
size_t vram_reserve_bytes(size_t total_vram) {
    static const i64 env = [] {
        const char *v = std::getenv("KRK_VRAM_RESERVE_MB");
        return (v && *v) ? static_cast<i64>(std::atoll(v)) : static_cast<i64>(-1);
    }();
    if (env >= 0) return static_cast<size_t>(env) * 1024u * 1024u;
    const size_t gib = static_cast<size_t>(1) << 30;
    const size_t share = total_vram / 16u;
    return share > gib ? share : gib;
}

""", '', tag='drop local reserve')

# every use now goes through the header
E = sub(E, 'const size_t vram_reserve = vram_reserve_bytes(total_vram);',
        'const size_t vram_reserve = krk::vram_reserve_bytes(total_vram);',
        tag='policy reserve')
E = sub(E, 'const size_t vram_reserve = vram_reserve_bytes(be_->device_total_bytes());',
        'const size_t vram_reserve = krk::vram_reserve_bytes(be_->device_total_bytes());',
        tag='warmup reserve')

# ---- 2. the refusal goes through the tested decision ----------------------
E = sub(E, """        size_t largest = 0;
        if (!krk::expert_budget_fits(budget, total_vram, used_now, vram_reserve,
                                     &largest, err)) {
            return false;
        }""",
"""        size_t largest = 0;
        if (!krk::expert_budget_fits(budget, total_vram, used_now, vram_reserve,
                                     &largest, err)) {
            return false;
        }""", tag='noop-check') if False else E

old_refuse = """        if (total_vram > 0 && budget > room_now) {
            if (err) {
                char b[512];
                std::snprintf(
                    b, sizeof(b),
                    "expert budget %d MiB does not fit: %.0f MiB of VRAM, "
                    "%.0f MiB already in use, %.0f MiB reserved -> the largest "
                    "budget this card can hold is %.0f MiB (--expert-cache-mb %.0f)",
                    cfg_.expert_cache_mb,
                    static_cast<f64>(total_vram) / 1048576.0,
                    static_cast<f64>(used_now) / 1048576.0,
                    static_cast<f64>(vram_reserve) / 1048576.0,
                    static_cast<f64>(room_now) / 1048576.0,
                    static_cast<f64>(room_now) / 1048576.0);
                *err = b;
            }
            return false;
        }"""
new_refuse = """        // The decision itself lives in krk/vram_guard.hpp so the test suite can
        // drive it: this refusal is the one path where being wrong either
        // changes the user's experiment quietly (a trim) or leaves a card with
        // nothing free (no refusal).
        if (!krk::expert_budget_fits(budget, total_vram, used_now, vram_reserve,
                                     &room_now, err))
            return false;"""
E = sub(E, old_refuse, new_refuse, tag='refusal via header')

# ---- 3. the host-RAM guard on the staging loop ----------------------------
E = sub(E, """    size_t staged = 0;
    size_t from = 0;
    i32 rounds = 0;
    while (from < depth) {
        if (out_of_time()) break;
        const size_t used = ec.warm_used_bytes();""",
"""    size_t staged = 0;
    size_t from = 0;
    i32 rounds = 0;
    // The host twin of the VRAM reserve, and for the same reason: the ram policy
    // above plans a 20% headroom once, before the KV tier, the workspaces and
    // the model's own buffers exist, and staging is what spends it. Ask the
    // machine every round instead of trusting that plan.
    const size_t host_reserve = krk::host_reserve_bytes(host_total_bytes());
    bool hit_host_reserve = false;
    while (from < depth) {
        if (out_of_time()) break;
        if (host_available_bytes() <= host_reserve) {
            hit_host_reserve = true;
            break;
        }
        const size_t used = ec.warm_used_bytes();""", tag='host guard')

E = sub(E, """    else if (hit_reserve) {
        char b[112];
        std::snprintf(b, sizeof(b),
                      " (stopped at the VRAM reserve: %.0f MiB kept free of "
                      "%.0f MiB total)",
                      static_cast<f64>(vram_reserve) / 1048576.0,
                      static_cast<f64>(be_->device_total_bytes()) / 1073741824.0);
        why = b;
    }""",
"""    else if (hit_reserve) {
        char b[112];
        std::snprintf(b, sizeof(b),
                      " (stopped at the VRAM reserve: %.0f MiB kept free of "
                      "%.0f MiB total)",
                      static_cast<f64>(vram_reserve) / 1048576.0,
                      static_cast<f64>(be_->device_total_bytes()) / 1073741824.0);
        why = b;
    } else if (hit_host_reserve) {
        char b[112];
        std::snprintf(b, sizeof(b),
                      " (stopped at the host RAM reserve: %.0f MiB kept free)",
                      static_cast<f64>(host_reserve) / 1048576.0);
        why = b;
    }""", tag='host report')

# ---- 4. the headroom line, before anything is sized ----------------------
E = sub(E, "    if (!configure_expert_cache(err)) return false;",
"""    // Every run states its headroom before it starts, dense models included.
    // The fault this guards against reached 0.00 GiB free, and the only line
    // that showed it printed at the END of the run -- after the display was
    // already gone. Reading it in the first lines is what makes a card-filling
    // configuration visible while it can still be changed.
    {
        const size_t vtotal = be_->device_total_bytes();
        const size_t vfree = be_->device_free_bytes();
        if (vtotal > 0) {
            const size_t vrsv = krk::vram_reserve_bytes(vtotal);
            KRK_INFO("vram headroom: %.2f GiB free of %.2f GiB total, %.2f GiB "
                     "reserved (KRK_VRAM_RESERVE_MB overrides)",
                     static_cast<f64>(vfree) / 1073741824.0,
                     static_cast<f64>(vtotal) / 1073741824.0,
                     static_cast<f64>(vrsv) / 1073741824.0);
            if (vfree < vrsv)
                KRK_WARN("vram headroom: free VRAM is BELOW the %.2f GiB reserve, "
                         "so an expert tier sized against the plan will be cut "
                         "short -- lower --ctx / --kv-hot-mb, or lower "
                         "KRK_VRAM_RESERVE_MB if this card is dedicated",
                         static_cast<f64>(vrsv) / 1073741824.0);
        }
    }
    if (!configure_expert_cache(err)) return false;""", tag='headroom')
save('src/engine.cpp', E, crlf_e)

# ---- 5. the tests -------------------------------------------------------
T, crlf_t = load('tests/test_kraken.cpp')
lines = T.split('\n')
first_krk = next(i for i, l in enumerate(lines) if l.startswith('#include "krk/'))
lines.insert(first_krk, '#include "krk/vram_guard.hpp"')
T = '\n'.join(lines)

T = sub(T, """    const size_t corpus = 4 * (a.expert_bytes() + b.expert_bytes());
    CHECK(corpus == 256, "the routed set is eight slots of 24 and 40 bytes");""",
"""    const size_t corpus = 4 * (a.expert_bytes() + b.expert_bytes());
    CHECK(corpus == 256, "the routed set is eight slots of 24 and 40 bytes");

    // The budget guard. It exists because `--expert-warmup` once filled a
    // 15.92 GiB card to 0.00 GiB free and took the display with it, and its
    // decision is pure arithmetic, so it is driven here with no device: an
    // explicit budget that cannot fit is refused, and the refusal names the
    // largest that does.
    {
#ifdef _WIN32
        _putenv_s("KRK_VRAM_RESERVE_MB", "");
        _putenv_s("KRK_RAM_RESERVE_MB", "");
#else
        unsetenv("KRK_VRAM_RESERVE_MB");
        unsetenv("KRK_RAM_RESERVE_MB");
#endif
        size_t largest = 0;
        std::string why;
        // 1000 total, 400 in use, 100 reserved -> 500 is the most that fits.
        CHECK(krk::expert_budget_room(1000, 400, 100) == 500,
              "room is total minus what is used minus the reserve");
        CHECK(krk::expert_budget_fits(500, 1000, 400, 100, &largest, &why),
              "a budget exactly at the room fits");
        CHECK(largest == 500, "and reports that room as the largest");
        CHECK(!krk::expert_budget_fits(501, 1000, 400, 100, &largest, &why),
              "one byte more is refused");
        CHECK(why.find("500") != std::string::npos,
              "and the refusal names the largest budget that fits");
        CHECK(!krk::expert_budget_fits(1, 1000, 900, 100, &largest, &why),
              "a card with only the reserve left refuses even one byte");
        CHECK(largest == 0, "and names zero as the largest");
        CHECK(krk::expert_budget_fits(1024, 0, 0, 0, &largest, &why),
              "a backend that cannot report capacity does not refuse");
        const size_t gib = static_cast<size_t>(1) << 30;
        CHECK(krk::vram_reserve_bytes(16 * gib) == gib,
              "a 16 GiB card reserves 1 GiB");
        CHECK(krk::vram_reserve_bytes(8 * gib) == gib,
              "a smaller card keeps the 1 GiB floor");
        CHECK(krk::vram_reserve_bytes(64 * gib) == 4 * gib,
              "a large card reserves a sixteenth");
        CHECK(krk::host_reserve_bytes(96 * gib) == 12 * gib,
              "96 GiB of host RAM reserves 12 GiB");
        CHECK(krk::host_reserve_bytes(4 * gib) == 2 * gib,
              "and the host reserve never drops below 2 GiB");
    }""", tag='test checks')
save('tests/test_kraken.cpp', T, crlf_t)

print('patched: src/engine.cpp, tests/test_kraken.cpp (+ include/krk/vram_guard.hpp)')
