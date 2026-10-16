#!/usr/bin/env python3
# Two guards for the fault measured on 2026-10-11 (card at 0.00 GiB free):
#  - the expert budget is refused (explicit) or clamped (auto) to what fits with
#    a reserve held back at PLAN time;
#  - the warm-up stops promoting against the reserve at PROMOTION time, asking
#    the driver, because the plan-time arithmetic cannot see the allocations
#    that come after it.
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
        print(old[:400]); sys.exit(1)
    return t.replace(old, new)

E, crlf_e = load('src/engine.cpp')
H, crlf_h = load('include/krk/engine.hpp')

# ---------------------------------------------------------------------------
# 1. the reserve itself, in the voice of the code around it
# ---------------------------------------------------------------------------
old = """// KRK_TOPK_SYNC restores the explicit hipDeviceSynchronize topk_argmax used to"""
new = """// The VRAM this engine will not spend. The expert tier's budget is decided at
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

// KRK_TOPK_SYNC restores the explicit hipDeviceSynchronize topk_argmax used to"""
E = sub(E, old, new, tag='reserve helper')

# ---------------------------------------------------------------------------
# 2. the warm-up stops on the reserve and says which reserve it kept
# ---------------------------------------------------------------------------
old = """    size_t promoted = 0;
    size_t promoted_bytes = 0;
    if (be_->caps().vram_free > 0) {
        for (size_t r = 0; r < depth; r++) {
            if (out_of_time()) break;
            bool room = false;"""
new = """    size_t promoted = 0;
    size_t promoted_bytes = 0;
    // Stop promoting while this much VRAM is still free. Checked per round,
    // against the driver's own count: the tier's budget came from a plan made
    // before the workspaces and the KV tier existed, so only the driver can
    // say what is actually left. A partly warm tier is a slow run; a card with
    // nothing free is a machine that has to be recovered by hand.
    const size_t vram_reserve = vram_reserve_bytes(be_->device_total_bytes());
    bool hit_reserve = false;
    if (be_->caps().vram_free > 0) {
        for (size_t r = 0; r < depth; r++) {
            if (out_of_time()) break;
            if (be_->device_free_bytes() <= vram_reserve) {
                hit_reserve = true;
                break;
            }
            bool room = false;"""
E = sub(E, old, new, tag='warmup guard')

old = """    const f64 ms = elapsed_ms();
    const bool cut = out_of_time();
    KRK_INFO("expert warmup: %s order, %d rounds, %zu experts/layer deep; WARM "
             "%.0f MiB of %.0f MiB in %zu slots (%.0f MiB read), VRAM +%zu "
             "experts (%.0f MiB, %zu slots); %.0f ms%s",
             ranked ? "ranked" : "even", rounds, depth,
             static_cast<f64>(ec.warm_used_bytes()) / 1048576.0,
             static_cast<f64>(warm_cap) / 1048576.0, ec.warm_slots(),
             static_cast<f64>(staged) / 1048576.0, promoted,
             static_cast<f64>(promoted_bytes) / 1048576.0, ec.resident_slots(),
             ms, cut ? " (stopped at the --expert-warmup-ms ceiling)" : "");"""
new = """    const f64 ms = elapsed_ms();
    const bool cut = out_of_time();
    std::string why;
    if (cut) why = " (stopped at the --expert-warmup-ms ceiling)";
    else if (hit_reserve) {
        char b[112];
        std::snprintf(b, sizeof(b),
                      " (stopped at the VRAM reserve: %.0f MiB kept free of "
                      "%.0f MiB total)",
                      static_cast<f64>(vram_reserve) / 1048576.0,
                      static_cast<f64>(be_->device_total_bytes()) / 1073741824.0);
        why = b;
    }
    KRK_INFO("expert warmup: %s order, %d rounds, %zu experts/layer deep; WARM "
             "%.0f MiB of %.0f MiB in %zu slots (%.0f MiB read), VRAM +%zu "
             "experts (%.0f MiB, %zu slots); %.0f ms%s",
             ranked ? "ranked" : "even", rounds, depth,
             static_cast<f64>(ec.warm_used_bytes()) / 1048576.0,
             static_cast<f64>(warm_cap) / 1048576.0, ec.warm_slots(),
             static_cast<f64>(staged) / 1048576.0, promoted,
             static_cast<f64>(promoted_bytes) / 1048576.0, ec.resident_slots(),
             ms, why.c_str());"""
E = sub(E, old, new, tag='warmup report')

# ---------------------------------------------------------------------------
# 3. the preflight: refuse an explicit budget that cannot fit, clamp the policy
# ---------------------------------------------------------------------------
old = """void Engine::configure_expert_cache() {
    const ModelConfig &mc = model_.cfg();
    if (!mc.is_moe || mc.n_expert <= 0) return;"""
new = """bool Engine::configure_expert_cache(std::string *err) {
    const ModelConfig &mc = model_.cfg();
    if (!mc.is_moe || mc.n_expert <= 0) return true;"""
E = sub(E, old, new, tag='signature')

old = """    size_t budget = 0;
    if (cfg_.expert_cache_mb > 0) {
        // An explicit restriction wins over the policy: the user has said how
        // much they want to spend, in expert bytes alone.
        budget = static_cast<size_t>(cfg_.expert_cache_mb) * 1024u * 1024u;
    } else if (be_->caps().vram_free == 0) {"""
new = """    // What the card can actually be asked for, with the reserve held back.
    const size_t vram_reserve = vram_reserve_bytes(total_vram);
    const size_t room_now = free_now > vram_reserve ? free_now - vram_reserve : 0;

    size_t budget = 0;
    if (cfg_.expert_cache_mb > 0) {
        // An explicit restriction wins over the policy: the user has said how
        // much they want to spend, in expert bytes alone.
        budget = static_cast<size_t>(cfg_.expert_cache_mb) * 1024u * 1024u;
        // ... but not over the card. A budget the machine cannot hold is
        // refused rather than trimmed: silently keeping less than the user
        // asked for turns their experiment into a different one, and the run
        // would report a residency they did not choose. Name the largest
        // budget that does fit, so the fix is one flag away.
        if (total_vram > 0 && budget > room_now) {
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
        }
    } else if (be_->caps().vram_free == 0) {"""
E = sub(E, old, new, tag='explicit refusal')

old = """    KRK_INFO("vram policy: %.1f GiB cap%s (%.1f GiB already in use, %.1f GiB "
             "demand, %.1f GiB free of %.1f GiB) -> expert budget %.1f MiB","""
new = """    // The plan cannot see the workspaces and the KV tier, which are allocated
    // after it, so it is clamped to the free memory measured HERE minus the
    // reserve, and the warm-up enforces the same reserve again at promotion
    // time. Both numbers are reported: they are the plan and the floor.
    if (total_vram > 0 && budget > room_now) budget = room_now;
    KRK_INFO("vram policy: %.1f GiB cap%s (%.1f GiB already in use, %.1f GiB "
             "demand, %.1f GiB free of %.1f GiB, %.1f GiB reserved) -> expert "
             "budget %.1f MiB","""
E = sub(E, old, new, tag='auto clamp')

old = """             static_cast<f64>(free_now) / static_cast<f64>(gib),
             static_cast<f64>(total_vram) / static_cast<f64>(gib),
             static_cast<f64>(budget) / (1024.0 * 1024.0));"""
new = """             static_cast<f64>(free_now) / static_cast<f64>(gib),
             static_cast<f64>(total_vram) / static_cast<f64>(gib),
             static_cast<f64>(vram_reserve) / static_cast<f64>(gib),
             static_cast<f64>(budget) / (1024.0 * 1024.0));"""
E = sub(E, old, new, tag='auto clamp args')

old = """    configure_expert_cache();"""
new = """    if (!configure_expert_cache(err)) return false;"""
E = sub(E, old, new, tag='call site')

save('src/engine.cpp', E, crlf_e)

old = """    void configure_expert_cache();"""
new = """    // Chooses the device-side expert budget. Returns false with *err set when
    // an explicit --expert-cache-mb cannot fit the card, which is a refusal
    // rather than a trim: see the reserve rule in the definition.
    bool configure_expert_cache(std::string *err);"""
H = sub(H, old, new, tag='declaration')
save('include/krk/engine.hpp', H, crlf_h)

print('patched: engine.cpp (vram_reserve_bytes, warm-up guard + report, budget refusal/clamp), engine.hpp (declaration)')
