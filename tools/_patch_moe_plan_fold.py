#!/usr/bin/env python3
# The MoE routing plan becomes ONE upload pair per layer (KRK_MOE_PLAN=0 keeps
# the per-visit pair), and the three remaining decode-path drains get switches.
import io, sys

def load(path):
    raw = io.open(path, 'r', encoding='utf-8', newline='').read()
    crlf = raw.count('\r\n') > 0
    txt = raw.replace('\r\n', '\n')
    return txt, crlf

def save(path, txt, crlf):
    out = txt.replace('\n', '\r\n') if crlf else txt
    io.open(path, 'w', encoding='utf-8', newline='').write(out)

def sub(txt, old, new, count=1, tag=''):
    n = txt.count(old)
    if n != count:
        print('ANCHOR FAIL %s: found %d, expected %d' % (tag, n, count))
        print('---- looking for ----')
        print(old[:400])
        sys.exit(1)
    return txt.replace(old, new)

E, crlf_e = load('src/engine.cpp')
H, crlf_h = load('include/krk/engine.hpp')

# --------------------------------------------------------------------------
# 1. switch accessors, beside the two that already exist
# --------------------------------------------------------------------------
old = """bool sync_expert() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_SYNC_EXPERT");
        return v && *v && std::string(v) != "0";
    }();
    return on;
}
"""
new = """bool sync_expert() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_SYNC_EXPERT");
        return v && *v && std::string(v) != "0";
    }();
    return on;
}

// KRK_TOPK_SYNC restores the explicit hipDeviceSynchronize topk_argmax used to
// issue between its launch and its 16-byte readback. `download` is a BLOCKING
// hipMemcpy on the same stream, so it already orders itself behind that launch
// -- the same argument the router readback above makes -- and the drain is a
// second, redundant wait that a decode token pays once. Off by default; the
// switch is the A/B arm the cost is measured against.
bool topk_sync() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_TOPK_SYNC");
        return v && *v && std::string(v) != "0";
    }();
    return on;
}

// KRK_LOGITS_SYNC=0 removes the drain in front of a full logits row readback.
// It sits on the fallback path (a row is only downloaded when the device-side
// argmax is unavailable), so it is off the plain decode path on a GPU -- but it
// is the same pattern as topk_argmax's, and a switch is what makes that a
// measurement rather than a claim. Default is the drain present, as shipped.
bool logits_sync() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_LOGITS_SYNC");
        return !(v && *v && std::string(v) == "0");
    }();
    return on;
}

// KRK_SPEC_ROW_SYNC=0 removes the per-round drain in front of the speculative
// loop's single-row logits recompute. Reachable only with a drafter, where the
// row is recomputed once per speculation round; default is the drain present.
bool spec_row_sync() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_SPEC_ROW_SYNC");
        return !(v && *v && std::string(v) == "0");
    }();
    return on;
}

// KRK_MOE_PLAN=0 restores one plan upload PAIR per expert visit. The default
// builds the layer's whole routing plan once and uploads it with two copies, so
// a visit reads its segment of the device buffer instead of issuing its own
// pair -- on a 256-expert layer at top-8 that is 16 launches per layer of the
// decode step down to 2.
bool moe_plan_grouped() {
    static const bool on = [] {
        const char *v = std::getenv("KRK_MOE_PLAN");
        return !(v && *v && std::string(v) == "0");
    }();
    return on;
}
"""
E = sub(E, old, new, tag='accessors')

# --------------------------------------------------------------------------
# 2. the plan/alpha buffers hold the whole LAYER's pairs, not one visit's
# --------------------------------------------------------------------------
old = """        ws_plan_ = alloc(static_cast<size_t>(C) * sizeof(i32));
        ws_alpha_ = alloc(static_cast<size_t>(C) * sizeof(f32));"""
new = """        // The whole layer's plan lives in one buffer: at most one (token,
        // expert) pair per selection slot, so C * k entries is the bound, and a
        // visit reads [off(e), off(e+1)) of it. It was C entries when each visit
        // uploaded its own segment.
        ws_plan_ = alloc(static_cast<size_t>(C) * k * sizeof(i32));
        ws_alpha_ = alloc(static_cast<size_t>(C) * k * sizeof(f32));"""
E = sub(E, old, new, tag='plan alloc')

# --------------------------------------------------------------------------
# 3. build the layer plan once, before the expert loop
# --------------------------------------------------------------------------
old = """        for (i32 e = 0; !expert_stub_ && e < ne; e++) {
            // The host arm owns this expert on this layer. Skipping it here is
            // what leaves its weights absent from both tiers.
            if (hybrid && hy_take_[static_cast<size_t>(e)]) continue;"""
new = """        // ---- one routing plan for the whole layer ---------------------------
        // The (token, slot) pairs are bucketed by expert once per layer, so a
        // visit reads [off(e), off(e+1)) of a plan that cost TWO copies instead
        // of two per expert. The row order is what the per-expert rescan below
        // collected -- tokens ascending, then slots ascending -- and the
        // bucketing is the same O(n*k) walk with a prefix sum, not the
        // O(n_expert * n * k) rescan it replaces. `KRK_MOE_PLAN=0` takes the
        // old path and is the arm the fold is measured against.
        const bool plan_grouped = moe_plan_grouped() && !expert_stub_;
        if (plan_grouped) {
            plan_off_.assign(static_cast<size_t>(ne) + 1, 0);
            for (i32 t = 0; t < n; t++) {
                const i32 *sel = moe_sel_.data() + static_cast<size_t>(t) * k;
                for (i32 s = 0; s < k; s++) {
                    const i32 e = sel[s];
                    if (e >= 0 && e < ne) plan_off_[static_cast<size_t>(e) + 1]++;
                }
            }
            for (i32 e = 0; e < ne; e++)
                plan_off_[static_cast<size_t>(e) + 1] +=
                    plan_off_[static_cast<size_t>(e)];
            plan_dev_.resize(static_cast<size_t>(plan_off_[static_cast<size_t>(ne)]));
            alpha_dev_.resize(plan_dev_.size());
            plan_cur_.assign(plan_off_.begin(), plan_off_.end() - 1);
            for (i32 t = 0; t < n; t++) {
                const i32 *sel = moe_sel_.data() + static_cast<size_t>(t) * k;
                const f32 *wt = moe_wt_.data() + static_cast<size_t>(t) * k;
                for (i32 s = 0; s < k; s++) {
                    const i32 e = sel[s];
                    if (e < 0 || e >= ne) continue;
                    const i32 at = plan_cur_[static_cast<size_t>(e)]++;
                    plan_dev_[static_cast<size_t>(at)] = t;
                    alpha_dev_[static_cast<size_t>(at)] = wt[s];
                }
            }
            if (!plan_dev_.empty()) {
                be_->upload_i32(static_cast<i32 *>(ws_plan_), plan_dev_.data(),
                                static_cast<i64>(plan_dev_.size()));
                be_->upload(static_cast<f32 *>(ws_alpha_), alpha_dev_.data(),
                            plan_dev_.size() * sizeof(f32));
            }
        }

        for (i32 e = 0; !expert_stub_ && e < ne; e++) {
            // The host arm owns this expert on this layer. Skipping it here is
            // what leaves its weights absent from both tiers.
            if (hybrid && hy_take_[static_cast<size_t>(e)]) continue;"""
E = sub(E, old, new, tag='layer plan')

# --------------------------------------------------------------------------
# 4. the visit reads its segment (or stages its own pair, under the switch)
# --------------------------------------------------------------------------
old = """            const i64 m = static_cast<i64>(group_rows_.size());
            plan_dev_.assign(group_rows_.begin(), group_rows_.end());
            alpha_dev_.assign(group_wt_.begin(), group_wt_.end());
            be_->upload_i32(static_cast<i32 *>(ws_plan_), plan_dev_.data(), m);
            be_->upload(static_cast<f32 *>(ws_alpha_), alpha_dev_.data(),
                        static_cast<size_t>(m) * sizeof(f32));

            // x̃ = xn[group_rows] — the permuted activation block.
            be_->gather_rows(ws_xg_, ws_xn_, static_cast<const i32 *>(ws_plan_), m,
                             n_embd_);"""
new = """            const i64 m = static_cast<i64>(group_rows_.size());
            const i32 *plan_i = nullptr;
            const f32 *plan_f = nullptr;
            if (plan_grouped) {
                // This expert's segment of the layer plan, already on the
                // device: no copy is issued here at all.
                plan_i = static_cast<const i32 *>(ws_plan_) + plan_off_[e];
                plan_f = static_cast<const f32 *>(ws_alpha_) + plan_off_[e];
            } else {
                plan_dev_.assign(group_rows_.begin(), group_rows_.end());
                alpha_dev_.assign(group_wt_.begin(), group_wt_.end());
                be_->upload_i32(static_cast<i32 *>(ws_plan_), plan_dev_.data(), m);
                be_->upload(static_cast<f32 *>(ws_alpha_), alpha_dev_.data(),
                            static_cast<size_t>(m) * sizeof(f32));
                plan_i = static_cast<const i32 *>(ws_plan_);
                plan_f = static_cast<const f32 *>(ws_alpha_);
            }

            // x̃ = xn[group_rows] — the permuted activation block.
            be_->gather_rows(ws_xg_, ws_xn_, plan_i, m, n_embd_);"""
E = sub(E, old, new, tag='visit plan')

old = """            be_->scatter_axpy_rows(ws_ffn_, ws_x2_, static_cast<const i32 *>(ws_plan_),
                                   static_cast<const f32 *>(ws_alpha_), m, n_embd_,"""
new = """            be_->scatter_axpy_rows(ws_ffn_, ws_x2_, plan_i, plan_f, m, n_embd_,"""
E = sub(E, old, new, tag='scatter')

# --------------------------------------------------------------------------
# 5. the drains
# --------------------------------------------------------------------------
old = """    be_->sync();
    // 16 bytes: the answer and the proof that it was not truncated. The old
    // path moved 993 kB and then compared 248320 of them on the host."""
new = """    if (topk_sync()) be_->sync();
    // 16 bytes: the answer and the proof that it was not truncated. The old
    // path moved 993 kB and then compared 248320 of them on the host."""
E = sub(E, old, new, tag='topk drain')

old = """    topk_valid_ = false;
    be_->sync();
    be_->download_f32(logits_host_, ws_logits_, n_vocab_);"""
new = """    topk_valid_ = false;
    if (logits_sync()) be_->sync();
    be_->download_f32(logits_host_, ws_logits_, n_vocab_);"""
E = sub(E, old, new, tag='logits drain')

old = """            head_compute(emitted - 1);
            be_->sync();
            be_->download_f32(logits_host_, ws_logits_, n_vocab_);"""
new = """            head_compute(emitted - 1);
            if (spec_row_sync()) be_->sync();
            be_->download_f32(logits_host_, ws_logits_, n_vocab_);"""
E = sub(E, old, new, tag='spec row drain', count=2)

save('src/engine.cpp', E, crlf_e)

# --------------------------------------------------------------------------
# 6. the two plan-staging vectors become members (no per-layer heap churn)
# --------------------------------------------------------------------------
old = """    std::vector<i32> plan_dev_;   // staging mirror for ws_plan_
    std::vector<f32> alpha_dev_;  // staging mirror for ws_alpha_"""
new = """    std::vector<i32> plan_dev_;   // staging mirror for ws_plan_
    std::vector<f32> alpha_dev_;  // staging mirror for ws_alpha_
    // The layer plan's segmentation: off(e) is expert e's first slot in
    // plan_dev_/alpha_dev_, and plan_cur_ is the bucket cursor the fill pass
    // walks. Both are members so a layer does not allocate.
    std::vector<i32> plan_off_;
    std::vector<i32> plan_cur_;"""
H = sub(H, old, new, tag='members')
save('include/krk/engine.hpp', H, crlf_h)

print('patched: engine.cpp (plan fold + 3 drain switches), engine.hpp (2 members)')
