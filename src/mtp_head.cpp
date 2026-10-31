// The HF `mtp.*` MTP head: qwen4exp / qwen4-mtp.
//
// This is the loader half of docs/PLAN-mtp-and-prefill.md steps 1.2-1.4. It
// reads the head set into the extended `MtpBlock` and checks every shape
// against the trunk it belongs to and against the head's own other tensors, so
// a head from a differently-pruned checkpoint is refused BY TENSOR NAME rather
// than loaded into a block that would draft plausible garbage.
//
// Three things it deliberately does:
//
//   * **One role table, not one artifact.** The converters disagree on the HF
//     spelling (`mtp.hyper_connection_mixer.input_mix_weight_down.weight` in
//     the vLLM-derived export, `mtp.hc_attn_down.weight` in the flattening one)
//     and on whether the dense prefix is one fused `eh_proj` or the
//     `fc_embedding` + `fc_hidden` pair. Both spellings and both forms load, and
//     which one a file used is reported, because "which artifact shape is this"
//     is a fact about the file, not something the loader should assume.
//
//   * **No copy of the expert bank.** `gate_up_proj` is one fused
//     `[2*n_ff_exp, n_embd, n_expert]` tensor. The gate rows are the first
//     `n_ff_exp` of every plane and the up rows the second; Gguf::
//     add_expert_plane_slice makes those two views with offset arithmetic and a
//     plane stride, so the 512-expert bank stays exactly one mapping region for
//     ExpertCache to page.
//
//   * **Nothing uploaded that is never read on the host.** The routed experts
//     are left in the file (637.5 MiB of the 810 MiB payload on the q2_0
//     repack); only the projections and norms become resident.
//
// What it does NOT do yet: run the block. Step 1.5 is the dense prefix on host,
// step 1.6 the block on the device. Until then `MtpBlock::hf` is true and the
// probe says so, which is the honest state -- a head that loads and validates
// but has no forward is not a drafter.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "krk/gguf.hpp"
#include "krk/model.hpp"
#include "krk/quant.hpp"

namespace krk {
namespace {

// One role and every spelling a converter has been seen to write for it. The
// empty slots are padding so the table is a fixed stride; the resolver stops at
// the first null. `%s` is the layer prefix (`mtp.layers.0.` or empty).
struct Role {
    const char *role;
    const char *names[5];
};

// Roles that belong to the one draft block. `%s` = "mtp.layers.0." .
const Role kBlockRoles[] = {
    {"hc_attn_norm",
     {"%shyper_connection_mixer.input_mix_norm_weight", "%shc_attn_norm.weight",
      nullptr}},
    {"hc_attn_down",
     {"%shyper_connection_mixer.input_mix_weight_down.weight",
      "%shc_attn_down.weight", nullptr}},
    {"hc_attn_up",
     {"%shyper_connection_mixer.input_mix_weight_up.weight",
      "%shc_attn_up.weight", nullptr}},
    {"hc_attn_inject",
     {"%shyper_connection_mixer.input_mix_inject_weight", "%shc_attn_inject.weight",
      "%shyper_connection_mixer.input_inject_weight", nullptr}},
    {"hc_ffn_norm",
     {"%shyper_connection_mixer.output_mix_norm_weight", "%shc_ffn_norm.weight",
      nullptr}},
    {"hc_ffn_down",
     {"%shyper_connection_mixer.output_mix_weight_down.weight",
      "%shc_ffn_down.weight", nullptr}},
    {"hc_ffn_up",
     {"%shyper_connection_mixer.output_mix_weight_up.weight",
      "%shc_ffn_up.weight", nullptr}},
    {"hc_ffn_inject",
     {"%shyper_connection_mixer.output_mix_inject_weight",
      "%shc_ffn_inject.weight", nullptr}},
    {"attn_q", {"%sself_attn.q_proj.weight", "%sattn_q.weight", nullptr}},
    {"attn_k", {"%sself_attn.k_proj.weight", "%sattn_k.weight", nullptr}},
    {"attn_v", {"%sself_attn.v_proj.weight", "%sattn_v.weight", nullptr}},
    {"attn_o", {"%sself_attn.o_proj.weight", "%sattn_output.weight", nullptr}},
    {"q_norm", {"%sself_attn.q_norm.weight", "%sattn_q_norm.weight", nullptr}},
    {"k_norm", {"%sself_attn.k_norm.weight", "%sattn_k_norm.weight", nullptr}},
    {"index_q",
     {"%sself_attn.indexer.q_proj.weight", "%sindexer.q_proj.weight",
      nullptr}},
    {"index_k",
     {"%sself_attn.indexer.k_proj.weight", "%sindexer.k_proj.weight",
      nullptr}},
    {"index_q_norm",
     {"%sself_attn.indexer.q_norm.weight", "%sindexer.q_norm.weight", nullptr}},
    {"index_k_norm",
     {"%sself_attn.indexer.k_norm.weight", "%sindexer.k_norm.weight", nullptr}},
    {"router", {"%smlp.gate.weight", "%sffn_gate_inp.weight", nullptr}},
    {"shexp_gate_inp",
     {"%smlp.shared_expert_gate.weight", "%sffn_gate_inp_shexp.weight",
      nullptr}},
    {"shexp_gate",
     {"%smlp.shared_expert.gate_proj.weight", "%sffn_gate_shexp.weight",
      nullptr}},
    {"shexp_up",
     {"%smlp.shared_expert.up_proj.weight", "%sffn_up_shexp.weight", nullptr}},
    {"shexp_down",
     {"%smlp.shared_expert.down_proj.weight", "%sffn_down_shexp.weight",
      nullptr}},
};

// Roles at the head's top level (`mtp.` with no layer prefix).
const Role kHeadRoles[] = {
    {"enorm", {"mtp.pre_fc_norm_embedding.weight", nullptr}},
    {"hnorm", {"mtp.pre_fc_norm_hidden.weight", nullptr}},
    {"eh_proj", {"mtp.eh_proj.weight", "mtp.layers.0.eh_proj.weight", nullptr}},
    {"fc_embedding", {"mtp.fc_embedding.weight", nullptr}},
    {"fc_hidden", {"mtp.fc_hidden.weight", nullptr}},
    {"hc_head_norm",
     {"mtp.final_mix_norm_weight", "mtp.hc_head_norm.weight",
      "mtp.hyper_connection_mixer.final_mix_norm_weight", nullptr}},
    {"hc_head_down",
     {"mtp.final_mix_weight_down.weight", "mtp.hc_head_down.weight",
      "mtp.hyper_connection_mixer.final_mix_weight_down.weight", nullptr}},
    {"hc_head_up",
     {"mtp.final_mix_weight_up.weight", "mtp.hc_head_up.weight",
      "mtp.hyper_connection_mixer.final_mix_weight_up.weight", nullptr}},
};

std::string fmt(const char *pat, const std::string &lp) {
    std::string out(pat);
    const size_t at = out.find("%s");
    if (at != std::string::npos) out.replace(at, 2, lp);
    return out;
}

// The fused bank, in the two spellings mtp_extract.py and the llama.cpp
// converters write. No `.weight`: it is a bare parameter.
const char *kGateUpNames[] = {"%smlp.experts.gate_up_proj",
                              "%sffn_gate_up_exps.weight",
                              "%smlp.experts.gate_up_proj.weight", nullptr};
const char *kDownNames[] = {"%smlp.experts.down_proj",
                            "%sffn_down_exps.weight",
                            "%smlp.experts.down_proj.weight", nullptr};

const GgufTensor *pick_named(const Gguf &g, const char *const *names,
                             const std::string &lp) {
    for (int i = 0; names[i]; i++) {
        const GgufTensor *t = g.tensor(fmt(names[i], lp));
        if (t) return t;
    }
    return nullptr;
}

}  // namespace

bool Model::load_mtp_hf_block(Gguf &g, const std::string &origin,
                              std::string *err) {
    const auto fail = [&](const std::string &m) {
        if (err) *err = origin + ": " + m;
        return false;
    };

    // ---- the block's layer prefix ------------------------------------------
    // `mtp.layers.<n>.` with the SMALLEST n, read from the tensors rather than
    // assumed: one draft block is what every qwen4exp export ships, but a
    // multi-block dump names them 0, 1, ... and picking the first in file order
    // would be picking whatever the writer happened to emit first.
    std::string lp = "mtp.layers.0.";
    i32 best = -1;
    for (const GgufTensor &t : g.tensors()) {
        static const char kPre[] = "mtp.layers.";
        if (t.name.rfind(kPre, 0) != 0) continue;
        const size_t dot = t.name.find('.', sizeof(kPre) - 1);
        if (dot == std::string::npos || dot == sizeof(kPre) - 1) continue;
        i32 n = 0;
        bool digits = true;
        for (size_t i = sizeof(kPre) - 1; i < dot; i++) {
            if (t.name[i] < '0' || t.name[i] > '9') { digits = false; break; }
            n = n * 10 + (t.name[i] - '0');
        }
        if (!digits) continue;
        if (best < 0 || n < best) {
            best = n;
            lp = t.name.substr(0, dot + 1);
        }
    }

    // ---- resolve every role ------------------------------------------------
    //
    // The resolver keeps NAMES, not pointers. `Gguf::tensor()` hands out
    // addresses inside a vector that the fused-bank views below append to, and a
    // pointer cached before that append points at freed storage -- which reads
    // as a plausible tensor with a garbage `n_elements` rather than as a crash.
    // Resolving on demand costs 31 linear scans over 31 entries, once, at load.
    struct Found {
        const char *role;
        std::string name;
    };
    std::vector<Found> found;
    const auto resolve = [&](const Role &r, const std::string &scope) {
        for (int i = 0; r.names[i]; i++) {
            const std::string n = fmt(r.names[i], scope);
            if (!g.tensor(n)) continue;
            found.push_back({r.role, n});
            return;
        }
        found.push_back({r.role, std::string()});
    };
    // The block roles take the layer prefix; the head roles their own fixed
    // spelling (their `%s` slots are unused, but resolve() still substitutes).
    for (const Role &r : kBlockRoles) resolve(r, lp);
    for (const Role &r : kHeadRoles) resolve(r, lp);

    const auto get = [&](const char *role) -> const GgufTensor * {
        for (const Found &f : found)
            if (std::strcmp(f.role, role) == 0)
                return f.name.empty() ? nullptr : g.tensor(f.name);
        return nullptr;
    };

    // ---- the fused expert bank, split into two views -----------------------
    const GgufTensor *gu = pick_named(g, kGateUpNames, lp);
    const GgufTensor *dn = pick_named(g, kDownNames, lp);
    if (!gu || !dn) {
        return fail(std::string("the head has no routed expert bank (") +
                    fmt(kGateUpNames[0], lp) + " and " + fmt(kDownNames[0], lp) +
                    ") -- an MTP block whose MoE is missing drafts from its "
                    "shared expert alone");
    }
    // A routed bank is [n_embd, 2*n_ff_exp, n_expert]: dim0 is the contiguous
    // inner dimension and dim1 the output rows, so the gate is rows
    // [0, n_ff_exp) of every expert's plane and the up rows are the second
    // half. `create_tensor_gate_up_exps` in llama.cpp fixes this order, and the
    // unfused `ffn_gate_exps`/`ffn_up_exps` banks use the same one.
    if (gu->n_dims != 3 || gu->ne[2] < 1)
        return fail(gu->name + " is " + std::to_string(gu->n_dims) +
                    "-D; a routed bank is [n_embd, 2*n_ff_exp, n_expert]");
    const i64 n_expert = static_cast<i64>(gu->ne[2]);
    const i64 n_embd_t = static_cast<i64>(gu->ne[0]);
    const i64 plane = static_cast<i64>(gu->ne[1]);
    if (plane % 2 != 0)
        return fail(gu->name + " is " + std::to_string(plane) +
                    " rows per expert; a fused gate+up bank is 2*n_ff_exp");
    const i64 n_ff_exp = plane / 2;
    if (dn->n_dims != 3 || static_cast<i64>(dn->ne[0]) != n_ff_exp ||
        static_cast<i64>(dn->ne[1]) != n_embd_t ||
        static_cast<i64>(dn->ne[2]) != n_expert)
        return fail(dn->name +
                    " does not match the fused bank: down_proj is [" +
                    std::to_string(dn->ne[0]) + ", " + std::to_string(dn->ne[1]) +
                    ", " + std::to_string(dn->ne[2]) + "] and the bank says [" +
                    std::to_string(n_ff_exp) + " x " + std::to_string(n_embd_t) +
                    " rows over " + std::to_string(n_expert) + " experts]");
    // Two views: gate = rows [0, n_ff_exp), up = rows [n_ff_exp, 2*n_ff_exp).
    // Both keep the source's plane stride, so expert e still lands on e.
    //
    // The names are COPIED before the directory grows: appending can reallocate
    // `tensors_`, and `gu`/`dn` point into it.
    const std::string gu_name = gu->name;
    const std::string dn_name = dn->name;
    const std::string gname = gu_name + ".gate_view";
    const std::string uname = gu_name + ".up_view";
    if (!g.add_expert_plane_slice(gname, gu_name, 0, n_ff_exp, n_embd_t) ||
        !g.add_expert_plane_slice(uname, gu_name, n_ff_exp, n_ff_exp, n_embd_t))
        return fail("could not split " + gu->name +
                    " into gate/up plane views (rows per plane " +
                    std::to_string(plane) + ")");
    // From here on `gu`/`dn` are stale: add_expert_plane_slice appended to the
    // directory vector. Re-resolve by name.
    gu = g.tensor(gu_name);
    dn = g.tensor(dn_name);
    const GgufTensor *vg = g.tensor(gname);
    const GgufTensor *vu = g.tensor(uname);
    if (!gu || !dn || !vg || !vu)
        return fail("the expert bank's plane views vanished from the directory");

    // ---- required roles ----------------------------------------------------
    const struct {
        const char *role;
        const char *what;
    } required[] = {
        {"enorm", "the embedding norm"},   {"hnorm", "the hidden norm"},
        {"router", "the expert router"},   {"attn_q", "the query projection"},
        {"attn_k", "the key projection"},  {"attn_v", "the value projection"},
        {"attn_o", "the output projection"},
        {"hc_attn_down", "the attention hyper-connection mixer"},
        {"hc_ffn_down", "the FFN hyper-connection mixer"},
        {"shexp_up", "the shared expert"}, {"shexp_down", "the shared expert"},
    };
    for (const auto &r : required) {
        if (get(r.role)) continue;
        return fail(std::string("the head has no ") + r.what + " (`" +
                    r.role + "`)");
    }
    const bool split = get("fc_embedding") != nullptr;
    if (split) {
        if (!get("fc_hidden"))
            return fail("the head has mtp.fc_embedding.weight but no "
                        "mtp.fc_hidden.weight -- the dense prefix needs both "
                        "halves or none (the fused form is mtp.eh_proj.weight)");
    } else if (!get("eh_proj")) {
        return fail("the head has neither mtp.eh_proj.weight nor the "
                    "mtp.fc_embedding + mtp.fc_hidden pair -- the dense prefix "
                    "that fuses the hidden state with the proposed token's "
                    "embedding is what makes this a next-token block");
    }

    // ---- geometry, against the trunk --------------------------------------
    MtpBlock &m = mtp_;
    m.hf = true;
    m.layer = best < 0 ? 0 : best;
    if (cfg_.n_embd <= 0)
        return fail("the trunk does not say how wide it is");
    m.n_embd = cfg_.n_embd;
    if (n_embd_t != cfg_.n_embd)
        return fail(gu->name + " carries " + std::to_string(n_embd_t) +
                    "-wide experts and this model is " +
                    std::to_string(cfg_.n_embd) +
                    " -- the head belongs to another checkpoint");
    const GgufTensor *hn = get("hnorm");
    const i64 hc_dim = hn->n_elements;
    const i64 en = get("enorm")->n_elements;
    if (en != cfg_.n_embd)
        return fail(std::string(get("enorm")->name) + " is " +
                    std::to_string(en) + " wide and this model is " +
                    std::to_string(cfg_.n_embd));
    if (hc_dim <= 0 || hc_dim % cfg_.n_embd != 0)
        return fail(std::string(hn->name) + " is " + std::to_string(hc_dim) +
                    " wide, which is not a whole number of " +
                    std::to_string(cfg_.n_embd) + "-wide lanes");
    m.n_lane = static_cast<i32>(hc_dim / cfg_.n_embd);
    m.n_head = cfg_.n_head;
    m.n_head_kv = cfg_.n_head_kv;
    m.head_dim = cfg_.head_dim;
    m.n_expert_used = cfg_.n_expert_used;

    // The dense prefix: `eh_proj` is [2*n_embd, n_embd] and each half in the
    // split form is [n_embd, n_embd]; the split form is applied per lane.
    const GgufTensor *ep = get("eh_proj");
    if (ep) {
        if (static_cast<i64>(ep->ne[0]) != 2 * cfg_.n_embd ||
            static_cast<i64>(ep->ne[1]) != cfg_.n_embd)
            return fail(std::string(ep->name) + " is [" +
                        std::to_string(ep->ne[0]) + ", " +
                        std::to_string(ep->ne[1]) +
                        "], and a fused prefix over [e | h_lane] is [" +
                        std::to_string(2 * cfg_.n_embd) + ", " +
                        std::to_string(cfg_.n_embd) + "]");
    } else {
        for (const char *r : {"fc_embedding", "fc_hidden"}) {
            const GgufTensor *t = get(r);
            if (static_cast<i64>(t->ne[0]) != cfg_.n_embd ||
                static_cast<i64>(t->ne[1]) != cfg_.n_embd)
                return fail(std::string(t->name) + " is [" +
                            std::to_string(t->ne[0]) + ", " +
                            std::to_string(t->ne[1]) +
                            "]; the split prefix is applied per lane, so each "
                            "half is [n_embd, n_embd]");
        }
    }
    m.eh_proj_split = split;

    // The mixers: down is [hc_dim, rank], up [rank, hc_dim], inject [n_lane].
    const auto mixer_geom = [&](const char *down, const char *up,
                                const char *inject) -> i32 {
        const GgufTensor *d = get(down);
        const GgufTensor *u = get(up);
        if (static_cast<i64>(d->ne[0]) != hc_dim)
            return -1;
        const i64 rank = static_cast<i64>(d->ne[1]);
        if (static_cast<i64>(u->ne[0]) != rank ||
            static_cast<i64>(u->ne[1]) != hc_dim)
            return -2;
        if (inject) {
            const GgufTensor *j = get(inject);
            if (j && j->n_elements != static_cast<i64>(m.n_lane)) return -3;
        }
        return static_cast<i32>(rank);
    };
    i32 rank = mixer_geom("hc_attn_down", "hc_attn_up", "hc_attn_inject");
    const auto why_mixer = [&](i32 r, const char *what) {
        if (r == -1)
            return fail(std::string("the ") + what + " mixer's down projection "
                        "is not " + std::to_string(hc_dim) +
                        " wide -- it must span every lane");
        if (r == -2)
            return fail(std::string("the ") + what + " mixer's up projection "
                        "does not mirror its down projection");
        return fail(std::string("the ") + what + " mixer's inject row is not " +
                    std::to_string(m.n_lane) + " long -- one weight per lane");
    };
    if (rank < 0) return why_mixer(rank, "attention");
    const i32 rank_ffn = mixer_geom("hc_ffn_down", "hc_ffn_up", "hc_ffn_inject");
    if (rank_ffn < 0) return why_mixer(rank_ffn, "FFN");
    if (rank_ffn != rank)
        return fail("the attention mixer is rank " + std::to_string(rank) +
                    " and the FFN mixer rank " + std::to_string(rank_ffn) +
                    " -- one block, one bottleneck");
    m.n_hc_rank = rank;
    if (get("hc_head_down")) {
        const i32 rh = mixer_geom("hc_head_down", "hc_head_up", nullptr);
        if (rh < 0)
            return fail("the head mixer (mtp.final_mix_*) does not mirror like "
                        "the other two");
    }

    // The router and the bank agree about how many experts there are.
    const GgufTensor *rt = get("router");
    if (static_cast<i64>(rt->ne[0]) != cfg_.n_embd ||
        static_cast<i64>(rt->ne[1]) != n_expert)
        return fail(std::string(rt->name) + " is [" +
                    std::to_string(rt->ne[0]) + ", " +
                    std::to_string(rt->ne[1]) + "] and the bank says [" +
                    std::to_string(cfg_.n_embd) + ", " +
                    std::to_string(n_expert) + "]");
    m.n_expert = static_cast<i32>(n_expert);
    m.n_ff_exp = static_cast<i32>(n_ff_exp);

    // Attention: q may pack a gate (n_head * head_dim * 2), k/v are GQA.
    const i64 qo = static_cast<i64>(get("attn_q")->ne[1]);
    const i64 ko = static_cast<i64>(get("attn_k")->ne[1]);
    if (cfg_.n_head > 0 && cfg_.head_dim > 0) {
        const i64 want = static_cast<i64>(cfg_.n_head) * cfg_.head_dim;
        if (qo != want && qo != 2 * want)
            return fail(std::string(get("attn_q")->name) + " is " +
                        std::to_string(qo) + " wide, and the trunk's " +
                        std::to_string(cfg_.n_head) + " x " +
                        std::to_string(cfg_.head_dim) + " says " +
                        std::to_string(want) + " (or twice that with a gate)");
        if (cfg_.n_head_kv > 0 &&
            ko != static_cast<i64>(cfg_.n_head_kv) * cfg_.head_dim)
            return fail(std::string(get("attn_k")->name) + " is " +
                        std::to_string(ko) + " wide and " +
                        std::to_string(cfg_.n_head_kv) + " kv head(s) x " +
                        std::to_string(cfg_.head_dim) + " says " +
                        std::to_string(static_cast<i64>(cfg_.n_head_kv) *
                                       cfg_.head_dim));
    }

    // The indexer is all four or none (llama.cpp's own qwen4exp loader makes the
    // same rule: two of them is a half-written export).
    i32 n_index = 0;
    for (const char *r : {"index_q", "index_k", "index_q_norm", "index_k_norm"})
        if (get(r)) ++n_index;
    if (n_index != 0 && n_index != 4)
        return fail("the head has " + std::to_string(n_index) +
                    " of the 4 QSA indexer tensors (q_proj, k_proj, q_norm, "
                    "k_norm) -- a partial indexer cannot score a block");
    if (n_index == 4) {
        const i64 kd = static_cast<i64>(get("index_k")->ne[1]);
        const i64 qd = static_cast<i64>(get("index_q")->ne[1]);
        if (kd <= 0 || qd % kd != 0)
            return fail(std::string(get("index_q")->name) + " is " +
                        std::to_string(qd) + " wide and " +
                        std::string(get("index_k")->name) + " is " +
                        std::to_string(kd) +
                        " -- the indexer's query is whole heads of its key");
        m.index_dim = static_cast<i32>(kd);
        m.n_index_head = static_cast<i32>(qd / kd);
        if (get("index_k_norm")->n_elements != kd)
            return fail(std::string(get("index_k_norm")->name) +
                        " does not span the indexer's key width");
    }

    // ---- convert: dense matrices verbatim, norms dequantized --------------
    if (!be_) return fail("no backend to upload the head to");
    const auto upload_q = [&](const GgufTensor *t, QuantTensor *out) {
        if (!t) return true;   // legal absence; the required roles are checked
        out->data = be_->alloc(t->n_bytes);
        if (!out->data) return false;
        be_->upload(out->data, t->data, t->n_bytes);
        out->type = t->type;
        out->n_in = static_cast<i64>(t->ne[0]);
        out->n_out = t->n_dims >= 2 ? static_cast<i64>(t->ne[1]) : 1;
        out->gs = 1.0f;
        m.bytes += static_cast<size_t>(t->n_bytes);
        return true;
    };
    const auto upload_f = [&](const GgufTensor *t) -> f32 * {
        if (!t) return nullptr;
        const i64 n = t->n_elements;
        f32 *host = static_cast<f32 *>(host_alloc(static_cast<size_t>(n) * 4));
        if (!host) return nullptr;
        dequant_row(t->type, t->data, host, n);
        f32 *dev = static_cast<f32 *>(be_->alloc(static_cast<size_t>(n) * 4));
        if (dev) be_->upload(dev, host, static_cast<size_t>(n) * 4);
        host_free(host);
        if (!dev) return nullptr;
        m.bytes += static_cast<size_t>(n) * 4;
        return dev;
    };

    bool ok = upload_q(get("attn_q"), &m.wq) && upload_q(get("attn_k"), &m.wk) &&
              upload_q(get("attn_v"), &m.wv) && upload_q(get("attn_o"), &m.wo) &&
              upload_q(get("router"), &m.router) &&
              upload_q(get("shexp_up"), &m.shexp_up) &&
              upload_q(get("shexp_down"), &m.shexp_down);
    if (ok) {
        ok = upload_q(get("shexp_gate"), &m.shexp_gate) &&
             upload_q(get("shexp_gate_inp"), &m.shexp_inp_gate);
    }
    if (ok) {
        if (split) {
            ok = upload_q(get("fc_embedding"), &m.fc_embedding) &&
                 upload_q(get("fc_hidden"), &m.fc_hidden);
        } else {
            ok = upload_q(get("eh_proj"), &m.eh_proj);
        }
    }
    if (ok && n_index == 4)
        ok = upload_q(get("index_q"), &m.index_q) &&
             upload_q(get("index_k"), &m.index_k);
    const auto load_mixer = [&](const char *dn_, const char *up,
                                const char *inj, MtpBlock::HcMixer *dst) {
        if (!ok) return;
        ok = upload_q(get(dn_), &dst->down) && upload_q(get(up), &dst->up);
        if (ok && inj && get(inj)) ok = upload_q(get(inj), &dst->inject);
    };
    load_mixer("hc_attn_down", "hc_attn_up", "hc_attn_inject", &m.hc_attn);
    load_mixer("hc_ffn_down", "hc_ffn_up", "hc_ffn_inject", &m.hc_ffn);
    if (ok && get("hc_head_down"))
        load_mixer("hc_head_down", "hc_head_up", nullptr, &m.hc_head);
    if (!ok)
        return fail("a weight upload was refused by the backend (allocation)");

    // Norms last: they are host-dequantized then uploaded, and a null here is a
    // real failure rather than a legal absence.
    m.attn_norm = upload_f(get("hc_attn_norm"));
    m.q_norm = upload_f(get("q_norm"));
    m.k_norm = upload_f(get("k_norm"));
    m.enorm = upload_f(get("enorm"));
    m.hnorm = upload_f(get("hnorm"));
    m.hc_attn.norm = upload_f(get("hc_attn_norm"));
    m.hc_ffn.norm = upload_f(get("hc_ffn_norm"));
    m.hc_head.norm = upload_f(get("hc_head_norm"));
    m.index_q_norm = upload_f(get("index_q_norm"));
    m.index_k_norm = upload_f(get("index_k_norm"));
    if (!m.enorm || !m.hnorm || !m.hc_attn.norm || !m.hc_ffn.norm)
        return fail("the head's norms could not be read");

    // The bank stays in the file for ExpertCache; only the two views are kept.
    m.exps_gate_up = gu;
    m.exps_gate = vg;
    m.exps_up = vu;
    m.exps_down = dn;
    m.n_ff_shexp = m.shexp_up.present() ? static_cast<i32>(m.shexp_up.n_out)
                                        : 0;
    m.present = true;

    KRK_INFO("MTP head %s: %d lane(s) x %d, %d experts top-%d, %d-wide FFN, "
             "%d-wide indexer, prefix %s, fwd=not implemented (plan 1.5-1.7), "
             "%.1f MiB resident, %lld MiB of experts left in the file",
             origin.c_str(), m.n_lane, m.n_embd, m.n_expert, m.n_expert_used,
             m.n_ff_exp, m.index_dim,
             split ? "fc_embedding+fc_hidden" : "fused eh_proj",
             static_cast<f64>(m.bytes) / 1048576.0,
             static_cast<long long>(gu->n_bytes) / 1048576LL);
    return true;
}

}  // namespace krk
