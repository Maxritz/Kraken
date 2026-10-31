// ============================================================================
//  dspark.cpp -- see include/krk/dspark.hpp.
//
//  The block is decoded in one non-causal attention pass and then walked left to
//  right through the Markov head. Those are deliberately two separate loops: the
//  transformer part has no dependence between rows (that is the point of a
//  parallel block drafter), and the bigram part has all of it. Collapsing them
//  would serialise the expensive half.
// ============================================================================
#include "krk/dspark.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace krk {

namespace {

inline char *act_at(void *base, size_t act_size, i64 elem) {
    return static_cast<char *>(base) + static_cast<size_t>(elem) * act_size;
}
inline const char *act_at(const void *base, size_t act_size, i64 elem) {
    return static_cast<const char *>(base) + static_cast<size_t>(elem) * act_size;
}

void rms_scale(f32 *out, const f32 *x, const f32 *w, i64 n, f32 eps) {
    f64 ss = 0.0;
    for (i64 i = 0; i < n; i++) ss += static_cast<f64>(x[i]) * x[i];
    const f32 inv =
        static_cast<f32>(1.0 / std::sqrt(ss / static_cast<f64>(n) + eps));
    for (i64 i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

void upload_act(Backend &be, void *dst, const f32 *src, i64 n) {
    if (be.act_type() == DType::F16) {
        std::vector<u16> tmp(static_cast<size_t>(n));
        for (i64 i = 0; i < n; i++)
            tmp[static_cast<size_t>(i)] = fp32_to_fp16(src[i]);
        be.upload(dst, tmp.data(), static_cast<size_t>(n) * 2);
    } else {
        be.upload(dst, src, static_cast<size_t>(n) * 4);
    }
}

void build_inv_freq(std::vector<f32> &tab, i32 n_dims, f32 base) {
    const i64 h = std::max<i32>(1, n_dims / 2);
    tab.assign(static_cast<size_t>(h), 0.0f);
    for (i64 i = 0; i < h; i++) {
        const f32 exponent = static_cast<f32>(2 * i) / static_cast<f32>(n_dims);
        tab[static_cast<size_t>(i)] = std::pow(base, -exponent);
    }
}

// `<name>` and `<name>.weight` are the same tensor under two spelling habits,
// and both appear in the wild for head sets. One helper so every read tries
// both instead of each site remembering.
const GgufTensor *tensor2(const Gguf &g, const std::string &name) {
    if (const GgufTensor *t = g.tensor(name)) return t;
    return g.tensor(name + ".weight");
}

} // namespace

bool dspark_arch(std::string_view arch) {
    // `dspark` is the head set this file implements. `deepseek4-dspark` and
    // `qwen4-mtp` are different objects -- an MTP shard and a gated 512-expert
    // block -- and admitting them here would mean running a graph they were not
    // trained for. They keep their own entries and their own refusals.
    return arch == "dspark";
}

DsparkDraft::~DsparkDraft() { unload(); }

void DsparkDraft::free_dev() {
    if (!be_) return;
    void *ptrs[] = {cap_,      feat_,     enc_out_,  ws_x_,     ws_xn_,
                    ws_q_,     ws_k_,     ws_v_,     ws_attn_,  ws_gate_,
                    ws_up_,    ws_logits_, ws_markov_, ws_mbias_, kcache_,
                    vcache_,   ws_gs_};
    for (void *p : ptrs)
        if (p) be_->release(p);
    f32 *fw[] = {out_norm_, aux_norm_};
    for (f32 *p : fw)
        if (p) be_->release(p);
    for (LayerWeights &L : layers_) {
        void *lp[] = {L.attn_norm, L.ffn_norm, L.q_norm, L.k_norm};
        for (void *p : lp)
            if (p) be_->release(p);
        QuantTensor *q[] = {&L.wq, &L.wk, &L.wv, &L.wo, &L.wgate, &L.wup, &L.wdown};
        for (QuantTensor *t : q)
            if (t->data) be_->release(t->data);
        L = LayerWeights{};
    }
    layers_.clear();
    QuantTensor *heads[] = {&fc_, &markov_w1_, &markov_w2_, &conf_};
    for (QuantTensor *t : heads) {
        if (t->data) be_->release(t->data);
        *t = QuantTensor{};
    }
}

void DsparkDraft::unload() {
    free_dev();
    cap_ = feat_ = enc_out_ = ws_x_ = ws_xn_ = ws_q_ = ws_k_ = ws_v_ = nullptr;
    ws_attn_ = ws_gate_ = ws_up_ = ws_logits_ = ws_markov_ = nullptr;
    ws_mbias_ = kcache_ = vcache_ = ws_gs_ = nullptr;
    out_norm_ = aux_norm_ = nullptr;
    gguf_.reset();
    be_ = nullptr;
    committed_ = 0;
}

void *DsparkDraft::capture_slot(i32 aux) const {
    if (!cap_ || aux < 0 || aux >= static_cast<i32>(target_layers_.size()))
        return nullptr;
    const size_t as = be_ ? be_->act_size() : 0;
    return act_at(cap_, as,
                  static_cast<i64>(aux) * chunk_ * static_cast<i64>(cfg_.n_embd));
}

bool DsparkDraft::load(Backend &be, const std::string &path, i64 chunk, i64 ctx,
                       i32 target_embd, i32 target_layer_count, i32 target_vocab,
                       std::string *err) {
    be_ = &be;
    chunk_ = std::max<i64>(1, chunk);
    gguf_ = std::make_unique<Gguf>();
    if (!gguf_->load(path, err)) return false;

    cfg_.arch = gguf_->get_str("general.architecture", "");
    cfg_.name = gguf_->get_str("general.name", "dspark");
    if (!dspark_arch(cfg_.arch)) {
        if (err)
            *err = "not a DSpark head set: general.architecture is '" +
                   cfg_.arch + "' (expected dspark)";
        return false;
    }

    auto key = [&](const char *suffix) { return cfg_.arch + suffix; };
    auto has = [&](const std::string &k) { return gguf_->find(k) != nullptr; };

    cfg_.n_layer = static_cast<i32>(gguf_->get_i64(key(".block_count"), 0));
    cfg_.n_embd = static_cast<i32>(gguf_->get_i64(key(".embedding_length"), 0));
    cfg_.n_ff = static_cast<i32>(gguf_->get_i64(key(".feed_forward_length"), 0));
    cfg_.n_head = static_cast<i32>(gguf_->get_i64(key(".attention.head_count"), 0));
    cfg_.n_head_kv = static_cast<i32>(
        gguf_->get_i64(key(".attention.head_count_kv"), cfg_.n_head));
    cfg_.head_dim = static_cast<i32>(gguf_->get_i64(key(".attention.key_length"), 0));
    if (cfg_.head_dim <= 0 && cfg_.n_head > 0)
        cfg_.head_dim = cfg_.n_embd / cfg_.n_head;
    cfg_.rms_eps =
        static_cast<f32>(gguf_->get_f64(key(".attention.layer_norm_rms_epsilon"), 1e-6));
    cfg_.n_vocab = static_cast<i32>(gguf_->get_i64(key(".vocab_size"), 0));
    const i32 declared_vocab = cfg_.n_vocab;
    if (target_vocab > cfg_.n_vocab) cfg_.n_vocab = target_vocab;
    if (declared_vocab > 0 && target_vocab > 0 && declared_vocab != target_vocab)
        KRK_WARN("dspark: head set declares a %d-token vocabulary, the target has "
                 "%d; the block borrows the target's head, so the wider of the "
                 "two sizes every workspace",
                 declared_vocab, target_vocab);
    if (cfg_.n_vocab <= 0) {
        if (err)
            *err = "DSpark head set has no vocabulary: neither the file nor the "
                   "target declares one";
        return false;
    }
    cfg_.n_ctx_train = static_cast<i32>(gguf_->get_i64(key(".context_length"), 0));
    if (cfg_.n_layer <= 0 || cfg_.n_embd <= 0 || cfg_.n_head <= 0 ||
        cfg_.head_dim <= 0 || cfg_.n_ff <= 0) {
        if (err) *err = "DSpark head set is missing core geometry";
        return false;
    }
    if (cfg_.n_head % cfg_.n_head_kv != 0) {
        if (err) *err = "DSpark head count is not a multiple of the KV head count";
        return false;
    }

    // ---- the capture list ------------------------------------------------
    if (const std::vector<i32> *tl = gguf_->get_i32_array(key(".target_layers")))
        target_layers_ = *tl;
    if (target_layers_.empty()) {
        if (err) *err = "DSpark head set declares no target_layers";
        return false;
    }
    for (i32 id : target_layers_) {
        if (id < 0 || id > target_layer_count) {
            if (err)
                *err = format("DSpark wants the input of layer %d, but the target "
                              "has %d layers and this drafter captures at most the "
                              "pre-final-norm state (%d)",
                              id, target_layer_count, target_layer_count);
            return false;
        }
    }

    block_size_ = static_cast<i32>(gguf_->get_i64(key(".block_size"), 16));
    if (block_size_ < 2) block_size_ = 16;

    mask_id_ = -1;
    if (has(key(".mask_token_id")))
        mask_id_ = static_cast<i32>(gguf_->get_i64(key(".mask_token_id"), -1));
    if (mask_id_ < 0)
        mask_id_ = static_cast<i32>(gguf_->get_i64("tokenizer.ggml.mask_token_id", -1));
    if (mask_id_ < 0)
        mask_id_ =
            static_cast<i32>(gguf_->get_i64("tokenizer.ggml.unknown_token_id", 0));

    // The block is non-causal by construction. The flag is here because the
    // file is the only thing that can say otherwise, and because it is the one
    // switch that changes what the block means, so a run has to be able to
    // report which way it went.
    causal_ = gguf_->get_bool(key(".attention.causal"), false);
    if (const char *e = std::getenv("KRK_DSPARK_CAUSAL")) causal_ = e[0] != '0';

    cfg_.rope_base = static_cast<f32>(gguf_->get_f64(key(".rope.freq_base"), 10000.0));
    i32 n_rot = static_cast<i32>(gguf_->get_i64(key(".rope.dimension_count"), 0));
    if (n_rot <= 0 || n_rot > cfg_.head_dim) n_rot = cfg_.head_dim;
    cfg_.rope_frac = static_cast<f32>(n_rot) / static_cast<f32>(cfg_.head_dim);
    cfg_.rope_scale = 1.0f;
    cfg_.rope_neox = gguf_->get_bool(key(".rope.neox"), false);
    build_inv_freq(inv_freq_, n_rot, cfg_.rope_base);

    // The fusion width is written by the file, but the only value it can have
    // that the target can actually feed is n_aux * target_embd. A file that
    // disagrees with the target it was handed is a refusal, not a resize.
    n_embd_enc_ = static_cast<i32>(
        gguf_->get_i64(key(".encoder.embd"), static_cast<i64>(target_embd) *
                                                 static_cast<i64>(target_layers_.size())));
    if (n_embd_enc_ != target_embd * static_cast<i32>(target_layers_.size())) {
        if (err)
            *err = format("DSpark fuses %d feature columns (%d layers x %d wide), "
                          "but this target provides %d x %d; the head set was "
                          "built for a different model",
                          n_embd_enc_, static_cast<i32>(target_layers_.size()),
                          target_embd, static_cast<i32>(target_layers_.size()),
                          target_embd);
        return false;
    }

    kv_cap_ = std::max<i64>(ctx + block_size_ + 8, 64);

    markov_on_ = gguf_->get_bool(key(".markov.enabled"), true);
    if (const char *e = std::getenv("KRK_DSPARK_MARKOV")) markov_on_ = e[0] != '0';
    markov_rank_ = static_cast<i32>(gguf_->get_i64(key(".markov.rank"), 0));
    if (const char *e = std::getenv("KRK_DSPARK_CONF"))
        conf_cut_ = static_cast<f32>(std::atof(e));

    if (!load_weights(err)) return false;

    // ---- workspace -------------------------------------------------------
    const size_t as = be_->act_size();
    const i64 E = cfg_.n_embd;
    const i64 n_aux = static_cast<i64>(target_layers_.size());
    const i64 q_dim = static_cast<i64>(cfg_.n_head) * cfg_.head_dim;
    const i64 kv_dim = static_cast<i64>(cfg_.n_head_kv) * cfg_.head_dim;
    cap_ = be_->alloc(static_cast<size_t>(n_aux * chunk_ * E) * as);
    feat_ = be_->alloc(static_cast<size_t>(chunk_ * n_embd_enc_) * as);
    enc_out_ = be_->alloc(static_cast<size_t>(chunk_ * E) * as);
    const i64 W = std::max<i64>(chunk_, block_size_);
    ws_x_ = be_->alloc(static_cast<size_t>(W * E) * as);
    ws_xn_ = be_->alloc(static_cast<size_t>(W * E) * as);
    ws_q_ = be_->alloc(static_cast<size_t>(W * q_dim) * as);
    ws_k_ = be_->alloc(static_cast<size_t>(W * kv_dim) * as);
    ws_v_ = be_->alloc(static_cast<size_t>(W * kv_dim) * as);
    ws_attn_ = be_->alloc(static_cast<size_t>(W * q_dim) * as);
    ws_gate_ = be_->alloc(static_cast<size_t>(W * cfg_.n_ff) * as);
    ws_up_ = be_->alloc(static_cast<size_t>(W * cfg_.n_ff) * as);
    ws_logits_ = be_->alloc(static_cast<size_t>(block_size_) *
                            static_cast<size_t>(cfg_.n_vocab) * as);
    // One scratch buffer serves the Markov head, the token embedding it reads,
    // and the confidence projection -- so it is sized for the widest of the
    // three, not for the rank. Sizing it by the rank alone left it NULL on a
    // file without a Markov head, and the confidence head then wrote through a
    // null pointer (found by the head set with KRK_DSPARK_MARKOV=0 in
    // test_dspark_head_set).
    {
        const i64 want = std::max<i64>(
            E, std::max<i64>(static_cast<i64>(block_size_) *
                                 std::max<i32>(1, markov_rank_),
                             1));
        ws_markov_ = be_->alloc(static_cast<size_t>(want) * as);
    }
    // The Markov head's own output row is a whole vocabulary wide, which is
    // wider than any FFN scratch on a small head set -- writing it through the
    // FFN buffer overran it by exactly the vocabulary deficit (found by this
    // group's fixture, which is 32 wide with a 260-token vocabulary).
    ws_mbias_ = be_->alloc(static_cast<size_t>(cfg_.n_vocab) * as);
    kcache_ = be_->alloc(static_cast<size_t>(kv_cap_ * cfg_.n_layer * kv_dim) * as);
    vcache_ = be_->alloc(static_cast<size_t>(kv_cap_ * cfg_.n_layer * kv_dim) * as);
    if (!cap_ || !feat_ || !enc_out_ || !ws_x_ || !ws_xn_ || !ws_q_ || !ws_k_ ||
        !ws_v_ || !ws_attn_ || !ws_gate_ || !ws_up_ || !ws_logits_ || !ws_mbias_ ||
        !kcache_ || !vcache_) {
        if (err) *err = "DSpark head set could not allocate its workspaces";
        return false;
    }
    be_->fill0(kcache_, static_cast<size_t>(kv_cap_ * cfg_.n_layer * kv_dim) * as);
    be_->fill0(vcache_, static_cast<size_t>(kv_cap_ * cfg_.n_layer * kv_dim) * as);

    feat_host_.assign(static_cast<size_t>(chunk_ * n_embd_enc_), 0.0f);
    tmp_host_.assign(static_cast<size_t>(cfg_.n_vocab), 0.0f);
    markov_bias_.assign(static_cast<size_t>(cfg_.n_vocab), 0.0f);
    block_tok_.assign(static_cast<size_t>(block_size_), mask_id_);
    last_conf_.assign(static_cast<size_t>(block_size_), 1.0f);
    markov_prev_.assign(static_cast<size_t>(block_size_), -1);
    return true;
}

bool DsparkDraft::load_weights(std::string *err) {
    const i64 q_dim = static_cast<i64>(cfg_.n_head) * cfg_.head_dim;
    const i64 kv_dim = static_cast<i64>(cfg_.n_head_kv) * cfg_.head_dim;

    auto upload_f32_w = [&](const GgufTensor *t) -> f32 * {
        if (!t) return nullptr;
        const i64 n = t->n_elements;
        f32 *host = static_cast<f32 *>(host_alloc(static_cast<size_t>(n) * 4));
        dequant_row(t->type, t->data, host, n);
        f32 *dev = static_cast<f32 *>(be_->alloc(static_cast<size_t>(n) * 4));
        be_->upload(dev, host, static_cast<size_t>(n) * 4);
        host_free(host);
        return dev;
    };
    auto upload_w = [&](const GgufTensor *t, const std::string &name,
                        i64 want_out, i64 want_in) -> QuantTensor {
        QuantTensor q;
        if (!t) return q;
        const i64 n_in = static_cast<i64>(t->ne[0]);
        const i64 n_out = t->n_elements / std::max<i64>(1, n_in);
        if (t->type == DType::Unknown || !dtype_supported(t->type)) {
            if (err) *err = "weight '" + name + "' uses an unsupported format";
            return q;
        }
        if (want_in > 0 && n_in != want_in) {
            if (err)
                *err = format("weight '%s' has inner width %lld, expected %lld",
                              name.c_str(), static_cast<long long>(n_in),
                              static_cast<long long>(want_in));
            return q;
        }
        if (want_out > 0 && n_out != want_out) {
            if (err)
                *err = format("weight '%s' has %lld rows, expected %lld",
                              name.c_str(), static_cast<long long>(n_out),
                              static_cast<long long>(want_out));
            return q;
        }
        if (t->type == DType::F32) {
            f32 *tmp = static_cast<f32 *>(
                host_alloc(static_cast<size_t>(t->n_elements) * 4));
            dequant_row(DType::F32, t->data, tmp, t->n_elements);
            u16 *half = static_cast<u16 *>(
                host_alloc(static_cast<size_t>(t->n_elements) * 2));
            for (i64 i = 0; i < t->n_elements; i++)
                half[i] = fp32_to_fp16(tmp[i]);
            q.data = be_->alloc(static_cast<size_t>(t->n_elements) * 2);
            be_->upload(q.data, half, static_cast<size_t>(t->n_elements) * 2);
            host_free(tmp);
            host_free(half);
            q.type = DType::F16;
        } else {
            q.data = be_->alloc(t->n_bytes);
            be_->upload(q.data, t->data, t->n_bytes);
            q.type = t->type;
        }
        q.n_in = n_in;
        q.n_out = n_out;
        q.gs = 1.0f;
        static const char *kW = ".weight";
        const size_t wn = std::strlen(kW);
        const std::string base =
            name.size() > wn && name.compare(name.size() - wn, wn, kW) == 0
                ? name.substr(0, name.size() - wn)
                : name;
        if (const GgufTensor *s = gguf_->tensor(base + ".scale")) {
            if (s->data && s->n_elements >= 1 && s->type == DType::F32) {
                q.gs = reinterpret_cast<const f32 *>(s->data)[0];
                if (q.gs != 1.0f && q.n_in > gs_in_) gs_in_ = q.n_in;
            }
        }
        return q;
    };

    fc_ = upload_w(tensor2(*gguf_, "fc"), "fc", 0, n_embd_enc_);
    if (!fc_.data) {
        if (err) *err = "DSpark head set has no fusion projection 'fc'";
        return false;
    }
    out_norm_ = upload_f32_w(tensor2(*gguf_, "output_norm"));
    if (!out_norm_) {
        if (err) *err = "DSpark head set has no 'output_norm'";
        return false;
    }

    const i64 n_aux = static_cast<i64>(target_layers_.size());
    aux_norm_host_.assign(static_cast<size_t>(cfg_.n_embd * n_aux), 1.0f);
    {
        // Stacked `enc.aux_norm` when present, otherwise one gain per aux, which
        // is how the per-layer spellings write it.
        if (const GgufTensor *t = tensor2(*gguf_, "enc.aux_norm")) {
            std::vector<f32> v(static_cast<size_t>(t->n_elements));
            dequant_row(t->type, t->data, v.data(), t->n_elements);
            const i64 per = t->n_elements / std::max<i64>(1, n_aux);
            for (i64 a = 0; a < n_aux; a++)
                for (i64 i = 0; i < cfg_.n_embd && i < per; i++)
                    aux_norm_host_[static_cast<size_t>(a * cfg_.n_embd + i)] =
                        v[static_cast<size_t>(a * per + i)];
        } else {
            for (i64 a = 0; a < n_aux; a++) {
                const std::string nm = format("aux_hidden_norm.%lld",
                                              static_cast<long long>(a));
                if (const GgufTensor *t = tensor2(*gguf_, nm)) {
                    std::vector<f32> v(static_cast<size_t>(t->n_elements));
                    dequant_row(t->type, t->data, v.data(), t->n_elements);
                    for (i64 i = 0; i < cfg_.n_embd && i < t->n_elements; i++)
                        aux_norm_host_[static_cast<size_t>(a * cfg_.n_embd + i)] =
                            v[static_cast<size_t>(i)];
                }
            }
        }
        aux_norm_ = static_cast<f32 *>(
            be_->alloc(static_cast<size_t>(aux_norm_host_.size()) * 4));
        be_->upload(aux_norm_, aux_norm_host_.data(),
                    aux_norm_host_.size() * 4);
    }

    layers_.assign(static_cast<size_t>(cfg_.n_layer), LayerWeights{});
    for (i32 l = 0; l < cfg_.n_layer; l++) {
        LayerWeights &L = layers_[static_cast<size_t>(l)];
        const std::string p = format("blk.%d.", l);
        auto T = [&](const char *suffix) { return tensor2(*gguf_, p + suffix); };
        L.attn_norm = upload_f32_w(T("attn_norm"));
        L.ffn_norm = upload_f32_w(T("ffn_norm"));
        L.q_norm = upload_f32_w(T("attn_q_norm"));
        L.k_norm = upload_f32_w(T("attn_k_norm"));
        L.wq = upload_w(T("attn_q"), p + "attn_q", q_dim, cfg_.n_embd);
        L.wk = upload_w(T("attn_k"), p + "attn_k", kv_dim, cfg_.n_embd);
        L.wv = upload_w(T("attn_v"), p + "attn_v", kv_dim, cfg_.n_embd);
        L.wo = upload_w(T("attn_output"), p + "attn_output", cfg_.n_embd, q_dim);
        L.wgate = upload_w(T("ffn_gate"), p + "ffn_gate", cfg_.n_ff, cfg_.n_embd);
        L.wup = upload_w(T("ffn_up"), p + "ffn_up", cfg_.n_ff, cfg_.n_embd);
        L.wdown = upload_w(T("ffn_down"), p + "ffn_down", cfg_.n_embd, cfg_.n_ff);
        if (!L.attn_norm || !L.ffn_norm || !L.wq.data || !L.wk.data ||
            !L.wv.data || !L.wo.data || !L.wgate.data || !L.wup.data || !L.wdown.data) {
            if (err)
                *err = format("DSpark block %d is missing a required tensor", l);
            return false;
        }
    }

    // ---- the two side heads, both optional -------------------------------
    if (markov_on_) {
        const GgufTensor *w1 = tensor2(*gguf_, "markov.w1");
        const GgufTensor *w2 = tensor2(*gguf_, "markov.w2");
        if (w1 && w2) {
            if (markov_rank_ <= 0) markov_rank_ = static_cast<i32>(w1->ne[0]);
            markov_w1_ = upload_w(w1, "markov.w1", markov_rank_, cfg_.n_embd);
            markov_w2_ = upload_w(w2, "markov.w2", cfg_.n_vocab, markov_rank_);
            if (!markov_w1_.data || !markov_w2_.data) {
                if (err) *err = "DSpark markov head has inconsistent shapes";
                return false;
            }
            if (!ws_gs_ && (markov_w1_.scaled() || markov_w2_.scaled())) {
                ws_gs_ = be_->alloc(static_cast<size_t>(gs_in_) *
                                    be_->act_size());
            }
        } else if (w1 || w2) {
            if (err)
                *err = "DSpark head set carries one half of the Markov head "
                       "(w1 without w2 or the reverse)";
            return false;
        }
    }
    conf_ = upload_w(tensor2(*gguf_, "confidence.proj"), "confidence.proj", 1,
                     cfg_.n_embd);
    if (!conf_.data && gguf_->tensor("confidence.proj") == nullptr &&
        gguf_->tensor("confidence.proj.weight") == nullptr)
        conf_ = QuantTensor{};
    return true;
}

namespace {

// One `dst = x * W^T` with the weight's NVFP4 global scale applied to the INPUT,
// which is the rule the trunk uses and the only order that does not overflow the
// f16 store.
void gemm_scaled(Backend &be, void *dst, const void *x, const QuantTensor &w,
                 i64 n_out, i64 rows, void *gs_ws) {
    const i64 n_in = w.n_in;
    if (!w.scaled() || gs_ws == nullptr) {
        be.gemm(dst, x, w.data, w.type, n_out, n_in, rows);
        return;
    }
    const size_t as = be.act_size();
    const f32 inv = 1.0f / w.gs;
    // Scale the input into the workspace, one row at a time.
    std::vector<f32> tmp(static_cast<size_t>(n_in));
    for (i64 r = 0; r < rows; r++) {
        be.download_f32(tmp.data(),
                        static_cast<const char *>(x) + static_cast<size_t>(r * n_in) * as,
                        n_in);
        for (i64 i = 0; i < n_in; i++) tmp[static_cast<size_t>(i)] *= inv;
        // The workspace is written through the backend's own element type.
        void *slot = static_cast<char *>(gs_ws) +
                     static_cast<size_t>(r * n_in) * as;
        if (be.act_type() == DType::F16) {
            std::vector<u16> half(static_cast<size_t>(n_in));
            for (i64 i = 0; i < n_in; i++)
                half[static_cast<size_t>(i)] = fp32_to_fp16(tmp[static_cast<size_t>(i)]);
            be.upload(slot, half.data(), static_cast<size_t>(n_in) * 2);
        } else {
            be.upload(slot, tmp.data(), static_cast<size_t>(n_in) * 4);
        }
    }
    be.gemm(dst, gs_ws, w.data, w.type, n_out, n_in, rows);
}

} // namespace

bool DsparkDraft::commit(i32 pos0, i32 n) {
    if (!be_ || n <= 0 || pos0 < 0) return false;
    if (static_cast<i64>(pos0 + n) > kv_cap_) return false;
    const size_t as = be_->act_size();
    const i64 E = cfg_.n_embd;
    const i64 n_aux = static_cast<i64>(target_layers_.size());
    const i64 W = n_embd_enc_;

    // ---- fuse the captured features --------------------------------------
    // Host side on purpose: the gain varies down the rows of the feature buffer,
    // which is a shape no existing kernel has. Cost is one round trip per
    // committed chunk, not per token.
    be_->download_f32(feat_host_.data(), cap_,
                      static_cast<i64>(n_aux * static_cast<i64>(n) * E));
    std::vector<f32> fused(static_cast<size_t>(n * W), 0.0f);
    for (i32 t = 0; t < n; t++) {
        f32 *dst = fused.data() + static_cast<size_t>(t * W);
        for (i64 a = 0; a < n_aux; a++) {
            const f32 *src = feat_host_.data() +
                             static_cast<size_t>((a * n + t) * E);
            rms_scale(dst + static_cast<size_t>(a * E), src,
                      aux_norm_host_.data() + static_cast<size_t>(a * E), E,
                      cfg_.rms_eps);
        }
    }
    upload_act(*be_, feat_, fused.data(), static_cast<i64>(n) * W);
    gemm_scaled(*be_, enc_out_, feat_, fc_, E, n, ws_gs_);

    // ---- inject K/V --------------------------------------------------------
    const i64 hd = cfg_.head_dim;
    const i64 nh = cfg_.n_head;
    const i64 nkv = cfg_.n_head_kv;
    const i64 kv_dim = nkv * hd;
    be_->rmsnorm(ws_xn_, enc_out_, nullptr, n, E, cfg_.rms_eps);
    gemm_scaled(*be_, ws_k_, ws_xn_, layers_[0].wk, kv_dim, n, ws_gs_);
    gemm_scaled(*be_, ws_v_, ws_xn_, layers_[0].wv, kv_dim, n, ws_gs_);
    // The injection rotates K only, but both backend entry points take a q
    // pointer and both dereference it (rope_apply() walks it unconditionally),
    // so the q slot is passed as scratch rather than as null. Nothing reads it
    // afterwards; the alternative is a second rotor entry point that exists
    // only to be handed a null.
    be_->qk_norm(ws_q_, ws_k_, nullptr, layers_[0].k_norm, 1, nkv, hd, n,
                 cfg_.rms_eps, false);
    be_->rope(ws_q_, ws_k_, nh, nkv, hd, n, pos0, inv_freq_.data(),
              cfg_.rope_scale, cfg_.rope_frac, cfg_.rope_neox);
    AttnDesc d;
    d.n_head = nh;
    d.n_kv = nkv;
    d.hd = hd;
    d.n_tok = n;
    d.pos0 = pos0;
    d.layer_stride = kv_cap_ * kv_dim;
    d.pos_stride = kv_dim;
    d.scale = static_cast<f32>(1.0 / std::sqrt(static_cast<f64>(hd)));
    d.causal = true;
    d.layer = 0;
    be_->kv_append(kcache_, vcache_, ws_k_, ws_v_, d);
    committed_ += n;
    return true;
}

i32 DsparkDraft::draft_block(const QuantTensor &tok_embd, const QuantTensor &lm_head,
                             i32 n_vocab, i32 id_last, i32 n_draft, i32 pos,
                             f32 *rows_out) {
    if (!be_ || n_draft <= 0 || n_vocab <= 0) return 0;
    if (n_draft > block_size_ - 1) n_draft = block_size_ - 1;
    const i32 nt = n_draft + 1;
    if (static_cast<i64>(pos + nt) > kv_cap_) return 0;

    const size_t as = be_->act_size();
    const i64 E = cfg_.n_embd;
    const i64 hd = cfg_.head_dim;
    const i64 nh = cfg_.n_head;
    const i64 nkv = cfg_.n_head_kv;
    const i64 kv_dim = nkv * hd;
    const i64 q_dim = nh * hd;

    for (i32 i = 0; i < nt; i++)
        block_tok_[static_cast<size_t>(i)] = i == 0 ? id_last : mask_id_;
    be_->embed(ws_x_, tok_embd.data, tok_embd.type, n_vocab, E,
               block_tok_.data(), nt);

    AttnDesc d;
    d.n_head = nh;
    d.n_kv = nkv;
    d.hd = hd;
    d.n_tok = nt;
    d.pos0 = pos;
    d.layer_stride = kv_cap_ * kv_dim;
    d.pos_stride = kv_dim;
    d.scale = static_cast<f32>(1.0 / std::sqrt(static_cast<f64>(hd)));
    // The block is non-causal: every candidate row sees the whole block, which
    // is what makes the candidates parallel instead of a chain.
    d.causal = causal_;

    for (i32 l = 0; l < cfg_.n_layer; l++) {
        const LayerWeights &L = layers_[static_cast<size_t>(l)];
        be_->rmsnorm(ws_xn_, ws_x_, L.attn_norm, nt, E, cfg_.rms_eps);
        gemm_scaled(*be_, ws_q_, ws_xn_, L.wq, q_dim, nt, ws_gs_);
        gemm_scaled(*be_, ws_k_, ws_xn_, L.wk, kv_dim, nt, ws_gs_);
        gemm_scaled(*be_, ws_v_, ws_xn_, L.wv, kv_dim, nt, ws_gs_);
        be_->qk_norm(ws_q_, ws_k_, L.q_norm, L.k_norm, nh, nkv, hd, nt,
                     cfg_.rms_eps, false);
        be_->rope(ws_q_, ws_k_, nh, nkv, hd, nt, pos, inv_freq_.data(),
                  cfg_.rope_scale, cfg_.rope_frac, cfg_.rope_neox);
        d.layer = l;
        be_->kv_append(kcache_, vcache_, ws_k_, ws_v_, d);
        be_->attention(ws_attn_, ws_q_, kcache_, vcache_, d);
        gemm_scaled(*be_, ws_x_, ws_attn_, L.wo, E, nt, ws_gs_);
        be_->add_residual(ws_x_, ws_attn_, nt, E, nullptr, nullptr);
        be_->rmsnorm(ws_xn_, ws_x_, L.ffn_norm, nt, E, cfg_.rms_eps);
        gemm_scaled(*be_, ws_gate_, ws_xn_, L.wgate, cfg_.n_ff, nt, ws_gs_);
        gemm_scaled(*be_, ws_up_, ws_xn_, L.wup, cfg_.n_ff, nt, ws_gs_);
        be_->silu_mul(ws_gate_, ws_gate_, ws_up_, static_cast<i64>(nt) * cfg_.n_ff,
                      cfg_.n_ff);
        gemm_scaled(*be_, ws_up_, ws_gate_, L.wdown, E, nt, ws_gs_);
        be_->add_residual(ws_x_, ws_up_, nt, E, nullptr, nullptr);
    }

    // ---- read every row out through the target's head --------------------
    be_->rmsnorm(ws_xn_, ws_x_, out_norm_, nt, E, cfg_.rms_eps);
    for (i32 i = 0; i < nt; i++)
        gemm_scaled(*be_,
                    act_at(ws_logits_, as, static_cast<i64>(i) * n_vocab),
                    act_at(ws_xn_, as, static_cast<i64>(i) * E), lm_head, n_vocab,
                    1, ws_gs_);

    // ---- the Markov recurrence over the block -----------------------------
    // Row k's bias comes from the token at row k-1. Row 0 is the already-known
    // `id_last`, so row 1 has a real predecessor; row k's own token is only
    // known once row k-1 has been read, so this loop is sequential on purpose
    // and it is cheap: one rank-wide projection and one vocab projection per
    // row, against a whole transformer stack in the parallel half.
    last_conf_.assign(static_cast<size_t>(nt), 1.0f);
    markov_prev_.assign(static_cast<size_t>(nt), -1);

    if (has_confidence()) {
        for (i32 i = 0; i < nt; i++) {
            void *row_logit = act_at(ws_logits_, as, static_cast<i64>(i) * n_vocab);
            (void)row_logit;
            // conf = sigmoid(proj . xn). The projection is one row, so the
            // backend GEMM writes a single value per candidate.
            gemm_scaled(*be_, ws_markov_, act_at(ws_xn_, as, static_cast<i64>(i) * E),
                        conf_, 1, 1, ws_gs_);
            be_->download_f32(tmp_host_.data(), ws_markov_, 1);
            last_conf_[static_cast<size_t>(i)] =
                static_cast<f32>(1.0 / (1.0 + std::exp(-tmp_host_[0])));
        }
    }

    i32 prev = id_last;
    i32 emitted = 0;
    for (i32 k = 1; k < nt; k++) {
        void *row = act_at(ws_logits_, as, static_cast<i64>(k) * n_vocab);
        if (has_markov() && markov_on_ && markov_rank_ > 0) {
            // bias = w2 @ relu(w1 @ emb(prev))
            be_->embed(ws_markov_, tok_embd.data, tok_embd.type, n_vocab, E,
                       &prev, 1);
            gemm_scaled(*be_, ws_x_, ws_markov_, markov_w1_, markov_rank_, 1,
                        ws_gs_);
            // relu
            be_->download_f32(tmp_host_.data(), ws_x_, markov_rank_);
            for (i32 i = 0; i < markov_rank_; i++)
                tmp_host_[static_cast<size_t>(i)] =
                    std::max(0.0f, tmp_host_[static_cast<size_t>(i)]);
            upload_act(*be_, ws_markov_, tmp_host_.data(), markov_rank_);
            gemm_scaled(*be_, ws_mbias_, ws_markov_, markov_w2_, n_vocab, 1,
                        ws_gs_);
            be_->download_f32(markov_bias_.data(), ws_mbias_, n_vocab);
            be_->download_f32(tmp_host_.data(), row, n_vocab);
            for (i32 v = 0; v < n_vocab; v++)
                tmp_host_[static_cast<size_t>(v)] +=
                    markov_bias_[static_cast<size_t>(v)];
            upload_act(*be_, row, tmp_host_.data(), n_vocab);
            markov_prev_[static_cast<size_t>(k)] = prev;
        }
        // The candidate for row k, which is the argmax of the (biased) row.
        be_->download_f32(tmp_host_.data(), row, n_vocab);
        i32 best = 0;
        f32 bv = tmp_host_[0];
        for (i32 v = 1; v < n_vocab; v++)
            if (tmp_host_[static_cast<size_t>(v)] > bv) {
                bv = tmp_host_[static_cast<size_t>(v)];
                best = v;
            }
        prev = best;

        // Confidence cut: stop proposing at the first row the head is not sure
        // about. Off unless KRK_DSPARK_CONF asks for it, because a block cut
        // short is cheaper but not obviously better.
        if (conf_cut_ > 0.0f && last_conf_[static_cast<size_t>(k)] < conf_cut_) break;

        std::memcpy(rows_out + static_cast<size_t>(emitted) * n_vocab,
                    tmp_host_.data(), static_cast<size_t>(n_vocab) * sizeof(f32));
        emitted++;
    }
    mask_rows_ = emitted;
    return emitted;
}

std::string DsparkDraft::describe() const {
    return format("dspark %s: %d blocks, %d embd, %d/%d heads, ff %d, block %d, "
                  "mask %d, %s block, markov %s (rank %d), confidence %s",
                  cfg_.name.c_str(), cfg_.n_layer, cfg_.n_embd, cfg_.n_head,
                  cfg_.n_head_kv, cfg_.n_ff, block_size_, mask_id_,
                  causal_ ? "causal" : "non-causal",
                  has_markov() ? "on" : "absent", markov_rank_,
                  has_confidence() ? "on" : "absent");
}

} // namespace krk
