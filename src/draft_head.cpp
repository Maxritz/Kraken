#include "krk/draft_head.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "krk/gguf.hpp"

namespace krk {
namespace {

const char *kMetaPrefix = "krk.draft.";

// Reads a `krk.draft.*` integer key, and refuses a file that omits a key the
// geometry needs: a zeroed width silently produces an empty forward, which looks
// like a head that proposes nothing.
bool need_u32(const Gguf &g, const std::string &name, i32 *out, std::string *err) {
    const i64 v = g.get_i64(kMetaPrefix + name, 0);
    if (!g.find(kMetaPrefix + name)) {
        if (err) *err = "draft head: missing key " + std::string(kMetaPrefix) + name;
        return false;
    }
    if (v <= 0) {
        if (err) *err = "draft head: " + std::string(kMetaPrefix) + name +
                        " must be positive, got " + std::to_string(v);
        return false;
    }
    *out = static_cast<i32>(v);
    return true;
}

// f16 weights, f32 arithmetic. A row is converted as it is read, which is one
// pass over the row and no persistent f32 copy.
f32 h2f(u16 h) {
    const u32 sign = static_cast<u32>(h & 0x8000u) << 16;
    const u32 exp = (h >> 10) & 0x1fu;
    const u32 man = h & 0x3ffu;
    u32 bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            // Subnormal: renormalise into an f32 exponent.
            u32 e = 127 - 15 + 1;
            u32 m = man;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                e--;
            }
            bits = sign | (e << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    f32 f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// y[o] = W[o, :] . x  for an f16 row-major matrix of [n_out, n_in].
void matvec(const std::vector<u16> &w, const f32 *x, i32 n_out, i32 n_in, f32 *y) {
    for (i32 o = 0; o < n_out; o++) {
        const u16 *row = w.data() + static_cast<size_t>(o) * static_cast<size_t>(n_in);
        f32 acc = 0;
        for (i32 i = 0; i < n_in; i++) acc += h2f(row[i]) * x[i];
        y[o] = acc;
    }
}

void rms_norm(const f32 *x, const std::vector<u16> &gain, i32 n, f32 eps, f32 *out) {
    f64 ss = 0;
    for (i32 i = 0; i < n; i++) ss += static_cast<f64>(x[i]) * x[i];
    const f32 inv = static_cast<f32>(1.0 / std::sqrt(ss / n + eps));
    for (i32 i = 0; i < n; i++) out[i] = x[i] * inv * h2f(gain[static_cast<size_t>(i)]);
}

// Rotary over one head's head_dim at `pos`, the NeoX pairing (i, i+hd/2), which
// is the one EAGLE heads are trained with.
void rope(f32 *v, i32 n_head, i32 head_dim, i32 pos, f32 theta) {
    for (i32 h = 0; h < n_head; h++) {
        f32 *base = v + static_cast<size_t>(h) * static_cast<size_t>(head_dim);
        const i32 half = head_dim / 2;
        for (i32 i = 0; i < half; i++) {
            const f32 inv = std::pow(theta, -static_cast<f32>(2 * i) /
                                                static_cast<f32>(head_dim));
            const f32 ang = static_cast<f32>(pos) * inv;
            const f32 c = std::cos(ang), s = std::sin(ang);
            const f32 a = base[i], b = base[i + half];
            base[i] = a * c - b * s;
            base[i + half] = a * s + b * c;
        }
    }
}

bool copy_tensor(const Gguf &g, const std::string &name, std::vector<u16> *out,
                 size_t want, std::string *err) {
    const GgufTensor *t = g.tensor(name);
    if (!t) {
        if (err) *err = "draft head: missing tensor " + name;
        return false;
    }
    if (t->type != DType::F16) {
        if (err) *err = "draft head: tensor " + name +
                        " must be F16, got type " +
                        std::to_string(static_cast<int>(t->type));
        return false;
    }
    const size_t n = t->n_bytes / 2;
    if (want && n != want) {
        if (err) *err = "draft head: tensor " + name + " has " + std::to_string(n) +
                        " values, geometry needs " + std::to_string(want);
        return false;
    }
    out->resize(n);
    std::memcpy(out->data(), t->data, n * 2);
    return true;
}

} // namespace

bool DraftHead::load(const std::string &path, std::string *err) {
    loaded_ = false;
    path_ = path;
    Gguf g;
    if (!g.load(path, err)) return false;

    std::string kind = "eagle3";
    if (g.find("krk.draft.kind")) kind = g.get_str("krk.draft.kind", "eagle3");
    if (kind != "eagle3") {
        if (err) *err = "draft head: kind '" + kind + "' is not supported (eagle3 is)";
        return false;
    }
    cfg_ = DraftHeadConfig{};
    cfg_.kind = kind;
    cfg_.target_arch = g.find("krk.draft.target_arch")
                           ? g.get_str("krk.draft.target_arch", "")
                           : std::string();
    if (!need_u32(g, "hidden", &cfg_.hidden, err)) return false;
    if (!need_u32(g, "target_hidden", &cfg_.target_hidden, err)) return false;
    if (!need_u32(g, "n_layer", &cfg_.n_layer, err)) return false;
    if (!need_u32(g, "n_head", &cfg_.n_head, err)) return false;
    if (!need_u32(g, "n_kv", &cfg_.n_kv, err)) return false;
    if (!need_u32(g, "head_dim", &cfg_.head_dim, err)) return false;
    if (!need_u32(g, "n_ff", &cfg_.n_ff, err)) return false;
    if (!need_u32(g, "draft_vocab", &cfg_.draft_vocab, err)) return false;
    if (!need_u32(g, "target_vocab", &cfg_.target_vocab, err)) return false;
    cfg_.n_in = 1;
    if (g.find("krk.draft.n_in")) {
        i32 n_in = 0;
        if (!need_u32(g, "n_in", &n_in, err)) return false;
        cfg_.n_in = n_in;
    }
    cfg_.rope_theta = g.find("krk.draft.rope_theta")
                          ? static_cast<f32>(g.get_f64("krk.draft.rope_theta", 10000.0))
                          : 10000.0f;
    cfg_.rms_eps = g.find("krk.draft.rms_eps")
                       ? static_cast<f32>(g.get_f64("krk.draft.rms_eps", 1e-5))
                       : 1e-5f;
    if (cfg_.n_kv > cfg_.n_head || cfg_.n_head % cfg_.n_kv != 0) {
        if (err) *err = "draft head: n_kv " + std::to_string(cfg_.n_kv) +
                        " does not divide n_head " + std::to_string(cfg_.n_head);
        return false;
    }
    // One layer is what the published EAGLE-3 heads are; the loader accepts more
    // only if it also carries the per-layer tensors, which the loop below reads.
    const size_t h = static_cast<size_t>(cfg_.hidden);
    const size_t th = static_cast<size_t>(cfg_.target_hidden);
    const size_t qd = static_cast<size_t>(cfg_.n_head) * static_cast<size_t>(cfg_.head_dim);
    const size_t kd = static_cast<size_t>(cfg_.n_kv) * static_cast<size_t>(cfg_.head_dim);

    if (!copy_tensor(g, "fc.weight", &fc_, h * th * static_cast<size_t>(cfg_.n_in + 1), err))
        return false;
    if (!copy_tensor(g, "norm.weight", &lnf_, h, err)) return false;
    if (!copy_tensor(g, "layers.0.input_layernorm.weight", &ln1_, h, err)) return false;
    if (!copy_tensor(g, "layers.0.post_attention_layernorm.weight", &ln2_, h, err))
        return false;
    if (!copy_tensor(g, "layers.0.self_attn.q_proj.weight", &wq_, qd * h, err)) return false;
    if (!copy_tensor(g, "layers.0.self_attn.k_proj.weight", &wk_, kd * h, err)) return false;
    if (!copy_tensor(g, "layers.0.self_attn.v_proj.weight", &wv_, kd * h, err)) return false;
    if (!copy_tensor(g, "layers.0.self_attn.o_proj.weight", &wo_, h * qd, err)) return false;
    if (!copy_tensor(g, "layers.0.mlp.gate_proj.weight", &gate_,
                     static_cast<size_t>(cfg_.n_ff) * h, err))
        return false;
    if (!copy_tensor(g, "layers.0.mlp.up_proj.weight", &up_,
                     static_cast<size_t>(cfg_.n_ff) * h, err))
        return false;
    if (!copy_tensor(g, "layers.0.mlp.down_proj.weight", &down_,
                     h * static_cast<size_t>(cfg_.n_ff), err))
        return false;
    if (!copy_tensor(g, "lm_head.weight", &lm_,
                     static_cast<size_t>(cfg_.draft_vocab) * h, err))
        return false;
    if (g.tensor("d2t")) {
        const GgufTensor *t = g.tensor("d2t");
        // F32, not GGML_TYPE_I32: the container describes float and quantized
        // tensor types only (an integer-typed entry is reported as unknown,
        // with no payload pointer), so an id map that a tool wrote as I32 is
        // not readable here at all. tools/eagle_to_gguf.py writes F32 for the
        // same reason; 4 bytes an id is 800 KiB at a 200k vocabulary.
        if (t->type != DType::F32) {
            if (err)
                *err = "draft head: d2t must be F32 (ids stored as floats); got "
                       "tensor type id " + std::to_string(t->type_id) +
                       " -- reconvert it with tools/eagle_to_gguf.py";
            return false;
        }
        if (t->n_elements <= 0) {
            if (err) *err = "draft head: d2t is empty";
            return false;
        }
        if (t->n_elements != cfg_.draft_vocab) {
            if (err)
                *err = "draft head: d2t holds " + std::to_string(t->n_elements) +
                       " id(s) but the draft vocabulary is " +
                       std::to_string(cfg_.draft_vocab);
            return false;
        }
        const size_t n = static_cast<size_t>(t->n_elements);
        d2t_.resize(n);
        const f32 *ids = reinterpret_cast<const f32 *>(t->data);
        for (size_t i = 0; i < n; i++) {
            const f32 v = ids[i];
            // A non-integral or out-of-range id is a broken map, and rounding a
            // wrong id into range would be a proposal for a token the head was
            // never trained to map to.
            if (!std::isfinite(v) || std::floor(v) != v) {
                if (err)
                    *err = "draft head: d2t[" + std::to_string(i) +
                           "] = " + std::to_string(v) + " is not an id";
                return false;
            }
            const i32 id = static_cast<i32>(v);
            if (id < 0 || id >= cfg_.target_vocab) {
                if (err) *err = "draft head: d2t[" + std::to_string(i) + "] = " +
                                std::to_string(id) +
                                " is outside the target vocabulary";
                return false;
            }
            d2t_[i] = id;
        }
        cfg_.has_d2t = true;
    }

    x_.resize(th * static_cast<size_t>(cfg_.n_in + 1));
    h_.resize(h);
    hn_.resize(h);
    q_.resize(qd);
    k_.resize(kd);
    v_.resize(kd);
    att_.resize(qd);
    proj_.resize(h);
    ff_.resize(static_cast<size_t>(cfg_.n_ff) * 2);
    out_.resize(h);
    loaded_ = true;
    return true;
}

bool DraftHead::check_target(i32 target_hidden, i32 target_vocab, std::string *err) const {
    if (!loaded_) {
        if (err) *err = "draft head: not loaded";
        return false;
    }
    if (cfg_.target_hidden != target_hidden) {
        if (err) *err = "draft head: built for a target hidden width of " +
                        std::to_string(cfg_.target_hidden) + ", this model's is " +
                        std::to_string(target_hidden);
        return false;
    }
    if (cfg_.target_vocab != target_vocab) {
        if (err) *err = "draft head: built for a target vocabulary of " +
                        std::to_string(cfg_.target_vocab) + ", this model's is " +
                        std::to_string(target_vocab) +
                        " -- a head from another tokenizer proposes ids the "
                        "verifier cannot accept";
        return false;
    }
    return true;
}

void DraftHead::reset(i32 ctx) {
    ctx_ = ctx > 0 ? ctx : 1;
    pos_ = 0;
    base_ = 0;
    const size_t n = static_cast<size_t>(ctx_) *
                     static_cast<size_t>(cfg_.n_kv) * static_cast<size_t>(cfg_.head_dim);
    k_cache_.assign(n, 0.0f);
    v_cache_.assign(n, 0.0f);
}

bool DraftHead::step(const f32 *feat, const f32 *tok_embd, i32 pos,
                     std::vector<f32> *logits, bool want_logits) {
    if (!loaded_ || !feat || !tok_embd) return false;
    if (want_logits && !logits) return false;
    if (pos < 0 || pos >= ctx_) return false;
    const i32 th = cfg_.target_hidden;
    const i32 h = cfg_.hidden;
    const i32 qd = cfg_.n_head * cfg_.head_dim;
    const i32 kd = cfg_.n_kv * cfg_.head_dim;

    for (i32 s = 0; s < cfg_.n_in; s++)
        std::memcpy(x_.data() + static_cast<size_t>(s) * th,
                    feat + static_cast<size_t>(s) * th, sizeof(f32) * th);
    std::memcpy(x_.data() + static_cast<size_t>(cfg_.n_in) * th, tok_embd,
                sizeof(f32) * th);

    matvec(fc_, x_.data(), h, th * (cfg_.n_in + 1), h_.data());

    rms_norm(h_.data(), ln1_, h, cfg_.rms_eps, hn_.data());
    matvec(wq_, hn_.data(), qd, h, q_.data());
    matvec(wk_, hn_.data(), kd, h, k_.data());
    matvec(wv_, hn_.data(), kd, h, v_.data());
    rope(q_.data(), cfg_.n_head, cfg_.head_dim, pos, cfg_.rope_theta);
    rope(k_.data(), cfg_.n_kv, cfg_.head_dim, pos, cfg_.rope_theta);

    f32 *kc = k_cache_.data() + static_cast<size_t>(pos) * kd;
    f32 *vc = v_cache_.data() + static_cast<size_t>(pos) * kd;
    std::memcpy(kc, k_.data(), sizeof(f32) * kd);
    std::memcpy(vc, v_.data(), sizeof(f32) * kd);

    // A prime step has already published what it is for. Skipping the rest is
    // what makes a windowed prime affordable: the block's attention, MLP and
    // head cost several times its own projections.
    if (!want_logits) {
        pos_ = pos + 1;
        return true;
    }        const i32 t0 = base_;
        const i32 group = cfg_.n_head / cfg_.n_kv;
    for (i32 hh = 0; hh < cfg_.n_head; hh++) {
        const i32 kvh = hh / group;
        const f32 *qp = q_.data() + static_cast<size_t>(hh) * cfg_.head_dim;
        f32 best = -1e30f;
        std::vector<f32> sc(static_cast<size_t>(pos - t0 + 1));
        for (i32 t = t0; t <= pos; t++) {
            const f32 *kp = k_cache_.data() + static_cast<size_t>(t) * kd +
                            static_cast<size_t>(kvh) * cfg_.head_dim;
            f32 acc = 0;
            for (i32 d = 0; d < cfg_.head_dim; d++) acc += qp[d] * kp[d];
            acc /= std::sqrt(static_cast<f32>(cfg_.head_dim));
            sc[static_cast<size_t>(t - t0)] = acc;
            if (acc > best) best = acc;
        }
        f32 sum = 0;
        for (i32 t = t0; t <= pos; t++) {
            const f32 e = std::exp(sc[static_cast<size_t>(t - t0)] - best);
            sc[static_cast<size_t>(t - t0)] = e;
            sum += e;
        }
        f32 *op = att_.data() + static_cast<size_t>(hh) * cfg_.head_dim;
        for (i32 d = 0; d < cfg_.head_dim; d++) op[d] = 0.0f;
        for (i32 t = t0; t <= pos; t++) {
            const f32 p = sc[static_cast<size_t>(t - t0)] / sum;
            const f32 *vp = v_cache_.data() + static_cast<size_t>(t) * kd +
                            static_cast<size_t>(kvh) * cfg_.head_dim;
            for (i32 d = 0; d < cfg_.head_dim; d++) op[d] += p * vp[d];
        }
    }
    matvec(wo_, att_.data(), h, qd, proj_.data());
    for (i32 i = 0; i < h; i++) h_[i] += proj_[i];

    rms_norm(h_.data(), ln2_, h, cfg_.rms_eps, hn_.data());
    matvec(gate_, hn_.data(), cfg_.n_ff, h, ff_.data());
    matvec(up_, hn_.data(), cfg_.n_ff, h, ff_.data() + cfg_.n_ff);
    for (i32 i = 0; i < cfg_.n_ff; i++) {
        const f32 g = ff_[static_cast<size_t>(i)];
        const f32 u = ff_[static_cast<size_t>(i) + cfg_.n_ff];
        // silu(g) * u
        ff_[static_cast<size_t>(i)] = (g / (1.0f + std::exp(-g))) * u;
    }
    matvec(down_, ff_.data(), h, cfg_.n_ff, proj_.data());
    for (i32 i = 0; i < h; i++) h_[i] += proj_[i];

    rms_norm(h_.data(), lnf_, h, cfg_.rms_eps, out_.data());
    logits->resize(static_cast<size_t>(cfg_.draft_vocab));
    matvec(lm_, out_.data(), cfg_.draft_vocab, h, logits->data());
    pos_ = pos + 1;
    return true;
}

i32 DraftHead::argmax(const std::vector<f32> &logits) {
    i32 best = -1;
    f32 bv = -1e30f;
    for (size_t i = 0; i < logits.size(); i++)
        if (logits[i] > bv) {
            bv = logits[i];
            best = static_cast<i32>(i);
        }
    return best;
}

i32 DraftHead::to_target(i32 draft_id) const {
    if (draft_id < 0) return -1;
    if (!cfg_.has_d2t) return draft_id < cfg_.target_vocab ? draft_id : -1;
    if (draft_id >= static_cast<i32>(d2t_.size())) return -1;
    return d2t_[static_cast<size_t>(draft_id)];
}

} // namespace krk
