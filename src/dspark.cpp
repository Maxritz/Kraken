// ============================================================================
//  dspark.cpp -- see include/krk/dspark.hpp.
//
//  Small file on purpose. The block decode is DFlash's; this adds exactly the
//  two things the reference puts on top of it, and the arithmetic is copied from
//  that reference line for line:
//
//      w1_prev = row(markov_w1, prev)
//      bias    = markov_w2 @ w1_prev            (no activation)
//      col     = row_logits + bias
//      prev    = argmax(col)
//
//      conf    = sigmoid(dot(conf_proj, [hidden_row ; w1_prev]) + conf_bias)
//
//  Both run on the host. The whole per-row cost is one rank-wide gather, one
//  rank x vocab matvec and one vocab-wide add; that is nothing beside the block
//  forward that produced the row, and keeping it on the host means the chain
//  (which is inherently serial) cannot serialise any device work.
// ============================================================================
#include "krk/dspark.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace krk {

namespace {

// Every spelling a converter has used for one of these tensors. The reference
// writes `markov_w1.weight`; a hand-repacked file uses `markov.w1`. Trying both
// costs two map lookups and stops a good file being refused over punctuation.
const GgufTensor *pick(const Gguf &g, std::initializer_list<const char *> names) {
    for (const char *n : names)
        if (const GgufTensor *t = g.tensor(n)) return t;
    return nullptr;
}

} // namespace

bool dspark_arch(std::string_view arch) {
    // A DSpark file is a DFlash file: the reference loads it with the DFlash
    // loader and adds the head when the tensors are there. So the arch strings
    // are DFlash's, plus `dspark` for anything that labels itself directly.
    return arch == "dspark" || dflash_arch(arch);
}

bool dspark_head_set(const Gguf &g) {
    return pick(g, {"markov_w1.weight", "markov_w1", "markov.w1.weight", "markov.w1"}) !=
           nullptr;
}

DsparkDraft::~DsparkDraft() { unload(); }

void DsparkDraft::unload() {
    if (be_) {
        void *ptrs[] = {ws_row_, ws_bias_, ws_feat_};
        for (void *p : ptrs)
            if (p) be_->release(p);
        QuantTensor *t[] = {&w1_, &w2_, &conf_};
        for (QuantTensor *q : t) {
            if (q->data && be_) be_->release(q->data);
            *q = QuantTensor{};
        }
        if (conf_bias_) be_->release(conf_bias_);
    }
    ws_row_ = ws_bias_ = ws_feat_ = nullptr;
    conf_bias_ = nullptr;
    base_.unload();
    gguf_.reset();
    be_ = nullptr;
}

bool DsparkDraft::load(Backend &be, const std::string &path, i64 chunk, i64 ctx,
                       i32 target_embd, i32 target_layer_count, i32 target_vocab,
                       std::string *err) {
    be_ = &be;
    gguf_ = std::make_unique<Gguf>();
    if (!gguf_->load(path, err)) return false;
    if (!dspark_head_set(*gguf_)) {
        if (err)
            *err = path + " has no markov_w1 tensor, so it is a plain DFlash "
                          "head set and not a DSpark one";
        return false;
    }
    // The base is the DFlash drafter: the same block, the same capture
    // contract, the same injection. It reads the file a second time; that is a
    // mapping of the header, not a copy of the weights.
    if (!base_.load(be, path, chunk, ctx, target_embd, target_layer_count,
                    target_vocab, err))
        return false;
    if (!load_head(err)) {
        base_.unload();
        return false;
    }
    const char *m = std::getenv("KRK_DSPARK_MARKOV");
    if (m) markov_on_ = m[0] != '0';
    if (const char *c = std::getenv("KRK_DSPARK_CONF"))
        conf_cut_ = static_cast<f32>(std::atof(c));
    return true;
}

bool DsparkDraft::load_head(std::string *err) {
    const i64 vocab = base_.cfg_n_vocab();
    const i64 n_embd = base_.n_embd();
    const GgufTensor *t1 =
        pick(*gguf_, {"markov_w1.weight", "markov_w1", "markov.w1.weight", "markov.w1"});
    const GgufTensor *t2 =
        pick(*gguf_, {"markov_w2.weight", "markov_w2", "markov.w2.weight", "markov.w2"});
    if (!t1 || !t2) {
        if (err)
            *err = "DSpark head set carries half a Markov head (w1 without w2)";
        return false;
    }
    // [rank, n_vocab] in ggml order: ne[0] is the rank.
    rank_ = static_cast<i32>(t1->ne[0]);
    if (rank_ <= 0 || static_cast<i64>(t1->ne[1]) != vocab) {
        if (err)
            *err = format("DSpark markov_w1 is [%lld, %lld] but this head set's "
                          "vocabulary is %lld",
                          static_cast<long long>(t1->ne[0]),
                          static_cast<long long>(t1->ne[1]),
                          static_cast<long long>(vocab));
        return false;
    }

    auto upload = [&](const GgufTensor *t, const char *name) -> QuantTensor {
        QuantTensor q;
        if (!dtype_supported(t->type)) {
            if (err) *err = std::string("DSpark '") + name + "' has an unsupported format";
            return q;
        }
        if (t->type == DType::F32) {
            f32 *host = static_cast<f32 *>(
                host_alloc(static_cast<size_t>(t->n_elements) * 4));
            dequant_row(DType::F32, t->data, host, t->n_elements);
            u16 *half = static_cast<u16 *>(
                host_alloc(static_cast<size_t>(t->n_elements) * 2));
            for (i64 i = 0; i < t->n_elements; i++)
                half[i] = fp32_to_fp16(host[i]);
            q.data = be_->alloc(static_cast<size_t>(t->n_elements) * 2);
            be_->upload(q.data, half, static_cast<size_t>(t->n_elements) * 2);
            host_free(host);
            host_free(half);
            q.type = DType::F16;
        } else {
            q.data = be_->alloc(t->n_bytes);
            be_->upload(q.data, t->data, t->n_bytes);
            q.type = t->type;
        }
        q.n_in = static_cast<i64>(t->ne[0]);
        q.n_out = static_cast<i64>(t->ne[1]);
        return q;
    };
    w1_ = upload(t1, "markov_w1");
    if (!w1_.data) return false;
    w2_ = upload(t2, "markov_w2");
    if (!w2_.data) return false;
    if (const GgufTensor *s = pick(*gguf_, {"markov_w2.scale", "markov.w2.scale"}))
        if (s->type == DType::F32 && s->n_elements >= 1)
            w2_scale_ = reinterpret_cast<const f32 *>(s->data)[0];

    // The confidence projection reads [hidden ; w1_prev], so its contraction is
    // n_embd + rank. It is optional, and so is its bias.
    // The reference's own spelling is `conf_proj.weight`
    // (llama-arch.cpp: {LLM_TENSOR_DSPARK_CONF_PROJ, "conf_proj"}), verified
    // against spiritbuun/buun-llama-cpp master on 2026-10-31. The longer
    // spellings are the pre-validation guesses and stay accepted so an older
    // repack still loads.
    if (const GgufTensor *tc = pick(*gguf_, {"conf_proj.weight",
                                             "conf_proj",
                                             "dspark_conf_proj.weight",
                                             "dspark_conf_proj",
                                             "confidence.proj.weight",
                                             "confidence.proj"})) {
        if (static_cast<i64>(tc->ne[0]) != n_embd + rank_) {
            if (err)
                *err = format("DSpark confidence projection contracts %lld "
                              "values but the head's input is [%lld ; %d] = %lld",
                              static_cast<long long>(tc->ne[0]),
                              static_cast<long long>(n_embd), rank_,
                              static_cast<long long>(n_embd + rank_));
            return false;
        }
        conf_ = upload(tc, "dspark_conf_proj");
        if (!conf_.data) return false;
        if (const GgufTensor *tb = pick(*gguf_, {"conf_proj.bias",
                                                 "dspark_conf_proj.bias",
                                                 "confidence.proj.bias"})) {
            f32 v = 0.0f;
            if (tb->type == DType::F32 && tb->n_elements >= 1)
                v = reinterpret_cast<const f32 *>(tb->data)[0];
            conf_bias_ = static_cast<f32 *>(be_->alloc(4));
            be_->upload(conf_bias_, &v, 4);
        }
    }

    const size_t as = be_->act_size();
    ws_row_ = be_->alloc(static_cast<size_t>(rank_) * as);
    ws_bias_ = be_->alloc(static_cast<size_t>(vocab) * as);
    ws_feat_ = be_->alloc(static_cast<size_t>(n_embd + rank_) * as);
    if (!ws_row_ || !ws_bias_ || !ws_feat_) {
        if (err) *err = "DSpark head could not allocate its workspaces";
        return false;
    }
    tmp_row_.assign(static_cast<size_t>(vocab), 0.0f);
    tmp_bias_.assign(static_cast<size_t>(vocab), 0.0f);
    hidden_.assign(static_cast<size_t>(base_.block_size()) *
                       static_cast<size_t>(n_embd),
                   0.0f);
    last_conf_.assign(static_cast<size_t>(base_.block_size()), 1.0f);
    markov_prev_.assign(static_cast<size_t>(base_.block_size()), -1);
    return true;
}

i32 DsparkDraft::draft_block(const QuantTensor &tok_embd, const QuantTensor &lm_head,
                             i32 n_vocab, i32 id_last, i32 n_draft, i32 pos,
                             f32 *rows_out) {
    if (!be_) return 0;
    const i32 got = base_.draft_block(tok_embd, lm_head, n_vocab, id_last, n_draft,
                                      pos, rows_out);
    last_rows_ = got;
    if (got <= 0) return got;
    markov_prev_.assign(static_cast<size_t>(got), -1);
    last_conf_.assign(static_cast<size_t>(got), 1.0f);
    if (!has_markov() || !markov_on_) return got;

    const i64 E = base_.n_embd();
    // The block's hidden rows feed the confidence head. Asking for them is a
    // download of got*n_embd floats; if the drafter cannot supply them the
    // confidence head is skipped rather than fed a substitute.
    i32 hrows = 0, hwidth = 0;
    const bool have_hidden =
        has_confidence() && base_.download_last_hidden(hidden_.data(), &hrows, &hwidth);

    i32 prev = id_last;
    for (i32 k = 0; k < got; k++) {
        // w1_prev = row(w1, prev). The backend's embed() is exactly a row
        // gather with dequantisation, which is what a vocabulary-indexed table
        // wants, so the same op serves a f16 or a quantised w1.
        be_->embed(ws_row_, w1_.data, w1_.type, n_vocab, rank_, &prev, 1);
        if (w2_scale_ != 1.0f) {
            // A quantised file's lora-style scale multiplies the INPUT of w2,
            // which is the row we just gathered.
            std::vector<f32> r(static_cast<size_t>(rank_));
            be_->download_f32(r.data(), ws_row_, rank_);
            for (i32 i = 0; i < rank_; i++)
                r[static_cast<size_t>(i)] *= w2_scale_;
            be_->upload(ws_row_, r.data(), static_cast<size_t>(rank_) * 4);
        }
        be_->gemm(ws_bias_, ws_row_, w2_.data, w2_.type, n_vocab, rank_, 1);

        f32 *row = rows_out + static_cast<size_t>(k) * n_vocab;
        be_->download_f32(tmp_bias_.data(), ws_bias_, n_vocab);
        std::memcpy(tmp_row_.data(), row, static_cast<size_t>(n_vocab) * 4);
        for (i32 v = 0; v < n_vocab; v++)
            tmp_row_[static_cast<size_t>(v)] +=
                tmp_bias_[static_cast<size_t>(v)];
        std::memcpy(row, tmp_row_.data(), static_cast<size_t>(n_vocab) * 4);

        // Greedy chain, as the reference does it: the conditioning token is the
        // argmax of the biased row, whatever the sampler later does with it.
        i32 best = 0;
        f32 bv = tmp_row_[0];
        for (i32 v = 1; v < n_vocab; v++)
            if (tmp_row_[static_cast<size_t>(v)] > bv) {
                bv = tmp_row_[static_cast<size_t>(v)];
                best = v;
            }
        markov_prev_[static_cast<size_t>(k)] = prev;

        if (have_hidden && k < hrows) {
            // conf = sigmoid(proj . [hidden_k ; w1_prev_k] + bias)
            const f32 *h = hidden_.data() + static_cast<size_t>(k) * E;
            std::vector<f32> feat(static_cast<size_t>(E + rank_));
            std::memcpy(feat.data(), h, static_cast<size_t>(E) * 4);
            std::vector<f32> w1row(static_cast<size_t>(rank_));
            be_->download_f32(w1row.data(), ws_row_, rank_);
            std::memcpy(feat.data() + E, w1row.data(),
                        static_cast<size_t>(rank_) * 4);
            // Both operands in the backend's activation format.
            const size_t as = be_->act_size();
            if (as == 2) {
                std::vector<u16> half(feat.size());
                for (size_t i = 0; i < feat.size(); i++)
                    half[i] = fp32_to_fp16(feat[i]);
                be_->upload(ws_feat_, half.data(), feat.size() * 2);
            } else {
                be_->upload(ws_feat_, feat.data(), feat.size() * 4);
            }
            be_->gemm(ws_bias_, ws_feat_, conf_.data, conf_.type, 1,
                      static_cast<i64>(feat.size()), 1);
            f32 z = 0.0f;
            be_->download_f32(&z, ws_bias_, 1);
            if (conf_bias_) {
                f32 b = 0.0f;
                be_->download_f32(&b, conf_bias_, 1);
                z += b;
            }
            last_conf_[static_cast<size_t>(k)] =
                static_cast<f32>(1.0 / (1.0 + std::exp(-static_cast<f64>(z))));
        }
        prev = best;
    }
    return got;
}

std::string DsparkDraft::describe() const {
    return format("%s + dspark head (%s, rank %d, confidence %s)",
                  base_.describe().c_str(),
                  has_markov() ? (markov_on_ ? "markov on" : "markov off")
                               : "no markov",
                  rank_, has_confidence() ? "on" : "absent");
}

} // namespace krk
