// ============================================================================
//  dflash.cpp — see include/krk/dflash.hpp.
//
//  Three forward passes, none of which looks like the target's:
//
//   * load(): read the head set. No token embedding and no output head, because
//     the drafter borrows both from the target at run time; every tensor here is
//     either the fusion projection, a norm weight, or one of the attention/FFN
//     matrices of a small dense stack.
//   * commit(): the only place the drafter touches the target's activations. The
//     target copies a handful of residual streams into the capture buffer; this
//     fuses them, projects K/V out of the result and writes those into a KV
//     cache of the drafter's own. Nothing is emitted.
//   * draft_block(): tokens in, logits out, with the injected cache as context.
//
//  The fusion step runs on the host, deliberately. Its arithmetic is a per-row
//  RMS norm over the feature width followed by a per-aux scale, which is not a
//  shape any existing kernel has (the weight varies down the rows of a buffer
//  the backend otherwise treats as uniform); doing it on the host costs one
//  round trip per *chunk* — a few hundred kilobytes per decode step — and buys
//  the absence of a new kernel in both backends. If this ever shows up in a
//  profile, the fix is a kernel, not a rearrangement.
// ============================================================================
#include "krk/dflash.hpp"

#include "krk/arch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace krk {

namespace {

inline char *act_at(void *base, size_t act_size, i64 elem) {
    return static_cast<char *>(base) + static_cast<size_t>(elem) * act_size;
}
inline const char *act_at(const void *base, size_t act_size, i64 elem) {
    return static_cast<const char *>(base) + static_cast<size_t>(elem) * act_size;
}

// Host-side RMS norm with a scale: out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i].
// The fusion layer is the only caller, and it needs main memory because the
// features arrive there to be concatenated anyway.
void rms_scale(f32 *out, const f32 *x, const f32 *w, i64 n, f32 eps) {
    f64 ss = 0.0;
    for (i64 i = 0; i < n; i++) ss += static_cast<f64>(x[i]) * x[i];
    const f32 inv = static_cast<f32>(1.0 / std::sqrt(ss / static_cast<f64>(n) + eps));
    for (i64 i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

// Upload f32 host data into an activation buffer. Activations live in the
// backend's own element type (f16 on the GPU), so a raw byte copy of f32 values
// is read back as half-precision noise -- denormals and infinities out of
// ordinary feature magnitudes, which is exactly what the fusion input is. Every
// other host->activation path in the engine converts here too; this one is the
// first that had to, because it is the first place a host-side computation feeds
// a GEMM directly.
void upload_act(Backend &be, void *dst, const f32 *src, i64 n) {
    if (be.act_type() == DType::F16) {
        std::vector<u16> tmp(static_cast<size_t>(n));
        for (i64 i = 0; i < n; i++) tmp[static_cast<size_t>(i)] = fp32_to_fp16(src[i]);
        be.upload(dst, tmp.data(), static_cast<size_t>(n) * 2);
    } else {
        be.upload(dst, src, static_cast<size_t>(n) * 4);
    }
}

// Plain RoPE table: n_dims/2 frequencies, base^(-2i/n_dims). The same shape the
// model loader builds for a checkpoint with no scaling keys, which is what both
// DFlash spellings are: the file declares `rope.freq_base` and nothing else.
void build_inv_freq(std::vector<f32> &tab, i32 n_dims, f32 base) {
    const i64 h = std::max<i32>(1, n_dims / 2);
    tab.assign(static_cast<size_t>(h), 0.0f);
    for (i64 i = 0; i < h; i++) {
        const f32 exponent = static_cast<f32>(2 * i) / static_cast<f32>(n_dims);
        tab[static_cast<size_t>(i)] = std::pow(base, -exponent);
    }
}

} // namespace

bool dflash_arch(std::string_view arch) {
    // `dflash` is what poolside's converter emits today; `qwen35-dflash-draft`
    // is the older spelling of the same object (see the header). Both are head
    // sets, and neither is anything else.
    return arch == "dflash" || arch == "qwen35-dflash-draft";
}

// The two spellings, in one place: each call tries the current name and then the
// older one, so the rest of the loader can read as if there were one file format.
static const GgufTensor *either(const Gguf &g, const char *current,
                               const char *legacy) {
    if (const GgufTensor *t = g.tensor(current)) return t;
    return legacy ? g.tensor(legacy) : nullptr;
}

DflashDraft::~DflashDraft() { unload(); }

void DflashDraft::unload() {
    if (be_) {
        be_->sync();
        for (void **p : {&cap_, &feat_, &enc_out_, &ws_x_, &ws_xn_, &ws_x2_,
                         &ws_q_, &ws_k_, &ws_v_, &ws_attn_, &ws_gate_, &ws_up_,
                         &ws_agate_, &ws_logits_, &kcache_, &vcache_}) {
            if (*p) {
                be_->release(*p);
                *p = nullptr;
            }
        }
        for (LayerWeights &L : layers_) {
            if (L.attn_norm) be_->release(L.attn_norm);
            if (L.ffn_norm) be_->release(L.ffn_norm);
            if (L.q_norm) be_->release(L.q_norm);
            if (L.k_norm) be_->release(L.k_norm);
            for (QuantTensor *q : {&L.wq, &L.wk, &L.wv, &L.wo, &L.wgate, &L.wup,
                                   &L.wdown, &L.wattn_gate})
                if (q->data) be_->release(q->data);
        }
        if (fc_.data) be_->release(fc_.data);
        if (enc_out_norm_) be_->release(enc_out_norm_);
        if (out_norm_) be_->release(out_norm_);
        if (aux_norm_) be_->release(aux_norm_);
    }
    layers_.clear();
    fc_ = QuantTensor{};
    enc_out_norm_ = nullptr;
    out_norm_ = nullptr;
    aux_norm_ = nullptr;
    aux_norm_host_.clear();
    inv_freq_.clear();
    feat_host_.clear();
    tmp_host_.clear();
    block_tok_.clear();
    gguf_.reset();
    be_ = nullptr;
}

bool DflashDraft::load(Backend &be, const std::string &path, i64 chunk, i64 ctx,
                       i32 target_embd, i32 target_layer_count, std::string *err) {
    be_ = &be;
    chunk_ = std::max<i64>(1, chunk);
    gguf_ = std::make_unique<Gguf>();
    if (!gguf_->load(path, err)) return false;

    cfg_.arch = gguf_->get_str("general.architecture", "");
    cfg_.name = gguf_->get_str("general.name", "dflash");
    if (!dflash_arch(cfg_.arch)) {
        if (err)
            *err = "not a DFlash head set: general.architecture is '" + cfg_.arch +
                   "' (expected dflash or qwen35-dflash-draft)";
        return false;
    }

    auto key = [&](const char *suffix) { return cfg_.arch + suffix; };
    // The dflash-specific keys travel one level down in the older spelling
    // (`qwen35-dflash-draft.dflash.block_size`) and sit directly under the arch
    // in the current one (`dflash.block_size`), so each read tries both.
    auto dkey = [&](const char *suffix) -> std::string {
        return cfg_.arch + ".dflash" + suffix;
    };
    auto has = [&](const std::string &k) { return gguf_->find(k) != nullptr; };

    cfg_.n_layer = static_cast<i32>(gguf_->get_i64(key(".block_count"), 0));
    cfg_.n_embd = static_cast<i32>(gguf_->get_i64(key(".embedding_length"), 0));
    cfg_.n_ff = static_cast<i32>(gguf_->get_i64(key(".feed_forward_length"), 0));
    cfg_.n_head = static_cast<i32>(gguf_->get_i64(key(".attention.head_count"), 0));
    cfg_.n_head_kv = static_cast<i32>(
        gguf_->get_i64(key(".attention.head_count_kv"), cfg_.n_head));
    cfg_.head_dim = static_cast<i32>(gguf_->get_i64(key(".attention.key_length"), 0));
    // No `n_embd % n_head` check here. The drafter's heads are wider than its
    // embedding divided by them (72 heads of 128 dims over 3072, i.e. 9216 query
    // columns): the head count is a property of the attention, not a way to cut
    // the residual stream up.
    if (cfg_.head_dim <= 0 && cfg_.n_head > 0)
        cfg_.head_dim = cfg_.n_embd / cfg_.n_head;
    cfg_.rms_eps = static_cast<f32>(
        gguf_->get_f64(key(".attention.layer_norm_rms_epsilon"), 1e-6));
    cfg_.n_vocab = static_cast<i32>(gguf_->get_i64(key(".vocab_size"), 0));
    // The drafter's own context ceiling. It is larger than any target's in both
    // files here, so the target's context is what actually bounds the cache.
    cfg_.n_ctx_train = static_cast<i32>(gguf_->get_i64(key(".context_length"), 0));

    if (cfg_.n_layer <= 0 || cfg_.n_embd <= 0 || cfg_.n_head <= 0 ||
        cfg_.head_dim <= 0 || cfg_.n_ff <= 0) {
        if (err) *err = "DFlash head set is missing core geometry";
        return false;
    }
    if (cfg_.n_head % cfg_.n_head_kv != 0) {
        if (err) *err = "DFlash head count is not a multiple of the KV head count";
        return false;
    }

    // ---- the capture list ------------------------------------------------
    // `target_layers` in the current spelling already holds *extract ids*: the
    // converter adds 1 to the checkpoint's own 0-based layer ids (the output of
    // layer i is the input of block i+1), and an id equal to the target's layer
    // count is the pre-final-norm state. The older spelling writes the raw
    // checkpoint ids, so the +1 is applied here.
    if (const std::vector<i32> *tl = gguf_->get_i32_array(key(".target_layers"))) {
        target_layers_ = *tl;
    } else if (const std::vector<i32> *tl =
                   gguf_->get_i32_array(dkey(".target_layers"))) {
        target_layers_ = *tl;
    } else if (const std::vector<i32> *tl =
                   gguf_->get_i32_array(dkey(".target_layer_ids"))) {
        // `target_layer_ids` is the checkpoint's own numbering, where an id means
        // "the output of layer i", which is the input of block i+1. That shift
        // is the single most load-bearing guess in this file, so it is a switch:
        // KRK_DFLASH_LAYER_OFFSET=0 reads them as extract ids directly.
        i32 off = 1;
        if (const char *env = std::getenv("KRK_DFLASH_LAYER_OFFSET"))
            off = std::atoi(env);
        target_layers_.clear();
        for (i32 v : *tl) target_layers_.push_back(v + off);
    }
    if (target_layers_.empty()) {
        if (err) *err = "DFlash head set declares no target_layers / target_layer_ids";
        return false;
    }
    // Deliberately NOT sorted. The aux-norm weights are stored one per capture
    // slot, so slot k pairs with id k; reordering the ids would pair a feature
    // with the wrong weight. Both files here happen to list them ascending,
    // which is exactly why the sort was invisible until it was not.
    for (i32 id : target_layers_) {
        if (id < 0 || id > target_layer_count) {
            if (err)
                *err = format("DFlash wants the input of layer %d, but the target "
                              "has %d layers and this drafter captures at most the "
                              "pre-final-norm state (%d)",
                              id, target_layer_count, target_layer_count);
            return false;
        }
    }
    if (has(dkey(".n_target_layers"))) {
        const i32 declared = static_cast<i32>(gguf_->get_i64(dkey(".n_target_layers"), 0));
        if (declared != static_cast<i32>(target_layers_.size())) {
            if (err)
                *err = format("DFlash declares %d target layers but lists %d ids",
                              declared, static_cast<i32>(target_layers_.size()));
            return false;
        }
    }

    block_size_ = static_cast<i32>(gguf_->get_i64(
        has(key(".block_size")) ? key(".block_size") : dkey(".block_size"), 16));
    if (block_size_ < 2) block_size_ = 16;

    // The mask token is a real vocabulary entry the drafter was trained to treat
    // as "no information"; the current spelling also records it in the tokenizer
    // block, which is where llama.cpp reads it.
    mask_id_ = -1;
    if (has(dkey(".mask_token_id")))
        mask_id_ = static_cast<i32>(gguf_->get_i64(dkey(".mask_token_id"), -1));
    if (mask_id_ < 0)
        mask_id_ = static_cast<i32>(
            gguf_->get_i64("tokenizer.ggml.mask_token_id", -1));
    if (mask_id_ < 0)
        mask_id_ = static_cast<i32>(
            gguf_->get_i64("tokenizer.ggml.unknown_token_id", 0));

    zero_mask_embed_ = gguf_->get_bool(
        has(dkey(".zero_mask_embedding")) ? dkey(".zero_mask_embedding")
                                          : key(".zero_mask_embedding"), false);

    // `decoder_arch = "laguna"` is the current spelling of two behavioural
    // statements at once: the injection input is normed by the layer's
    // attn_norm, and the noise block is causal. The older files state neither
    // (their `context_kv_layer_norm` even says 0) but carry the tensors of that
    // same decoder block, and both the converter that repacks them for llama.cpp
    // and the engine that produced them derive the contract from those tensors
    // rather than from the flag. So: a gated (or aux-normed) third-party head
    // set is a Laguna decoder, whatever its metadata claims.
    const std::string decoder_arch = gguf_->get_str(key(".decoder_arch"), "");
    const bool third_party = cfg_.arch != "dflash";
    const bool gate_tensors = gguf_->tensor("blk.0.attn_gate.weight") != nullptr;
    const bool aux_tensors =
        gguf_->tensor("enc.aux_norm.weight") != nullptr ||
        gguf_->tensor("dflash.aux_hidden_norm.0.weight") != nullptr;
    decoder_laguna_ = decoder_arch == "laguna" ||
                      (third_party && (gate_tensors || aux_tensors)) ||
                      gguf_->get_bool(key(".context_kv_layer_norm"), false);
    kv_norm_ = decoder_laguna_;
    causal_ = decoder_laguna_;
    if (const char *env = std::getenv("KRK_DFLASH_CAUSAL"))
        causal_ = env[0] == '1';
    if (const char *env = std::getenv("KRK_DFLASH_KVNORM"))
        kv_norm_ = env[0] == '1';

    // A single rotor: DFlash has one rope, deliberately, and both the injection
    // and the block use it for every layer. The file declares no scaling keys,
    // so this is plain rope — the interpolated table is not reachable here.
    cfg_.rope_base = static_cast<f32>(gguf_->get_f64(key(".rope.freq_base"), 10000.0));
    i32 n_rot = static_cast<i32>(gguf_->get_i64(key(".rope.dimension_count"), 0));
    if (n_rot <= 0 || n_rot > cfg_.head_dim) n_rot = cfg_.head_dim;
    cfg_.rope_frac = static_cast<f32>(n_rot) / static_cast<f32>(cfg_.head_dim);
    cfg_.rope_scale = 1.0f;
    cfg_.rope_neox = arch_spec(cfg_.arch).rope_neox;
    cfg_.attn_gate = arch_spec(cfg_.arch).attn_gate;
    cfg_.qk_norm = true;
    build_inv_freq(inv_freq_, n_rot, cfg_.rope_base);
    if (const char *env = std::getenv("KRK_DFLASH_BLOCK_POS"))
        block_base_ = std::atoi(env);

    // Interleaved sliding window: the older file carries one (all layers, 512),
    // the current one does not. The pattern is stored as a bool per layer.
    layer_window_.assign(static_cast<size_t>(cfg_.n_layer), 0);
    {
        const i64 win = gguf_->get_i64(key(".attention.sliding_window"), 0);
        if (win > 0) {
            cfg_.swa_window = static_cast<i32>(win);
            bool any = false;
            if (const std::vector<i32> *pat =
                    gguf_->get_i32_array(key(".attention.sliding_window_pattern"))) {
                for (i32 l = 0; l < cfg_.n_layer; l++) {
                    const bool on = l < static_cast<i32>(pat->size()) &&
                                    (*pat)[static_cast<size_t>(l)] != 0;
                    if (on) {
                        layer_window_[static_cast<size_t>(l)] = cfg_.swa_window;
                        any = true;
                    }
                }
            } else {
                for (i32 l = 0; l < cfg_.n_layer; l++)
                    layer_window_[static_cast<size_t>(l)] = cfg_.swa_window;
                any = true;
            }
            if (!any) cfg_.swa_window = 0;
        }
    }

    // The drafter's own runtime masks the noise block per layer: sliding-window
    // layers are causal so a mask row cannot read the rows after it, full layers
    // are block-visible. llama.cpp's Laguna decoder instead flips one global
    // switch for every layer, so a native `dflash` file follows that and a
    // third-party head set follows the engine it was exported for.
    layer_causal_.assign(static_cast<size_t>(cfg_.n_layer), causal_ ? 1 : 0);
    if (third_party) {
        for (i32 l = 0; l < cfg_.n_layer; l++)
            layer_causal_[static_cast<size_t>(l)] =
                layer_window_[static_cast<size_t>(l)] > 0 ? 1 : 0;
    }

    // ---- capacity --------------------------------------------------------
    // Counted for the contract line above: per-layer masks outvote the global
    // flag, and a mis-derived one is invisible in the tensor checks.
    n_causal_layers_ = 0;
    for (i32 c : layer_causal_) n_causal_layers_ += c ? 1 : 0;
    kv_cap_ = std::min<i64>(std::max<i64>(ctx, 1), std::max<i64>(cfg_.n_ctx_train, 1));
    if (kv_cap_ <= 0) kv_cap_ = 2048;
    cfg_.n_ctx_train = static_cast<i32>(kv_cap_);
    if (!load_weights(err)) {
        unload();
        return false;
    }

    // ---- workspaces ------------------------------------------------------
    const size_t as = be.act_size();
    const i64 q_dim = static_cast<i64>(cfg_.n_head) * cfg_.head_dim;
    const i64 kv_dim = static_cast<i64>(cfg_.n_head_kv) * cfg_.head_dim;
    const i64 C = chunk_;
    n_embd_enc_ = static_cast<i32>(target_layers_.size()) * target_embd;
    if (static_cast<i64>(fc_.n_in) != n_embd_enc_) {
        if (err)
            *err = format("DFlash fusion projection is %lld wide, but %d captured "
                          "layers x the target's %d embedding is %d",
                          static_cast<long long>(fc_.n_in),
                          static_cast<i32>(target_layers_.size()), target_embd,
                          n_embd_enc_);
        unload();
        return false;
    }

    cap_ = be.alloc(static_cast<size_t>(capture_stride() * n_aux()) * as);
    feat_ = be.alloc(static_cast<size_t>(C * n_embd_enc_) * as);
    enc_out_ = be.alloc(static_cast<size_t>(C * cfg_.n_embd) * as);
    ws_x_ = be.alloc(static_cast<size_t>(C * cfg_.n_embd) * as);
    ws_xn_ = be.alloc(static_cast<size_t>(C * cfg_.n_embd) * as);
    ws_x2_ = be.alloc(static_cast<size_t>(C * cfg_.n_embd) * as);
    ws_q_ = be.alloc(static_cast<size_t>(C * q_dim) * as);
    ws_k_ = be.alloc(static_cast<size_t>(C * kv_dim) * as);
    ws_v_ = be.alloc(static_cast<size_t>(C * kv_dim) * as);
    ws_attn_ = be.alloc(static_cast<size_t>(C * q_dim) * as);
    ws_gate_ = be.alloc(static_cast<size_t>(C * cfg_.n_ff) * as);
    ws_up_ = be.alloc(static_cast<size_t>(C * cfg_.n_ff) * as);
    if (cfg_.attn_gate)
        ws_agate_ = be.alloc(static_cast<size_t>(C * q_dim) * as);
    // One row per candidate, so the whole block's readback is a single transfer
    // instead of one sync per row. block_size-1 is the most the block can ask
    // for, and it is the same ceiling the driver clamps the window to.
    ws_logits_ = be.alloc(static_cast<size_t>(block_size_ - 1) *
                          static_cast<size_t>(cfg_.n_vocab) * as);
    kcache_ = be.alloc(static_cast<size_t>(cfg_.n_layer * kv_cap_ * kv_dim) * as);
    vcache_ = be.alloc(static_cast<size_t>(cfg_.n_layer * kv_cap_ * kv_dim) * as);
    be.fill0(kcache_, static_cast<size_t>(cfg_.n_layer * kv_cap_ * kv_dim) * as);
    be.fill0(vcache_, static_cast<size_t>(cfg_.n_layer * kv_cap_ * kv_dim) * as);

    feat_host_.resize(static_cast<size_t>(C) * static_cast<size_t>(n_embd_enc_));
    tmp_host_.resize(static_cast<size_t>(C) * static_cast<size_t>(cfg_.n_embd));
    block_tok_.resize(static_cast<size_t>(block_size_));
    be.sync();

    KRK_INFO("dflash drafter: %s (%s) %d layers, %d embd, %d/%d heads, ff %d, "
             "block %d, mask %d, ctx %lld, capture [",
             cfg_.name.c_str(), cfg_.arch.c_str(), cfg_.n_layer, cfg_.n_embd,
             cfg_.n_head, cfg_.n_head_kv, cfg_.n_ff, block_size_, mask_id_,
             static_cast<long long>(kv_cap_));
    for (size_t i = 0; i < target_layers_.size(); i++)
        KRK_INFO("dflash capture:   %d%s", target_layers_[i],
                 i + 1 < target_layers_.size() ? "," : "]");
    KRK_INFO("dflash contract: %s injection, %s block (%d/%d layers causal), "
             "aux norms %s, rope %d/%d base %.0f, decoder from %s%s",
             kv_norm_ ? "normed" : "raw", causal_ ? "causal" : "non-causal",
             n_causal_layers_, static_cast<i32>(cfg_.n_layer),
             aux_norm_ ? "stacked" : "absent", static_cast<i32>(n_rot),
             cfg_.head_dim, static_cast<f64>(cfg_.rope_base),
             decoder_arch.empty() ? "tensors" : "metadata",
             zero_mask_embed_ ? ", mask embedding zeroed" : "");
    return true;
}

bool DflashDraft::load_weights(std::string *err) {
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
    // Same rule as the model loader: an F32 matrix is downcast to F16 so every
    // GEMM lands on one kernel family, and a quantized block stays packed.
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
            for (i64 i = 0; i < t->n_elements; i++) half[i] = fp32_to_fp16(tmp[i]);
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
        return q;
    };

    // ---- the fusion projection and the two encoder/decoder norms ----------
    {
        const GgufTensor *t = either(*gguf_, "fc.weight", "dflash.fc.weight");
        if (!t) {
            if (err) *err = "DFlash head set has no fusion projection (fc.weight)";
            return false;
        }
        fc_ = upload_w(t, t->name, 0, 0);
        if (!fc_.present()) return false;
        // The projection reads the concatenated feature blocks and writes the
        // drafter's width; anything else is a file for a different target.
        if (fc_.n_out != cfg_.n_embd) {
            if (err)
                *err = format("DFlash fusion projection writes %lld columns, "
                              "expected the drafter's %d",
                              static_cast<long long>(fc_.n_out), cfg_.n_embd);
            return false;
        }
    }
    {
        const GgufTensor *t = either(*gguf_, "enc.output_norm.weight",
                                     "dflash.hidden_norm.weight");
        if (!t) {
            if (err) *err = "DFlash head set has no encoder output norm";
            return false;
        }
        enc_out_norm_ = upload_f32_w(t);
        if (t->n_elements != cfg_.n_embd) {
            if (err) *err = "DFlash encoder output norm is not the drafter's width";
            return false;
        }
    }
    // ---- the per-aux feature norms ---------------------------------------
    // Stacked in the current spelling ([n_embd, n_aux] in one tensor), separate
    // in the older one (`dflash.aux_hidden_norm.{k}.weight`). Both mean the same
    // buffer afterwards: aux block k is scaled by the k-th n_embd-long slice.
    {
        const GgufTensor *t = gguf_->tensor("enc.aux_norm.weight");
        if (t) {
            if (t->n_elements !=
                static_cast<i64>(cfg_.n_embd) * static_cast<i64>(n_aux())) {
                if (err)
                    *err = "DFlash aux norm does not have one n_embd row per "
                           "captured layer";
                return false;
            }
            aux_norm_host_.assign(static_cast<size_t>(t->n_elements), 0.0f);
            dequant_row(t->type, t->data, aux_norm_host_.data(), t->n_elements);
        } else {
            aux_norm_host_.assign(
                static_cast<size_t>(cfg_.n_embd) * static_cast<size_t>(n_aux()), 0.0f);
            for (i32 a = 0; a < n_aux(); a++) {
                const GgufTensor *n = gguf_->tensor(
                    format("dflash.aux_hidden_norm.%d.weight", a));
                if (!n) {
                    if (err)
                        *err = format("DFlash head set has neither enc.aux_norm nor "
                                      "aux_hidden_norm.%d",
                                      a);
                    return false;
                }
                if (n->n_elements != cfg_.n_embd) {
                    if (err) *err = "DFlash aux hidden norm is not the drafter's width";
                    return false;
                }
                dequant_row(n->type, n->data,
                            aux_norm_host_.data() + static_cast<size_t>(a) * cfg_.n_embd,
                            cfg_.n_embd);
            }
        }
        // On the device only because the multiply is easier to reason about with
        // the two sides in the same layout; the host copy is what commit() reads.
        aux_norm_ = static_cast<f32 *>(
            be_->alloc(aux_norm_host_.size() * sizeof(f32)));
        be_->upload(aux_norm_, aux_norm_host_.data(),
                    aux_norm_host_.size() * sizeof(f32));
    }
    // ---- decoder final norm ----------------------------------------------
    // The block's last projection is the *target's* lm_head, but the norm that
    // feeds it belongs to the drafter: the reference reads `output_norm.weight`
    // off the draft file and only borrows the head itself.
    {
        const GgufTensor *t = gguf_->tensor("output_norm.weight");
        if (!t) {
            if (err) *err = "DFlash head set has no output_norm.weight";
            return false;
        }
        if (t->n_elements != cfg_.n_embd) {
            if (err) *err = "DFlash output norm is not the drafter's width";
            return false;
        }
        out_norm_ = upload_f32_w(t);
    }

    // ---- per-layer -------------------------------------------------------
    layers_.resize(static_cast<size_t>(cfg_.n_layer));
    for (i32 l = 0; l < cfg_.n_layer; l++) {
        LayerWeights &L = layers_[static_cast<size_t>(l)];
        auto bn = [&](const char *fmt) { return format(fmt, l); };

        L.attn_norm = upload_f32_w(gguf_->tensor(bn("blk.%d.attn_norm.weight")));
        L.ffn_norm = upload_f32_w(gguf_->tensor(bn("blk.%d.ffn_norm.weight")));
        if (!L.attn_norm || !L.ffn_norm) {
            if (err) *err = format("DFlash layer %d is missing its norm weights", l);
            return false;
        }
        const std::string wq = format("blk.%d.attn_q.weight", l);
        const std::string wk = format("blk.%d.attn_k.weight", l);
        const std::string wv = format("blk.%d.attn_v.weight", l);
        const std::string wo = format("blk.%d.attn_output.weight", l);
        L.wq = upload_w(gguf_->tensor(wq), wq, q_dim, cfg_.n_embd);
        L.wk = upload_w(gguf_->tensor(wk), wk, kv_dim, cfg_.n_embd);
        L.wv = upload_w(gguf_->tensor(wv), wv, kv_dim, cfg_.n_embd);
        L.wo = upload_w(gguf_->tensor(wo), wo, cfg_.n_embd, q_dim);
        if (!L.wq.present() || !L.wk.present() || !L.wv.present() || !L.wo.present())
            return false;
        L.q_norm = upload_f32_w(gguf_->tensor(bn("blk.%d.attn_q_norm.weight")));
        L.k_norm = upload_f32_w(gguf_->tensor(bn("blk.%d.attn_k_norm.weight")));
        {
            const GgufTensor *g = gguf_->tensor(bn("blk.%d.attn_gate.weight"));
            if (g) {
                // Per-head (n_head rows) or per-element (q_dim rows), told apart
                // by the stored width, exactly as the laguna target arch does.
                const i64 rows = g->n_elements / std::max<i64>(1, static_cast<i64>(g->ne[0]));
                L.wattn_gate = upload_w(g, bn("blk.%d.attn_gate.weight"), rows,
                                        cfg_.n_embd);
                if (!L.wattn_gate.present()) return false;
            }
        }
        const std::string wg = format("blk.%d.ffn_gate.weight", l);
        const std::string wu = format("blk.%d.ffn_up.weight", l);
        const std::string wd = format("blk.%d.ffn_down.weight", l);
        L.wgate = upload_w(gguf_->tensor(wg), wg, cfg_.n_ff, cfg_.n_embd);
        L.wup = upload_w(gguf_->tensor(wu), wu, cfg_.n_ff, cfg_.n_embd);
        L.wdown = upload_w(gguf_->tensor(wd), wd, cfg_.n_embd, cfg_.n_ff);
        if (!L.wgate.present() || !L.wup.present() || !L.wdown.present())
            return false;
    }
    return true;
}

void *DflashDraft::capture_slot(i32 aux) const {
    return act_at(cap_, be_->act_size(), static_cast<i64>(aux) * capture_stride());
}

bool DflashDraft::commit(i32 pos0, i32 n) {
    if (!be_ || n <= 0) return false;
    if (n > chunk_) n = static_cast<i32>(chunk_);
    const size_t as = be_->act_size();
    const i32 n_aux = this->n_aux();
    const i32 E = cfg_.n_embd;
    const i64 enc = n_embd_enc_;

    // ---- fuse: per-aux RMS norm, then concatenate in aux order -------------
    // The layout fc reads is token-major with the aux blocks inside it
    // (`feat[i * n_embd_enc + a * n_embd]`), which is the same order the
    // reference driver memcpy's into its feature buffer.
    i64 n_bad = 0;
    for (i32 a = 0; a < n_aux; a++) {
        be_->download_f32(tmp_host_.data(),
                          act_at(cap_, as, static_cast<i64>(a) * capture_stride()),
                          static_cast<i64>(n) * E);
        const f32 *w = aux_norm_host_.data() + static_cast<size_t>(a) * E;
        for (i32 i = 0; i < n; i++) {
            f32 *dst = feat_host_.data() + static_cast<size_t>(i) * enc +
                       static_cast<size_t>(a) * E;
            rms_scale(dst, tmp_host_.data() + static_cast<size_t>(i) * E, w, E,
                      cfg_.rms_eps);
        }
    }
    // A target's residual stream carries massive activations (attention sinks
    // reach |x| ~ 1e6 in the pre-final-norm state), and f16 tops out at 65504.
    // One poisoned row would NaN the drafter's whole cache, so a non-finite
    // value is clamped rather than propagated -- the same trade the reference
    // driver makes, for the same reason.
    if (be_->act_type() == DType::F16) {
        for (f32 &v : feat_host_) {
            if (!std::isfinite(v)) {
                v = v != v ? 0.0f : (v > 0.0f ? 65504.0f : -65504.0f);
                n_bad++;
            }
        }
        if (n_bad > 0) {
            static bool warned = false;
            if (!warned) {
                KRK_WARN("dflash: clamped %lld non-finite feature values (f16 "
                         "range); draft quality may drop on the affected rows",
                         static_cast<long long>(n_bad));
                warned = true;
            }
        }
    }
    upload_act(*be_, feat_, feat_host_.data(),
               static_cast<i64>(n) * enc);

    // ---- fc, then the encoder's output norm -------------------------------
    // `enc_out_` is what the reference calls inp_g: the drafter's context K/V
    // source, not an embedding.
    be_->gemm(ws_x2_, feat_, fc_.data, fc_.type, E, static_cast<i64>(enc), n);
    be_->rmsnorm(enc_out_, ws_x2_, enc_out_norm_, n, E, cfg_.rms_eps);
    {   // KRK_DFLASH_KV_ZERO=1 empties the injection. The block's rows must
        // move when the context they read is emptied; if they do not, the block
        // is not reading the context at all, whatever the cache says.
        static const bool kz = std::getenv("KRK_DFLASH_KV_ZERO") != nullptr;
        if (kz) {
            be_->fill0(enc_out_, static_cast<i64>(n) * E);
            KRK_INFO("dflash: injection emptied (KRK_DFLASH_KV_ZERO)");
        }
    }
    // KRK_DFLASH_DEBUG=1: the fused context, before any projection reads it. Its
    // scale says whether the capture arrived at all: a row of zeros means the
    // feature blocks never made it into the buffer, and a row of 1e6 means the
    // target's massive activations survived the norm instead of being scaled out.
    if (std::getenv("KRK_DFLASH_DEBUG")) {
        be_->download_f32(tmp_host_.data(), enc_out_, static_cast<i64>(n) * E);
        f32 lo = 1e30f, hi = -1e30f;
        f64 ss = 0.0;
        for (i64 i = 0; i < static_cast<i64>(n) * E; i++) {
            lo = std::min(lo, tmp_host_[static_cast<size_t>(i)]);
            hi = std::max(hi, tmp_host_[static_cast<size_t>(i)]);
            ss += static_cast<f64>(tmp_host_[static_cast<size_t>(i)]) *
                  tmp_host_[static_cast<size_t>(i)];
        }
        std::fprintf(stderr,
                     "[dfdbg] commit pos=%d n=%d enc_out min %.6g max %.6g rms "
                     "%.6g | row0 moved %.6g\n",
                     pos0, n, static_cast<f64>(lo), static_cast<f64>(hi),
                     std::sqrt(ss / (static_cast<f64>(n) * E)),
                     [&] {
                         // feat rms is a constant by construction (each aux
                         // block is RMS-normed before it is scaled by its own
                         // weight), so it cannot say whether the context moved.
                         // Row 0 against the previous commit can.
                         static std::vector<f32> prev;
                         const f32 *cur = tmp_host_.data();
                         f64 d = -1.0;
                         if (prev.size() == static_cast<size_t>(E)) {
                             d = 0.0;
                             for (i64 i = 0; i < E; i++) {
                                 const f64 dv = static_cast<f64>(cur[i]) -
                                                static_cast<f64>(prev[static_cast<size_t>(i)]);
                                 d += dv * dv;
                             }
                             d = std::sqrt(d / static_cast<f64>(E));
                         }
                         prev.assign(cur, cur + static_cast<size_t>(E));
                         return d;
                     }());
        std::fflush(stderr);
    }

    // ---- inject K/V for every layer ---------------------------------------
    const i64 hd = cfg_.head_dim;
    const i64 nh = cfg_.n_head;
    const i64 nkv = cfg_.n_head_kv;
    const i64 kv_dim = nkv * hd;
    AttnDesc d;
    d.n_head = nh;
    d.n_kv = nkv;
    d.hd = hd;
    d.n_tok = n;
    d.pos0 = pos0;
    d.layer_stride = kv_cap_ * kv_dim;
    d.pos_stride = kv_dim;
    for (i32 l = 0; l < cfg_.n_layer; l++) {
        const LayerWeights &L = layers_[static_cast<size_t>(l)];
        const void *kv_inp = enc_out_;
        if (kv_norm_) {
            be_->rmsnorm(ws_xn_, enc_out_, L.attn_norm, n, E, cfg_.rms_eps);
            kv_inp = ws_xn_;
        }
        be_->gemm(ws_k_, kv_inp, L.wk.data, L.wk.type, kv_dim, E, n);
        be_->gemm(ws_v_, kv_inp, L.wv.data, L.wv.type, kv_dim, E, n);
        // Only K is carried through the head norm and the rotation; the query
        // half of rope() is handed the scratch buffer, which is rotated and
        // discarded. Passing a null there would have to be a special case in two
        // backends for an op that costs a fraction of the projections.
        // The drafter is the qwen3-dflash family, whose QK-norm is head_dim
        // wide and per head, so the OLMoE whole-row convention never applies.
        be_->qk_norm(nullptr, ws_k_, nullptr, L.k_norm, 0, nkv, hd, n,
                     cfg_.rms_eps, false);
        be_->rope(ws_q_, ws_k_, nh, nkv, hd, n, pos0, inv_freq_.data(),
                  cfg_.rope_scale, cfg_.rope_frac, cfg_.rope_neox);
        d.layer = l;
        be_->kv_append(kcache_, vcache_, ws_k_, ws_v_, d);
    }
    committed_ += n;
    return true;
}

i32 DflashDraft::draft_block(const QuantTensor &tok_embd, const QuantTensor &lm_head,
                             i32 n_vocab, i32 id_last, i32 n_draft, i32 pos,
                             f32 *rows_out) {
    if (!be_ || n_draft <= 0 || n_vocab <= 0) return 0;
    if (n_draft > block_size_ - 1) n_draft = block_size_ - 1;
    const bool dfdbg_kv = std::getenv("KRK_DFLASH_DEBUG") != nullptr;
    const i32 nt = n_draft + 1; // id_last + n_draft masks

    // The block is placed so that mask row k lands on position pos+k: id_last
    // goes back to the position it already occupies, and the masks take the
    // positions the candidates are for. That is the layout the reference driver
    // builds ([id_last, <mask> * n_draft] at positions pos .. pos+n_draft) read
    // from the next free position instead of the last filled one, and it is what
    // makes block row k the candidate for pos+k rather than for pos+k+1.
    const i32 blk_pos = pos + block_base_;
    if (blk_pos < 0) return 0;

    const size_t as = be_->act_size();
    const i64 E = cfg_.n_embd;
    const i64 hd = cfg_.head_dim;
    const i64 nh = cfg_.n_head;
    const i64 nkv = cfg_.n_head_kv;
    const i64 kv_dim = nkv * hd;
    const i64 q_dim = nh * hd;

    for (i32 i = 0; i < nt; i++)
        block_tok_[static_cast<size_t>(i)] = i == 0 ? id_last : mask_id_;
    be_->embed(ws_x_, tok_embd.data, tok_embd.type, n_vocab, E, block_tok_.data(), nt);
    // Some drafters are trained with the mask rows carrying no token information
    // at all; the flag says so and the embedding is cleared for those rows.
    if (zero_mask_embed_ && nt > 1)
        be_->fill0(act_at(ws_x_, as, E),
                   static_cast<size_t>(nt - 1) * static_cast<size_t>(E) * as);

    AttnDesc d;
    d.n_head = nh;
    d.n_kv = nkv;
    d.hd = hd;
    d.n_tok = nt;
    d.pos0 = blk_pos;
    d.layer_stride = kv_cap_ * kv_dim;
    d.pos_stride = kv_dim;
    d.scale = static_cast<f32>(1.0 / std::sqrt(static_cast<f64>(hd)));
    d.causal = causal_;

    for (i32 l = 0; l < cfg_.n_layer; l++) {
        const LayerWeights &L = layers_[static_cast<size_t>(l)];
        be_->rmsnorm(ws_xn_, ws_x_, L.attn_norm, nt, E, cfg_.rms_eps);
        if (L.wq.type == L.wk.type && L.wk.type == L.wv.type) {
            void *qkv[3] = {ws_q_, ws_k_, ws_v_};
            const void *wqkv[3] = {L.wq.data, L.wk.data, L.wv.data};
            const i64 nqkv[3] = {q_dim, kv_dim, kv_dim};
            be_->gemm_group(qkv, ws_xn_, wqkv, L.wq.type, nqkv, E, 3, nt);
        } else {
            be_->gemm(ws_q_, ws_xn_, L.wq.data, L.wq.type, q_dim, E, nt);
            be_->gemm(ws_k_, ws_xn_, L.wk.data, L.wk.type, kv_dim, E, nt);
            be_->gemm(ws_v_, ws_xn_, L.wv.data, L.wv.type, kv_dim, E, nt);
        }
        be_->qk_norm(ws_q_, ws_k_, L.q_norm, L.k_norm, nh, nkv, hd, nt,
                     cfg_.rms_eps, false);
        be_->rope(ws_q_, ws_k_, nh, nkv, hd, nt, blk_pos, inv_freq_.data(),
                  cfg_.rope_scale, cfg_.rope_frac, cfg_.rope_neox);
        d.layer = l;
        d.window = layer_window_[static_cast<size_t>(l)];
        d.causal = layer_causal_[static_cast<size_t>(l)] != 0;
        be_->kv_append(kcache_, vcache_, ws_k_, ws_v_, d);
        be_->attention(ws_attn_, ws_q_, kcache_, vcache_, d);
        if (dfdbg_kv && (l == 0 || l == cfg_.n_layer - 1)) {
            // Row 1 of the block: what attention actually produced, and the K
            // row the injection wrote at the last committed position. If the
            // block ignores its context, either the first is ~0 or the second
            // is, and which one it is settles where the context is lost.
            const i64 koff = static_cast<i64>(d.layer) * d.layer_stride +
                             static_cast<i64>(pos - 1) * d.pos_stride;
            const i64 kmax = static_cast<i64>(cfg_.n_layer) * d.layer_stride;
            f64 ar = 0.0, kr = 0.0;
            be_->download_f32(tmp_host_.data(),
                              act_at(ws_attn_, as, static_cast<i64>(1) * q_dim),
                              q_dim);
            for (i64 i = 0; i < q_dim; i++)
                ar += static_cast<f64>(tmp_host_[static_cast<size_t>(i)]) *
                      static_cast<f64>(tmp_host_[static_cast<size_t>(i)]);
            if (pos >= 1 && koff + kv_dim <= kmax) {
                be_->download_f32(tmp_host_.data(), act_at(kcache_, as, koff),
                                  kv_dim);
                for (i64 i = 0; i < kv_dim; i++)
                    kr += static_cast<f64>(tmp_host_[static_cast<size_t>(i)]) *
                          static_cast<f64>(tmp_host_[static_cast<size_t>(i)]);
            }
            std::fprintf(stderr,
                         "[dfkv] l=%d row1 attn rms %.6g | injected K[pos-1] rms "
                         "%.6g | committed %lld blk_pos %d nt %d",
                         l, std::sqrt(ar / static_cast<f64>(q_dim)),
                         std::sqrt(kr / static_cast<f64>(kv_dim)),
                         static_cast<long long>(committed_), blk_pos, nt);
            std::fputc(10, stderr);
        }

        if (cfg_.attn_gate && L.wattn_gate.present()) {
            // Softplus output gate off the same hidden state q/k/v read -- the
            // laguna drafter's own gate, per-head or per-element by width.
            const i64 g_out = static_cast<i64>(L.wattn_gate.n_out);
            be_->gemm(ws_agate_, ws_xn_, L.wattn_gate.data, L.wattn_gate.type,
                      g_out, E, nt);
            be_->softplus_act(ws_agate_, static_cast<i64>(nt) * g_out);
            if (g_out == nh)
                be_->mul_head_broadcast(ws_attn_, ws_agate_, nt, nh, hd);
            else
                be_->mul_act(ws_attn_, ws_agate_, static_cast<i64>(nt) * q_dim);
        }
        be_->gemm(ws_x2_, ws_attn_, L.wo.data, L.wo.type, E, q_dim, nt);
        be_->add_inplace(ws_x_, ws_x2_, static_cast<i64>(nt) * E);

        be_->rmsnorm(ws_xn_, ws_x_, L.ffn_norm, nt, E, cfg_.rms_eps);
        if (L.wgate.type == L.wup.type) {
            void *gu[2] = {ws_gate_, ws_up_};
            const void *wgu[2] = {L.wgate.data, L.wup.data};
            const i64 ngu[2] = {cfg_.n_ff, cfg_.n_ff};
            be_->gemm_group(gu, ws_xn_, wgu, L.wgate.type, ngu, E, 2, nt);
        } else {
            be_->gemm(ws_gate_, ws_xn_, L.wgate.data, L.wgate.type, cfg_.n_ff, E, nt);
            be_->gemm(ws_up_, ws_xn_, L.wup.data, L.wup.type, cfg_.n_ff, E, nt);
        }
        be_->silu_mul(ws_gate_, ws_gate_, ws_up_, static_cast<i64>(nt) * cfg_.n_ff);
        be_->gemm(ws_x2_, ws_gate_, L.wdown.data, L.wdown.type, E, cfg_.n_ff, nt);
        be_->add_inplace(ws_x_, ws_x2_, static_cast<i64>(nt) * E);
    }

    // Mask rows only: row 0 carries id_last, whose token is already known, so
    // its logits are not a draft. Row k (1-based) is the candidate for pos+k-1.
    for (i32 r = 1; r < nt; r++) {
        const void *row = act_at(ws_x_, as, static_cast<i64>(r) * E);
        be_->rmsnorm(ws_xn_, row, out_norm_, 1, E, cfg_.rms_eps);
        be_->gemm(act_at(ws_logits_, as, static_cast<i64>(r - 1) * n_vocab), ws_xn_,
                  lm_head.data, lm_head.type, n_vocab, E, 1);
    }
    be_->download_f32(rows_out, ws_logits_,
                      static_cast<i64>(n_draft) * n_vocab);
    blocks_++;
    // KRK_DFLASH_DEBUG=1: print each row's top few candidates with their
    // log-probs, so a drafter that is merely displaced by a position can be told
    // from one whose features never reached it. The distinction is the whole
    // diagnosis: a displaced row still puts the right token first among its
    // neighbours, a starved one puts the same few tokens everywhere.
    if (std::getenv("KRK_DFLASH_DEBUG")) {
        for (i32 r = 0; r < n_draft; r++) {
            const f32 *row = rows_out + static_cast<size_t>(r) * n_vocab;
            f32 best[3] = {-1e30f, -1e30f, -1e30f};
            i32 bid[3] = {-1, -1, -1};
            for (i32 v = 0; v < n_vocab; v++) {
                for (int c = 0; c < 3; c++) {
                    if (row[v] > best[c]) {
                        for (int d = 2; d > c; d--) {
                            best[d] = best[d - 1];
                            bid[d] = bid[d - 1];
                        }
                        best[c] = row[v];
                        bid[c] = v;
                        break;
                    }
                }
            }
            f64 mean = 0.0;
            for (i32 v = 0; v < n_vocab; v++) mean += row[v];
            mean /= n_vocab;
            std::fprintf(stderr,
                         "[dfdbg] blk=%lld pos=%d row=%d -> %d(%.6g) %d(%.6g) "
                         "%d(%.6g) | mean %.6g\n",
                         static_cast<long long>(blocks_), blk_pos, r, bid[0],
                         static_cast<f64>(best[0]), bid[1], static_cast<f64>(best[1]),
                         bid[2], static_cast<f64>(best[2]), mean);
        }
        std::fflush(stderr);
    }
    return n_draft;
}

} // namespace krk
