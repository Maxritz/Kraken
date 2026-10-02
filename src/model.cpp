// model.cpp — resolve GGUF metadata + tensor schema into device-resident weights.
#include "krk/model.hpp"

#include <algorithm>
#include <cmath>

namespace krk {

namespace {

std::string blk_key(const char *fmt, i32 layer) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), fmt, layer);
    return std::string(buf);
}

} // namespace

bool Model::load(Backend &be, const std::string &path, std::string *err) {
    be_ = &be;
    if (!gguf_.load(path, err)) return false;

    // ---- architecture -----------------------------------------------------
    cfg_.arch = gguf_.get_str("general.architecture", "llama");
    cfg_.name = gguf_.get_str("general.name", "model");
    const std::string a = cfg_.arch;

    auto key = [&a](const char *suffix) { return a + suffix; };

    // block_count counts every block in the file, which includes any
    // next-token-prediction (MTP) blocks. Those are extra prediction heads
    // appended to the stack, not transformer layers: they carry attention and a
    // FFN but no ssm tensors, so loading them as layers fails on the missing
    // delta-net weights. The converter records how many in
    // `<arch>.nextn_predict_layers` (an array, one entry per head).
    const i32 block_count = static_cast<i32>(gguf_.get_i64(key(".block_count"), 0));
    i32 nextn = 0;
    if (const std::vector<i32> *nl = gguf_.get_i32_array(key(".nextn_predict_layers"))) {
        if (!nl->empty()) nextn = (*nl)[0];
    } else {
        nextn = static_cast<i32>(gguf_.get_i64(key(".nextn_predict_layers"), 0));
    }
    if (nextn < 0 || nextn > block_count) nextn = 0;
    cfg_.n_layer_nextn = nextn;
    cfg_.n_layer = block_count - nextn;
    cfg_.n_embd = static_cast<i32>(gguf_.get_i64(key(".embedding_length"), 0));
    cfg_.n_ff = static_cast<i32>(gguf_.get_i64(key(".feed_forward_length"), 0));
    cfg_.n_head = static_cast<i32>(gguf_.get_i64(key(".attention.head_count"), 0));
    cfg_.n_head_kv =
        static_cast<i32>(gguf_.get_i64(key(".attention.head_count_kv"), cfg_.n_head));
    cfg_.n_ctx_train =
        static_cast<i32>(gguf_.get_i64(key(".context_length"), 2048));
    cfg_.head_dim = static_cast<i32>(gguf_.get_i64(key(".attention.key_length"), 0));
    if (cfg_.head_dim == 0 && cfg_.n_head > 0)
        cfg_.head_dim = cfg_.n_embd / cfg_.n_head;
    cfg_.rms_eps =
        static_cast<f32>(gguf_.get_f64(key(".attention.layer_norm_rms_epsilon"), 1e-5));
    cfg_.rope_base = static_cast<f32>(gguf_.get_f64(key(".rope.freq_base"), 10000.0));
    cfg_.rope_scale = static_cast<f32>(gguf_.get_f64(key(".rope.scaling.factor"), 1.0));
    {
        const std::string st = gguf_.get_str(key(".rope.scaling.type"), "");
        if (st == "none" || st.empty()) cfg_.rope_scale = 1.0f;
        // YaRN needs the frequency corrections of rope.cpp on the host; the
        // linear factor is a good first-order approximation and is what the
        // engine applies to inv_freq at load time.
    }
    cfg_.tied_embeddings = (gguf_.tensor("output.weight") == nullptr);

    // ---- Mixture-of-Experts geometry -------------------------------------
    // Qwen2-MoE / Qwen3-MoE (and the other `*moe` archs) describe routed
    // experts with these keys; a dense model simply has n_expert == 0.
    cfg_.n_expert = static_cast<i32>(gguf_.get_i64(key(".expert_count"), 0));
    cfg_.n_expert_used = static_cast<i32>(
        gguf_.get_i64(key(".expert_used_count"), cfg_.n_expert > 0 ? 1 : 0));
    cfg_.n_ff_exp = static_cast<i32>(gguf_.get_i64(key(".expert_feed_forward_length"), 0));
    cfg_.n_ff_shexp = static_cast<i32>(
        gguf_.get_i64(key(".expert_shared_feed_forward_length"), 0));
    cfg_.n_expert_shared = static_cast<i32>(gguf_.get_i64(
        key(".expert_shared_count"), cfg_.n_ff_shexp > 0 ? 1 : 0));
    // is_moe is only confirmed once a layer actually exposes expert tensors, so
    // a checkpoint that merely declares expert_count cannot misroute the FFN.
    if (cfg_.n_expert > 0) {
        if (cfg_.n_expert_used <= 0) cfg_.n_expert_used = cfg_.n_expert;
        if (cfg_.n_expert_used > cfg_.n_expert) cfg_.n_expert_used = cfg_.n_expert;
        if (cfg_.n_ff_exp <= 0) cfg_.n_ff_exp = cfg_.n_ff > 0 ? cfg_.n_ff : 0;
    }

    // Gemma-style models scale the embedding by sqrt(n_embd); the schema here
    // only needs to know the norm epsilon and head geometry.
    if (cfg_.arch == "gemma" || cfg_.arch == "gemma2" || cfg_.arch == "gemma3") {
        cfg_.rms_eps = static_cast<f32>(
            gguf_.get_f64(key(".attention.layer_norm_rms_epsilon"), 1e-6));
        cfg_.rope_base = 10000.0f;
    }

    // ---- gated delta net geometry (Qwen3.5 / Qwen3-Next) -----------------
    // These archs interleave linear-attention ("recurrent") layers with real
    // attention every `full_attention_interval` layers. The ssm.* keys carry
    // the recurrent block's shape; the tensor schema is validated per layer
    // further down, where the split is actually known.
    cfg_.recurrent = (a == "qwen35" || a == "qwen35moe");
    if (cfg_.recurrent) {
        cfg_.full_attention_interval = static_cast<i32>(
            gguf_.get_i64(key(".full_attention_interval"), 4));
        if (cfg_.full_attention_interval <= 0) cfg_.full_attention_interval = 4;
        cfg_.ssm_d_state = static_cast<i32>(gguf_.get_i64(key(".ssm.state_size"), 0));
        cfg_.ssm_dt_rank = static_cast<i32>(gguf_.get_i64(key(".ssm.time_step_rank"), 0));
        cfg_.ssm_n_group = static_cast<i32>(gguf_.get_i64(key(".ssm.group_count"), 0));
        cfg_.ssm_d_conv = static_cast<i32>(gguf_.get_i64(key(".ssm.conv_kernel"), 4));
        const i32 ssm_inner =
            static_cast<i32>(gguf_.get_i64(key(".ssm.inner_size"), 0));
        if (cfg_.ssm_d_state <= 0 || cfg_.ssm_dt_rank <= 0 || cfg_.ssm_n_group <= 0) {
            if (err)
                *err = "arch '" + a +
                       "' is a gated delta net but its ssm.* geometry is missing";
            return false;
        }
        cfg_.ssm_key_dim = cfg_.ssm_n_group * cfg_.ssm_d_state;
        cfg_.ssm_value_dim = ssm_inner > 0 ? ssm_inner
                                           : cfg_.ssm_dt_rank * cfg_.ssm_d_state;
        cfg_.ssm_conv_dim = cfg_.ssm_key_dim * 2 + cfg_.ssm_value_dim;
        if (cfg_.ssm_value_dim != cfg_.ssm_dt_rank * cfg_.ssm_d_state) {
            if (err)
                *err = "arch '" + a + "': ssm.inner_size (" +
                       std::to_string(cfg_.ssm_value_dim) +
                       ") disagrees with time_step_rank * state_size (" +
                       std::to_string(cfg_.ssm_dt_rank * cfg_.ssm_d_state) + ")";
            return false;
        }
        if (cfg_.ssm_d_conv <= 0 || cfg_.ssm_d_conv > 16) {
            if (err)
                *err = "arch '" + a + "': ssm.conv_kernel " +
                       std::to_string(cfg_.ssm_d_conv) + " is out of range";
            return false;
        }
        // A recurrent model needs a backend that can carry its state; refusing
        // here is much better than decoding garbage on one that cannot.
        if (!be.gdn_supported()) {
            if (err)
                *err = "arch '" + a +
                       "' needs the gated delta net kernels, which this backend "
                       "does not implement";
            return false;
        }
        // Partial rotary: the converter records how many dims are rotated.
        const i32 rope_dim =
            static_cast<i32>(gguf_.get_i64(key(".rope.dimension_count"), 0));
        cfg_.rope_dim = rope_dim > 0 ? std::min(rope_dim, cfg_.head_dim) : 0;
        if (cfg_.rope_dim > 0 && cfg_.head_dim > 0)
            cfg_.rope_frac = static_cast<f32>(cfg_.rope_dim) /
                             static_cast<f32>(cfg_.head_dim);
        if (const std::vector<i32> *sec = gguf_.get_i32_array(key(".rope.dimension_sections"))) {
            for (size_t i = 0; i < sec->size() && i < 4; i++)
                cfg_.rope_sections[i] = (*sec)[i];
        }
    }

    if (cfg_.n_layer <= 0 || cfg_.n_embd <= 0 || cfg_.n_head <= 0) {
        if (err) *err = "GGUF metadata is missing core geometry (arch='" + a + "')";
        return false;
    }
    if (cfg_.n_embd % cfg_.n_head != 0) {
        if (err) *err = "n_embd is not divisible by head count";
        return false;
    }
    if (cfg_.n_head % cfg_.n_head_kv != 0) {
        if (err) *err = "head count is not a multiple of the KV head count";
        return false;
    }
    // An all-MoE model legitimately omits `feed_forward_length`; fall back to
    // the first expert workspace that is actually described.
    if (cfg_.n_ff <= 0) cfg_.n_ff = cfg_.n_ff_shexp > 0 ? cfg_.n_ff_shexp : cfg_.n_ff_exp;
    if (cfg_.n_ff <= 0) {
        if (err) *err = "feed_forward_length is missing";
        return false;
    }

    const i64 q_dim = static_cast<i64>(cfg_.n_head) * cfg_.head_dim;
    const i64 kv_dim = static_cast<i64>(cfg_.n_head_kv) * cfg_.head_dim;

    // ---- helpers ----------------------------------------------------------
    auto require_tensor = [&](const std::string &name, std::string *e) -> const GgufTensor * {
        const GgufTensor *t = gguf_.tensor(name);
        if (!t) {
            if (e) *e = "required tensor '" + name + "' is missing";
            return nullptr;
        }
        if (!dtype_supported(t->type)) {
            if (e)
                *e = "tensor '" + name + "' uses unsupported format " +
                     dtype_name(t->type) + " (type id " +
                     std::to_string(t->type_id) + ")";
            return nullptr;
        }
        if (!dtype_row_aligned(t->type, static_cast<i64>(t->ne[0]))) {
            if (e) *e = "tensor '" + name + "' row length is not block aligned";
            return nullptr;
        }
        return t;
    };

    // Uploads a tensor verbatim (quantized blocks stay packed).
    auto upload_raw = [&](const GgufTensor *t, DType as) -> QuantTensor {
        QuantTensor q;
        q.data = be.alloc(t->n_bytes);
        be.upload(q.data, t->data, t->n_bytes);
        q.type = as;
        q.n_in = static_cast<i64>(t->ne[0]);
        q.n_out = t->n_dims >= 2 ? static_cast<i64>(t->ne[1]) : 1;
        if (t->n_dims > 2) q.n_out = t->n_elements / std::max<i64>(1, q.n_in);
        weight_bytes_ += static_cast<i64>(t->n_bytes);
        return q;
    };

    // The short-convolution kernel is stored the OTHER way round from the op
    // contract. GGUF's ne[0] is the fastest axis, and for ssm_conv1d that axis
    // is the tap, so the file holds kern[c * ksize + j] — each channel's taps
    // contiguous — while conv1d_silu indexes kern[j * conv_dim + c], the newest
    // tap last. (ggml's ssm_conv reads c[i0 + i1 * d_conv], the same
    // channel-major order, which is how llama.cpp feeds it unmodified.)
    // Transposing once here costs 4 x 8192 values at load time and leaves the
    // op — and its per-layer tap cache — on a single stride.
    auto upload_conv1d = [&](const std::string &name) -> QuantTensor {
        QuantTensor q;
        const GgufTensor *t = gguf_.tensor(name);
        if (!t || t->n_dims < 2) return q;
        const i64 ksize = static_cast<i64>(t->ne[0]);
        const i64 chan = static_cast<i64>(t->ne[1]);
        const i64 n = ksize * chan;
        if (ksize <= 0 || chan <= 0 || n != t->n_elements) return q;
        f32 *host = static_cast<f32 *>(host_alloc(static_cast<size_t>(n) * 4));
        dequant_row(t->type, t->data, host, n);
        std::vector<f32> tap(static_cast<size_t>(n));
        for (i64 c = 0; c < chan; c++)
            for (i64 j = 0; j < ksize; j++)
                tap[static_cast<size_t>(j * chan + c)] =
                    host[static_cast<size_t>(c * ksize + j)];
        host_free(host);
        q.data = be.alloc(static_cast<size_t>(n) * 4);
        be.upload(q.data, tap.data(), static_cast<size_t>(n) * 4);
        q.type = DType::F32;
        q.n_in = ksize;
        q.n_out = chan;
        weight_bytes_ += n * 4;
        return q;
    };

    // Uploads a 1-D f32 tensor (norm weights, biases, per-head norms).
    auto upload_f32 = [&](const std::string &name) -> f32 * {
        const GgufTensor *t = gguf_.tensor(name);
        if (!t) return nullptr;
        const i64 n = t->n_elements;
        f32 *host = static_cast<f32 *>(host_alloc(static_cast<size_t>(n) * 4));
        dequant_row(t->type, t->data, host, n);
        f32 *dev = static_cast<f32 *>(be.alloc(static_cast<size_t>(n) * 4));
        be.upload(dev, host, static_cast<size_t>(n) * 4);
        host_free(host);
        return dev;
    };

    // Uploads a 2-D matmul weight; F32 is downcast to F16 so every GEMM lands
    // on one kernel family.
    auto upload_weight = [&](const std::string &name, bool optional,
                             std::string *e) -> QuantTensor {
        const GgufTensor *t = gguf_.tensor(name);
        if (!t) {
            if (!optional && e) *e = "required weight '" + name + "' is missing";
            return {};
        }
        if (!dtype_supported(t->type)) {
            if (e)
                *e = "weight '" + name + "' uses unsupported format " +
                     dtype_name(t->type) + " (type id " + std::to_string(t->type_id) +
                     ")";
            return {};
        }
        if (t->type == DType::F32) {
            const i64 n_in = static_cast<i64>(t->ne[0]);
            const i64 n_out = t->n_elements / std::max<i64>(1, n_in);
            // already row-aligned, so a single flat conversion is valid
            f32 *tmp = static_cast<f32 *>(
                host_alloc(static_cast<size_t>(t->n_elements) * 4));
            dequant_row(DType::F32, t->data, tmp, t->n_elements);
            u16 *half = static_cast<u16 *>(
                host_alloc(static_cast<size_t>(t->n_elements) * 2));
            for (i64 i = 0; i < t->n_elements; i++)
                half[i] = fp32_to_fp16(tmp[i]);
            QuantTensor q;
            q.data = be.alloc(static_cast<size_t>(t->n_elements) * 2);
            be.upload(q.data, half, static_cast<size_t>(t->n_elements) * 2);
            host_free(tmp);
            host_free(half);
            q.type = DType::F16;
            q.n_in = n_in;
            q.n_out = n_out;
            weight_bytes_ += t->n_elements * 2;
            return q;
        }
        return upload_raw(t, t->type);
    };

    // ---- token embedding + head ------------------------------------------
    {
        std::string e;
        const GgufTensor *te = require_tensor("token_embd.weight", &e);
        if (!te) {
            if (err) *err = e;
            return false;
        }
        tok_embd_ = upload_raw(te, te->type);
        cfg_.n_vocab = static_cast<i32>(te->ne[1]);
    }
    if (cfg_.tied_embeddings) {
        out_head_ = tok_embd_;
    } else {
        std::string e;
        const GgufTensor *oh = require_tensor("output.weight", &e);
        if (!oh) {
            if (err) *err = e;
            return false;
        }
        out_head_ = upload_raw(oh, oh->type);
    }
    out_norm_ = upload_f32("output_norm.weight");
    if (!out_norm_) {
        if (err) *err = "output_norm.weight is missing";
        return false;
    }

    // ---- per-layer weights ------------------------------------------------
    layers_.resize(static_cast<size_t>(cfg_.n_layer));
    for (i32 l = 0; l < cfg_.n_layer; l++) {
        LayerWeights &L = layers_[static_cast<size_t>(l)];
        std::string e;

        auto need = [&](const std::string &n) -> QuantTensor {
            QuantTensor q = upload_weight(n, false, &e);
            if (!q.present() && err && e.empty()) e = "missing weight '" + n + "'";
            return q;
        };

        L.attn_norm = upload_f32(blk_key("blk.%d.attn_norm.weight", l));
        L.ffn_norm = upload_f32(blk_key("blk.%d.ffn_norm.weight", l));
        // Qwen3.5 renamed the post-attention norm; when the classic key is
        // absent this is the same role (norm before the FFN), so accept either
        // rather than rejecting a checkpoint that spells it differently.
        if (!L.ffn_norm)
            L.ffn_norm = upload_f32(blk_key("blk.%d.post_attention_norm.weight", l));
        if (!L.attn_norm || !L.ffn_norm) {
            if (err) *err = "layer " + std::to_string(l) + " is missing its norm weights";
            return false;
        }

        // ---- which kind of layer is this? ---------------------------------
        if (cfg_.recurrent) {
            // An explicit layer_types array wins. Otherwise the converter's
            // default is "every full_attention_interval'th layer attends",
            // counted from 1 — layers 3, 7, 11... for an interval of 4. That
            // makes the *last* layer attend whenever the depth is a multiple of
            // the interval, which is why MTP blocks must be excluded above
            // rather than special-cased here.
            L.gdn = true;
            if (const std::vector<i32> *rec =
                    gguf_.get_i32_array(key(".attention.recurrent_layers"))) {
                if (l < static_cast<i32>(rec->size()))
                    L.gdn = (*rec)[static_cast<size_t>(l)] != 0;
            } else {
                L.gdn = ((l + 1) % cfg_.full_attention_interval) != 0;
            }
        }

        if (L.gdn) {
            L.wqkv = need(blk_key("blk.%d.attn_qkv.weight", l));
            L.wqkv_gate = need(blk_key("blk.%d.attn_gate.weight", l));
            L.ssm_conv1d = upload_conv1d(blk_key("blk.%d.ssm_conv1d.weight", l));
            L.ssm_out = need(blk_key("blk.%d.ssm_out.weight", l));
            L.ssm_dt = upload_f32(blk_key("blk.%d.ssm_dt.bias", l));
            L.ssm_a = upload_f32(blk_key("blk.%d.ssm_a", l));
            L.ssm_alpha = need(blk_key("blk.%d.ssm_alpha.weight", l));
            L.ssm_beta = need(blk_key("blk.%d.ssm_beta.weight", l));
            L.ssm_norm = upload_f32(blk_key("blk.%d.ssm_norm.weight", l));
        } else {
            L.wq = need(blk_key("blk.%d.attn_q.weight", l));
            L.wk = need(blk_key("blk.%d.attn_k.weight", l));
            L.wv = need(blk_key("blk.%d.attn_v.weight", l));
            L.wo = need(blk_key("blk.%d.attn_output.weight", l));
        }

        // A layer is MoE when it carries routed-expert tensors. This is decided
        // per layer: hybrid stacks (dense early layers, MoE later ones) load
        // correctly without extra metadata.
        const std::string exps_gate = blk_key("blk.%d.ffn_gate_exps.weight", l);
        L.moe = gguf_.tensor(exps_gate) != nullptr;
        if (!L.moe) {
            L.wgate = need(blk_key("blk.%d.ffn_gate.weight", l));
            L.wup = need(blk_key("blk.%d.ffn_up.weight", l));
            L.wdown = need(blk_key("blk.%d.ffn_down.weight", l));
        } else {
            // The router is small and always resident. The routed experts are
            // only *described* here — nothing is uploaded, which is what keeps
            // a 235B-A22B model from needing 235B of VRAM at load time.
            L.router = need(blk_key("blk.%d.ffn_gate_inp.weight", l));
            const GgufTensor *g = gguf_.tensor(exps_gate);
            const GgufTensor *up = gguf_.tensor(blk_key("blk.%d.ffn_up_exps.weight", l));
            const GgufTensor *dn =
                gguf_.tensor(blk_key("blk.%d.ffn_down_exps.weight", l));
            if (!g || !up || !dn) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " has a partial routed-expert tensor set";
                return false;
            }
            for (const GgufTensor *t : {g, up, dn}) {
                if (!dtype_supported(t->type)) {
                    if (err)
                        *err = "expert tensor '" + t->name +
                               "' uses unsupported format " + dtype_name(t->type);
                    return false;
                }
            }
            L.experts.gate = g;
            L.experts.up = up;
            L.experts.down = dn;
            L.experts.n_expert = static_cast<i32>(g->ne[2]);
            L.experts.n_embd = static_cast<i64>(g->ne[0]);
            L.experts.n_ff_exp = static_cast<i64>(g->ne[1]);

            // Shared expert (Qwen2-MoE): one always-on expert, small enough to
            // keep resident.
            L.shexp_gate =
                upload_weight(blk_key("blk.%d.ffn_gate_shexp.weight", l), true, &e);
            L.shexp_up =
                upload_weight(blk_key("blk.%d.ffn_up_shexp.weight", l), true, &e);
            L.shexp_down =
                upload_weight(blk_key("blk.%d.ffn_down_shexp.weight", l), true, &e);
            // Qwen3.5 gates the shared expert with a per-token scalar built
            // from a 1-D row; the Qwen2/Qwen3 schema has no such vector.
            L.shexp_inp_gate =
                upload_weight(blk_key("blk.%d.ffn_gate_inp_shexp.weight", l), true, &e);
        }
        if (!e.empty()) {
            if (err) *err = e;
            return false;
        }

        // geometric sanity: the tensor schema must match the metadata
        if (L.gdn) {
            // The recurrent block has no head geometry at all: one fused
            // [q|k|v] projection and one output projection, both sized by the
            // ssm.* keys.
            if (L.wqkv.n_out != cfg_.ssm_conv_dim || L.wqkv.n_in != cfg_.n_embd) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " attn_qkv disagrees with the declared ssm geometry";
                return false;
            }
            if (L.wqkv_gate.n_out != cfg_.ssm_value_dim) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " attn_gate disagrees with ssm.inner_size";
                return false;
            }
            if (L.ssm_out.n_in != cfg_.ssm_value_dim || L.ssm_out.n_out != cfg_.n_embd) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " ssm_out disagrees with ssm.inner_size/embedding_length";
                return false;
            }
            if (L.ssm_conv1d.n_in != cfg_.ssm_d_conv ||
                L.ssm_conv1d.n_out != cfg_.ssm_conv_dim) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " ssm_conv1d disagrees with ssm.conv_kernel/inner_size";
                return false;
            }
            if (!L.ssm_dt || !L.ssm_a || !L.ssm_norm || !L.ssm_alpha.present() ||
                !L.ssm_beta.present()) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " is missing its ssm state parameters";
                return false;
            }
        } else {
            // Qwen3.5's attending layers pack a per-head output gate into the
            // query projection, so its row count is twice the head width.
            const i64 expect_q = (a == "qwen35" || a == "qwen35moe") ? 2 * q_dim : q_dim;
            if (L.wq.n_out != expect_q || L.wk.n_out != kv_dim ||
                L.wv.n_out != kv_dim || L.wo.n_in != q_dim) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " attention tensors disagree with the declared head geometry";
                return false;
            }
        }
        if (!L.moe && (L.wgate.n_out != cfg_.n_ff || L.wdown.n_in != cfg_.n_ff)) {
            if (err)
                *err = "layer " + std::to_string(l) + " FFN tensors disagree with n_ff";
            return false;
        }
        if (L.moe) {
            if (L.router.n_out != cfg_.n_expert || L.router.n_in != cfg_.n_embd) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " router disagrees with expert_count/embedding_length";
                return false;
            }
            if (L.experts.n_expert != cfg_.n_expert ||
                L.experts.n_ff_exp != cfg_.n_ff_exp) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " routed-expert tensors disagree with the metadata";
                return false;
            }
            if (L.shexp_gate.present() && L.shexp_gate.n_out != cfg_.n_ff_shexp) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " shared-expert width disagrees with the metadata";
                return false;
            }
            cfg_.is_moe = true;
        }

        L.q_bias = upload_f32(blk_key("blk.%d.attn_q.bias", l));
        L.k_bias = upload_f32(blk_key("blk.%d.attn_k.bias", l));
        L.v_bias = upload_f32(blk_key("blk.%d.attn_v.bias", l));
        cfg_.has_bias = cfg_.has_bias || (L.q_bias != nullptr);

        L.q_norm = upload_f32(blk_key("blk.%d.attn_q_norm.weight", l));
        L.k_norm = upload_f32(blk_key("blk.%d.attn_k_norm.weight", l));
        if (L.q_norm || L.k_norm) cfg_.qk_norm = true;
    }

    // ---- rotary inverse frequencies (host) --------------------------------
    // Partial rotary: Qwen3.5 rotates only the first `rope_dim` dims of each
    // head and leaves the tail untouched, so the table is rope_dim/2 long and
    // the engine passes the fraction to the rotation op.
    const i64 rope_dim = cfg_.rope_dim > 0 ? cfg_.rope_dim : cfg_.head_dim;
    const i64 half = rope_dim / 2;
    inv_freq_.resize(static_cast<size_t>(half));
    for (i64 i = 0; i < half; i++) {
        const f32 exponent = static_cast<f32>(2 * i) / static_cast<f32>(rope_dim);
        inv_freq_[static_cast<size_t>(i)] = std::pow(cfg_.rope_base, -exponent);
    }

    KRK_INFO("loaded %s (%s): %d layers, %d embd, %d/%d heads, hd=%d, ff=%d, vocab=%d",
             cfg_.name.c_str(), cfg_.arch.c_str(), cfg_.n_layer, cfg_.n_embd,
             cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim, cfg_.n_ff, cfg_.n_vocab);
    if (cfg_.recurrent) {
        KRK_INFO("gated delta net: %d recurrent layers of %d (every %d attends), "
                 "ssm %d key x %d value heads, state %d, conv kernel %d, rope %d/%d",
                 recurrent_layers(), cfg_.n_layer, cfg_.full_attention_interval,
                 cfg_.ssm_n_group, cfg_.ssm_dt_rank, cfg_.ssm_d_state,
                 cfg_.ssm_d_conv, rope_dim, cfg_.head_dim);
    }
    return true;
}

void Model::unload() {
    if (!be_) return;
    // The expert cache owns backend allocations too; drop them first.
    experts_.clear();
    for (auto &L : layers_) {
        if (L.attn_norm) be_->release(L.attn_norm);
        if (L.ffn_norm) be_->release(L.ffn_norm);
        if (L.q_bias) be_->release(L.q_bias);
        if (L.k_bias) be_->release(L.k_bias);
        if (L.v_bias) be_->release(L.v_bias);
        if (L.q_norm) be_->release(L.q_norm);
        if (L.k_norm) be_->release(L.k_norm);
        for (f32 *f : {L.ssm_dt, L.ssm_a, L.ssm_norm})
            if (f) be_->release(f);
        for (QuantTensor *q : {&L.wq, &L.wk, &L.wv, &L.wo, &L.wgate, &L.wup,
                               &L.wdown, &L.router, &L.shexp_gate, &L.shexp_up,
                               &L.shexp_down, &L.shexp_inp_gate, &L.wqkv,
                               &L.wqkv_gate, &L.ssm_conv1d, &L.ssm_out,
                               &L.ssm_alpha, &L.ssm_beta})
            if (q->present()) be_->release(q->data);
    }
    layers_.clear();
    if (tok_embd_.present()) be_->release(tok_embd_.data);
    if (!cfg_.tied_embeddings && out_head_.present()) be_->release(out_head_.data);
    if (out_norm_) be_->release(out_norm_);
    tok_embd_ = {};
    out_head_ = {};
    out_norm_ = nullptr;
    be_ = nullptr;
}

} // namespace krk
