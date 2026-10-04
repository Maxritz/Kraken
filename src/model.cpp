// model.cpp — resolve GGUF metadata + tensor schema into device-resident weights.
#include "krk/model.hpp"
#include "krk/arch.hpp"
#include "krk/verdict.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>

namespace krk {

namespace {

std::string blk_key(const char *fmt, i32 layer) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), fmt, layer);
    return std::string(buf);
}

} // namespace

// Load-phase timeline (KRK_PHASE=1). The engine reports one "load N ms" line,
// which cannot say whether the time went to the mapping, the metadata, the
// embedding, the per-layer loop, or the transfers -- and the answer decides
// what is worth optimising. See docs/KNOWLEDGE-MODELS.md §6.
struct LoadPhase {
    const char *on = std::getenv("KRK_PHASE");
    f64 t0 = now();
    f64 last = t0;
    static f64 now() {
        return static_cast<f64>(std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count()) / 1000.0;
    }
    void mark(const char *what) {
        if (!on) return;
        const f64 t = now();
        std::fprintf(stderr, "[load ] %-24s %8.1f ms  (total %8.1f ms)\n", what,
                     t - last, t - t0);
        std::fflush(stderr);
        last = t;
    }
};

bool Model::load(Backend &be, const std::string &path, std::string *err) {
    LoadPhase lp;
    be_ = &be;
    if (!gguf_.load(path, err)) return false;
    lp.mark("gguf map + metadata");

    // Large weights are pulled in with read() rather than DMA'd out of the cold
    // mapping: 6.2 GB/s single-threaded and 16.3 with four handles, against
    // 3.87 for a cold mapping, because the mapping's cost is one soft page fault
    // per 4 KiB (measured 1208.8 ms vs 109.1 ms for the 8B). The backend owns
    // the pipeline and overlaps the reads with the copies; it falls back to the
    // direct copy if pinned memory cannot be had, so this is an optimisation,
    // not a requirement. See docs/KNOWLEDGE-MODELS.md.
    be.set_weight_pull(path, gguf_.file().data(), gguf_.file().size(),
                       gguf_.data_section_off());

    // ---- architecture -----------------------------------------------------
    cfg_.arch = gguf_.get_str("general.architecture", "llama");
    cfg_.name = gguf_.get_str("general.name", "model");
    const std::string a = cfg_.arch;

    // What this file is, and whether this build can run it: the arch table, the
    // quantization formats, and the tensors that name unimplemented pieces, all
    // answered from the metadata before any weight is uploaded (verdict.hpp).
    // An unrecognized arch is not fatal — most llama-schema variants load and
    // run — but it is reported, because the alternative is a model that decodes
    // nonsense while the log says nothing.
    const ModelVerdict verdict = assess_model(gguf_);
    const ArchSpec &aspec = *verdict.spec;
    for (const std::string &n : verdict.notes)
        KRK_WARN("architecture '%s': %s (known: %s)", a.c_str(), n.c_str(),
                 arch_known_names().c_str());
    if (!verdict.runnable) {
        if (err) {
            std::string msg = "cannot run this file ('" +
                              (verdict.arch.empty() ? std::string("no arch") : verdict.arch) +
                              "')";
            for (const std::string &b : verdict.blockers) msg += ". " + b;
            *err = msg;
        }
        return false;
    }

    auto key = [&a](const char *suffix) { return a + suffix; };

    // block_count counts every block in the file, which includes any
    // next-token-prediction (MTP) blocks. Those are extra prediction heads
    // appended to the stack, not transformer layers: they carry attention and a
    // FFN but no ssm tensors, so loading them as layers fails on the missing
    // delta-net weights. How many there are is `nextn_predict_layers`, read by
    // the verdict (verdict.hpp) so the loader and the report on a file cannot
    // disagree about it.
    const i32 block_count = static_cast<i32>(gguf_.get_i64(key(".block_count"), 0));
    const i32 nextn = verdict.mtp_blocks;
    cfg_.n_layer_nextn = nextn;
    cfg_.n_layer = block_count - nextn;
    cfg_.n_embd = static_cast<i32>(gguf_.get_i64(key(".embedding_length"), 0));
    cfg_.n_ff = static_cast<i32>(gguf_.get_i64(key(".feed_forward_length"), 0));
    cfg_.n_head = static_cast<i32>(gguf_.get_i64(key(".attention.head_count"), 0));
    // A hybrid arch stores the head count per layer (laguna is 48 heads on the
    // full-attention layers and 72 on the windowed ones). Keep the whole vector
    // and let `n_head` be its maximum, which is the width the shared query
    // workspace has to hold; the layer's own width is n_head_at().
    if (const std::vector<i32> *hc =
            gguf_.get_i32_array(key(".attention.head_count"))) {
        cfg_.n_head_layer = *hc;
        for (i32 v : *hc)
            if (v > cfg_.n_head) cfg_.n_head = v;
    }
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
    cfg_.logit_softcap =
        static_cast<f32>(gguf_.get_f64(key(".final_logit_softcapping"), 0.0));
    {
        const std::string st = gguf_.get_str(key(".rope.scaling.type"), "");
        if (st == "none" || st.empty()) cfg_.rope_scale = 1.0f;
        // YaRN needs the frequency corrections of rope.cpp on the host; the
        // linear factor is a good first-order approximation and is what the
        // engine applies to inv_freq at load time.
    }
    // ---- rotary: how many dims rotate, and YaRN -------------------------
    // `rope.dimension_count` is the number of head dims the rotation covers and
    // may be narrower than head_dim (partial rotary). llama.cpp reads it the
    // same way — default to the full head, let the key override — so a file
    // that declares it means the same thing to both engines.
    cfg_.rope_dim = static_cast<i32>(gguf_.get_i64(key(".rope.dimension_count"), 0));
    if (cfg_.rope_dim <= 0 || (cfg_.head_dim > 0 && cfg_.rope_dim > cfg_.head_dim))
        cfg_.rope_dim = cfg_.head_dim;
    if (cfg_.head_dim > 0)
        cfg_.rope_frac = static_cast<f32>(cfg_.rope_dim) /
                         static_cast<f32>(cfg_.head_dim);
    cfg_.rope_yarn = gguf_.get_str(key(".rope.scaling.type"), "") == "yarn";

    // ---- hybrid full / sliding-window attention (laguna) -----------------
    // The windowed layers of a hybrid stack carry their own window, their own
    // rotated width and their own rotary table; the full layers keep the keys
    // above. laguna-S alternates full / window x3 — period 4 with the full
    // layer first, which is also what llama.cpp's laguna loader hard-codes.
    cfg_.swa_window = 0;
    if (const i64 win = gguf_.get_i64(key(".attention.sliding_window"), 0); win > 0) {
        cfg_.swa_window = static_cast<i32>(win);
        cfg_.swa_period = 4;
        cfg_.rope_base_swa =
            static_cast<f32>(gguf_.get_f64(key(".rope.freq_base_swa"), 10000.0));
        const i32 rd = static_cast<i32>(
            gguf_.get_i64(key(".rope.dimension_count_swa"), cfg_.rope_dim));
        cfg_.rope_dim_swa =
            (rd > 0 && (cfg_.head_dim <= 0 || rd <= cfg_.head_dim)) ? rd
                                                                   : cfg_.head_dim;
        cfg_.rope_frac_swa = cfg_.head_dim > 0
                                 ? static_cast<f32>(cfg_.rope_dim_swa) /
                                       static_cast<f32>(cfg_.head_dim)
                                 : 1.0f;
    }

    // The attention output gate is a property of the schema, not of a tensor:
    // the loader has to know to look for it before the layer loop starts.
    cfg_.attn_gate = aspec.attn_gate;
    // RoPE's pair convention is a property of the export, so it comes from
    // the arch table rather than from the tensors. KRK_ROPE_NEOX=0/1 forces
    // it, for bisecting a file that disagrees with llama.cpp: with the wrong
    // pairing a model still runs, and still emits fluent-looking text.
    cfg_.rope_neox = aspec.rope_neox;
    if (const char *rn = std::getenv("KRK_ROPE_NEOX"))
        cfg_.rope_neox = rn[0] == '1';

    // ---- router shape ----------------------------------------------------
    // 1 is softmax (the Qwen convention) and 2 is sigmoid (laguna, and the HF
    // default when the key is absent). The bias tensor itself is found per
    // layer; this records only the activation.
    cfg_.n_dense_lead =
        static_cast<i32>(gguf_.get_i64(key(".leading_dense_block_count"), 0));
    cfg_.router_sigmoid =
        static_cast<i32>(gguf_.get_i64(key(".expert_gating_func"), 1)) == 2;
    // Tri-state: -1 keeps the engine's legacy always-normalize for files that
    // do not declare it (see ModelConfig::expert_w_norm).
    cfg_.expert_w_norm =
        gguf_.find(key(".expert_weights_norm")) != nullptr
            ? (gguf_.get_bool(key(".expert_weights_norm")) ? 1 : 0)
            : -1;
    cfg_.expert_w_scale =
        static_cast<f32>(gguf_.get_f64(key(".expert_weights_scale"), 1.0));

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
    if (aspec.gemma_norm) {
        cfg_.rms_eps = static_cast<f32>(
            gguf_.get_f64(key(".attention.layer_norm_rms_epsilon"), 1e-6));
        cfg_.rope_base = 10000.0f;
    }

    // ---- gated delta net geometry (Qwen3.5 / Qwen3-Next) -----------------
    // These archs interleave linear-attention ("recurrent") layers with real
    // attention every `full_attention_interval` layers. The ssm.* keys carry
    // the recurrent block's shape; the tensor schema is validated per layer
    // further down, where the split is actually known.
    cfg_.recurrent = (aspec.shape == ArchShape::Recurrent ||
                      aspec.shape == ArchShape::RecurrentMoe);
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
        // Interleaved multi-rotary sections (Phi). `rope.dimension_count`
        // itself is read once above, for every arch rather than only this one.
        if (const std::vector<i32> *sec = gguf_.get_i32_array(key(".rope.dimension_sections"))) {
            for (size_t i = 0; i < sec->size() && i < 4; i++)
                cfg_.rope_sections[i] = (*sec)[i];
        }
    }

    if (cfg_.n_layer <= 0 || cfg_.n_embd <= 0 || cfg_.n_head <= 0) {
        if (err) *err = "GGUF metadata is missing core geometry (arch='" + a + "')";
        return false;
    }
    // A per-layer head count makes this identity meaningless: laguna's
    // windowed layers carry 72 heads of 128 dims over a 3072-wide embedding,
    // which is not a whole number of heads' worth of embedding at all. head_dim
    // is declared in the file (`attention.key_length`) and is what the loader
    // and the kernels actually use, so only check the identity when there is a
    // single head count for it to hold for.
    if (cfg_.n_head_layer.empty() && cfg_.n_embd % cfg_.n_head != 0) {
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
    // Total upload accounting: the per-tensor dump only covers tensors over
    // 32 MiB, so it cannot say what fraction of load is actually transfer.
    u64 up_bytes = 0;
    f64 up_ms = 0;
    // Size histogram: the copy-size sweep shows the transfer rate is set by
    // the NUMBER of copies (4 KiB -> 0.04 GB/s, 16 MiB -> 10.6 GB/s), so the
    // thing to know is how the model's tensors are distributed.
    u64 bucket_n[8] = {0};
    u64 bucket_b[8] = {0};
    auto bucket_of = [](size_t bytes) {
        if (bytes < 4096u) return 0;
        if (bytes < 16384u) return 1;
        if (bytes < 65536u) return 2;
        if (bytes < 262144u) return 3;
        if (bytes < 1048576u) return 4;
        if (bytes < 4194304u) return 5;
        if (bytes < 16777216u) return 6;
        return 7;
    };
    auto upload_timed = [&](void *dst, const void *src, size_t bytes) {
        const f64 t0 = LoadPhase::now();
        be.upload(dst, src, bytes);
        up_ms += LoadPhase::now() - t0;
        up_bytes += bytes;
        const int bi = bucket_of(bytes);
        bucket_n[bi]++;
        bucket_b[bi] += bytes;
    };
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
        const f64 a0 = LoadPhase::now();
        q.data = be.alloc(t->n_bytes);
        const f64 a1 = LoadPhase::now();
        upload_timed(q.data, t->data, t->n_bytes);
        const f64 a2 = LoadPhase::now();
        if (std::getenv("KRK_PHASE") && t->n_bytes > (32u << 20))
            std::fprintf(stderr,
                         "[alloc] %-24s alloc %8.1f ms  upload %8.1f ms  (%.2f GB/s)\n",
                         t->name.c_str(), a1 - a0, a2 - a1,
                         static_cast<f64>(t->n_bytes) / 1073741824.0 /
                             ((a2 - a1) / 1000.0));
        if (std::getenv("KRK_PHASE")) std::fflush(stderr);
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
        upload_timed(q.data, tap.data(), static_cast<size_t>(n) * 4);
        q.type = DType::F32;
        q.n_in = ksize;
        q.n_out = chan;
        weight_bytes_ += n * 4;
        return q;
    };

    // Uploads a 1-D f32 tensor (norm weights, biases, per-head norms).
    // Latches the one-shot router-bias spelling warning, below.
    bool warned_bias = false;

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
            upload_timed(q.data, half, static_cast<size_t>(t->n_elements) * 2);
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

    lp.mark("arch + metadata resolve");
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
        lp.mark("  token_embd upload");
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
        lp.mark("  output.weight upload");
    }
    out_norm_ = upload_f32("output_norm.weight");
    lp.mark("  output_norm upload");
    if (!out_norm_) {
        if (err) *err = "output_norm.weight is missing";
        return false;
    }

    lp.mark("token_embd + output head");

    // ---- per-layer weights ------------------------------------------------
    // A layer's own head count. The model maximum (`cfg_.n_head`) is only what
    // the shared query workspace is sized against; every per-layer check below
    // has to use this.
    const auto head_at = [&](i32 layer) -> i64 {
        if (cfg_.n_head_layer.empty())
            return static_cast<i64>(cfg_.n_head);
        if (layer < 0 || layer >= static_cast<i32>(cfg_.n_head_layer.size()))
            return static_cast<i64>(cfg_.n_head);
        return static_cast<i64>(cfg_.n_head_layer[static_cast<size_t>(layer)]);
    };

    layers_.resize(static_cast<size_t>(cfg_.n_layer));
    for (i32 l = 0; l < cfg_.n_layer; l++) {
        LayerWeights &L = layers_[static_cast<size_t>(l)];
        if (l == 1) lp.mark("  layer 0 only");
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
            // laguna's per-head output gate is a projection of its own off the
            // same hidden state q/k/v read — not a slice of attn_q — so it is a
            // weight like any other and is uploaded with them.
            if (cfg_.attn_gate)
                L.wattn_gate = need(blk_key("blk.%d.attn_gate.weight", l));
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
            // A per-expert selection bias (laguna): added to the router's
            // *probabilities* before the top-k, and deliberately not to the
            // weights the selected experts are mixed with.
            //
            // Two spellings exist in the wild and the difference is not
            // cosmetic. laguna-xs2-Q4_K_M ships `blk.N.exp_probs_b.bias`, with
            // no `ffn_` prefix; looking only for the prefixed name returned
            // nullptr for every layer of every laguna model, `cfg_.router_bias`
            // stayed false, and expert selection ran un-biased -- silently, with
            // no warning and no failure, because a missing bias tensor and a
            // model without one look identical from here. That is a plausible
            // cause of this engine answering laguna incorrectly while every
            // other MoE model it runs is coherent, and it was invisible because
            // laguna has never been run against the CPU reference (7 s/token).
            // Accept either spelling, and say which one this file used.
            {
                const char *spellings[2] = {"blk.%d.exp_probs_b.bias",
                                           "blk.%d.ffn_exp_probs_b.bias"};
                for (const char *fmt : spellings) {
                    const GgufTensor *bt = gguf_.tensor(blk_key(fmt, l));
                    if (!bt) continue;
                    const i64 n = bt->n_elements;
                    // Host memory: the selection top-k runs here, not on the
                    // device. See the note on LayerWeights::router_bias.
                    f32 *host = static_cast<f32 *>(
                        host_alloc(static_cast<size_t>(n) * sizeof(f32)));
                    dequant_row(bt->type, bt->data, host, n);
                    L.router_bias = host;
                    // Once per model. laguna has 39 routed layers, and 39
                    // identical lines on every load is a warning nobody reads
                    // -- which is how the first one would have stayed unread
                    // anyway.
                    if (strcmp(fmt, spellings[0]) == 0 && !warned_bias) {
                        warned_bias = true;
                        KRK_WARN("router selection bias found as "
                                 "exp_probs_b.bias (no ffn_ prefix); the engine "
                                 "expected the prefixed spelling, so builds "
                                 "before this one ran this model's expert "
                                 "selection un-biased");
                    }
                    break;
                }
                if (L.router_bias) cfg_.router_bias = true;
            }
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
            // The head count is per *layer* on a hybrid stack (laguna runs 48
            // heads on its full layers and 72 on its windowed ones), so the
            // expected width is this layer's own -- comparing against the
            // model maximum would reject every layer that is not the widest.
            const i64 lq_dim = head_at(l) * cfg_.head_dim;
            const i64 expect_q = aspec.q_output_gate ? 2 * lq_dim : lq_dim;
            if (L.wq.n_out != expect_q || L.wk.n_out != kv_dim ||
                L.wv.n_out != kv_dim || L.wo.n_in != lq_dim) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " attention tensors disagree with the declared head geometry";
                return false;
            }
            // laguna's output gate is one scalar per query head, projected
            // from the embedding, so its shape is a second, independent check
            // of the head count this layer is supposed to have.
            if (cfg_.attn_gate &&
                (L.wattn_gate.n_out != head_at(l) || L.wattn_gate.n_in != cfg_.n_embd)) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " attention output gate disagrees with the head count";
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

    lp.mark("per-layer loop (all weights)");

    // ---- rotary inverse frequencies (host) --------------------------------
    // Partial rotary: a layer rotates only the first `rope_dim` dims of each
    // head and leaves the tail untouched, so the table is rope_dim/2 long and
    // the engine passes the fraction to the rotation op.
    //
    // YaRN is folded in *here*, into the frequency itself. Its correction is a
    // per-dimension weight on the frequency and not a function of position, so
    // nothing about it has to run per token; doing it once also leaves the
    // kernel exactly what it was (a position times a frequency) and keeps the
    // position-scaling path free for the linear scaling it already implements.
    //
    // The magnitude half of YaRN is deliberately left out: its mscale is
    // 1 + 0.1*ln(factor) applied to cos/sin, and the reference driver
    // (llama.cpp's llama_context) pre-divides the checkpoint's
    // `yarn_attn_factor` by exactly that before handing it to the rope op. For
    // a checkpoint whose yarn_attn_factor is 1 the two cancel and the attention
    // scale is plain 1/sqrt(head_dim), which is what rope_attn_scale staying at
    // 1.0 means; a checkpoint that asks for something else would need the
    // mscale folded into the attention scale instead.
    //
    // Folding the factor in also means rope_scale must go back to 1.0: the
    // interpolated frequency already carries the 1/factor, and the kernel
    // divides the *position* by rope_scale, so leaving both would apply it
    // twice.
    const i64 half = std::max<i64>(1, cfg_.rope_dim / 2);
    auto build_inv_freq = [](std::vector<f32> &tab, i32 n_dims, f32 base) {
        const i64 h = std::max<i32>(1, n_dims / 2);
        tab.assign(static_cast<size_t>(h), 0.0f);
        for (i64 i = 0; i < h; i++) {
            const f32 exponent =
                static_cast<f32>(2 * i) / static_cast<f32>(n_dims);
            tab[static_cast<size_t>(i)] = std::pow(base, -exponent);
        }
    };
    auto build_yarn = [](std::vector<f32> &tab, i32 n_dims, f32 base, f32 factor,
                         i32 n_ctx_orig, f32 beta_fast, f32 beta_slow) {
        // YaRN (Peng et al., jquesnelle/yarn), matching ggml's rope_yarn +
        // rope_yarn_corr_dims so this table is the one llama.cpp builds at run
        // time: the extrapolated frequency fades into the interpolated one over
        // the pair range [low, high], and everything past `high` — the
        // low-frequency end, which is where long context aliases — is fully
        // interpolated. Note that low and high are computed in *dim* units but
        // compared against pair indices; that is what both references do.
        const i64 h = std::max<i32>(1, n_dims / 2);
        tab.assign(static_cast<size_t>(h), 0.0f);
        const f32 two_pi = 6.28318530717958647692f;
        const f32 l2b = 2.0f * std::log(base);
        f32 low = std::floor(static_cast<f32>(n_dims) *
                             std::log(static_cast<f32>(n_ctx_orig) /
                                      (beta_fast * two_pi)) / l2b);
        f32 high = std::ceil(static_cast<f32>(n_dims) *
                             std::log(static_cast<f32>(n_ctx_orig) /
                                      (beta_slow * two_pi)) / l2b);
        low = std::max(0.0f, low);
        high = std::min(static_cast<f32>(n_dims - 1), high);
        if (high <= low) high = low + 0.001f;
        const f32 fscale = factor > 0.0f ? 1.0f / factor : 1.0f;
        for (i64 i = 0; i < h; i++) {
            const f32 extrap =
                std::pow(base, -static_cast<f32>(2 * i) / static_cast<f32>(n_dims));
            const f32 interp = fscale * extrap;
            const f32 y = (static_cast<f32>(i) - low) / std::max(0.001f, high - low);
            const f32 ramp = 1.0f - std::min(1.0f, std::max(0.0f, y));
            tab[static_cast<size_t>(i)] = interp * (1.0f - ramp) + extrap * ramp;
        }
    };
    if (cfg_.rope_yarn) {
        const f32 factor = cfg_.rope_scale;
        const i32 orig = static_cast<i32>(gguf_.get_i64(
            key(".rope.scaling.original_context_length"), cfg_.n_ctx_train));
        const f32 beta_fast = static_cast<f32>(
            gguf_.get_f64(key(".rope.scaling.yarn_beta_fast"), 32.0));
        const f32 beta_slow = static_cast<f32>(
            gguf_.get_f64(key(".rope.scaling.yarn_beta_slow"), 1.0));
        build_yarn(inv_freq_, cfg_.rope_dim, cfg_.rope_base, factor, orig,
                   beta_fast, beta_slow);
        cfg_.rope_scale = 1.0f;
        KRK_INFO("rope: YaRN over %d of %d dims, base %.0f, factor %.1f, "
                 "original context %d, beta_fast/slow %.0f/%.0f",
                 cfg_.rope_dim, cfg_.head_dim, static_cast<f64>(cfg_.rope_base),
                 static_cast<f64>(factor), orig, static_cast<f64>(beta_fast),
                 static_cast<f64>(beta_slow));
    } else {
        build_inv_freq(inv_freq_, cfg_.rope_dim, cfg_.rope_base);
    }
    // The windowed layers of a hybrid stack keep plain RoPE: llama.cpp passes
    // them an ext_factor of 0, which is the same statement.
    if (cfg_.swa_window > 0)
        build_inv_freq(inv_freq_swa_, cfg_.rope_dim_swa, cfg_.rope_base_swa);
    if (cfg_.rope_neox)
        KRK_INFO("rope pairing: NeoX half-split (channel j with j + %d)",
                 cfg_.rope_dim / 2);

    KRK_INFO("loaded %s (%s): %d layers, %d embd, %d/%d heads, hd=%d, ff=%d, vocab=%d",
             cfg_.name.c_str(), cfg_.arch.c_str(), cfg_.n_layer, cfg_.n_embd,
             cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim, cfg_.n_ff, cfg_.n_vocab);
    if (cfg_.recurrent) {
        KRK_INFO("gated delta net: %d recurrent layers of %d (every %d attends), "
                 "ssm %d key x %d value heads, state %d, conv kernel %d, rope %d/%d",
                 recurrent_layers(), cfg_.n_layer, cfg_.full_attention_interval,
                 cfg_.ssm_n_group, cfg_.ssm_dt_rank, cfg_.ssm_d_state,
                 cfg_.ssm_d_conv, cfg_.rope_dim, cfg_.head_dim);
    }
    if (cfg_.swa_window > 0) {
        KRK_INFO("hybrid attention: %d of every %d layers attend fully, the rest "
                 "within %d tokens; rope %d/%d dims at base %.0f on the windowed "
                 "layers",
                 cfg_.swa_period - 1, cfg_.swa_period, cfg_.swa_window,
                 cfg_.rope_dim_swa, cfg_.head_dim,
                 static_cast<f64>(cfg_.rope_base_swa));
    }
    upload_bytes_ = up_bytes;
    upload_ms_ = up_ms;
    if (std::getenv("KRK_PHASE"))
        std::fprintf(stderr,
                     "[load ] TOTAL upload: %.1f MiB in %.1f ms = %.2f GB/s\n",
                     static_cast<f64>(up_bytes) / 1048576.0, up_ms,
                     static_cast<f64>(up_bytes) / 1073741824.0 / (up_ms / 1000.0));
    if (std::getenv("KRK_PHASE")) {
        static const char *nm[8] = {"<4K", "4-16K", "16-64K", "64-256K",
                                    "256K-1M", "1-4M", "4-16M", ">16M"};
        for (int i = 0; i < 8; i++)
            if (bucket_n[i])
                std::fprintf(stderr, "[load ]   %-9s %5llu copies  %8.1f MiB\n",
                             nm[i], static_cast<unsigned long long>(bucket_n[i]),
                             static_cast<f64>(bucket_b[i]) / 1048576.0);
        std::fflush(stderr);
    }
    lp.mark("post-load (rope table etc)");
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
        // The router bias is HOST memory (see LayerWeights::router_bias): the
        // top-k that reads it runs on the host. It used to be freed through
        // be_->release() on the strength of having been filled by upload_f32(),
        // which is the same device-pointer assumption the read side got wrong.
        if (L.router_bias) {
            host_free(L.router_bias);
            L.router_bias = nullptr;
        }
        for (QuantTensor *q : {&L.wq, &L.wk, &L.wv, &L.wo, &L.wgate, &L.wup,
                               &L.wdown, &L.router, &L.shexp_gate, &L.shexp_up,
                               &L.shexp_down, &L.shexp_inp_gate, &L.wqkv,
                               &L.wqkv_gate, &L.wattn_gate, &L.ssm_conv1d,
                               &L.ssm_out,
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
