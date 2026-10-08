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
    // head_count_kv can be a PER-LAYER array (nemotron_h_moe: 52 entries, 2 on
    // its six attention layers, 0 on the 46 that have none; the gemma4 files
    // in this collection: {1,8}); the scalar read above sees an array and
    // yields 0, and kv_dim 0 then divides by zero the first time an attention
    // layer resolves its KV slot. Keep the whole vector, as head_count does
    // above. The maximum is what a shared workspace and one KV slot are sized
    // against, and a layer's own width is kv_dim_at().
    //
    // Assuming the two were always the same number is what charged a hybrid
    // that keeps KV on 6 of its 52 layers for all 52, and what turned a
    // per-layer width into "attention tensors disagree with the declared head
    // geometry" below.
    if (const std::vector<i32> *hckv =
            gguf_.get_i32_array(key(".attention.head_count_kv"))) {
        cfg_.n_head_kv_layer = *hckv;
        for (i32 v : *hckv)
            if (v > cfg_.n_head_kv) cfg_.n_head_kv = v;
    }
    cfg_.n_ctx_train =
        static_cast<i32>(gguf_.get_i64(key(".context_length"), 2048));
    cfg_.head_dim = static_cast<i32>(gguf_.get_i64(key(".attention.key_length"), 0));
    // Whether the file states the head width itself or has to be inferred
    // from the embedding. A file that states it need not satisfy the
    // n_embd/n_head identity at all: Qwen3.5's gated attention runs 24 heads
    // of 256 dims over a 5120-wide embedding, because q_proj is 2*n_head*d
    // (12288) -- it emits the query gate alongside the query. The identity is
    // only a model promise in the inferred case, so only that case is checked.
    const bool head_dim_declared = cfg_.head_dim > 0;
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
    // nemotron_h / nemotron_h_moe gate their experts with a *sigmoid*, and the
    // reference hardcodes it (llama.cpp's build_moe_ffn call in nemotron-h
    // passes LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID), while the converter writes
    // no expert_gating_func key at all -- so the generic softmax default is
    // wrong for every file of the family, and softmax over 128 logits is a
    // different selection as well as a different weight.
    const bool nemotron_h = (a == "nemotron_h" || a == "nemotron_h_moe");
    cfg_.router_sigmoid = static_cast<i32>(
        gguf_.get_i64(key(".expert_gating_func"), nemotron_h ? 2 : 1)) == 2;
    cfg_.ffn_relu_sqr = nemotron_h;
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
        // nemotron_h_moe: inner_size is independent of dt_rank * d_state
        // GDN (qwen35): inner_size == dt_rank * d_state
        if (a == "nemotron_h" || a == "nemotron_h_moe") {
            cfg_.ssm_value_dim = ssm_inner > 0 ? ssm_inner : cfg_.ssm_dt_rank * cfg_.ssm_d_state;
            cfg_.ssm_inner_size = ssm_inner > 0 ? ssm_inner : cfg_.ssm_dt_rank * cfg_.ssm_d_state;
        } else {
            cfg_.ssm_value_dim = ssm_inner > 0 ? ssm_inner : cfg_.ssm_dt_rank * cfg_.ssm_d_state;
            if (cfg_.ssm_value_dim != cfg_.ssm_dt_rank * cfg_.ssm_d_state) {
                if (err)
                    *err = "arch '" + a + "': ssm.inner_size (" +
                           std::to_string(cfg_.ssm_value_dim) +
                           ") disagrees with time_step_rank * state_size (" +
                           std::to_string(cfg_.ssm_dt_rank * cfg_.ssm_d_state) + ")";
                return false;
            }
            cfg_.ssm_inner_size = cfg_.ssm_value_dim;
        }
        cfg_.ssm_conv_dim = cfg_.ssm_key_dim * 2 + cfg_.ssm_value_dim;
        if (cfg_.ssm_d_conv <= 0 || cfg_.ssm_d_conv > 16) {
            if (err)
                *err = "arch '" + a + "': ssm.conv_kernel " +
                       std::to_string(cfg_.ssm_d_conv) + " is out of range";
            return false;
        }
        // nemotron_h_moe uses Mamba-2 blocks (different from GDN).
        if (a == "nemotron_h" || a == "nemotron_h_moe") {
            cfg_.ssm_inner_size = static_cast<i32>(
                gguf_.get_i64(key(".ssm.inner_size"), 4096));
            cfg_.has_d_param = gguf_.find(key(".ssm.d_state")) != nullptr ||
                              gguf_.find(key("nemotron_h_moe.ssm.d_state")) != nullptr;
            cfg_.nemotron_moe = true;
        } else {
            // A recurrent model needs a backend that can carry its state; refusing
            // here is much better than decoding garbage on one that cannot.
            if (!be.gdn_supported()) {
                if (err)
                    *err = "arch '" + a +
                           "' needs the gated delta net kernels, which this backend "
                           "does not implement";
                return false;
            }
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
    // single head count for it to hold for, and it is not a promise a file
    // that declares `attention.key_length` makes either.
    // Both of these are integer divisions on values that come straight from the
    // file, so a zero has to be refused rather than divided by: the kernel took
    // SIGFPE here, core dumped, before reading a single tensor, on any model
    // whose head-count metadata is absent or empty. A model we cannot use is
    // not a model we should crash on.
    if (cfg_.n_head <= 0) {
        if (err) *err = "model declares no attention head count";
        return false;
    }
    if (cfg_.n_head_layer.empty() && !head_dim_declared &&
        cfg_.n_embd % cfg_.n_head != 0) {
        if (err) *err = "n_embd is not divisible by head count";
        return false;
    }
    if (cfg_.n_head_kv <= 0 || cfg_.n_head % cfg_.n_head_kv != 0) {
        // nemotron_h_moe has unusual head geometry - bypass strict check
        if (a != "nemotron_h" && a != "nemotron_h_moe") {
            if (err) *err = "head count is not a multiple of the KV head count";
            return false;
        }
    }
    // An all-MoE model legitimately omits `feed_forward_length`; fall back to
    // the first expert workspace that is actually described.
    if (cfg_.n_ff <= 0) cfg_.n_ff = cfg_.n_ff_shexp > 0 ? cfg_.n_ff_shexp : cfg_.n_ff_exp;
    if (cfg_.n_ff <= 0) {
        if (err) *err = "feed_forward_length is missing";
        return false;
    }

    const i64 q_dim = static_cast<i64>(cfg_.n_head) * cfg_.head_dim;
    // No single kv_dim here on purpose: the KV width is per layer
    // (kv_dim_at(l)), and the maximum is only what the workspaces and one KV
    // slot are sized against. A local constant is how the geometry check below
    // came to reject every layer that is not the widest.

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
    // The single f32 of a "<tensor>.scale" sidecar, read straight out of the
    // mapping: it is 4 bytes and never leaves the host. NOT upload_weight,
    // which would convert it to f16 and put it on the device -- the scales are
    // applied on the host's behalf, not by a kernel.
    auto one_scale = [&](const std::string &name) -> f32 {
        const GgufTensor *t = gguf_.tensor(name);
        if (!t || !t->data || t->n_elements < 1) return 1.0f;
        if (t->type != DType::F32) return 1.0f;
        return reinterpret_cast<const f32 *>(t->data)[0];
    };

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

    // ---- prism.hadamard activation transform -----------------------------
    // PrismML's llama.cpp fork folds a normalized block-1024 Sylvester
    // Walsh-Hadamard plus a per-width +-1 sign vector into a handful of
    // matmul WEIGHTS, so the engine has to transform those matmuls'
    // ACTIVATIONS (and un-rotate the one table stored latent). Every check
    // that fork's loader makes, this makes too, and anything it cannot
    // express here is refused rather than loaded and decoded as fluent
    // garbage -- which is exactly the failure a missing transform produces:
    // every block read from the right place, every number wrong.
    //
    // `had_layer` carries the per-layer result forward to the weight loop.
    std::vector<u8> had_layer;
    {
        auto fail = [&](const std::string &m) {
            if (err) *err = m;
            return false;
        };
        const bool has_ver = gguf_.find("prism.hadamard.version") != nullptr;
        const bool has_any =
            gguf_.find("prism.hadamard.weight_names") != nullptr;
        if (has_any && !has_ver)
            return fail("prism.hadamard.weight_names is present but "
                        "prism.hadamard.version is not");
        if (has_ver) {
            HadamardPlan plan;
            const i64 ver = gguf_.get_i64("prism.hadamard.version", 0);
            // Version 2 is the tied-output variant (the head reads the
            // rotated embedding table); version 1 requires output.weight to
            // be present and forbids tying.
            if (ver != 1)
                return fail("prism.hadamard.version " + std::to_string(ver) +
                            " is not implemented (this engine folds version 1;"
                            " version 2 is the tied-output variant)");
            if (gguf_.get_bool("prism.hadamard.tied_output", false))
                return fail("prism.hadamard.tied_output is not implemented");
            const i64 block = gguf_.get_i64("prism.hadamard.block_size", 0);
            if (block < 64 || block > 4096 || (block & (block - 1)) != 0)
                return fail("prism.hadamard.block_size " +
                            std::to_string(block) +
                            " is not a power of two in [64, 4096]");
            const std::string transform =
                gguf_.get_str("prism.hadamard.transform");
            if (transform != "normalized-sylvester-walsh-hadamard")
                return fail("prism.hadamard transform '" + transform +
                            "' is not implemented (only the normalized "
                            "Sylvester Walsh-Hadamard is)");
            const std::string axis = gguf_.get_str("prism.hadamard.axis");
            if (axis != "input-last-dimension")
                return fail("prism.hadamard axis '" + axis +
                            "' is not implemented (only input-last-dimension)");
            const std::string sign_mode =
                gguf_.get_str("prism.hadamard.sign_mode");
            if (sign_mode != "identity" && sign_mode != "explicit")
                return fail("prism.hadamard sign_mode '" + sign_mode +
                            "' is not implemented (identity or explicit)");
            const std::vector<std::string> *names =
                gguf_.get_str_array("prism.hadamard.weight_names");
            if (!names || names->empty())
                return fail("prism.hadamard.weight_names is missing or empty");

            // Classify one folded weight: which layer it belongs to (-1 for
            // the untiered output head) and which transform site of that
            // layer it feeds. A name no site answers to is refused -- an
            // expert tensor, a shared expert, a bias, anything this engine
            // would read without transforming.
            auto classify = [](const std::string &nm, i32 *layer,
                               int *bit) -> bool {
                *layer = -1;
                if (nm == "output.weight") {
                    *bit = 64;
                    return true;
                }
                if (nm.compare(0, 4, "blk.") != 0) return false;
                size_t q = 4;
                i32 lay = 0;
                bool digit = false;
                while (q < nm.size() && nm[q] >= '0' && nm[q] <= '9') {
                    lay = lay * 10 + (nm[q] - '0');
                    digit = true;
                    q++;
                }
                if (!digit || q >= nm.size() || nm[q] != '.') return false;
                const std::string rest = nm.substr(q + 1);
                int b = -1;
                if (rest == "attn_q.weight" || rest == "attn_k.weight" ||
                    rest == "attn_v.weight" || rest == "attn_qkv.weight")
                    b = 1;
                else if (rest == "attn_gate.weight")
                    b = 2;
                else if (rest == "attn_output.weight")
                    b = 4;
                else if (rest == "ffn_gate.weight" || rest == "ffn_up.weight")
                    b = 8;
                else if (rest == "ffn_down.weight")
                    b = 16;
                else if (rest == "ssm_out.weight")
                    b = 32;
                else
                    return false;
                *layer = lay;
                *bit = b;
                return true;
            };

            had_layer.assign(static_cast<size_t>(std::max<i32>(cfg_.n_layer, 0)),
                             0);
            std::vector<i64> widths; // distinct folded input widths
            for (const std::string &nm : *names) {
                i32 lay = -1;
                int bit = -1;
                if (!classify(nm, &lay, &bit))
                    return fail("prism.hadamard: weight '" + nm +
                                "' is on no activation-transform site this "
                                "engine implements");
                if (bit == 64) {
                    plan.head_folded = true;
                } else if (lay >= cfg_.n_layer) {
                    // An MTP block: a prediction head this engine does not
                    // run at all, so there is no activation to transform.
                } else if (lay >= 0) {
                    had_layer[static_cast<size_t>(lay)] |=
                        static_cast<u8>(bit);
                } else {
                    return fail("prism.hadamard: weight '" + nm +
                                "' names a negative layer");
                }
                const GgufTensor *t = gguf_.tensor(nm);
                if (!t)
                    return fail("prism.hadamard names weight '" + nm +
                                "' but the file has no such tensor");
                const i64 n_in = static_cast<i64>(t->ne[0]);
                if (n_in <= 0 || n_in % block != 0)
                    return fail("prism.hadamard: block " +
                                std::to_string(block) +
                                " does not divide input width " +
                                std::to_string(n_in) + " of '" + nm + "'");
                if (n_in > plan.max_width) plan.max_width = n_in;
                if (std::find(widths.begin(), widths.end(), n_in) ==
                    widths.end())
                    widths.push_back(n_in);
            }

            // The inverse side: one table, looked up by row, stored rotated.
            // The engine applies the transform right after the lookup, so any
            // other name here would load and stay silently rotated.
            if (const std::vector<std::string> *inv =
                    gguf_.get_str_array("prism.hadamard.inverse_weight_names")) {
                for (const std::string &nm : *inv) {
                    if (nm != "token_embd.weight")
                        return fail("prism.hadamard: inverse table '" + nm +
                                    "' is not a site this engine implements "
                                    "(only token_embd.weight)");
                    plan.embd_inverse = true;
                }
            }
            if (plan.embd_inverse && cfg_.tied_embeddings)
                return fail("a latent token embedding on a TIED output needs "
                            "prism.hadamard version 2, which is not "
                            "implemented");

            // Sign vectors: parsed and validated against the host copy first,
            // uploaded only once nothing below can still fail.
            std::vector<std::pair<i32, std::vector<f32>>> host_signs;
            if (sign_mode == "explicit") {
                const std::vector<i32> *sw =
                    gguf_.get_i32_array("prism.hadamard.sign_widths");
                const std::vector<i32> *sv =
                    gguf_.get_i32_array("prism.hadamard.sign_values");
                if (!sw || sw->empty() || !sv)
                    return fail("prism.hadamard sign_mode is explicit but "
                                "sign_widths/sign_values are missing");
                size_t off = 0;
                for (const i32 w : *sw) {
                    if (w <= 0 || (w % block) != 0 ||
                        off + static_cast<size_t>(w) > sv->size())
                        return fail("prism.hadamard sign width " +
                                    std::to_string(w) + " is invalid");
                    std::vector<f32> host(static_cast<size_t>(w));
                    for (i32 i = 0; i < w; i++) {
                        const i32 v = (*sv)[off + static_cast<size_t>(i)];
                        if (v != 1 && v != -1)
                            return fail("prism.hadamard sign values must be "
                                        "+-1, not " + std::to_string(v));
                        host[static_cast<size_t>(i)] = static_cast<f32>(v);
                    }
                    host_signs.emplace_back(w, std::move(host));
                    off += static_cast<size_t>(w);
                }
                if (off != sv->size())
                    return fail("prism.hadamard sign_values holds " +
                                std::to_string(sv->size()) +
                                " values but the widths account for " +
                                std::to_string(off));
                for (const i64 w : widths) {
                    bool found = false;
                    for (const auto &hs : host_signs)
                        if (static_cast<i64>(hs.first) == w) found = true;
                    if (!found)
                        return fail("prism.hadamard has no sign vector for "
                                    "input width " + std::to_string(w));
                }
            }

            // ssm_out carries an extra grouped-V permutation ahead of the
            // signs when the converter says so; it needs the delta net's head
            // geometry to express it.
            plan.gdn_v_grouped =
                gguf_.get_bool("prism.hadamard.gdn_v_grouped", false);
            bool any_ssm = false;
            for (const u8 f : had_layer)
                if (f & 32) any_ssm = true;
            if (any_ssm && !cfg_.recurrent)
                return fail("prism.hadamard folds ssm_out but this model has "
                            "no gated-delta-net layers");
            if (any_ssm && plan.gdn_v_grouped &&
                (cfg_.ssm_dt_rank <= 0 || cfg_.ssm_n_group <= 0 ||
                 cfg_.ssm_value_dim % cfg_.ssm_dt_rank != 0 ||
                 cfg_.ssm_dt_rank % cfg_.ssm_n_group != 0))
                return fail("prism.hadamard gdn_v_grouped does not divide this "
                            "model's delta-net head geometry");

            plan.active = true;
            plan.block = static_cast<i32>(block);
            had_ = plan;
            for (const auto &hs : host_signs) {
                f32 *dev = static_cast<f32 *>(
                    be.alloc(static_cast<size_t>(hs.first) * 4));
                upload_timed(dev, hs.second.data(),
                             static_cast<size_t>(hs.first) * 4);
                had_signs_.emplace_back(hs.first, dev);
            }
            KRK_INFO("prism.hadamard: %zu folded weight(s)%s, block %d, "
                     "%zu sign vector(s), widest input %lld",
                     names->size(),
                     plan.embd_inverse ? " + latent token embedding" : "",
                     plan.block, had_signs_.size(),
                     static_cast<long long>(plan.max_width));
        }
    }

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
        if (!had_layer.empty()) {
            const u8 f = had_layer[static_cast<size_t>(l)];
            L.h_attn_in = (f & 1) != 0;
            L.h_attn_gate = (f & 2) != 0;
            L.h_attn_out = (f & 4) != 0;
            L.h_ffn_in = (f & 8) != 0;
            L.h_ffn_down = (f & 16) != 0;
            L.h_ssm_out = (f & 32) != 0;
        }

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
        // nemotron_h / nemotron_h_moe have no second norm at all: the converter
        // applies attn_norm before BOTH sub-blocks (the file carries one norm
        // per layer and the graph reads it twice), so the FFN consumes
        // norm(x) through the same weights. A missing post-attention norm is
        // therefore a legitimate shape on this family, not a truncated file.
        // Aliasing the pointer would double-release in Model::unload, which
        // frees attn_norm and ffn_norm independently, so this is a second
        // upload of the same 10.5 KiB rather than a copy of the pointer.
        if (!L.ffn_norm && L.attn_norm)
            L.ffn_norm = upload_f32(blk_key("blk.%d.attn_norm.weight", l));
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
            } else if (cfg_.full_attention_interval > 0) {
                L.gdn = ((l + 1) % cfg_.full_attention_interval) != 0;
            } else {
                // No usable interval: treat every layer as a full-attention
                // layer, which is what an interval of 1 means. This used to be
                // an unguarded modulo and any model that reached this branch
                // without the clamp above took its course -- SIGFPE, core
                // dumped, before a single tensor was read. Laguna-XS.2-IQ4_XS
                // did exactly that on gfx1031.
                L.gdn = false;
            }
            // nemotron_h_moe: override L.gdn based on actual layer type
            // Even layers (0,2,4...) are SSM, odd layers (1,3,5...) are expert
            if (cfg_.nemotron_moe) {
                const std::string ssm_in_name = blk_key("blk.%d.ssm_in.weight", l);
                L.gdn = (gguf_.tensor(ssm_in_name) != nullptr);
                L.nemotron_ssm = L.gdn;
            }
        }

        if (L.gdn) {
            if (L.nemotron_ssm) {
                // Mamba-2 SSM block (nemotron_h / nemotron_h_moe): no
                // QKV, no gate, no alpha/beta. Instead: an input
                // projection (ssm_in), optional D parameter (ssm_d),
                // timestep bias (ssm_dt), the A log-state parameter
                // (ssm_a), the short conv (ssm_conv1d), the output
                // projection (ssm_out) and its norm.
                L.ssm_conv1d = upload_conv1d(blk_key("blk.%d.ssm_conv1d.weight", l));
                L.ssm_conv_bias = upload_f32(blk_key("blk.%d.ssm_conv1d.bias", l));
                L.ssm_in = need(blk_key("blk.%d.ssm_in.weight", l));
                L.ssm_out = need(blk_key("blk.%d.ssm_out.weight", l));
                L.ssm_dt = upload_f32(blk_key("blk.%d.ssm_dt.bias", l));
                L.ssm_a = upload_f32(blk_key("blk.%d.ssm_a", l));
                L.ssm_d = upload_f32(blk_key("blk.%d.ssm_d", l));
                L.ssm_norm = upload_f32(blk_key("blk.%d.ssm_norm.weight", l));
            } else {
                L.wqkv = need(blk_key("blk.%d.attn_qkv.weight", l));
                L.wqkv_gate = need(blk_key("blk.%d.attn_gate.weight", l));
                L.ssm_conv1d = upload_conv1d(blk_key("blk.%d.ssm_conv1d.weight", l));
                L.ssm_out = need(blk_key("blk.%d.ssm_out.weight", l));
                L.ssm_dt = upload_f32(blk_key("blk.%d.ssm_dt.bias", l));
                L.ssm_a = upload_f32(blk_key("blk.%d.ssm_a", l));
                L.ssm_alpha = need(blk_key("blk.%d.ssm_alpha.weight", l));
                L.ssm_beta = need(blk_key("blk.%d.ssm_beta.weight", l));
                L.ssm_norm = upload_f32(blk_key("blk.%d.ssm_norm.weight", l));
            }
        } else if (gguf_.tensor(blk_key("blk.%d.attn_q.weight", l)) ||
                   gguf_.tensor(blk_key("blk.%d.attn_qkv.weight", l)) ||
                   (!cfg_.nemotron_moe && !cfg_.recurrent &&
                    !gguf_.tensor(blk_key("blk.%d.ffn_up_exps.weight", l)))) {
            // Attention layers. Only 6 of nemotron_h_moe's 52 layers carry
            // attention at all (the rest are mamba-2 or pure-MoE blocks), so
            // the q/k/v/output upload is keyed on the tensors being present,
            // not on the layer merely not being recurrent. An ordinary stack
            // requires every layer to attend EXCEPT a layer that carries routed
            // experts: that is a pure-MoE block, the same kind this family runs
            // without attention, and the engine already gives a layer with no
            // q/k/v/output the short path (the `if (L.wq.present())` branch of
            // the forward pass: the residual is unchanged and the FFN reads the
            // norm of the same stream). A layer with neither attention nor
            // experts is still a truncated file, which is what keeps this from
            // accepting half a download as a model.
            L.wq = need(blk_key("blk.%d.attn_q.weight", l));
            L.wk = need(blk_key("blk.%d.attn_k.weight", l));
            L.wv = need(blk_key("blk.%d.attn_v.weight", l));
            L.wo = need(blk_key("blk.%d.attn_output.weight", l));
            // laguna's per-head output gate is a projection of its own off the
            // same hidden state q/k/v read — not a slice of attn_q — so it is a
            // weight like any other and is uploaded with them.
            if (cfg_.attn_gate)
                L.wattn_gate = need(blk_key("blk.%d.attn_gate.weight", l));
        } else {
            // A recurrent nemotron layer that is neither mamba-2 nor
            // attention: a pure-MoE block. Nothing to upload here.
            KRK_WARN("layer %d has no attention tensors; running as a pure FFN block", l);
        }
        // A layer is MoE when it carries routed-expert tensors. This is decided
        // per layer: hybrid stacks (dense early layers, MoE later ones) load
        // correctly without extra metadata.
        // Qwen3-style MoE: routed experts gated by ffn_gate_exps.weight, with the
        // router projection in ffn_gate_inp.weight and up/down in ffn_up_exps /
        // ffn_down_exps. nemotron_h_moe-style MoE: the routed experts are ffn_up_exps /
        // ffn_down_exps and the router projection is ffn_gate_inp.weight, but there is
        // no ffn_gate_exps.weight — the layer is MoE when ffn_up_exps.weight is present
        // and ffn_gate.weight is not (i.e. the per-token gate is absent).
        const std::string exps_gate = blk_key("blk.%d.ffn_gate_exps.weight", l);
        const bool has_exps_up = gguf_.tensor(blk_key("blk.%d.ffn_up_exps.weight", l)) != nullptr;
        const bool has_dense_gate = gguf_.tensor(blk_key("blk.%d.ffn_gate.weight", l)) != nullptr;
        // nemotron_h_moe alternates mamba-2 blocks (NO ffn of any kind: the
        // layer is ssm_in/conv/ssm_out and nothing else) with attention + MoE
        // blocks. An FFN-less layer is a legitimate shape on this family, so
        // the dense-FFN requirement below applies only when the layer carries
        // dense-ffn tensors at all.
        const bool has_dense_ffn =
            gguf_.tensor(blk_key("blk.%d.ffn_up.weight", l)) != nullptr;
        L.moe = gguf_.tensor(exps_gate) != nullptr || (has_exps_up && !has_dense_gate);
        if (!L.moe && has_dense_ffn) {
            L.wgate = need(blk_key("blk.%d.ffn_gate.weight", l));
            L.wup = need(blk_key("blk.%d.ffn_up.weight", l));
            L.wdown = need(blk_key("blk.%d.ffn_down.weight", l));
        } else if (L.moe) {
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
            // nemotron_h_moe routes with up/down only — there is no
            // ffn_gate_exps — so a null gate is a complete set on this family,
            // not a partial one. What IS partial: either of the two tensors
            // every schema carries being missing.
            if (!up || !dn) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " has a partial routed-expert tensor set";
                return false;
            }
            for (const GgufTensor *t : {g, up, dn}) {
                if (t && !dtype_supported(t->type)) {
                    if (err)
                        *err = "expert tensor '" + t->name +
                               "' uses unsupported format " + dtype_name(t->type);
                    return false;
                }
            }
            L.experts.gate = g;
            L.experts.up = up;
            L.experts.down = dn;
            // The NVFP4 global scales, if this bank is NVFP4. Absent on every
            // other format, and a missing sidecar is 1.0 rather than an error,
            // so the loader stays format-agnostic here.
            L.experts.up_scale =
                gguf_.tensor(blk_key("blk.%d.ffn_up_exps.scale", l));
            L.experts.down_scale =
                gguf_.tensor(blk_key("blk.%d.ffn_down_exps.scale", l));
            L.experts.gate_scale = g ? gguf_.tensor(exps_gate + ".scale")
                                     : nullptr;
            {
                const GgufTensor *src = nullptr;
                if (g && g->n_dims == 3)
                    src = g;
                else if (up && up->n_dims == 3)
                    src = up;
                else if (dn && dn->n_dims == 3)
                    src = dn;
                if (!src) {
                    if (err)
                        *err = "layer " + std::to_string(l) +
                               " is marked MoE but its routed-expert tensors are not 3-D";
                    return false;
                }
                // ne = [n_in, n_ff, n_expert] for BOTH schemas -- Qwen3's
                // ffn_gate_exps and nemotron's ffn_up_exps are laid out the same
                // way. inspect prints shapes largest-stride-first, so its
                // "128x1856x2688" is ne[2]xne[1]xne[0]; reading the expert dim
                // off ne[0] sets n_expert=2688 and every MoE layer is refused
                // against expert_count. tools/cpu_expert_ceiling.cpp says this in
                // its header for the same reason.
                L.experts.n_expert = static_cast<i32>(src->ne[2]);
                L.experts.n_ff_exp = static_cast<i64>(src->ne[1]);
                L.experts.n_embd = static_cast<i64>(src->ne[0]);
            }

            // Shared expert (when the file declares one). Both Qwen2/Qwen3-MoE and
            // nemotron_h_moe spell the shared expert as a 2-D gate/up/down set
            // (ffn_gate_shexp / ffn_up_shexp / ffn_down_shexp). A file that carries
            // this set does not carry ffn_gate_inp_shexp.
            // Gated on up+down, not on the gate: nemotron_h_moe's shared
            // expert is ffn_up_shexp/ffn_down_shexp with no ffn_gate_shexp, so
            // requiring the gate skipped the shared expert entirely -- the same
            // defect as the routed bank's, one tensor over. up/down are what
            // every schema carries.
            if (gguf_.tensor(blk_key("blk.%d.ffn_up_shexp.weight", l)) &&
                gguf_.tensor(blk_key("blk.%d.ffn_down_shexp.weight", l))) {
                L.shexp_gate = upload_weight(blk_key("blk.%d.ffn_gate_shexp.weight", l), true, &e);
                L.shexp_up = upload_weight(blk_key("blk.%d.ffn_up_shexp.weight", l), true, &e);
                L.shexp_down = upload_weight(blk_key("blk.%d.ffn_down_shexp.weight", l), true, &e);
                L.shexp_up_scale =
                    one_scale(blk_key("blk.%d.ffn_up_shexp.scale", l));
                L.shexp_down_scale =
                    one_scale(blk_key("blk.%d.ffn_down_shexp.scale", l));
                L.shexp_gate_scale =
                    one_scale(blk_key("blk.%d.ffn_gate_shexp.scale", l));
            }
            // Qwen3.5 gates the shared expert with a per-token scalar built from a
            // 1-D row; the Qwen2/Qwen3/nemotron schema has no such vector.
            L.shexp_inp_gate =
                upload_weight(blk_key("blk.%d.ffn_gate_inp_shexp.weight", l), true, &e);
        }
        if (!e.empty()) {
            if (err) *err = e;
            return false;
        }

        // geometric sanity: the tensor schema must match the metadata
        if (L.gdn) {
            // The gated-delta-net block has no head geometry at all: one fused
            // [q|k|v] projection and one output projection, both sized by the
            // ssm.* keys. nemotron's mamba-2 block has no wqkv or z gate at all
            // (it projects straight to the block width with ssm_in), so its
            // geometry is checked against ssm_in/ssm_out further down.
            if (L.nemotron_ssm) {
                // Mamba-2's in_proj is one fused row [xBC | z | dt]:
                //   xBC = inner + 2*(n_group*d_state)   (the conv block)
                //   z   = inner                          (the output gate)
                //   dt  = n_heads                        (per-head step, softplus'd)
                // time_step_rank doubles as the head count on this family.
                const i64 bc_w = cfg_.ssm_inner_size +
                                 2 * cfg_.ssm_key_dim;
                const i64 in_w = bc_w + cfg_.ssm_inner_size + cfg_.ssm_dt_rank;
                if (L.ssm_in.n_out != in_w || L.ssm_in.n_in != cfg_.n_embd) {
                    if (err)
                        *err = "layer " + std::to_string(l) +
                               " ssm_in disagrees with the declared ssm geometry";
                    return false;
                }
                cfg_.ssm_conv_dim = static_cast<i32>(bc_w);
            } else if (L.wqkv.n_out != cfg_.ssm_conv_dim ||
                       L.wqkv.n_in != cfg_.n_embd) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " attn_qkv disagrees with the declared ssm geometry";
                return false;
            }
            if (!L.nemotron_ssm && L.wqkv_gate.n_out != cfg_.ssm_value_dim) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " attn_gate disagrees with ssm.inner_size";
                return false;
            }
            if (L.ssm_out.n_in != (L.nemotron_ssm ? cfg_.ssm_inner_size
                                                  : cfg_.ssm_value_dim) ||
                L.ssm_out.n_out != cfg_.n_embd) {
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
            if (L.nemotron_ssm) {
                if (!L.ssm_dt || !L.ssm_a || !L.ssm_norm || !L.ssm_in.present() ||
                    !L.ssm_out.present() || !L.ssm_conv1d.present()) {
                    if (err)
                        *err = "layer " + std::to_string(l) +
                               " is missing its nemotron ssm state parameters";
                    return false;
                }
                // Mamba-2 D parameter is optional in this engine's schema; absent is
                // fine, which is why has_d_param is a config flag rather than a load
                // requirement.
            } else {
                if (!L.ssm_dt || !L.ssm_a || !L.ssm_norm || !L.ssm_alpha.present() ||
                    !L.ssm_beta.present()) {
                    if (err)
                        *err = "layer " + std::to_string(l) +
                               " is missing its ssm state parameters";
                    return false;
                }
            }
        } else if (L.moe) {
            // Qwen3.5's attending layers pack a per-head output gate into the
            // query projection, so its row count is twice the head width.
            // The head count is per *layer* on a hybrid stack (laguna runs 48
            // heads on its full layers and 72 on its windowed ones), so the
            // expected width is this layer's own -- comparing against the
            // model maximum would reject every layer that is not the widest.
            //
            // nemotron_h_moe's pure-MoE layers have no attention tensors, so
            // the head-geometry check only applies to layers that carry one.
            const i64 lq_dim = head_at(l) * cfg_.head_dim;
            // The KV width is this layer's own as well. The same reasoning as
            // the query above applies to it with one extra case: head_count_kv
            // is an array on a hybrid (nemotron_h_moe 0/2, gemma4 {1,8}), so
            // comparing against the stack maximum refused every layer that does
            // not carry the widest KV -- and reported it as the file disagreeing
            // with its own declared geometry.
            const i64 lkv_dim = kv_dim_at(l);
            const i64 expect_q = aspec.q_output_gate ? 2 * lq_dim : lq_dim;
            const bool layer_has_attn = L.wq.present();
            if (layer_has_attn &&
                (L.wq.n_out != expect_q || L.wk.n_out != lkv_dim ||
                 L.wv.n_out != lkv_dim || L.wo.n_in != lq_dim)) {
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
        if (!L.moe && has_dense_ffn &&
            (L.wgate.n_out != cfg_.n_ff || L.wdown.n_in != cfg_.n_ff)) {
            if (err)
                *err = "layer " + std::to_string(l) + " FFN tensors disagree with n_ff";
            return false;
        }
        if (L.moe) {
            // ---- router geometry ------------------------------------------
            if (L.router.n_out != cfg_.n_expert || L.router.n_in != cfg_.n_embd) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " router disagrees with expert_count/embedding_length";
                return false;
            }
            // ---- routed-expert geometry (per-layer) ---------------------
            // The expert tensors are only *described* here so the cache can know
            // per-expert byte boundaries; nothing is uploaded at load time.
            //
            // Nemotron-style MoE (nemotron_h_moe) does not carry ffn_gate_exps,
            // so the geometry is read off the up tensor instead. The sanity that
            // matters for the forward path's intermediate workspace is the
            // routed n_ff_exp; the shared-expert sanity is the one that gates the
            // shared branch.
            {
                const GgufTensor *src = nullptr;
                if (L.experts.up && L.experts.up->n_dims == 3)
                    src = L.experts.up;
                else if (L.experts.down && L.experts.down->n_dims == 3)
                    src = L.experts.down;
                // ffn_gate_exps (Qwen3) is also 3-D and can stand in when the
                // up/down naming is the other convention.
                else if (L.experts.gate && L.experts.gate->n_dims == 3)
                    src = L.experts.gate;
                if (src) {
                    // Same ne order as the assignment site above: expert dim is
                    // ne[2] for every converter seen so far.
                    L.experts.n_expert = static_cast<i32>(src->ne[2]);
                    L.experts.n_ff_exp = static_cast<i64>(src->ne[1]);
                    L.experts.n_embd = static_cast<i64>(src->ne[0]);
                }
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
        // Qwen3/Gemma3 store the QK-norm weight as one head (head_dim wide);
        // OLMoE stores it across the whole projected row (n_embd wide) because
        // its reference applies the norm before splitting into heads. Nothing
        // in the schema says which, and the mistake is invisible in the output:
        // the model still loads, still answers, just wrongly -- and wrongly on
        // the CPU reference too, because both backends call the same op. So
        // decide from the stored width, and refuse a width that is neither
        // rather than reading it as whichever happened to fit.
        if (cfg_.qk_norm && cfg_.n_embd > 0 && cfg_.head_dim > 0 &&
            cfg_.n_embd != cfg_.head_dim) {
            const i64 wide = cfg_.n_embd, per_head = cfg_.head_dim;
            cfg_.qk_norm_wide = false;
            for (int i = 0; i < 2; i++) {
                const char *nm = i == 0 ? "blk.%d.attn_q_norm.weight"
                                        : "blk.%d.attn_k_norm.weight";
                const std::string name = blk_key(nm, l);
                const GgufTensor *t = gguf_.tensor(name);
                if (!t) continue;
                const i64 w = t->n_elements;
                if (w == wide) cfg_.qk_norm_wide = true;
                else if (w != per_head) {
                    if (err)
                        *err = format("%s is %lld wide, which is neither "
                                      "head_dim (%lld) nor the full row (%lld); "
                                      "this engine does not know how to apply it",
                                      name.c_str(), static_cast<long long>(w),
                                      static_cast<long long>(per_head),
                                      static_cast<long long>(wide));
                    return false;
                }
            }
        }
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
                               &L.ssm_out, &L.ssm_in,
                               &L.ssm_alpha, &L.ssm_beta})
            if (q->present()) be_->release(q->data);
        if (L.ssm_d) be_->release(L.ssm_d);
    }
    layers_.clear();
    for (auto &sg : had_signs_)
        if (sg.second) be_->release(sg.second);
    had_signs_.clear();
    had_ = HadamardPlan{};
    if (tok_embd_.present()) be_->release(tok_embd_.data);
    if (!cfg_.tied_embeddings && out_head_.present()) be_->release(out_head_.data);
    if (out_norm_) be_->release(out_norm_);
    tok_embd_ = {};
    out_head_ = {};
    out_norm_ = nullptr;
    be_ = nullptr;
}

} // namespace krk
