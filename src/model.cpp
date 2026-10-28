// model.cpp — resolve GGUF metadata + tensor schema into device-resident weights.
#include "krk/host_time.hpp"
#include "krk/model.hpp"
#include "krk/arch.hpp"
#include "krk/verdict.hpp"
#include "krk/vram_guard.hpp"
#include "krk/backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>

namespace krk {

namespace {

std::string blk_key(const char *fmt, i32 layer) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), fmt, layer);
    return std::string(buf);
}

// Every QuantTensor a layer owns. One list, because two loops have to agree
// with LayerWeights: the per-tensor release in unload(), and the binding of a
// streamed entry to the tensor that a promotion re-points. A field added to
// LayerWeights and missed here leaks its allocation, or -- worse -- keeps
// pointing into the ring after the layer was promoted into its own base.
std::vector<QuantTensor *> qt_fields(LayerWeights &L) {
    return {&L.wq,         &L.wk,        &L.wv,          &L.wo,
            &L.wgate,      &L.wup,       &L.wdown,       &L.router,
            &L.shexp_gate, &L.shexp_up,  &L.shexp_down,  &L.shexp_inp_gate,
            &L.wqkv,       &L.wqkv_gate, &L.wattn_gate,
            &L.ssm_conv1d, &L.ssm_out,   &L.ssm_in,
            &L.wq_direct,    &L.wkv_a_mqa,   &L.wkv_b,      &L.wkv_b16,
            // The modern MLA layouts (src/model.cpp: the MLA branch of the
            // layer loop). A field added to LayerWeights and missed here
            // leaks its allocation -- or, on the streamed path, keeps
            // pointing into the ring after the layer was promoted.
            &L.wq_a,      &L.wq_b,      &L.wkv,
            &L.wkv_a,     &L.wk_b,      &L.wv_b,
            &L.wo_a,      &L.wo_b,
            // The V4 f16 staging: five more allocations a layer, and missing
            // them here leaks every one of them on unload.
            &L.wq_a16,    &L.wq_b16,    &L.wkv16,
            &L.wo_a16,    &L.wo_b16,
            &L.ssm_alpha,  &L.ssm_beta,
            // LatentMoE's pair: the narrow projection the routed bank reads and
            // the one that lifts its result back to n_embd.
            &L.latent_down, &L.latent_up};
}

} // namespace

const char *mla_layout_name(MlaLayout l) {
    switch (l) {
        case MlaLayout::Legacy: return "legacy lite";
        case MlaLayout::Split:  return "split (per-head K/V)";
        case MlaLayout::Shared: return "shared-latent";
        case MlaLayout::None:   break;
    }
    return "none";
}

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
    model_path_ = path;
    {
    // Mapping + metadata: on a 74 GiB file this is the parse of every tensor
    // entry, and it happens before anything else can start.
    KRK_TIME_HOST("load.map");
    if (!gguf_.load(path, err)) return false;
    }
    lp.mark("gguf map + metadata");

    // Large weights are pulled in with read() rather than DMA'd out of the cold
    // mapping: 6.2 GB/s single-threaded and 16.3 with four handles, against
    // 3.87 for a cold mapping, because the mapping's cost is one soft page fault
    // per 4 KiB (measured 1208.8 ms vs 109.1 ms for the 8B). The backend owns
    // the pipeline and overlaps the reads with the copies; it falls back to the
    // direct copy if pinned memory cannot be had, so this is an optimisation,
    // not a requirement. See docs/KNOWLEDGE-MODELS.md.
    // Build the shard table so the backend can turn source pointers into
    // file offsets for tensors living in any shard.
    std::vector<WeightSource> sources;
    sources.reserve(gguf_.shard_count());
    for (i32 s = 0; s < static_cast<i32>(gguf_.shard_count()); s++) {
        const GgufShard &shard = gguf_.shards()[s];
        sources.push_back({
            shard.file->path(),           // path to the shard file
            static_cast<const u8 *>(shard.file->data()),  // map base
            shard.file->size(),           // map size
            shard.data_off,               // tensor data offset within this shard
        });
    }
    be.set_weight_sources(sources);

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
    // Development escape hatch, and it is loader-only on purpose:
    // KRK_ARCH_ALLOW_EXPERIMENTAL=1 lets the tensors of a refused schema be
    // read, so a half-ported architecture can be checked stage by stage.
    // Nothing user-facing sets it, and the warning says what it is -- the
    // "generation will refuse" this used to promise was not true: Engine::init
    // has no verdict check, so a file the hatch let through generated whatever
    // the engine's own blocks could produce, blockers or not.
    const char *allow_exp = std::getenv("KRK_ARCH_ALLOW_EXPERIMENTAL");
    const bool experimental_load =
        allow_exp && allow_exp[0] == '1' && verdict.spec != nullptr;
    if (!verdict.runnable && experimental_load) {
        KRK_WARN("architecture '%s': loading an UNSUPPORTED schema for development "
                 "(KRK_ARCH_ALLOW_EXPERIMENTAL=1); the loader's blockers are not "
                 "enforced for this run",
                 verdict.arch.empty() ? "no arch" : verdict.arch.c_str());
    } else if (!verdict.runnable) {
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
    // Two keys name the same epsilon. RMSNorm families write
    // `layer_norm_rms_epsilon`; the LayerNorm families (gpt2, gptneox, gptj,
    // phi2) write `layer_norm_epsilon`. Reading only the first left every one
    // of those files on the 1e-5 default -- right for all four in practice and
    // wrong as a rule, since the file is what says.
    cfg_.rms_eps = static_cast<f32>(gguf_.get_f64(
        key(".attention.layer_norm_rms_epsilon"),
        gguf_.get_f64(key(".attention.layer_norm_epsilon"), 1e-5)));
    cfg_.rope_base = static_cast<f32>(gguf_.get_f64(key(".rope.freq_base"), 10000.0));
    cfg_.rope_scale = static_cast<f32>(gguf_.get_f64(key(".rope.scaling.factor"), 1.0));
    cfg_.logit_softcap =
        static_cast<f32>(gguf_.get_f64(key(".final_logit_softcapping"), 0.0));
    cfg_.attn_softcap =
        static_cast<f32>(gguf_.get_f64(key(".attention.logit_softcapping"), 0.0));
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
    // Whether this file gives its windowed layers a rotary table of their own.
    // A file that declares neither `rope.freq_base_swa` nor
    // `rope.dimension_count_swa` has ONE table for the whole stack -- gpt-oss
    // is that shape -- so the windowed layers must rotate with exactly the
    // full layers' table, YaRN included, rather than a 10000-based
    // approximation of it.
    bool swa_own_rope = false;
    if (const i64 win = gguf_.get_i64(key(".attention.sliding_window"), 0); win > 0) {
        cfg_.swa_window = static_cast<i32>(win);
        // Which layers are windowed is a property of the architecture, not of
        // the metadata: the GGUF states the window and nothing about where it
        // applies. laguna (the default) is period 4 with the full layer first;
        // gpt-oss is period 2 with the windowed layer first, which the arch
        // table declares.
        cfg_.swa_period = aspec.swa_period > 0 ? aspec.swa_period : 4;
        cfg_.swa_phase = aspec.swa_window_first ? 1 : 0;
        const f64 swa_base = gguf_.get_f64(key(".rope.freq_base_swa"), 0.0);
        const i64 swa_dim = gguf_.get_i64(key(".rope.dimension_count_swa"), 0);
        swa_own_rope = swa_base > 0.0 || swa_dim > 0;
        cfg_.rope_base_swa = swa_base > 0.0 ? static_cast<f32>(swa_base)
                                           : cfg_.rope_base;
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
    cfg_.q_output_gate = aspec.q_output_gate;
    // muse-glimmer gates with a sigmoid where laguna gates with a softplus.
    cfg_.attn_gate_sigmoid = aspec.attn_gate_sigmoid;
    // Per-branch post norms, and the GeGLU FFN that comes with the gemma stack.
    // `post_norm_eps` 0 means "use the model's rms_eps", which is the gemma case;
    // muse-glimmer states its own 1e-8 for these two norms.
    cfg_.post_branch_norm = aspec.post_branch_norm;
    // Stored as given; 0 means "the model's rms_eps", resolved by the engine at
    // the call -- the gemma eps override is applied further down, AFTER this
    // point, so resolving it here would capture the pre-override value.
    cfg_.post_norm_eps = static_cast<f32>(aspec.post_norm_eps);
    cfg_.gelu_ffn = aspec.gelu_ffn;
    cfg_.gpt2_style = aspec.gpt2_style;
    cfg_.fused_qkv = aspec.fused_qkv;
    cfg_.fused_gate_up = aspec.fused_gate_up;
    cfg_.ln_bias = aspec.ln_bias;
    cfg_.learned_pos = aspec.learned_pos;
    cfg_.parallel_residual = aspec.parallel_residual;
    cfg_.no_gate_ffn = aspec.no_gate_ffn;
    cfg_.embd_norm = aspec.embd_norm;
    cfg_.parallel_two_norm = aspec.parallel_two_norm;
    // The arch entry supplies the family's own constant (BLOOM's 8.0); the
    // file's key wins when it has one, which is how MPT states its value.
    cfg_.alibi_max_bias = static_cast<f32>(gguf_.get_f64(
        key(".attention.max_alibi_bias"), aspec.alibi_max_bias));
    cfg_.rope_swa_only = aspec.rope_swa_only;
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
    // A file that is silent on it takes its architecture's default, and the two
    // answers are not close: gpt-oss routes 128 experts and its top four hold
    // ~13% of the softmax mass, so renormalizing them multiplies each layer's
    // expert contribution by 7.6 and the residual stream grows instead of
    // holding. llama.cpp leaves `expert_weights_norm` false for this arch.
    if (cfg_.expert_w_norm < 0 && aspec.expert_w_norm >= 0)
        cfg_.expert_w_norm = aspec.expert_w_norm;
    cfg_.ffn_swiglu_oai = aspec.swiglu_oai;
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
    // `expert_used_count` and `expert_feed_forward_length` may be ARRAYS, one
    // entry per block. Nemotron Bolt 74B-A9B writes both that way: top-k is 0
    // on its Mamba-2 blocks and 9 on its 40 MoE blocks, and the expert width
    // varies (2048/1536/...). `get_i64` on an array yields the default, so a
    // scalar-only read turned that file into "top-1 of a 496-wide router that
    // is only 240 wide" -- two different wrong answers in one number.
    // The array is the truth; the model-level scalars become the maxima, which
    // is what a workspace has to be sized for.
    if (const std::vector<i32> *used_arr = gguf_.get_i32_array(key(".expert_used_count"))) {
        cfg_.layer_n_expert_used = *used_arr;
        i32 mx = 0;
        for (i32 v : *used_arr)
            if (v > mx) mx = v;
        if (mx > 0) cfg_.n_expert_used = mx;
    }
    if (const std::vector<i32> *ff_arr =
            gguf_.get_i32_array(key(".expert_feed_forward_length"))) {
        i32 mx = 0;
        for (i32 v : *ff_arr)
            if (v > mx) mx = v;
        if (mx > 0) cfg_.n_ff_exp = mx;
    }
    // `feed_forward_length` is per-block in the same files (0 on the Mamba-2
    // blocks, 2048/1536 on the MoE ones). Nothing routes by a dense width per
    // layer here, so the maximum is the only safe reading: it bounds every
    // workspace that a dense FFN block could ask for.
    if (const std::vector<i32> *ffl =
            gguf_.get_i32_array(key(".feed_forward_length"))) {
        i32 mx = 0;
        for (i32 v : *ffl)
            if (v > mx) mx = v;
        if (mx > 0) cfg_.n_ff = mx;
    }
    cfg_.n_ff_shexp = static_cast<i32>(
        gguf_.get_i64(key(".expert_shared_feed_forward_length"), 0));
    cfg_.n_expert_shared = static_cast<i32>(gguf_.get_i64(
        key(".expert_shared_count"), cfg_.n_ff_shexp > 0 ? 1 : 0));
    // Files without `expert_shared_feed_forward_length` stack n_shared
    // experts of width n_ff_exp (deepseek2: 2 x 1408 = 2816); the width is
    // then metadata-derivable, not a default of 0 that refuses the file at
    // the shared-expert check below.
    if (cfg_.n_expert > 0 && cfg_.n_ff_shexp <= 0 && cfg_.n_expert_shared > 0 &&
        cfg_.n_ff_exp > 0)
        cfg_.n_ff_shexp = cfg_.n_ff_exp * cfg_.n_expert_shared;
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
        // LFM2.5 is "recurrent" in the layer-interleaving sense only: its
        // non-attention layers are a short convolution, and it carries no
        // ssm.* key at all. The one number it does declare is the conv
        // kernel (`shortconv.l_cache`), and that is the whole geometry the
        // block needs -- so this branch must run before the delta-net shape
        // checks, not through them.
        if (aspec.family && std::string_view(aspec.family) == "lfm2") {
            cfg_.lfm_conv = true;
            cfg_.full_attention_interval = static_cast<i32>(
                gguf_.get_i64(key(".full_attention_interval"), 4));
            if (cfg_.full_attention_interval <= 0)
                cfg_.full_attention_interval = 4;
            cfg_.ssm_d_conv = static_cast<i32>(
                gguf_.get_i64(key(".shortconv.l_cache"), 3));
            if (cfg_.ssm_d_conv <= 0 || cfg_.ssm_d_conv > 16) {
                if (err)
                    *err = "arch '" + a + "': shortconv.l_cache " +
                           std::to_string(cfg_.ssm_d_conv) +
                           " is out of range";
                return false;
            }
        } else {
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
        } // !lfm2: the delta-net / Mamba-2 geometry below is not this stack's
        // Interleaved multi-rotary sections (Phi). `rope.dimension_count`
        // itself is read once above, for every arch rather than only this one.
        if (const std::vector<i32> *sec = gguf_.get_i32_array(key(".rope.dimension_sections"))) {
            for (size_t i = 0; i < sec->size() && i < 4; i++)
                cfg_.rope_sections[i] = (*sec)[i];
        }
    }

    // ---- MLA (DeepSeek-V2/V3/V4) ----------------------------------------
    // Latent ranks off the file's own attention metadata. The names matter:
    // exporters write q_lora_rank (missing on lite -- the query is direct),
    // kv_lora_rank (missing on V4 -- the latent is key_length wide there),
    // key_length and value_length. There are no qk_nope_head_dim-style keys;
    // reading those refused every real file (Coder-V2-Lite proved it: it
    // carries key_length=192/value_length=128/kv_lora_rank=512 and none of
    // the old names). nope is key_length - rope_dim.
    if (aspec.mla) {
        cfg_.mla = true;
        cfg_.mla_q_lora = static_cast<i32>(
            gguf_.get_i64(key(".attention.q_lora_rank"), 0));
        cfg_.mla_kv_lora = static_cast<i32>(
            gguf_.get_i64(key(".attention.kv_lora_rank"), 0));
        const i64 key_len = gguf_.get_i64(key(".attention.key_length"), 0);
        const i64 val_len = gguf_.get_i64(key(".attention.value_length"), 0);
        if (cfg_.mla_kv_lora <= 0) cfg_.mla_kv_lora = static_cast<i32>(key_len);
        cfg_.mla_qk_rope = cfg_.rope_dim;
        cfg_.mla_qk_nope = static_cast<i32>(key_len) - cfg_.mla_qk_rope;
        cfg_.mla_v_dim = static_cast<i32>(val_len);
        if (cfg_.mla_kv_lora <= 0 || key_len <= 0 || val_len <= 0 ||
            cfg_.mla_qk_nope <= 0 || cfg_.mla_qk_rope <= 0) {
            if (err)
                *err = "arch '" + a + "': MLA ranks are incomplete " +
                       "(kv_lora_rank/key_length/value_length must be > 0; " +
                       "nope = key_length - rope.dimension_count)";
            return false;
        }
    }

    // ---- DeepSeek-V4 extras ---------------------------------------------
    // Grouped output, window, hash-route depth, compressed rope base and the
    // per-layer ratios. compress_ratios must cover every layer: a short
    // array is a truncated download, not a default. get_i64_array widens any
    // integral element width, so the check is on the count, not the type.
    if (aspec.dsv4) {
        cfg_.dsv4 = true;
        cfg_.dsv4_o_groups = static_cast<i32>(
            gguf_.get_i64(key(".attention.output_group_count"), 0));
        cfg_.dsv4_o_lora = static_cast<i32>(
            gguf_.get_i64(key(".attention.output_lora_rank"), 0));
        cfg_.dsv4_window = static_cast<i32>(
            gguf_.get_i64(key(".attention.sliding_window"), 0));
        cfg_.dsv4_hash_layers = static_cast<i32>(
            gguf_.get_i64(key(".hash_layer_count"), 0));
        cfg_.dsv4_compress_rope_base = static_cast<f32>(
            gguf_.get_f64(key(".attention.compress_rope_freq_base"), 0.0));
        std::vector<i64> ratios;
        const bool have_ratios =
            gguf_.get_i64_array(key(".attention.compress_ratios"), &ratios);
        if (have_ratios) {
            cfg_.dsv4_ratio.assign(ratios.begin(), ratios.end());
            for (i64 v : ratios)
                if (v != 0 && v != 4 && v != 128) {
                    if (err)
                        *err = "arch '" + a + "': compress_ratios carries " +
                               std::to_string(v) + " (want 0, 4 or 128)";
                    return false;
                }
        }
        if (cfg_.dsv4_o_groups <= 0 || cfg_.dsv4_o_lora <= 0 ||
            cfg_.dsv4_window <= 0 || cfg_.dsv4_compress_rope_base <= 0.0f ||
            !have_ratios ||
            static_cast<i32>(ratios.size()) < cfg_.n_layer) {
            if (err)
                *err = "arch '" + a + "': V4 metadata is incomplete " +
                       "(output_group_count/output_lora_rank/sliding_window/"
                       "compress_rope_freq_base must be > 0 and "
                       "compress_ratios must cover every layer)";
            return false;
        }
    }

    // ---- qwen4exp: hyper-connections, PLE, QSA ---------------------------
    // Read only for the schema flags, so a file of another family costs four
    // pointer tests and nothing else.
    if (aspec.hyper_connection) {
        cfg_.hc = true;
        cfg_.hc_n = static_cast<i32>(gguf_.get_i64(key(".hyper_connection.count"), 0));
        cfg_.hc_low_rank =
            static_cast<i32>(gguf_.get_i64(key(".hyper_connection.low_rank"), 0));
        // A count of 1 has nothing to mix. transformers, vLLM and SGLang all
        // raise on it, and a silent 1-lane read would make the mixer an
        // expensive identity.
        if (cfg_.hc_n <= 1 || cfg_.hc_low_rank <= 0) {
            if (err)
                *err = "arch '" + a + "': hyper_connection.count (" +
                       std::to_string(cfg_.hc_n) + ") x low_rank (" +
                       std::to_string(cfg_.hc_low_rank) +
                       ") is not a mixer (count must be > 1, rank > 0)";
            return false;
        }
        cfg_.no_out_norm = aspec.no_out_norm;
        cfg_.gdn_gate_sigmoid = aspec.hyper_connection;
        // `aspec.qsa` says the SCHEMA can index; whether THIS FILE does is
        // proved by its own metadata. A qwen4exp file without an indexer (a
        // development fixture, or a checkpoint exported before the indexer
        // landed) carries no compress ratios and no head geometry, and
        // demanding them would refuse the residual stack it does have. When the
        // keys ARE there the completeness check below is unchanged, so a
        // half-declared indexer is still refused rather than run as if it had
        // none -- that is the direction that decodes wrong text.
        const bool has_indexer =
            gguf_.find(key(".attention.indexer.head_count")) != nullptr ||
            gguf_.find(key(".attention.compress_ratios")) != nullptr;
        cfg_.qsa = aspec.qsa && has_indexer;
        if (cfg_.qsa) {
            cfg_.idx_n_head =
                static_cast<i32>(gguf_.get_i64(key(".attention.indexer.head_count"), 0));
            cfg_.idx_head_dim =
                static_cast<i32>(gguf_.get_i64(key(".attention.indexer.key_length"), 0));
            cfg_.idx_top_k =
                static_cast<i32>(gguf_.get_i64(key(".attention.indexer.top_k"), 0));
            if (cfg_.idx_n_head <= 0 || cfg_.idx_head_dim <= 0 || cfg_.idx_top_k <= 0) {
                if (err)
                    *err = "arch '" + a + "': the QSA indexer metadata is incomplete "
                           "(head_count/key_length/top_k must all be > 0)";
                return false;
            }
            if (const std::vector<i32> *cr =
                    gguf_.get_i32_array(key(".attention.compress_ratios"))) {
                cfg_.qsa_ratio = *cr;
            }
        }
    }
    if (aspec.ple) {
        const std::vector<i32> *layers = gguf_.get_i32_array(key(".ple.layers"));
        if (layers && !layers->empty()) {
            cfg_.ple = true;
            cfg_.ple_layers = *layers;
            cfg_.ple_ngram = static_cast<i32>(gguf_.get_i64(key(".ple.ngram_size"), 0));
            cfg_.ple_heads_per_ngram =
                static_cast<i32>(gguf_.get_i64(key(".ple.heads_per_ngram"), 0));
            cfg_.ple_conv_kernel =
                static_cast<i32>(gguf_.get_i64(key(".ple.conv_kernel"), 0));
            cfg_.ple_head_dim = static_cast<i32>(
                gguf_.get_i64(key(".embedding_length_per_layer_input"), 0));
            cfg_.ple_eos_token =
                static_cast<i32>(gguf_.get_i64(key(".ple.eos_token_id"), -1));
            // All three of these are 64-bit arrays in the REAP file, and two of
            // them genuinely need the width: the offsets reach 300,001,275 and
            // the multipliers are 2^45-scale constants. Reading them through
            // get_i32_array() found nothing and refused a file that declares
            // every one of these keys -- see Gguf::get_i64_array.
            if (!gguf_.get_i64_array(key(".ple.layer_multipliers"),
                                     &cfg_.ple_multipliers))
                cfg_.ple_multipliers.clear();
            // The head ranges are stored as uint64 in the file; the loader keeps
            // them as i64 and refuses a set that does not cover every head.
            if (!gguf_.get_i64_array(key(".ple.head_offsets"),
                                     &cfg_.ple_head_offsets))
                cfg_.ple_head_offsets.clear();
            if (!gguf_.get_i64_array(key(".ple.head_vocab_sizes"),
                                     &cfg_.ple_head_vocab_sizes))
                cfg_.ple_head_vocab_sizes.clear();
            const i32 heads = (cfg_.ple_ngram - 1) * cfg_.ple_heads_per_ngram;
            if (cfg_.ple_ngram < 2 || cfg_.ple_heads_per_ngram <= 0 ||
                cfg_.ple_head_dim <= 0 || cfg_.ple_conv_kernel <= 0 ||
                static_cast<i32>(cfg_.ple_head_offsets.size()) < heads ||
                static_cast<i32>(cfg_.ple_head_vocab_sizes.size()) < heads ||
                static_cast<i32>(cfg_.ple_multipliers.size()) < cfg_.ple_ngram) {
                if (err)
                    *err = "arch '" + a + "': the PLE metadata is incomplete (needs "
                           "ngram>=2, heads_per_ngram, embedding_length_per_layer_"
                           "input, conv_kernel, and one offset/vocab per head)";
                return false;
            }
            for (i32 l : cfg_.ple_layers) {
                if (l < 0 || l >= cfg_.n_layer) {
                    if (err)
                        *err = "arch '" + a + "': PLE layer " + std::to_string(l) +
                               " is outside the " + std::to_string(cfg_.n_layer) +
                               "-layer stack";
                    return false;
                }
            }
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
        if (stream_cur_ && ring_) {
            // Streamed layer: no per-tensor device allocation and no copy at
            // load. The tensor points into the shared ring, and stage_layer()
            // refills that ring from this source before the layer runs. 256-byte
            // alignment keeps every row of every quant block naturally aligned.
            const size_t aligned = (t->n_bytes + 255u) & ~size_t(255u);
            // The slot this layer's parity owns. Fixed at load time, so a
            // layer's addresses never change while the GPU may be reading them.
            q.data = const_cast<u8 *>(slot_base(stream_layer_index_)) +
                     stream_off_;
            stream_off_ += aligned;
            stream_[static_cast<size_t>(stream_layer_index_)].push_back(
                StreamEntry{t, q.data, t->n_bytes});
            stream_bytes_total_ += t->n_bytes;
            q.type = as;
            q.n_in = static_cast<i64>(t->ne[0]);
            q.n_out = t->n_dims >= 2 ? static_cast<i64>(t->ne[1]) : 1;
            if (t->n_dims > 2) q.n_out = t->n_elements / std::max<i64>(1, q.n_in);
            weight_bytes_ += static_cast<i64>(t->n_bytes);
            return q;
        }
        // A resident per-layer weight, carved from the one slab when there is
        // room: the bump keeps the tensors in loader order, which is the order
        // the kernels read them in.
        const size_t slab_need = (t->n_bytes + 255u) & ~size_t(255u);
        if (slab_ && stream_layer_index_ >= 0 &&
            slab_off_ + slab_need <= slab_bytes_) {
            q.data = slab_ + slab_off_;
            slab_off_ += slab_need;
        } else {
            q.data = be.alloc(t->n_bytes);
        }
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

    // Reads an int32 tensor as DATA. The values ARE the data -- an
    // expert-id table keyed by token id -- so they are read as integers and
    // kept on the host, where the route is resolved. Never uploaded: a
    // device copy would be a weight in all but name, which is why
    // upload_weight refuses the int32 format outright.
    auto read_i32_table = [&](const std::string &name, std::vector<i32> &out,
                              i32 &per_token, i32 &n_tokens,
                              std::string *e) -> bool {
        const GgufTensor *t = gguf_.tensor(name);
        if (!t) {
            if (e) *e = "int32 table '" + name + "' is missing";
            return false;
        }
        if (t->type != DType::I32) {
            if (e)
                *e = "table '" + name + "' is " + dtype_name(t->type) +
                     ", expected an int32 table (ggml type 26)";
            return false;
        }
        if (t->n_dims < 1 || t->n_elements <= 0 || !t->data) {
            if (e) *e = "table '" + name + "' is empty";
            return false;
        }
        // n_elements is positive (checked above) and 4 bytes wide, so the
        // widening to size_t is exact; comparing the two types directly was a
        // signed/unsigned mix.
        if (t->n_bytes < static_cast<size_t>(t->n_elements) * 4u) {
            if (e)
                *e = "table '" + name + "' is truncated (" +
                     std::to_string(t->n_bytes) + " bytes for " +
                     std::to_string(t->n_elements) + " int32s)";
            return false;
        }
        // ne[0] is the FASTEST axis in GGUF, so the flat copy is token-major:
        // `per_token` ids per token, `n_tokens` tokens.
        per_token = static_cast<i32>(t->ne[0]);
        n_tokens = t->n_dims >= 2 ? static_cast<i32>(t->ne[1]) : 1;
        if (t->n_dims > 2)
            n_tokens = static_cast<i32>(t->n_elements / std::max<i64>(1, per_token));
        out.resize(static_cast<size_t>(t->n_elements));
        std::memcpy(out.data(), t->data,
                    static_cast<size_t>(t->n_elements) * sizeof(i32));
        return true;
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

    // ---- a weight's NVFP4 global scale, found by name ----------------------
    // The sidecar is `<tensor>.scale` beside `<tensor>.weight`, and it is the
    // one number a consumer cannot get from the quantized bytes. Resolved HERE,
    // once per uploaded weight, so that a call site can neither forget to look
    // for it nor disagree with another call site about where it lives.
    auto with_sidecar_scale = [&](QuantTensor q, const std::string &wname) {
        static const char *kWeight = ".weight";
        const size_t wn = std::strlen(kWeight);
        const std::string base =
            (wname.size() > wn && wname.compare(wname.size() - wn, wn, kWeight) == 0)
                ? wname.substr(0, wname.size() - wn)
                : wname;
        q.gs = one_scale(base + ".scale");
        // The engine scales a scaled weight's INPUT, and that copy has to be as
        // wide as the widest scaled weight's contraction. Recorded here because
        // this is the only place that sees the sidecar at all.
        if (q.gs != 1.0f && q.n_in > max_scaled_in_) max_scaled_in_ = q.n_in;
        return q;
    };

    auto upload_weight = [&](const std::string &name, bool optional,
                             std::string *e) -> QuantTensor {
        const GgufTensor *t = gguf_.tensor(name);
        if (!t) {
            if (!optional && e) *e = "required weight '" + name + "' is missing";
            return {};
        }
        // An int32 tensor is DATA, not a weight. DeepSeek-V4 ships one per
        // hash layer -- `ffn_gate_tid2eid.weight`, [n_expert_used, n_vocab]
        // expert ids -- and the loader reads it as a table. Letting one
        // through here would put a table of ids where a coefficient matrix
        // belongs, and the GEMM would read the ids as numbers: finite,
        // fluent, and not the model. So the format is refused HERE, at the
        // one place that knows the difference between data and a weight.
        if (t->type == DType::I32) {
            if (e)
                *e = "weight '" + name +
                     "' is an int32 table (ggml type 26), not a matmul "
                     "weight; a data tensor has to be read as data";
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
            return with_sidecar_scale(q, name);
        }
        return with_sidecar_scale(upload_raw(t, t->type), name);
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
        // A lookup TABLE, not a matmul: its rows are copied out by index, so a
        // per-tensor scale has no projection to ride and is deliberately not
        // taken here. (A tied file shares this tensor with the head below, and
        // the head's own gs is read there.)
        tok_embd_ = upload_raw(te, te->type);
        if (cfg_.learned_pos) {
            // Learned absolute positions, [n_ctx_train, n_embd], added to the
            // token embedding before the first block. There is no rotary table
            // anywhere in this family, and building one would rotate vectors
            // that already carry their position.
            // OPTIONAL, not required. GPT-2 and Phi-2 always carry the table;
            // MPT carries it on some exports and relies on ALiBi alone on
            // others, and refusing the second kind would reject a complete
            // file. `learned_pos` says the family CAN use a table, and the
            // forward adds it only when the tensor is there.
            const GgufTensor *pe = gguf_.tensor("position_embd.weight");
            if (!pe) {
                // nothing to load; the position signal is ALiBi's
            } else {
            pos_embd_ = upload_raw(pe, pe->type);
            if (pe->n_elements !=
                static_cast<i64>(cfg_.n_ctx_train) * cfg_.n_embd) {
                if (err)
                    *err = "position_embd.weight is " +
                           std::to_string(pe->n_elements) +
                           " elements, expected n_ctx_train*n_embd = " +
                           std::to_string(static_cast<i64>(cfg_.n_ctx_train) *
                                          cfg_.n_embd);
                return false;
            }
            }
        }
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
        out_head_ = with_sidecar_scale(upload_raw(oh, oh->type), "output.weight");
        lp.mark("  output.weight upload");
    }
    if (cfg_.no_out_norm) {
        // The final hyper-connection mixer IS the output norm: the file carries
        // hc_head_norm/down/up instead of output_norm.weight. The norm weight is
        // kept here so the final mix has its gamma; the projector and the wide
        // stream are read by the engine's last mix.
        //
        // Two spellings are in the wild for the same three tensors: the
        // REAP-288 export writes `output_hc_{norm,down,up}`, while the fixtures
        // (and the first file this ran on) use `hc_head_*`. Both are tried in
        // the file's own order -- hc_head first, so nothing that loaded before
        // changes -- and the shape check below decides whether what was found is
        // a mixer at all.
        auto head_name = [&](const char *first, const char *second) {
            return gguf_.tensor(first) ? std::string(first)
                                       : std::string(second);
        };
        const std::string hc_norm = head_name("hc_head_norm.weight",
                                              "output_hc_norm.weight");
        const std::string hc_down = head_name("hc_head_down.weight",
                                              "output_hc_down.weight");
        const std::string hc_up = head_name("hc_head_up.weight",
                                            "output_hc_up.weight");
        hc_head_norm_ = upload_f32(hc_norm);
        std::string he;
        hc_head_down_ = upload_weight(hc_down, false, &he);
        if (!hc_head_down_.present()) {
            if (err) *err = hc_down + " is missing";
            return false;
        }
        hc_head_up_ = upload_weight(hc_up, false, &he);
        if (!hc_head_up_.present()) {
            if (err) *err = hc_up + " is missing";
            return false;
        }
        // The engine's final rmsnorm still runs, with the first lane's gamma:
        // the reference mixes the lanes first, which is the stage that lands
        // with the hyper-connection forward pass.
        out_norm_ = hc_head_norm_;
        if (!out_norm_) {
            if (err) *err = hc_norm + " is missing";
            return false;
        }
        lp.mark("  hc_head upload");
    } else {
        out_norm_ = upload_f32("output_norm.weight");
        lp.mark("  output_norm upload");
        // LFM2 names its final norm `token_embd_norm.weight` and ships no
        // `output_norm` at all. The two are the same tensor under a different
        // name -- one RMSNorm over the last hidden row, with the output head
        // reading its result -- so the fallback is a name, not a graph change.
        // Done by name rather than by arch so a converter that renames it on
        // some other family does not need a table row to be told.
        if (!out_norm_) {
            out_norm_ = upload_f32("token_embd_norm.weight");
            if (out_norm_) lp.mark("  token_embd_norm upload (no output_norm)");
        }
        if (!out_norm_) {
            if (err) *err = "output_norm.weight is missing";
            return false;
        }
        if (cfg_.ln_bias) {
            // The final LayerNorm's bias. Without it the last hidden row keeps
            // its mean, which shifts every logit by a constant per row -- and
            // a constant on the LOGIT axis does not divide out of a softmax
            // unless every row shares it, so this changes the next-token
            // distribution rather than just its scale.
            // Loaded when the file has one, not required. GPT-NeoX and Phi-2
            // carry it; GPT-J is a LayerNorm with a weight and NO bias, so
            // demanding the tensor would refuse a complete file -- and the
            // same distinction already exists for the per-layer norms.
            out_norm_bias_ = upload_f32("output_norm.bias");
            // BLOOM's embedding norm. The file carries it BESIDE the output
            // norm and beside every per-layer norm, so reading the wrong one is
            // a same-shape substitution that still decodes -- which is why it
            // is loaded only under the arch flag that says this family has it.
            if (cfg_.embd_norm) {
                embd_norm_ = upload_f32("token_embd_norm.weight");
                embd_norm_bias_ = upload_f32("token_embd_norm.bias");
                if (!embd_norm_) {
                    if (err)
                        *err = "bloom file has no token_embd_norm.weight "
                               "(this family normalizes the embedding)";
                    return false;
                }
            }
        }
    }

    // The head bias (`output.bias`, Phi-MoE). It belongs OUTSIDE the LayerNorm
    // branch above: Phi-MoE is an RMSNorm family, so a load placed inside
    // `if (cfg_.ln_bias)` never ran and the bias was silently dropped. The A/B
    // that caught it is a file with a nonzero bias whose logits did not move by
    // a single digit. Optional everywhere: no other family writes one.
    if (!out_bias_) out_bias_ = upload_f32("output.bias");

    if (cfg_.ple) {
        const GgufTensor *t = gguf_.tensor("per_layer_token_embd.weight");
        if (!t) {
            if (err) *err = "per_layer_token_embd.weight is missing";
            return false;
        }
        // Rows, not bytes: one row is head_dim values of the file's own type and
        // the gather reads them straight out of the mapping. Uploading this
        // table would be 30+ GiB of VRAM for rows a token touches sixteen of.
        const u64 n_rows = static_cast<u64>(t->ne[1]);
        i64 min_rows = 0;
        for (size_t h = 0; h < cfg_.ple_head_offsets.size(); h++)
            min_rows = std::max<i64>(
                min_rows, cfg_.ple_head_offsets[h] + cfg_.ple_head_vocab_sizes[h]);
        if (static_cast<i64>(n_rows) < min_rows) {
            if (err)
                *err = "per_layer_token_embd has " + std::to_string(n_rows) +
                       " rows, too few for the PLE head ranges (" +
                       std::to_string(min_rows) + ")";
            return false;
        }
        ple_table_ = t->data;
        ple_table_type_ = t->type;
        // BYTES for one row, in the file's own type: the gather reads the row
        // straight out of the mapping, so an f16 or quantized table moves
        // half (or a quarter) as many bytes as an f32 one and the stride has
        // to follow. (This used to be elements per row, which is the same
        // number only for an 8-bit type.)
        ple_row_bytes_ =
            static_cast<u64>(dtype_row_bytes(t->type, cfg_.ple_head_dim));
        KRK_INFO("ple: %llu rows of %d read from the mapping (%llu B/row), "
                 "%d head(s) per token",
                 static_cast<unsigned long long>(n_rows), cfg_.ple_head_dim,
                 static_cast<unsigned long long>(ple_row_bytes_),
                 (cfg_.ple_ngram - 1) * cfg_.ple_heads_per_ngram);
    }

    lp.mark("token_embd + output head");

    // ---- dense weight residency -----------------------------------------
    // Every dense matrix used to be uploaded at load, so a model whose weights
    // did not fit the card filled VRAM to 0 bytes free and left the KV tier
    // one slot per plane -- the 15.9 GiB Qwen3.8-27B run measured 0.00 GiB free
    // and 3.4 tok/s. Here the layers that fit stay resident and the rest are
    // staged through one ring before their layer runs, so the KV cache is sized
    // against the memory that is actually left over.
    //
    // The layer byte totals are read off the file's own tensor directory (every
    // tensor under `blk.N.`), which over-counts only the tiny f32 norms and is
    // therefore a safe upper bound for the ring.
    std::vector<size_t> layer_bytes(
        static_cast<size_t>(std::max<i32>(cfg_.n_layer, 0)), 0);
    for (const GgufTensor &t : gguf_.tensors()) {
        if (t.name.compare(0, 4, "blk.") != 0) continue;
        // Routed-expert banks are NOT loaded here -- ExpertCache pages them --
        // so charging them to a layer's streamed bytes made a 25 GiB MoE look
        // like dense layers that did not fit and streamed 20 layers that did.
        // The shared expert (`_shexp`) is an ordinary dense upload and stays.
        if (t.name.find("_exps") != std::string::npos) continue;
        size_t q = 4;
        i64 lay = 0;
        while (q < t.name.size() && t.name[q] >= '0' && t.name[q] <= '9') {
            lay = lay * 10 + (t.name[q] - '0');
            q++;
        }
        if (q < t.name.size() && t.name[q] == '.' && lay >= 0 &&
            lay < static_cast<i64>(layer_bytes.size()))
            layer_bytes[static_cast<size_t>(lay)] += t.n_bytes;
    }

    // The CPU backend has no device/host split: streaming there would copy host
    // memory to host memory and buy nothing. `weight_budget_ == 0` is automatic.
    const bool can_stream = be.has_device_memory() &&
                            be.device_total_bytes() > 0;
    size_t weight_budget = weight_budget_;
    // An explicit --weight-vram-mb is a promise, not a hint: the second ring slot
    // must fit in whatever headroom it left, never come out of the plan. An
    // automatic budget has the 1 GiB slack below for exactly this.
    const bool budget_is_auto = weight_budget == 0;
    if (can_stream && weight_budget == 0) {
        const size_t total = be.device_total_bytes();
        const size_t freeb = be.device_free_bytes();
        const size_t reserve = krk::vram_reserve_bytes(total);
        // Workspaces, the KV slot and the ring itself are allocated after this
        // point, so leave a gigabyte for them rather than planning to the last
        // byte the way the old path did.
        const size_t slack = 1u << 30;
        weight_budget =
            freeb > reserve + slack ? freeb - reserve - slack : 0;
    }
    // One-hop-ahead staging needs a second ring slot the size of the largest
    // layer, and that comes out of the same VRAM the resident prefix is planned
    // against. Reserve it BEFORE the split so the plan cannot spend it, and drop
    // the double buffer rather than strand the ring when there is too little to
    // reserve. KRK_WEIGHT_AHEAD=0 forces the single-slot arm for an A/B.
    // OFF by default, and the measurement is why: on the 27B the copy of one
    // layer is 9-10 ms over the link while that layer's compute is 1.7 ms, so a
    // two-slot window -- whose cover is exactly one layer -- moves the wait into
    // device idle and leaves the step at 192 -> 197 ms. It wins only where a
    // layer's compute exceeds its copy, which no PCIe-streamed weight set on
    // this card reaches. Kept behind KRK_WEIGHT_AHEAD=1 because it is the right
    // pipeline the day that ratio inverts (a faster link, or a stack whose
    // streamed layers are the compute-heavy ones).
    bool want_ahead = false;
    if (const char *av = std::getenv("KRK_WEIGHT_AHEAD"))
        want_ahead = std::string(av) == "1";
    size_t max_layer_bytes = 0;
    for (size_t b : layer_bytes) max_layer_bytes = std::max(max_layer_bytes, b);
    max_layer_bytes = (max_layer_bytes + 255u) & ~size_t(255u);
    if (can_stream && want_ahead && weight_budget > 0 &&
        weight_budget < (size_t(1) << 60) &&
        weight_budget > 2 * max_layer_bytes) {
        // An automatic budget is planned to keep 1 GiB of slack, and the second
        // slot is paid out of that. An explicit one is not touched: taking a
        // layer's worth of residency away to buy a slot streams one MORE layer
        // per token, which is the opposite of the intended trade.
        if (budget_is_auto) weight_budget -= max_layer_bytes;
        ring_slots_ = 2;
    } else {
        want_ahead = false;
    }

    i32 first_stream = -1;
    if (can_stream && weight_budget > 0) {
        size_t cum = static_cast<size_t>(up_bytes);
        for (i32 l = 0; l < cfg_.n_layer; l++) {
            const size_t lb = layer_bytes[static_cast<size_t>(l)];
            if (cum + lb > weight_budget) {
                first_stream = l;
                break;
            }
            cum += lb;
        }
    }
    if (first_stream >= 0) {
        size_t need = 0;
        for (i32 l = first_stream; l < cfg_.n_layer; l++)
            need = std::max(need, layer_bytes[static_cast<size_t>(l)]);
        need = (need + 255u) & ~size_t(255u);
        // ring_slots_ was decided above. The second slot is the prefetch target:
        // one layer's copy in flight while another layer computes.
        if (ring_slots_ == 2) {
            // alloc() ABORTS on failure, so ask before taking it: a second slot
            // that does not fit must cost speed, not the run.
            const size_t freeb = be.device_free_bytes();
            const size_t floor = krk::vram_reserve_bytes(be.device_total_bytes());
            if (freeb < 2 * need + floor) {
                KRK_WARN("weight residency: %.1f MiB free, no room for a second "
                         "%.1f MiB ring slot -- staging stays one layer deep",
                         static_cast<f64>(freeb) / 1048576.0,
                         static_cast<f64>(need) / 1048576.0);
                ring_slots_ = 1;
            }
        }
        ring_ = be.alloc(need * static_cast<size_t>(ring_slots_));
        stage_host_ = ring_ ? be.alloc_host(need) : nullptr;
        // The spare pinned buffer the next layer is read into while this one's
        // DMA is in flight. Two buffers, alternating; without the second the
        // read waits for the copy to land. Its absence costs the overlap, not
        // the run, so a failed allocation just turns the pipeline off.
        if (const char *pv = std::getenv("KRK_WEIGHT_PREFETCH"))
            prefetch_enabled_ = std::string(pv) != "0";
        if (stage_host_ && prefetch_enabled_) {
            prefetch_host_ = be.alloc_host(need);
            if (!prefetch_host_) {
                prefetch_enabled_ = false;
                KRK_WARN("weight residency: no spare pinned buffer for the read "
                         "pipeline (%.1f MiB) -- reads will wait for copies",
                         static_cast<f64>(need) / 1048576.0);
            }
        }
        read_buf_[0] = stage_host_;
        read_buf_[1] = prefetch_host_;
        if (!ring_ || !stage_host_) {
            KRK_WARN("weight residency: could not allocate a %.1f MiB streaming "
                     "ring; keeping every weight on the device (VRAM pressure "
                     "will decide the KV tier)",
                     static_cast<f64>(need) / 1048576.0);
            if (ring_) be.release(ring_);
            if (stage_host_) be.release_host(stage_host_);
            ring_ = nullptr;
            stage_host_ = nullptr;
            first_stream = -1;
        } else {
            ring_bytes_ = need * static_cast<size_t>(ring_slots_);
            slot_bytes_ = ring_slots_ > 1 ? need : 0;
            layer_streamed_.assign(static_cast<size_t>(cfg_.n_layer), 0);
            stream_.assign(static_cast<size_t>(cfg_.n_layer), std::vector<StreamEntry>());
            for (i32 l = first_stream; l < cfg_.n_layer; l++)
                layer_streamed_[static_cast<size_t>(l)] = 1;
            streamed_layers_ = static_cast<size_t>(cfg_.n_layer - first_stream);
            KRK_INFO("weight residency: %.2f GiB device budget, layer(s) %d..%d "
                     "streamed from the file through a %.1f MiB ring; the rest "
                     "stays resident",
                     static_cast<f64>(weight_budget) / 1073741824.0,
                     first_stream, cfg_.n_layer - 1,
                     static_cast<f64>(need) / 1048576.0);
            if (ring_slots_ > 1)
                KRK_INFO("weight residency: %.1f MiB ring is TWO slots -- one "
                         "layer's copy stays in flight while the layer before it "
                         "computes",
                         static_cast<f64>(ring_bytes_) / 1048576.0);
        }
    } else if (can_stream) {
        size_t total_model = static_cast<size_t>(up_bytes);
        for (size_t b : layer_bytes) total_model += b;
        if (weight_budget >= (size_t(1) << 60))
            KRK_INFO("weight residency: streaming disabled; all %.2f GiB of "
                     "weights resident",
                     static_cast<f64>(total_model) / 1073741824.0);
        else
            KRK_INFO("weight residency: all %.2f GiB of weights resident "
                     "(%.2f GiB budget)",
                     static_cast<f64>(total_model) / 1073741824.0,
                     static_cast<f64>(weight_budget) / 1073741824.0);
    }

    // ---- one slab for the resident weights ---------------------------------
    // Every resident per-layer weight goes into ONE device allocation instead of
    // several hundred. Same bytes, same addresses-per-kernel; the difference the
    // test measures is page locality. KRK_WEIGHT_SLAB=0 restores per-tensor
    // allocations for the A/B.
    {
        // OFF by default: measured, and layout is not the cause. With promotion
        // disabled in both arms -- so the resident set, the streamed set and the
        // policy are identical -- 12.4 GiB of resident weights in ONE slab is
        // 230.4 ms/step against 225.4 ms in ~700 separate allocations, output md5
        // identical. The slab also reserves its full size up front, which costs
        // ~1.9 GiB of promotion headroom (10 promotions become 1) and is why the
        // first A/B looked 20% worse. KRK_WEIGHT_SLAB=1 keeps the arm.
        const char *sv = std::getenv("KRK_WEIGHT_SLAB");
        const bool want_slab = sv && *sv && std::string(sv) != "0";
        if (want_slab && can_stream) {
            size_t want = static_cast<size_t>(up_bytes);
            for (i32 l = 0; l < cfg_.n_layer; l++) {
                if (!layer_streamed_.empty() &&
                    layer_streamed_[static_cast<size_t>(l)])
                    continue;
                want += layer_bytes[static_cast<size_t>(l)];
            }
            // Alignment per tensor is 256 B and the loader rounds each one, so
            // the sum above is a floor: give it 2% and a 16 MiB floor of slack
            // rather than let a rounding difference push tensors back out.
            want = want + want / 50u + (size_t(16) << 20);
            want = (want + 4095u) & ~static_cast<size_t>(4095u);
            slab_ = static_cast<u8 *>(be.alloc(want));
            slab_bytes_ = slab_ ? want : 0;
            slab_off_ = 0;
            KRK_INFO("weight residency: %s slab of %.0f MiB for the resident "
                     "per-layer weights",
                     slab_ ? "one contiguous" : "could not allocate a",
                     static_cast<f64>(slab_bytes_) / 1048576.0);
        }
    }

    // ---- per-layer weights ------------------------------------------------
    // The tensor upload loop: every dense weight the device keeps resident, one
    // [stats ] load line used to cover this and the mapping together.
    KRK_TIME_HOST("load.weights");
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
        stream_cur_ = layer_streamed(l);
        stream_off_ = 0;
        stream_layer_index_ = l;
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
        // Falcon-40B's second pre-attention norm. Optional: 7B has none, and
        // then the FFN's norm is attn_norm, which the fallback below aliases.
        L.attn_norm_2 = upload_f32(blk_key("blk.%d.attn_norm_2.weight", l));
        L.attn_norm_2_bias = upload_f32(blk_key("blk.%d.attn_norm_2.bias", l));
        if (cfg_.ln_bias) {
            // LayerNorm, so each pre-norm has a bias beside its weight. A
            // LayerNorm run without its bias is a different function, not a
            // rounding difference: every row keeps its mean instead of being
            // centred, and the error compounds over eight blocks.
            L.attn_norm_bias = upload_f32(blk_key("blk.%d.attn_norm.bias", l));
            L.ffn_norm_bias = upload_f32(blk_key("blk.%d.ffn_norm.bias", l));
            // Loaded, not required. The layer norms take their bias whenever
            // the file has one: GPT-2 and GPT-NeoX carry both, Phi-2 carries
            // both, and GPT-J -- which is a LayerNorm with weights and no
            // biases anywhere -- carries neither. Requiring them would refuse a
            // complete GPT-J file, and the per-layer norm is where the
            // difference actually lives, since the same `ln_bias` flag also
            // sizes the LayerNorm the forward runs.
        }
        // The two branch post norms. They are NOT the FFN's input norm under
        // another name -- the file carries `ffn_norm` beside them (gemma3) --
        // and the fallback below deliberately does not look at them, because
        // reading `post_attention_norm` as the pre-FFN norm is exactly what
        // this family's graph does not do: gemma normalizes the attention
        // output and THEN adds it, so a loader that aliased the two would run
        // the layer with the right weights in the wrong place.
        if (cfg_.post_branch_norm) {
            L.attn_post_norm = upload_f32(blk_key("blk.%d.post_attention_norm.weight", l));
            L.ffn_post_norm = upload_f32(blk_key("blk.%d.post_ffw_norm.weight", l));
            if (!L.attn_post_norm || !L.ffn_post_norm) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " is missing a branch post norm (post_attention_norm / "
                           "post_ffw_norm), which this architecture's graph runs";
                return false;
            }
        }
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
        // qwen4exp has no plain attn_norm/ffn_norm at all: each sub-block reads
        // the hyper-connection mixer's own grouped norm instead, and requiring
        // one here would refuse a file whose norms are all present under the
        // names the mixer uses.
        if (!cfg_.hc && (!L.attn_norm || !L.ffn_norm)) {
            if (err) *err = "layer " + std::to_string(l) + " is missing its norm weights";
            return false;
        }

        // BitNet's SubLN: an RMSNorm on each sub-block's INTERMEDIATE output,
        // present on that family and on nothing else, so it is read when it is
        // there rather than required -- absent keeps the graph a file already
        // had (see LayerWeights::attn_sub_norm for the two positions, which are
        // not the pre-norms'). Both widths are checked, because each is indexed
        // by a column count that comes from elsewhere in the graph (`aw` for
        // the context, n_ff for the gated intermediate): a wrong width is an
        // out-of-bounds weight read, not a wrong answer.
        {
            const std::string sa = blk_key("blk.%d.attn_sub_norm.weight", l);
            const std::string sf = blk_key("blk.%d.ffn_sub_norm.weight", l);
            L.attn_sub_norm = upload_f32(sa);
            L.ffn_sub_norm = upload_f32(sf);
            const GgufTensor *ta = gguf_.tensor(sa);
            const GgufTensor *tf = gguf_.tensor(sf);
            if (L.attn_sub_norm && ta && ta->n_elements != cfg_.n_embd) {
                if (err)
                    *err = "layer " + std::to_string(l) + " attn_sub_norm has " +
                           std::to_string(ta->n_elements) +
                           " elements; the attention context is " +
                           std::to_string(cfg_.n_embd) + " wide";
                return false;
            }
            if (L.ffn_sub_norm && tf && tf->n_elements != cfg_.n_ff) {
                if (err)
                    *err = "layer " + std::to_string(l) + " ffn_sub_norm has " +
                           std::to_string(tf->n_elements) +
                           " elements; the FFN intermediate is " +
                           std::to_string(cfg_.n_ff) + " wide";
                return false;
            }
        }

        // Attention sinks (deepseek4, gpt-oss): one fp32 scalar per query
        // head, uploaded like every other 1-D weight. Absent is not an
        // error -- it is the whole rest of the model set, and the kernels
        // only add the term when the pointer is present.
        L.sinks = upload_f32(blk_key("blk.%d.attn_sinks.weight", l));

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
            // lfm2moe: the same derivation, off the mixer's own tensor, and it
            // has to decide BOTH ways. Its stack is 3 conv : 1 attention with
            // attention at indices 2, 6, ... and no `attention.recurrent_layers`
            // array, so the interval rule above (every 4th from 1) marks layer 2
            // as a delta net and layer 3 as attention -- both wrong. What the
            // file says unambiguously is which layers carry a shortconv block,
            // so on this family that is the rule: a layer with one is a conv
            // layer, a layer without one is an attention layer.
            if (cfg_.lfm_conv) {
                L.lfm_conv =
                    gguf_.tensor(blk_key("blk.%d.shortconv.in_proj.weight", l)) !=
                    nullptr;
                L.gdn = L.lfm_conv;
            } else if (gguf_.tensor(
                           blk_key("blk.%d.shortconv.in_proj.weight", l))) {
                // An unlisted arch string that still carries the mixer: same
                // rule, one direction only (there is no table row to say the
                // rest of the stack is attention).
                L.gdn = true;
                L.lfm_conv = true;
            }
        }

        if (L.gdn) {
            if (L.lfm_conv) {
                // LFM2.5's short convolution: one projection emitting B, C and
                // the convolved signal's own X, the depthwise taps, and the
                // output projection. There is no q/k/v, no state matrix and no
                // gate chain -- the conv plus the two slices is the whole
                // mixer.
                L.sc_in = need(blk_key("blk.%d.shortconv.in_proj.weight", l));
                L.sc_out = need(blk_key("blk.%d.shortconv.out_proj.weight", l));
                L.sc_conv = upload_conv1d(blk_key("blk.%d.shortconv.conv.weight", l));
                if (!L.sc_conv.present()) {
                    if (err)
                        *err = blk_key("blk.%d.shortconv.conv.weight", l) +
                               " is missing";
                    return false;
                }
                // The taps are read as f32 by the kernel. Every file here
                // stores them that way; a quantised one would be a different
                // dequant path, so it is named instead of guessed at.
                if (L.sc_conv.type != DType::F32) {
                    if (err)
                        *err = blk_key("blk.%d.shortconv.conv.weight", l) +
                               " is not F32; the short-convolution taps are read "
                               "as f32";
                    return false;
                }
                // 3*n_embd of OUTPUT, three equal slices of n_embd: the file's
                // in_proj is [n_embd, 3*n_embd], so n_out is the width the
                // three slices come out of.
                if (L.sc_in.n_in != static_cast<i64>(cfg_.n_embd) ||
                    L.sc_in.n_out != 3 * static_cast<i64>(cfg_.n_embd)) {
                    if (err)
                        *err = blk_key("blk.%d.shortconv.in_proj.weight", l) +
                               " is " + std::to_string(L.sc_in.n_in) + "x" +
                               std::to_string(L.sc_in.n_out) +
                               "; the mixer needs n_embd x 3*n_embd = " +
                               std::to_string(static_cast<i64>(cfg_.n_embd)) +
                               "x" +
                               std::to_string(3 * static_cast<i64>(cfg_.n_embd));
                    return false;
                }
                if (cfg_.ssm_d_conv <= 0) cfg_.ssm_d_conv = 3;
            } else if (L.nemotron_ssm) {
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
                // NVFP4's global scale rides the tensors themselves (gs); it is
                // 1.34e-4 on the Bolt file, against a 4096-wide projection.
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
        } else if (cfg_.mla) {
            // ---- MLA (DeepSeek-V2/V3/V4) --------------------------------
            // Three layouts, told apart by the TENSORS a layer carries and
            // NOT by the arch name: one arch can ship more than one, and a
            // file is what it carries.
            //
            //   Legacy  attn_q + attn_kv_a[_mqa] + attn_kv_b     (V2-Lite)
            //   Split   attn_q[_a/_b] + attn_kv_a[_mqa] +
            //           attn_k_b/attn_v_b                          (the V3 class)
            //   Shared  attn_q_a/attn_q_b + attn_kv +
            //           attn_output_a/attn_output_b                (DeepSeek-V4)
            //
            // They differ in the KV path, which is the whole point of MLA.
            // Legacy and Split compress the KV into a latent and unfold it
            // per head at runtime; Shared keeps ONE latent for every head
            // and folds the head count into a GROUPED output projection
            // instead (n_head heads in n_groups groups, each group through
            // its own rank-o_lora matrix, the groups summed).
            //
            // The old code refused the modern names outright, which was
            // honest but said nothing: the message named the legacy layout
            // for a file that carried neither. Now the layout is read, the
            // tensors are loaded, and the geometry is checked against the
            // file's own metadata. The ATTENTION itself is still not
            // implemented here, and that is refused by the ENGINE
            // (ArchSpec::attention_implemented), not by the loader: the
            // loader's job is to read what is there, and a loader that
            // refuses a file it can read teaches nothing about the file.
            //
            // The compressed-latent projection has TWO spellings in the
            // wild: `attn_kv_a_mqa` (what llama.cpp has called it since the
            // V3 class arrived, and what the V2-Lite exports carry) and
            // `attn_kv_a` (the name this engine's gap map uses). Both are
            // read, and the file says which one it has.
            const bool has_q_a =
                gguf_.tensor(blk_key("blk.%d.attn_q_a.weight", l)) != nullptr;
            const bool has_direct_q =
                gguf_.tensor(blk_key("blk.%d.attn_q.weight", l)) != nullptr;
            const bool has_kv =
                gguf_.tensor(blk_key("blk.%d.attn_kv.weight", l)) != nullptr;
            const bool has_kv_a_mqa =
                gguf_.tensor(blk_key("blk.%d.attn_kv_a_mqa.weight", l)) != nullptr;
            const bool has_k_b =
                gguf_.tensor(blk_key("blk.%d.attn_k_b.weight", l)) != nullptr;
            const bool has_v_b =
                gguf_.tensor(blk_key("blk.%d.attn_v_b.weight", l)) != nullptr;
            const bool has_kv_b =
                gguf_.tensor(blk_key("blk.%d.attn_kv_b.weight", l)) != nullptr;

            MlaLayout layout = MlaLayout::None;
            if (has_kv)
                layout = MlaLayout::Shared;
            else if (has_k_b || has_v_b)
                layout = MlaLayout::Split;
            else if (has_kv_b)
                layout = MlaLayout::Legacy;
            if (layout == MlaLayout::None) {
                e = "layer " + std::to_string(l) +
                    " carries no recognisable MLA attention block (expected "
                    "attn_q/attn_kv_a_mqa/attn_kv_b, attn_q_a/attn_kv_a_mqa/"
                    "attn_k_b/attn_v_b, or attn_q_a/attn_kv/attn_output_a)";
                if (err) *err = e;
                return false;
            }
            L.mla = true;
            // Every layer must agree. A file whose layers disagree is a mix,
            // not a model, and the layer that differs is exactly where a
            // wrong answer would come from.
            if (l == 0) {
                cfg_.mla_layout = layout;
            } else if (cfg_.mla_layout != layout) {
                e = "layer " + std::to_string(l) + " carries the " +
                    mla_layout_name(layout) + " MLA layout but layer 0 carries "
                    "the " + mla_layout_name(cfg_.mla_layout) + " one";
                if (err) *err = e;
                return false;
            }

            // The geometry every layout is checked against, off the file's
            // own metadata: the query trunk's rank, the compressed latent's
            // rank, the per-head key width (nope + rope) and the value width.
            const i64 want_q = static_cast<i64>(cfg_.n_head) *
                               (cfg_.mla_qk_nope + cfg_.mla_qk_rope);
            auto bad = [&](const char *nm, i64 got_in, i64 want_in,
                           i64 got_out, i64 want_out) {
                if (got_in != want_in || got_out != want_out) {
                    e = "layer " + std::to_string(l) + " '" + nm + "' is [" +
                        std::to_string(got_in) + " x " +
                        std::to_string(got_out) + "], metadata wants [" +
                        std::to_string(want_in) + " x " +
                        std::to_string(want_out) + "]";
                    return true;
                }
                return false;
            };
            auto bad_norm = [&](const char *nm, const GgufTensor *t,
                                i64 want) {
                if (!t || t->n_elements == want) return false;
                e = "layer " + std::to_string(l) + " '" + nm + "' has " +
                    std::to_string(t->n_elements) +
                    " elements, metadata wants " + std::to_string(want);
                return true;
            };

            // The query trunk. A file with a q_lora_rank carries attn_q_a
            // (the low-rank projection), its norm and attn_q_b; a file
            // without one projects the query directly (attn_q). The V3 class
            // has both forms -- the query trunk and the KV path are
            // independent choices -- so this is decided by the file, not by
            // the layout.
            if (has_q_a) {
                L.wq_a = need(blk_key("blk.%d.attn_q_a.weight", l));
                L.wq_a_norm =
                    upload_f32(blk_key("blk.%d.attn_q_a_norm.weight", l));
                L.wq_b = need(blk_key("blk.%d.attn_q_b.weight", l));
                const GgufTensor *tqn =
                    gguf_.tensor(blk_key("blk.%d.attn_q_a_norm.weight", l));
                if (!L.wq_a.present() || !L.wq_a_norm || !L.wq_b.present()) {
                    if (e.empty())
                        e = "layer " + std::to_string(l) +
                            " has an incomplete MLA query trunk";
                    if (err) *err = e;
                    return false;
                }
                if (bad("attn_q_a", L.wq_a.n_in, cfg_.n_embd,
                        L.wq_a.n_out, cfg_.mla_q_lora) ||
                    bad("attn_q_b", L.wq_b.n_in, cfg_.mla_q_lora,
                        L.wq_b.n_out, want_q) ||
                    bad_norm("attn_q_a_norm", tqn, cfg_.mla_q_lora)) {
                    if (err) *err = e;
                    return false;
                }
            } else if (has_direct_q) {
                L.wq_direct = need(blk_key("blk.%d.attn_q.weight", l));
                if (!L.wq_direct.present() ||
                    bad("attn_q", L.wq_direct.n_in, cfg_.n_embd,
                        L.wq_direct.n_out, want_q)) {
                    if (e.empty())
                        e = "layer " + std::to_string(l) +
                            " has an incomplete MLA attention block";
                    if (err) *err = e;
                    return false;
                }
            } else {
                e = "layer " + std::to_string(l) +
                    " carries no MLA query projection (attn_q or attn_q_a)";
                if (err) *err = e;
                return false;
            }

            if (layout == MlaLayout::Legacy) {
                // ---- legacy lite: MQA kv, unsplit wkv_b --------------------
                // The compressed latent and the decoupled rope key ride
                // BESIDE each other (kv_lora + rope wide), and the per-head
                // K and V are ONE unsplit table: [kv_lora, n_head*(nope+v)].
                const std::string kv_a =
                    has_kv_a_mqa ? blk_key("blk.%d.attn_kv_a_mqa.weight", l)
                                 : blk_key("blk.%d.attn_kv_a.weight", l);
                L.wkv_a_mqa = need(kv_a);
                L.wkv_a_norm =
                    upload_f32(blk_key("blk.%d.attn_kv_a_norm.weight", l));
                L.wkv_b = need(blk_key("blk.%d.attn_kv_b.weight", l));
                L.wo = need(blk_key("blk.%d.attn_output.weight", l));
                const GgufTensor *tn =
                    gguf_.tensor(blk_key("blk.%d.attn_kv_a_norm.weight", l));
                if (!L.wkv_a_mqa.present() || !L.wkv_a_norm ||
                    !L.wkv_b.present() || !L.wo.present()) {
                    if (e.empty())
                        e = "layer " + std::to_string(l) +
                            " has an incomplete MLA attention block";
                    if (err) *err = e;
                    return false;
                }
                const i64 want_mqa =
                    static_cast<i64>(cfg_.mla_kv_lora + cfg_.mla_qk_rope);
                const i64 want_kvb =
                    static_cast<i64>(cfg_.n_head) *
                    (cfg_.mla_qk_nope + cfg_.mla_v_dim);
                const i64 want_wo =
                    static_cast<i64>(cfg_.n_head) * cfg_.mla_v_dim;
                if (bad("attn_kv_a", L.wkv_a_mqa.n_in, cfg_.n_embd,
                        L.wkv_a_mqa.n_out, want_mqa) ||
                    bad("attn_kv_b", L.wkv_b.n_in, cfg_.mla_kv_lora,
                        L.wkv_b.n_out, want_kvb) ||
                    bad("attn_output", L.wo.n_in, want_wo, L.wo.n_out,
                        cfg_.n_embd) ||
                    bad_norm("attn_kv_a_norm", tn, cfg_.mla_kv_lora)) {
                    if (err) *err = e;
                    return false;
                }
                // ---- the f16 copy the attention kernel reads ----------------
                // attn_kv_b is a BLOCKED quantized table, and MLA reads it as
                // one small [kv_lora, nope+v] matrix per head folded into the
                // score and the value -- not as the `gemm()` shape a quantized
                // weight is uploaded for. Dequantizing it once here keeps the
                // kernel's index arithmetic identical to the file's and the
                // host oracle's, which is what makes a mismatch arithmetic
                // rather than a different weight; the block walk is the same
                // `dequant_row` every other path uses, so there is no second
                // quantizer to be wrong in a slightly different way.
                {
                    const GgufTensor *tb =
                        gguf_.tensor(blk_key("blk.%d.attn_kv_b.weight", l));
                    const i64 n_in = cfg_.mla_kv_lora;
                    const i64 n_out = want_kvb;
                    const size_t nelem =
                        static_cast<size_t>(n_in) * static_cast<size_t>(n_out);
                    const size_t rb = dtype_row_bytes(tb->type, n_in);
                    std::vector<f32> row(static_cast<size_t>(n_in));
                    std::vector<u16> half(nelem);
                    const u8 *src = static_cast<const u8 *>(tb->data);
                    for (i64 r = 0; r < n_out; r++) {
                        // The file is row-major over [n_head*(nope+v)] rows of
                        // kv_lora: `row(m)` is the m-th row of the SAME flat
                        // array the quantized tensor holds, so the staging
                        // copies bytes and nothing else.
                        dequant_row(tb->type, src + static_cast<size_t>(r) * rb,
                                    row.data(), n_in);
                        for (i64 k = 0; k < n_in; k++)
                            half[static_cast<size_t>(r * n_in + k)] =
                                fp32_to_fp16(row[static_cast<size_t>(k)]);
                    }
                    void *dst = be.alloc(nelem * 2);
                    if (!dst) {
                        if (err)
                            *err = "layer " + std::to_string(l) +
                                   ": could not allocate " +
                                   std::to_string(nelem * 2 / 1048576u) +
                                   " MiB for the MLA f16 staging";
                        return false;
                    }
                    upload_timed(dst, half.data(), nelem * 2);
                    L.wkv_b16.data = dst;
                    L.wkv_b16.type = DType::F16;
                    L.wkv_b16.n_in = n_in;
                    L.wkv_b16.n_out = n_out;
                }
            } else if (layout == MlaLayout::Split) {
                // ---- V3 class: a COMPRESSED latent, and the per-head K and
                // V as two separate tables ----------------------------------
                // The latent is compressed (kv_lora wide) and the decoupled
                // rope key rides BESIDE it, so the projection is
                // [n_embd, kv_lora + rope] and the norm covers the latent
                // ALONE. attn_k_b is [nope, kv_lora, n_head] and attn_v_b is
                // [kv_lora, v_dim, n_head] -- both three-dimensional, one
                // matrix per head, and in both the contraction index is ne[0]
                // (the K table folds the query's nope part, the V table folds
                // the latent).
                const std::string kv_a =
                    has_kv_a_mqa ? blk_key("blk.%d.attn_kv_a_mqa.weight", l)
                                 : blk_key("blk.%d.attn_kv_a.weight", l);
                L.wkv_a = need(kv_a);
                L.wkv_norm = upload_f32(blk_key("blk.%d.attn_kv_a_norm.weight", l));
                L.wk_b = need(blk_key("blk.%d.attn_k_b.weight", l));
                L.wv_b = need(blk_key("blk.%d.attn_v_b.weight", l));
                L.wo = need(blk_key("blk.%d.attn_output.weight", l));
                const GgufTensor *tkn =
                    gguf_.tensor(blk_key("blk.%d.attn_kv_a_norm.weight", l));
                if (!L.wkv_a.present() || !L.wkv_norm || !L.wk_b.present() ||
                    !L.wv_b.present() || !L.wo.present()) {
                    if (e.empty())
                        e = "layer " + std::to_string(l) +
                            " has an incomplete split MLA attention block";
                    if (err) *err = e;
                    return false;
                }
                const i64 want_kva =
                    static_cast<i64>(cfg_.mla_kv_lora + cfg_.mla_qk_rope);
                const i64 want_kb =
                    static_cast<i64>(cfg_.mla_kv_lora) * cfg_.n_head;
                const i64 want_vb =
                    static_cast<i64>(cfg_.mla_v_dim) * cfg_.n_head;
                const i64 want_wo =
                    static_cast<i64>(cfg_.n_head) * cfg_.mla_v_dim;
                if (bad("attn_kv_a", L.wkv_a.n_in, cfg_.n_embd,
                        L.wkv_a.n_out, want_kva) ||
                    bad("attn_k_b", L.wk_b.n_in, cfg_.mla_qk_nope,
                        L.wk_b.n_out, want_kb) ||
                    bad("attn_v_b", L.wv_b.n_in, cfg_.mla_kv_lora,
                        L.wv_b.n_out, want_vb) ||
                    bad("attn_output", L.wo.n_in, want_wo, L.wo.n_out,
                        cfg_.n_embd) ||
                    bad_norm("attn_kv_a_norm", tkn, cfg_.mla_kv_lora)) {
                    if (err) *err = e;
                    return false;
                }
            } else {
                // ---- V4: ONE shared latent, a GROUPED output projection ----
                // The latent is not compressed per head: one row of
                // key_length serves every head, and the rope is its TAIL
                // (the last rope_dim of it), so the norm covers the whole
                // latent. The head count is folded into the output
                // projection instead: attn_output_a is
                // [n_head*key_length/o_groups, o_lora_rank*o_groups] in the
                // file and attn_output_b is
                // [o_lora_rank*o_groups, n_embd] -- the groups are summed on
                // the way out.
                if (cfg_.dsv4_o_groups <= 0 || cfg_.dsv4_o_lora <= 0) {
                    e = "layer " + std::to_string(l) +
                        ": the shared-latent MLA layout needs "
                        "attention.output_group_count and "
                        "attention.output_lora_rank, and the file has none";
                    if (err) *err = e;
                    return false;
                }
                L.wkv = need(blk_key("blk.%d.attn_kv.weight", l));
                L.wkv_norm = upload_f32(blk_key("blk.%d.attn_kv_a_norm.weight", l));
                L.wo_a = need(blk_key("blk.%d.attn_output_a.weight", l));
                L.wo_b = need(blk_key("blk.%d.attn_output_b.weight", l));
                const GgufTensor *tkn =
                    gguf_.tensor(blk_key("blk.%d.attn_kv_a_norm.weight", l));
                if (!L.wkv.present() || !L.wkv_norm || !L.wo_a.present() ||
                    !L.wo_b.present()) {
                    if (e.empty())
                        e = "layer " + std::to_string(l) +
                            " has an incomplete shared-latent MLA attention "
                            "block";
                    if (err) *err = e;
                    return false;
                }
                // The shared latent is key_length wide: nope + rope, the rope
                // being the tail.
                const i64 want_latent =
                    static_cast<i64>(cfg_.mla_qk_nope + cfg_.mla_qk_rope);
                const i64 want_woa_in =
                    static_cast<i64>(cfg_.n_head) * want_latent /
                    cfg_.dsv4_o_groups;
                const i64 want_woa_out =
                    static_cast<i64>(cfg_.dsv4_o_lora) * cfg_.dsv4_o_groups;
                if (bad("attn_kv", L.wkv.n_in, cfg_.n_embd,
                        L.wkv.n_out, want_latent) ||
                    bad("attn_output_a", L.wo_a.n_in, want_woa_in,
                        L.wo_a.n_out, want_woa_out) ||
                    bad("attn_output_b", L.wo_b.n_in, want_woa_out,
                        L.wo_b.n_out, cfg_.n_embd) ||
                    bad_norm("attn_kv_a_norm", tkn, want_latent)) {
                    if (err) *err = e;
                    return false;
                }

                // ---- the f16 staging the shared-latent ops read ------------
                // Every table this layout's kernels touch is staged here, once,
                // for `wkv_b16`'s reason above: the grouped output is addressed
                // as per-group slices (see LayerWeights::wo_a16) and the host
                // oracle the device is judged against reads f16. The query
                // trunk is staged only when the file HAS one -- inventing a
                // tensor is not the loader's job, and a shared file with no
                // trunk is a refusal for the ENGINE to make.
                i64 stage_bytes = 0;
                auto stage16 = [&](const char *what, QuantTensor &dst,
                                   const std::string &tname, i64 n_in,
                                   i64 n_out) {
                    const GgufTensor *t = gguf_.tensor(tname);
                    if (!t) {
                        e = "layer " + std::to_string(l) +
                            ": the shared-latent path stages '" +
                            std::string(what) + "' as f16 and the file has no "
                            "such tensor";
                        return false;
                    }
                    const size_t nelem = static_cast<size_t>(n_in) *
                                         static_cast<size_t>(n_out);
                    const size_t rb = dtype_row_bytes(t->type, n_in);
                    std::vector<f32> row(static_cast<size_t>(n_in));
                    std::vector<u16> half(nelem);
                    const u8 *src = static_cast<const u8 *>(t->data);
                    for (i64 r = 0; r < n_out; r++) {
                        dequant_row(t->type, src + static_cast<size_t>(r) * rb,
                                    row.data(), n_in);
                        for (i64 i = 0; i < n_in; i++)
                            half[static_cast<size_t>(r * n_in + i)] =
                                fp32_to_fp16(row[static_cast<size_t>(i)]);
                    }
                    void *dev = be.alloc(nelem * 2);
                    if (!dev) {
                        e = "layer " + std::to_string(l) +
                            ": could not allocate " +
                            std::to_string(nelem * 2 / 1048576u) +
                            " MiB for the '" + std::string(what) +
                            "' f16 staging";
                        return false;
                    }
                    upload_timed(dev, half.data(), nelem * 2);
                    dst.data = dev;
                    dst.type = DType::F16;
                    dst.n_in = n_in;
                    dst.n_out = n_out;
                    stage_bytes += static_cast<i64>(nelem * 2);
                    return true;
                };
                const i64 q_head = static_cast<i64>(cfg_.n_head) * want_latent;
                bool staged =
                    stage16("attn_kv", L.wkv16,
                            blk_key("blk.%d.attn_kv.weight", l), cfg_.n_embd,
                            want_latent) &&
                    stage16("attn_output_a", L.wo_a16,
                            blk_key("blk.%d.attn_output_a.weight", l),
                            want_woa_in, want_woa_out) &&
                    stage16("attn_output_b", L.wo_b16,
                            blk_key("blk.%d.attn_output_b.weight", l),
                            want_woa_out, cfg_.n_embd);
                if (staged && has_q_a)
                    staged = stage16("attn_q_a", L.wq_a16,
                                     blk_key("blk.%d.attn_q_a.weight", l),
                                     cfg_.n_embd, cfg_.mla_q_lora) &&
                             stage16("attn_q_b", L.wq_b16,
                                     blk_key("blk.%d.attn_q_b.weight", l),
                                     cfg_.mla_q_lora, q_head);
                if (!staged) {
                    if (err) *err = e;
                    return false;
                }
                // Reported, not hidden: this is VRAM the expert tier cannot
                // reclaim, so a 10 GB budget has to count it.
                if (l == 0 && std::getenv("KRK_PHASE"))
                    std::fprintf(stderr,
                                 "[load ] %-24s %7.3f MiB/layer\n",
                                 "mla f16 staging",
                                 static_cast<f64>(stage_bytes) / 1048576.0);
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
            // The flag says the SCHEMA has a fused projection; whether this
            // file uses one is a fact about the file. Keying off the tensor
            // lets a mixed fixture be loaded, and lets a file that carries both
            // shapes take the one it actually has.
            const bool has_fused_qkv =
                gguf_.tensor(blk_key("blk.%d.attn_qkv.weight", l)) != nullptr;
            if (cfg_.gpt2_style || (cfg_.fused_qkv && has_fused_qkv)) {
                // One projection whose output row is Q | K | V -- three
                // contiguous blocks, so the split is a stride and not a gather.
                // The block widths differ by family (see phi3_spec).
                // Q | K | V as three contiguous n_embd blocks, with one
                // 3*n_embd bias covering all of them. The file carries no
                // separate q/k/v tensors at all, so this is not an
                // optimisation here -- it is the only shape that exists.
                L.wqkv_fused = need(blk_key("blk.%d.attn_qkv.weight", l));
                L.qkv_bias = upload_f32(blk_key("blk.%d.attn_qkv.bias", l));
                if (!L.wqkv_fused.present()) {
                    if (e.empty())
                        e = "layer " + std::to_string(l) +
                            " is missing its fused attn_qkv projection";
                    if (err) *err = e;
                    return false;
                }
                // The fused weight must be exactly three embeddings wide, or
                // the Q|K|V split below reads past its own row and every head
                // after the first is silently somebody else's. Cheap to check,
                // and a wrong-width file cannot be rescued downstream.
                const i64 kv_wide =
                    static_cast<i64>(cfg_.n_head_kv) * cfg_.head_dim;
                const i64 qkv_wide =
                    (cfg_.fused_qkv && !cfg_.gpt2_style)
                        ? static_cast<i64>(cfg_.n_embd) + 2 * kv_wide
                        : 3 * static_cast<i64>(cfg_.n_embd);
                if (L.wqkv_fused.n_out != qkv_wide) {
                    if (err)
                        *err = "layer " + std::to_string(l) +
                               " attn_qkv is " +
                               std::to_string(L.wqkv_fused.n_out) +
                               " wide, expected Q + 2*(n_head_kv*head_dim) = " +
                               std::to_string(qkv_wide);
                    return false;
                }
            } else {
                L.wq = need(blk_key("blk.%d.attn_q.weight", l));
                L.wk = need(blk_key("blk.%d.attn_k.weight", l));
                L.wv = need(blk_key("blk.%d.attn_v.weight", l));
            }
            L.wo = need(blk_key("blk.%d.attn_output.weight", l));
            // Attention-output bias (gpt-oss). One f32 per output channel,
            // added to the projection's result before the residual. Absent on
            // every other family in the collection, and a missing bias and a
            // model without one look the same from here -- which is why it was
            // never noticed that this tensor existed and was ignored.
            L.wo_bias = upload_f32(blk_key("blk.%d.attn_output.bias", l));
            // The plain family's MLP biases. GPT-NeoX carries both, GPT-J
            // neither, and a missing tensor uploads as null with both
            // applications guarded -- so one flag covers both files.
            if (cfg_.ln_bias) {
                L.wup_bias = upload_f32(blk_key("blk.%d.ffn_up.bias", l));
                L.wdown_bias = upload_f32(blk_key("blk.%d.ffn_down.bias", l));
            }
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

        // ---- qwen4exp: the two hyper-connection mixers ---------------------
        if (cfg_.hc) {
            L.hc_attn_norm = upload_f32(blk_key("blk.%d.hc_attn_norm.weight", l));
            L.hc_attn_down = need(blk_key("blk.%d.hc_attn_down.weight", l));
            L.hc_attn_up = need(blk_key("blk.%d.hc_attn_up.weight", l));
            L.hc_attn_inject = need(blk_key("blk.%d.hc_attn_inject.weight", l));
            L.hc_ffn_norm = upload_f32(blk_key("blk.%d.hc_ffn_norm.weight", l));
            L.hc_ffn_down = need(blk_key("blk.%d.hc_ffn_down.weight", l));
            L.hc_ffn_up = need(blk_key("blk.%d.hc_ffn_up.weight", l));
            L.hc_ffn_inject = need(blk_key("blk.%d.hc_ffn_inject.weight", l));
            const i64 wide = static_cast<i64>(cfg_.n_embd) * cfg_.hc_n;
            if (!L.hc_attn_norm || !L.hc_ffn_norm ||
                L.hc_attn_down.n_in != wide || L.hc_attn_down.n_out != cfg_.hc_low_rank ||
                L.hc_attn_up.n_in != cfg_.hc_low_rank || L.hc_attn_up.n_out != wide ||
                L.hc_attn_inject.n_in != wide || L.hc_attn_inject.n_out != cfg_.hc_n ||
                L.hc_ffn_down.n_in != wide || L.hc_ffn_down.n_out != cfg_.hc_low_rank ||
                L.hc_ffn_up.n_in != cfg_.hc_low_rank || L.hc_ffn_up.n_out != wide ||
                L.hc_ffn_inject.n_in != wide || L.hc_ffn_inject.n_out != cfg_.hc_n) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           ": hyper-connection tensors disagree with " +
                           std::to_string(wide) + " -> " +
                           std::to_string(cfg_.hc_low_rank) + " -> " +
                           std::to_string(wide) + " (" + std::to_string(cfg_.hc_n) +
                           " lanes)";
                return false;
            }
        }
        // A layer's QSA indexer is keyed on its own compress ratio.
        if (cfg_.qsa && l < static_cast<i32>(cfg_.qsa_ratio.size()))
            L.qsa_ratio = cfg_.qsa_ratio[static_cast<size_t>(l)];
        if (L.qsa_ratio > 0) {
            L.idx_q_proj = need(blk_key("blk.%d.indexer.q_proj.weight", l));
            L.idx_k_proj = need(blk_key("blk.%d.indexer.k_proj.weight", l));
            L.idx_q_norm = upload_f32(blk_key("blk.%d.indexer.q_norm.weight", l));
            L.idx_k_norm = upload_f32(blk_key("blk.%d.indexer.k_norm.weight", l));
            // Host copies for the selection, which norms the pooled keys and
            // the indexer queries itself (dequant_row into a vector: the same
            // move the router's own host-side bias makes).
            for (int kk = 0; kk < 2; kk++) {
                const GgufTensor *nt = gguf_.tensor(
                    blk_key(kk == 0 ? "blk.%d.indexer.q_norm.weight"
                                    : "blk.%d.indexer.k_norm.weight", l));
                if (!nt) continue;
                std::vector<f32> &dst = kk == 0 ? L.idx_q_norm_h : L.idx_k_norm_h;
                dst.assign(static_cast<size_t>(nt->n_elements), 0.0f);
                dequant_row(nt->type, nt->data, dst.data(), nt->n_elements);
            }
            if (!L.idx_q_proj.present() || !L.idx_k_proj.present() ||
                !L.idx_q_norm || !L.idx_k_norm) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " declares a compress ratio (" +
                           std::to_string(L.qsa_ratio) +
                           ") but carries no QSA indexer tensors";
                return false;
            }
        }
        if (cfg_.ple) {
            for (i32 pl : cfg_.ple_layers) {
                if (pl != l) continue;
                L.ple = true;
                L.ple_key = need(blk_key("blk.%d.ple_key.weight", l));
                L.ple_value = need(blk_key("blk.%d.ple_value.weight", l));
                L.ple_conv1d = upload_conv1d(blk_key("blk.%d.ple_conv1d.weight", l));
                L.ple_norm_key = upload_f32(blk_key("blk.%d.ple_norm_key.weight", l));
                L.ple_norm_query = upload_f32(blk_key("blk.%d.ple_norm_query.weight", l));
                L.ple_norm_conv = upload_f32(blk_key("blk.%d.ple_norm_conv.weight", l));
                if (!L.ple_key.present() || !L.ple_value.present() ||
                    !L.ple_conv1d.present() || !L.ple_norm_key ||
                    !L.ple_norm_query || !L.ple_norm_conv) {
                    if (err)
                        *err = "layer " + std::to_string(l) +
                               " is a PLE layer but is missing one of its tensors";
                    return false;
                }
            }
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
            // GPT-2's MLP is up -> gelu -> down. There is no gate tensor, and
            // asking for one would fail the load on a file that is complete;
            // reading ffn_up as a gate instead would halve the intermediate
            // width and run the wrong activation over it.
            const bool ffn_fused =
                cfg_.fused_gate_up &&
                gguf_.tensor(blk_key("blk.%d.ffn_gate.weight", l)) == nullptr;
            if (ffn_fused) {
                // gate | up in one row. There is no ffn_gate tensor to read;
                // treating ffn_up as the gate instead -- what happens if this
                // flag is missed -- halves the intermediate width and gates the
                // wrong operand, which is a wrong answer rather than a failure.
                L.wup_fused = need(blk_key("blk.%d.ffn_up.weight", l));
                const i64 want = 2 * static_cast<i64>(cfg_.n_ff);
                if (L.wup_fused.n_out != want) {
                    if (err)
                        *err = "layer " + std::to_string(l) + " ffn_up is " +
                               std::to_string(L.wup_fused.n_out) +
                               " wide, expected gate|up = " +
                               std::to_string(want);
                    return false;
                }
            } else {
                if (!cfg_.no_gate_ffn)
                    L.wgate = need(blk_key("blk.%d.ffn_gate.weight", l));
                L.wup = need(blk_key("blk.%d.ffn_up.weight", l));
            }
            L.wdown = need(blk_key("blk.%d.ffn_down.weight", l));
        } else if (L.moe) {
            // The router is small and always resident. The routed experts are
            // only *described* here — nothing is uploaded, which is what keeps
            // a 235B-A22B model from needing 235B of VRAM at load time.
            L.router = need(blk_key("blk.%d.ffn_gate_inp.weight", l));
            // DeepSeek-V4's HASH ROUTE: the first `hash_layer_count` layers do
            // not pick their experts from the router at all. The file carries
            // a table of expert ids per TOKEN ID, and a token's route is a
            // lookup in it. The table is an int32 tensor
            // ([n_expert_used, n_vocab] ids), which is DATA, not a weight: it
            // is read as integers, on the host, and never multiplied.
            // Reading it as a weight would be wrong twice over -- the format
            // is not one the device can dequantize, and the numbers are ids,
            // not coefficients.
            if (cfg_.dsv4 && l < cfg_.dsv4_hash_layers) {
                i32 per_token = 0, n_tokens = 0;
                std::string te;
                if (!read_i32_table(blk_key("blk.%d.ffn_gate_tid2eid.weight", l),
                                    L.tid2eid, per_token, n_tokens, &te)) {
                    if (err) *err = te;
                    return false;
                }
                L.tid2eid_per_token = per_token;
                L.tid2eid_n_tokens = n_tokens;
                // The geometry is the file's own: one id per selected expert,
                // one column per token id. A table of another shape is another
                // route, and looking it up anyway would send a token to an
                // expert the file never named.
                if (per_token != cfg_.n_expert_used || n_tokens != cfg_.n_vocab) {
                    if (err)
                        *err = "layer " + std::to_string(l) +
                               ": ffn_gate_tid2eid is [" + std::to_string(per_token) +
                               " x " + std::to_string(n_tokens) +
                               "], metadata wants [" +
                               std::to_string(cfg_.n_expert_used) + " x " +
                               std::to_string(cfg_.n_vocab) + "]";
                    return false;
                }
            }
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
                                 "exp_probs_b.bias (no ffn_ prefix) and APPLIED: "
                                 "it shifts selection, not the mixing weights. "
                                 "Builds before this loader accepted only the "
                                 "prefixed ffn_exp_probs_b.bias spelling and "
                                 "silently ran such a model's expert selection "
                                 "un-biased");
                    }
                    break;
                }
                if (L.router_bias) cfg_.router_bias = true;
            }
            // llama.cpp's spelling of the same idea, and a different
            // arithmetic: `ffn_gate_inp.bias` is added to the router's LOGITS,
            // before the softmax (`build_moe_ffn`: the bias is folded into
            // `logits`, and the SOFTMAX_WEIGHT gating func then does
            // `probs = logits`). Adding it after the softmax instead -- the
            // laguna reading -- selects different experts and mixes them with
            // different weights, with nothing in either shape to show for it.
            // gpt-oss is the family that carries this one.
            if (!L.router_bias) {
                if (const GgufTensor *bt =
                        gguf_.tensor(blk_key("blk.%d.ffn_gate_inp.bias", l))) {
                    const i64 n = bt->n_elements;
                    f32 *host = static_cast<f32 *>(
                        host_alloc(static_cast<size_t>(n) * sizeof(f32)));
                    dequant_row(bt->type, bt->data, host, n);
                    L.router_bias = host;
                    if (!warned_bias) {
                        warned_bias = true;
                        KRK_INFO("router bias: ffn_gate_inp.bias, added to the "
                                 "logits before the softmax (llama.cpp's "
                                 "convention), %lld experts",
                                 static_cast<long long>(n));
                    }
                }
                if (L.router_bias) {
                    cfg_.router_bias = true;
                    cfg_.router_bias_pre = true;
                }
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
            // Per-expert FFN biases (gpt-oss): [n_expert, n_ff] on gate and up,
            // [n_expert, n_embd] on down. They belong to the expert, not to the
            // token, so the expert group's rows all get the same vector -- the
            // shape `add_bias_cols` already has. Device-resident and loaded up
            // front: the routed path touches them on every token, and a
            // per-expert fetch would put a host round trip inside the group.
            L.exp_gate_bias = upload_f32(blk_key("blk.%d.ffn_gate_exps.bias", l));
            L.exp_up_bias = upload_f32(blk_key("blk.%d.ffn_up_exps.bias", l));
            L.exp_down_bias = upload_f32(blk_key("blk.%d.ffn_down_exps.bias", l));
            if (L.exp_gate_bias || L.exp_up_bias || L.exp_down_bias)
                KRK_INFO("layer %d: per-expert FFN biases loaded (gate=%d up=%d "
                         "down=%d)",
                         l, L.exp_gate_bias != nullptr, L.exp_up_bias != nullptr,
                         L.exp_down_bias != nullptr);
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
            }
            // Qwen3.5 gates the shared expert with a per-token scalar built from a
            // 1-D row; the Qwen2/Qwen3/nemotron schema has no such vector.
            L.shexp_inp_gate =
                upload_weight(blk_key("blk.%d.ffn_gate_inp_shexp.weight", l), true, &e);
            // ---- LatentMoE: the routed experts' latent ---------------------
            // Nemotron 3.5 keeps the routed bank in a narrow space: this pair
            // projects n_embd -> n_latent and back, and the experts are
            // [n_latent x n_ff_exp x n_expert] rather than n_embd wide. The
            // shared expert above stays full width -- it is a different tensor
            // set for exactly that reason. Both halves or neither: one of the
            // pair describes a block whose routed leg has no input or no exit,
            // and the geometry check below refuses it rather than running the
            // experts on the hidden state and clipping every element.
            if (gguf_.tensor(blk_key("blk.%d.ffn_latent_down.weight", l)) ||
                gguf_.tensor(blk_key("blk.%d.ffn_latent_up.weight", l))) {
                L.latent_down =
                    upload_weight(blk_key("blk.%d.ffn_latent_down.weight", l), true, &e);
                L.latent_up =
                    upload_weight(blk_key("blk.%d.ffn_latent_up.weight", l), true, &e);
            }
        }
        if (!e.empty()) {
            if (err) *err = e;
            return false;
        }

        // geometric sanity: the tensor schema must match the metadata
        // LFM2.5's mixer is checked where it is LOADED instead (three equal
        // n_embd slices, an F32 tap row, an n_embd output projection): it has
        // no ssm.* geometry to agree with, and every check below would compare
        // an absent tensor's default against a width this stack never sets.
        if (L.gdn && !L.lfm_conv) {
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
        // GPT-2 has no gate tensor: its intermediate width is ffn_up's output.
        // Checking wgate here would fail every complete gpt2 file, because an
        // absent gate reads as width 0 rather than as "no gate".
        // Three shapes reach here, and each says the width from a different
        // tensor: gpt2 has ffn_up alone, every split family has ffn_gate beside
        // it, and Phi-3 has neither -- its gate and up are the two halves of
        // ffn_up, which the load above already checked is 2*n_ff wide.
        const bool ffn_width_ok =
            L.wup_fused.present()
                ? (L.wup_fused.n_out == 2 * static_cast<i64>(cfg_.n_ff) &&
                   L.wdown.n_in == cfg_.n_ff)
                : cfg_.no_gate_ffn
                      ? (L.wup.n_out == cfg_.n_ff && L.wdown.n_in == cfg_.n_ff)
                      : (L.wgate.n_out == cfg_.n_ff &&
                         L.wdown.n_in == cfg_.n_ff);
        if (!L.moe && has_dense_ffn && !ffn_width_ok) {
            if (err)
                *err = "layer " + std::to_string(l) + " FFN tensors disagree with n_ff";
            return false;
        }
        if (L.moe) {
            // ---- how many experts THIS layer has --------------------------
            // `expert_count` is the model's count; a REAP export prunes each
            // layer independently and records what each layer KEPT in
            // `reap.layer.keep_count` (one entry per block).
            // `reap.layer.expert_count` is the count BEFORE pruning -- 256 on
            // the V4 file here, against 160 in every bank -- so reading that
            // one would trade one wrong expectation for another. Where
            // keep_count is present it is this layer's truth, and the router
            // and the routed bank are both checked against it, because the
            // router is 160 wide to match the 160-expert bank. Without the key
            // the expert_count check stands: a pruned bank and a corrupt one
            // look the same from the tensors alone.
            const std::vector<i32> *reap_kept =
                gguf_.get_i32_array("reap.layer.keep_count");
            // ---- routed-expert geometry (per-layer) ---------------------
            // The expert tensors are only *described* here so the cache can know
            // per-expert byte boundaries; nothing is uploaded at load time.
            //
            // Nemotron-style MoE (nemotron_h_moe) does not carry ffn_gate_exps,
            // so the geometry is read off the up tensor instead. The sanity that
            // matters for the forward path's intermediate workspace is the
            // routed n_ff_exp; the shared-expert sanity is the one that gates the
            // shared branch.
            //
            // This block now runs BEFORE the router check on purpose: the bank's
            // ne[2] is this layer's expert count, and the router has to match
            // IT. `expert_count` is a CAP, not an equality -- the packed hybrid
            // merges (Nemotron Bolt 74B-A9B) carry 240/304/368/432/496 experts
            // across 40 MoE blocks under one `expert_count` of 496, so checking
            // every router against the cap refuses a file whose tensors agree
            // with each other perfectly.
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
            // A REAP export prunes each layer independently and records what
            // each layer KEPT; where it is present it is the expectation. The
            // bank is the authority otherwise.
            const i32 want_expert =
                (reap_kept && l < static_cast<i32>(reap_kept->size()) &&
                 (*reap_kept)[l] > 0)
                    ? (*reap_kept)[l]
                    : (L.experts.n_expert > 0 ? L.experts.n_expert
                                              : cfg_.n_expert);
            // ---- router geometry ------------------------------------------
            // The router must agree with this layer's own bank: a 240-wide
            // router over a 240-expert bank routes correctly whatever the cap
            // says, and a 496-wide router over it would index experts that do
            // not exist.
            if (L.router.n_out != want_expert || L.router.n_in != cfg_.n_embd) {
                if (err)
                    *err = "layer " + std::to_string(l) + " router is [" +
                           std::to_string(L.router.n_in) + " x " +
                           std::to_string(L.router.n_out) +
                           "], this layer's routed bank is [" +
                           std::to_string(cfg_.n_embd) + " x " +
                           std::to_string(want_expert) + "]";
                return false;
            }
            // The cap still binds: a layer may be narrower than the declared
            // maximum (that is the packed-merge shape) but never wider, because
            // every router row, plan buffer and dispatch slot is sized by it.
            if (want_expert > cfg_.n_expert) {
                if (err)
                    *err = "layer " + std::to_string(l) + " has " +
                           std::to_string(want_expert) +
                           " experts, more than the declared expert_count " +
                           std::to_string(cfg_.n_expert);
                return false;
            }
            if (L.experts.n_ff_exp > cfg_.n_ff_exp) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " routed-expert width " +
                           std::to_string(L.experts.n_ff_exp) +
                           " exceeds the declared expert_feed_forward_length " +
                           std::to_string(cfg_.n_ff_exp);
                return false;
            }
            if (L.shexp_gate.present() && L.shexp_gate.n_out != cfg_.n_ff_shexp) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " shared-expert width disagrees with the metadata";
                return false;
            }
            // ---- LatentMoE geometry (per layer) ---------------------------
            // The routed bank's input width is `L.experts.n_embd`. On most
            // families that IS n_embd, and the block has no latent pair. On
            // Nemotron 3.5 it is a narrow latent -- 1024 against a 4096-wide
            // hidden state on Bolt 74B -- and the layer must carry the pair that
            // gets there and back. Feeding the hidden state to a 1024-wide bank
            // is not a slow answer, it is a wrong one: every row read lands on
            // the wrong stride and the routed sum clips to garbage.
            if (L.latent_down.present() != L.latent_up.present()) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " carries only one of ffn_latent_down / "
                           "ffn_latent_up; the routed leg needs both or "
                           "neither";
                return false;
            }
            if (L.latent_moe()) {
                const i64 ew = L.experts.n_embd;
                if (ew <= 0 || L.latent_down.n_in != cfg_.n_embd ||
                    L.latent_down.n_out != ew || L.latent_up.n_in != ew ||
                    L.latent_up.n_out != cfg_.n_embd) {
                    if (err)
                        *err = "layer " + std::to_string(l) +
                               " latent projection is [" +
                               std::to_string(L.latent_down.n_in) + " x " +
                               std::to_string(L.latent_down.n_out) + "] -> [" +
                               std::to_string(L.latent_up.n_in) + " x " +
                               std::to_string(L.latent_up.n_out) +
                               "], which does not join the hidden state (" +
                               std::to_string(cfg_.n_embd) +
                               ") to this layer's routed bank (" +
                               std::to_string(ew) + ") and back";
                    return false;
                }
            } else if (L.experts.n_embd > 0 && L.experts.n_embd != cfg_.n_embd) {
                if (err)
                    *err = "layer " + std::to_string(l) +
                           " routed bank reads " +
                           std::to_string(L.experts.n_embd) +
                           "-wide rows but the layer carries no latent "
                           "projection to reach it (hidden state is " +
                           std::to_string(cfg_.n_embd) + ")";
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

    // ---- bind streamed entries, and size their promotion -------------------
    // A streamed tensor's QuantTensor already points into the ring, so the
    // engine can use it before this runs; what the binding adds is the way
    // BACK from the entry to the tensor, which a promotion needs to re-point,
    // and the layer's aligned footprint, which is the size of the allocation
    // that replaces the ring for it. Both are facts about the layer the upload
    // could not see (the QuantTensor is returned by value).
    layer_stream_bytes_.assign(static_cast<size_t>(cfg_.n_layer), 0);
    // stream_ is only sized when some layer actually streams; a model that fits
    // entirely leaves it empty, and indexing it here is a crash on every run
    // that needs no streaming at all (measured: all three qwen35moe 35B files).
    for (i32 l = 0; l < cfg_.n_layer && l < static_cast<i32>(stream_.size()); l++) {
        std::vector<StreamEntry> &es = stream_[static_cast<size_t>(l)];
        if (es.empty()) continue;
        std::vector<QuantTensor *> qs = qt_fields(layers_[static_cast<size_t>(l)]);
        size_t bytes = 0;
        for (StreamEntry &e : es) {
            bytes += (e.bytes + 255u) & ~size_t(255u);
            for (QuantTensor *q : qs) {
                if (q->present() && q->data == e.dst) {
                    e.q = q;
                    break;
                }
            }
            if (!e.q)
                KRK_WARN("weight residency: layer %d has a streamed tensor that "
                         "no layer field owns -- it can never be promoted, so it "
                         "will be staged every step", l);
        }
        layer_stream_bytes_[static_cast<size_t>(l)] = bytes;
    }
    // KRK_WEIGHT_PROMOTE=0 restores a fixed resident prefix: every layer past
    // the load-time budget is then staged on every step, whatever VRAM is left.
    if (const char *v = std::getenv("KRK_WEIGHT_PROMOTE"))
        promote_enabled_ = std::string(v) != "0";

    // ---- pinned host mirror of the streamed layers -------------------------
    // A streamed layer's bytes do not change between tokens, so reading them out
    // of the mapping on every pass repeats work that was already done. Measured
    // on the 15.97 GiB Qwen3.8-27B file at --ctx 512: 9 streamed layers per
    // token, 64.70 GiB of re-reads across a 32-token bench, and the read was
    // ~55% of decode -- the mapping is page-cache hot, so the cost is the read
    // path and not the disk.
    //
    // The mirror is one pinned allocation holding every streamed layer at its
    // own offset, filled once here through the same reader a staging pass used.
    // Afterwards a staging pass is a DMA straight out of pinned memory, which is
    // also this card's fastest source (16.4 GB/s pinned against 8.9 GB/s through
    // the narrow-aperture staging ring). The per-token cost becomes the copy
    // and no longer includes the read.
    {
        const auto refused = [] {
            const char *e = std::getenv("KRK_WEIGHT_HOST_CACHE");
            return e && *e && e[0] == '0';
        };
        size_t streamed_total = 0;
        for (size_t l = 0; l < layer_stream_bytes_.size(); l++)
            if (l < stream_.size() && !stream_[l].empty() &&
                layer_streamed(static_cast<i32>(l)))
                streamed_total += layer_stream_bytes_[l];
        // The cap is a ceiling on pinned (non-swappable) RAM, not a tuning knob:
        // 8 GiB by default, KRK_WEIGHT_HOST_CACHE_MB=<n> to change it. A set
        // above the cap keeps the read path, which is slower and still correct.
        size_t cap = size_t(8) << 30;
        if (const char *e = std::getenv("KRK_WEIGHT_HOST_CACHE_MB")) {
            if (*e) {
                const long long mb = std::atoll(e);
                cap = mb > 0 ? static_cast<size_t>(mb) * 1048576u : 0;
            }
        }
        if (!refused() && streamed_total > 0 && streamed_total <= cap)
            build_stream_host_mirror(streamed_total);
        // The one-hop pipeline needs a source that does not move: with the
        // mirror, every copy reads a fixed slice of pinned memory. Without it the
        // per-token read would have to stay valid until its copy landed, which is
        // what the existing two-buffer read path already does, so the double
        // buffer is simply not enabled there.
        ring_ahead_ = ring_slots_ > 1 && stream_host_on_;
        if (ring_slots_ > 1 && !ring_ahead_)
            KRK_WARN("weight residency: no pinned host mirror, so the second "
                     "ring slot cannot be used -- staging stays one layer deep");
        else if (!refused() && streamed_total > cap)
            KRK_INFO("weight residency: streamed set is %.2f GiB, above the "
                     "%.2f GiB pinned host-mirror cap -- staging keeps "
                     "re-reading the file (KRK_WEIGHT_HOST_CACHE_MB)",
                     static_cast<f64>(streamed_total) / 1073741824.0,
                     static_cast<f64>(cap) / 1073741824.0);
    }

    // The nextn/MTP block, when the file carries one and the caller asked for
    // it. Not an error if it is absent: most files have none.
    {
        std::string merr;
        if (!load_mtp_block(&merr) && !merr.empty()) {
            if (err) *err = merr;
            return false;
        }
    }

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
        // Kept for MLA's softmax scale, which the reference builds from these
        // two numbers (see ModelConfig::rope_yarn_log_mul). The frequency table
        // above does not need them; the attention scale does.
        cfg_.rope_yarn_factor = factor;
        cfg_.rope_yarn_log_mul = static_cast<f32>(
            gguf_.get_f64(key(".rope.scaling.yarn_log_multiplier"), 0.0));
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
    // them an ext_factor of 0, which is the same statement. A file with no
    // windowed-table keys of its own (`swa_own_rope` false) has one table for
    // both layer kinds, so nothing is built here and `inv_freq_at()` returns
    // the full-layer table -- YaRN included, which is what gpt-oss needs.
    if (cfg_.swa_window > 0 && swa_own_rope)
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

// The nextn/MTP block. `n_layer_nextn` blocks sit after the stack and are not
// transformer layers of the stack, so `Model::load` excludes them from
// `cfg_.n_layer` and nothing in the main path touches them. This loads one when
// the caller asks (KRK_MTP_LOAD=1): the block plus the three adapters that fuse
// the target's hidden state with the last token's embedding. Routed experts are
// left in the file for ExpertCache to page, since they are 855 MB on a 35B-A3B.
bool Model::load_mtp_block(std::string *err) {
    mtp_ = MtpBlock{};
    // ---- a sidecar nextn file -------------------------------------------
    // A `-noMTP` trunk has no block of its own (nextn_predict_layers is absent
    // or 0), so the in-file branch below returns before doing anything. When a
    // path was given, the block comes from that file instead: same tensor names,
    // same index convention (the block sits past the stack), and its mapping has
    // to outlive the block because the routed banks are `const GgufTensor *`
    // into it.
    if (!mtp_path_.empty()) {
        mtp_gguf_.reset(new Gguf());
        if (!mtp_gguf_->load(mtp_path_, err)) {
            mtp_gguf_.reset();
            return false;
        }
        const std::string side_arch = mtp_gguf_->get_str("general.architecture", "");
        // A file whose tensors are all `mtp.*` is an MTP head, and that is
        // decided by its TENSORS, not by its header: the converters disagree on
        // the spelling ("qwen4exp-mtp" for the q2_0 repack, "qwen4-mtp" for the
        // BF16 original of the same 31 tensors), and a header comparison refused
        // one of them as "another family" when it is the same head.
        i32 hf_tensors = 0;
        for (const GgufTensor &t : mtp_gguf_->tensors())
            if (t.name.rfind("mtp.", 0) == 0) ++hf_tensors;
        if (hf_tensors > 0) {
            // ... and it is the same head this trunk's own block would be, in HF
            // names and in a shape this loader does not run: `load_mtp_block_from`
            // expects `blk.N.attn_*` + `blk.N.ffn_*`, while these blocks carry a
            // hyper-connection mixer, a QSA indexer's attention and their own
            // routed MoE. Named here so the gap is the file's own -- refusing it
            // as "not a nextn block" was both wrong and unhelpful.
            if (err)
                *err = mtp_path_ + " carries " + std::to_string(hf_tensors) +
                       " `mtp.*` tensor(s): the same head this trunk's own block "
                       "would be, in HF names and in a shape this loader does not "
                       "run (hyper-connection mixer, QSA indexer attention, its own "
                       "routed MoE). Reading it needs that forward; the verify loop "
                       "(--mtp-spec) is already here and unchanged. See "
                       "docs/AUDIT-mtp-drafter.md.";
            mtp_gguf_.reset();
            return false;
        }
        if (!side_arch.empty() && side_arch != cfg_.arch) {
            if (err)
                *err = mtp_path_ + " is a '" + side_arch + "' file and this model "
                       "is '" + cfg_.arch + "' -- a head from another family "
                       "proposes tokens this trunk would never accept";
            mtp_gguf_.reset();
            return false;
        }
        // The block index is the smallest `blk.<n>.` at or past the trunk's own
        // stack. Reading it from the tensors rather than from block_count keeps
        // this correct for a sidecar whose header counts only its own blocks.
        i32 block = -1;
        for (const GgufTensor &t : mtp_gguf_->tensors()) {
            if (t.name.rfind("blk.", 0) != 0) continue;
            const size_t dot = t.name.find('.', 4);
            if (dot == std::string::npos || dot == 4) continue;
            i32 k = 0;
            for (size_t i = 4; i < dot; i++) {
                if (t.name[i] < '0' || t.name[i] > '9') { k = -1; break; }
                k = k * 10 + (t.name[i] - '0');
            }
            if (k < cfg_.n_layer) continue; // a trunk block, not a head
            if (block < 0 || k < block) block = k;
        }
        if (block < 0) {
            if (err)
                *err = mtp_path_ + " carries no blk.<n>.* block past this model's " +
                       std::to_string(cfg_.n_layer) +
                       " layers -- it is not a nextn block for this trunk";
            mtp_gguf_.reset();
            return false;
        }
        const bool ok = load_mtp_block_from(*mtp_gguf_, block, mtp_path_, err);
        if (!ok) mtp_gguf_.reset();
        return ok;
    }
    if (cfg_.n_layer_nextn <= 0) return true;
    // Two switches upload the block: KRK_MTP_LOAD for a run that wants it, and
    // KRK_MTP_PROBE (--mtp-probe), which cannot do anything without it. Keeping
    // them one gate means the probe never measures a block that is not there.
    const auto wanted = [](const char *name) {
        const char *v = std::getenv(name);
        return v && v[0] != '\0' && v[0] != '0';
    };
    if (!mtp_load_ && !wanted("KRK_MTP_LOAD") && !wanted("KRK_MTP_PROBE"))
        return true;
    return load_mtp_block_from(gguf_, cfg_.n_layer, model_path_, err);
}

bool Model::load_mtp_block_from(const Gguf &g, i32 l, const std::string &origin,
                                std::string *err) {
    const auto key = [&](const char *suffix) {
        return std::string("blk.") + std::to_string(l) + suffix;
    };
    const auto tensor = [&](const std::string &name) { return g.tensor(name); };

    const GgufTensor *tq = tensor(key(".attn_q.weight"));
    const GgufTensor *tk = tensor(key(".attn_k.weight"));
    const GgufTensor *tv = tensor(key(".attn_v.weight"));
    const GgufTensor *to = tensor(key(".attn_output.weight"));
    if (!tq || !tk || !tv || !to) {
        // A nextn block without attention is a shape this loader does not know.
        // Refusing here is the honest answer: silently skipping it would make a
        // drafter that drafts from nothing.
        if (err)
            *err = origin + " has no readable " + key(".attn_q.weight") +
                   " for its nextn block";
        return false;
    }
    for (DType t : {tq->type, tk->type, tv->type, to->type}) {
        if (!dtype_supported(t)) {
            if (err)
                *err = "nextn block " + std::to_string(l) + " uses unsupported "
                       "format " + dtype_name(t) + " (type id " +
                       std::to_string(tq->type_id) + ")";
            return false;
        }
    }

    MtpBlock &m = mtp_;
    m.layer = l;
    m.n_embd = static_cast<i32>(tq->ne[0]);
    if (m.n_embd != cfg_.n_embd) {
        if (err)
            *err = "nextn block " + std::to_string(l) + " in " + origin +
                   " is " + std::to_string(m.n_embd) +
                   " wide and this model is " + std::to_string(cfg_.n_embd) +
                   " -- the block's workspaces are the trunk's";
        return false;
    }
    // GGUF ne[] is (in, out): attn_q's OUT is query+gate packed on the arches
    // that pack a gate and plain q on the ones that do not, and attn_k's OUT is
    // the KV width -- so a shape alone cannot say how many heads there are.
    //
    // The counts come from the MODEL config, which is the only place that knows
    // them: a per-block key does not exist in a real file, and the trunk is the
    // authority for a block that belongs to it. `<arch>.attention.head_count_kv`
    // can be an ARRAY with one entry per layer (nemotron_h_moe ships 53 of
    // them), and the block's own index selects its entry -- reading the scalar
    // key of such a file gives the first layer's count and a wrong KV width.
    // Every number is then CHECKED against the tensor shapes below, so a wrong
    // guess is a refusal with the two numbers rather than a block that runs and
    // attends over the wrong slice.
    const std::string ap = cfg_.arch + ".";
    m.head_dim = static_cast<i32>(g.get_i64(key(".attention.key_length"), 0));
    if (m.head_dim <= 0)
        m.head_dim = static_cast<i32>(g.get_i64(ap + "attention.key_length", 0));
    if (m.head_dim <= 0) m.head_dim = cfg_.head_dim;
    if (m.head_dim <= 0) {
        if (const GgufTensor *qn = tensor(key(".attn_q_norm.weight")))
            m.head_dim = static_cast<i32>(qn->n_elements);
    }
    if (m.head_dim <= 0) {
        if (err) *err = "nextn block " + std::to_string(l) + " in " + origin +
                        ": nothing says how wide a head is";
        return false;
    }
    i32 n_head = static_cast<i32>(g.get_i64(ap + "attention.head_count", 0));
    if (n_head <= 0) n_head = cfg_.n_head;
    i32 n_kv = 0;
    if (const std::vector<i32> *a = g.get_i32_array(ap + "attention.head_count_kv")) {
        if (!a->empty())
            n_kv = (l >= 0 && l < static_cast<i32>(a->size())) ? (*a)[static_cast<size_t>(l)]
                                                               : a->back();
    } else {
        n_kv = static_cast<i32>(g.get_i64(ap + "attention.head_count_kv", 0));
    }
    if (n_kv <= 0) n_kv = cfg_.n_head_kv;
    const i64 qout = static_cast<i64>(tq->ne[1]);
    const i64 kout = static_cast<i64>(tk->ne[1]);
    if (n_head <= 0) n_head = static_cast<i32>(qout / (2 * m.head_dim));
    if (n_kv <= 0) n_kv = static_cast<i32>(kout / m.head_dim);
    if (n_head <= 0 || n_kv <= 0) {
        if (err) *err = "nextn block " + std::to_string(l) + " in " + origin +
                        " has no usable attention geometry";
        return false;
    }
    if (qout != static_cast<i64>(n_head) * m.head_dim &&
        qout != 2 * static_cast<i64>(n_head) * m.head_dim) {
        if (err)
            *err = "nextn block " + std::to_string(l) + " in " + origin + ": the "
                   "query projection is " + std::to_string(qout) + " wide, and " +
                   std::to_string(n_head) + " head(s) x " +
                   std::to_string(m.head_dim) + " say " +
                   std::to_string(static_cast<i64>(n_head) * m.head_dim) + " or " +
                   std::to_string(2 * static_cast<i64>(n_head) * m.head_dim);
        return false;
    }
    if (kout != static_cast<i64>(n_kv) * m.head_dim) {
        if (err)
            *err = "nextn block " + std::to_string(l) + " in " + origin + ": the "
                   "key projection is " + std::to_string(kout) + " wide, and " +
                   std::to_string(n_kv) + " kv head(s) x " +
                   std::to_string(m.head_dim) + " say " +
                   std::to_string(static_cast<i64>(n_kv) * m.head_dim) +
                   " -- the block and the trunk disagree about its attention";
        return false;
    }
    // The block belongs to this trunk, so it has to use the trunk's head count:
    // a sidecar that disagrees (a head from a same-family model with a
    // different attention shape) runs its own rows over a KV plane the engine
    // sized for the other geometry.
    if (cfg_.n_head > 0 && n_head != cfg_.n_head) {
        if (err)
            *err = "nextn block " + std::to_string(l) + " in " + origin + " has " +
                   std::to_string(n_head) + " head(s) and this model runs " +
                   std::to_string(cfg_.n_head) +
                   " -- a head from another model, not this trunk's";
        return false;
    }
    if (cfg_.n_head_kv > 0 && n_kv != cfg_.n_head_kv) {
        if (err)
            *err = "nextn block " + std::to_string(l) + " in " + origin + " has " +
                   std::to_string(n_kv) + " kv head(s) and this model runs " +
                   std::to_string(cfg_.n_head_kv);
        return false;
    }
    m.n_head = n_head;
    m.n_head_kv = n_kv;

    // Verbatim upload for the quantized matrices, dequantized f32 for the norms
    // and biases. The same shapes the per-layer loader uses; the shared head is
    // the model's own output.weight, so no head is loaded here.
    const auto upload_q = [&](const GgufTensor *t, QuantTensor *out) {
        out->data = be_->alloc(t->n_bytes);
        if (!out->data) return false;
        be_->upload(out->data, t->data, t->n_bytes);
        out->type = t->type;
        out->n_in = static_cast<i64>(t->ne[0]);
        out->n_out = t->n_dims >= 2 ? static_cast<i64>(t->ne[1]) : 1;
        // The block's own weights carry NVFP4 global scales like the trunk's,
        // and this uploader is separate from the layer one -- the same sidecar
        // rule applies, or a block whose checkpoint is NVFP4 (the Lightning
        // head set) runs every projection unscaled.
        static const char *kW = ".weight";
        const size_t wn = std::strlen(kW);
        const std::string base =
            t->name.size() > wn && t->name.compare(t->name.size() - wn, wn, kW) == 0
                ? t->name.substr(0, t->name.size() - wn)
                : t->name;
        out->gs = 1.0f;
        if (const GgufTensor *s = tensor(base + ".scale")) {
            if (s->data && s->n_elements >= 1 && s->type == DType::F32) {
                out->gs = reinterpret_cast<const f32 *>(s->data)[0];
                if (out->gs != 1.0f && out->n_in > max_scaled_in_)
                    max_scaled_in_ = out->n_in;
            }
        }
        m.bytes += static_cast<size_t>(t->n_bytes);
        return true;
    };
    const auto upload_f = [&](const std::string &name) -> f32 * {
        const GgufTensor *t = tensor(name);
        if (!t) return nullptr;
        const i64 n = t->n_elements;
        f32 *host = static_cast<f32 *>(host_alloc(static_cast<size_t>(n) * 4));
        dequant_row(t->type, t->data, host, n);
        f32 *dev = static_cast<f32 *>(be_->alloc(static_cast<size_t>(n) * 4));
        if (dev) be_->upload(dev, host, static_cast<size_t>(n) * 4);
        host_free(host);
        m.bytes += static_cast<size_t>(n) * 4;
        return dev;
    };

    bool ok = upload_q(tq, &m.wq) && upload_q(tk, &m.wk) &&
              upload_q(tv, &m.wv) && upload_q(to, &m.wo);
    if (ok) {
        if (const GgufTensor *t = tensor(key(".nextn.eh_proj.weight")))
            ok = upload_q(t, &m.eh_proj);
    }
    if (ok && tensor(key(".ffn_gate_inp.weight")))
        ok = upload_q(tensor(key(".ffn_gate_inp.weight")), &m.router);
    {
        const struct {
            const char *name;
            QuantTensor *dst;
        } sh[4] = {
            {".ffn_gate_shexp.weight", &m.shexp_gate},
            {".ffn_up_shexp.weight", &m.shexp_up},
            {".ffn_down_shexp.weight", &m.shexp_down},
            {".ffn_gate_inp_shexp.weight", &m.shexp_inp_gate},
        };
        for (int i = 0; ok && i < 4; i++)
            if (const GgufTensor *t = tensor(key(sh[i].name)))
                ok = upload_q(t, sh[i].dst);
    }
    if (ok) {
        // Dense FFN (`ffn_gate/up/down`, or ungated `ffn_up/down`). Uploaded
        // like any other dense matrix: the block is one layer, not a bank.
        if (const GgufTensor *t = tensor(key(".ffn_gate.weight")))
            ok = upload_q(t, &m.ffn_gate);
        if (ok) {
            if (const GgufTensor *t = tensor(key(".ffn_up.weight")))
                ok = upload_q(t, &m.ffn_up);
        }
        if (ok) {
            if (const GgufTensor *t = tensor(key(".ffn_down.weight")))
                ok = upload_q(t, &m.ffn_down);
        }
        if (ok && m.ffn_gate.present()) m.n_ff = static_cast<i32>(m.ffn_gate.n_out);
        if (ok && m.n_ff == 0 && m.ffn_up.present())
            m.n_ff = static_cast<i32>(m.ffn_up.n_out);
    }
    if (ok) {
        m.attn_norm = upload_f(key(".attn_norm.weight"));
        m.post_attn_norm = upload_f(key(".post_attention_norm.weight"));
        m.q_norm = upload_f(key(".attn_q_norm.weight"));
        m.k_norm = upload_f(key(".attn_k_norm.weight"));
        m.enorm = upload_f(key(".nextn.enorm.weight"));
        m.hnorm = upload_f(key(".nextn.hnorm.weight"));
        m.shared_head_norm = upload_f(key(".nextn.shared_head_norm.weight"));
        if (const GgufTensor *t = tensor(key(".ffn_gate_inp.bias"))) {
            const i64 n = t->n_elements;
            m.router_bias = static_cast<f32 *>(host_alloc(static_cast<size_t>(n) * 4));
            if (m.router_bias)
                dequant_row(t->type, t->data, m.router_bias, n);
        }
        m.exps_gate = tensor(key(".ffn_gate_exps.weight"));
        m.exps_up = tensor(key(".ffn_up_exps.weight"));
        m.exps_down = tensor(key(".ffn_down_exps.weight"));
        // An UNGATED expert bank has no gate tensor at all (nemotron_h_moe's
        // nextn block is up/down only), so the bank is whatever of the three is
        // there and the width comes from the tensor that carries it: up/gate put
        // the FFN width in ne[1], down puts it in ne[0]. Reading n_expert from
        // the gate alone made this block load as "no experts and no dense FFN"
        // and the probe refused a block that was right there in the file.
        const GgufTensor *any_exp =
            m.exps_gate ? m.exps_gate : (m.exps_up ? m.exps_up : m.exps_down);
        if (any_exp) m.n_expert = static_cast<i32>(any_exp->ne[2]);
        if (m.exps_up)
            m.n_ff_exp = static_cast<i32>(m.exps_up->ne[1]);
        else if (m.exps_gate)
            m.n_ff_exp = static_cast<i32>(m.exps_gate->ne[1]);
        else if (m.exps_down)
            m.n_ff_exp = static_cast<i32>(m.exps_down->ne[0]);
        // expert_used_count is a MODEL-level key (qwen35moe.expert_used_count),
        // not a per-block one, so key() -- which prefixes blk.<n>. -- cannot
        // reach it. The parsed config already has the number.
        m.n_expert_used = cfg_.n_expert_used;
        // The shared expert can be ungated too, and then its width is the up
        // projection's -- a gate-only reading reported 0 and dropped the whole
        // shared path out of the block's FFN.
        if (m.shexp_up.present()) m.n_ff_shexp = static_cast<i32>(m.shexp_up.n_out);
        else if (m.shexp_gate.present())
            m.n_ff_shexp = static_cast<i32>(m.shexp_gate.n_out);
    }
    if (!ok) {
        if (err) *err = "nextn block " + std::to_string(l) +
                        ": a weight upload failed (device allocation refused)";
        return false;
    }
    m.present = true;
    // One line per shape: an MoE block names its routed bank, a dense block names
    // its FFN. Printing "0 experts x top-0 of 0-wide FFN" for a dense block made
    // the report read as a broken block rather than a different one.
    if (m.n_expert > 0) {
        KRK_INFO("nextn (MTP) block %d loaded: %d head(s) x %d, %d experts x "
                 "top-%d of %d-wide FFN%s, eh_proj %lldx%lld, adapters "
                 "enorm/hnorm/shared_head_norm %s, %.1f MiB resident, experts "
                 "left in the file",
                 l, m.n_head, m.head_dim, m.n_expert, m.n_expert_used,
                 m.n_ff_exp,
                 m.n_ff_shexp > 0 ? " + shared expert" : "",
                 static_cast<long long>(m.eh_proj.n_in),
                 static_cast<long long>(m.eh_proj.n_out),
                 (m.enorm && m.hnorm && m.shared_head_norm) ? "present"
                                                           : "MISSING",
                 static_cast<f64>(m.bytes) / 1048576.0);
    } else {
        KRK_INFO("nextn (MTP) block %d loaded: %d head(s) x %d, dense %d-wide "
                 "%s FFN, eh_proj %lldx%lld, adapters enorm/hnorm/"
                 "shared_head_norm %s, %.1f MiB resident",
                 l, m.n_head, m.head_dim, m.n_ff,
                 m.ffn_gate.present() ? "SwiGLU" : "ungated",
                 static_cast<long long>(m.eh_proj.n_in),
                 static_cast<long long>(m.eh_proj.n_out),
                 (m.enorm && m.hnorm && m.shared_head_norm) ? "present"
                                                           : "MISSING",
                 static_cast<f64>(m.bytes) / 1048576.0);
    }
    return true;
}

// A streamed layer that measured free VRAM can hold is kept instead of being
// re-staged on every step. The load-time budget is a decision made before the
// KV cache and the workspaces exist, and on this card it left 2 GiB free at the
// end of a run while 17 layers were being re-read every token -- 100% of the
// step. Promoting on first use turns that leftover into residency, and the
// floor is the same reserve the allocator is guarded by, measured rather than
// predicted, so it cannot over-commit: the check is against bytes that are free
// right now, after every other allocation has been made.
bool Model::promote_layer(i32 l) {
    if (!promote_enabled_ || !be_ || !ring_ || !stage_host_) return false;
    if (l < 0 || l >= static_cast<i32>(stream_.size())) return false;
    std::vector<StreamEntry> &es = stream_[static_cast<size_t>(l)];
    if (es.empty()) return false;
    const size_t bytes = layer_stream_bytes_[static_cast<size_t>(l)];
    const size_t total = be_->device_total_bytes();
    const size_t floor = vram_reserve_bytes(total);
    const size_t freeb = be_->device_free_bytes();
    if (bytes == 0 || freeb < bytes + floor) return false;
    const u64 t0 = now_us();
    void *base = be_->alloc(bytes);
    if (!base) return false;
    // The same read and copy one staging pass does, into memory that is kept.
    // No drain is needed first: the ring is not touched at all here, and each
    // batch reads into a region of the pinned buffer the previous batch already
    // finished with (its upload is ordered on the default stream).
    size_t off = 0;
    size_t moved = 0;
    for (size_t b = 0; b < es.size(); b += 32) {
        Backend::ReadReq reqs[32];
        void *dst[32];
        const void *src[32];
        size_t len[32];
        int n = 0;
        for (size_t i = b; i < es.size() && n < 32; i++) {
            const StreamEntry &e = es[i];
            reqs[n].dst = static_cast<u8 *>(stage_host_) + off;
            reqs[n].mapped_src = e.src->data;
            reqs[n].bytes = e.bytes;
            dst[n] = static_cast<u8 *>(base) + off;
            src[n] = reqs[n].dst;
            len[n] = e.bytes;
            off += (e.bytes + 255u) & ~size_t(255u);
            moved += e.bytes;
            n++;
        }
        be_->read_host_batch(reqs, n);
        // `src` is stage_host_, which is alloc_host() memory -- already pinned.
        // The PAGEABLE call is what used to be here, and it costs the host a
        // full bounce copy through the 4 MiB ring: measured on this card at
        // 8.8 GB/s for exactly this promotion loop, against 27.5 GB/s for the
        // pinned batch the staging path uses. The two differ only in which side
        // of the copy the driver stages, and this source needs none of it.
        // Each batch writes a fresh region of stage_host_ (`off` only grows),
        // so the copies can stay in flight; the default stream is ordered after
        // them by the backend's event, which is what makes the tensors below
        // safe to read immediately.
        be_->upload_pinned_batch(dst, src, len, n);
    }
    // Re-point the layer's own tensors. `dst` keeps its ring-relative offset,
    // so only the address the engine reads through changes.
    for (const StreamEntry &e : es)
        if (e.q)
            e.q->data = static_cast<u8 *>(base) +
                        (static_cast<const u8 *>(e.dst) - slot_base(l));
    promoted_.emplace_back(base, bytes);
    promoted_bytes_ += moved;
    promoted_layers_++;
    layer_streamed_[static_cast<size_t>(l)] = 0;
    if (streamed_layers_) streamed_layers_--;
    promote_ms_ += static_cast<f64>(now_us() - t0) / 1000.0;
    KRK_INFO("weight residency: layer %d promoted resident (%.1f MiB, "
             "%.0f MiB still free) -- it will not be staged again",
             l, static_cast<f64>(moved) / 1048576.0,
             static_cast<f64>(freeb - bytes) / 1048576.0);
    return true;
}

void Model::read_layer_to(void *host, const std::vector<StreamEntry> &es,
                          size_t *bytes) {
    // Batches of 32: a layer carries far fewer, but a fixed stack array must
    // not silently drop the tail (a dropped tensor is a wrong answer, not a
    // slow one).
    size_t off = 0;
    for (size_t base = 0; base < es.size(); base += 32) {
        Backend::ReadReq reqs[32];
        int n = 0;
        for (size_t i = base; i < es.size() && n < 32; i++) {
            const StreamEntry &e = es[i];
            reqs[n].dst = static_cast<u8 *>(host) + off;
            reqs[n].mapped_src = e.src->data;
            reqs[n].bytes = e.bytes;
            off += (e.bytes + 255u) & ~size_t(255u);
            if (bytes) *bytes += e.bytes;
            n++;
        }
        be_->read_host_batch(reqs, n);
    }
}

void Model::upload_layer_from(const void *host, const std::vector<StreamEntry> &es) {
    size_t off = 0;
    for (size_t base = 0; base < es.size(); base += 32) {
        void *dst[32];
        const void *src[32];
        size_t len[32];
        int n = 0;
        for (size_t i = base; i < es.size() && n < 32; i++) {
            const StreamEntry &e = es[i];
            dst[n] = e.dst;
            src[n] = static_cast<const u8 *>(host) + off;
            len[n] = e.bytes;
            off += (e.bytes + 255u) & ~size_t(255u);
            n++;
        }
        // A pinned source: the backend DMAs straight out of this buffer and the
        // default stream waits on its event, so this returns as soon as the
        // copies are queued rather than when they land.
        be_->upload_pinned_batch(dst, src, len, n);
    }
}

// Fill the pinned host mirror described at its call site in load(): one
// allocation, every streamed layer at its own offset, read once. The offsets
// are the same aligned offsets read_layer_to already writes, so a layer's
// mirror slice is byte-for-byte what a staging pass would have read for it.
void Model::build_stream_host_mirror(size_t streamed_total) {
    if (!be_ || streamed_total == 0) return;
    void *base = be_->alloc_host(streamed_total);
    if (!base) {
        KRK_WARN("weight residency: no %.2f GiB pinned host mirror for the "
                 "streamed layers -- they will be re-read every step "
                 "(KRK_WEIGHT_HOST_CACHE=0 silences this arm)",
                 static_cast<f64>(streamed_total) / 1073741824.0);
        return;
    }
    stream_host_ = base;
    stream_host_size_ = streamed_total;
    stream_host_off_.assign(stream_.size(), 0);
    size_t off = 0;
    for (size_t l = 0; l < stream_.size(); l++) {
        if (stream_[l].empty() || !layer_streamed(static_cast<i32>(l))) continue;
        stream_host_off_[l] = off;
        off += layer_stream_bytes_[l];
    }
    const u64 t0 = now_us();
    for (i32 l = 0; l < cfg_.n_layer; l++) {
        if (!layer_streamed(l)) continue;
        const std::vector<StreamEntry> &es = stream_[static_cast<size_t>(l)];
        if (es.empty()) continue;
        read_layer_to(static_cast<u8 *>(stream_host_) +
                          stream_host_off_[static_cast<size_t>(l)],
                      es, nullptr);
    }
    stream_host_fill_ms_ = static_cast<f64>(now_us() - t0) / 1000.0;
    stream_host_on_ = true;
    KRK_INFO("weight residency: %.2f GiB pinned host mirror of %zu streamed "
             "layer(s) filled in %.0f ms -- staging DMAs from it instead of "
             "re-reading the file",
             static_cast<f64>(streamed_total) / 1073741824.0, streamed_layers_,
             stream_host_fill_ms_);
}

i32 Model::next_streamed(i32 after) const {
    for (i32 l = after + 1; l < static_cast<i32>(stream_.size()); l++)
        if (layer_streamed(l)) return l;
    return -1;
}

// Which ring slot layer `l` owns. Parity, not order of use: it is fixed at load
// so a layer's tensor addresses never move.
const u8 *Model::slot_base(i32 l) const {
    if (!ring_ || slot_bytes_ == 0) return static_cast<const u8 *>(ring_);
    return static_cast<const u8 *>(ring_) +
           (static_cast<size_t>(l) & 1u) * slot_bytes_;
}

i32 Model::first_streamed() const {
    for (i32 l = 0; l < static_cast<i32>(stream_.size()); l++)
        if (layer_streamed(l)) return l;
    return -1;
}

// Fill the slot the NEXT streamed layer owns while this layer's kernels run.
//
// Two orderings make that safe and neither blocks the host:
//   * the copy stream follows `ready`, an event recorded here on the default
//     stream. That position covers every kernel enqueued so far, which is
//     exactly the previous reader of the target slot (the layer two before the
//     one being filled). This layer's own kernels are not enqueued yet, which is
//     why the copy may start now.
//   * the default stream follows the copy before the target layer runs, in
//     stage_layer.
void Model::issue_ahead(i32 l) {
    if (!ring_ahead_ || !be_ || !stream_host_on_) return;
    i32 nl = next_streamed(l);
    // Wrap at the end of the stack: the last layer then fills the slot the next
    // pass begins with, and that copy runs under the logits and the sample.
    if (nl < 0) nl = first_streamed();
    if (nl < 0 || nl == l) return;
    const std::vector<StreamEntry> &es = stream_[static_cast<size_t>(nl)];
    if (es.empty()) return;
    if (inflight_layer_ >= 0) {
        // Reachable only when one pass stages the same layer twice: wait the
        // older copy out rather than overwrite the bytes it is reading.
        be_->wait_marker_stream(inflight_ticket_);
        be_->release_marker(inflight_ticket_);
        inflight_ticket_ = nullptr;
        inflight_layer_ = -1;
    }
    void *ready = be_->make_marker();
    if (ready) be_->record_marker(ready);
    const u8 *hb = static_cast<const u8 *>(stream_host_) +
                   stream_host_off_[static_cast<size_t>(nl)];
    const u8 *dstb = slot_base(nl);
    size_t off = 0;
    void *ticket = nullptr;
    const size_t total = es.size();
    for (size_t b = 0; b < total; b += 32) {
        void *dst[32];
        const void *src[32];
        size_t len[32];
        int n = 0;
        for (size_t i = b; i < total && n < 32; i++) {
            const StreamEntry &e = es[i];
            dst[n] = const_cast<u8 *>(dstb) + off;
            src[n] = hb + off;
            len[n] = e.bytes;
            off += (e.bytes + 255u) & ~size_t(255u);
            n++;
        }
        // Only the first batch needs the readers' marker: the rest are ordered
        // behind it on the same copy stream.
        void *t = be_->prefetch_after(dst, src, len, n, b == 0 ? ready : nullptr);
        if (b + 32 < total) {
            // Every batch but the last is ordered by its successor on the copy
            // stream, so its ticket is not the one to wait on.
            if (t) be_->release_marker(t);
        } else {
            ticket = t;
        }
    }
    if (ready) be_->release_marker(ready);
    if (ticket) {
        inflight_layer_ = nl;
        inflight_ticket_ = ticket;
    }
}

void Model::stage_layer(i32 l) {
    if (!be_ || !ring_ || l < 0 || l >= static_cast<i32>(stream_.size())) return;
    // A promoted layer is resident now; its entries still exist (unload reads
    // them) but its tensors no longer point into the ring.
    if (!layer_streamed(l)) return;
    if (promote_layer(l)) return;
    const std::vector<StreamEntry> &es = stream_[static_cast<size_t>(l)];
    if (es.empty()) return;

    const u64 t0 = now_us();
    if (ring_ahead_ && inflight_layer_ == l) {
        // The copy for this layer has been in flight since the previous layer's
        // staging pass. What is left is one device-side ordering -- the host does
        // not wait, so the time booked here is the enqueue, not the transfer.
        be_->wait_marker_stream(inflight_ticket_);
        be_->release_marker(inflight_ticket_);
        inflight_ticket_ = nullptr;
        inflight_layer_ = -1;
        ahead_copies_++;
        stage_bytes_ += layer_stream_bytes_[static_cast<size_t>(l)];
        stage_calls_++;
        stage_ms_ += static_cast<f64>(now_us() - t0) / 1000.0;
    } else {
        // Cold: the first streamed layer of the run, or a layer whose prefetch
        // was skipped. Fill it through the single-slot path so a miss costs
        // speed, never correctness.
        if (ring_ahead_) ahead_cold_++;
        if (!stage_layer_now(l, es)) return;
    }
    // The slot just settled, and every kernel of the layers before this one is
    // enqueued: this is the one point where the next copy can be issued safely.
    issue_ahead(l);
}

// The single-slot pass: drain, read, DMA. Without the double buffer this is what
// every streamed layer pays; with it, only a cold layer does.
bool Model::stage_layer_now(i32 l, const std::vector<StreamEntry> &es) {
    // The ring is reused by the next layer, and the previous layer's kernels
    // may still be reading it: the copy is published TO the default stream, not
    // FROM it. Drain once before overwriting, or layer l reads layer l+1's
    // bytes -- which is fluent-looking garbage, not a crash. It also releases
    // the pinned buffer this pass is about to write into.
    const u64 t0 = now_us();
    be_->sync();
    const u64 t_sync = now_us();

    size_t bytes = 0;
    const void *src = nullptr;
    // With the pinned mirror, this layer's bytes are already in host memory and
    // no read happens at all: the pass is the drain plus the DMA. Without it,
    // the pass reads the layer into a pinned buffer and DMAs from there, and the
    // next layer's read rides on the spare buffer while this copy is in flight.
    if (stream_host_on_) {
        for (const StreamEntry &e : es) bytes += e.bytes;
        src = static_cast<const u8 *>(stream_host_) +
              stream_host_off_[static_cast<size_t>(l)];
    } else {
        // Which pinned buffer holds this layer. The previous staging pass read it
        // while THIS layer's predecessor was still copying, which is the overlap
        // this function exists to create; on the first streamed layer there is
        // nothing to inherit and the read happens here.
        int slot = -1;
        if (prefetched_layer_ == l && read_buf_[prefetched_slot_]) slot = prefetched_slot_;
        if (slot < 0) {
            slot = 0;
            const u64 tr = now_us();
            read_layer_to(read_buf_[slot], es, &bytes);
            stage_read_ms_ += static_cast<f64>(now_us() - tr) / 1000.0;
        } else {
            for (const StreamEntry &e : es) bytes += e.bytes;
        }
        prefetched_layer_ = -1;
        src = read_buf_[slot];

        // Read the NEXT layer into the spare buffer while this layer's copy is
        // still moving over PCIe. The copy is a DMA on a side stream, so the host
        // thread is free immediately; the read is the half that used to wait.
        const i32 nl = prefetch_enabled_ ? next_streamed(l) : -1;
        const int nslot = 1 - slot;
        if (read_buf_[nslot] && nl >= 0) {
            const u64 tr = now_us();
            read_layer_to(read_buf_[nslot], stream_[static_cast<size_t>(nl)], nullptr);
            stage_prefetch_ms_ += static_cast<f64>(now_us() - tr) / 1000.0;
            prefetched_layer_ = nl;
            prefetched_slot_ = nslot;
        }
    }

    const u64 tx = now_us();
    upload_layer_from(src, es);
    stage_xfer_ms_ += static_cast<f64>(now_us() - tx) / 1000.0;

    stage_sync_ms_ += static_cast<f64>(t_sync - t0) / 1000.0;
    stage_ms_ += static_cast<f64>(now_us() - t0) / 1000.0;
    stage_calls_++;
    stage_bytes_ += bytes;
    return true;
}

void Model::unload() {
    if (!be_) return;
    if (out_bias_) {
        be_->release(out_bias_);
        out_bias_ = nullptr;
    }
    // The expert cache owns backend allocations too; drop them first.
    experts_.clear();
    for (auto &L : layers_) {
        if (L.attn_norm) be_->release(L.attn_norm);
        if (L.attn_post_norm) be_->release(L.attn_post_norm);
        if (L.ffn_post_norm) be_->release(L.ffn_post_norm);
        if (L.ffn_norm) be_->release(L.ffn_norm);
        if (L.attn_norm_2) be_->release(L.attn_norm_2);
        if (L.attn_norm_2_bias) be_->release(L.attn_norm_2_bias);
        if (L.attn_sub_norm) be_->release(L.attn_sub_norm);
        if (L.ffn_sub_norm) be_->release(L.ffn_sub_norm);
        if (L.wup_bias) be_->release(L.wup_bias);
        if (L.wdown_bias) be_->release(L.wdown_bias);
        if (L.q_bias) be_->release(L.q_bias);
        if (L.k_bias) be_->release(L.k_bias);
        if (L.v_bias) be_->release(L.v_bias);
        if (L.q_norm) be_->release(L.q_norm);
        if (L.k_norm) be_->release(L.k_norm);
        if (L.sinks) be_->release(L.sinks);
        if (L.wkv_a_norm) be_->release(L.wkv_a_norm);
        if (L.wq_a_norm) be_->release(L.wq_a_norm);
        if (L.wkv_norm) be_->release(L.wkv_norm);
        // Device-resident like every other 1-D weight: the wo GEMM and the
        // expert groups read them on the device.
        for (f32 *f : {L.wo_bias, L.exp_gate_bias, L.exp_up_bias, L.exp_down_bias})
            if (f) be_->release(f);
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
        // A streamed tensor's data points into the shared ring, not into its
        // own allocation; releasing it would free one interior pointer per
        // tensor. The ring is released once, below.
        auto in_ring = [&](const void *p) {
            return ring_ && p >= ring_ &&
                   p < static_cast<const u8 *>(ring_) + ring_bytes_;
        };
        // ... and the resident slab is the same story, one level down.
        auto in_slab = [&](const void *p) {
            return slab_ && p >= slab_ && p < slab_ + slab_bytes_;
        };
        // A promoted layer's tensors are interior pointers into one base that
        // is released once below; releasing each of them would free one
        // interior pointer per tensor.
        auto in_promoted = [&](const void *p) {
            for (const auto &b : promoted_) {
                const u8 *lo = static_cast<const u8 *>(b.first);
                if (p >= lo && p < lo + b.second) return true;
            }
            return false;
        };
        for (QuantTensor *q : qt_fields(L))
            if (q->present() && !in_ring(q->data) && !in_promoted(q->data) &&
                !in_slab(q->data))
                be_->release(q->data);
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
    if (prefetch_host_) be_->release_host(prefetch_host_);
    prefetch_host_ = nullptr;
    read_buf_[0] = nullptr;
    read_buf_[1] = nullptr;
    prefetched_layer_ = -1;
    if (inflight_ticket_) be_->release_marker(inflight_ticket_);
    inflight_ticket_ = nullptr;
    inflight_layer_ = -1;
    ring_ahead_ = false;
    ring_slots_ = 1;
    slot_bytes_ = 0;
    ahead_copies_ = 0;
    ahead_cold_ = 0;
    ahead_wait_ms_ = 0;
    if (stream_host_) be_->release_host(stream_host_);
    stream_host_ = nullptr;
    stream_host_off_.clear();
    stream_host_on_ = false;
    stream_host_size_ = 0;
    stream_host_fill_ms_ = 0;
    for (const auto &b : promoted_) be_->release(b.first);
    promoted_.clear();
    promoted_bytes_ = 0;
    promoted_layers_ = 0;
    promote_ms_ = 0;
    layer_stream_bytes_.clear();
    if (slab_) be_->release(slab_);
    slab_ = nullptr;
    slab_bytes_ = 0;
    slab_off_ = 0;
    if (stage_host_) be_->release_host(stage_host_);
    if (ring_) be_->release(ring_);
    stage_host_ = nullptr;
    ring_ = nullptr;
    ring_bytes_ = 0;
    stream_.clear();
    layer_streamed_.clear();
    // The nextn block is not a layer in `layers_`, so its allocations are not
    // reached by the loop above and have to be released here. They are owned
    // device buffers (plus one HOST router bias), never ring or promoted
    // pointers, so releasing each one is correct: nothing else aliases them.
    // This runs whether or not the block finished loading, because a failed
    // upload leaves the allocations it already made behind.
    for (QuantTensor *q : {&mtp_.wq, &mtp_.wk, &mtp_.wv, &mtp_.wo,
                           &mtp_.router, &mtp_.shexp_gate, &mtp_.shexp_up,
                           &mtp_.shexp_down, &mtp_.shexp_inp_gate,
                           &mtp_.eh_proj})
        if (q->present()) be_->release(q->data);
    for (f32 *f : {mtp_.attn_norm, mtp_.post_attn_norm, mtp_.q_norm,
                   mtp_.k_norm, mtp_.enorm, mtp_.hnorm,
                   mtp_.shared_head_norm})
        if (f) be_->release(f);
    if (mtp_.router_bias) host_free(mtp_.router_bias);
    mtp_ = MtpBlock{};
    tok_embd_ = {};
    out_head_ = {};
    out_norm_ = nullptr;
    be_ = nullptr;
}

} // namespace krk
