// ============================================================================
//  model.hpp — GGUF -> device-resident weight set.
//
//  Covers the dense LLaMA family: llama, mistral, qwen2, qwen3 (QK-norm),
//  smollm, granite, and any arch speaking the same tensor schema — plus the
//  Mixture-of-Experts Qwen family (qwen2moe, qwen3moe) and any arch that emits
//  the standard `ffn_*_exps` expert tensors. Expert weights are NOT uploaded at
//  load time; see expert_cache.hpp. The IQ* formats are still rejected with a
//  message naming the offending tensor instead of being silently mis-decoded.
// ============================================================================
#ifndef KRK_MODEL_HPP
#define KRK_MODEL_HPP

#include "krk/backend.hpp"
#include "krk/expert_cache.hpp"
#include "krk/chat_template.hpp"
#include "krk/gguf.hpp"

#include <cmath> // std::tanh in apply_logit_softcap, called from every path
#include <utility> // std::pair in the promoted-layer table

namespace krk {

// QuantTensor (a resident weight matrix) is declared in expert_cache.hpp, which
// this header includes; the same descriptor covers eager and lazy weights.

// The prism.hadamard activation transform (PrismML's llama.cpp fork, the
// Ternary-Bonsai-2 exports). A few matmul weights are stored with their
// INPUT dimension rotated: multiplied by a per-width +-1 sign vector and
// then by a normalized Sylvester Walsh-Hadamard over consecutive blocks of
// `block` elements. Nothing has to be done to the weights -- the engine
// transforms the matching ACTIVATION before each such matmul (and, for the
// one table stored rotated, once after the embedding lookup).
//
// Inactive for every file that does not carry the keys, so the plan costs a
// bool test per site and nothing else.
struct HadamardPlan {
    bool active = false;
    i32 block = 1024;           // FWHT block size, a power of two
    bool embd_inverse = false;  // token_embd.weight stores rotated rows
    bool head_folded = false;   // output.weight's input is transformed
    bool gdn_v_grouped = false; // ssm_out's input carries the grouped-V perm
    i64 max_width = 0;          // widest folded input; sizes the scratch
};

struct LayerWeights {
    f32 *attn_norm = nullptr; // f32 [n_embd]
    // ---- GPT-2 -----------------------------------------------------------
    // The two pre-norms are LayerNorms, so each carries a bias beside its
    // weight. Null on every other family, and null keeps their graph exactly
    // as it was: the engine only looks at these when cfg_.gpt2_style is set.
    f32 *attn_norm_bias = nullptr; // f32 [n_embd]
    // Falcon-40B's second pre-attention norm (`attn_norm_2`). Optional on the
    // family: Falcon-7B has one norm and reuses it for both arms.
    f32 *attn_norm_2 = nullptr;       // f32 [n_embd]
    f32 *attn_norm_2_bias = nullptr;  // f32 [n_embd]
    f32 *ffn_norm_bias = nullptr;  // f32 [n_embd]
    // The FUSED query/key/value projection, [n_embd -> 3*n_embd]. Its output
    // row is Q | K | V, three contiguous n_embd blocks, and one bias covers
    // all three. wq/wk/wv stay empty on this family; the forward splits the
    // fused result instead of running three projections, which is also what
    // the file expects -- there are no separate q/k/v tensors to run.
    QuantTensor wqkv_fused;   // distinct from the delta-net wqkv below
    // Phi-3's fused gate|up projection, [n_embd -> 2*n_ff]: one row holds the
    // gate's n_ff channels then the up's, and the file carries no ffn_gate.
    QuantTensor wup_fused;
    f32 *qkv_bias = nullptr;  // f32 [3*n_embd]
    QuantTensor wq, wk, wv, wo;
    // ---- MoVA: the attention VALUES are a routed expert set (k2-horizon) ----
    // This family has no `attn_v.weight`. V is built from the same mix of the
    // block input that Q and K come from, but through a router of its own:
    //
    //   gate    = attn_v_gate @ x                    [n_value_expert]
    //   probs   = softmax or sigmoid(gate)
    //   sel     = top-k(probs + attn_v_gate_b)       (bias shifts SELECTION only)
    //   V       = sum_k probs[sel_k] * silu(W_sel_k @ x)
    //
    // so a layer carrying this cannot take the wv path at all, and a file with
    // both is malformed rather than one of them being redundant. `attn_v_exps`
    // is resident -- n_value_expert is a handful, not the 128/256 of the FFN
    // bank -- so it is a plain 3-D tensor, [n_embd, kvw, n_value_expert], not
    // an ExpertSource.
    QuantTensor attn_v_gate;      // [n_embd -> n_value_expert]
    f32 *attn_v_gate_b = nullptr; // f32 [n_value_expert], optional, HOST
    QuantTensor attn_v_exps;      // [n_embd -> kvw*n_value_expert]
    f32 *q_bias = nullptr; // optional [n_head*hd]
    f32 *k_bias = nullptr; // optional [n_kv*hd]
    f32 *v_bias = nullptr; // optional [n_kv*hd]
    f32 *q_norm = nullptr; // optional per-head RMSNorm weight [hd]
    f32 *k_norm = nullptr;
    // Attention sink logit per query head [n_head], nullable like every other
    // optional 1-D weight: a file without `blk.N.attn_sinks.weight` is the
    // normal case and keeps the arithmetic it had (see AttnDesc::sinks).
    f32 *sinks = nullptr;
    // Attention-output bias (`attn_output.bias`, gpt-oss): one f32 per output
    // channel, added to the wo projection's result before the residual. DEVICE
    // memory -- the GEMM that produces the result runs there, and the add is
    // one launch with the same shape as `add_bias_cols`.
    f32 *wo_bias = nullptr;
    // The plain family biases its MLP projections (`ffn_up.bias` /
    // `ffn_down.bias`, GPT-NeoX). Absent on GPT-J, which has no biases at all,
    // so both applications are guarded on the pointer.
    f32 *wup_bias = nullptr;
    f32 *wdown_bias = nullptr;
    f32 *ffn_norm = nullptr; // f32 [n_embd]
    // Branch post norms (gemma2/3/4, muse-glimmer): applied to the sub-block's
    // output BEFORE the residual add, each with its own weight. Null on every
    // other family, and null is what keeps their graph unchanged.
    f32 *attn_post_norm = nullptr; // f32 [n_embd]
    f32 *ffn_post_norm = nullptr;  // f32 [n_embd]
    // ---- gemma4's routed layers (see ArchSpec::moe_two_branch) ------------
    // The expert branch's own input norm, and the two branch output norms.
    // `ffn_post_norm` above is the BLOCK's norm on the summed branches, so a
    // layer of this family carries three of them and they are not
    // interchangeable: swapping any two still produces fluent text from the
    // right weights in the wrong place.
    f32 *ffn_pre_norm2 = nullptr;  // f32 [n_embd]
    f32 *ffn_post_norm1 = nullptr; // f32 [n_embd]
    f32 *ffn_post_norm2 = nullptr; // f32 [n_embd]
    // The router's per-element input scale (`ffn_gate_inp.scale`). Present
    // exactly when `ffn_pre_norm2` is: both are the routed shape.
    f32 *router_scale = nullptr;   // f32 [n_embd]
    // ---- gemma4's per-layer input embeddings -----------------------------
    // The PLE contribution is a fourth stage inside the block:
    //   cur = gelu(inp_gate @ cur) * ple_slice[l]
    //   cur = post_norm(proj @ cur)
    //   x  += cur
    // `ple_slice` is the layer's own [n_embd_per_layer, n_tok] block of the
    // per-layer input, computed once before the stack.
    QuantTensor ple_inp_gate;   // [n_embd, n_embd_per_layer]
    QuantTensor ple_proj;       // [n_embd_per_layer, n_embd]
    f32 *ple_post_norm = nullptr;  // f32 [n_embd]
    // gemma4's per-layer output scalar (`layer_output_scale.weight`, one
    // value). Multiplied into the residual at the end of the block. 1.0 means
    // the file carries none.
    f32 out_scale = 1.0f;
    // BitNet's SubLN (bitnet-b1.58-2B-4T): a SECOND RMSNorm per sub-block, and
    // it sits on the sub-block's intermediate output rather than on the
    // pre-normed input --
    //
    //   x -> attn_norm -> qkv -> attention -> attn_sub_norm -> wo  -> +x
    //   x -> ffn_norm  -> gate/up -> silu(gate)*up -> ffn_sub_norm -> down
    //
    // -- so the plausible reading (norm the layer's input a second time) runs
    // the right weights on the wrong operand and decodes nonsense rather than
    // failing. The WIDTH is the shape that tells them apart, from the pre-norms
    // and from each other: `attn_sub_norm` is n_embd wide because the attention
    // context is, and `ffn_sub_norm` is n_ff wide because the gated FFN
    // intermediate is. Null on every other family, and null is what keeps their
    // graph exactly as it was.
    f32 *attn_sub_norm = nullptr; // f32 [n_embd]
    f32 *ffn_sub_norm = nullptr;  // f32 [n_ff]
    QuantTensor wgate, wup, wdown;

    // ---- qwen4exp: hyper-connections, PLE, indexer (optional) ------------
    // `hc_*` are the two mixers per layer (before the token mixer and before the
    // FFN). `hc_attn_norm` is [n_embd, hc] read lane-major; the down/up pair is
    // the [hc*n_embd -> low_rank -> hc*n_embd] bottleneck and `inject` maps the
    // wide stream to one weight per lane. `ple_*` is the n-gram module on the
    // layers `ple.layers` names, and `idx_*` is the QSA indexer on a full
    // attention layer.
    f32 *hc_attn_norm = nullptr;
    QuantTensor hc_attn_down, hc_attn_up, hc_attn_inject;
    f32 *hc_ffn_norm = nullptr;
    QuantTensor hc_ffn_down, hc_ffn_up, hc_ffn_inject;
    // ---- DeepSeek-V4.1's hyper-connections: the SINKHORN mixer -------------
    // A DIFFERENT mixer from the qwen4exp triple above, and a file carries one
    // or the other -- never both, which the loader enforces. `hc_*_fn` is the
    // full-rank [hc*n_embd -> (2+hc)*hc] projection that emits all three
    // pieces at once, `hc_*_base` its [mix_dim] bias and `hc_*_scale` the [3]
    // per-piece gains. See hc_sinkhorn_* in krk/hc.hpp for the algebra and
    // hc_sinkhorn_split() for the layout trap (the file's comb block is
    // dst-fastest, the host code keeps dst-major rows).
    QuantTensor hc_attn_fn, hc_ffn_fn;
    f32 *hc_attn_base = nullptr;   // f32 [mix_dim]
    f32 *hc_ffn_base = nullptr;
    f32 *hc_attn_scale = nullptr;  // f32 [3]
    f32 *hc_ffn_scale = nullptr;
    // ---- Engram (V4.1): conditional memory ---------------------------------
    // `engram_kv` maps the gathered n-gram rows to (hc + 1) blocks of n_embd
    // -- the first `hc` are the per-lane keys, the last is the value that gets
    // added. `engram_gate` is [n_embd, hc]. Both only on the layers
    // `ple.layers` names.
    QuantTensor engram_kv;
    // The gate is applied ELEMENTWISE to a product of two norms, so it has to
    // be f32: the reference casts a quantized gate to F32 for that reason
    // ("quantized copy has no binary-op kernel", deepseek41.cpp). Loaded as a
    // dequantized weight rather than kept in its file type.
    f32 *engram_gate = nullptr;
    bool ple = false;
    QuantTensor ple_key, ple_value, ple_conv1d;
    f32 *ple_norm_key = nullptr;
    f32 *ple_norm_query = nullptr;
    f32 *ple_norm_conv = nullptr;
    i32 qsa_ratio = 0; // > 0 when this layer carries the QSA indexer
    QuantTensor idx_q_proj, idx_k_proj;
    f32 *idx_q_norm = nullptr;
    f32 *idx_k_norm = nullptr;
    // HOST copies of the two indexer norms. The selection runs on the host --
    // the pooled keys, the scores and the top-k are all host work, because the
    // block scores have to be turned into a per-row cell list before any
    // kernel can use them -- so these are read where the selection reads them
    // rather than downloaded from the device copy once per forward.
    std::vector<f32> idx_q_norm_h, idx_k_norm_h;

    // ---- Mixture-of-Experts (optional) -----------------------------------
    // `moe` selects the routed-expert FFN path. `router` is the small gating
    // matrix [n_expert, n_embd] and is always resident. The routed experts are
    // described by `experts` and materialized lazily by ExpertCache.
    bool moe = false;
    QuantTensor router;                           // ffn_gate_inp.weight
    // Per-expert selection bias (laguna), added to the router's probabilities
    // before the top-k and deliberately not to the weights.
    //
    // HOST memory, and that is load-bearing rather than incidental: the top-k it
    // feeds runs on the host, on the router row this engine already downloads.
    // It was declared next to the QuantTensors and filled with upload_f32(),
    // which returns a DEVICE pointer, so `rb[e]` in the selection score read
    // VRAM from the CPU and took the process down. The bias was never loaded --
    // the tensor name did not match -- so the path had never run and the bug sat
    // in it unnoticed. n_expert floats is 1 KiB a layer; there is no reason for
    // it to be on the device at all.
    f32 *router_bias = nullptr;  // HOST f32[n_expert], owned; exp_probs_b.bias
    // Per-expert FFN biases (`ffn_gate_exps.bias` / `ffn_up_exps.bias` /
    // `ffn_down_exps.bias`, gpt-oss): whole tables [n_expert, n_ff] (or
    // [n_expert, n_embd] for down), DEVICE memory, indexed by expert rather
    // than by token. They are applied after each projection and before the
    // activation, which is where the reference puts them -- so a layer that
    // carries them cannot take the fused gate/silu pair, whose output is
    // already post-activation.
    f32 *exp_gate_bias = nullptr;  // [n_expert * n_ff_exp]
    f32 *exp_up_bias = nullptr;    // [n_expert * n_ff_exp]
    f32 *exp_down_bias = nullptr;  // [n_expert * n_embd]
    // Per-head attention output gate: one scalar per query head, off the
    // pre-attention hidden state.
    QuantTensor wattn_gate;                       // attn_gate.weight [n_embd, n_head]
    // The shared expert's NVFP4 global scales live on each tensor's `gs` (see
    // QuantTensor), not in fields beside them: one rule for every weight.
    QuantTensor shexp_gate, shexp_up, shexp_down; // shared expert, if any
    QuantTensor shexp_inp_gate;                   // [n_embd] per-token gate
    ExpertSource experts;                         // lazily-loaded routed experts
    // ---- LatentMoE (Nemotron 3.5 Bolt) -----------------------------------
    // The routed experts do NOT read the hidden state on this family: the block
    // projects n_embd -> n_latent (`ffn_latent_down`), runs the bank in that
    // narrow space and projects back (`ffn_latent_up`). The shared expert and
    // everything outside the routed leg stay full width, which is why only the
    // experts are latent. Both tensors present, or neither: half a pair is a
    // layout the forward cannot run and the loader refuses it.
    QuantTensor latent_down, latent_up;           // [n_embd, n_latent] / [n_latent, n_embd]
    bool latent_moe() const {
        return latent_down.present() && latent_up.present();
    }
    // ---- MLA legacy (DeepSeek-V2-Lite: direct q, MQA kv, unsplit wkv_b) --
    // attn_q is the direct per-head query (no q_lora trunk on lite);
    // attn_kv_a_mqa emits [kv_lora + rope] (latent plus decoupled key);
    // attn_kv_a_norm norms the latent at write; attn_kv_b is the UNSPLIT
    // legacy table [kv_lora, n_head*(nope+v)]. wo is the shared output.
    // This layer runs multi-head latent attention. Set by the loader from the
    // tensors the block carries; the geometry that follows from it (a
    // kv_lora+rope cache row, no per-head K/V) lives on Model, because it is
    // the same question `layer_has_kv()` asks.
    bool mla = false;
    QuantTensor wq_direct;    // [n_embd, n_head*(nope+rope)]
    QuantTensor wkv_a_mqa;    // [n_embd, kv_lora + rope]
    f32 *wkv_a_norm = nullptr;  // DEVICE f32[kv_lora], like every norm
    QuantTensor wkv_b;        // [kv_lora, n_head*(nope+v_dim)]
    // The same table, dequantized to f16 in the FILE's own element order
    // ([kv_lora, n_head*(nope+v_dim)] row-major: feature j at row j, the latent
    // k its fastest axis), because MLA reads it as 27 small matrices folded
    // into score and value -- a shape `gemm()` cannot address, since head h
    // owns a COLUMN slice of a row-major matrix and the rows interleave the
    // heads. Staged once at load, so the kernel's indices are the file's
    // indices and a mismatch is arithmetic rather than a different weight.
    QuantTensor wkv_b16;

    // ---- MLA modern split (DeepSeek-V3 class / V4) ------------------------
    // The query trunk: attn_q_a [n_embd, q_lora] -> attn_q_a_norm ->
    // attn_q_b [q_lora, n_head*(nope+rope)]. The latent KV path is one of two
    // shapes, told apart by the tensors the layer carries:
    //
    //   Split   attn_kv_a [n_embd, kv_lora+rope] with the rms norm over that
    //           whole row, and the per-head K and V as two separate tables
    //           attn_k_b [kv_lora, n_head*nope] and attn_v_b
    //           [kv_lora, n_head*v_dim] (the V3 class);
    //   Shared  ONE shared latent attn_kv [n_embd, key_length] with the norm
    //           over it, the rope living in its TAIL rather than beside it
    //           (V4).
    //
    // The output projection follows the same split: a plain attn_output on V3,
    // a grouped pair attn_output_a [n_embd, o_lora_rank*n_groups] and
    // attn_output_b [o_lora_rank*n_groups, n_embd] on V4.
    QuantTensor wq_a;          // [n_embd, q_lora]
    f32 *wq_a_norm = nullptr;  // DEVICE f32[q_lora], the query trunk's norm
    QuantTensor wq_b;          // [q_lora, n_head*(nope+rope)]
    QuantTensor wkv;           // Shared: [n_embd, key_length]
    f32 *wkv_norm = nullptr;   // the latent's rms gamma
    QuantTensor wkv_a;         // Split: [n_embd, kv_lora+rope]
    QuantTensor wk_b;          // Split: [kv_lora, n_head*nope]
    QuantTensor wv_b;          // Split: [kv_lora, n_head*v_dim]
    QuantTensor wo_a;          // Shared: [n_embd, o_lora_rank*n_groups]
    QuantTensor wo_b;          // Shared: [o_lora_rank*n_groups, n_embd]
    // The five tables above, dequantized to f16 once at load in the FILE's own
    // element order -- the same staging and the same reason as `wkv_b16` on
    // the legacy path. The shared-latent ops read the grouped output as
    // per-group slices of a [heads_per_group*key_length, o_groups*o_lora]
    // matrix, which `gemm()` cannot address: a group owns a COLUMN block and
    // its rows are that group's concatenated heads, so the two are interleaved
    // in a way no output-stride argument expresses. The host oracle the device
    // is judged against reads these same f16 numbers, so the query trunk and
    // the latent are staged too and the whole layer runs one dtype.
    //
    // These are ALLOCATIONS, not views. On V4's real geometry
    // (n_embd 4096, key_length 512, 8 groups, o_lora 1024) the two output
    // tables are 128 MiB a layer and a 43-layer file stages ~5.4 GiB. A VRAM
    // budget has to carry that, which is why the loader reports the number
    // rather than letting it appear as unexplained residency.
    QuantTensor wq_a16;        // [n_embd, q_lora]
    QuantTensor wq_b16;        // [q_lora, n_head*key_length]
    QuantTensor wkv16;         // [n_embd, key_length]
    QuantTensor wo_a16;        // [heads_per_group*key_length, o_groups*o_lora]
    QuantTensor wo_b16;        // [o_groups*o_lora, n_embd]

    // ---- DeepSeek-V4 compressed attention (CSA/HCA), optional -----------
    // Host f32 copies, in the file's [in, out] order. The state rows and the
    // pool are computed by the host oracle in src/dsv4_compress.cpp -- they are
    // per-token projections of the layer's input, not GEMM operands the engine
    // tiles, and the tables are small (on V4-Flash: 4096x1024 for the CSA
    // wkv/wgate pair against 67 MiB for the query trunk alone). A layer with
    // compress_ratio 0 carries none of them.
    //
    // `comp_*` is the attention's own compressor. `idx_*` is the lightning
    // indexer's, which exists only on CSA layers and is always TWO-wide (its
    // pool is the overlapping one) -- that is the reference's own shape, not a
    // choice this engine makes.
    std::vector<f32> comp_wkv;    // [n_embd, coff*hd]
    std::vector<f32> comp_wgate;  // [n_embd, coff*hd]
    std::vector<f32> comp_ape;    // [coff*hd, ratio]
    std::vector<f32> comp_norm;   // [hd]
    std::vector<f32> idx_proj;    // [n_embd, idx_n_head]
    std::vector<f32> idx_q_b;     // [q_lora, idx_n_head*idx_head]
    std::vector<f32> idx_wkv;     // [n_embd, 2*idx_head]
    std::vector<f32> idx_wgate;   // [n_embd, 2*idx_head]
    std::vector<f32> idx_ape;     // [2*idx_head, ratio]
    std::vector<f32> idx_norm;    // [idx_head]

    // The hash route's expert-id table (DeepSeek-V4): `ffn_gate_tid2eid.weight`
    // is an INT32 tensor -- ggml type 26 -- holding one row per selected
    // expert and one column per token id. It is DATA, not a weight: it is
    // read as integers and looked up, never multiplied. Host memory, because
    // the route is a host-side lookup (the same reason the router bias is).
    // Empty on every other family.
    //
    // The copy is the file's own flat array, and GGUF's ne[0] is the FASTEST
    // axis, so the layout is TOKEN-major: `tid2eid[token * tid2eid_per_token
    // + slot]` is the expert id the file names for that token and that slot.
    // The two counts are the tensor's ne[0] and ne[1] -- the per-token slot
    // count and the token count -- and they are what the loader checks the
    // file's metadata against.
    std::vector<i32> tid2eid;
    i32 tid2eid_per_token = 0;
    i32 tid2eid_n_tokens = 0;

    // ---- gated delta net (Qwen3.5 / Qwen3-Next, optional) ---------------
    // A recurrent layer replaces attention with a short depthwise convolution
    // over the projected [q|k|v] plus a delta-rule state update, so its
    // geometry is unrelated to the head counts. A full-attention layer in the
    // same stack is ordinary attention whose query projection additionally
    // packs a per-head output gate (unpacked by the engine before attention).
    bool gdn = false;
    QuantTensor wqkv;        // [n_embd, conv_dim]  -> [q | k | v]
    QuantTensor wqkv_gate;   // [n_embd, value_dim]  the z gate
    QuantTensor ssm_conv1d;  // [ksize, conv_dim]    depthwise
    // The mamba-2 checkpoint carries a conv bias; the gated delta net's does
    // not, so this is null there and the op adds nothing. HOST-loaded f32.
    f32 *ssm_conv_bias = nullptr; // [conv_dim]
    QuantTensor ssm_out;     // [value_dim, n_embd]
    f32 *ssm_dt = nullptr;   // [n_v_head]  timestep bias (no .weight suffix)
    f32 *ssm_a = nullptr;    // [n_v_head]  stored as -exp(A_log)
    // ---- Nemotron Mamba-2 SSM layer (optional) ---------------------------
    // nemotron_h_moe uses Mamba-2 blocks which have different tensors than GDN:
    // ssm_in (input projection), ssm_d (optional D parameter), no ssm_alpha/beta.
    bool nemotron_ssm = false;  // This layer is a Mamba-2 SSM block (not GDN)
    // ---- LFM2.5 short convolution (`lfm2moe`) ---------------------------
    // A third recurrent block, and the only one that is not a scan: the whole
    // mixer is a depthwise causal convolution over the last `kern` rows of its
    // own input, gated by one of the two branches its projection also emits.
    //   B, C, X = in_proj(x)            [n, 3*n_embd], three 2048-wide slices
    //   y[t]    = sum_k conv[k, c] * X[t - (kern-1) + k, c]
    //   z       = B * y + C * X
    //   out     = out_proj(z)
    // No activation between the conv and the gate, which is what separates it
    // from the gated delta net's `conv1d_silu`.
    bool lfm_conv = false;
    QuantTensor sc_in;          // [n_embd, 3*n_embd]  B | C | X
    QuantTensor sc_out;         // [n_embd, n_embd]
    // The depthwise taps, already in the `w[k*ch + c]` layout every conv in
    // this engine uses (`upload_conv1d`). F32 in every file that has one --
    // 24 KiB a layer -- and the loader refuses another type rather than
    // reinterpreting it.
    QuantTensor sc_conv;        // [kern * n_embd]
    QuantTensor ssm_in;         // [inner_size, n_embd] - Mamba-2 input projection
    // NVFP4's global scale needs no field of its own: it is a property of the
    // TENSOR, so it lives on QuantTensor::gs -- see ssm_in.gs / ssm_out.gs.
    f32 *ssm_d = nullptr;       // [dt_rank] - Mamba-2 D parameter (optional)
    // Projected per head, so these are matmul weights like any other (the
    // loader downcasts an F32 checkpoint tensor to F16).
    QuantTensor ssm_alpha;  // [n_embd, n_v_head]
    QuantTensor ssm_beta;   // [n_embd, n_v_head]
    // Plain RMSNorm weight over d_state. The converter adds 1 to every norm in
    // the model EXCEPT this one, so it is uploaded verbatim, not via the
    // zero-centred path.
    f32 *ssm_norm = nullptr;
    // ---- prism.hadamard folded inputs (false on every other model) -------
    // Which of this layer's matmuls read a ROTATED weight, so their input has
    // to be transformed first (Backend::hadamard_act, plan above). They are
    // per site rather than one flag because the same buffer feeds weights that
    // are NOT folded: the attention-norm output also feeds ssm_alpha/ssm_beta
    // and a MoE router untouched, so the transformed copy goes to a scratch
    // buffer and only the folded matmul reads it.
    bool h_attn_in = false;   // attn-norm out -> attn_q / attn_k / attn_v /
                              //                 attn_qkv
    bool h_attn_gate = false; // attn-norm out -> attn_gate (Qwen3.5's packed
                              //                 gate, laguna's separate one)
    bool h_attn_out = false;  // attention result -> attn_output
    bool h_ffn_in = false;    // ffn-norm out -> ffn_gate / ffn_up
    bool h_ffn_down = false;  // silu(gate)*up -> ffn_down
    bool h_ssm_out = false;   // delta-rule out -> ssm_out (grouped-V perm)
};

// The final-logit bound some archs declare (`final_logit_softcapping`). A free
// function so the formula has one home and can be checked on its own; the engine
// applies it in the one place logits become host-visible.
inline f32 apply_logit_softcap(f32 x, f32 cap) {
    return cap > 0.0f ? cap * std::tanh(x / cap) : x;
}

// Which MLA attention layout a file carries. The three are told apart by the
// TENSORS a layer holds, not by the arch name: an arch can ship more than one
// and a file is what it carries.
//
//   Legacy  attn_q + attn_kv_a_mqa + attn_kv_b          (DeepSeek-V2-Lite)
//   Split   attn_q_a/attn_q_b + attn_kv_a + attn_k_b/attn_v_b
//                                                      (the V3 class)
//   Shared  attn_q_a/attn_q_b + attn_kv + attn_output_a/attn_output_b
//                                                      (DeepSeek-V4)
//
// The difference is the KV path, which is the whole point of MLA. Legacy and
// Split compress the KV into a latent and unfold it per head at runtime;
// Shared keeps ONE latent for every head and folds the head count into a
// GROUPED output projection instead.
enum class MlaLayout { None, Legacy, Split, Shared };

// The layout's name, for a refusal message. "none" for MlaLayout::None.
const char *mla_layout_name(MlaLayout l);

// The DeepSeek V2/V3 YaRN attention scale: mscale^2 / sqrt(key_length), with
// mscale = 1 + 0.1*mscale_all_dim*ln(factor) and mscale_all_dim recovered from
// the file's `rope.scaling.yarn_log_multiplier` (which the converter writes as
// 0.1*mscale_all_dim). `log_mul` is the RAW key, not the divided one.
//
// `deepseek_yarn_mscale` returns the bare multiplier (1.0 -- exactly, not
// approximately -- when the file is not YaRN-scaled or carries no key), which
// is what ModelConfig stores; `deepseek_kq_scale` is the finished number.
//
// Free functions, not members, because they are the thing under test: the
// whole point of B7 is that this arithmetic can be checked against the
// reference's without a model in the room.
f32 deepseek_yarn_mscale(f32 factor, f32 log_mul);
f32 deepseek_kq_scale(f32 factor, f32 log_mul, i32 key_length);

struct ModelConfig {
    std::string arch = "llama";
    std::string name;
    i32 n_vocab = 0;
    i32 n_embd = 0;
    i32 n_layer = 0;
    // Next-token-prediction (MTP) blocks present in the file beyond n_layer.
    // They are prediction heads, not transformer layers, and are not loaded.
    i32 n_layer_nextn = 0;
    i32 n_head = 0;
    i32 n_head_kv = 0;
    i32 head_dim = 0;
    i32 n_ff = 0;
    i32 n_ctx_train = 0;
    f32 rope_base = 10000.0f;
    f32 rope_scale = 1.0f;      // position scaling for linear RoPE
    // Per-layer position divisor, when the stack's windowed half rotates with
    // plain rope (ArchSpec::swa_plain_rope) and the file scales linearly.
    // Linear scaling divides the POSITION, so unlike the frequency tables it
    // cannot be re-derived per layer kind from one stack-wide value: the
    // windowed layers would rotate every position by the full layer's factor.
    // Empty means "use rope_scale everywhere", which is every other stack.
    std::vector<f32> rope_scale_layer;
    // NOTE: there is deliberately no `rope_attn_scale` here. The YaRN
    // magnitude (mscale) is NOT folded into the attention scale by this
    // engine: the frequency half of YaRN is folded into the table (see
    // Model::load's build_yarn), and the magnitude half is left out on the
    // measurement that a converted checkpoint's `yarn_attn_factor` already
    // cancels it. A field that nothing assigns is a promise the graph does not
    // keep, so the multiply is gone rather than carried at 1.0. The DeepSeek
    // family's `mscale^2` correction is tracked separately (docs/TODO-
    // alignment-2026-10-29.md, B7) because it needs the file's
    // `yarn_log_multiplier` and is not a constant.
    f32 rope_frac = 1.0f;       // fraction of head dims rotated (Phi partial RoPE)
    f32 rms_eps = 1e-5f;
    // Gemma 2/3/4 bound the output logits with cap * tanh(logit / cap). The
    // transform is monotone, so argmax and every sampler that only compares
    // probabilities are unaffected by *ranking* — it matters for the value of
    // the log-probs a sampler sees and for anything that mixes logits from
    // different sources. 0 disables it. See apply_logit_softcap.
    f32 logit_softcap = 0.0f;
    // Gemma 2 also bounds the ATTENTION logits: cap * tanh(score / cap) between
    // the QK dot product and the softmax (`attention.logit_softcapping`, 50 on
    // every gemma2 file). It is a different function from the final-logit one
    // above -- it changes the probabilities, not just their log-scale -- so it
    // belongs inside attention (AttnDesc::softcap). 0 disables it.
    f32 attn_softcap = 0.0f;
    // muse-glimmer multiplies the head's output by `<arch>.logit_scale` BEFORE
    // the final-logit softcap (`cur = ggml_scale(cur, hparams.f_logit_scale)` in
    // src/models/muse-glimmer.cpp). It is not the same thing as the softcap and
    // it does not divide out: with a cap the composed function is
    // cap*tanh(s*x/cap), not s*cap*tanh(x/cap), and even without one the scale
    // changes the temperature-relative distribution the sampler draws from. The
    // reference REQUIRES the key on that family (`get_key` with no optional
    // flag), so a muse-glimmer file always carries it; every other family
    // leaves it at 1.0 and the multiply is skipped.
    f32 logit_scale = 1.0f;
    // OLMo-1 bounds the attention projections: `attention.clamp_kqv` (8.0 in
    // every OLMo-1 config, `clip_qkv` on the HF side) clamps q, k and v to
    // [-c, +c] after the projection and the bias, before the rotary. 0 means
    // the file declares no clamp, which is every other family.
    f32 attn_clamp_kqv = 0.0f;
    // The head-input multiplier: MiniCPM3 scales the normed row by
    // `n_embd_base/n_embd` (256/n_embd) between the final norm and `output`.
    // Applied in Engine::head_compute, which is the reference's order. 1.0
    // skips it.
    f32 lm_head_scale = 1.0f;
    // The layer's residual scale also applies to the FFN branch's output, not
    // only the attention branch's (MiniCPM3 scales both). false is the
    // attention-only shape every other family has.
    bool residual_scale_ffn = false;
    bool tied_embeddings = false;
    bool qk_norm = false;
    // OLMoE and its relatives normalize Q/K *before* the head split: one RMS
    // over the whole projected [n_embd] row, weight width n_embd. Qwen3 and
    // Gemma3 normalize per head, weight width head_dim. The two are not
    // interchangeable and the difference is silent -- both backends share this
    // one op, so reading a 2048-wide weight as 16 heads of 128 throws away
    // 15/16 of it and yields a fluent-but-wrong model on *every* path. The
    // stored tensor width is what tells them apart (see model.cpp).
    bool qk_norm_wide = false;

    // ---- per-branch post norms (gemma2/3/4, muse-glimmer) ------------------
    // `x + norm(attn_out)` rather than `x + attn_out`: each sub-block's output is
    // RMS-normed with its OWN weight before it reaches the residual, which also
    // means the projection cannot fold its residual store into the GEMM epilogue
    // (the norm has to read the projection first).
    bool post_branch_norm = false;
    // Every norm in the stack is weightless and the file carries no norm
    // tensors at all (OLMo-1; see ArchSpec::noweight_norm). The loader stops
    // requiring `attn_norm`/`ffn_norm`/`output_norm` and the engine hands the
    // null weight to Backend::rmsnorm, which is its weightless case.
    bool noweight_norm = false;
    f32 post_norm_eps = 0.0f; // 0 keeps rms_eps (muse-glimmer's are 1e-8)
    // The gate activation differs by family while the tensor shape does not:
    // laguna softplus's `attn_gate`, muse-glimmer sigmoids it.
    bool attn_gate_sigmoid = false;
    // The gate is per ELEMENT of the attention row, not per head (muse-glimmer).
    bool attn_gate_wide = false;
    // A weightless RMS norm on the embedding before layer 0 (muse-glimmer).
    bool embd_rms_noweight = false;
    // gelu(gate) * up instead of silu(gate) * up (gemma).
    bool gelu_ffn = false;
    // ---- GPT-2 -----------------------------------------------------------
    // LayerNorm instead of RMSNorm (mean-subtracting, with a bias), learned
    // absolute position embeddings, a fused attn_qkv whose three outputs share
    // one projection and one bias, and a plain GELU MLP. See
    // ArchSpec::gpt2_style -- these are one architecture, not four switches.
    bool gpt2_style = false;
    // Fused projections, as named by ArchSpec (see arch.hpp). Both are off for
    // every family that carries separate q/k/v and gate/up tensors.
    bool fused_qkv = false;
    bool fused_gate_up = false;
    // The plain pre-norm family's pieces (see ArchSpec). `gpt2_style` still
    // means the whole GPT-2 bundle; these name the parts so gptneox/gptj/phi2
    // can take what they have and nothing they do not.
    bool ln_bias = false;
    bool learned_pos = false;
    bool parallel_residual = false;
    bool no_gate_ffn = false;
    // see ArchSpec::norm_bias
    bool norm_bias = false;
    // ALiBi's scale, and BLOOM's embedding norm (see ArchSpec).
    f32 alibi_max_bias = 0.0f;
    bool embd_norm = false;
    bool parallel_two_norm = false;
    // Rotate only on the windowed layers (muse-glimmer's full layers are NoPE).
    bool rope_swa_only = false;

    // ---- hybrid full / sliding-window attention (laguna) ------------------
    // Some archs alternate layer kinds. Every layer whose index is not a
    // multiple of `swa_period` attends only the last `swa_window` positions,
    // and carries its own rotary table: laguna-S is 48 layers of period 4
    // (full at il % 4 == 0), window 512, full layers rotating 64 of 128 dims
    // under YaRN and the windowed ones rotating all 128 under plain RoPE.
    // `swa_period` 0 or 1 means every layer attends fully.
    i32 swa_period = 0;
    // The windowed/full pattern stated EXPLICITLY by the file
    // (`attention.sliding_window_pattern`, one entry per layer; gemma4 writes
    // it). When present it is the authority and `swa_period` is not consulted:
    // a period is a guess about the pattern, and a family that declares the
    // pattern is not asking for one to be guessed.
    std::vector<i32> swa_pattern;
    // ---- gemma's embedding scale -----------------------------------------
    // llama.cpp builds the input embedding scaled by sqrt(n_embd) for this
    // family, and it is not a detail: every downstream norm sees a row an
    // order of magnitude away from what the model was trained on, so the
    // residual grows (or shrinks) geometrically with depth. 1.0 means no scale.
    f32 embd_scale = 1.0f;
    // ---- gemma4 per-layer input embeddings (PLE) -------------------------
    // Width of ONE layer's slice; 0 means the family has none. The per-layer
    // token table is [n_layer * n_embd_per_layer, n_vocab] and the model
    // projection is [n_embd, n_layer * n_embd_per_layer].
    i32 n_embd_per_layer = 0;
    // How many of the LAST layers write no KV and attend an earlier layer's
    // cache (`attention.shared_kv_layers`). 0 on every other family.
    i32 n_kv_shared = 0;
    // The value path is RMS-normalized with no weight (gemma4).
    bool v_norm_weightless = false;
    // The family runs the per-layer embedding module (gemma4). Mirrors
    // `ArchSpec::per_layer_embd`, which is where the declaration lives; the
    // loader and the forward read this one so a family that does not have it
    // pays no branch in the hot loop beyond a bool.
    bool per_layer_embd = false;
    // Every layer multiplies its output by `blk.N.layer_output_scale` (gemma4).
    bool layer_out_scale = false;
    // A routed layer of this family is two separately-normed branches plus a
    // detached router input, not the generic MoE block (gemma4). Mirrors
    // `ArchSpec::moe_two_branch`.
    bool moe_two_branch = false;
    // Every Nth layer skips rotary (SmolLM3); 0 rotates every layer.
    i32 no_rope_step = 0;
    // Phase of the pattern: is_swa(l) == ((l + swa_phase) % swa_period) != 0.
    // 0 is the historical (full layer first) reading; 1 flips the pair so the
    // *windowed* layer comes first (gpt-oss).
    i32 swa_phase = 0;
    i32 swa_window = 0;
    i32 rope_dim_swa = 0;
    f32 rope_frac_swa = 1.0f;
    // Defaults to the model's own `rope.freq_base`, not to 10000: a file that
    // declares one rotary base for the whole stack means it for every layer,
    // and a windowed layer silently rotated at 10000 while its full layers used
    // 150000 is a wrong answer that no shape check catches (gpt-oss declares
    // exactly one base, for all 36 layers).
    f32 rope_base_swa = 10000.0f;
    // `n_head` above is the head count of the *full* layers. When a model
    // varies it per layer (`attention.head_count` stored as an array) the whole
    // vector is kept here, and `n_head` is its maximum — which is what the
    // workspaces size to, since the query projection is the widest one.
    // Per-layer head WIDTH. gemma4 declares two attention geometries in one
    // stack: global layers of 512 dims and sliding-window layers of 256
    // (`attention.key_length` vs `attention.key_length_swa`). Empty means every
    // layer uses `head_dim`, which is every other family. The stack maximum is
    // head_dim (workspaces are sized against it) and head_dim_at() is the
    // layer's own -- the same split as n_head_layer/n_head.
    std::vector<i32> head_dim_layer;
    // The attention scale, when the family does not use the default
    // 1/sqrt(head_dim). gemma4 sets f_attention_scale = 1.0 and applies no
    // pre-attention scaling at all, which is not expressible by tweaking the
    // default. 0 means "unset, use the default".
    //
    // Set from ArchSpec::attn_scale and overridden by `<arch>.attention.scale`
    // when the file states one, with the reference's own "0 = default"
    // convention. Both halves were missing until 2026-10-29: Engine's
    // helper already read this field and its comment already named gemma4,
    // but nothing ever ASSIGNED it, so every gemma4 file ran at 1/sqrt(512).
    f32 attn_scale = 0.0f;
    std::vector<i32> n_head_layer;
    // Same idea for the KV side: a per-layer array (nemotron_h_moe: 2 on its
    // six attention layers, 0 on the 46 that have none, gemma4: {1,8}) with
    // n_head_kv holding the maximum, which is what a shared workspace and one
    // KV slot are measured against.
    std::vector<i32> n_head_kv_layer;
    // Per-head softplus output gate (laguna `attn_gate.weight`): one scalar per
    // query head, projected from the *same* hidden state q/k/v read, then
    // softplus'd and multiplied into the attention result before wo. Distinct
    // from the packed query gate of Qwen3.5, which is a sigmoid off the query
    // projection rather than a separate tensor.
    bool attn_gate = false;
    // The query projection packs a per-head output gate beside q (Qwen3.5,
    // qwen4exp): the stored row is 2*n_head*head_dim, the q half first. It is
    // geometry, so it lives here rather than being re-derived from the arch
    // STRING at engine init -- see the note on q_proj_ in Engine::init. An
    // aliased or unlisted label used to answer that question differently from
    // the loader, which changes the graph instead of failing.
    bool q_output_gate = false;
    // The attention branch's output is multiplied by this before it joins the
    // residual (minicpm3: `depth / sqrt(n_layer)`, already divided at load).
    // 1.0 everywhere else, and 1.0 keeps the multiply folded away.
    f32 attn_out_scale = 1.0f;
    // ---- qwen4exp (Qwen3.8-Flash-Next) -----------------------------------
    // The residual is `hc` lanes of n_embd, mixed through a low-rank bottleneck
    // (4 x 2560 through 320 on the REAP-288 file). `ple` is the n-gram hash
    // embedding table, `qsa_ratio` the per-layer block size of the QSA indexer
    // (0 on a layer without one), and `no_out_norm` says the final mixer
    // carries the norm -- there is no output_norm.weight in the file.
    // ---- MoVA (k2-horizon): a routed set of attention values ---------------
    // `n_value_expert` > 0 means every MoE layer builds V by routing to
    // `n_value_expert_used` of them instead of projecting with attn_v.weight.
    // Read from `<arch>.attention.value_expert_count` / `.value_expert_used_count`.
    i32 n_value_expert = 0;
    i32 n_value_expert_used = 0;
    // ---- k2-horizon's GROUPED RMS norm -------------------------------------
    // `<arch>.attention.group_norm_groups`: the norm divides the row into this
    // many equal groups and takes the RMS over each GROUP, not over the whole
    // row, then applies one elementwise weight across all of it. 1 (or 0, the
    // absent key) is the ordinary norm and must keep the fast path.
    i32 n_norm_groups = 1;
    // Router gating for the value experts; shares the FFN's convention
    // (`router_sigmoid`), because llama.cpp reads one `expert_gating_func`.
    bool hc = false;
    i32 hc_n = 0;
    i32 hc_low_rank = 0;
    // ---- V4.1's variant of the same idea ---------------------------------
    // `hc_sinkhorn` says the mixer is the (2+hc)*hc one above rather than
    // qwen4exp's low-rank bottleneck. `single_pass` is V4.1's lag: a sub-block
    // collapses with the pre-mix computed ONE SUB-LAYER EARLIER. With it off,
    // each sub-block computes and uses its own pre-mix (which is what an
    // ablation export looks like). `no_hc_head` says the file has no head
    // mixer at all -- V4.1 collapses the final lane stack with the LAST FFN's
    // pre-mix, and the presence or absence of hc_head_fn is what says which.
    bool hc_sinkhorn = false;
    bool hc_single_pass = false;
    bool hc_no_head = false;
    i32 hc_sinkhorn_iters = 0;
    f32 hc_eps = 0.0f;
    // Engram's token map: `engram.token_map[token]` is the id the hash uses,
    // and a token outside it is DEAD (see krk/engram.hpp).
    std::vector<u32> engram_token_map;
    i64 engram_pad_id = 0;
    bool ple = false;
    i32 ple_ngram = 0;
    i32 ple_heads_per_ngram = 0;
    i32 ple_head_dim = 0;
    i32 ple_conv_kernel = 0;
    i32 ple_eos_token = -1;
    std::vector<i64> ple_multipliers;   // [ngram]
    std::vector<i64> ple_head_offsets;  // [n_heads]
    std::vector<i64> ple_head_vocab_sizes;
    std::vector<i32> ple_layers;        // layer indices carrying a PLE module

    // Is layer `l` one of the layers `ple.layers` names? Used by the loader for
    // both families' per-layer modules, because the LIST is the authority -- a
    // layer carrying the tensors but absent from the list is a malformed file,
    // not a layer to run (see the check in Model::load).
    bool dsv4_engram_layer(i32 l) const {
        for (i32 v : ple_layers)
            if (v == l) return true;
        return false;
    }
    bool qsa = false;
    // ---- MLA (DeepSeek-V2/V3/V4) --------------------------------------
    // The latent geometry, read off the file's own attention metadata. The
    // real keys are q_lora_rank (absent on lite: the query is a direct
    // projection), kv_lora_rank (absent on V4: the latent IS key_length),
    // key_length (full per-head K width: nope + rope) and value_length --
    // NOT qk_nope_head_dim/qk_rope_head_dim/v_head_dim, which no exporter
    // writes and which used to refuse every real file at load. The nope
    // split is key_length - rope.dimension_count. A layer's KV cache row is
    // kv_lora + rope wide (576 floats on V2-Lite: 512 + 64), not
    // n_kv x head_dim x 2, and there are no per-head K/V rows at all -- the
    // per-head projections are absorbed at load.
    bool mla = false;
    i32 mla_q_lora = 0;   // 0 on lite: direct attn_q, no q trunk
    i32 mla_kv_lora = 0;  // kv_lora_rank, or key_length when absent (V4)
    i32 mla_qk_nope = 0;  // derived: key_length - rope_dim
    i32 mla_qk_rope = 0;  // = rope_dim for MLA (the rotation covers the tail)
    i32 mla_v_dim = 0;    // = value_length
    // Which of the three layouts this file carries, decided by the loader
    // from the tensors and required to be the SAME on every layer.
    MlaLayout mla_layout = MlaLayout::None;
    // ---- DeepSeek-V4 extras -------------------------------------------
    // Dense-core grouping (wo_a/wo_b), the sliding window, the hash-route
    // layer count, the compressed-rope base and the per-layer compression
    // ratios (0 = dense). Read for every dsv4 file; the loader refuses a
    // file that does not carry them rather than guessing.
    bool dsv4 = false;
    i32 dsv4_o_groups = 0;
    i32 dsv4_o_lora = 0;
    i32 dsv4_window = 0;
    i32 dsv4_hash_layers = 0;
    f32 dsv4_compress_rope_base = 0.0f;
    std::vector<i32> dsv4_ratio;  // per layer, 0 = dense core
    i32 idx_n_head = 0;
    i32 idx_head_dim = 0;
    i32 idx_top_k = 0;
    std::vector<i32> qsa_ratio;         // per layer, 0 = no indexer
    bool no_out_norm = false;
    // The recurrent layer's output gate is `sigmoid(z)` rather than `silu(z)`.
    // qwen4exp's build_norm_gated is the one numerical difference from Qwen3.5's
    // gated delta net.
    bool gdn_gate_sigmoid = false;
    // RoPE pair convention, read off the arch table (ArchSpec::rope_neox): true
    // pairs channel j with j + rope_dim/2 (NeoX half-split), false pairs the
    // adjacent channels (2i, 2i+1). It travels with the model because it is a
    // property of how the file was exported, not of the kernel, and both the
    // prefill and the decode path have to agree on it.
    bool rope_neox = false;
    // Router shape. `router_bias` is a per-expert selection bias
    // (`ffn_exp_probs_b.bias`) added to the *probs* before the top-k but not to
    // the weights the experts are combined with; `router_sigmoid` selects
    // sigmoid over softmax; the selected weights are then optionally
    // sum-normalized (`expert_w_norm`) and scaled (`expert_w_scale`).
    bool router_bias = false;
    // Where the bias enters. laguna's `exp_probs_b.bias` is added to the
    // *probabilities* (`false`); llama.cpp's `ffn_gate_inp.bias` convention
    // (gpt-oss) adds it to the router LOGITS, before the softmax -- so it
    // changes both the top-k and the weights, and the two are not
    // interchangeable: a monotone map does not commute with a bias added after
    // it, so the same tensor applied in the wrong place selects different
    // experts. Absent means the laguna reading, which is what this engine ran
    // before the flag existed.
    bool router_bias_pre = false;
    bool router_sigmoid = false;
    // The FFN's activation is squared ReLU rather than a gated SiLU
    // (nemotron_h/nemotron_h_moe: the reference builds every FFN of the family
    // with LLM_FFN_RELU_SQR, and its files carry no ffn_gate for the gate-less
    // expert/shared-expert shapes to multiply against).
    bool ffn_relu_sqr = false;
    // gpt-oss's expert activation (SwiGLU-OAI): min(gate, 7) *
    // sigmoid(1.702 * gate) * (clamp(up, -7, 7) + 1). A different function from
    // silu(gate) * up, not a different spelling of it -- see the note on
    // Backend::swiglu_oai.
    bool ffn_swiglu_oai = false;
    // Whether the rope table was built with YaRN's frequency correction, i.e.
    // whether `rope_scale` has already been folded into `inv_freq` and must be
    // left at 1.0 for the rotation op.
    bool rope_yarn = false;
    // The file's rope.scaling.factor and rope.scaling.yarn_log_multiplier --
    // the INPUTS of DeepSeek's YaRN attention scale, which is applied to the
    // MLA score through `mla_kq_mscale` below.
    //
    // The arithmetic, all of it from the reference and none of it inferred
    // (llama.cpp @ f5222f25: src/llama-context.cpp:269-300 builds
    // cparams.yarn_attn_factor, src/models/deepseek2.cpp:431-437 consumes it):
    //
    //   factor        = 1 / rope_freq_scale                 (40 on a V2/V3 file)
    //   mscale_all    = yarn_log_multiplier / 0.1           (the loader stores
    //                                                        0.1*mscale_all_dim)
    //   attn_factor   = get_mscale(factor, mscale)
    //                 / get_mscale(factor, mscale_all)      (== 1 for deepseek)
    //                   * 1/(1 + 0.1*ln(factor))            (the cancel step)
    //   mscale        = attn_factor*(1+0.1*ln(factor))
    //                   * (1 + 0.1*mscale_all*ln(factor))
    //
    // which collapses to  mscale = 1 + 0.1*mscale_all*ln(factor), i.e.
    // `1 + yarn_log_multiplier * ln(factor)` in the file's own units, and the
    // MLA score scale is mscale^2 / sqrt(key_length). DeepSeek's own modelling
    // code (`yarn_get_mscale` in modelling_deepseek_v2.py) computes the same
    // number, which is why this is not "llama.cpp's quirk".
    //
    // It is applied to the V2/V3 path only. deepseek4/deepseek41 pass a plain
    // 1/sqrt(n_embd_head) (src/models/deepseek4.cpp:1195-1206,
    // deepseek41.cpp:427), so the V4 shared-latent path keeps its own scale.
    f32 rope_yarn_factor = 1.0f;
    f32 rope_yarn_log_mul = 0.0f;
    // mscale above, resolved once at load. 1.0 whenever the file is not
    // YaRN-scaled or carries no yarn_log_multiplier, so every non-DeepSeek and
    // every non-YaRN file keeps a scale of exactly 1/sqrt(key_length).
    f32 mla_kq_mscale = 1.0f;
    // 1 = sum-normalize the selected expert weights, 0 = keep the raw
    // probabilities, -1 = the file does not declare the key. The engine has
    // always normalized, and every model in the collection that *does* declare
    // it declares 1, so the absent value has to keep the legacy behaviour
    // rather than silently switching every Qwen MoE to unnormalized weights.
    i32 expert_w_norm = -1;
    f32 expert_w_scale = 1.0f;
    // Leading layers whose FFN is dense rather than MoE
    // (`leading_dense_block_count`). REPORTED, not consumed: the layer loop
    // dispatches on the tensors a layer actually carries (`ffn_gate_inp`
    // present or not), so a mixed stack needs no rule -- see the glm4moe row
    // in src/arch.cpp. Kept because it is a file fact the loader proves it read
    // (test_lfm2moe checks it) and a diagnostic worth printing.
    i32 n_dense_lead = 0;

    // Mixture-of-Experts geometry (is_moe == false for dense models).
    // ---- masked diffusion (LLaDA2.x, `diffuse`) ---------------------------
    // `diffusion` says the stack is BIDIRECTIONAL: every forward is a denoising
    // pass over the whole sequence, so attention has no causal mask and no KV
    // row is reused between steps. The rest is the file's own schedule, read
    // rather than defaulted, because a wrong block length changes which tokens
    // are committed together and therefore the output text.
    bool diffusion = false;
    i32 diffuse_mask_id = -1;    // `diffuse.mask_token_id`
    i32 diffuse_block_length = 0;
    i32 diffuse_steps = 0;       // 0 = the sampler's default
    i32 diffuse_moe_block_size = 0;

    bool is_moe = false;
    // Group-limited selection (DeepSeek-V3 / MiniMax / LLaDA2's Block Routing):
    // experts are contiguous blocks of `n_expert_group`, only the
    // `n_expert_group_used` best groups stay eligible for the top-k. 0 or 1
    // means "no grouping", and the mask helper is then a no-op. See
    // krk/moe_route.hpp for the rule and the three ways to get it wrong.
    i32 n_expert_group = 0;
    i32 n_expert_group_used = 0;
    i32 n_expert = 0;        // routed experts per MoE layer (the CAP)
    i32 n_expert_used = 0;   // top-k experts activated per token
    i32 n_ff_exp = 0;        // routed-expert hidden width
    i32 n_ff_shexp = 0;      // shared-expert hidden width (0 when absent)
    i32 n_expert_shared = 0; // always-on shared experts per layer
    // Per-layer routed-expert top-k. `expert_used_count` is a scalar in every
    // Qwen/DeepSeek file, but a packed hybrid merge (Nemotron Bolt 74B-A9B)
    // writes it as one entry per block -- 0 on the Mamba-2 blocks, 9 on the
    // MoE ones -- so a single global number cannot describe it. Empty means
    // every layer uses `n_expert_used`. The per-layer expert COUNT needs no
    // vector here: the routed bank's own ne[2] is each layer's truth and lives
    // on LayerWeights::experts, with `n_expert` demoted to a cap.
    std::vector<i32> layer_n_expert_used;

    // ---- gated delta net (Qwen3.5 / Qwen3-Next) --------------------------
    // `recurrent` is true when at least one layer is a delta-net layer, which
    // is what makes the engine allocate and carry the recurrent state.
    bool recurrent = false;
    // LFM2.5: the recurrent layers are short convolutions, not delta nets. The
    // FILE decides each layer's kind (it carries `shortconv.in_proj` on exactly
    // the conv ones), so this flag only says the stack is that family -- the
    // index arithmetic of `full_attention_interval` does not describe it.
    bool lfm_conv = false;
    i32 full_attention_interval = 4; // every Nth layer keeps real attention
    i32 ssm_d_state = 128;           // state width == key/value head dim
    i32 ssm_dt_rank = 32;            // value heads
    i32 ssm_n_group = 16;            // key heads
    i32 ssm_d_conv = 4;              // short-convolution kernel taps
    i32 ssm_key_dim = 0;             // ssm_n_group * ssm_d_state
    i32 ssm_value_dim = 0;           // ssm_dt_rank * ssm_d_state
    i32 ssm_conv_dim = 0;            // ssm_key_dim*2 + ssm_value_dim
    // ---- nemotron_h_moe (Mamba-2 + MoE) -----------------------------------
    // nemotron_h_moe uses Mamba-2 blocks (different from GDN) alternating with
    // MoE expert layers. Even layers are Mamba-2 SSM, odd layers are experts.
    bool nemotron_moe = false;        // This is a nemotron_h_moe model
    bool has_d_param = false;         // ssm_d parameter present (Mamba-2 specific)
    i32 ssm_inner_size = 0;           // Mamba-2 inner projection dimension
    // Rotated width when it is narrower than head_dim (partial RoPE). 0 means
    // the whole head is rotated.
    i32 rope_dim = 0;
    // Interleaved multi-rotary section sizes over the rotated pairs. Recorded
    // for completeness; with a single text stream every position is equal, so
    // the sections collapse to one uniform rotation (see the engine's rope
    // call). 0 when the model does not use them.
    i32 rope_sections[4] = {0, 0, 0, 0};
};

// A model's next-token-prediction (MTP/nextn) block: the extra block a
// Qwen3.5-family file appends after the stack, which the engine does NOT run as
// a layer. It is a full attention + MoE block plus three adapters that fuse the
// target's hidden state with the last token's embedding:
//
//   x = eh_proj( cat( hnorm(hidden), enorm(embed(t)) ) )
//   x = attn(x) -> moe(x) -> shared_head_norm(x) -> the model's own output head
//
// Loaded only when asked for (KRK_MTP_LOAD=1, or a future --mtp-draft): the
// experts alone are ~855 MB on Tiel-35B-A3B, so a plain run must not pay for
// them. Nothing in the main stack reads this struct.
struct MtpBlock {
    bool present = false;
    i32 layer = -1;      // the file's block index (blk.<layer>.*)
    i32 n_embd = 0;
    i32 n_head = 0;
    i32 n_head_kv = 0;
    i32 head_dim = 0;
    i32 n_expert = 0;
    i32 n_expert_used = 0;
    i32 n_ff_exp = 0;
    i32 n_ff_shexp = 0;
    i32 n_ff = 0;        // dense FFN width (0 when the block is MoE-only)
    size_t bytes = 0;    // weights uploaded, for the report
    // The block itself.
    QuantTensor wq, wk, wv, wo;
    QuantTensor router;
    QuantTensor shexp_gate, shexp_up, shexp_down;
    // Qwen3.5 scales the shared expert by a sigmoid of a per-token projection
    // (`ffn_gate_inp_shexp`), the same gate the stack's own layers carry.
    QuantTensor shexp_inp_gate;
    // A block whose FFN is dense carries ffn_gate/up/down instead of routed
    // banks. Both shapes load, and the forward runs the one the file has: the
    // qwen35 dense family's nextn heads are this shape, and running attention
    // over a block whose FFN was never loaded is a fluent wrong answer.
    QuantTensor ffn_gate, ffn_up, ffn_down;
    f32 *attn_norm = nullptr;
    f32 *post_attn_norm = nullptr;
    f32 *q_norm = nullptr;
    f32 *k_norm = nullptr;
    f32 *router_bias = nullptr; // HOST, like LayerWeights::router_bias
    // The nextn adapters and the shared head's norm.
    QuantTensor eh_proj;        // [2*n_embd, n_embd]
    f32 *enorm = nullptr;
    f32 *hnorm = nullptr;
    f32 *shared_head_norm = nullptr;
    // Routed experts are NOT uploaded here: they go through ExpertCache like
    // any other layer's, once the drafter runs (see the mtp plan in docs).
    const GgufTensor *exps_gate = nullptr;
    const GgufTensor *exps_up = nullptr;
    const GgufTensor *exps_down = nullptr;
};

// One dense tensor that is NOT resident: its source in the GGUF mapping and
// the address inside the streaming ring it is staged at. A streamed layer's
// QuantTensors point into that ring, so layer decisions (present(), dtypes,
// shapes) are unchanged and only the bytes move.
struct StreamEntry {
    const GgufTensor *src = nullptr;
    void *dst = nullptr;
    size_t bytes = 0;
    // The layer tensor that points at `dst`, bound after the layer is built.
    // Promotion re-points it into a permanent allocation, so this is what
    // moves the layer's own weights rather than the ring's copy of them.
    QuantTensor *q = nullptr;
};

class Model {
public:
    Model() = default;
    ~Model() = default;
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;

    bool load(Backend &be, const std::string &path, std::string *err);
    void unload();

    // ---- dense weight residency ------------------------------------------
    // Budget for dense weights in device memory, in bytes, set BEFORE load().
    // 0 (the default) is automatic: keep everything resident when it fits and
    // stream whole layers from the file once it does not, leaving VRAM for KV
    // and workspaces. A positive value is an explicit ceiling. The CPU backend
    // never streams (there is no device/host split to page across).
    void set_weight_budget(size_t bytes) { weight_budget_ = bytes; }
    // Upload the file's nextn (MTP) block. Set BEFORE load(), like the weight
    // budget: the block is parsed once, during load, so a caller that asks
    // afterwards would be asking a loader that has already run. KRK_MTP_LOAD=1
    // is the environment spelling; --mtp-probe sets this.
    void set_mtp_load(bool on) { mtp_load_ = on; }
    // The nextn block can also arrive as its OWN file: a `-noMTP` quant of a
    // model whose MTP was published separately (Nemotron 3.5 Lightning ships
    // `mtp-<model>.gguf` beside a `-noMTP` trunk). The sidecar carries the same
    // `blk.<n>.*` tensors the in-file block does, at the block index past the
    // trunk's stack, plus its own copy of the config; this loader reads the
    // block out of it and keeps the mapping alive for the routed banks.
    // Set BEFORE load(), like set_mtp_load: the block is parsed once.
    void set_mtp_path(std::string path) { mtp_path_ = std::move(path); }
    const std::string &mtp_path() const { return mtp_path_; }
    // Refill the streaming ring with layer l's weights. A no-op for a resident
    // layer, so the engine's layer loop calls it unconditionally.
    void stage_layer(i32 l);
    // The nextn/MTP block, when this file carries one and it was asked for.
    const MtpBlock &mtp() const { return mtp_; }
    // A streamed layer is promoted into an allocation of its own the first
    // time the engine needs it, when measured free VRAM can hold it beside the
    // reserve. Streaming is then only what genuinely does not fit, which is a
    // measurement rather than a load-time guess.
    size_t promoted_layers() const { return promoted_layers_; }
    size_t promoted_bytes() const { return promoted_bytes_; }
    f64 promote_ms() const { return promote_ms_; }
    size_t streamed_bytes() const { return stream_bytes_total_; }
    size_t streamed_layers() const { return streamed_layers_; }
    size_t ring_bytes() const { return ring_bytes_; }
    u64 stage_calls() const { return stage_calls_; }
    u64 stage_bytes() const { return stage_bytes_; }
    f64 stage_ms() const { return stage_ms_; }
    // One-hop-ahead staging: the ring slot count (1 when it is off), and how the
    // copies that filled a slot were paid for -- in flight under the previous
    // layer (`ahead_copies`), or synchronously because the pipeline had no
    // prefetched slot to consume (`ahead_cold`). A cold count that grows with
    // the token count means the ring is not actually running ahead.
    int ring_slots() const { return ring_slots_; }
    u64 ahead_copies() const { return ahead_copies_; }
    u64 ahead_cold() const { return ahead_cold_; }
    f64 ahead_wait_ms() const { return ahead_wait_ms_; }
    // Where a staged layer's wall time went: the drain that orders ring reuse,
    // the file read into pinned staging, and the host->device copy. Only useful
    // as a split -- a combined total cannot say which half to fix.
    f64 stage_sync_ms() const { return stage_sync_ms_; }
    f64 stage_read_ms() const { return stage_read_ms_; }
    f64 stage_xfer_ms() const { return stage_xfer_ms_; }
    // The pinned host mirror: its total bytes, and how long filling it took at
    // load. A non-zero size means staging no longer reads the file per token.
    size_t stream_host_bytes() const {
        return stream_host_ ? stream_host_size_ : 0;
    }
    f64 stream_host_fill_ms() const { return stream_host_fill_ms_; }
    // Read time that was NOT paid inside a staging pass: the next layer was
    // read into the spare pinned buffer while this one's copy was in flight.
    // It is wall time the pass would otherwise have spent, so it is reported
    // beside the split rather than inside it.
    f64 stage_prefetch_ms() const { return stage_prefetch_ms_; }
    bool layer_streamed(i32 l) const {
        return l >= 0 && l < static_cast<i32>(layer_streamed_.size()) &&
               layer_streamed_[static_cast<size_t>(l)] != 0;
    }

    i64 n_ctx() const { return cfg_.n_ctx_train; }
    // Widest KV row on the whole stack. An MLA row is latent + decoupled key,
    // which need not be wider or narrower than n_kv x head_dim (V2-Lite: 576
    // against 3072), so the maximum is what anything sized off this must hold.
    i64 kv_dim() const {
        const i64 head_geom = static_cast<i64>(cfg_.n_head_kv) * cfg_.head_dim;
        if (cfg_.mla_kv_lora <= 0) return head_geom;
        const i64 mla = static_cast<i64>(cfg_.mla_kv_lora) + cfg_.mla_qk_rope;
        return mla > head_geom ? mla : head_geom;
    }
    // Widest query projection any layer asks for. A model with a per-layer head
    // count (laguna) is smaller on some layers than others, so this is what a
    // shared workspace has to be sized against; the width of one particular
    // layer is q_dim_at().
    i64 q_dim() const { return static_cast<i64>(cfg_.n_head) * cfg_.head_dim; }

    // ---- per-layer geometry (laguna alternates two layer kinds) ----------
    i32 n_head_at(i32 layer) const {
        if (cfg_.n_head_layer.empty()) return cfg_.n_head;
        if (layer < 0 || layer >= static_cast<i32>(cfg_.n_head_layer.size()))
            return cfg_.n_head;
        return cfg_.n_head_layer[static_cast<size_t>(layer)];
    }
    // The layer's own head width (see cfg_.head_dim_layer).
    i32 head_dim_at(i32 layer) const {
        if (cfg_.head_dim_layer.empty()) return cfg_.head_dim;
        if (layer < 0 || layer >= static_cast<i32>(cfg_.head_dim_layer.size()))
            return cfg_.head_dim;
        const i32 d = cfg_.head_dim_layer[static_cast<size_t>(layer)];
        return d > 0 ? d : cfg_.head_dim;
    }
    i64 q_dim_at(i32 layer) const {
        return static_cast<i64>(n_head_at(layer)) * head_dim_at(layer);
    }

    // ---- per-layer KV geometry --------------------------------------------
    // Wide as the widest layer (workspace and KV-slot sizing); the width one
    // layer actually reads and writes is kv_dim_at().
    i32 n_head_kv_at(i32 layer) const {
        if (cfg_.n_head_kv_layer.empty()) return cfg_.n_head_kv;
        if (layer < 0 || layer >= static_cast<i32>(cfg_.n_head_kv_layer.size()))
            return cfg_.n_head_kv;
        return cfg_.n_head_kv_layer[static_cast<size_t>(layer)];
    }
    // True when this layer runs multi-head latent attention.
    bool layer_is_mla(i32 layer) const {
        return layer >= 0 && layer < static_cast<i32>(layers_.size()) &&
               layers_[static_cast<size_t>(layer)].mla;
    }
    i64 kv_dim_at(i32 layer) const {
        // An MLA layer's row is the compressed latent followed by the
        // decoupled rope key, so it is NOT n_kv x head_dim: header says
        // n_kv = 16 and the row is 576 (V2-Lite), which is the whole point of
        // the compression. Reading the head geometry here would size the plane
        // 5x too wide and the kernel would read every row past its end.
        if (layer_is_mla(layer)) {
            // The shared-latent layout (V4) caches ONE row and keeps the rope in
            // its TAIL, so the row is the latent itself. The lite layout stores
            // the compressed latent and a decoupled rope key BESIDE it, which is
            // the +qk_rope. Sizing V4 by the lite rule leaves every plane 64
            // columns too wide and makes `pos_stride` -- the row the kernels
            // index by -- disagree with the slot the tier handed out.
            const i64 row = static_cast<i64>(cfg_.mla_kv_lora);
            if (cfg_.mla_layout == MlaLayout::Shared) return row;
            return row + cfg_.mla_qk_rope;
        }
        return static_cast<i64>(n_head_kv_at(layer)) * head_dim_at(layer);
    }
    // True when this layer reads and writes KV: the ones carrying q/k/v or an
    // MLA latent. A mamba/SSM block and a pure-MoE block append nothing, so
    // they hold no KV, are charged for none, and get no plane in the cache.
    // This is the same predicate the engine's attention branch uses.
    // The layer whose KV cache this one attends. Equal to `layer` unless the
    // file declares shared-KV layers (gemma4's `attention.shared_kv_layers`),
    // in which case the last N layers reuse an earlier one:
    //
    //     source = n_layer_kv_from_start - (is_swa(layer) ? 2 : 1)
    //
    // which is llama.cpp's rule, and the pairing is the file's own: a windowed
    // layer shares with the last windowed layer, a full one with the last full
    // layer, so the geometry (head count, width, window) always matches the
    // cache it reads. A shared layer must not append: it would allocate a plane
    // nobody reads, and reading its own plane would attend an empty cache.
    i32 kv_source_layer(i32 layer) const {
        const i32 start = cfg_.n_layer - cfg_.n_kv_shared;
        if (cfg_.n_kv_shared <= 0 || layer < start || start < 2) return layer;
        return start - (is_swa(layer) ? 2 : 1);
    }
    // True when this layer writes KV of its own.
    bool layer_writes_kv(i32 layer) const {
        return layer_has_kv(layer) && kv_source_layer(layer) == layer;
    }

    bool layer_has_kv(i32 layer) const {
        if (layer < 0 || layer >= static_cast<i32>(layers_.size())) return false;
        const LayerWeights &w = layers_[static_cast<size_t>(layer)];
        // A fused QKV layer keeps its query inside the fused tensor and so has
        // no standalone `wq` to test. Asking only about `wq` declares every
        // such layer KV-free, and the KV plane for the whole stack then sizes
        // to zero: the cache hands out a null base and the first append writes
        // through it. Measured, GPT-2 TinyStories: `KV=0 MiB`, SIGSEGV in
        // layer 0's first decode.
        return w.wq.present() || w.mla || w.wqkv_fused.present();
    }
    size_t kv_layer_count() const {
        size_t n = 0;
        for (size_t l = 0; l < layers_.size(); l++)
            if (layer_has_kv(static_cast<i32>(l))) n++;
        return n;
    }
    // True when this layer attends a bounded window instead of the whole past.
    bool is_swa(i32 layer) const {
        // The phase decides which end of the pattern is windowed. laguna is
        // period 4 with the FULL layer first (phase 0); gpt-oss is period 2
        // with the WINDOWED layer first (phase 1). Reading the pattern off the
        // wrong end runs a full-attention kernel on layers the checkpoint
        // trained with a 128-token window and vice versa: the shapes fit, so
        // nothing complains, and every layer sees the wrong context.
        // A pattern the FILE states outranks any period derived from the arch
        // table: gemma4 writes `attention.sliding_window_pattern` per layer, and
        // a period inferred here would be a guess about a fact the file already
        // gave. Out-of-range layers fall through to the period rule.
        if (layer >= 0 &&
            layer < static_cast<i32>(cfg_.swa_pattern.size()) &&
            cfg_.swa_window > 0)
            return cfg_.swa_pattern[static_cast<size_t>(layer)] != 0;
        return cfg_.swa_period > 1 && cfg_.swa_window > 0 &&
               ((layer + cfg_.swa_phase) % cfg_.swa_period) != 0;
    }
    // Rotary table this layer rotates with: the windowed layers of a hybrid
    // model carry their own (plain RoPE over the full head, where the full
    // layers run YaRN over a prefix). Falls back to the single table when the
    // model has no per-layer-kind rotary.
    const std::vector<f32> &inv_freq_at(i32 layer) const {
        return is_swa(layer) && !inv_freq_swa_.empty() ? inv_freq_swa_ : inv_freq_;
    }
    f32 rope_frac_at(i32 layer) const {
        return is_swa(layer) ? cfg_.rope_frac_swa : cfg_.rope_frac;
    }
    // The position divisor this layer rotates with. See
    // ModelConfig::rope_scale_layer: empty means the stack-wide value, which
    // is what every family without a window -- and every stack whose file
    // scales by YaRN, whose factor is already inside the table -- uses.
    f32 rope_scale_at(i32 layer) const {
        if (layer >= 0 && static_cast<size_t>(layer) < cfg_.rope_scale_layer.size())
            return cfg_.rope_scale_layer[static_cast<size_t>(layer)];
        return cfg_.rope_scale;
    }
    i64 n_ff() const { return cfg_.n_ff; }
    i64 n_embd() const { return cfg_.n_embd; }
    // ---- per-layer MoE geometry -----------------------------------------
    // The routed bank's own width is each layer's expert count (the loader
    // fills LayerWeights::experts from the tensors), and `n_expert` is the cap
    // the file declares. A packed merge really does carry different counts per
    // layer -- Nemotron Bolt 74B-A9B has 240/304/368/432/496 across its 40 MoE
    // blocks against one `expert_count` of 496 -- so anything that sizes or
    // routes by layer has to ask these, not the config.
    i32 n_expert_at(i32 layer) const {
        if (layer < 0 || layer >= static_cast<i32>(layers_.size()))
            return cfg_.n_expert;
        const i32 ne = static_cast<i32>(layers_[static_cast<size_t>(layer)].experts.n_expert);
        return ne > 0 ? ne : cfg_.n_expert;
    }
    // Top-k for one layer: the file's per-layer array when it has one, else the
    // model-level number, always clamped to that layer's own expert count. A k
    // larger than ne would select the same expert twice (the loop skips taken
    // ids, so the tail slots stay -1) and mix a plan of the wrong width.
    i32 n_expert_used_at(i32 layer) const {
        i32 k = cfg_.n_expert_used;
        if (layer >= 0 && layer < static_cast<i32>(cfg_.layer_n_expert_used.size())) {
            const i32 kv = cfg_.layer_n_expert_used[static_cast<size_t>(layer)];
            if (kv > 0) k = kv;
        }
        const i32 ne = n_expert_at(layer);
        if (ne > 0 && k > ne) k = ne;
        return k;
    }
    i64 n_ff_exp_at(i32 layer) const {
        if (layer < 0 || layer >= static_cast<i32>(layers_.size()))
            return cfg_.n_ff_exp;
        const i64 w = layers_[static_cast<size_t>(layer)].experts.n_ff_exp;
        return w > 0 ? w : cfg_.n_ff_exp;
    }
    // True when every MoE layer has the same expert count and top-k. The
    // device plan and the one-launch dispatch publish a UNIFORM table (one
    // [n_layer x n_expert] grid), so a heterogeneous stack keeps the host path
    // instead of reading past the end of its own table for the wide layers.
    bool moe_geometry_uniform() const {
        i32 first = -1;
        for (i32 l = 0; l < static_cast<i32>(layers_.size()); l++) {
            if (!layers_[static_cast<size_t>(l)].moe) continue;
            if (first < 0) {
                first = l;
                continue;
            }
            if (n_expert_at(l) != n_expert_at(first) ||
                n_expert_used_at(l) != n_expert_used_at(first) ||
                n_ff_exp_at(l) != n_ff_exp_at(first))
                return false;
        }
        return true;
    }
    // Widest FFN a layer may need (dense, routed expert, or shared expert).
    i64 n_ff_ws() const {
        i64 w = cfg_.n_ff;
        if (cfg_.n_ff_exp > w) w = cfg_.n_ff_exp;
        if (cfg_.n_ff_shexp > w) w = cfg_.n_ff_shexp;
        return w;
    }

    const ModelConfig &cfg() const { return cfg_; }
    const std::vector<LayerWeights> &layers() const { return layers_; }

    // ---- gated delta net geometry ---------------------------------------
    bool is_recurrent() const { return cfg_.recurrent; }
    // Does this stack mix with LFM2.5's short convolution? Asked once at init
    // to size the workspace, because the delta-net buffers are shaped by
    // metadata this family does not carry.
    bool has_lfm_conv() const {
        for (const LayerWeights &L : layers_)
            if (L.lfm_conv) return true;
        return false;
    }
    // Can a speculative round rewind the sequence state to an accepted prefix?
    //
    // A delta-rule state cannot: it is a running product of decay and rank-1
    // updates with no inverse, so a rejected suffix is unrecoverable and the
    // sequence rewrite is what makes speculation safe there (src/engine.cpp,
    // `all_accepted`). A shortconv ring can, and not because the ring holds
    // history: it can because the engine keeps the ring DEEPER than the window
    // (Engine::sc_hist_), so a row the round is about to rewrite is never the
    // row the next convolution reads. Rewinding is then pointer arithmetic.
    bool speculative_rewind_ok() const {
        if (has_lfm_conv()) return true;
        return !is_recurrent();
    }
    i64 conv_dim() const { return cfg_.ssm_conv_dim; }
    i64 value_dim() const { return cfg_.ssm_value_dim; }
    i64 key_dim() const { return cfg_.ssm_key_dim; }
    // Number of delta-net layers, i.e. how many recurrent states the engine
    // has to carry.
    i32 recurrent_layers() const {
        i32 n = 0;
        for (const LayerWeights &L : layers_)
            if (L.gdn) n++;
        return n;
    }
    // Index of a delta-net layer within the recurrent-state arrays (layers are
    // numbered for the KV cache but the states are only allocated for the
    // recurrent ones).
    i32 recurrent_index(i32 layer) const {
        i32 n = 0;
        for (i32 i = 0; i < layer && i < static_cast<i32>(layers_.size()); i++)
            if (layers_[static_cast<size_t>(i)].gdn) n++;
        return n;
    }
    const QuantTensor &tok_embd() const { return tok_embd_; }
    const QuantTensor &out_head() const { return out_head_; }
    // The widest CONTRACTION among the weights that carry an NVFP4 global
    // scale; 0 when the file has none. The engine sizes the staging buffer it
    // scales such a weight's input into, and allocates nothing when this is 0.
    i64 max_scaled_in() const { return max_scaled_in_; }
    // BLOOM normalizes the token embedding before block 0; both null on every
    // other family (ArchSpec::embd_norm).
    const f32 *embd_norm() const { return embd_norm_; }
    const f32 *embd_norm_bias() const { return embd_norm_bias_; }
    const f32 *out_norm() const { return out_norm_; }
    // `output.bias`, one f32 per vocabulary row. Not every family has one --
    // Phi-MoE does, and adds it to the logits after the head's projection, so a
    // file that carries it with the tensor ignored gets every logit shifted by
    // a per-token amount. Null when absent, like every other optional weight.
    const f32 *out_bias() const { return out_bias_; }
    const f32 *out_norm_bias() const { return out_norm_bias_; }
    const QuantTensor &pos_embd() const { return pos_embd_; }
    // qwen4exp has no output_norm: the final hyper-connection mixer carries it,
    // with its own [n_embd, hc] gamma. Null on every other family.
    const f32 *hc_head_norm() const { return hc_head_norm_; }
    // qwen4exp's PLE table is a pointer into the mapping, not an upload: the
    // accessors exist so a test can assert the loader found it and sized one row.
    const QuantTensor &hc_head_fn() const { return hc_head_fn_; }
    const f32 *hc_head_base() const { return hc_head_base_; }
    const f32 *hc_head_scale() const { return hc_head_scale_; }
    const u8 *ple_table() const { return ple_table_; }
    // Row count of the shared per-layer table. The gather needs it to refuse an
    // id past the end (krk/engram.cpp's gather zeroes such a row), and it is
    // the file's own ne[1], not derived from the head ranges.
    u64 ple_table_rows() const { return ple_table_rows_; }
    // The file's own type for one row of the table: the gather dequantizes one
    // row at a time out of the mapping, so the stride and the type travel
    // together.
    u64 ple_row_bytes() const { return ple_row_bytes_; }
    DType ple_table_type() const { return ple_table_type_; }
    const QuantTensor &hc_head_down() const { return hc_head_down_; }
    const QuantTensor &hc_head_up() const { return hc_head_up_; }
    // ---- gemma4 per-layer input embeddings -------------------------------
    // The token table is [n_layer*n_embd_per_layer, n_vocab] and stays on the
    // device (it is a full embedding matrix, gathered a row per token); the
    // model projection and its norm are read every pass.
    QuantTensor ple_tok_embd;      // [n_layer*n_embd_per_layer, n_vocab]
    QuantTensor ple_model_proj;    // [n_embd, n_layer*n_embd_per_layer]
    f32 *ple_proj_norm = nullptr;  // f32 [n_embd_per_layer]
    // The same norm weight repeated once per layer. The file carries ONE
    // [n_embd_per_layer] vector and the graph applies it to every layer's slice
    // of the projection; the grouped norm kernel indexes its weight by group
    // (`w + g * width`), so the repetition is the shape that kernel wants and
    // is built once here instead of a second kernel path that would have to
    // special-case a shared weight.
    std::vector<f32> ple_proj_norm_rep;

    // NeoX inverse frequencies, length head_dim/2, host-resident (tiny).
    const std::vector<f32> &inv_freq() const { return inv_freq_; }
    i64 weight_bytes() const { return weight_bytes_; }
    const Gguf &gguf() const { return gguf_; }

    // ---- chat template ---------------------------------------------------
    // The Jinja program the file carries (`tokenizer.chat_template`), or empty.
    // It is the file's own statement of the byte layout its turns were trained
    // on, which is why it is read rather than guessed from the arch -- see
    // include/krk/chat_template.hpp for why the layout is correctness and not
    // cosmetics.
    const std::string &chat_template() const { return chat_template_; }
    // What to actually render with: the file's template when it has one this
    // engine can run, otherwise the builtin for the family. Never empty.
    const std::string &chat_template_or_builtin() const {
        return chat_template_.empty() ? builtin_template_ : chat_template_;
    }
    // "file", or the builtin's name -- for a log line that has to say which
    // template produced the prompt, because a silent fallback is exactly the
    // failure this whole module exists to prevent.
    const char *chat_template_source() const {
        return chat_template_.empty() ? builtin_name_ : "file";
    }
    // A caller's override (--chat-template): set BEFORE load(), since the
    // resolution happens there.
    void set_chat_template_override(std::string t) {
        chat_template_override_ = std::move(t);
    }
    // The prism.hadamard plan for this file (inactive for everything else).
    const HadamardPlan &had() const { return had_; }
    // The +-1 sign vector for a folded input `width` wide, in the backend's
    // address space (device memory on the GPU); null when the file declares
    // sign_mode=identity, and never null for a width the plan folded -- the
    // loader refuses that case.
    const f32 *had_signs(i64 width) const {
        for (const auto &sg : had_signs_)
            if (static_cast<i64>(sg.first) == width) return sg.second;
        return nullptr;
    }

    // Lazily-loaded routed experts. Override the residency budget after load to
    // trade reload traffic against memory.
    ExpertCache &experts() { return experts_; }
    const ExpertCache &experts() const { return experts_; }
    // host_bytes sizes the cache's second tier in page-locked host memory; 0
    // disables it and evicted experts go back to the GGUF mapping.
    void set_expert_budget(size_t bytes, size_t host_bytes = 0, size_t slot_budget = 0) {
        experts_.configure(be_, bytes, host_bytes, slot_budget);
    }
    // How many (layer, expert) slots the routed set has (0 when dense). The
    // counterpart of total_expert_bytes() for the residency report: the device
    // budget is sized to cover the whole set, and this is how many slots that
    // is, so the cache can report the set rather than an estimate of it.
    size_t expert_slot_count() const {
        size_t n = 0;
        for (const LayerWeights &L : layers_) {
            if (!L.moe) continue;
            const i64 ne = L.experts.n_expert > 0 ? L.experts.n_expert : 1;
            n += static_cast<size_t>(ne);
        }
        return n;
    }

    // Largest single-expert footprint across the MoE layers (0 when dense).
    // What the load phase copied to the device and how long it took. The
    // per-size breakdown stays behind KRK_PHASE; this is the total the run
    // report needs on every run.
    size_t upload_bytes() const { return upload_bytes_; }
    f64 upload_ms() const { return upload_ms_; }

    size_t max_expert_bytes() const {
        size_t m = 0;
        for (const LayerWeights &L : layers_)
            if (L.moe && L.experts.expert_bytes() > m) m = L.experts.expert_bytes();
        return m;
    }
    // Bytes needed to keep the *whole routed set* resident (0 when dense):
    // every expert of every MoE layer, not one expert per layer.
    //
    // It summed only expert_bytes() per layer, which is the footprint of a
    // single expert and understated the answer by exactly n_expert. The
    // budget is then compared against this in configure_expert_cache, so a
    // 4-expert layer got a budget for 1 expert's worth of the whole model:
    // 152.6 MiB where ~610 MiB was needed, ~28 of 112 expert rows resident
    // against 56 accessed per token, and a 0.0% hit rate — every expert
    // re-fetched from host memory on every step. Qwen3-MoE-4x0.6B-2.4B ran at
    // 16.0 tok/s; with the correct total the budget covers the whole set and
    // it runs at 77.5 tok/s (4.8x), with 4096 MiB buying nothing further.
    size_t total_expert_bytes() const {
        size_t t = 0;
        for (const LayerWeights &L : layers_) {
            if (!L.moe) continue;
            const i64 ne = L.experts.n_expert > 0 ? L.experts.n_expert : 1;
            t += L.experts.expert_bytes() * static_cast<size_t>(ne);
        }
        return t;
    }

    // Bytes of file this model has mapped, summed over shards (0 when the
    // model was not loaded from a mapping). The mapping is the third
    // residency tier: it is the disk source, and every page it touches is
    // page cache -- RAM that a WARM copy cannot spend a second time.
    size_t mapped_bytes() const {
        size_t n = 0;
        for (const GgufShard &sh : gguf_.shards())
            if (sh.file) n += sh.file->size();
        return n;
    }

private:
    Backend *be_ = nullptr;
    // Dense weight streaming: layers whose weights live in the GGUF mapping and
    // are copied through one reusable device ring before the layer runs. All
    // streamed layers share one ring, so a run's VRAM cost is the resident
    // prefix plus one layer, not the whole model.
    size_t weight_budget_ = 0;   // bytes, 0 = automatic
    std::vector<u8> layer_streamed_;
    std::vector<std::vector<StreamEntry>> stream_;
    // One contiguous device slab for every RESIDENT per-layer weight, carved at
    // load time. Four hundred hipMalloc'd buffers of 1-15 MiB and one slab hold
    // the same bytes, but a kernel walks the slab's pages in order instead of
    // crossing a heap. Built to test whether allocation layout explains the
    // in-engine read rate; null means the per-tensor path (which is correct, just
    // more scattered). Released once, and every interior pointer into it is
    // skipped by the per-tensor release.
    u8 *slab_ = nullptr;
    size_t slab_bytes_ = 0;
    size_t slab_off_ = 0;
    void *ring_ = nullptr;
    size_t ring_bytes_ = 0;
    void *stage_host_ = nullptr; // pinned staging buffer, ring-sized
    // Layers that were streamed once and are now resident in a permanent
    // allocation of their own (base, bytes). Freed by unload, and skipped by
    // the per-tensor release because every tensor in them is an interior
    // pointer into that base.
    std::vector<std::pair<void *, size_t>> promoted_;
    // Dense streaming pipeline. Two pinned buffers: the one this layer's copy
    // reads from, and the one the next layer is read into. alternating them is
    // what lets the read (8.9 GB/s) and the DMA (a side stream) run at once.
    void *prefetch_host_ = nullptr;
    void *read_buf_[2] = {nullptr, nullptr};
    i32 prefetched_layer_ = -1;
    int prefetched_slot_ = 0;
    bool prefetch_enabled_ = true;
    f64 stage_prefetch_ms_ = 0;
    // ---- one-hop-ahead double buffering ---------------------------------
    // The ring above is ONE layer deep, so every step is [copy layer l] then
    // [compute layer l]: the measured 27B step is 79 ms of staging DMA and
    // 84 ms of projections that never overlap. Two slots and a prefetch issued
    // one layer early put the copy inside the compute that precedes it.
    //
    // A layer's slot is chosen by its parity and its tensor addresses are baked
    // at load time, so nothing is re-pointed per token while the GPU may be
    // reading. The slot a copy writes was read by the layer two before it: the
    // copy stream is ordered behind a marker recorded at the point those kernels
    // were enqueued (Backend::prefetch_after), and the default stream is ordered
    // behind the copy at the point the layer runs (wait_marker_stream).
    //
    // Requires the pinned host mirror: with it, the copy's source never changes.
    // Without it, a per-token read would have to stay valid until the copy
    // landed, and the existing read pipeline is already what that path gets.
    int ring_slots_ = 1;
    size_t slot_bytes_ = 0;      // stride between slots; 0 when there is one
    bool ring_ahead_ = false;    // the pipeline is live (mirror present)
    i32 inflight_layer_ = -1;    // layer whose copy is in flight
    void *inflight_ticket_ = nullptr;
    u64 ahead_copies_ = 0, ahead_cold_ = 0;
    f64 ahead_wait_ms_ = 0;
    const u8 *slot_base(i32 l) const;
    i32 first_streamed() const;
    // Issue the copy that fills `next_streamed(l)`'s slot, ordered behind every
    // kernel enqueued so far. Called from the END of stage_layer(l), which is
    // the one point where both facts hold.
    void issue_ahead(i32 l);
    // The synchronous single-slot pass: drain, read, DMA. Returns false when the
    // layer has nothing to stage.
    bool stage_layer_now(i32 l, const std::vector<StreamEntry> &es);
    // A pinned host mirror of every streamed layer, one allocation, each layer
    // at its own offset. Streamed bytes do not change between tokens, so the
    // per-token file read is pure repetition: the mirror is filled once at load
    // and every staging pass afterwards is a DMA straight out of pinned memory.
    // Null when the set is above the cap or the allocation failed, in which case
    // stage_layer falls back to the read path.
    void *stream_host_ = nullptr;
    std::vector<size_t> stream_host_off_; // per-layer offset into stream_host_
    bool stream_host_on_ = false;
    size_t stream_host_size_ = 0;
    f64 stream_host_fill_ms_ = 0;
    void build_stream_host_mirror(size_t streamed_total);
    // Read layer `l`'s tensors into `host` at the ring's own offsets, and copy
    // them from there into the ring. Split so the read of the next layer can
    // happen between the two.
    void read_layer_to(void *host, const std::vector<StreamEntry> &es,
                       size_t *bytes);
    void upload_layer_from(const void *host, const std::vector<StreamEntry> &es);
    i32 next_streamed(i32 after) const;
    std::vector<size_t> layer_stream_bytes_; // aligned ring bytes per streamed layer
    size_t promoted_bytes_ = 0;
    i32 promoted_layers_ = 0;
    f64 promote_ms_ = 0;
    // Promotion is on by default; KRK_WEIGHT_PROMOTE=0 restores the fixed
    // resident prefix for an A/B of the policy itself.
    bool promote_enabled_ = true;
    // Promotion into a permanent allocation, in stage_layer's own order: the
    // read + copy a staging pass would do anyway, kept. False when this layer
    // is not streamed, when the switch is off, or when it does not fit.
    bool promote_layer(i32 l);
    size_t stream_bytes_total_ = 0;
    size_t streamed_layers_ = 0;
    u64 stage_calls_ = 0;
    u64 stage_bytes_ = 0;
    f64 stage_ms_ = 0;
    f64 stage_sync_ms_ = 0;
    f64 stage_read_ms_ = 0;
    f64 stage_xfer_ms_ = 0;
    bool stream_cur_ = false;    // inside a streamed layer at load time
    size_t stream_off_ = 0;      // ring write offset for the current layer
    i32 stream_layer_index_ = -1;
    Gguf gguf_;
    std::string chat_template_;
    std::string builtin_template_ = builtin_chat_template("", "");
    const char *builtin_name_ = kBuiltinDefault;
    std::string chat_template_override_;
    ModelConfig cfg_;
    MtpBlock mtp_;
    bool mtp_load_ = false;
    std::string model_path_;            // for messages that name the file
    std::string mtp_path_;              // sidecar nextn file, empty for in-model
    std::unique_ptr<Gguf> mtp_gguf_;    // keeps the sidecar's mapping alive
    // The body both paths share: read block `l` out of `g`. `origin` names the
    // file in a refusal, so a sidecar that does not fit does not read as the
    // trunk being broken.
    bool load_mtp_block_from(const Gguf &g, i32 l, const std::string &origin,
                             std::string *err);
    // Upload the nextn block, if the file has one. Called at the end of load()
    // and only when KRK_MTP_LOAD is set: it is ~855 MB of experts on a 35B-A3B
    // and nothing in the main stack reads it.
    bool load_mtp_block(std::string *err);
    std::vector<LayerWeights> layers_;
    QuantTensor tok_embd_, out_head_;
    i64 max_scaled_in_ = 0;
    f32 *embd_norm_ = nullptr;
    f32 *embd_norm_bias_ = nullptr;
    f32 *out_norm_ = nullptr;
    f32 *out_bias_ = nullptr;
    // ---- GPT-2 -----------------------------------------------------------
    // Learned absolute position embeddings, [n_ctx_train, n_embd], added to
    // the token embedding before the first block. Held in its original dtype
    // like every other weight, so the read is a dequantizing row copy, not a
    // widened resident table.
    QuantTensor pos_embd_;
    // LayerNorm bias on the final norm, beside out_norm_.
    f32 *out_norm_bias_ = nullptr;
    // qwen4exp's final mixer *is* the output norm (there is no output_norm):
    // the same lane-mix read the blocks use, applied once at the top of the
    // stack. `ple_table_` is the flat n-gram embedding table, kept as a POINTER
    // INTO THE MAPPING because it is 320,001,536 rows of 160 -- tens of GB that
    // must never be uploaded. `ple_row_bytes` is one row's stride.
    f32 *hc_head_norm_ = nullptr;
    QuantTensor hc_head_down_, hc_head_up_;
    // V4.1's head mixer: hc_head_fn {hc*n_embd -> hc}, base [hc], scale [1].
    // Absent on a single-pass file, where the last FFN's pre-mix collapses the
    // stack instead -- that absence is what hc_no_head records.
    QuantTensor hc_head_fn_;
    f32 *hc_head_base_ = nullptr;
    f32 *hc_head_scale_ = nullptr;
    const u8 *ple_table_ = nullptr;
    u64 ple_row_bytes_ = 0;
    u64 ple_table_rows_ = 0;
    DType ple_table_type_ = DType::F32;
    std::vector<f32> inv_freq_;
    std::vector<f32> inv_freq_swa_;
    // prism.hadamard. The sign vectors are uploaded once at load and released
    // in unload(); the plan itself is read through had()/had_signs().
    HadamardPlan had_;
    std::vector<std::pair<i32, f32 *>> had_signs_; // (width, device f32[width])
    i64 weight_bytes_ = 0;
    ExpertCache experts_;
    size_t upload_bytes_ = 0;
    f64 upload_ms_ = 0;
};

} // namespace krk

#endif // KRK_MODEL_HPP
