// ============================================================================
//  engine.hpp — the decode loop, workspace management and streaming.
// ============================================================================
#ifndef KRK_ENGINE_HPP
#define KRK_ENGINE_HPP

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "krk/backend.hpp"
#include "krk/h2d_probe.hpp"
#include "krk/hc.hpp"
#include "krk/ple.hpp"
#include "krk/expert_cpu.hpp"
#include "krk/kv_tier.hpp"
#include "krk/dsv4_route.hpp"
#include "krk/draft_head.hpp"
#include "krk/model.hpp"
#include "krk/sampler.hpp"
#include "krk/tokenizer.hpp"
#include "krk/moe_predictor.hpp"

namespace krk {

class DflashDraft;

struct EngineConfig {
    std::string model_path;
    // A chat template to use INSTEAD of the file's (`--chat-template NAME`,
    // where NAME is a builtin, or a path to a .jinja file). Empty keeps the
    // file's own, which is what a file that carries one should always get.
    std::string chat_template;
    i32 n_ctx = 4096;        // KV capacity (tokens)
    // Prompt tokens per batched forward. 512 because the routed expert set is
    // re-read once per chunk: 35B-A3B at a 4096-token prompt measured 397.1
    // tok/s at 256 against 462.6 at 512, and the exact-rung hashes agree at
    // chunk 1, 8 and 512 (test_prefill_chunk_bitexact's 512 arm).
    i32 prefill_chunk = 512;
    // Diagnostics: fold every stage of every prefill row into one hash
    // (see Engine::prefill_trace_hash). --prefill-hash sets it; it is the
    // only observable that sees a one-ulp difference in a row that is
    // not the prompt's last, which is what chunk invariance is about.
    bool trace_hash = false;
    i32 threads = 0;         // CPU backend only
    u64 seed = 0;
    bool warmup = false;     // run a 1-token pass at init to page in weights

    // Residency budget for lazily-loaded MoE expert weights, in MiB. 0 means
    // auto: a share of free VRAM on the GPU, a fixed share on the CPU. Below the
    // cost of one expert the model still runs — it just reloads every step.
    // KV residency (see kv_tier.hpp). HOT is VRAM, WARM is host RAM, COLD is
    // a spill directory. 0 = auto for HOT (whatever free VRAM is left after
    // the weights and workspaces) and for WARM (half of free host RAM); a
    // negative WARM, or an empty COLD dir, disables that tier.
    //
    // When the whole KV fits the budget this is inert: the cache is one flat
    // allocation and every call site is unchanged. Tiering only engages once
    // the KV genuinely does not fit, which is what lets a long context run at
    // all instead of being refused.
    i32 kv_hot_mb = 0;
    i32 kv_warm_mb = 0;
    std::string kv_cold_dir;
    // K/V cache element format. 0 = f16/f32 (the default, and the only arm any
    // model ran before this existed: its arithmetic is unchanged). 1 = int8
    // with one f32 scale per 32 values, see KvQuant in backend.hpp. The CPU
    // reference implements both; a backend that cannot read the layout refuses
    // the run by name instead of running f16 kernels over int8 bytes.
    i32 kv_quant = 0;

    // Dense weight budget in MiB. 0 (the default) is automatic: weights stay
    // resident while they fit, and whole layers are streamed from the file once
    // they do not, leaving VRAM for KV and workspaces. A positive value caps
    // the resident set explicitly (--weight-vram-mb); a negative value disables
    // streaming and keeps the old upload-everything behaviour. KRK_WEIGHT_VRAM_MB
    // is the environment spelling of the same number.
    i32 weight_vram_mb = 0;

    i32 expert_cache_mb = 0;
    // Hard cap on resident (layer, expert) slots, regardless of budget. 0 = auto.
    i32 expert_cache_slots = 0;
    // WARM tier budget in MiB: pageable host RAM that holds the expert set
    // beside the device's hot subset, and the tier every cold read lands in
    // before it is promoted (see expert_cache.cpp). Sized independently of the
    // device budget, because the two answer different questions — VRAM holds
    // what is about to be used, RAM holds what was just used.
    // -1 (the default) sizes it from the machine: the expert set, capped by half
    // of what is free after a reserve, so 24/32/48/96 GiB boxes each get a
    // proportional tier. 0 disables it (HOT only, every miss re-reads the
    // file), and a positive value is that many MiB — the explicit restriction.
    i32 expert_warm_mb = -1;
    // Pinned prefix of WARM, in MiB. <0 keeps KRK_HOST_PIN_MB's behaviour (off
    // unless that variable is set); 0 pins nothing; N pins the first N MiB of the
    // tier in admission -- i.e. rank -- order. Those are the experts promoted
    // most often, and a promotion out of pinned memory is one DMA with no host
    // memcpy in front of it: measured 22.7 GB/s against 7.6.
    i32 expert_pin_mb = -1;
    // How many tokens the caller expects to generate, when it knows. The seed
    // fraction at init depends on it: a short reply is worth a half-seeded
    // pool, a long one is not (see warm_experts). 0 = unknown, read as long.
    i32 expected_reply_tokens = 0;
    // Fill WARM ahead of the run rather than on demand. Off by default: an eager
    // sweep costs the whole corpus in wall time and RAM before the first token,
    // and read-through already admits every expert the run touches.
    bool expert_warm_prefetch = false;
    // Expert warmup: a bounded startup phase that stages the RANKED expert set
    // into WARM and then promotes the top of that order into VRAM, so the first
    // tokens are not paying compulsory file reads and a tier smaller than the
    // routed set holds what routing wants rather than what happened to be
    // touched first. -1 (the default) turns it on when it can help -- a WARM
    // tier exists and the routed set is larger than the device budget, which is
    // exactly the case where the ORDER of the two tiers decides the miss rate.
    // 1 forces it, 0 disables it; KRK_EXPERT_WARMUP=0/1 overrides both, so the
    // A/B is one environment variable away.
    i32 expert_warmup = -1;
    // Residency preference in the router: a near-tie between two experts is
    // given to the one already in VRAM (then to the one in WARM) instead of to
    // the marginally higher logit. The value is the margin, as a fraction of the
    // top probability -- 0.05 means "any candidate within 5% of the best score is
    // near-equal". 0 (the default) is off, and <0 reads KRK_ROUTER_TIE_MARGIN.
    // It is a real trade: the route changes, so the text can change too. What it
    // buys is fewer promotions, i.e. fewer bytes per token, which on a tier
    // smaller than the routed set is the whole cost of a step.
    f32 router_tie_margin = -1.0f;
    // Wall-time ceiling for that phase, in ms. The default is inside the
    // 15-30 s a cold start can afford; a phase that has not filled the tiers by
    // then stops where it is, having spent the time on the hottest experts
    // first. 0 removes the ceiling (fill the tiers however long it takes).
    i32 expert_warmup_ms = 20000;
    // The RAM class to plan the host tier for, in GiB: 16, 24, 32, 48, 64 or
    // 96, i.e. the machine the run should behave as if it were on. 0 (the
    // default) is the installed RAM. This is the spec's tiering table applied
    // to a class instead of to the box, so a deployment can be measured and
    // tuned on a 96 GiB workstation for a 32 GiB target without pretending the
    // extra RAM away by hand.
    i32 ram_tier_gb = 0;
    // Total device memory this run is allowed to plan for, in MiB. 0 (the
    // default) is the stepped policy in Engine::configure_expert_cache: 6 GiB
    // to begin with, +2 GiB at a time while the card has the headroom and the
    // model actually needs more, never above 14 GiB. A positive value is that
    // many MiB and skips the policy entirely.
    i32 vram_cap_mb = 0;
    // The class form of the same restriction, in GiB: behave as if the card
    // were this big. The twin of ram_tier_gb, and what a plan made for one
    // machine carries to another -- a 12 GiB card and an 8 GiB card want a
    // number that is a property of the CARD, not a MiB figure that has to be
    // recomputed per box. Clamped to the memory actually installed, so the same
    // command line is safe on the big machine. Ignored when vram_cap_mb is set,
    // which is the exact figure.
    i32 vram_tier_gb = 0;
    // Host memory this run is allowed to plan for, in MiB. 0 (the default) is
    // a quarter of installed RAM; a positive value is that many MiB and bounds
    // every host tier inside the process. The mapping and the dense trunk sit
    // outside it, which is what the reserve is for.
    i32 host_ram_mb = 0;

    // ---- hybrid CPU + GPU expert compute --------------------------------
    //
    // In an MoE the expert weights, not the activations, are what has to move:
    // one decode token routes k experts per layer, one activation row is 8 KiB,
    // and one expert's three slices are ~1.95 MiB on a model like laguna. So an
    // expert that is NOT resident in VRAM is far cheaper to compute where its
    // bytes already are -- the mapping -- than to read in and promote. When this
    // is on, an expert that would have been a miss is computed on the host by
    // CpuExpertPool while the device runs the experts that ARE resident, and the
    // two partial sums are merged.
    //
    // It is a decode optimization and it is deliberately bounded to a small row
    // count: the merge is two [n, n_embd] transfers per layer, the host cost
    // scales with the rows an expert carries, and at prefill widths the second
    // of those dominates whatever the split saved. Combined with a small
    // --expert-cache-mb it is also the lever that frees VRAM and host RAM: the
    // experts the host computes need no resident copy in either tier.
    bool hybrid_experts = false;
    // Host worker threads for that split. 0 = min(hardware, 8).
    i32 hybrid_threads = 0;
    // Widest row count the host arm runs at. 0 disables the row bound.
    i32 hybrid_max_rows = 8;
    // Fraction (0..1) of the device tier's OVERFLOW -- the experts it could not
    // hold -- that the host arm takes. 1.0 (the default) takes all of it, which
    // is the policy: an expert that would have been a read plus a promotion is
    // computed where its bytes already are. Anything less sends that share of
    // the overflow back to the device to be promoted as it always was, which is
    // the A/B knob that sweeps the CPU/GPU balance at a fixed cache size.
    //
    // It cannot reach a RESIDENT expert. That was the old semantics ("share of
    // the resident set forced to the host") and it is the one thing the policy
    // forbids: an expert whose weights are already in VRAM has nothing to save
    // by moving its one decode row to the host, and pulling it off the device
    // would make the [stats] hybrid policy line untrue. Residency decides this,
    // not a fraction.
    f32 hybrid_frac = 1.0f;

    // ---- nextn (MTP) probe -------------------------------------------------
    // Runs the file's own nextn block for one step after prefill and reports
    // the token it would draft against the target's own argmax at that
    // position. It is a measurement, not a decoder: nothing is emitted from it,
    // and it is off unless --mtp-probe / KRK_MTP_PROBE=1 asked for it. The same
    // switch makes Model::load upload the block, because a probe that runs
    // against an absent block would report a measurement of nothing.
    bool mtp_probe = false;
    // Decode with the file's own nextn block as the drafter: proposals come
    // from the block's chain (the target's hidden row, then the block's own
    // output), verification from the target's one batched forward, and the
    // block's KV is rolled back and re-fed from the accepted tokens' captured
    // target rows. Greedy only, like every speculative path here. --mtp-spec /
    // KRK_MTP_SPEC=1; also loads the block.
    bool mtp_spec = false;
    // A nextn block in its OWN file, for a trunk that ships without one
    // (`-noMTP` quants whose MTP was published separately). Same block shape,
    // same names, read from the sidecar and driven by --mtp-probe/--mtp-spec.
    std::string mtp_path;

    // ---- external draft head (EAGLE-3) ------------------------------------
    // Path to a converted head (tools/eagle_to_gguf.py). Empty = none. The
    // head is loaded after the target, checked against its geometry and its
    // vocabulary, and then drives the same verify loop --draft-tokens already
    // controls. Needs no device memory of its own.
    std::string draft_head;
    // Positions a round primes the head over, most recent last: 0 = the whole
    // prompt. A head's fusion projection is O(hidden x target_hidden) per
    // position, so the window is what keeps a long prompt's prime from being
    // seconds of host arithmetic.
    i32 draft_head_prime = 256;
};

// Hybrid CPU+GPU expert-compute telemetry. The split is only interesting if you
// can see how it landed: how many experts each arm took, how many rows the host
// carried, and the wall time the host arm cost. Read by the end-of-run [stats]
// line and by --profile (which also gets it as the "moe.cpu_experts" host span).
struct HybridStats {
    i32 threads = 0;
    u64 cpu_experts = 0;
    u64 gpu_experts = 0;
    u64 cpu_rows = 0;
    f64 cpu_ms = 0.0;
    // Experts classified for the host while they were still VRAM-resident. The
    // policy says this cannot happen -- `cpu` is only ever set from the
    // overflow, which requires !in_vram -- so a non-zero value here is a defect
    // in the classification rather than a tuning outcome, and the report prints
    // it beside the policy it invalidates. Counted instead of asserted because
    // it is one read-only residency probe on an expert that was just probed,
    // and because the suite has no group that opens the device to reach the
    // real path (AGENTS.md): a counter is what makes "hot experts run on the
    // GPU" checkable in a run instead of argued from a comment.
    u64 policy_violations = 0;
};

struct StreamSink {
    void (*fn)(void *user, const char *text, i32 token, bool done) = nullptr;
    void *user = nullptr;
    void emit(const char *text, i32 token, bool done) const {
        if (fn) fn(user, text, token, done);
    }
};

struct GenerateParams {
    std::string prompt;
    SampleParams sampler;
    i32 max_tokens = 256;
    std::vector<std::string> stop;
    StreamSink sink;
    bool echo_prompt = false;
    // Debug: when > 0, write the top-k candidates and their log-probs for
    // every sampled token to stderr, at full precision. Two runs that pick
    // different tokens are then distinguishable by eye: identical log-probs
    // up to a step and different ones after it means drift, while the very
    // first step already differing means something raced.
    i32 debug_topk = 0;

    // Prompt tokens already resident in the KV cache, from position 0. A
    // server that keeps the cache across requests sets this after rolling the
    // cache back to the longest common prefix it found, and the prefill then
    // starts at `reuse_tokens` instead of at 0 -- the rows below it already
    // hold exactly these tokens at exactly these positions, so the result is
    // the one a full prefill would have produced.
    //
    // 0 (the default) is the historical behaviour: prefill the whole prompt.
    // It is ignored, with a warning, for a recurrent model (whose state cannot
    // be reconstructed by a rollback) and for the speculative paths (which
    // prefill the prompt themselves), so a caller cannot corrupt a run by
    // setting it when it does not apply.
    i32 reuse_tokens = 0;

    // Force the argmax on the host from the downloaded logits row instead of
    // on the device via logits_topk. Costs a 993 kB download per token at this
    // vocab, and buys determinism: the device argmax path is the one carrying
    // the laguna race (see docs/PERF-ANALYSIS.md section 9), so this is the
    // correct path until that is fixed. --debug-topk reaches the same code but
    // also prints per token, which serialises the run and confounds any test of
    // it -- hence a flag that changes one variable and not the other.
    bool sample_host = false;
};

enum Finish : int {
    FinishLength = 0,
    FinishStop = 1,
    FinishEos = 2,
    FinishContext = 3,
    FinishError = 4,
};

enum LogitMode : int {
    LogitsNone = 0, // no vocab projection (mid-prefill chunks)
    LogitsLast = 1, // project the final row only (prefill end, plain decode)
    LogitsAll = 2,  // project every row (speculative verification)
};

struct GenerateResult {
    std::string text;
    std::vector<i32> tokens;
    int finish = FinishLength;
    i32 prompt_tokens = 0;
    // How many of prompt_tokens the prefill was allowed to skip because the KV
    // cache already held them (see GenerateParams::reuse_tokens). 0 for a run
    // that prefilled the whole prompt.
    i32 reused_tokens = 0;
    i32 generated = 0;
    f64 prefill_ms = 0;
    f64 decode_ms = 0;
    // How many decode steps decode_ms actually covers. It is NOT `generated`:
    // the first generated token comes out of the prefill pass's last-position
    // logits, so the timer starts after prefill and times one step fewer than
    // the number of tokens emitted. Dividing by `generated` understated the
    // per-token cost and printed 2304 tok/s for --max-tokens 1, which is not a
    // speed any machine can produce: one step still reads every weight.
    // Counted rather than inferred, so it stays right if the loop changes.
    i64 decode_steps = 0;
    f64 load_ms = 0;
    // Non-empty when inference stopped before the requested token count. Kept
    // on the result so API callers can distinguish resource refusal from a
    // normal context/stop-token finish.
    std::string error;
};

// ---- the expert index file -----------------------------------------------
// A parsed `<model>.krakenexperts.json`, layer-major. `valid` is false when the
// text does not describe the model it was asked about: a ranking measured on
// other weights is worse than no ranking, so every reader refuses it.
struct ExpertIndexRows {
    std::vector<double> mass; // [n_layer * n_expert] summed P(expert)
    std::vector<u64> hits;    // [n_layer * n_expert] times selected
    i64 tokens = 0;           // token positions the file was measured over
    bool valid = false;
};

// Reads the format write_expert_index produces. `src_size` is the model file's
// size when known; 0 skips that check, which only a fixture wants.
ExpertIndexRows expert_index_parse(const std::string &text, i32 n_layer,
                                   i32 n_expert, const std::string &arch,
                                   unsigned long long src_size);

// p' = retain * p_old + (1 - retain) * p_run, per layer, each input normalised
// to a distribution first so a long run cannot outvote a short one by its token
// count. Hits add, and the stored mass keeps its meaning ("summed probability
// over N positions") by being scaled to the larger of the two counts. `old` is
// overwritten with the blend; a layer the run never routed through keeps its
// stored row.
void expert_index_blend(ExpertIndexRows *old, i32 n_expert,
                        const std::vector<double> &run_mass,
                        const std::vector<u64> &run_hits, i64 run_tokens,
                        double retain);

class Engine {
public:
    Engine() = default;
    ~Engine();

    bool init(Backend *be, const EngineConfig &cfg, std::string *err);
    bool moe_prediction_enabled() const { return moe_predict_enabled_; }
    u64 moe_predictions_issued() const { return moe_predictor_.predictions_issued(); }
    u64 moe_prediction_candidates() const { return moe_predictor_.candidates(); }
    u64 moe_prediction_hits() const { return moe_predictor_.candidate_hits(); }
    u64 moe_prediction_staged() const { return moe_predictor_.staged(); }
    Backend::ExpertPoolStats expert_pool_stats() const {
        return be_ ? be_->expert_pool_stats() : Backend::ExpertPoolStats{};
    }
    void shutdown();

    bool generate(const GenerateParams &p, GenerateResult *res);

    // ---- draft model (speculative decoding) ------------------------------
    // Greedy speculative decoding: the draft engine proposes up to
    // `spec_decode_tokens` continuations, the target verifies them in ONE
    // batched forward, and the longest matching prefix is kept. Because only
    // argmax branches are pursued, accepted tokens are bit-identical to plain
    // greedy decoding — sampling is rejected with an error instead of changing
    // the distribution.
    bool load_draft(const std::string &path, std::string *err);
    void unload_draft();
    void set_draft_window(i32 n) { draft_tokens_ = n > 0 ? n : 1; }
    bool has_draft() const { return draft_ != nullptr || dflash_ != nullptr; }
    // Whether the draft is a full MODEL (as opposed to a DFlash head set, which
    // has no Model of its own). draft_model() dereferences draft_, so this is
    // the guard for reading it: has_draft() is true for both kinds.
    bool has_draft_model() const { return draft_ != nullptr; }
    const Model &draft_model() const { return draft_->model(); }
    // The draft engine itself, so a test can compare its KV against the
    // target's while both are still alive (see self_draft_kv_parity).
    const Engine *draft_engine() const { return draft_; }
    i32 draft_window() const { return draft_tokens_; }

    // ---- external draft head (EAGLE-3) -----------------------------------
    // A head is not a model and not a nextn block inside the target: it is a
    // small GGUF beside the model that reads the TARGET's hidden state and the
    // last token's embedding, and proposes `--draft-tokens` continuations for
    // the same verify loop. Convert one with tools/eagle_to_gguf.py. The head
    // runs on the HOST (one block, milliseconds), so it competes with the
    // target's tiers for no VRAM at all. Refused on a recurrent target, whose
    // delta-rule state cannot be rewound after a rejected suffix.
    bool load_draft_head(const std::string &path, std::string *err);
    void unload_draft_head();
    bool has_draft_head() const { return head_ != nullptr; }
    const DraftHead *draft_head() const { return head_; }

    // ---- DFlash drafter ---------------------------------------------------
    // `--draft` takes either a full model of the target's family (the path
    // above) or a Poolside DFlash head set, which is not a model at all: it has
    // no embedding and no head, reads the *hidden states* of a few target layers
    // through a fusion encoder, injects their K/V into a cache of its own, and
    // drafts a masked block against it. Which one a path names is read off the
    // file's architecture, so the caller does not have to know.
    bool load_dflash(const std::string &path, std::string *err);
    void unload_dflash();
    bool has_dflash() const { return dflash_ != nullptr; }
    const DflashDraft *dflash() const { return dflash_; }

    // Test/diagnostic access to one draft_block call: exactly the speculative
    // loop's block, wrapped so a caller outside the engine (the suite, a probe
    // tool) can read the rows without a full generation. `rows.data` is one f32
    // candidate-logits row per proposer, row k answering position pos+k.
    struct DflashRows {
        std::vector<f32> data;
    };
    i32 dflash_block_size() const;
    i32 draft_block_rows(const QuantTensor &tok_embd, const QuantTensor &lm_head,
                         i32 n_vocab, i32 id_last, i32 n_draft, i32 pos,
                         DflashRows &out);

    // ---- MTP (nextn) block probe -----------------------------------------
    // One replay of the file's own nextn block over the prompt just prefilled,
    // and the comparison that says whether it drafts the target's token:
    // `draft_id` is the block's argmax for position pos + 1, `target_id` is the
    // target's own argmax for the same position (read from the prefill's last
    // logits row BEFORE the block's head overwrites it). Equal ids mean a
    // speculative round would have accepted that token.
    struct MtpProbe {
        bool ok = false;
        i32 pos = 0;        // position the block attended at
        i32 draft_id = -1;  // the block's prediction of position pos + 1
        i32 target_id = -1; // the target's prediction of position pos + 1
        f32 draft_logit = 0.0f;
        f32 target_logit = 0.0f;
        // Where the target's token sat in the DRAFTER's own distribution, and
        // the logit it got there. The pair separates the two failure modes a
        // low acceptance rate has: a head that puts the right token second is a
        // calibration/quality problem, while a head that puts it at rank 4000
        // is not reading the same distribution it was trained on
        // (wrong hidden capture, wrong norm, wrong concat). Top-1 agreement
        // alone cannot tell those apart, because both report 14%.
        i32 target_rank = 0;   // 1 = the drafter had it first
        f32 target_in_draft = 0.0f;
        // The capture's own fidelity: the captured pre-final-norm row pushed
        // through the model's OWN output norm and head (KRK_MTP_CAPCHECK=1).
        // It skips the nextn block entirely, so it is the row, not the block:
        // if this disagrees with the target the stored hidden is the wrong row
        // (an off-by-one position, or a copy that lost its exponent), and if it
        // agrees, every token the drafter loses is the block's doing.
        i32 capture_id = -1;
        f32 capture_logit = 0.0f;
        f64 ms = 0.0;
        std::string note; // why it did not run, when ok is false
    };
    MtpProbe mtp_probe_step();
    bool mtp_probe_enabled() const { return mtp_probe_; }
    bool mtp_spec_active() const { return mtp_spec_; }
    // Admits the nextn block's routed set to VRAM once and pins it. Called from
    // init, after the expert budget has been clamped to what is really free.
    bool mtp_pin_experts();

    // ---- KV cache introspection -------------------------------------------
    // Positions [0, kv_pos()) hold keys/values; kv_pos() is also the position
    // the next forward() will write to.
    i64 kv_pos() const { return kv_pos_; }
    i64 kv_capacity() const { return kv_cap_; }

    // ---- decode-batch telemetry -------------------------------------------
    // How many single-row layers the batched MoE path computed, and how many
    // kept the per-expert loop. The pair is the only observable that says WHY
    // a decode step took the loop: the batch resolves its whole routed set
    // before launching, so it refuses when the expert cache cannot hold that
    // set (see Engine::moe_decode_batch). A run whose cache is too small to
    // hold the routed set must therefore report `ran == 0` -- that is what
    // test_moe_batch_fallback pins, because the alternative (batching into
    // aliased slots) produced plausible text from the wrong weights.
    u64 moe_batch_layers() const { return moe_batch_layers_; }
    u64 moe_batch_skips() const { return moe_batch_skips_; }
    // Logical rollback to `pos` (must be <= kv_pos()). The occupied region is
    // simply treated as shorter — no zeroing is needed because every forward
    // fully overwrites the rows it touches and attention never reads past
    // kv_pos(). This is the primitive speculative decoding builds on.
    // A recurrent model is the exception: the delta rule has no inverse, so
    // only a rewind to 0 is honoured (the server's per-request isolation,
    // after which the caller replays the prompt and rebuilds the state).
    bool kv_rollback(i64 pos);

    // Speculation telemetry (valid after a run with a draft loaded).
    u64 draft_proposed() const { return draft_proposed_; }
    u64 draft_accepted() const { return draft_accepted_; }
    // Mean acceptance a rejection-sampling verifier would report, summed over
    // proposals under the served regime (temp 1.0, top-k 20): the number a
    // model card's acceptance length is measured with, against the greedy
    // match rate above it.
    f64 draft_sampled_expected() const { return draft_sampled_e_; }
    u64 draft_sampled_n() const { return draft_sampled_n_; }
    u64 spec_steps() const { return spec_steps_; }

    // KV tier telemetry, summed over the K and V planes. The counters existed
    // in KvTierCache but nothing read them after startup, so a run could page
    // the whole cache through WARM and COLD and report nothing about it. This
    // is what the end-of-run [stats] line is built from.
    void kv_tier_stats(KvTierStats *out) const {
        KvTierStats a, b;
        kvt_k_.stats(&a);
        kvt_v_.stats(&b);
        out->tiered = a.tiered || b.tiered;
        out->layers = a.layers;
        out->hot = a.hot;
        out->warm = a.warm;
        out->cold = a.cold;
        out->slots = a.slots;
        // Both planes, so this is the VRAM the HOT tier actually holds and not
        // half of it. layer_bytes stays per-plane: it is a slot's size, and a
        // slot is one plane's copy of one layer.
        out->hot_bytes = a.hot_bytes + b.hot_bytes;
        out->layer_bytes = a.layer_bytes;
        out->promotions_from_warm = a.promotions_from_warm + b.promotions_from_warm;
        out->promotions_from_cold = a.promotions_from_cold + b.promotions_from_cold;
        out->evictions_to_warm = a.evictions_to_warm + b.evictions_to_warm;
        out->evictions_to_cold = a.evictions_to_cold + b.evictions_to_cold;
        out->promotion_refused = a.promotion_refused + b.promotion_refused;
        // Counted per plane, so this is pages, and the same lost layer counts
        // twice -- once for K and once for V. Reported apart from ->COLD
        // because it is the one number here that says the run is wrong.
        out->evictions_dropped = a.evictions_dropped + b.evictions_dropped;
        out->migrate_bytes = a.migrate_bytes + b.migrate_bytes;
    }

    // Hybrid CPU + GPU expert compute (see EngineConfig::hybrid_experts).
    bool hybrid_enabled() const { return hybrid_on_; }
    void hybrid_stats(HybridStats *out) const {
        out->threads = hybrid_threads_live_;
        out->cpu_experts = hy_cpu_experts_;
        out->gpu_experts = hy_gpu_experts_;
        out->cpu_rows = hy_cpu_rows_;
        out->cpu_ms = hy_cpu_ms_;
        out->policy_violations = hy_policy_violations_;
    }

    const KvTierCache &kvt_k() const { return kvt_k_; }
    const KvTierCache &kvt_v() const { return kvt_v_; }
    const Model &model() const { return model_; }
    // The host->device promotion probe this run MEASURED: the three rates, and
    // with them which pageable path and which pinned prefix it selected. Read
    // this instead of parsing the log line -- and, in the suite, compare the
    // engine's actual choice against it: asserting a fixed verdict from a
    // wall-clock measurement asserts the test machine's scheduler, not the code.
    const H2dProbe &h2d_probe() const { return h2d_probe_; }
    // The probe's cross-run ledger as this init left it: how many runs this
    // machine has probed, how many reused a cached verdict, and how much time
    // the measurements actually cost. The end of a run prints it -- one run's
    // verdict is a hit rate of 100% or 0% and says nothing on its own.
    const H2dProbeTally &h2d_tally() const { return h2d_tally_; }
    const Tokenizer &tokenizer() const { return tok_; }

    // The one place that decides how a prompt string becomes token ids, so the
    // `--tokenize` dump cannot disagree with what `generate()` actually feeds
    // the model. The rule that matters: a chat template which emitted
    // `bos_token` has already opened the sequence, so `add_special` must not
    // add a second BOS.
    int tokenize_prompt(const std::string &prompt, std::vector<i32> &ids) const;
    // Debug: logits from the most recent forward (n_vocab floats).
    const f32 *last_logits() const { return logits_host_; }
    // Test hook: the resident plane a layer's attention reads, as the cache
    // handed it to the kernels. `which` is 0 for K and 1 for V. A shared-KV
    // layer returns the SAME pointer as the layer it shares with (that aliasing
    // is the feature), and a test can also read the rows to check what the
    // value norm did to them.
    const void *kv_plane_debug(i32 which, i32 layer);
    // Test hook, qwen4exp only: the hyper-connection lane stack after the last
    // forward pass, as f32 host rows [n_tok, hc, n_embd], or nullptr when the
    // file carries no lane stack (or the backend keeps it on device only). The
    // suite pins the identical-lane invariant through the REAL forward path
    // with this -- hc.cpp proves the arithmetic, this proves the wiring.
    const f32 *hc_debug_lanes(i32 *hc_out, i64 *width_out) const;
    // Test hook, qwen4exp only: the PLE table rows the LAST forward pass hashed,
    // as [n_tok, n_heads] (head fastest), or nullptr when the file carries no
    // PLE block. The hash is host work on both backends, so this is a plain
    // pointer into the engine's own staging buffer.
    const i64 *ple_debug_rows(i32 *n_tok, i32 *n_heads) const;
    // Test hook, qwen4exp PLE only: the block's three terms from the LAST
    // forward, each `hc * n_embd` wide and laid out lane-major like the stack
    // they feed. `gated` is the RAW `value * gate` the residual receives;
    // `conv_in` is the lane-normed copy the convolution reads; `conv_out` is
    // that convolution after its silu. The reference (`build_ple` in
    // src/models/qwen4exp.cpp) adds `gated` to the hidden stack and uses the
    // normalized copy for the conv only, so these are two different rows: the
    // one time they shared a buffer the block put a per-lane unit-RMS row on
    // the residual and the model's text was garbage, which is why the suite
    // checks the split. Null on a device backend, like hc_debug_lanes.
    const f32 *ple_debug_terms(f32 **conv_in, f32 **conv_out, i64 *n_tok) const;
    // Test hook, qwen4exp only: the cells the LAST forward pass's indexer
    // selected for layer `l`, ascending, and empty when that layer carries no
    // indexer (or the run was dense, which is every context shorter than the
    // layer's `top_k + ratio - 1` budget). The selection is host work -- pooled
    // keys, scores and top-k all run there -- so this reads the engine's own
    // buffers, and it is what pins the rule against an oracle in the suite.
    const std::vector<i64> &qsa_debug_cells(i32 layer) const;
    // That layer's compress ratio (0 when it has no indexer), so a caller can
    // tell "dense because everything fits" from "no indexer at all".
    i32 qsa_debug_ratio(i32 layer) const;
    // Hash of every dumped activation stage of the LAST prefill, folded
    // per (stage, layer) in sorted-key order. Bit-exact: two prefills of
    // the same prompt at different --chunk values must return the same
    // number, and the first kernel that stops agreeing shows up here even
    // when the difference is absorbed before the logits row.
    u64 prefill_trace_hash() const;
    const DeviceCaps &device() const { return be_ ? be_->caps() : no_caps_; }

    // ---- routing scan (public surface; see the members for why) ----------
    // Turns the routing scan on, and (optionally) the expert-stubs-only
    // approximation. Must be called before init(): the buffers are sized there.
    void set_expert_scan(bool scan, bool stub) {
        expert_scan_ = scan;
        expert_stub_ = scan && stub;
    }
    // Writes <path> with each expert's routing mass, sorted hottest-first.
    void write_expert_index(const std::string &path);
    // Pulls the device histograms into expert_mass_/expert_hits_ (no-op when
    // they are not in use). Called before either writer, so the file is always
    // built from the complete picture.
    void scan_dev_fold();
    // Counts routing during a NORMAL run -- no stub, no index write -- so the
    // run can feed what it actually routed back into the index. Must be set
    // before init(): the counters are sized there.
    void set_expert_track(bool on) { expert_track_ = on; }
    // Folds this run's observed routing into <path>. The next load then ranks
    // the tiers by what the workload ran rather than by the measurement prompt
    // alone, so a repeated workload converges on its own hot set instead of on
    // the scan's generic one. `retain` is the weight the existing file keeps
    // (0.5 = half); the run contributes the rest. Refuses, and leaves the file
    // alone, when it describes a different model; writes a fresh one when there
    // is no file yet.
    void merge_expert_index(const std::string &path, double retain);
    // Routing residency preference, for the end-of-run report: was it on, at
    // what margin, and how many picks it moved.
    bool router_pref() const { return router_pref_; }
    f32 router_tie_margin() const { return router_tie_margin_; }
    u64 router_picks() const { return router_picks_; }
    u64 router_subs() const { return router_subs_; }
    u64 router_hot() const { return router_hot_; }
    // Why the last forward refused, when it did. Public because a gate that
    // asserts a run FAILS has to be able to read the reason: "the run stopped"
    // and "the run stopped for the right cause" are different claims, and only
    // the second one proves the check that fired is the check under test.
    std::string forward_failure() const { return forward_failure_error(); }

private:
    std::string kv_resource_error() const;
    std::string expert_resource_error() const;
    // Prefers forward_error_ (expert refusal) over the generic KV message.
    std::string forward_failure_error() const;
    const std::string &forward_error() const { return forward_error_; }
    // One batched forward over n tokens starting at pos0. LogitMode picks how
    // much of the output head runs: none (mid-prefill), the last row (plain
    // decode), or every row (speculative verification).
    bool forward(const i32 *toks, i32 n, i32 pos0, bool want_logits);
    bool forward_core(const i32 *toks, i32 n, i32 pos0, LogitMode mode);
    // GPT-2 runs its own loop rather than threading four family-specific
    // branches through forward_core: LayerNorm with a bias, learned absolute
    // positions, a fused attn_qkv whose row is Q|K|V, and an ungated GELU MLP.
    // Those four are one architecture, and a separate loop keeps every other
    // family's graph literally unchanged.
    // The plain pre-norm family (GPT-2, GPT-NeoX, GPT-J, Phi-2): LayerNorm,
// a fused or separate QKV, optional rotary or learned positions, an
// optional parallel residual and an optional gated MLP. Which of those
// are on is the arch table's business, not this loop's.
    bool plain_forward(const i32 *toks, i32 n, i32 pos0);
    // LayerNorm with a bias, and the Q|K|V split of a fused projection row.
    // Both are GPT-2-only and both go through the backend's own download /
    // upload, so they are correct on every backend rather than only the one the
    // author happened to run. Neither is hot: a LayerNorm here is E = 64 wide
    // and the split is a strided copy, against E-by-3E GEMMs either side.
    void layernorm(void *out, const void *x, const f32 *w, const f32 *b,
                   i64 rows, i64 n, f32 eps);
    void split_qkv(void *q, void *k, void *v, const void *qkv, i64 rows, i64 E);
    // This pass's input token ids (forward_core's `toks`), for the ONE consumer
    // that needs a vocabulary id rather than a position: DeepSeek-V4's hash
    // route selects experts BY TOKEN. A row index is not a token id -- on a
    // multi-row prefill the two differ, and the route then sends every prompt
    // row but the first to another token's experts (measured: chunk 1 and chunk
    // 8 disagreed on the whole-graph fixture). Not owned; valid for the pass.
    const i32 *fwd_toks_ = nullptr;
    i32 fwd_toks_n_ = 0;
    // Attributes the expert tier's stage timers to the phase the engine is
    // running. Prefill and decode promotions are the same DMA behind different
    // row counts, and the end-of-run report splits them apart
    // (ExpertCache::set_phase). The draft engine's tier moves with the
    // target's, so it is attributed to the same phase.
    void expert_phase(ExpertCache::Phase p);
    // Runs the output head on one row of ws_x_ into ws_logits_.
    void head_compute(i32 row);
    // prism.hadamard fold: dst <- transform(src), see Backend::hadamard_act.
    // `inverse` picks the order used right after a latent embedding lookup,
    // `gdn_perm` adds ssm_out's grouped-V permutation ahead of both.
    void had_xform(void *dst, const void *src, i64 rows, i64 width,
                   bool inverse, bool gdn_perm);

    // KRK_DUMP: append the last row of `buf` (width x rows elements, f32 host
    // or f16 device) to the dump file, one text line per call. A CPU run and a
    // HIP run of the same prompt then differ at exactly one line per stage.
    // `row_scale` (nullable) is a per-row f32 multiplier the row was stored
    // divided by; it is applied on the host so the trace shows true scale.
    void dump_row(const char *what, i32 layer, const void *buf, i64 width, i64 rows,
                  const f32 *row_scale = nullptr);
    // Folds every row of `buf` into the prefill trace hash. The row-local
    // stages a chunk composition must leave unchanged; unlike the text dump it
    // covers ALL rows and never a stage that only one chain emits.
    void trace_rows(const char *what, i32 layer, const void *buf, i64 width,
                    i64 rows);
    // Zero-fill rollback for the (rare) shrink-a-lot path; the common small
    // rollback in speculative decoding is pure pointer arithmetic.
    bool kv_rollback_zero(i64 pos);
    // Routed-expert FFN for one layer: router -> top-k -> grouped per-expert
    // FFN -> optional shared expert. Tokens that picked the same expert are
    // permuted into one block, so each expert runs three batched GEMMs
    // regardless of how many tokens chose it. Experts are paged in through the
    // model's cache.
    bool moe_ffn(const LayerWeights &L, i32 layer, i32 n);

    // One decode token's routed experts as BATCHED launches: gate/up for up to
    // four experts per group launch (they all read the same activation row at
    // one token), one activation launch for all of them, one launch for their
    // down projections (each reading its own row), and the scatter per expert
    // unchanged -- the routed sum's order and its storage exponents are the
    // per-expert path's. Returns false having issued nothing when the layer
    // cannot be expressed this way, so the caller's per-expert loop runs.
    bool moe_decode_batch(const LayerWeights &L, i32 layer, i32 k, i32 ne,
                          i64 ff_exp, bool use_plan_off, bool inplace_ok);
    // Set when moe_ffn (or the MTP block path) refuses a routed expert:
    // forward_core returns false and the next error surface prefers this
    // over the generic KV message. Cleared at each forward_core entry.
    std::string forward_error_;
    // The dense (non-MoE) SwiGLU FFN both layer shapes share: gate/up on one
    // fused launch, silu, down, residual. Shared because the recurrent path's
    // copy was the stale one and had drifted off the fused group.
    void dense_ffn(const LayerWeights &L, i32 l, i64 n,
                   void *dst = nullptr, const f32 *post_norm = nullptr);
    // gemma4's routed block: the shared MLP and the expert bank are two
    // separately-normed branches off the same post-attention row, and the
    // router reads a weightless norm of that row rather than either branch's
    // input. `moe_ffn` cannot express it (one norm, one bank, one residual
    // add), so the block is assembled here from both.
    bool gemma_moe_ffn(const LayerWeights &L, i32 layer, i64 n);
    // dst += x * W^T as ONE launch where the backend can fold the add into the
    // projection's epilogue (Backend::gemm_accumulate), and as the
    // gemm()-into-scratch + add_inplace() pair everywhere else -- including
    // under KRK_DUMP, whose projection stage reads `scratch`.
    // `gs` is the weight's NVFP4 global scale (QuantTensor::gs): the folded
    // epilogue cannot express it, so a scaled weight takes the two-call path.
    void gemm_residual(void *dst, const void *x, const void *w, DType wt,
                       i64 n_out, i64 n_in, i64 rows, void *scratch,
                       f32 gs = 1.0f);
    // dst = x * W^T, with the weight's NVFP4 global scale applied.
    //
    // Every DENSE weight goes through this instead of Backend::gemm() so that a
    // `"<tensor>.scale"` sidecar cannot be honoured at one call site and missed
    // at another: the scale is a property of the tensor (QuantTensor::gs), and
    // folding it into the projection's RESULT is exact because the projection is
    // linear -- gs*(W x) is the same number as (gs*W) x, and everything
    // downstream then sees exactly what a model with the true weights sees. On a
    // weight with no sidecar it is one plain gemm() call, which is every
    // non-NVFP4 file.
    void gemm_w(void *dst, const void *x, const QuantTensor &w, i64 n_out,
                i64 n_in, i64 rows);
    // The vocabulary head, one row: the projection plus the head's own bias
    // (`output.bias`) when the file carries one. Every path that forms logits
    // goes through here so the bias cannot be present on one and missing on
    // another -- the device top-k, the host sampling path and the MTP probe all
    // read the same ws_logits_ buffer, and a bias applied after the download
    // would leave the device argmax choosing from a different distribution.
    void logit_head(const void *xn);
    // ---- qwen4exp hyper-connections --------------------------------------
    // The lane stack replaces the single residual stream: each sub-block is
    // bracketed by a mix (lanes -> one row, left in ws_xn_ AND ws_x_) and a
    // combine (the block's output -> every lane). `hc_enter` also saves the
    // mixed row, because the block adds its output onto the row it was handed
    // and the combine needs the block output alone.
    void hc_enter(const f32 *norm_w, const QuantTensor &down,
                  const QuantTensor &up, const QuantTensor &inject, i32 n);
    void hc_leave(i32 n);
    // The head's mixer: row `row` of the lane stack -> one f32 row in ws_xn_.
    // There is no injection (nothing is scattered back) and no output_norm --
    // this mixer IS the final norm on this family.
    void hc_head_mix(i32 row, const f32 *norm_w, const QuantTensor &down,
                     const QuantTensor &up);
    // ---- qwen4exp per-layer embedding (PLE) ------------------------------
    // The n-gram block that runs on the layers `ple.layers` names, BEFORE that
    // layer's hyper-connection mixer: hash -> mmap'd table rows -> key/value ->
    // per-lane gate -> dilated depthwise conv -> residual into the lane stack.
    // It is the only place this engine reads a weight straight out of the file
    // mapping on the hot path, which is why the rows are computed on the host.
    void ple_block(const LayerWeights &L, i32 l, i32 n, i64 pos0);
    // ---- qwen4exp QSA indexer --------------------------------------------
    // Pooled keys per block of `ratio` cells -> one score per block -> top-k
    // blocks -> the cells of those blocks. Returns true when this row set is a
    // SELECTION (some visible cell was dropped) and the caller must therefore
    // attend over the chosen cells instead of the whole cache; false is the
    // dense case, which is every context inside the layer's budget and is
    // exactly what the reference does there too.
    bool qsa_select_block(const LayerWeights &L, i32 l, i32 n, i64 pos0,
                          const void *xn);
    // Dense causal attention over each row's selected cells, gathered into a
    // compact buffer: `attention()` has no mask, and the cell list is per
    // QUERY row, so the compact form is the one that needs no new kernel.
    void qsa_attention(i32 l, i32 n, const AttnDesc &d);
    // Chooses the expert residency budget from cfg_ and the device.
    // Chooses the device-side expert budget. Returns false with *err set when
    // an explicit --expert-cache-mb cannot fit the card, which is a refusal
    // rather than a trim: see the reserve rule in the definition.
    bool configure_expert_cache(std::string *err);
    // Fills the two residency tiers ahead of the run, hottest-first: WARM from
    // the ranked expert list, then the top of that order onto the device. The
    // point is the ORDER, not the volume -- a tier smaller than the routed set
    // holds what routing actually wants instead of whatever the first few
    // tokens happened to touch -- and the wall-time ceiling is what keeps the
    // phase inside a cold start. See EngineConfig::expert_warmup.
    void warm_experts();
    // The hot ring: the top of the expert index admitted into VRAM and pinned,
    // charged to a budget the LRU pool does not see (see ExpertCache's
    // set_pin_budget / admit_hot). Runs between the warmup and the pool clamp,
    // so the clamp sizes the pool from the VRAM that is really free after the
    // ring has taken its bytes. KRK_EXPERT_PIN=0 is the A/B arm;
    // KRK_EXPERT_PIN_MB=<n> sizes the ring (default: half the measured room
    // beside the VRAM floor).
    void pin_hot_experts();
    // Publishes the device dispatch table (ExpertCache::publish_dispatch_table)
    // when the whole routed set is device-resident, so a layer can pick its
    // experts' matrices without a host pointer per expert. Called at the end of
    // init -- after the warmup and the pins, because both move pointers -- and
    // silent when the residency promise cannot be made.
    void publish_expert_table();
    // Re-derives the expert pool's byte budget from the VRAM that is free NOW.
    // The init-time clamp runs once, before the run; the KV tier promotes
    // COLD->HOT as the context grows, so the bytes it named can be gone by the
    // time decode is deep. Called once per forward from forward() -- a
    // hipMemGetInfo is a query, not a drain -- and it only ever LOWERS the
    // budget, because raising it past what init measured is how the pool grows
    // into the card and turns every alloc into a 30 ms driver eviction.
    void reclamp_expert_budget(const char *why);
    // total = weights + kv + experts + workspaces + free + unaccounted, every
    // term measured. Called at the end of init and at teardown, so a buffer
    // allocated after init (a per-forward scratch, a promoted layer) shows up as
    // a change in `unaccounted` instead of being folded into `free`.
    void log_vram_ledger(const char *when);
    // Every workspace allocation made through init's `alloc` lambda, in bytes.
    // It exists so the VRAM ledger can print a measured sum instead of leaving
    // the remainder of the card to be inferred by subtracting four numbers from
    // the total -- which is how "scratch = 3.0 GiB" stood without a ledger, and
    // that number decides whether the 8 GB and 6 GB tiers can run at all.
    size_t ws_bytes_ = 0;
    // The KV cache's own bytes, both planes, as allocated at init. A member so
    // the ledger can be printed at teardown as well as during init.
    size_t kv_bytes_ = 0;
    // Top-k' lookahead: how many experts the router ranks (absolute), or 0 for
    // just the k it selects. KRK_EXPERT_LOOKAHEAD=<k'>. The extras are only
    // staged when they are already in WARM (a copy, never a read).
    i32 expert_lookahead_ = 0;
    u32 reclamp_tick_ = 0;
    // Set once this run has stored a residual row divided by a power of two
    // (Backend::add_residual's exponent). It disables the folded projection
    // store (Backend::gemm_accumulate), which divides its contribution by the
    // row's CURRENT exponent and so cannot hold a sum that needs a bigger one.
    // Sticky and one-way: see the note in Engine::forward.
    bool resid_scaled_ = false;
    // `<model>.krakenexperts.json` (see write_expert_index) as a hottest-first
    // (layer, expert) list. Empty when the file is missing or does not describe
    // this exact model: a ranking measured on other weights is worse than no
    // ranking, so a stale index is refused rather than trusted.
    std::vector<std::pair<i32, i32>> load_expert_index() const;
    // Whether the ranked warmup should run for this configuration: false for a
    // routing scan, a dense model, or a run with no WARM tier, and otherwise
    // cfg_.expert_warmup, whose default (-1) is ON for every MoE with a device
    // tier and is the corpus-vs-budget comparison only on the CPU backend.
    // KRK_EXPERT_WARMUP=0/1 overrides all of it, which is what makes the A/B
    // one variable wide.
    bool expert_warmup_wanted() const;
    // The one index writer: the scan and the adaptation pass fill the same
    // arrays, so they must not produce two formats -- the loader reads both.
    void write_expert_index_rows(const std::string &path,
                                 const std::vector<double> &mass,
                                 const std::vector<u64> &hits, i64 tokens,
                                 const char *mode) const;
    // One gated-delta-net (recurrent) layer: the fused [q|k|v] projection, the
    // causal short conv, the per-head gate/forget, the delta rule, and the
    // gated output projection. Advances the layer's recurrent state.
    void gdn_forward(const LayerWeights &L, i32 l, i32 n);
    // LFM2.5's short-convolution mixer (the layers whose block carries
    // `shortconv.in_proj`). `pos0` is the absolute position of the first row:
    // the history ring is keyed by it.
    void lfm_conv_forward(const LayerWeights &L, i32 l, i32 n, i64 pos0);
    // One multi-head latent attention layer (DeepSeek-V2/V3/V4, legacy lite
    // layout): the [kv_lora + rope] projection, the normed latent written to
    // the KV plane, and the absorbed score/value pass over it. Leaves the
    // attention result in ws_attn_ at `L.wo.n_in` columns per row, which is
    // where the ordinary output projection picks it up. False when a geometry
    // this engine cannot run reaches it -- a refusal, not a wrong answer.
    bool mla_forward(const LayerWeights &L, i32 l, i32 n, i32 pos0);
    // The same layer in the SHARED-latent layout (DeepSeek-V4): the query trunk,
    // ONE normed latent row per token used as both K (rope-rotated on read) and
    // V, and the grouped low-rank output. Unlike the lite path this one writes
    // the layer's FINAL [n, n_embd] attention output (wo_a/wo_b are inside), so
    // the caller must not run `attn_output` over it a second time.
    bool mla_shared_forward(const LayerWeights &L, i32 l, i32 n, i32 pos0);
    // The same layer when the file declares a compress ratio for it (V4
    // CSA/HCA): the query trunk and the grouped output are unchanged, the
    // ATTENTION reads two row families -- the raw window and the pooled keys
    // this method maintains. See Engine::comp_* for the streams.
    bool mla_shared_forward_compressed(const LayerWeights &L, i32 l, i32 n,
                                       i32 pos0);
    // Clears the convolution windows and delta-rule states (f32, one set per
    // recurrent layer). Only correct to call at a sequence boundary.
    void recurrent_reset();
    // Downloads the last row of logits into logits_host_. With device_only true the
    // row stays in VRAM and is reduced to a single greedy argmax by
    // topk_argmax() instead; see fetch_logits() for when that is legal. The
    // flag reads the same way as the caller's, on purpose: an inverted sense
    // here silently disables the fast path with identical output.
    void fetch_logits(bool device_only = false);
    // Device top-k, reduced to the one number greedy decoding needs. Returns
    // false when the caller must use the host row instead: a CPU backend, or a
    // cut so wide the candidate buffer overflowed (a whole shelf of tied
    // logits, which is what a softcapped model produces at the very top).
    bool topk_argmax(i32 *out);
    // Greedy speculative step; false when the draft cannot help.
    bool generate_speculative(const GenerateParams &p, GenerateResult *res,
                              const std::vector<i32> &ids, f64 *prefill_ms,
                              f64 *decode_ms, i64 *decode_steps);
    // The DFlash variant: the drafter is a head set, not an engine, so the
    // proposals come from a masked block instead of a second model's decode
    // loop, and the target's captured activations are what feed it.
    bool generate_speculative_dflash(const GenerateParams &p, GenerateResult *res,
                                     const std::vector<i32> &ids, f64 *prefill_ms,
                                     f64 *decode_ms, i64 *decode_steps);
    // Copies the residual stream entering `layer` (or, for mc.n_layer, the
    // pre-final-norm state) into the DFlash drafter's capture buffer. Inert when
    // no drafter is loaded, which is every run that does not ask for one.
    void dflash_capture(i32 layer, i32 n);

    // ---- nextn (MTP) probe ------------------------------------------------
    // Copies this pass's pre-final-norm hidden rows and their token ids into
    // the probe's capture buffer, so the block can be replayed over the whole
    // prompt rather than over its last row alone.
    void mtp_capture(const i32 *toks, i32 pos0, i32 n);
    // One block forward at `pos`: the captured hidden row (hnorm) and the
    // position's token embedding (enorm) are fused by eh_proj, then the block's
    // own attention runs against its own KV plane (written at `pos`), and its
    // routed + shared experts run through the model's ExpertCache. With
    // want_logits the shared head writes ws_logits_ for position pos + 1.
    bool mtp_block_forward(const MtpBlock &m, i32 pos, bool want_logits);
    // The block's routed + shared expert FFN. Same residency path as a stack
    // layer, keyed on the block's own index, so its experts page in and out of
    // the same budget as everything else.
    bool mtp_ffn(const MtpBlock &m, i32 n);
    void mtp_dense_ffn(const MtpBlock &m, i32 n, void *acc, i32 *exp);
    // One token's embedding row into mtp_e_, with the latent-embedding
    // un-rotation the stack applies at the same point.
    void mtp_embed(i32 tok);
    // Feeds every captured position in [mtp_fed_, pos) through the block, with
    // no head. This is what makes the block's KV describe the same history the
    // target's does; without it the drafter reads rows it invented.
    bool mtp_prime(i64 pos);
    // Greedy speculative decode driven by the nextn block.
    // The round the external draft head proposes; same verify/accept/realign
    // shape as the nextn block's, with the proposal side on the host. Added to
    // the engine's declarations below by load_draft_head's section.
    bool generate_speculative_head(const GenerateParams &p, GenerateResult *res,
                                   const std::vector<i32> &ids, f64 *prefill_ms,
                                   f64 *decode_ms, i64 *decode_steps);
    bool generate_speculative_mtp(const GenerateParams &p, GenerateResult *res,
                                  const std::vector<i32> &ids, f64 *prefill_ms,
                                  f64 *decode_ms, i64 *decode_steps);

    Backend *be_ = nullptr;
    Model model_;
    Tokenizer tok_;
    EngineConfig cfg_;
    DeviceCaps no_caps_;

    // The draft engine owns its own Model and KV cache but shares this
    // backend, so the two models can interleave kernels without copies.
    Engine *draft_ = nullptr;
    // The DFlash head set, if `--draft` named one. Shares this engine's backend
    // and this engine's activations; owns its own KV cache and workspaces.
    DflashDraft *dflash_ = nullptr;
    i32 draft_tokens_ = 0;
    u64 draft_proposed_ = 0;
    u64 draft_accepted_ = 0;
    f64 draft_sampled_e_ = 0.0;
    u64 draft_sampled_n_ = 0;
    u64 spec_steps_ = 0;

    i64 q_dim_ = 0, kv_dim_ = 0, n_embd_ = 0, n_ff_ = 0, n_vocab_ = 0;
    // Row width of the attention query buffer: Qwen3.5 packs a per-head output
    // gate behind the query, so its projection is 2*q_dim wide before the
    // engine splits it apart.
    i64 q_proj_ = 0;
    i64 conv_dim_ = 0, value_dim_ = 0;
    i64 kv_cap_ = 0;
    i64 kv_pos_ = 0;
    i64 chunk_ = 1;

    // ---- nextn (MTP) probe state (allocated only when asked for) ----------
    // The block attends over the prompt, so the capture is per position -- one
    // row is not enough to judge it. Its KV is its own plane pair: sharing the
    // stack's would mix two different key/value spaces in one cache, which is
    // the "two instances, one namespace" defect this repo has been burned by.
    bool mtp_probe_ = false; // the capture, block KV and workspaces exist
    bool mtp_spec_ = false;  // decode with the block as the drafter
    // Where a speculative round's wall time goes. Proposals, the verification
    // pass, the hybrid state snapshot and the block re-feed are the four terms;
    // a round that loses is a round whose verify is not amortized by the rows
    // it covers, and these say which of them it actually is.
    f64 mtp_propose_ms_ = 0.0;
    f64 mtp_verify_ms_ = 0.0;
    f64 mtp_snap_ms_ = 0.0;
    f64 mtp_refeed_ms_ = 0.0;
    // Where the propose pass's time goes: the block's own expert acquires, in
    // wall time, and how many of them were a device miss. The propose pass is a
    // single block forward, so if it is slow the answer is either the acquire
    // path (promotion) or the kernels, and those want opposite fixes.
    f64 mtp_acq_ms_ = 0.0;
    // The round's SECOND target forward. Every round pays exactly one: a
    // pre-check miss forwards the token it just emitted, a full match forwards
    // the bonus token so the next round has its state. It is not part of verify
    // (that is the batched pass over the proposals) and not part of refeed (that
    // is the block), so without its own term it shows up only as the gap
    // between the round and the sum of the phases -- which is how it went
    // unmeasured. `n` counts it: 1.0 forward per round is the whole point of
    // comparing against a plain step's 1.0 per TOKEN.
    f64 mtp_extra_ms_ = 0.0;
    u64 mtp_extra_n_ = 0;
    u64 mtp_acq_n_ = 0;
    u64 mtp_acq_miss_ = 0;
    // --mtp-probe asked for the per-step comparison. Separate from mtp_probe_,
    // which only says the buffers exist: a speculation run allocates the same
    // buffers and must not print probe lines.
    bool mtp_probe_report_ = false;
    // A hybrid model's delta-rule state has no inverse, so a rejected
    // speculative suffix cannot be rewound through kv_rollback. These hold the
    // pre-verification state (f32, the same size the engine allocated) and the
    // round restores them before replaying the accepted prefix.
    void *mtp_rec_snap_ = nullptr;
    void *mtp_conv_snap_ = nullptr;
    size_t mtp_rec_snap_bytes_ = 0;
    size_t mtp_conv_snap_bytes_ = 0;
    i64 mtp_kv_dim_ = 0;
    void *mtp_kv_k_ = nullptr;
    void *mtp_kv_v_ = nullptr;
    void *mtp_hidden_ = nullptr; // [kv_cap, n_embd] pre-final-norm rows
    void *mtp_h_ = nullptr;      // one hidden row, the block's hnorm input
    void *mtp_e_ = nullptr;      // one embedding row, the block's enorm input
    void *mtp_cat_ = nullptr;    // [2, n_embd] the eh_proj input
    void *mtp_gate_ = nullptr;   // the packed output gate's half
    void *mtp_sgate_ = nullptr;  // one f32: the shared expert's sigmoid gate
    void *mtp_rep_ = nullptr;    // one i32: the block's residual exponent
    std::vector<i32> mtp_tok_;   // token id per captured position
    i64 mtp_captured_ = 0;       // positions captured by the last pass
    // Positions already fed through the block's own KV. The block's attention
    // is causal over its own plane, so a second probe only has to append the
    // rows since the last one -- replaying the prompt every step would make the
    // per-step acceptance measurement quadratic for no new information.
    i64 mtp_fed_ = 0;
    // ---- row fidelity (KRK_MTP_ROWFID=1) ----------------------------------
    // For every HISTORY row the block builds, the block's row is compared with
    // the stack's own captured hidden at the same position: cosine between the
    // two rows, and the rank of the token that actually follows that position
    // under the stack's own head applied to each. The cosine says how far the
    // block's reconstruction drifts; the two ranks say whether that drift costs
    // anything on its own terms. Stored per row so the drift can be read
    // against row index rather than as one average.
    std::vector<i64> rowfid_pos_;
    std::vector<f32> rowfid_cos_;
    std::vector<i32> rowfid_blk_;
    std::vector<i32> rowfid_cap_;
    std::vector<f32> rowfid_a_; // one host row, block side
    std::vector<f32> rowfid_b_; // one host row, capture side
    // ---- propose-pass split (KRK_MTP_PROP_SPLIT=1) -------------------------
    // Host enqueue times are useless here: the block's work sits in the queue
    // until something drains it, so every phase reads as its launch cost. The
    // split therefore drains between phases (be_->sync()), which costs a little
    // latency but attributes the DEVICE time to the phase that earned it.
    f64 prop_embed_ms_ = 0, prop_block_ms_ = 0, prop_argmax_ms_ = 0;
    f64 prop_head_ms_ = 0; // charged inside prop_block_ms_, split out for the report
    u64 prop_timed_ = 0;

    // ---- external draft head state (allocated only when one is loaded) -----
    // The capture (one row per position, like the MTP probe's) and the two
    // host rows it is read into. `head_prime` walks the head over the captured
    // rows in [head_fed_, pos); `head_embed` puts one token's embedding on the
    // host, which is the other half of the head's input.
    void head_capture(const i32 *toks, i32 pos0, i32 n);
    bool head_embed(i32 tok);
    bool head_prime(i64 pos, std::string *why);
    DraftHead *head_ = nullptr;
    void *head_hidden_ = nullptr; // [kv_cap, n_embd] pre-final-norm rows
    void *head_e_ = nullptr;      // one embedding row, downloaded to the host
    std::vector<i32> head_tok_;   // token id per captured position
    std::vector<f32> head_row_;   // one host row: the captured hidden state
    std::vector<f32> head_emb_;   // one host row: a token embedding
    std::vector<f32> head_logits_;
    i64 head_captured_ = 0; // positions captured by the last forward
    i64 head_fed_ = 0;      // positions fed through the head's own KV
    // How many recent positions a round primes the head over. The head's fc is
    // O(hidden x target_hidden) per position, so a whole 4k prompt is hundreds
    // of GFLOP of host arithmetic; the window is the knob that keeps a round
    // cheap, and `0` means the whole prompt (the exact form).
    i32 head_prime_max_ = 256;
    f64 head_prime_ms_ = 0.0;
    f64 head_propose_ms_ = 0.0;

    // HOT/WARM/COLD KV residency. Two planes (K and V), one layer-sized slot
    // each. When untiered both hand back a pointer into one flat allocation
    // and d.layer_stride stays kv_cap * kv_dim, so the call sites below are
    // identical either way.
    KvTierCache kvt_k_;
    KvTierCache kvt_v_;
    bool kv_tiered_ = false;
    // Gated delta net: f32, one block per recurrent layer, persistent across
    // forwards. conv_state_ is the short conv's last (ksize-1) steps; rec_state_
    // is the delta rule's [n_v_head, d_state, hd] matrix. This is the model's
    // memory of the sequence — unlike the KV cache it cannot be recomputed
    // from a position, which is why rollback to 0 resets it.
    void *conv_state_ = nullptr;
    void *rec_state_ = nullptr;
    i32 rec_layers_ = 0;
    i64 conv_state_span_ = 0; // f32 per recurrent layer
    // ---- LFM2.5 short convolution (`lfm2moe`) ----------------------------
    // A different recurrent block: the mixer is `out_proj(B*conv(X) + C*X)` and
    // its memory of the sequence is a ring of the last `kern-1` X rows, keyed
    // by absolute position. sc_state_ is one span per LAYER (not per recurrent
    // layer) because the ring is indexed by the layer's own number, and the
    // activation width is the backend's -- f16 on the card, f32 on the
    // reference -- so the span is in elements and every address multiplies by
    // act_size.
    void *ws_sc_ = nullptr;   // [chunk, 3*n_embd]  the B | C | X projection
    void *ws_sc2_ = nullptr;  // [chunk, n_embd]    the gated convolution
    void *sc_state_ = nullptr;
    i64 sc_state_span_ = 0;   // elements per layer
    // Ring DEPTH (rows per layer), not the window: `kern - 1` rows are what the
    // convolution reads, and the rest is slack a speculative round needs. A
    // round verifies k+1 rows, accepts a prefix and then feeds the replacement
    // token at pos+ra -- which reads rows the verify pass produced. With a ring
    // exactly kern-1 deep those rows have been overwritten, so the rewind would
    // need a snapshot and a replay; with slack, `kv_rollback` is pointer
    // arithmetic and the conv side needs nothing at all. The slot function is
    // `pos % hist` in both cases, so a deeper ring is bit-identical.
    i64 sc_hist_ = 0;
    i64 rec_state_span_ = 0;  // f32 per recurrent layer
    // Mamba-2 layout ([head][p][s] vs GDN's [head][s][p] — same bytes, and the
    // scan never mixes them) and the per-layer conv bias, loaded only when a
    // checkpoint actually carries one. ws_m2in_ holds the fused [z|xBC|dt]
    // projection row, which is WIDER than the conv row (10304 vs 6144 on
    // nemotron-30B) and so cannot share ws_qkv_ with the GDN path.
    i64 mamba2_state_span_ = 0; // f32 per nemotron mamba-2 layer
    void *ws_m2in_ = nullptr;
    void mamba2_forward(const LayerWeights &L, i32 l, i32 n);
    // workspaces, sized for `chunk_` rows
    void *ws_x_ = nullptr;    // [chunk, n_embd]
    void *ws_xn_ = nullptr;   // [chunk, n_embd]
    // [chunk, n_embd] -- Falcon's second parallel norm, null unless
    // ArchSpec::parallel_two_norm.
    void *ws_xn2_ = nullptr;
    void *ws_x2_ = nullptr;   // [chunk, n_embd]
    // qwen4exp's lane stack and its mixer workspaces. All null unless the file
    // declares hyper_connection.count > 1; `hc_active_` is the single switch
    // the forward pass tests, so every other family runs the code it ran.
    bool hc_active_ = false;
    HcGeometry hc_;
    void *ws_hc_ = nullptr;     // [chunk, hc*n_embd] the lanes
    void *ws_hc_n_ = nullptr;   // [chunk, hc*n_embd] normed lanes
    void *ws_hc_g_ = nullptr;   // [chunk, hc*n_embd] mixer up-projection
    void *ws_hc_lo_ = nullptr;  // [chunk, low_rank]
    void *ws_hc_inj_ = nullptr; // [chunk, hc]
    void *ws_hc_mix_ = nullptr; // [chunk, n_embd] the mixed row + its negation
    // PLE workspaces. Null unless the file carries the block; `ple_state_` is
    // the conv history, one [(kern-1)*dil, hc*n_embd] window per layer, which is
    // what makes a chunked prefill and a token-at-a-time decode agree.
    bool ple_active_ = false;
    PleConfig ple_;
    i64 ple_hist_ = 0;          // (kern - 1) * ngram rows of history
    std::vector<i64> ple_seq_;  // the sequence's tokens, position-indexed
    std::vector<i64> ple_rows_; // [chunk * n_heads] this pass's rows
    i32 ple_rows_n_ = 0;        // how many tokens the last pass hashed
    std::vector<f32> ple_emb_;  // host staging for the gather
    std::vector<u16> ple_emb16_;
    void *ws_ple_emb_ = nullptr;   // [chunk, n_heads*head_dim] gathered rows
    void *ws_ple_key_ = nullptr;   // [chunk, hc*n_embd] key + its norm
    void *ws_ple_qry_ = nullptr;   // [chunk, hc*n_embd] lane-normed query
    void *ws_ple_gate_ = nullptr;  // [hc, chunk] per-lane gates (lane-major)
    void *ws_ple_val_ = nullptr;   // [chunk, n_embd] value
    void *ws_ple_gated_ = nullptr; // [chunk, hc*n_embd] value broadcast x gate
    // The conv's input: the lane-normed copy of the gated value. The residual
    // adds the RAW gated row (the reference's `gated`), so the two cannot share
    // a buffer.
    void *ws_ple_norm_ = nullptr;  // [chunk, hc*n_embd] lane-normed gated
    void *ws_ple_pad_ = nullptr;   // [hist + chunk, hc*n_embd] conv window
    void *ws_ple_conv_ = nullptr;  // [chunk, hc*n_embd] conv output
    void *ple_state_ = nullptr;    // [n_layer, hist, hc*n_embd] history
    // QSA indexer state. `keys` is the host copy of every RAW indexer key the
    // layer has produced, position-indexed -- pooling precedes norm and
    // rotation, so the raw row is what has to be kept. Pooled keys, scores and
    // the per-row cell lists follow from it each pass; nothing about them is
    // carried across passes, which is what keeps a rolled-back KV honest.
    struct QsaState {
        std::vector<f32> keys;   // [kv_cap, idx_dim]
        std::vector<f32> pooled; // [blocks, idx_dim]
        std::vector<f32> score;  // [chunk, blocks]
    };
    bool qsa_active_ = false;
    i32 qsa_dim_ = 0;    // indexer.key_length
    i32 qsa_heads_ = 0;  // indexer.head_count
    i32 qsa_top_k_ = 0;  // indexer.top_k, in CELLS
    i32 qsa_ratio_max_ = 0;
    f32 qsa_eps_ = 1e-6f;
    std::vector<QsaState> qsa_state_;      // per layer, empty where no indexer
    std::vector<f32> qsa_q_;               // host [chunk * heads * idx_dim]
    std::vector<f32> qsa_k_;               // host [chunk * idx_dim]
    std::vector<std::vector<i64>> qsa_rows_; // this pass's cell list per row
    std::vector<i64> qsa_last_;            // last row's cells (debug hook)
    void *ws_qsa_q_ = nullptr;  // [chunk, heads*idx_dim]
    void *ws_qsa_k_ = nullptr;  // [chunk, idx_dim]
    void *ws_qsa_kc_ = nullptr; // compact K [cell budget, kv_dim]
    void *ws_qsa_vc_ = nullptr; // compact V [cell budget, kv_dim]
    void *ws_q_ = nullptr;    // [chunk, q_dim]
    // [chunk, q_proj] the raw query projection: Qwen3.5 packs a per-head
    // output gate behind the query, so its projection is twice as wide as the
    // query it has to be split into. Null when the model does not pack one.
    void *ws_qpack_ = nullptr;
    // MLA's projection row: latent + decoupled key, NOT the MHA q/k/v shape.
    // Null unless the model runs latent attention; see mla_forward().
    void *ws_mla_ = nullptr;  // [chunk, kv_lora + rope] (the latent on Shared)
    // Shared-latent MLA's query trunk needs the normed `q_a` row to live
    // somewhere while `q_b` reads it, so these two exist only on that layout:
    // the trunk's intermediate [chunk, q_lora] and the grouped output's f32
    // [chunk, gout] accumulator (a per-call allocation would be a device malloc
    // on every layer of every token).
    void *ws_mlaqn_ = nullptr; // [chunk, q_lora]
    void *ws_mlay_ = nullptr;  // f32 [chunk, o_groups * o_lora]
    // The per-head attention stack [chunk, n_head * kv_lora], kept in f32 on
    // EVERY backend: it is the input of the grouped output projection, and a
    // device that rounded it to the activation type would mix a different
    // number than the host oracle the kernels are judged against. `ws_attn_` is
    // act-width and cannot hold it on a HIP build (f16 there, f32 on the CPU).
    void *ws_mlaa_ = nullptr;

    // ---- DeepSeek-V4 compressed attention (CSA/HCA) ----------------------
    // Host f32, one set per COMPRESSED layer: the pool is the oracle's
    // arithmetic (src/dsv4_compress.cpp) and the op it feeds reads these
    // through `mla_shared_attention_compressed`, so they never become device
    // buffers. They exist only on a file whose `attention.compress_ratios`
    // names a non-zero ratio, and the backend must say
    // `mla_shared_compressed_supported()` -- a device that cannot run the two
    // row families would otherwise read the attention table for both.
    //
    // `comp_state_kv_`/`comp_state_sc_` are a two-block ring of per-token state
    // rows (the pool reads the previous block and the current one), kept as two
    // arrays because the projection emits a value and a score per token and the
    // pool consumes them in different halves. `comp_pooled_` holds the pooled
    // keys in block order and `comp_kpos_` each block's FIRST token -- the rope
    // angle, not the block index. `idx_state_*`/`idx_key_` are the CSA
    // indexer's own, which is always the two-wide overlapping compressor and
    // whose pooled key is stored ALREADY ROTATED (the score is a plain dot
    // against the rotated query).
    std::vector<i32> comp_ratio_;               // [layer], 0 = dense
    std::vector<std::vector<f32>> comp_state_kv_;  // [layer][2*ratio*state_width]
    std::vector<std::vector<f32>> comp_state_sc_;  // [layer][2*ratio*state_width]
    std::vector<std::vector<f32>> comp_pooled_;    // [layer][blocks * kv_lora]
    std::vector<std::vector<i64>> comp_kpos_;      // [layer][blocks]
    std::vector<std::vector<f32>> idx_state_kv_;   // [layer][2*ratio*2*idx_head]
    std::vector<std::vector<f32>> idx_state_sc_;   // [layer][2*ratio*2*idx_head]
    std::vector<std::vector<f32>> idx_key_;        // [layer][blocks * idx_head]
    std::vector<i64> comp_nblocks_;                // per layer, blocks so far
    std::vector<f32> comp_inv_freq_;               // [qk_rope/2], compress base
    std::vector<f32> comp_pool_kv_;                // scratch, pool_rows*state_width
    std::vector<f32> comp_pool_sc_;                // scratch, pool_rows*state_width
    std::vector<f32> comp_row_kv_;                 // scratch, state_width
    std::vector<f32> comp_row_sc_;                 // scratch, state_width
    std::vector<f32> comp_idx_kv_;                 // scratch, 2*idx_head
    std::vector<f32> comp_idx_sc_;                 // scratch, 2*idx_head
    std::vector<f32> comp_iq_;                     // scratch, idx_n_head*idx_head
    std::vector<f32> comp_iw_;                     // scratch, idx_n_head
    std::vector<f32> comp_score_;                  // scratch, blocks
    std::vector<i32> comp_sel_;                    // scratch, top-k
    void *ws_k_ = nullptr;    // [chunk, kv_dim]
    void *ws_v_ = nullptr;    // [chunk, kv_dim]
    void *ws_attn_ = nullptr; // [chunk, q_dim]
    // GPT-2's fused attn_qkv projection lands here: [chunk, 3*n_embd], one row
    // of Q|K|V. Distinct from the GDN path's ws_qkv_ below, which is a
    // conv-shaped buffer for a different mixer.
    void *ws_qkv_fused_ = nullptr;
    void *ws_gate_ = nullptr; // [chunk, max(n_ff, n_ff_exp, n_ff_shexp)]
    void *ws_up_ = nullptr;   // [chunk, same]
    // prism.hadamard scratch: the transformed copy of a folded matmul's
    // input, widest folded input wide. One buffer, reused at every site --
    // no two of them are live at the same time. Null unless the file folds.
    void *ws_had_ = nullptr;  // [chunk, widest folded input]
    void *ws_logits_ = nullptr;
    // Gated delta net workspaces (null unless the model is recurrent)
    void *ws_qkv_ = nullptr;      // [chunk, conv_dim] fused q|k|v, post-conv
    void *ws_z_ = nullptr;        // [chunk, value_dim] the z gate
    void *ws_h_ = nullptr;        // [chunk, value_dim] delta-rule output
    void *ws_ssm_ = nullptr;      // [chunk, n_v_head] alpha/gate scalars
    void *ws_beta_ = nullptr;     // [chunk, n_v_head] forget gates
    void *ws_agate_ = nullptr;    // [chunk, q_dim] full-attn output gates
    // MoE-only workspaces (null for dense models)
    void *ws_router_ = nullptr; // [chunk, n_expert]
    void *ws_ffn_ = nullptr;    // [chunk, n_embd] MoE output accumulator
    // One f32 scalar per token: the shared expert's gate. It used to live in
    // `ws_ssm_`, which is allocated only for a RECURRENT stack, so the first
    // non-hybrid family to carry `ffn_gate_inp_shexp` (qwen2moe) dereferenced a
    // null pointer in its first forward.
    void *ws_shexp_g_ = nullptr; // [chunk] shared-expert gate
    // ---- gemma4's two-branch routed block --------------------------------
    // `ws_g4_post_` is the post-attention row the two branches both read (the
    // residual has to survive until the block's own add at the end), `ws_g4_mlp_`
    // is the shared MLP's finished output, and `ws_router_in_` is the router's
    // detached input: a WEIGHTLESS norm of the post-attention row, scaled by
    // 1/sqrt(n_embd) and then elementwise by `ffn_gate_inp.scale`. None of the
    // three exists on a family without `moe_two_branch`.
    void *ws_g4_post_ = nullptr;  // [chunk, n_embd] post-attention row
    void *ws_g4_mlp_ = nullptr;   // [chunk, n_embd] shared MLP branch
    void *ws_router_in_ = nullptr; // [chunk, n_embd] router input
    // Per-call overrides, set only around a two-branch layer: the row the
    // experts read, and "leave the residual alone, the caller will assemble
    // the block". Both are nullptr/false on every other family.
    const void *moe_in_ = nullptr;
    bool moe_no_residual_ = false;

    // ---- gemma4 per-layer input embeddings -------------------------------
    // `ws_ppl_` holds the gathered per-layer token table for these tokens and
    // `ws_ppl_in_` the projection that is added to it; both are [chunk,
    // n_layer*n_embd_per_layer], because the model projection is ONE matrix
    // whose output spans every layer. Only a file that declares the module
    // allocates them.
    void *ws_ppl_ = nullptr;      // [chunk, n_layer*npl] token rows * sqrt(npl)
    void *ws_ppl_in_ = nullptr;   // [chunk, n_layer*npl] the finished table
    void *ws_ppl_g_ = nullptr;    // [chunk, npl] one layer's gated input
    void *ws_ppl_o_ = nullptr;    // [chunk, n_embd] one layer's contribution
    i64 ppl_width_ = 0;           // n_layer * npl, 0 when the module is off
    i64 ppl_npl_ = 0;             // n_embd_per_layer
    // Batched-expert staging: rows are permuted activations and their
    // intermediate buffers, both [chunk, ...]. Chunked prefill bounds the plan
    // size, so chunk_ slots always suffice. The speculative verification pass
    // uses the same buffers with n = draft window + 1.
    void *ws_xg_ = nullptr;     // [chunk, n_embd]   gathered token rows
    void *ws_gateg_ = nullptr;  // [chunk, n_ff_ws]   gathered gate/intermediate
    void *ws_upg_ = nullptr;    // [chunk, n_ff_ws]   gathered up projection
    void *ws_plan_ = nullptr;   // device i32 row ids for the current expert group
    void *ws_alpha_ = nullptr;  // device f32 gate weights for the current group
    void *ws_scale_ = nullptr;  // device f32 row multipliers from a scaled silu
    // The batched-expert path's per-expert buffers: at one decode token every
    // routed expert owns ONE row, so k of them need k rows of each width. Sized
    // for kMoeBatch experts (the launch caps a group at that many); a layer with
    // a wider top-k runs the batch in chunks of that size.
    void *ws_ebatch_ = nullptr;  // [kMoeBatch][2 * n_ff_ws] gate then up, per expert
    void *ws_edown_ = nullptr;   // [kMoeBatch][n_embd]     down projection, per expert
    void *ws_ebscale_ = nullptr; // [kMoeBatch] f32       row scale per expert
    // One i32 per token row: the power of two the residual stream is stored
    // divided by for that row (Backend::add_residual's contract), zeroed at the
    // start of every forward pass so it describes this pass's rows only.
    void *ws_rexp_ = nullptr;
    // The MoE staging buffer's own exponent, per token row: the routed experts'
    // sum is accumulated at true scale and a sum of two in-range rows can still
    // leave f16 when the shared expert is folded in, so that fold renormalizes
    // through the same contract and `ws_ffn_` carries an exponent too.
    void *ws_fexp_ = nullptr;
    // One f32 per token row: the power of two that row's gated FFN intermediate
    // was stored divided by (Backend::silu_mul's row-scaled form). Only the
    // layers with an `ffn_sub_norm` ask for it, and their consumer is that
    // RMSNorm, which is scale-invariant and reads the direction -- so nothing
    // folds it back into the graph; the trace applies it so a dump stays at
    // true scale (Engine::dump_row).
    void *ws_fscl_ = nullptr;
    // ---- LatentMoE staging (null on every family without ffn_latent_*) ----
    // `ws_xlat_` is the block's projection into the experts' narrow space and
    // `ws_ffn_lat_` is the routed sum accumulated there; `ws_xlat_out_` is that
    // sum lifted back to n_embd, added into `ws_ffn_` where every other family
    // already accumulates. Width is the widest latent the file declares.
    // The scaled-input staging for an NVFP4 weight whose sidecar is not 1.0:
    // [chunk, widest scaled contraction]. Null on every file without one.
    void *ws_gs_ = nullptr;
    void *ws_xlat_ = nullptr;     // [chunk, latent]
    void *ws_ffn_lat_ = nullptr;  // [chunk, latent]
    void *ws_xlat_out_ = nullptr; // [chunk, n_embd]
    f32 *logits_host_ = nullptr;
    H2dProbe h2d_probe_;
    H2dProbeTally h2d_tally_;
    std::vector<i32> tok_scratch_;
    // Device top-k. The candidate buffer is sized for a degenerate cut; k is
    // one today (an argmax) but the buffers allow more without reallocating.
    static constexpr i32 kTopK = 1;
    static constexpr i32 kTopKCap = 4096;
    void *topk_scratch_ = nullptr; // 4096 B header + kTopKCap u64 keys
    // One 16-byte device landing zone for the whole op: i32 id | f32 value |
    // i64 collected count. Keeping the three fields adjacent is what turns the
    // readback into a single 16-byte transfer instead of three.
    void *topk_out_ = nullptr;
    alignas(8) char topk_out_h_[16] = {};
    i64 topk_n_ = 0;    // the collected count of the last successful call
    i32 topk_id_ = 0;   // its argmax, valid only while topk_valid_
    bool topk_valid_ = false;
    // Host-side router scratch, sized [chunk, n_expert]
    std::vector<f32> router_host_;
    // Device-side router probabilities ([chunk, n_expert] f32), written by
    // Backend::router_softmax. Allocated for every backend and used only when
    // KRK_DEVICE_SOFTMAX=1, so "is the device path available" stays a runtime
    // question rather than a different allocation path with a null to forget.
    void *ws_routerp_ = nullptr;
    std::vector<f32> dev_probs_; // its host copy, for the drift comparison
    u64 ds_rows_ = 0;            // rows compared
    u64 ds_diff_ = 0;            // elements whose f32 bits differ
    f64 ds_max_ = 0.0;           // largest |device - host| probability
    // The device selection and plan (KRK_DEV_PLAN=1). `ws_ids_` is [chunk, k]
    // i32 and `ws_wts_` the same shape in f32; `ws_off_` is the plan's
    // [n_expert + 1] prefix. The plan itself lands in ws_plan_/ws_alpha_, which
    // already exist for the host plan, so the per-expert loop needs no change.
    void *ws_ids_ = nullptr;
    void *ws_wts_ = nullptr;
    void *ws_off_ = nullptr;
    std::vector<i32> dev_ids_;
    std::vector<f32> dev_wts_;
    u64 dp_rows_ = 0;            // layers whose device plan was compared
    u64 dp_sel_diff_ = 0;        // (slot) ids that differ from the host
    u64 dp_wt_diff_ = 0;         // (slot) weights that differ from the host
    u64 dp_wt_1ulp_ = 0;         // (slot) weights one ulp from the host
    u64 dp_used_ = 0;            // layers that ran the device plan
    // The published device dispatch table (KRK_DEV_PTR=1): the cache's own
    // pointer, so a promotion that invalidated it is noticed by comparing the
    // two rather than by trusting a flag that was true at publish time.
    void *dev_table_ = nullptr;
    // How many times the publish was retried while the router filled the tiers,
    // and whether its "not yet" reason has been printed (once is enough).
    static constexpr i32 kTableAttempts = 8;

    // How many experts the batched decode path puts in one launch group. It is
    // 8 because that is what the backend's group matrix array holds; the
    // gate/up half of a group is two matrices per expert, so one gate/up launch
    // covers four experts at a time.
    static constexpr i32 kMoeBatch = 8;
    i32 table_attempts_ = 0;
    bool table_why_logged_ = false;
    u64 dp_launch_layers_ = 0;   // layers that ran the one-launch dispatch
    u64 dp_launch_skips_ = 0;    // layers the dispatch could not serve (shape/residency)
    // Chunks a host-backed table deliberately does not serve: the table kernel
    // computes one row at a time, so a prefill chunk keeps the per-expert path.
    // Counted apart from the fallbacks, because a designed exclusion reported as
    // a fallback hides a real one.
    u64 dp_chunk_skips_ = 0;
    u64 moe_batch_layers_ = 0;   // decode layers the batched path computed
    u64 moe_batch_skips_ = 0;    // decode layers that kept the per-expert loop
    // Scratch for the layer-wide expert prefetch: the routed ids, deduped.
    std::vector<i32> prefetch_ids_;
    std::vector<u8> prefetch_seen_;
    // The ids handed to the cache's read-ahead worker (the next MoE layer's
    // prediction, filtered to what is not in WARM yet). Reused per layer so the
    // issue site allocates nothing.
    std::vector<i32> ahead_ids_;
    // Previous-token route sets retained for the legacy
    // KRK_MOE_NEXT_PREFETCH experiment. The adaptive predictor has independent
    // per-layer histories and only proposes cache staging hints.
    std::vector<std::vector<i32>> moe_hint_;
    // Architecture-gated online cache predictor. Candidate IDs affect only
    // stage_ahead(); moe_sel_/moe_wt_ continue to come from the native router.
    MoeRoutePredictor moe_predictor_;
    std::vector<i32> moe_predict_ids_;
    bool moe_predict_enabled_ = false;
    // KRK_DUMP destination (see Engine::init): empty means the per-stage dump
    // hook in forward_core is inert.
    std::string dump_path_;
    // -1 = the architecture's own delta-net gate (silu on qwen3.5, sigmoid on
    // qwen4exp), 0 = force silu, 1 = force sigmoid (KRK_GDN_GATE).
    i32 gdn_gate_ = -1;
    // KRK_DUMP_ROUTE=1: print each token's top-k routing decision.
    bool dump_route_ = false;
    // Residency preference in the router (see EngineConfig::router_tie_margin).
    // `router_pref_` is false unless a margin was asked for, and the selection
    // path is then the same arithmetic it has always been -- the switch is one
    // process apart, which is the only way its cost is measurable.
    bool router_pref_ = false;
    f32 router_tie_margin_ = 0.0f;
    u64 router_picks_ = 0; // (token, slot) picks since the last reset
    u64 router_subs_ = 0;  // picks where a resident expert beat the top scorer
    // Picks that landed on an expert already in VRAM, counted in BOTH arms: it is
    // the number the preference exists to raise, and a counter that only runs
    // when the switch is on cannot show a control.
    u64 router_hot_ = 0;
    bool trace_hash_on_ = false;
    std::map<std::string, u64> trace_stage_; // key -> running hash
    std::vector<f32> moe_prob_;
    // Scratch for the routing logits when the file's bias belongs BEFORE the
    // softmax (gpt-oss): the router row is downloaded as it came out of the
    // GEMM and this holds the biased copy the top-k and the softmax both read,
    // so the two never disagree about which experts were picked.
    std::vector<f32> moe_logit_;
    // The routing plan for the current chunk: which tokens each expert got.
    std::vector<i32> moe_sel_;    // [chunk, k] expert id per (token, slot)
    std::vector<f32> moe_wt_;     // [chunk, k] renormalized gate weight
    std::vector<i32> group_rows_; // [<=chunk] token ids in the current group
    std::vector<f32> group_wt_;   // [<=chunk] their gate weights

    // ---- hybrid CPU + GPU expert compute (EngineConfig::hybrid_experts) --
    //
    // All host-side. hy_xn_ is the activation the host arm reads (downloaded
    // from ws_xn_ exactly once per layer), hy_out_ is what CpuExpertPool
    // accumulates into, and hy_ffn_/hy_f16_ are the merge scratch: the device's
    // ws_ffn_ comes down, the host contribution is added, and the result goes
    // back up as f16. No device buffer is added, so turning this on cannot cost
    // VRAM -- which is the whole point, since the VRAM it frees has to go to the
    // KV cache.
    //
    // hy_job_rows_/hy_job_wt_ are the flattened per-expert row lists a Job
    // points into. They are RESERVED to n * k before anything is written into
    // them: every (token, slot) pair contributes at most one row per expert, so
    // that reserve is an upper bound and no push_back can reallocate a buffer a
    // Job already holds a pointer to.
    bool hybrid_on_ = false;
    i32 hybrid_threads_ = 0;
    i32 hybrid_max_rows_ = 8;
    i32 hybrid_frac_permille_ = 1000;
    i32 hybrid_threads_live_ = 0;
    u64 hy_cpu_experts_ = 0;
    u64 hy_gpu_experts_ = 0;
    u64 hy_policy_violations_ = 0;
    u64 hy_cpu_rows_ = 0;
    f64 hy_cpu_ms_ = 0.0;
    std::vector<u8> hy_take_;                 // [n_expert] 1 => host computes it
    std::vector<CpuExpertPool::Job> hy_jobs_; // the host arm's work list
    std::vector<i32> hy_job_rows_;
    std::vector<f32> hy_job_wt_;
    std::vector<f32> hy_xn_;
    std::vector<f32> hy_out_;
    std::vector<f32> hy_ffn_;
    std::vector<u16> hy_f16_;

    // ---- expert routing scan (--expert-scan) ----------------------------
    //
    // A hot-expert list cannot be built from the model file: which experts a
    // token routes to is a function of the residual stream, so it has to be
    // measured. These accumulate what the routers actually chose, weighted by
    // the routing probability rather than by a count -- a count cannot tell you
    // whether the top 8 of 128 carry 90% of the traffic or 40%, and that
    // number is what sizes a residency budget.
    //
    // `expert_stub_` runs the routers with every expert FFN replaced by
    // nothing, which is the cheap approximation: the routers only need the
    // dense path, so the whole expert tensor set stays unmapped. It is only
    // valid if stubbing does not change what the routers choose, which is an
    // assumption with nothing behind it until it is measured -- see
    // tools/expert_scan_diff.py.
    std::vector<double> expert_mass_; // [n_layer * n_expert] summed P(expert)
    std::vector<u64> expert_hits_;     // [n_layer * n_expert] times selected
    // The same two histograms, ON THE DEVICE, where the routing already is.
    // Non-null only when the backend can accumulate (Backend::scan_accum) and a
    // scan or an adapt run asked for them; when they are, the host plan stops
    // counting and the numbers arrive once, after the run, from
    // scan_dev_fold(). That is what stops a tracking run from paying a router
    // readback per layer for eight integers.
    void *scan_hits_dev_ = nullptr;
    void *scan_mass_dev_ = nullptr;
    i64 expert_scan_tokens_ = 0;       // token positions folded in so far
    bool expert_scan_ = false;         // accumulate
    bool expert_stub_ = false;         // skip the expert GEMMs while scanning
    bool expert_track_ = false;        // count routing on a normal run (adapt)
    bool expert_stubbed_ran_ = false;  // stub was requested and did run

    std::vector<i32> plan_dev_;   // staging mirror for ws_plan_
    std::vector<f32> alpha_dev_;  // staging mirror for ws_alpha_
    // The layer plan's segmentation: off(e) is expert e's first slot in
    // plan_dev_/alpha_dev_, and plan_cur_ is the bucket cursor the fill pass
    // walks. Both are members so a layer does not allocate.
    std::vector<i32> plan_off_;
    std::vector<i32> plan_cur_;
    // The batched decode path's own staging: the layer's routed experts in
    // ascending order and the resident pointers their launches read. Both are
    // resolved before the first launch, so a refusal can leave the layer to the
    // per-expert loop without a contribution counted twice.
    std::vector<i32> batch_ids_;          // this layer's routed experts, ascending
    std::vector<const ResidentExpert *> batch_res_;
};

} // namespace krk

#endif // KRK_ENGINE_HPP
