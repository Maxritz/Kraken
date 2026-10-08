// main_cli.cpp — the kraken command line front end.
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "engine_usage.hpp"
#include "krk/backend.hpp"
#include "krk/engine.hpp"
#include "krk/host_time.hpp"

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace {

using namespace krk;

struct Args {
    std::string model;
    std::string prompt;
    std::string system;
    std::string draft;
    int max_tokens = 256;
    int ctx = 4096;
    int chunk = 256;
    int device = 0;
    int threads = 0;
    int expert_cache_mb = 0;    // MoE expert residency budget (MiB), 0 = auto
    int vram_cap_mb = 0;        // total device memory to plan for (MiB), 0 = policy
    int vram_tier_gb = 0;       // plan for a card of this class (GiB), 0 = off
    int host_ram_mb = 0;        // host memory to plan for (MiB), 0 = policy
    int expert_cache_slots = 0; // MoE resident (layer, expert) slot cap, 0 = auto
    int expert_warm_mb = -1;     // MoE WARM tier in pageable host RAM (MiB): <0 auto, 0 off
    int kv_hot_mb = 0;           // KV HOT (VRAM) budget, MiB: 0 auto
    int kv_warm_mb = -1;         // KV WARM (host RAM) budget, MiB: <0 auto, 0 off
    std::string kv_cold_dir;     // KV COLD spill directory: empty disables
    bool expert_warm_prefetch = false; // fill WARM at load instead of on demand
    int expert_warmup = -1;      // ranked WARM+VRAM warmup at load: <0 auto, 0 off
    int expert_warmup_ms = 20000; // wall-time ceiling for that phase, 0 = none
    int ram_tier_gb = 0;         // plan the host tier for this RAM class (GiB)
    // Hybrid CPU + GPU expert compute: experts that are not VRAM-resident are
    // computed on the host out of the mapping while the device runs the
    // resident ones (see EngineConfig::hybrid_experts).
    bool hybrid_experts = false;
    int hybrid_threads = 0;     // host worker threads, 0 = min(hardware, 8)
    int hybrid_max_rows = 8;    // rows the host arm runs at, 0 = unbounded
    double hybrid_frac = 1.0;   // share of RESIDENT experts forced to the host
    int draft_tokens = 4;       // speculative decoding window
    bool cpu = false;
    bool greedy = false;
    bool tokenize = false;
    bool chat = false;
    bool info = false;
    bool bench = false;
    bool profile = false;
    bool expert_scan = false;   // measure routing and write an expert index
    bool expert_stub = false;   // run the routers with every expert FFN removed
    int debug_topk = 0; // stderr dump of top-k logprobs per token
    // argmax on the host from the downloaded row, not on the device
    bool sample_host = false;
    bool verbose = false;
    bool interactive = true;
    // Whether --prompt/-p appeared at all. It has to be tracked separately
    // from prompt.empty(): an empty string is a legitimate prompt (generate
    // from an empty context), but conflating the two made `--prompt ""` fall
    // through to the interactive loop and block on stdin, so a script that
    // interpolated an empty file hung instead of running.
    bool prompt_given = false;
    f32 temp = 0.8f;
    int top_k = 40;
    f32 top_p = 0.95f;
    f32 min_p = 0.05f;
    f32 rep_pen = 1.1f;
    u64 seed = 0;
};

void usage() {
    std::printf(
        "%s %s — RDNA-native GGUF inference\n\n"
        "usage: kraken --model model.gguf [options]\n\n"
        "  --prompt STR          prompt to complete; an empty value still counts\n"
        "                        as given (omit the flag for interactive)\n"
        "  --max-tokens N        tokens to generate (default 256)\n"
        "  --ctx N               KV capacity in tokens (default 4096)\n"
        "  --chunk N             prefill batch size (default 256)\n"
        "  --temp F              sampling temperature, 0 = greedy (default 0.8)\n"
        "  --top-k N             top-k cutoff (default 40)\n"
        "  --top-p F             nucleus cutoff (default 0.95)\n"
        "  --min-p F             min-p cutoff (default 0.05)\n"
        "  --repeat-penalty F    repetition penalty (default 1.1)\n"
        "  --seed N              RNG seed\n"
        "  --greedy              force argmax decoding\n"
        "  --tokenize            print the prompt's token ids and exit\n"
        "  --chat                wrap the prompt in a ChatML template\n"
        "  --system STR          system message for --chat\n"
        "  --cpu                 use the scalar reference backend\n"
        "  --device N            HIP device index (default 0)\n"
        "  --expert-cache-mb N   MoE expert residency budget in MiB (0 = auto)\n"
        "  --host-ram-mb N       host memory to plan for, in MiB. 0 uses a\n"
        "                        quarter of installed RAM. Bounds every host\n"
        "                        tier inside the process; the model mapping and\n"
        "                        the dense trunk sit outside it.\n"
        "  --kv-hot-mb N          KV HOT tier in VRAM, in MiB. 0 (default) takes\n"
        "                        whatever VRAM the weights and workspaces leave\n"
        "                        free. The KV is split into layer-granular slots;\n"
        "                        if the whole cache fits, this is inert and the\n"
        "                        cache is one flat allocation as before. Lower it\n"
        "                        to make a long context run by paging the\n"
        "                        overflow through WARM instead of refusing it.\n"
        "  --kv-warm-mb N         KV WARM tier in pageable host RAM, in MiB.\n"
        "                        <0 (default) sizes it from the geometry: with no\n"
        "                        --kv-cold-dir it covers every KV-carrying layer,\n"
        "                        because a page that fits nowhere is DROPPED, its\n"
        "                        layer is zero-filled, and the run decodes WRONG\n"
        "                        text. A smaller N is therefore obeyed only when a\n"
        "                        spill directory exists; otherwise WARM is raised to\n"
        "                        cover the cache and the deviation is reported.\n"
        "                        Buffers are allocated lazily, so a big WARM costs\n"
        "                        nothing until an eviction lands in it.\n"
        "  --kv-cold-dir DIR      KV COLD tier: spill directory for pages evicted\n"
        "                        past WARM. Unset (default) keeps them in RAM and\n"
        "                        recomputes instead of writing; a directory that\n"
        "                        cannot be written is ignored and reported, since a\n"
        "                        failed spill is a dropped page.\n"
        "  --vram-tier N         plan for a card of this class, in GiB (8, 12,\n"
        "                        16, 24 ...). Clamped to the card that is actually\n"
        "                        installed, so `--vram-tier 12` on a 16 GiB box\n"
        "                        behaves as the 12 GiB one does. The class form of\n"
        "                        --vram-cap-mb, which wins if both are given.\n"
        "  --vram-cap-mb N       total device memory to plan for, in MiB. 0 uses\n"
        "                        the policy: 6 GiB, then +2 GiB at a time while\n"
        "                        the card has room and the model needs it, up to\n"
        "                        14 GiB. Anything the run does not plan for -- a\n"
        "                        long KV context, another application -- is what\n"
        "                        the headroom is for.\n"
        "  --expert-cache-slots N  cap resident (layer, expert) slots (0 = auto)\n"
        "  --expert-warm-mb N     WARM expert cache in pageable host RAM. Cold\n"
        "                        reads land here before VRAM, VRAM evictions stay\n"
        "                        here, and WARM eviction never writes to the file\n"
        "                        (MiB; <0 auto = the expert set, capped by half of\n"
        "                        free RAM after a reserve; 0 off).\n"
        "                        --expert-l2-mb is the old spelling of this flag.\n"
        "  --expert-warm-prefetch  fill WARM at load instead of on demand (costs\n"
        "                        RAM and time before the first token)\n"
        "  --expert-warmup N      ranked expert warmup at load: -1 (default) runs\n"
        "                        it when the routed expert set exceeds VRAM, 1\n"
        "                        forces it, 0 disables it. Stages the hot experts\n"
        "                        into WARM and promotes the top of that order into\n"
        "                        VRAM before the first token, so a tier smaller\n"
        "                        than the routed set holds what routing wants.\n"
        "                        Needs <model>.krakenexperts.json to rank by;\n"
        "                        without it the tiers fill in an even order.\n"
        "  --expert-warmup-ms N   wall-time ceiling for that phase, in ms (0 =\n"
        "                        none). Default 20000; the hottest experts go in\n"
        "                        first, so running out costs coverage, not the top.\n"
        "  --ram-tier N           plan the host RAM tiers for this class of\n"
        "                        machine, in GiB: 16, 24, 32, 48, 64 or 96. 0 (the\n"
        "                        default) is the installed RAM. Clamped to what is\n"
        "                        actually installed, so a 32 GiB tier is the same\n"
        "                        run on a 32 GiB box and a smaller one here.\n"
        "  --hybrid-experts 1    compute the experts the device tier cannot hold\n"
        "                        on the host, out of the weight mapping, while the\n"
        "                        device runs the resident ones. The experts it\n"
        "                        takes cost no VRAM and no WARM RAM, so it frees\n"
        "                        memory for the KV cache. Inert when the budget\n"
        "                        already holds the routed set.\n"
        "                        MEASURED, Qwen3-MoE-4x0.6B Q4_K_M, 16 MiB expert\n"
        "                        budget: this LOSES -- 5.5 tok/s against 17.2 for\n"
        "                        promoting instead, at 5.4 ms of host time per\n"
        "                        expert against ~2.0 ms of read+promote. The host\n"
        "                        arm is a per-layer barrier the device waits on, so\n"
        "                        the split only pays when the file is slower than\n"
        "                        the CPU. See docs/test-results.md section 9.\n"
        "  --hybrid-threads N    host threads for that split (0 = min(hw, 8))\n"
        "  --hybrid-max-rows N   widest row count the host arm runs at (default 8;\n"
        "                        0 = unbounded. It is a decode optimization.)\n"
        "  --hybrid-frac F       share (0..1) of the RESIDENT experts forced to the\n"
        "                        host anyway; the knob that sweeps the split\n"
        "  --draft MODEL         draft model for greedy speculative decoding\n"
        "  --draft-tokens N      speculation window (default 4)\n"
        "  --info                print model and device info, then exit\n"
        "  --bench               prefill/decode benchmark on a fixed prompt\n"
        "  --profile             per-op device timeline for the last decode\n"
        "                        step(s), with the idle gap before each op\n"
        "                        (KRK_PROFILE_STEPS/FROM/OPS tune the window)\n"
        "  --debug-topk N        dump the top-N candidates and their log-probs\n"
        "  --sample-host         take the argmax on the host from the downloaded\n"
        "                        logits row instead of on the device. Costs about a\n"
        "                        1 MB download per token; the device argmax path is\n"
        "                        the one carrying the laguna race, so this is the\n"
        "                        correct path until that is fixed\n"
        "                        for every token to stderr (at full precision)\n"
        "  --expert-scan        measure which experts the routers actually pick\n"
        "                        and write <model>.krakenexperts.json\n"
        "  --expert-stub        requires --expert-scan: run only the routers,\n"
        "                        with every expert FFN skipped (no expert\n"
        "                        weights are read). The list is then\n"
        "                        approximate; the two modes exist so they can\n"
        "                        be compared. On its own the flag does\n"
        "                        nothing, so it is refused rather than ignored\n"
        "  -v                    verbose logging\n"
        "  -h, --help            this message\n",
        kEngineName, kEngineVersion);
}

bool parse(int argc, char **argv, Args *a) {
    for (int i = 1; i < argc; i++) {
        const std::string f = argv[i];
        auto next = [&](const char *what) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "kraken: %s requires a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (f == "--model" || f == "-m") a->model = next("--model");
        else if (f == "--prompt" || f == "-p") {
            a->prompt = next("--prompt");
            a->prompt_given = true;
        }
        else if (f == "--system") a->system = next("--system");
        else if (f == "--draft" || f == "-d") a->draft = next("--draft");
        else if (f == "--draft-tokens")
            a->draft_tokens = std::atoi(next("--draft-tokens"));
        else if (f == "--max-tokens" || f == "-n") a->max_tokens = std::atoi(next("--max-tokens"));
        else if (f == "--ctx" || f == "-c") a->ctx = std::atoi(next("--ctx"));
        else if (f == "--chunk") a->chunk = std::atoi(next("--chunk"));
        else if (f == "--temp") a->temp = static_cast<f32>(std::atof(next("--temp")));
        else if (f == "--top-k") a->top_k = std::atoi(next("--top-k"));
        else if (f == "--top-p") a->top_p = static_cast<f32>(std::atof(next("--top-p")));
        else if (f == "--min-p") a->min_p = static_cast<f32>(std::atof(next("--min-p")));
        else if (f == "--repeat-penalty") a->rep_pen = static_cast<f32>(std::atof(next("--repeat-penalty")));
        else if (f == "--seed") a->seed = std::strtoull(next("--seed"), nullptr, 10);
        else if (f == "--device") a->device = std::atoi(next("--device"));
        else if (f == "--threads") a->threads = std::atoi(next("--threads"));
        else if (f == "--host-ram-mb")
            a->host_ram_mb = std::atoi(next("--host-ram-mb"));
        else if (f == "--kv-hot-mb")
            a->kv_hot_mb = std::atoi(next("--kv-hot-mb"));
        else if (f == "--kv-warm-mb")
            a->kv_warm_mb = std::atoi(next("--kv-warm-mb"));
        else if (f == "--kv-cold-dir")
            a->kv_cold_dir = next("--kv-cold-dir");
        else if (f == "--vram-cap-mb")
            a->vram_cap_mb = std::atoi(next("--vram-cap-mb"));
        else if (f == "--vram-tier" || f == "--vram-tier-gb")
            a->vram_tier_gb = std::atoi(next("--vram-tier"));
        else if (f == "--expert-cache-mb")
            a->expert_cache_mb = std::atoi(next("--expert-cache-mb"));
        else if (f == "--expert-cache-slots")
            a->expert_cache_slots = std::atoi(next("--expert-cache-slots"));
        else if (f == "--expert-warm-mb" || f == "--expert-l2-mb")
            a->expert_warm_mb = std::atoi(next("--expert-warm-mb"));
        else if (f == "--expert-warm-prefetch")
            a->expert_warm_prefetch = std::atoi(next("--expert-warm-prefetch")) != 0;
        else if (f == "--expert-warmup")
            a->expert_warmup = std::atoi(next("--expert-warmup"));
        else if (f == "--expert-warmup-ms")
            a->expert_warmup_ms = std::atoi(next("--expert-warmup-ms"));
        else if (f == "--ram-tier" || f == "--ram-tier-gb")
            a->ram_tier_gb = std::atoi(next("--ram-tier"));
        else if (f == "--hybrid-experts")
            a->hybrid_experts = std::atoi(next("--hybrid-experts")) != 0;
        else if (f == "--hybrid-threads")
            a->hybrid_threads = std::atoi(next("--hybrid-threads"));
        else if (f == "--hybrid-max-rows")
            a->hybrid_max_rows = std::atoi(next("--hybrid-max-rows"));
        else if (f == "--hybrid-frac")
            a->hybrid_frac = std::atof(next("--hybrid-frac"));
        else if (f == "--greedy") a->greedy = true;
        else if (f == "--tokenize") a->tokenize = true;
        else if (f == "--chat") a->chat = true;
        else if (f == "--cpu") a->cpu = true;
        else if (f == "--info") a->info = true;
        else if (f == "--bench") a->bench = true;
        else if (f == "--profile") a->profile = true;
        else if (f == "--expert-scan") a->expert_scan = true;
        else if (f == "--expert-stub") a->expert_stub = true;
        else if (f == "--debug-topk")
            a->debug_topk = std::atoi(next("--debug-topk"));
        else if (f == "--sample-host")
            a->sample_host = true;
        else if (f == "-v" || f == "--verbose") a->verbose = true;
        else if (f == "-h" || f == "--help") { usage(); std::exit(0); }
        else {
            std::fprintf(stderr, "kraken: unknown option '%s'\n", f.c_str());
            return false;
        }
    }
    if (a->model.empty()) {
        std::fprintf(stderr, "kraken: --model is required\n");
        return false;
    }
    return true;
}

std::string chatml(const std::string &system, const std::string &user) {
    std::string s;
    if (!system.empty()) s += "<|im_start|>system\n" + system + "<|im_end|>\n";
    s += "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    return s;
}

struct SinkCtx {
    bool chat = false;
    std::string hold;
};

// Streams generated text, hiding the ChatML end marker from the transcript.
void sink_cb(void *user, const char *text, i32 token, bool done) {
    SinkCtx *c = static_cast<SinkCtx *>(user);
    (void)token;
    if (done) {
        std::fflush(stdout);
        return;
    }
    if (!text) return;
    if (!c->chat) {
        std::fputs(text, stdout);
        std::fflush(stdout);
        return;
    }
    c->hold += text;
    const std::string marker = "<|im_end|>";
    size_t pos;
    while ((pos = c->hold.find(marker)) != std::string::npos) {
        std::fputs(c->hold.substr(0, pos).c_str(), stdout);
        c->hold.erase(0, pos + marker.size());
    }
    if (c->hold.size() > marker.size()) {
        const size_t keep = marker.size() - 1;
        const size_t emit = c->hold.size() - keep;
        std::fputs(c->hold.substr(0, emit).c_str(), stdout);
        c->hold.erase(0, emit);
    }
    std::fflush(stdout);
}

// One block per run, always on: what the load cost, what the run cost, and what
// the expert cache did. Every number here already existed as a counter — a
// paging policy nobody can read back is not a measurement. Goes to stderr so
// stdout stays exactly the generated text (the coherence check depends on it).
static void print_run_stats(FILE *out, Engine &engine, const GenerateResult &r,
                            f64 load_ms, size_t dev_free, size_t dev_total) {
    const Model &m = engine.model();
    const ModelConfig &mc = m.cfg();
    const f64 ptps = r.prefill_ms > 0 ? r.prompt_tokens / (r.prefill_ms / 1000.0) : 0.0;
    // Over decode_steps, not over r.generated. The first generated token comes
    // out of the prefill pass, so the decode timer covers one step fewer than
    // the tokens emitted; dividing by `generated` understated the per-token
    // cost, and at --max-tokens 1 it printed 2304 tok/s, which no machine can
    // do -- one decode step still has to read every weight from VRAM.
    const f64 dtps = r.decode_steps > 0
                         ? r.decode_steps / (r.decode_ms / 1000.0)
                         : 0.0;
    const f64 dper = r.decode_steps > 0 ? r.decode_ms / r.decode_steps : 0.0;
    std::fprintf(out, "\n[stats ] load      %.0f ms", load_ms);
    if (m.upload_bytes() > 0) {
        const f64 gb = static_cast<f64>(m.upload_bytes()) / 1073741824.0;
        const f64 gbs = m.upload_ms() > 0 ? gb / (m.upload_ms() / 1000.0) : 0.0;
        std::fprintf(out, ", %.2f GiB uploaded of %.2f GiB weights in %.0f ms = %.1f GB/s",
                     gb, static_cast<f64>(m.weight_bytes()) / 1073741824.0,
                     m.upload_ms(), gbs);
    }
    std::fprintf(out, "\n[stats ] prefill   %d tok in %.1f ms = %.1f tok/s\n",
                 r.prompt_tokens, r.prefill_ms, ptps);
    // "n tok in T ms" counts every emitted token; the rate is over the steps
    // that were actually timed, and the step count is printed so the two can
    // never be confused again.
    // A speculative run's unit of work is a round, not a step, and one round
    // emits several tokens. Reporting rounds/decode_ms as "tok/s" understated
    // the rate by the accept rate; reporting tokens/decode_ms against a step
    // count that was never incremented printed 0.0 tok/s. The two are printed
    // as what they are.
    const bool spec_run = engine.has_draft() && engine.spec_steps() > 0;
    if (spec_run) {
        const f64 spec_tps = r.decode_ms > 0 ? 1000.0 * r.generated / r.decode_ms : 0.0;
        const f64 ms_round = r.decode_steps > 0
                                 ? r.decode_ms / static_cast<f64>(r.decode_steps)
                                 : 0.0;
        std::fprintf(out,
                     "[stats ] decode    %d tok in %.1f ms over %lld speculation "
                     "rounds = %.1f tok/s (%.2f ms/round)\n",
                     r.generated, r.decode_ms,
                     static_cast<long long>(r.decode_steps), spec_tps, ms_round);
    } else {
        std::fprintf(out,
                     "[stats ] decode    %d tok in %.1f ms over %lld timed steps "
                     "= %.1f tok/s (%.1f ms/step)\n",
                     r.generated, r.decode_ms,
                     static_cast<long long>(r.decode_steps), dtps, dper);
    }
    if (dev_total > 0)
        std::fprintf(out,
                     "[stats ] device    %.2f GiB in use of %.2f GiB (%.2f GiB free)\n",
                     static_cast<f64>(dev_total - dev_free) / 1073741824.0,
                     static_cast<f64>(dev_total) / 1073741824.0,
                     static_cast<f64>(dev_free) / 1073741824.0);

    // KV tier traffic. These counters live in KvTierCache and were collected
    // from the start, but the only thing that ever read them was the load-time
    // one-liner, so a run that paged the whole cache through WARM and out to
    // COLD reported nothing. It is printed for every model -- a dense model
    // tiers too -- and printed before the MoE-only block below, which used to
    // return first and hide it for everything that is not an MoE.
    {
        KvTierStats ks;
        engine.kv_tier_stats(&ks);
        if (ks.tiered) {
            const f64 mig_mib = static_cast<f64>(ks.migrate_bytes) / 1048576.0;
            const f64 layer_mib = static_cast<f64>(ks.layer_bytes) / 1048576.0;
            std::fprintf(out,
                         "[stats ] kv        tiered: %lld KV layer(s) x %.2f MiB per "
                         "plane, %lld HOT slots (%.2f MiB VRAM) | promoted WARM->HOT %lld, "
                         "COLD->HOT %lld | evicted HOT->WARM %lld, ->COLD %lld | "
                         "%.1f MiB migrated\n",
                         static_cast<long long>(ks.layers), layer_mib,
                         static_cast<long long>(ks.slots),
                         static_cast<f64>(ks.hot_bytes) / 1048576.0,
                         static_cast<long long>(ks.promotions_from_warm),
                         static_cast<long long>(ks.promotions_from_cold),
                         static_cast<long long>(ks.evictions_to_warm),
                         static_cast<long long>(ks.evictions_to_cold), mig_mib);
            // A dropped page is not a spilled one. `->COLD` above is a page on
            // disk that comes back; this is a page that went nowhere, so its
            // layer is zero-filled and the TEXT OF THIS RUN IS NOT
            // TRUSTWORTHY. Printed last and in full sentences on purpose: the
            // counter that used to absorb these said "->COLD", so a run that
            // lost history every step read as a working disk tier.
            if (ks.evictions_dropped > 0) {
                std::fprintf(out,
                             "[stats ] kv        DROPPED %lld page(s) with nowhere "
                             "to put them: their KV is gone and their layers "
                             "were zero-filled. THIS RUN'S OUTPUT IS WRONG -- "
                             "raise --kv-warm-mb or set --kv-cold-dir.\n",
                             static_cast<long long>(ks.evictions_dropped));
            }
        } else {
            std::fprintf(out,
                         "[stats ] kv        flat: whole cache resident in VRAM "
                         "(no tiering traffic)\n");
        }
    }

    // Host-phase table goes before the MoE-only block so a dense model gets it
    // too: --profile used to print the device op table and then nothing about
    // the host, and the host is where a launch-bound step's time actually goes.
    if (!mc.is_moe) {
        krk::HostTime::get().report(out);
        return;
    }
    const ExpertCache &ec = m.experts();
    const u64 acq = ec.acquires();
    const f64 token_div = r.generated > 0 ? static_cast<f64>(r.generated) : 1.0;
    std::fprintf(out,
                 "[stats ] experts   %.0f MiB VRAM budget, %zu/%zu slots resident, "
                 "%zu pinned | WARM %.0f MiB pageable, %zu held (%.0f MiB, "
                 "%.0f MiB prefetched)\n",
                 static_cast<f64>(ec.budget_bytes()) / 1048576.0, ec.resident_slots(),
                 ec.capacity_slots(), ec.pinned_slots(),
                 static_cast<f64>(ec.warm_capacity_bytes()) / 1048576.0, ec.warm_slots(),
                 static_cast<f64>(ec.warm_used_bytes()) / 1048576.0,
                 static_cast<f64>(ec.bytes_staged()) / 1048576.0);
    // The three tiers reported apart. A single "hit rate" reads as "the
    // cache works 20% of the time" when VRAM 20% / RAM 80% / file 0% is
    // exactly the policy working: the tiers answer different questions and
    // only the last one is the storage path.
    const f64 share = acq > 0 ? 100.0 / static_cast<f64>(acq) : 0.0;
    std::fprintf(out,
                 "[stats ] cache     %llu acquires | HOT hits %llu (%.1f%%) | "
                 "WARM hits %llu (%.1f%%) | COLD misses %llu (%.1f%%)\n",
                 static_cast<unsigned long long>(acq),
                 static_cast<unsigned long long>(ec.hits()), share * static_cast<f64>(ec.hits()),
                 static_cast<unsigned long long>(ec.host_hits()),
                 share * static_cast<f64>(ec.host_hits()),
                 static_cast<unsigned long long>(ec.loads()),
                 share * static_cast<f64>(ec.loads()));
    const f64 prom_ms = ec.promote_ms();
    const f64 prom_mib = static_cast<f64>(ec.bytes_promoted()) / 1048576.0;
    std::fprintf(out,
                 "[stats ] transfer  %llu promotions, %.0f MiB in %.1f ms = %.1f GB/s "
                 "(%.1f us/expert) | %llu VRAM evictions kept in WARM, %.0f MiB | "
                 "%.0f MiB read from the file (%.2f MiB/tok)\n",
                 static_cast<unsigned long long>(ec.promotions()), prom_mib, prom_ms,
                 prom_ms > 0 ? prom_mib / 1024.0 / (prom_ms / 1000.0) : 0.0,
                 ec.promotions() > 0
                     ? prom_ms * 1000.0 / static_cast<f64>(ec.promotions())
                     : 0.0,
                 static_cast<unsigned long long>(ec.demotions()),
                 static_cast<f64>(ec.bytes_demoted()) / 1048576.0,
                 static_cast<f64>(ec.bytes_loaded()) / 1048576.0,
                 static_cast<f64>(ec.bytes_loaded()) / 1048576.0 / token_div);
    // The expert path split. promote_ms() alone cannot say whether a token is
    // slow because of the file, the allocator or the copy, and those three have
    // three different fixes, so all four are timed and printed together.
    std::fprintf(out,
                 "[stats ] expert-path  read %8.1f ms | room %8.1f ms | "
                 "alloc %8.1f ms | xfer %8.1f ms | (promote total %8.1f ms)\n",
                 ec.read_ms(), ec.room_ms(), ec.alloc_ms(), ec.xfer_ms(),
                 prom_ms);
    // The hybrid split, when it was on. Printed even when the host arm ran
    // nothing, because "the split bought nothing here" is a result too, and the
    // only way to see it is for the line to appear anyway.
    if (engine.hybrid_enabled()) {
        HybridStats hs;
        engine.hybrid_stats(&hs);
        std::fprintf(out,
                     "[stats ] hybrid    %llu expert(s) on the host over %llu row(s) "
                     "in %.1f ms (%.3f ms/expert, %d thread(s)) | device took %llu\n",
                     static_cast<unsigned long long>(hs.cpu_experts),
                     static_cast<unsigned long long>(hs.cpu_rows), hs.cpu_ms,
                     hs.cpu_experts ? hs.cpu_ms / static_cast<f64>(hs.cpu_experts)
                                    : 0.0,
                     hs.threads,
                     static_cast<unsigned long long>(hs.gpu_experts));
    }
    std::fprintf(out,
                 "[stats ] warm      %.0f MiB capacity, %.1f%% in use | %llu admissions, "
                 "%llu evictions, %llu rejects | %llu device evictions, %llu decay events\n",
                 static_cast<f64>(ec.warm_capacity_bytes()) / 1048576.0,
                 ec.warm_capacity_bytes() > 0
                     ? 100.0 * static_cast<f64>(ec.warm_used_bytes()) /
                           static_cast<f64>(ec.warm_capacity_bytes())
                     : 0.0,
                 static_cast<unsigned long long>(ec.warm_admissions()),
                 static_cast<unsigned long long>(ec.warm_evictions()),
                 static_cast<unsigned long long>(ec.warm_rejects()),
                 static_cast<unsigned long long>(ec.evictions()),
                 static_cast<unsigned long long>(ec.decay_events()));
    krk::HostTime::get().report(out);
}

int run_bench(Engine &engine, const Args &a, f64 load_ms, f64 start_ms) {
    GenerateParams p;
    p.prompt =
        "The history of computing is a history of abstraction: from relays to "
        "vacuum tubes, from transistors to integrated circuits, and now from "
        "monolithic kernels to distributed accelerators. In each transition the "
        "same question returns — what is the smallest primitive that still "
        "expresses the computation?";
    p.max_tokens = a.max_tokens;
    p.sampler.temp = a.temp;
    p.sampler.top_k = a.top_k;
    p.sampler.top_p = a.top_p;
    p.sampler.min_p = a.min_p;
    p.sampler.repeat_penalty = a.rep_pen;
    p.sampler.greedy = a.greedy;
    p.sampler.seed = a.seed;

    // Sample this process's per-engine GPU utilization for the duration of
    // the run, so the report below can state which engine actually did the
    // work (Compute 0 / 3D / Copy) instead of leaving it to a profiler.
    EngineUsage usage;
    usage.start();

    GenerateResult r;
    if (!engine.generate(p, &r)) {
        usage.stop();
        std::fprintf(stderr, "kraken: benchmark run failed\n");
        return 1;
    }
    usage.stop();
    const f64 prefill_tps = r.prompt_tokens > 0 ? r.prompt_tokens / (r.prefill_ms / 1000.0) : 0;
    const f64 decode_tps = r.decode_steps > 0
                               ? r.decode_steps / (r.decode_ms / 1000.0)
                               : 0;
    std::printf("\n");
    std::printf("model          %s\n", engine.model().cfg().name.c_str());
    std::printf("device         %s\n", engine.device().name.c_str());
    std::printf("prefill        %d tokens in %.1f ms  (%.1f tok/s)\n", r.prompt_tokens,
                r.prefill_ms, prefill_tps);
    std::printf("decode         %d tokens in %.1f ms  (%.1f tok/s)\n", r.generated,
                r.decode_ms, decode_tps);
    // The load phase is a third of start-to-end on every model measured (8B:
    // 1531 ms of load against 2973 ms of decode), and it stays invisible here
    // unless it is printed. Reporting only prefill and decode invites
    // optimising two thirds of the process.
    if (load_ms >= 0) {
        // start_ms is the uptime before the load began (argument parsing plus
        // backend creation), so total_ms is start-to-end minus only the tail
        // after the last decode token. Without this line the report covers two
        // thirds of the process and the load phase is invisible.
        const f64 total_ms = start_ms + load_ms + r.prefill_ms + r.decode_ms;
        std::printf("startup        %.1f ms  (arg parse + backend create)\n", start_ms);
        std::printf("load           %.1f ms  (%.0f%% of start-to-end)\n", load_ms,
                    total_ms > 0 ? 100.0 * load_ms / total_ms : 0.0);
        std::printf("total          %.1f ms  (startup + load + prefill + decode)\n", total_ms);
    }
    usage.report(stdout);
    print_run_stats(stdout, engine, r, load_ms, 0, 0);
    if (engine.has_draft() && engine.spec_steps() > 0) {
        const f64 rate =
            engine.draft_proposed() > 0
                ? 100.0 * static_cast<f64>(engine.draft_accepted()) /
                      static_cast<f64>(engine.draft_proposed())
                : 0.0;
        std::printf("speculation   %llu steps, %llu/%llu draft tokens accepted "
                    "(%.1f%%)\n",
                    static_cast<unsigned long long>(engine.spec_steps()),
                    static_cast<unsigned long long>(engine.draft_accepted()),
                    static_cast<unsigned long long>(engine.draft_proposed()), rate);
    }
    if (engine.model().cfg().is_moe) {
        const ExpertCache &ec = engine.model().experts();
        const u64 acq = ec.acquires();
        const u64 hits = ec.hits();
        std::printf("experts        %llu loads, %llu evictions, %.1f%% hit rate, "
                    "%zu/%zu pinned\n",
                    static_cast<unsigned long long>(ec.loads()),
                    static_cast<unsigned long long>(ec.evictions()),
                    acq > 0 ? 100.0 * static_cast<f64>(hits) / static_cast<f64>(acq)
                            : 0.0,
                    ec.pinned_slots(), ec.resident_slots());
        if (ec.warm_capacity_bytes() > 0)
            std::printf("expert WARM    %.0f MiB pageable, %zu expert(s) held, "
                        "%llu admissions, %llu promotions, %llu evictions, %llu rejects\n",
                        static_cast<f64>(ec.warm_capacity_bytes()) / (1024.0 * 1024.0),
                        ec.warm_slots(),
                        static_cast<unsigned long long>(ec.warm_admissions()),
                        static_cast<unsigned long long>(ec.promotions()),
                        static_cast<unsigned long long>(ec.warm_evictions()),
                        static_cast<unsigned long long>(ec.warm_rejects()));
    }
    // The hybrid split, when it ran. Printed unconditionally once enabled: a
    // split that moved nothing is exactly as informative as one that moved
    // everything, and the point of the line is to say which experts went where
    // without a second run to find out.
    if (engine.hybrid_enabled()) {
        HybridStats hs;
        engine.hybrid_stats(&hs);
        std::printf("hybrid experts host %llu expert(s) over %llu row(s) in %.1f ms "
                    "(%.3f ms/expert) on %d thread(s); device took %llu\n",
                    static_cast<unsigned long long>(hs.cpu_experts),
                    static_cast<unsigned long long>(hs.cpu_rows), hs.cpu_ms,
                    hs.cpu_experts ? hs.cpu_ms / static_cast<f64>(hs.cpu_experts)
                                   : 0.0,
                    hs.threads,
                    static_cast<unsigned long long>(hs.gpu_experts));
    }
    return 0;
}

} // namespace

// Process-phase timeline (KRK_PHASE=1). `--bench` reports prefill and decode,
// but not what else the process spends its time on -- and on the 8B that was
// 1531 ms of load against 2973 ms of decode, i.e. a third of start-to-end that
// the benchmark never mentioned. This prints every phase's wall time to stderr
// so the budget is attributed rather than guessed. Costs one getenv when off.
struct PhaseClock {
    const char *on;
    f64 t0;
    f64 last;
    PhaseClock() {
        on = std::getenv("KRK_PHASE");
        t0 = now();
        last = t0;
    }
    static f64 now() {
        return static_cast<f64>(std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count()) / 1000.0;
    }
    void mark(const char *what) {
        if (!on) return;
        const f64 t = now();
        std::fprintf(stderr, "[phase] %-22s %8.1f ms  (total %8.1f ms)\n",
                     what,
                     t - last, t - t0);
        std::fflush(stderr);
        last = t;
    }
};

int main(int argc, char **argv) {
    PhaseClock ph;
    ph.mark("process start");
    Args a;
    if (!parse(argc, argv, &a)) return 2;
    log_set_level(a.verbose ? Log::Debug : Log::Info);

    // The per-op timeline lives in the HIP backend (src/hip/op_time.hpp), which
    // this translation unit cannot include — it is compiled for both backends.
    // The handshake is the environment, read once when the profiler's singleton
    // is constructed at the first instrumented op, so setting it here (before
    // the backend exists) is early enough. `--profile` implies timing every op:
    // a filtered timeline is missing most of the step.
    if (a.profile) {
#ifdef _WIN32
        _putenv_s("KRK_PROFILE", "1");
#else
        setenv("KRK_PROFILE", "1", 1);
#endif
    }

    std::string err;
    Backend *be = nullptr;
    if (a.cpu) {
        be = make_cpu_backend();
    } else {
        be = make_hip_backend(a.device, &err);
        if (!be) {
            KRK_WARN("%s", err.c_str());
            KRK_WARN("falling back to the CPU reference backend");
            be = make_cpu_backend();
        }
    }

    EngineConfig cfg;
    cfg.model_path = a.model;
    cfg.n_ctx = a.ctx;
    cfg.prefill_chunk = a.chunk;
    cfg.threads = a.threads;
    cfg.seed = a.seed;
    cfg.expert_cache_mb = a.expert_cache_mb;
    cfg.vram_cap_mb = a.vram_cap_mb;
    cfg.vram_tier_gb = a.vram_tier_gb;
    cfg.host_ram_mb = a.host_ram_mb;
    cfg.expert_cache_slots = a.expert_cache_slots;
    cfg.expert_warm_mb = a.expert_warm_mb;
    cfg.kv_hot_mb = a.kv_hot_mb;
    cfg.kv_warm_mb = a.kv_warm_mb;
    cfg.kv_cold_dir = a.kv_cold_dir;
    cfg.expert_warm_prefetch = a.expert_warm_prefetch;
    cfg.expert_warmup = a.expert_warmup;
    cfg.expert_warmup_ms = a.expert_warmup_ms;
    cfg.ram_tier_gb = a.ram_tier_gb;
    cfg.hybrid_experts = a.hybrid_experts;
    cfg.hybrid_threads = a.hybrid_threads;
    cfg.hybrid_max_rows = a.hybrid_max_rows;
    cfg.hybrid_frac = static_cast<f32>(a.hybrid_frac);

    ph.mark("backend create");
    Engine engine;
    engine.set_expert_scan(a.expert_scan, a.expert_stub);
    const f64 load_t0 = PhaseClock::now();
    if (!engine.init(be, cfg, &err)) {
        std::fprintf(stderr, "kraken: %s\n", err.c_str());
        delete be;
        return 1;
    }
    // Wall time of the load, and this process's uptime at the same moment: the
    // benchmark reports both so start-to-end is a number it prints rather than
    // one the reader has to reconstruct from KRK_PHASE.
    const f64 bench_load_ms = PhaseClock::now() - load_t0;
    const f64 bench_start_ms = load_t0 - ph.t0;
    ph.mark("engine.init (load)");
    if (a.expert_stub && !a.expert_scan) {
        // --expert-stub on its own does NOTHING: set_expert_scan stores
        // `scan && stub`, so the stub arm never reaches the engine and the run
        // looks exactly like a normal one. That is not a cosmetic trap -- it
        // is how this repository came to record "the MoE expert path is ~3% of
        // a decode token" when it is 91% (docs/PERF-ANALYSIS.md section B3,
        // retracted). Measured on laguna-xs2 with the flag passed alone:
        // 10602 acquires, 4590 COLD misses, 98.8 ms/token, i.e. no stub at all;
        // with --expert-scan added: 8.5 ms/token.
        std::fprintf(stderr,
                     "kraken: --expert-stub only means something together with "
                     "--expert-scan (it stubs the experts the scan measures); "
                     "passing it alone is a silent no-op\n");
        return 2;
    }

    if (!a.draft.empty()) {
        engine.set_draft_window(a.draft_tokens);
        if (!engine.load_draft(a.draft, &err)) {
            std::fprintf(stderr, "kraken: draft model: %s\n", err.c_str());
            engine.shutdown();
            delete be;
            return 1;
        }
    }

    std::printf("%s %s | %s\n", kEngineName, kEngineVersion, be->describe().c_str());

    if (a.expert_scan && a.prompt.empty() && !a.bench) {
        // Nothing to measure without tokens: a scan over an empty prompt would
        // write an index with zero mass in every expert, which reads as "this
        // model routes nowhere" rather than as "this scan measured nothing".
        // An empty --prompt is a distinct mistake from a missing one, so it
        // gets its own message rather than being told to pass a flag it passed.
        std::fprintf(stderr,
                     a.prompt_given
                         ? "kraken: --expert-scan needs a non-empty --prompt "
                           "(or --bench): there are no token positions to "
                           "measure the routers on\n"
                         : "kraken: --expert-scan needs --prompt (or --bench) "
                           "so there are token positions to measure the routers "
                           "on\n");
        engine.shutdown();
        delete be;
        return 2;
    }

    if (a.info) {
        const ModelConfig &mc = engine.model().cfg();
        std::printf("\narchitecture   %s (%s)\n", mc.arch.c_str(), mc.name.c_str());
        std::printf("embd           %d\n", mc.n_embd);
        std::printf("layers         %d\n", mc.n_layer);
        std::printf("heads          %d q / %d kv, head_dim %d\n", mc.n_head, mc.n_head_kv,
                    mc.head_dim);
        std::printf("ffn            %d\n", mc.n_ff);
        std::printf("vocab          %d\n", mc.n_vocab);
        std::printf("context        %d train\n", mc.n_ctx_train);
        std::printf("rope           base %.0f, scale %.3f, frac %.3f\n", mc.rope_base,
                    mc.rope_scale, mc.rope_frac);
        std::printf("tied embd      %s\n", mc.tied_embeddings ? "yes" : "no");
        std::printf("qk norm        %s\n", mc.qk_norm ? "yes" : "no");
        if (mc.is_moe) {
            std::printf("moe            %d experts, top-%d%s\n", mc.n_expert,
                        mc.n_expert_used,
                        mc.n_ff_shexp > 0 ? " + shared expert" : "");
            std::printf("expert ffn     %d (shared %d)\n", mc.n_ff_exp, mc.n_ff_shexp);
            const ExpertCache &ec = engine.model().experts();
            std::printf("expert cache   %.0f MiB budget, %zu/%zu slots resident, "
                        "policy LFU+aging (pin at %u)\n",
                        static_cast<f64>(ec.budget_bytes()) / (1024.0 * 1024.0),
                        ec.resident_slots(), ec.capacity_slots(),
                        ExpertCache::kPinThreshold);
            if (ec.warm_capacity_bytes() > 0)
                std::printf("expert WARM    %.0f MiB pageable host tier "
                            "(cold reads are read-through; VRAM misses promote)\n",
                            static_cast<f64>(ec.warm_capacity_bytes()) /
                                (1024.0 * 1024.0));
        }
        std::printf("weights        %.2f GiB on device\n",
                    static_cast<f64>(engine.model().weight_bytes()) /
                        (1024.0 * 1024.0 * 1024.0));
        std::printf("tokenizer      %s, vocab %d, bos %d, eos %d\n",
                    engine.tokenizer().model() == TokModel::Bpe ? "byte-BPE" : "SPM",
                    engine.tokenizer().n_vocab(), engine.tokenizer().bos(),
                    engine.tokenizer().eos());
        engine.shutdown();
        delete be;
        return 0;
    }

    ph.mark("banner + draft");
    if (a.bench) {
        const int rc = run_bench(engine, a, bench_load_ms, bench_start_ms);
        ph.mark("run_bench (prefill+decode)");
        // The bench arm used to return here, which silently dropped the index
        // for `--expert-scan --bench` -- the very combination the argument
        // validation above advertises as the way to scan without a prompt. The
        // scan's positions came from the bench's own prefill and decode, so
        // this is the same index a --prompt run would have written, just never
        // written. `--expert-scan --bench` was a no-op that looked like a pass.
        if (a.expert_scan) engine.write_expert_index(a.model + ".krakenexperts.json");
        engine.shutdown();
        ph.mark("engine.shutdown");
        delete be;
        ph.mark("delete be -> exit");
        return rc;
    }

    // --tokenize: what the model actually sees. A reference comparison is a
    // token-id list, so template markers and the tokenizer's special-token
    // rules can be diffed instead of inferred from the generated text.
    if (a.tokenize) {
        const std::string text =
            a.prompt_given ? (a.chat ? chatml(a.system, a.prompt) : a.prompt)
                           : std::string();
        std::vector<i32> ids;
        const int n = engine.tokenizer().encode(text, ids, true);
        std::printf("%d tokens\n[", n);
        for (int i = 0; i < n; i++) std::printf("%s%d", i ? ", " : "", ids[i]);
        std::printf("]\n");
        for (int i = 0; i < n; i++)
            std::printf("%6d  %s\n", ids[i],
                        engine.tokenizer().piece(ids[i]).c_str());
        engine.shutdown();
        delete be;
        return 0;
    }

    SinkCtx sink;
    sink.chat = a.chat;

    const f64 load_ms = bench_load_ms;
    auto generate_once = [&](const std::string &prompt) -> bool {
        GenerateParams p;
        p.prompt = prompt;
        p.max_tokens = a.max_tokens;
        p.sampler.temp = a.temp;
        p.sampler.top_k = a.top_k;
        p.sampler.top_p = a.top_p;
        p.sampler.min_p = a.min_p;
        p.sampler.repeat_penalty = a.rep_pen;
        p.sampler.greedy = a.greedy;
        p.sampler.seed = a.seed;
        p.stop = {"<|im_end|>", "<|eot_id|>"};
        p.debug_topk = a.debug_topk;
        p.sample_host = a.sample_host;
        p.sink.fn = sink_cb;
        p.sink.user = &sink;

        GenerateResult r;
        if (!engine.generate(p, &r)) return false;
        print_run_stats(stderr, engine, r, load_ms,
                        be->device_free_bytes(), be->device_total_bytes());
        // Speculation telemetry. It lived only in run_bench, which meant the one
        // run a user actually does -- start the CLI, ask a question, read the
        // answer -- could not say whether the draft earned its keep. The number
        // that matters is the accept rate: a drafter that proposes four tokens
        // and lands one is still a win, and one that lands none is pure cost.
        if (engine.has_draft() && engine.spec_steps() > 0) {
            const f64 rate =
                engine.draft_proposed() > 0
                    ? 100.0 * static_cast<f64>(engine.draft_accepted()) /
                          static_cast<f64>(engine.draft_proposed())
                    : 0.0;
            std::fprintf(stderr,
                         "[stats ] speculation %llu rounds, %llu/%llu draft "
                         "tokens accepted (%.1f%%)\n",
                         static_cast<unsigned long long>(engine.spec_steps()),
                         static_cast<unsigned long long>(engine.draft_accepted()),
                         static_cast<unsigned long long>(engine.draft_proposed()),
                         rate);
        }
        return true;
    };

    int rc = 0;
    if (a.prompt_given) {
        rc = generate_once(a.chat ? chatml(a.system, a.prompt) : a.prompt) ? 0 : 1;
        std::printf("\n");
    } else if (a.interactive) {
        std::string line;
        while (true) {
            std::fprintf(stderr, a.chat ? "user> " : "> ");
            std::fflush(stderr);
            if (!std::getline(std::cin, line)) break;
            if (line == "/quit" || line == "/exit") break;
            if (line.empty()) continue;
            sink.hold.clear();
            if (!generate_once(a.chat ? chatml(a.system, line) : line)) {
                rc = 1;
                break;
            }
            std::printf("\n");
        }
    }

    if (a.expert_scan) {
        // Written after the run, beside the model, so the list travels with the
        // weights and a later load on a different card can still apply it. The
        // file stores each expert's routing mass, not a chosen hot set: which
        // experts deserve VRAM depends on the card (6 GB and 12 GB are both
        // targets) and on the free host tier, so the choice belongs at load
        // time, not here.
        engine.write_expert_index(a.model + ".krakenexperts.json");
    }

    engine.shutdown();
    delete be;
    return rc;
}
