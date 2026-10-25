// main_cli.cpp — the kraken command line front end.
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "engine_usage.hpp"
#include "krk/backend.hpp"
#include "krk/cli_value.hpp"
#include "krk/engine.hpp"
#include "krk/host_time.hpp"
#include "krk/trace.hpp"

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
    // An EAGLE-3 head (tools/eagle_to_gguf.py): a small GGUF beside the model
    // that reads the target's own hidden state and proposes for the same verify
    // loop. Not a model, so it has no engine of its own.
    std::string draft_head;
    int draft_head_prime = 256; // positions a round primes the head over
    int max_tokens = 256;
    int ctx = 4096;
    int chunk = 256;
    int kv_quant = 0; // 0 = f16 KV, 1 = int8 (see EngineConfig::kv_quant)
    int device = 0;
    int threads = 0;
    int expert_cache_mb = 0;    // MoE expert residency budget (MiB), 0 = auto
    int vram_cap_mb = 0;        // total device memory to plan for (MiB), 0 = policy
    int vram_tier_gb = 0;       // plan for a card of this class (GiB), 0 = off
    int host_ram_mb = 0;        // host memory to plan for (MiB), 0 = policy
    int expert_cache_slots = 0; // MoE resident (layer, expert) slot cap, 0 = auto
    int expert_warm_mb = -1;     // MoE WARM tier in pageable host RAM (MiB): <0 auto, 0 off
    int expert_pin_mb = -1;      // pinned prefix of that tier (MiB): <0 no override
    f32 router_resident = -1.0f; // router near-tie margin toward resident experts
    int weight_vram_mb = 0;      // dense weight VRAM budget, MiB: 0 auto, <0 off
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
    // Run the file's own nextn (MTP) block for one step after prefill and print
    // the token it drafts against the target's argmax. Also makes the loader
    // upload the block; nothing is emitted from the comparison.
    bool mtp_probe = false;
    std::string mtp_path;   // --mtp FILE: a nextn block in a file of its own
    // Decode with the file's own nextn block as the drafter (greedy only).
    bool mtp_spec = false;
    // The rows>1 GEMM rung: the rung lottery (matrix cores where present)
    // instead of the exact grid. DEFAULT ON since the measurement below; the
    // exact grid keeps every prompt row's arithmetic independent of --chunk and
    // costs 36% of the chunk on a MoE model that is already streaming weights.
    //
    // 27B Q4_K_M, 191-token prompt: 45.5 tok/s exact against 196.0 with the
    // lottery, md5-identical output at chunk 64/128/256/512. Qwen3.5-35B-A3B
    // Q4_K_XL, same prompt: 146.3 -> 231.1 tok/s (+58%), output md5-identical,
    // but the PREFILL HASH then differs between chunk sizes (chunk 128
    // 1fcaeaa8..., chunk 256 3b7cab4a...) -- so --prefill-hash forces the exact
    // rung back on, which is the only thing that gate can measure.
    bool prefill_fast = true;
    // True only when the caller ASKED for the fast rung, so the --prefill-hash
    // note below fires on a real conflict instead of on every run.
    bool prefill_fast_explicit = false;
    int hybrid_threads = 0;     // host worker threads, 0 = min(hardware, 8)
    int hybrid_max_rows = 8;    // rows the host arm runs at, 0 = unbounded
    double hybrid_frac = 1.0;   // share of RESIDENT experts forced to the host
    int draft_tokens = 4;       // speculative decoding window
    // Whether --draft-tokens was given. The window has a measured default that
    // differs by drafter (see the set_draft_window call), and an explicit flag
    // always wins over it.
    bool draft_tokens_given = false;
    bool cpu = false;
    bool greedy = false;
    bool tokenize = false;
    // Print one hash of the prompt's last-position logits and exit (see
    // --prefill-hash): the chunk-invariance gate's one observable.
    bool prefill_hash = false;
    bool chat = false;
    bool info = false;
    bool bench = false;
    bool profile = false;
    bool trace = false;
    std::string trace_path;
    bool expert_scan = false;   // measure routing and write an expert index
    bool expert_stub = false;   // run the routers with every expert FFN removed
    bool expert_adapt = false;  // fold this run's routing into the index
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
        "  --kv-quant f16|int8   K/V cache format (default f16). int8 stores each\n"
        "                        row as bytes plus one f32 scale per 32 values, 1.8x\n"
        "                        smaller; both backends implement it, and on the card\n"
        "                        decode and prefill agree bit for bit (MLA and the\n"
        "                        nextn/draft block are still refused by name)\n"
        "  --temp F              sampling temperature, 0 = greedy (default 0.8)\n"
        "  --top-k N             top-k cutoff (default 40)\n"
        "  --top-p F             nucleus cutoff (default 0.95)\n"
        "  --min-p F             min-p cutoff (default 0.05)\n"
        "  --repeat-penalty F    repetition penalty (default 1.1)\n"
        "  --seed N              RNG seed\n"
        "  --greedy              force argmax decoding\n"
        "  --tokenize            print the prompt's token ids and exit\n"
        "  --prefill-hash         prefill the prompt, print a hash of the\n"
        "                        last-position logits and exit. Two runs\n"
        "                        that differ only in --chunk print the same\n"
        "                        hash iff the multi-row forward is bit-exact\n"
        "                        (the chunk-invariance gate). Forces the exact\n"
        "                        rung: the default fast rung is not chunk-\n"
        "                        invariant, so the gate could not see anything.\n"
        "  --chat                wrap the prompt in a ChatML template\n"
        "  --system STR          system message for --chat\n"
        "  --cpu                 use the scalar reference backend\n"
        "  --device N            HIP device index (default 0)\n"
        "  --expert-cache-mb N   MoE expert residency budget in MiB (0 = auto)\n"
        "  --host-ram-mb N       host memory to plan for, in MiB. 0 uses a\n"
        "                        quarter of installed RAM. Bounds every host\n"
        "                        tier inside the process; the model mapping and\n"
        "                        the dense trunk sit outside it.\n"
        "  --fast-prefill         same as --prefill fast, which is now the\n"
        "                         default; kept so existing scripts work\n"
        "  --weight-vram-mb N     dense weights kept in VRAM, in MiB. 0 (default)\n"
        "                        keeps them resident while they fit and streams\n"
        "                        whole layers from the file once they do not,\n"
        "                        which frees VRAM for KV. Negative disables it.\n"
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
        "  --expert-pin-mb N      Pin the first N MiB of that tier, in rank order.\n"
        "                         A promotion out of pinned memory is one DMA with\n"
        "                         no host memcpy in front of it: measured 22.7 GB/s\n"
        "                         against 7.6 on a 120B MoE, +18-25%% decode. It is\n"
        "                         RAM that cannot be paged out, so it is off unless\n"
        "                         asked for; size it to the per-run promotion bytes.\n"
        "  --router-resident P    Residency preference in the router: a pick within\n"
        "                        P of the top score (a fraction of it, so 0.05 is\n"
        "                        5%%) goes to the expert already in VRAM, then to\n"
        "                        the one in WARM. Fewer promotions per token at the\n"
        "                        cost of a near-tie; 0 or absent is off.\n"
        "  --expert-cache-slots N  cap resident (layer, expert) slots (0 = auto)\n"
        "  --expert-warm-mb N     WARM expert cache in host RAM. Cold reads land\n"
        "                        here before VRAM, VRAM evictions stay here, and\n"
        "                        WARM eviction never writes to the file. MiB;\n"
        "                        <0 auto = the expert corpus capped by the largest\n"
        "                        commitment that leaves the OS its reserve: 60%% of\n"
        "                        installed (9.6 GiB on a 16 GiB machine, 57.6 on a\n"
        "                        96 GiB one), or free RAM minus that reserve when\n"
        "                        that is smaller. 0 off.\n"
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
        "  --mtp-spec            decode with the file's own nextn (MTP) block as\n"
        "                        the drafter: its proposals are verified by one\n"
        "                        batched target forward and the longest matching\n"
        "                        prefix is kept, so a round emits several tokens\n"
        "                        per target pass. Greedy only; --draft-tokens\n"
        "                        sets the proposal window (default 4). Output is\n"
        "                        bit-identical to plain greedy decoding.\n"
        "                        KRK_MTP_SPEC=1 is the environment spelling.\n"
        "  --mtp FILE            nextn (MTP) block in a SEPARATE GGUF, for a\n"
        "                        trunk that ships without one (-noMTP quants\n"
        "                        whose MTP was published beside them). Same\n"
        "                        block, same names: --mtp-probe/--mtp-spec\n"
        "                        drive it exactly as an in-file block\n"
        "  --prefill exact|fast  fast (default) takes the rows>1 GEMM to the\n"
        "                        RDNA4 matrix cores: +58%% chunk throughput on\n"
        "                        Qwen3.5-35B-A3B, output identical, but the\n"
        "                        prefill HASH then differs between chunk sizes.\n"
        "                        exact keeps every prompt row's arithmetic\n"
        "                        independent of --chunk and costs that 36%%.\n"
        "                        --prefill-hash always uses exact.\n"
        "  --mtp-probe           run the file's own nextn (MTP) block for one\n"
        "                        step after prefill and print the token it\n"
        "                        drafts against the target's argmax: a match is\n"
        "                        the token a speculative round would accept.\n"
        "                        Measurement only -- generation continues as it\n"
        "                        would have without it. KRK_MTP_PROBE=1 is the\n"
        "                        environment spelling.\n"
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
        "  --hybrid-frac F       share (0..1) of the device tier's OVERFLOW the\n"
        "                        host arm takes (default 1.0 = all of it). It\n"
        "                        cannot move a VRAM-resident expert to the host:\n"
        "                        residency decides placement. The knob that sweeps\n"
        "                        the split at a fixed cache size.\n"
        "  --draft MODEL         draft model for greedy speculative decoding\n"
        "  --draft-tokens N      speculation window: how many tokens a round\n"
        "                        may propose (default 4 for a draft model, 1\n"
        "                        for --mtp-spec -- measured: on the Lightning\n"
        "                        sidecar k=1 decoded 30.3 tok/s against 14.8 at\n"
        "                        k=4, because a full-accept round also collects\n"
        "                        the target's next token)\n"
        "  --draft-head FILE     EAGLE-3 draft head (a GGUF from\n"
        "                        tools/eagle_to_gguf.py); needs no extra VRAM and\n"
        "                        no draft model. Greedy is a prefix match; under\n"
        "                        sampling the round uses the min(1,p/q) rule, so\n"
        "                        the text stays the target's distribution\n"
        "  --draft-head-prime N  positions a round primes the head over, most\n"
        "                        recent last (default 256); 0 = the whole prompt,\n"
        "                        which is exact and costs O(prompt) host FLOP\n"
        "  --info                print model and device info, then exit\n"
        "  --bench               prefill/decode benchmark on a fixed prompt\n"
        "  --trace [PATH]        Chrome-trace timeline (Perfetto/tracing):\n"
        "                        host functions, backend ops, device spans,\n"
        "                        transfers, syncs and counters. PATH sets\n"
        "                        KRK_TRACE_PATH. Bounded window only; compare\n"
        "                        untraced wall time.\n"
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
        "  --expert-adapt       fold this run's own routing into\n"
        "                        <model>.krakenexperts.json when it ends, so the\n"
        "                        next load's tier plan follows what the workload\n"
        "                        actually ran. Runs overlap the scan and needs no\n"
        "                        --expert-scan; a workload repeated converges on\n"
        "                        its own hot set. KRK_EXPERT_RETAIN=<0..1> is the\n"
        "                        weight the existing file keeps (default 0.5)\n"
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
        // A value that is not the number it claims to be USED TO arrive as 0:
        // std::atoi("abc") is 0 and std::atoi("4096x") is 4096, so `--ctx abc`
        // ran with a zero-token context while `--ctx 4096x` ran with 4096 -- a
        // typo read as a choice, silently, with no line anywhere saying so.
        // These parse the WHOLE string and exit 2 on the rest, which is the
        // contract the enum flags already had and the one a script can act on.
        // They exit rather than return false for the same reason `next` does:
        // the caller is a chain of `else if`, so a value that fails here has no
        // branch to return from, and parse()'s bool would read a stray `return
        // 2` as SUCCESS (the defect that made `--prefill bogus` run).
        auto next_i = [&](const char *what) -> i32 {
            const char *v = next(what);
            long long n = 0;
            if (!krk::whole_int(v, &n)) {
                std::fprintf(stderr, "kraken: %s needs a whole number, got '%s'\n",
                             what, v);
                std::exit(2);
            }
            return static_cast<i32>(n);
        };
        auto next_f = [&](const char *what) -> f32 {
            const char *v = next(what);
            double d = 0;
            if (!krk::whole_float(v, &d)) {
                std::fprintf(stderr, "kraken: %s needs a number, got '%s'\n", what, v);
                std::exit(2);
            }
            return static_cast<f32>(d);
        };
        auto next_u = [&](const char *what) -> u64 {
            const char *v = next(what);
            unsigned long long n = 0;
            if (!krk::whole_uint(v, &n)) {
                std::fprintf(stderr,
                             "kraken: %s needs a non-negative whole number, got "
                             "'%s'\n",
                             what, v);
                std::exit(2);
            }
            return static_cast<u64>(n);
        };
        if (f == "--model" || f == "-m") a->model = next("--model");
        else if (f == "--prompt" || f == "-p") {
            a->prompt = next("--prompt");
            a->prompt_given = true;
        }
        else if (f == "--system") a->system = next("--system");
        else if (f == "--draft" || f == "-d") a->draft = next("--draft");
        else if (f == "--draft-tokens")
        {
            a->draft_tokens = next_i("--draft-tokens");
            a->draft_tokens_given = true;
        }
        else if (f == "--draft-head") a->draft_head = next("--draft-head");
        else if (f == "--draft-head-prime")
            a->draft_head_prime = next_i("--draft-head-prime");
        else if (f == "--max-tokens" || f == "-n") a->max_tokens = next_i("--max-tokens");
        else if (f == "--ctx" || f == "-c") a->ctx = next_i("--ctx");
        else if (f == "--kv-quant") {
            const std::string v = next("--kv-quant");
            if (v == "int8") a->kv_quant = 1;
            else if (v == "f16") a->kv_quant = 0;
            else {
                std::fprintf(stderr, "kraken: --kv-quant needs f16|int8, got '%s'\n", v.c_str());
                // parse() is a BOOL: `return 2` converts to true and the caller
                // reads it as "parsed fine", so a typo here ran the default f16
                // cache instead of refusing. main() turns false into exit 2.
                return false;
            }
        }
        else if (f == "--chunk") a->chunk = next_i("--chunk");
        // Prefill GEMM policy. The default is the 'exact' grid, because
        // test_prefill_chunk_bitexact asserts that a prefill reproduces DECODE's
        // association byte for byte at every chunk size, and the throughput-first
        // rung lottery does not: at rows > 8 an intermediate activation trace
        // differs (the logits do not). That invariant costs 4x the prefill --
        // 27B Q4_K_M, 191-token prompt: 45.5 tok/s exact against 196.0 with the
        // lottery (gemm(wmma)), output md5 identical at chunk 64/128/256/512.
        // This flag is the opt-out for a caller who wants the speed more than the
        // trace-level reproducibility; KRK_GEMM_EXACT=0 is the same switch.
        else if (f == "--fast-prefill") {
            a->prefill_fast = true;
            a->prefill_fast_explicit = true;
        }
        else if (f == "--temp") a->temp = next_f("--temp");
        else if (f == "--top-k") a->top_k = next_i("--top-k");
        else if (f == "--top-p") a->top_p = next_f("--top-p");
        else if (f == "--min-p") a->min_p = next_f("--min-p");
        else if (f == "--repeat-penalty") a->rep_pen = next_f("--repeat-penalty");
        else if (f == "--seed") a->seed = next_u("--seed");
        else if (f == "--device") a->device = next_i("--device");
        else if (f == "--threads") a->threads = next_i("--threads");
        else if (f == "--host-ram-mb")
            a->host_ram_mb = next_i("--host-ram-mb");
        else if (f == "--weight-vram-mb")
            a->weight_vram_mb = next_i("--weight-vram-mb");
        else if (f == "--kv-hot-mb")
            a->kv_hot_mb = next_i("--kv-hot-mb");
        else if (f == "--kv-warm-mb")
            a->kv_warm_mb = next_i("--kv-warm-mb");
        else if (f == "--kv-cold-dir")
            a->kv_cold_dir = next("--kv-cold-dir");
        else if (f == "--vram-cap-mb")
            a->vram_cap_mb = next_i("--vram-cap-mb");
        else if (f == "--vram-tier" || f == "--vram-tier-gb")
            a->vram_tier_gb = next_i("--vram-tier");
        else if (f == "--expert-cache-mb")
            a->expert_cache_mb = next_i("--expert-cache-mb");
        else if (f == "--expert-cache-slots")
            a->expert_cache_slots = next_i("--expert-cache-slots");
        else if (f == "--expert-warm-mb" || f == "--expert-l2-mb")
            a->expert_warm_mb = next_i("--expert-warm-mb");
        else if (f == "--expert-pin-mb")
            a->expert_pin_mb = next_i("--expert-pin-mb");
        else if (f == "--router-resident")
            a->router_resident = next_f("--router-resident");
        else if (f == "--expert-warm-prefetch")
            a->expert_warm_prefetch = next_i("--expert-warm-prefetch") != 0;
        else if (f == "--expert-warmup")
            a->expert_warmup = next_i("--expert-warmup");
        else if (f == "--expert-warmup-ms")
            a->expert_warmup_ms = next_i("--expert-warmup-ms");
        else if (f == "--ram-tier" || f == "--ram-tier-gb")
            a->ram_tier_gb = next_i("--ram-tier");
        else if (f == "--mtp") {
            a->mtp_path = next("--mtp");
            // Asking for the block implies loading it: a sidecar that was named
            // and then never read would be a flag that does nothing.
            a->mtp_probe = true;
        } else if (f == "--mtp-probe")
            a->mtp_probe = true;
        else if (f == "--mtp-spec")
            a->mtp_spec = true;
        else if (f == "--prefill") {
            const std::string v = next("--prefill");
            if (v == "fast") {
                a->prefill_fast = true;
                a->prefill_fast_explicit = true;
            } else if (v == "exact") {
                a->prefill_fast = false;
            } else {
                std::fprintf(stderr, "kraken: --prefill needs exact|fast, got '%s'\n", v.c_str());
                return false; // see --kv-quant above: 2 would read as success
            }
        }
        else if (f == "--hybrid-experts")
            a->hybrid_experts = next_i("--hybrid-experts") != 0;
        else if (f == "--hybrid-threads")
            a->hybrid_threads = next_i("--hybrid-threads");
        else if (f == "--hybrid-max-rows")
            a->hybrid_max_rows = next_i("--hybrid-max-rows");
        else if (f == "--hybrid-frac")
            a->hybrid_frac = next_f("--hybrid-frac");
        else if (f == "--greedy") a->greedy = true;
        else if (f == "--tokenize") a->tokenize = true;
        else if (f == "--prefill-hash") a->prefill_hash = true;
        else if (f == "--chat") a->chat = true;
        else if (f == "--cpu") a->cpu = true;
        else if (f == "--info") a->info = true;
        else if (f == "--bench") a->bench = true;
        else if (f == "--profile") a->profile = true;
        else if (f == "--trace") {
            a->trace = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') a->trace_path = argv[++i];
        }
        else if (f == "--profile-log" || f == "--profile_lib") {
            // The profiler tables (host phases, components, device ops) go to
            // this file instead of stderr: a comparison run needs the artifact,
            // and a table on a console scrolls away. KRK_PROFILE_LOG does the
            // same thing through the environment.
            if (i + 1 < argc && argv[i + 1][0] != '-') {
#ifdef _WIN32
                _putenv_s("KRK_PROFILE_LOG", argv[++i]);
#else
                setenv("KRK_PROFILE_LOG", argv[++i], 1);
#endif
                a->profile = true;
            }
        }
        else if (f == "--expert-scan") a->expert_scan = true;
        else if (f == "--expert-stub") a->expert_stub = true;
        else if (f == "--expert-adapt") a->expert_adapt = true;
        else if (f == "--debug-topk")
            a->debug_topk = next_i("--debug-topk");
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
// The activation-range readout as an invariant, not a report.
//
// Two counters and one exponent are the whole answer to "did this run compute
// the model's own numbers". `silu`, `experts-sum` and `residual` count stores
// that left f16 and were clipped -- each one a value the f32 reference carries
// exactly and this engine does not -- and an exponent at the clamp means the
// storage scale itself hit its cap, which is only reachable when the value is
// beyond anything an exponent can cover (an `inf` in the row) or when the scale
// climbed on its own (the ratchet that made Laguna-XS-2.1 report 2^-30 for a
// value that needed 2^-5, and S-2.1 2^-60 beside `residual 0`).
//
// A run that trips this is not a run whose text can be trusted, so it does not
// get to exit 0 with the fact buried in the log: the verdict is printed as an
// error and the process exit status carries it. `KRK_RANGE_STRICT=0` demotes it
// back to a log line, for the probes that read a clipped model on purpose.
static constexpr int kRexClamp = 60;

// Set by a run whose activation range was violated in strict mode; the
// CLI's generate path returns bool, so the verdict reaches the exit
// status through here rather than through its own return value.
static int g_range_exit = 0;

static bool range_violated(unsigned long long silu, unsigned long long sum,
                           unsigned long long resid, int rexp_max) {
    return silu != 0 || sum != 0 || resid != 0 || rexp_max >= kRexClamp;
}

static bool range_strict() {
    const char *env = std::getenv("KRK_RANGE_STRICT");
    return !(env && env[0] == '0');
}

static void range_verdict(FILE *out, unsigned long long silu,
                          unsigned long long sum, unsigned long long resid,
                          int rexp_max, const char *where) {
    if (!range_violated(silu, sum, resid, rexp_max)) return;
    std::fprintf(out,
                 "[%s ] range     %s: %llu silu, %llu experts-sum and %llu "
                 "residual store(s) left f16 and were clipped%s; compare "
                 "against the --cpu arm before trusting this output\n",
                 range_strict() ? "error" : "warn", where,
                 silu, sum, resid,
                 rexp_max >= kRexClamp
                     ? ", and the storage exponent reached the clamp"
                     : "");
}

static int range_exit_code(unsigned long long silu, unsigned long long sum,
                           unsigned long long resid, int rexp_max) {
    if (!range_strict()) return 0;
    return range_violated(silu, sum, resid, rexp_max) ? 2 : 0;
}

static void print_run_stats(FILE *out, Engine &engine, const GenerateResult &r,
                            f64 load_ms, size_t dev_free, size_t dev_total,
                            unsigned long long sat_silu, unsigned long long sat_sum,
                            unsigned long long sat_resid, int rexp_max,
                            Backend *be) {
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
    const bool spec_run =
        (engine.has_draft() || engine.mtp_spec_active()) && engine.spec_steps() > 0;
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
    // The H2D probe's ledger, which only means anything across runs: this init
    // either reused a cached verdict or paid for a measurement, and the question
    // "is the cache earning its keep" is answered by how often it has been the
    // former. Silent when nothing has been counted (a dense model has no
    // promotion path, so it never probes and must not print zeros every run).
    {
        const krk::H2dProbeTally &tally = engine.h2d_tally();
        const std::string h2d = krk::h2d_tally_text(tally);
        if (!h2d.empty()) std::fprintf(out, "[stats ] h2d       %s\n", h2d.c_str());
        // And per probe size, because the sizes are not interchangeable: a 64 MiB
        // verdict is not the one a 16 MiB model asks for, and its measurement
        // costs several times as much. The total above is the headline; these are
        // the rows it is made of, and only the sizes this machine has actually
        // probed appear.
        for (size_t i = 0; i < tally.buckets.size(); i++) {
            const std::string bucket = krk::h2d_tally_bucket_text(tally.buckets[i]);
            if (!bucket.empty())
                std::fprintf(out, "[stats ] h2d         %s\n", bucket.c_str());
        }
    }

    // Activation range. Every bound in the f16 activation path is a bounded
    // wrong answer, and printing nothing about it is how two laguna files read as
    // "runnable" while decoding nonsense. Printed for every run, dense models
    // included, so zeros are a stated fact rather than a silence.
    std::fprintf(out,
                 "[stats ] range     f16 activation bounds fired: silu %llu, "
                 "experts-sum %llu, residual %llu%s\n",
                 sat_silu, sat_sum, sat_resid,
                 (sat_silu || sat_sum || sat_resid)
                     ? "  (a value left f16 and was clipped; the f32 reference "
                       "carries it exactly)"
                     : "");
    // The residual stream's own answer to the same question, printed with equal
    // regularity: a non-zero exponent means the hidden state left f16 and is
    // being carried by the storage scaling instead of clipped. It is the one
    // response to the range problem that is a fix rather than a bound, so a run
    // says whether it is in play instead of leaving it to be inferred.
    std::fprintf(out,
                 "[stats ] range     residual rows stored scaled: max exponent %d%s\n",
                 rexp_max,
                 rexp_max > 0
                     ? "  (the hidden state left f16 and rides a power-of-two "
                       "storage scale; every norm puts it back exactly)"
                     : "");

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
    const auto pool = engine.expert_pool_stats();
    std::fprintf(out, "[stats ] active-stage issued %llu, consumed %llu, demand wait %.3f ms\n",
                 static_cast<unsigned long long>(ec.active_staged()),
                 static_cast<unsigned long long>(ec.active_used()), ec.active_wait_ms());
    // The overlap probe (KRK_STAGE_TIMING=1) answers the only question the
    // staging switch turns on: did the copies actually run beside the reader
    // kernels, or did the step merely gain a stream? Without it the A/B
    // result is a number with no mechanism behind it.
    if (pool.staged_measured) {
        std::fprintf(out,
                     "[stats ] stage-timing %llu staged copies: %.3f ms copy, "
                     "%.3f ms reader window, %.3f ms overlapped, hidden %llu/%llu\n",
                     static_cast<unsigned long long>(pool.staged_measured),
                     pool.staged_copy_ms, pool.staged_reader_ms,
                     pool.staged_overlap_ms,
                     static_cast<unsigned long long>(pool.staged_hidden),
                     static_cast<unsigned long long>(pool.staged_measured));
    }
    if (pool.allocations || pool.reuses)
        std::fprintf(out,
                     "[stats ] pool-alloc where: trim %.1f ms | driver query %.1f ms "
                     "(%llu probe(s)) | new cell %.1f ms\n",
                     pool.trim_ms, pool.probe_ms,
                     static_cast<unsigned long long>(pool.probes),
                     pool.malloc_ms);
    std::fprintf(out, "[stats ] expert-pool budget %zu committed %zu live %zu reusable %zu bytes; "
                     "alloc %llu reuse %llu trim %llu guard-fallback %llu | "
                     "release %llu mark(s) for %llu slice(s), reader-done %llu | "
                     "honour device %llu host %llu implicit %llu, host-wait %.3f ms\n",
                 pool.budget, pool.committed, pool.live, pool.reusable,
                 static_cast<unsigned long long>(pool.allocations),
                 static_cast<unsigned long long>(pool.reuses),
                 static_cast<unsigned long long>(pool.trims),
                 static_cast<unsigned long long>(pool.guard_fallbacks),
                 static_cast<unsigned long long>(pool.release_marks),
                 static_cast<unsigned long long>(pool.release_slices),
                 static_cast<unsigned long long>(pool.reader_marks),
                 static_cast<unsigned long long>(pool.order_device),
                 static_cast<unsigned long long>(pool.order_host),
                 static_cast<unsigned long long>(pool.order_implicit),
                 pool.honor_host_ms);
    const u64 acq = ec.acquires();
    const f64 token_div = r.generated > 0 ? static_cast<f64>(r.generated) : 1.0;
    std::fprintf(out,
                 "[stats ] experts   %.0f MiB VRAM budget, %zu/%zu slots resident, "
                 "%zu pinned | WARM %.0f MiB (%.0f pinned of a %.0f budget), "
                 "%zu held (%.0f MiB, %.0f MiB prefetched)\n",
                 static_cast<f64>(ec.budget_bytes()) / 1048576.0, ec.resident_slots(),
                 ec.capacity_slots(), ec.pinned_slots(),
                 static_cast<f64>(ec.warm_capacity_bytes()) / 1048576.0,
                 static_cast<f64>(ec.warm_pinned_bytes()) / 1048576.0,
                 static_cast<f64>(ec.warm_pin_budget_bytes()) / 1048576.0,
                 ec.warm_slots(),
                 static_cast<f64>(ec.warm_used_bytes()) / 1048576.0,
                 static_cast<f64>(ec.bytes_staged()) / 1048576.0);
    // The three tiers reported apart. A single "hit rate" reads as "the
    // cache works 20% of the time" when VRAM 20% / RAM 80% / file 0% is
    // exactly the policy working: the tiers answer different questions and
    // only the last one is the storage path.
    const f64 share = acq > 0 ? 100.0 / static_cast<f64>(acq) : 0.0;
    // IN PLACE is its own column, not part of WARM hits: it is the acquisitions
    // that spent neither a copy nor a VRAM slot, which is the whole point of the
    // tier on a card smaller than the routed set.
    std::fprintf(out,
                 "[stats ] cache     %llu acquires | HOT hits %llu (%.1f%%) | "
                 "WARM hits %llu (%.1f%%) | in place %llu (%.1f%%) | "
                 "COLD misses %llu (%.1f%%)\n",
                 static_cast<unsigned long long>(acq),
                 static_cast<unsigned long long>(ec.hits()), share * static_cast<f64>(ec.hits()),
                 static_cast<unsigned long long>(ec.host_hits()),
                 share * static_cast<f64>(ec.host_hits()),
                 static_cast<unsigned long long>(ec.inplace_hits()),
                 share * static_cast<f64>(ec.inplace_hits()),
                 static_cast<unsigned long long>(ec.loads()),
                 share * static_cast<f64>(ec.loads()));
    if (ec.prefetch_warm_issued() > 0 || ec.prefetch_warm_taken() > 0)
        std::fprintf(out,
                     "[stats ] read-ahead %llu job(s), %llu expert(s) taken "
                     "(%.0f MiB read on the worker instead of inside a layer)\n",
                     static_cast<unsigned long long>(ec.prefetch_warm_issued()),
                     static_cast<unsigned long long>(ec.prefetch_warm_taken()),
                     static_cast<f64>(ec.prefetch_warm_bytes()) / 1048576.0);
    // A promote-ahead pass that staged nothing is the most likely reason an arm
    // "did not move", and the cache's own counters are what tells that apart
    // from a pass whose copies were simply free.
    if (ec.lookahead_staged() > 0 || ec.lookahead_rejected() > 0)
        std::fprintf(out,
                     "[stats ] lookahead %llu staged, %llu rejected by the budget, "
                     "%llu consumed\n",
                     static_cast<unsigned long long>(ec.lookahead_staged()),
                     static_cast<unsigned long long>(ec.lookahead_rejected()),
                     static_cast<unsigned long long>(ec.active_used()));
    if (engine.router_pref())
        std::fprintf(out,
                     "[stats ] router    %llu picks, %llu (%.1f%%) went to an "
                     "already-resident expert over a near-equal one (margin %.1f%% "
                     "of the top score)\n",
                     static_cast<unsigned long long>(engine.router_picks()),
                     static_cast<unsigned long long>(engine.router_subs()),
                     engine.router_picks() > 0
                         ? 100.0 * static_cast<f64>(engine.router_subs()) /
                               static_cast<f64>(engine.router_picks())
                         : 0.0,
                     static_cast<f64>(engine.router_tie_margin()) * 100.0);
    if (ec.acquires() > 0)
        std::fprintf(out,
                     "[stats ] router    hot share %llu/%llu picks (%.1f%%) landed "
                     "on an expert already in VRAM%s\n",
                     static_cast<unsigned long long>(engine.router_hot()),
                     static_cast<unsigned long long>(ec.acquires()),
                     100.0 * static_cast<f64>(engine.router_hot()) /
                         static_cast<f64>(ec.acquires()),
                     engine.router_pref() ? "" : " (preference off)");
    const f64 prom_ms = ec.promote_ms();
    const f64 prom_mib = static_cast<f64>(ec.bytes_promoted()) / 1048576.0;
    // The batched-promotion arm is off by default, so the clause is only there
    // when it ran; the us/expert figure is the comparable one either way.
    char batch_note[64] = "";
    if (ec.promote_batches() > 0)
        std::snprintf(batch_note, sizeof batch_note, " in %llu batch%s",
                      static_cast<unsigned long long>(ec.promote_batches()),
                      ec.promote_batches() == 1 ? "" : "es");
    std::fprintf(out,
                 "[stats ] transfer  %llu promotions, %.0f MiB in %.1f ms = %.1f GB/s "
                 "(%.1f us/expert%s) | %llu VRAM evictions kept in WARM, %.0f "
                 "MiB | "
                 "%.0f MiB read from the file (%.2f MiB/tok)\n",
                 static_cast<unsigned long long>(ec.promotions()), prom_mib, prom_ms,
                 prom_ms > 0 ? prom_mib / 1024.0 / (prom_ms / 1000.0) : 0.0,
                 ec.promotions() > 0
                     ? prom_ms * 1000.0 / static_cast<f64>(ec.promotions())
                     : 0.0,
                 batch_note,
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
    // The room row's two halves. A victim search walks the whole slot map, so on
    // a tier with 14k WARM entries one eviction is thousands of node visits --
    // this line is what says whether the scan or the release is the cost.
    if (ec.scans() > 0)
        std::fprintf(out,
                     "[stats ] evict       %llu scan(s), %llu slot visits, "
                     "%.1f ms scanning + %.1f ms releasing = %.0f ns/visit, "
                     "%.0f visits/scan | index %zu/%zu keys/resident\n",
                     static_cast<unsigned long long>(ec.scans()),
                     static_cast<unsigned long long>(ec.scan_visits()),
                     ec.scan_ms(), ec.retire_ms(),
                     ec.scan_visits() > 0
                         ? ec.scan_ms() * 1e6 /
                               static_cast<f64>(ec.scan_visits())
                         : 0.0,
                     ec.scans() > 0
                         ? static_cast<f64>(ec.scan_visits()) /
                               static_cast<f64>(ec.scans())
                         : 0.0,
                     ec.vram_index_keys(), ec.vram_index_resident());
    // The same four stages split by phase. The merged row above cannot say
    // which side a fix belongs on: a promotion inside a prefill chunk is the
    // same DMA behind up to `--chunk` rows, a decode promotion is behind one,
    // and load-time warmup is neither. Printed only when a phase ran, so a
    // run with no promotion in one of them does not grow a line of zeros.
    const f64 pf_prom = ec.prefill_promote_ms(), dc_prom = ec.decode_promote_ms();
    if (pf_prom > 0.0 || dc_prom > 0.0) {
        std::fprintf(out,
                     "[stats ] expert-path split  prefill: read %7.1f | room %7.1f | "
                     "alloc %7.1f | xfer %7.1f | promote %7.1f ms\n"
                     "[stats ]                     decode: read %7.1f | room %7.1f | "
                     "alloc %7.1f | xfer %7.1f | promote %7.1f ms\n",
                     ec.prefill_read_ms(), ec.prefill_room_ms(),
                     ec.prefill_alloc_ms(), ec.prefill_xfer_ms(), pf_prom,
                     ec.decode_read_ms(), ec.decode_room_ms(),
                     ec.decode_alloc_ms(), ec.decode_xfer_ms(), dc_prom);
        const f64 other = prom_ms - pf_prom - dc_prom;
        if (other > 0.5)
            std::fprintf(out,
                         "[stats ]                     other: %.0f ms of promote "
                         "time outside both phases (load-time warmup/prefetch)\n",
                         other);
    }
    // The pinned half of the tier, measured on its OWN stream: the promote
    // timers above wrap `upload_pinned_batch`, which only ENQUEUES the copy, so
    // the rate they imply for a pinned promotion is an enqueue rate. Measured
    // wrong once: 7098 MiB "in 107 ms" = 64.6 GB/s on a link that does 27. This
    // line is the event-timed truth, printed only when that path carried bytes.
    if (be && be->pinned_dma_batches() > 0) {
        const f64 pms = be->pinned_dma_ms();
        const f64 pmib = static_cast<f64>(be->pinned_dma_bytes()) / 1048576.0;
        // The RATE is over the batches the bracket actually covered, never over
        // every byte: a batch that found all bracket slots busy went untimed.
        const f64 tmib = static_cast<f64>(be->pinned_dma_timed_bytes()) / 1048576.0;
        const unsigned long long merged = be->pinned_merged();
        std::fprintf(out,
                     "[stats ] pinned    %llu batch(es), %.0f MiB on the copy stream; "
                     "%.0f MiB over %llu timed batch(es) in %.1f ms = %.1f GB/s | "
                     "%llu per-tensor copy(ies) coalesced into their neighbours\n"
                     "[stats ]           the promote timers above wrap the ENQUEUE of a "
                     "pinned copy, not its flight: read this line for the rate, and the "
                     "step time for its cost. Every batch here is one expert (about "
                     "1.7 MiB), and bar_probe puts a single copy of that size at 12.6 "
                     "GB/s against 28.3 for 1 GiB -- the fixed per-copy latency, not "
                     "the link, is what a small promotion pays. 0 coalesced means the "
                     "entries were not adjacent, so a decode difference is not from it\n",
                     be->pinned_dma_batches(), pmib, tmib,
                     be->pinned_dma_timed_batches(), pms,
                     pms > 0 ? tmib / 1024.0 / (pms / 1000.0) : 0.0, merged);
    }
    // The draft engine owns its OWN expert tier, and its budget is derived from
    // the free VRAM left after the target has already taken its share. On a
    // model whose routed set is bigger than the card that can come out at a
    // handful of slots, and every drafter forward pass then thrashes -- while
    // this report read the target's cache only and showed a healthy 95% HOT.
    // Measured on laguna-xs2 as a self-draft: the drafter's tier holds 4 of 9984
    // slots, promotes at 4566 us/expert against 700 us healthy, and turns one
    // round into 4.6 s at 81% acceptance. Without this line the only number a
    // caller sees is "1.6 tok/s", with nothing in the log to attribute it to.
    if (engine.has_draft_model()) {
        const ExpertCache &dec = engine.draft_model().experts();
        const u64 dacq = dec.acquires();
        const f64 dshare = dacq > 0 ? 100.0 / static_cast<f64>(dacq) : 0.0;
        std::fprintf(out,
                     "[stats ] draft     %.0f MiB VRAM budget, %zu/%zu slots "
                     "resident | %llu acquires | HOT %llu (%.1f%%) | WARM %llu | "
                     "COLD %llu | %llu promotions at %.1f us/expert%s\n",
                     static_cast<f64>(dec.budget_bytes()) / 1048576.0,
                     dec.resident_slots(), dec.capacity_slots(),
                     static_cast<unsigned long long>(dacq),
                     static_cast<unsigned long long>(dec.hits()),
                     dshare * static_cast<f64>(dec.hits()),
                     static_cast<unsigned long long>(dec.host_hits()),
                     static_cast<unsigned long long>(dec.loads()),
                     static_cast<unsigned long long>(dec.promotions()),
                     dec.promotions() > 0
                         ? dec.promote_ms() * 1000.0 / static_cast<f64>(dec.promotions())
                         : 0.0,
                     // The signature that matters is the HIT RATE, not the
                     // per-expert time: a tier holding 4 of 9984 slots cannot
                     // serve any reuse at all, so it reads 0% HOT while a
                     // starved allocator makes each promotion 1-5 ms. Gated on
                     // a real acquire count so a two-token probe is not accused.
                     (dacq >= 1000 && dshare * static_cast<f64>(dec.hits()) < 50.0)
                         ? "  <- the drafter's tier cannot hold its routed set; "
                           "every expert is a promotion, so a round costs the "
                           "tier and not the drafter"
                         : "");
    }
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
        // The policy, stated where the split is reported. It is a property of
        // the classification, not a measurement: `cpu` is only ever set on an
        // expert that is NOT VRAM-resident, so a resident expert is never sent
        // to the host whatever the fractions say. A run that cannot say this
        // has to have it inferred from the cache counters, which is how "the
        // hot experts run on the GPU" stayed a claim instead of a line.
        std::fprintf(out,
                     "[stats ] hybrid    policy: only experts the device tier could "
                     "not hold were candidates for the host; every VRAM-resident "
                     "expert ran on the device%s | %llu violation(s)\n",
                     hs.cpu_experts == 0 ? " (the host arm ran nothing)" : "",
                     static_cast<unsigned long long>(hs.policy_violations));
        if (hs.policy_violations > 0) {
            // Not a tuning outcome. Say it in full, next to the line it
            // contradicts, because the alternative is a placement nobody can
            // see from the cache counters.
            std::fprintf(out,
                         "[stats ] hybrid    %llu expert(s) were sent to the host "
                         "while VRAM-resident, which the policy forbids: the "
                         "placement above is NOT what this run did.\n",
                         static_cast<unsigned long long>(hs.policy_violations));
        }
    }
    std::fprintf(out,
                 "[stats ] warm      %.0f MiB capacity, %.1f%% in use | %llu admissions, "
                 "%llu evictions, %llu rejects | %llu device evictions, %llu decay events"
                 " | %llu MiB pinned\n",
                 static_cast<f64>(ec.warm_capacity_bytes()) / 1048576.0,
                 ec.warm_capacity_bytes() > 0
                     ? 100.0 * static_cast<f64>(ec.warm_used_bytes()) /
                           static_cast<f64>(ec.warm_capacity_bytes())
                     : 0.0,
                 static_cast<unsigned long long>(ec.warm_admissions()),
                 static_cast<unsigned long long>(ec.warm_evictions()),
                 static_cast<unsigned long long>(ec.warm_rejects()),
                 static_cast<unsigned long long>(ec.evictions()),
                 static_cast<unsigned long long>(ec.decay_events()),
                 static_cast<unsigned long long>(ec.warm_pinned_bytes() / 1048576u));
    krk::HostTime::get().report(out);
}

int run_bench(Engine &engine, Backend *be, const Args &a, f64 load_ms,
              f64 start_ms) {
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
    const bool spec_run =
        (engine.has_draft() || engine.mtp_spec_active()) && engine.spec_steps() > 0;
    const f64 decode_count = spec_run ? r.generated : r.decode_steps;
    const f64 decode_tps = decode_count > 0
                               ? decode_count / (r.decode_ms / 1000.0)
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
    // The bench report says nothing about device memory, but it must not claim
    // zero saturated stores just because it does not print them either: the
    // counters are read wherever the line is printed.
    unsigned long long sat[3] = {0, 0, 0};
    if (be) be->activation_saturation(sat);
    const int rexp_max_bench = be ? be->residual_exponent_max() : 0;
    print_run_stats(stdout, engine, r, load_ms, 0, 0, sat[0], sat[1], sat[2],
                    rexp_max_bench, be);
    range_verdict(stdout, sat[0], sat[1], sat[2], rexp_max_bench, "bench");
    if (engine.has_draft() && engine.spec_steps() > 0 &&
        engine.draft_accepted() == 0) {
        std::fprintf(stderr,
                     "[stats ] speculation accepted no draft tokens in %llu "
                     "round(s): the drafter is pure cost at this rate\n",
                     static_cast<unsigned long long>(engine.spec_steps()));
    }
    if ((engine.has_draft() || engine.mtp_spec_active()) &&
        engine.spec_steps() > 0) {
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
        if (engine.draft_sampled_n() > 0)
            std::printf(
                "sampled-accept %.1f%% of %llu scored proposals (temp1 "
                "topk20, rejection-sampling verifier)\n",
                100.0 * engine.draft_sampled_expected() /
                    static_cast<f64>(engine.draft_sampled_n()),
                static_cast<unsigned long long>(engine.draft_sampled_n()));
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
    // A benchmark whose numbers came from clipped stores is measuring a
    // different model, so the gate sees it in the status too.
    return range_exit_code(sat[0], sat[1], sat[2], rexp_max_bench);
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

// How much of the existing expert index an --expert-adapt run keeps. 0.5 makes
// two identical runs agree and a changed workload move the ranking within a
// couple of runs; 1 would ignore the run and 0 would forget every earlier one.
static double adapt_retain() {
    const char *e = std::getenv("KRK_EXPERT_RETAIN");
    if (!e || !e[0]) return 0.5;
    char *end = nullptr;
    const double v = std::strtod(e, &end);
    if (end == e || v < 0.0 || v > 1.0) {
        std::fprintf(stderr,
                     "kraken: KRK_EXPERT_RETAIN='%s' is not a weight in "
                     "[0,1] -- using 0.5\n", e);
        return 0.5;
    }
    return v;
}

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
    // Sherlock trace timeline: Chrome-trace JSON for Perfetto. --trace enables
    // the recorder; KRK_TRACE=1 does the same without the flag. The path is
    // --trace's optional value, else KRK_TRACE_PATH, else kraken-trace.json.
    // finish() is atexit-registered so every exit path (bench, hash, error)
    // flushes one file; it is idempotent so an explicit call is also safe.
    if (a.trace) {
#ifdef _WIN32
        _putenv_s("KRK_TRACE", "1");
        if (!a.trace_path.empty()) _putenv_s("KRK_TRACE_PATH", a.trace_path.c_str());
#else
        setenv("KRK_TRACE", "1", 1);
        if (!a.trace_path.empty()) setenv("KRK_TRACE_PATH", a.trace_path.c_str(), 1);
#endif
    }
    {
        const char *te = std::getenv("KRK_TRACE");
        if ((te && *te && std::strcmp(te, "0") != 0) || a.trace) {
            ::krk::Trace::get().force_enable(a.trace_path.c_str());
            std::atexit([] { ::krk::Trace::get().finish(); });
            ::krk::Trace::get().instant("config", "trace-start");
        }
    }

    std::string err;
    Backend *be = nullptr;
    if (a.cpu) {
        be = make_cpu_backend();
    } else {
        // --prefill maps onto the backend's own GEMM rung switch, which is read
        // once at construction. The rung lottery is the default: on
        // Qwen3.5-35B-A3B it is 146.3 -> 231.1 tok/s (+58%) with md5-identical
        // output, and it costs the exact grid's chunk-invariance, which nothing
        // but --prefill-hash consumes.
        //
        // KRK_GEMM_EXACT is set rather than KRK_GEMM_PATH so a caller's own
        // KRK_GEMM_PATH (dp4a/simt probes) still decides WHICH rung the ladder
        // may use -- this flag only decides whether the exact grid outranks it.
        // An EXPLICIT KRK_GEMM_EXACT outranks this default. That is not
        // politeness: the gate's own falsifier runs the same prompt with
        // KRK_GEMM_EXACT=0 and REQUIRES the hashes to disagree, so pinning the
        // rung unconditionally would quietly turn the falsifier into a second
        // copy of the test.
        const char *exact_env = std::getenv("KRK_GEMM_EXACT");
        const bool exact_forced = exact_env && *exact_env;
        if (a.prefill_hash) {
            // The gate's contract is that a --chunk difference is the only
            // difference. Under the fast rung the hash differs between chunk
            // sizes for reasons that have nothing to do with chunking, so the
            // gate pins the rung it was written for — unless the caller pinned
            // one themselves.
            if (!exact_forced) {
#ifdef _WIN32
                _putenv_s("KRK_GEMM_EXACT", "1");
#else
                setenv("KRK_GEMM_EXACT", "1", 1);
#endif
                if (a.prefill_fast_explicit)
                    std::fprintf(stderr,
                                 "kraken: --prefill-hash pins the exact GEMM "
                                 "rung (the fast rung is not chunk-\n"
                                 "invariant)\n");
            }
        } else if (a.prefill_fast && !exact_forced) {
#ifdef _WIN32
            _putenv_s("KRK_GEMM_EXACT", "0");
#else
            setenv("KRK_GEMM_EXACT", "0", 1);
#endif
        }
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
    cfg.kv_quant = a.kv_quant;
    cfg.trace_hash = a.prefill_hash;
    cfg.threads = a.threads;
    cfg.seed = a.seed;
    cfg.expert_cache_mb = a.expert_cache_mb;
    cfg.vram_cap_mb = a.vram_cap_mb;
    cfg.vram_tier_gb = a.vram_tier_gb;
    cfg.host_ram_mb = a.host_ram_mb;
    cfg.expert_cache_slots = a.expert_cache_slots;
    cfg.expert_warm_mb = a.expert_warm_mb;
    cfg.expert_pin_mb = a.expert_pin_mb;
    cfg.router_tie_margin = a.router_resident;
    cfg.draft_head = a.draft_head;
    cfg.draft_head_prime = a.draft_head_prime;
    cfg.weight_vram_mb = a.weight_vram_mb;
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
    {
        const char *env = std::getenv("KRK_MTP_PROBE");
        cfg.mtp_path = a.mtp_path;
    cfg.mtp_probe = a.mtp_probe ||
                        (env && env[0] != '\0' && env[0] != '0');
        const char *senv = std::getenv("KRK_MTP_SPEC");
        cfg.mtp_spec = a.mtp_spec ||
                       (senv && senv[0] != '\0' && senv[0] != '0');
    }

    ph.mark("backend create");
    Engine engine;
    engine.set_expert_scan(a.expert_scan, a.expert_stub);
    if (a.expert_adapt) engine.set_expert_track(true);
    const f64 load_t0 = PhaseClock::now();
    if (!engine.init(be, cfg, &err)) {
        std::fprintf(stderr, "kraken: %s\n", err.c_str());
        // A refused startup must not free the backend: the engine holds
        // pointers into the backend's buffers and its own destructor releases
        // them through that pointer, so deleting the backend here freed it and
        // then unwound past the engine -- SIGSEGV (exit 139) on a path whose
        // whole job is to print one line and stop. `exit` skips that unwind.
        std::exit(1);
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

    // The window is the same knob for both drafters: --draft-tokens N is how
    // many tokens a round may propose, whoever proposes them.
    if (!a.draft.empty() || cfg.mtp_spec) {
        // The window the curve chose. Measured on the Lightning 30B sidecar
        // (32 tokens, same prompt, replay-isolated verify): k=1 30.3 tok/s,
        // k=2 23.3, k=3 18.5, k=4 14.8 -- fewer proposals win because a
        // full-accept round emits its draft AND the target's next token from
        // the same verify row, so k=1 collects 1.68 tokens per round against
        // k=4's 1.39 while paying one row of verify instead of four.
        // --draft-tokens still overrides, and a draft MODEL keeps its default:
        // the curve is this drafter's, whose deep proposals decay.
        int win = a.draft_tokens;
        if (cfg.mtp_spec && a.draft.empty() && !a.draft_tokens_given) win = 1;
        engine.set_draft_window(win);
    }
    if (!a.draft.empty()) {
        if (!engine.load_draft(a.draft, &err)) {
            std::fprintf(stderr, "kraken: draft model: %s\n", err.c_str());
            engine.shutdown();
            delete be;
            return 1;
        }
    }
    if (!a.draft_head.empty()) {
        engine.set_draft_window(a.draft_tokens);
        if (!engine.load_draft_head(a.draft_head, &err)) {
            // The loader's message already names the head.
            std::fprintf(stderr, "kraken: %s\n", err.c_str());
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
        const int rc = run_bench(engine, be, a, bench_load_ms, bench_start_ms);
        ph.mark("run_bench (prefill+decode)");
        // The bench arm used to return here, which silently dropped the index
        // for `--expert-scan --bench` -- the very combination the argument
        // validation above advertises as the way to scan without a prompt. The
        // scan's positions came from the bench's own prefill and decode, so
        // this is the same index a --prompt run would have written, just never
        // written. `--expert-scan --bench` was a no-op that looked like a pass.
        if (a.expert_scan) engine.write_expert_index(a.model + ".krakenexperts.json");
        if (a.expert_adapt)
            engine.merge_expert_index(a.model + ".krakenexperts.json",
                                      adapt_retain());
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

    // --prefill-hash: one prefill of the prompt, then two hashes, and exit.
    // Two runs that differ only in --chunk print the same numbers iff the
    // multi-row forward is bit-identical to the single-row one -- the property
    // chunk invariance is made of, and what `kraken-tests
    // test_prefill_chunk_bitexact` gates on.
    //
    // The first is `prefill-hash`: the embedding, every layer's attention
    // output and every layer's hidden state, over ALL prompt rows and folded
    // per (stage, layer). That is the one that sees a one-ulp change in a row
    // that is not the prompt's last -- the shape the old tiled kernel's
    // reduction order produced. The second is `logits`: the last position's
    // whole logits row, which is what the text depends on. Both cover the
    // WHOLE row, never the sampled token: hashing the text would let a drift
    // survive whenever the argmax happened not to move.
    if (a.prefill_hash) {
        GenerateParams p;
        p.prompt = a.prompt_given
                       ? (a.chat ? chatml(a.system, a.prompt) : a.prompt)
                       : std::string();
        p.max_tokens = 1; // the loop stops after the prefill's first token
        p.sampler.greedy = true;
        p.sampler.temp = 0.0f;
        p.sample_host = true;
        GenerateResult r;
        if (!engine.generate(p, &r)) {
            std::fprintf(stderr, "kraken: prefill failed: %s\n",
                         r.error.empty() ? "generation returned false"
                                         : r.error.c_str());
            std::exit(1);
        }
        // The MODEL's vocab, not the tokenizer's: they can differ in a
        // fixture, and the logits row is exactly n_vocab floats long -- reading
        // one past it hashes whatever buffer the allocator put next, which is
        // how this gate first reported differences that were its own bug.
        const i64 n = engine.model().cfg().n_vocab;
        const f32 *row = engine.last_logits();
        if (!row || n <= 0) {
            std::fprintf(stderr, "kraken: no host logits row to hash\n");
            std::exit(1);
        }
        u64 h = 1469598103934665603ull;
        const unsigned char *b = reinterpret_cast<const unsigned char *>(row);
        for (i64 i = 0; i < n * 4; i++) {
            h ^= b[i];
            h *= 1099511628211ull;
        }
        std::printf(
            "prefill-hash %016llx logits %016llx prompt_tokens=%d first=%d\n",
            static_cast<unsigned long long>(engine.prefill_trace_hash()),
            static_cast<unsigned long long>(h), r.prompt_tokens,
            r.tokens.empty() ? -1 : r.tokens[0]);
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
        // A failed run has always reported WHY in r.error, and this path used to
        // drop it: `--mtp-spec` on a pool that cannot place the block's first
        // expert exited 1 with an empty stderr, which reads as a crash. The
        // error is the only thing that says which of the refusals it was.
        if (!engine.generate(p, &r)) {
            if (!r.error.empty())
                std::fprintf(stderr, "kraken: generate: %s\n", r.error.c_str());
            return false;
        }
        unsigned long long sat[3] = {0, 0, 0};
        be->activation_saturation(sat);
        const int rexp_max_run = be->residual_exponent_max();
        print_run_stats(stderr, engine, r, load_ms,
                        be->device_free_bytes(), be->device_total_bytes(),
                        sat[0], sat[1], sat[2], rexp_max_run, be);
        range_verdict(stderr, sat[0], sat[1], sat[2], rexp_max_run,
                      "generate");
        // The status the caller gets, not just a line in the log: exit 2 is
        // reserved for exactly this, so a script that pipes the text cannot
        // mistake a clipped run for a clean one.
        if (range_exit_code(sat[0], sat[1], sat[2], rexp_max_run) != 0)
            g_range_exit = 2;
        // Speculation telemetry. It lived only in run_bench, which meant the one
        // run a user actually does -- start the CLI, ask a question, read the
        // answer -- could not say whether the draft earned its keep. The number
        // that matters is the accept rate: a drafter that proposes four tokens
        // and lands one is still a win, and one that lands none is pure cost.
        if (engine.has_draft() && engine.spec_steps() > 0 &&
            engine.draft_accepted() == 0) {
            std::fprintf(stderr,
                         "[stats ] speculation accepted no draft tokens in %llu "
                         "round(s): the drafter is pure cost at this rate\n",
                         static_cast<unsigned long long>(engine.spec_steps()));
        }
        if ((engine.has_draft() || engine.mtp_spec_active()) &&
            engine.spec_steps() > 0) {
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
            if (engine.draft_sampled_n() > 0)
                std::fprintf(
                    stderr,
                    "[stats ] speculation sampled-accept %.1f%% of %llu "
                    "scored proposals (temp1 topk20, rejection-sampling "
                    "verifier)\n",
                    100.0 * engine.draft_sampled_expected() /
                        static_cast<f64>(engine.draft_sampled_n()),
                    static_cast<unsigned long long>(engine.draft_sampled_n()));
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
    if (a.expert_adapt) {
        // After the run, beside the model: the tier plan this run's traffic
        // implies, blended with whatever the last measurement wrote. Best
        // effort -- a failed merge must not turn a good generation into an
        // error code, and merge_expert_index already says why it refused.
        engine.merge_expert_index(a.model + ".krakenexperts.json",
                                  adapt_retain());
    }

    engine.shutdown();
    delete be;
    return rc != 0 ? rc : g_range_exit;
}
