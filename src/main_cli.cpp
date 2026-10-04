// main_cli.cpp — the kraken command line front end.
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "engine_usage.hpp"
#include "krk/backend.hpp"
#include "krk/engine.hpp"

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
    int expert_cache_slots = 0; // MoE resident (layer, expert) slot cap, 0 = auto
    int expert_warm_mb = -1;     // MoE WARM tier in pageable host RAM (MiB): <0 auto, 0 off
    bool expert_warm_prefetch = false; // fill WARM at load instead of on demand
    int draft_tokens = 4;       // speculative decoding window
    bool cpu = false;
    bool greedy = false;
    bool chat = false;
    bool info = false;
    bool bench = false;
    bool profile = false;
    bool expert_scan = false;   // measure routing and write an expert index
    bool expert_stub = false;   // run the routers with every expert FFN removed
    int debug_topk = 0; // stderr dump of top-k logprobs per token
    bool verbose = false;
    bool interactive = true;
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
        "  --prompt STR          prompt to complete (omit for interactive mode)\n"
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
        "  --chat                wrap the prompt in a ChatML template\n"
        "  --system STR          system message for --chat\n"
        "  --cpu                 use the scalar reference backend\n"
        "  --device N            HIP device index (default 0)\n"
        "  --expert-cache-mb N   MoE expert residency budget in MiB (0 = auto)\n"
        "  --expert-cache-slots N  cap resident (layer, expert) slots (0 = auto)\n"
        "  --expert-warm-mb N     WARM expert cache in pageable host RAM. Cold\n"
        "                        reads land here before VRAM, VRAM evictions stay\n"
        "                        here, and WARM eviction never writes to the file\n"
        "                        (MiB; <0 auto = the expert set, capped by half of\n"
        "                        free RAM after a reserve; 0 off).\n"
        "                        --expert-l2-mb is the old spelling of this flag.\n"
        "  --expert-warm-prefetch  fill WARM at load instead of on demand (costs\n"
        "                        RAM and time before the first token)\n"
        "  --draft MODEL         draft model for greedy speculative decoding\n"
        "  --draft-tokens N      speculation window (default 4)\n"
        "  --info                print model and device info, then exit\n"
        "  --bench               prefill/decode benchmark on a fixed prompt\n"
        "  --profile             per-op device timeline for the last decode\n"
        "                        step(s), with the idle gap before each op\n"
        "                        (KRK_PROFILE_STEPS/FROM/OPS tune the window)\n"
        "  --debug-topk N        dump the top-N candidates and their log-probs\n"
        "                        for every token to stderr (at full precision)\n"
        "  --expert-scan        measure which experts the routers actually pick\n"
        "                        and write <model>.krakenexperts.json\n"
        "  --expert-stub        with --expert-scan: run only the routers, with\n"
        "                        every expert FFN skipped (no expert weights\n"
        "                        are read). The list is then approximate; the\n"
        "                        two modes exist so they can be compared\n"
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
        else if (f == "--prompt" || f == "-p") a->prompt = next("--prompt");
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
        else if (f == "--expert-cache-mb")
            a->expert_cache_mb = std::atoi(next("--expert-cache-mb"));
        else if (f == "--expert-cache-slots")
            a->expert_cache_slots = std::atoi(next("--expert-cache-slots"));
        else if (f == "--expert-warm-mb" || f == "--expert-l2-mb")
            a->expert_warm_mb = std::atoi(next("--expert-warm-mb"));
        else if (f == "--expert-warm-prefetch")
            a->expert_warm_prefetch = std::atoi(next("--expert-warm-prefetch")) != 0;
        else if (f == "--greedy") a->greedy = true;
        else if (f == "--chat") a->chat = true;
        else if (f == "--cpu") a->cpu = true;
        else if (f == "--info") a->info = true;
        else if (f == "--bench") a->bench = true;
        else if (f == "--profile") a->profile = true;
        else if (f == "--expert-scan") a->expert_scan = true;
        else if (f == "--expert-stub") a->expert_stub = true;
        else if (f == "--debug-topk")
            a->debug_topk = std::atoi(next("--debug-topk"));
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
    const f64 dtps = r.decode_ms > 0 ? r.generated / (r.decode_ms / 1000.0) : 0.0;
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
    std::fprintf(out, "[stats ] decode    %d tok in %.1f ms = %.1f tok/s (%.1f ms/tok)\n",
                 r.generated, r.decode_ms, dtps,
                 r.generated > 0 ? r.decode_ms / r.generated : 0.0);
    if (dev_total > 0)
        std::fprintf(out,
                     "[stats ] device    %.2f GiB in use of %.2f GiB (%.2f GiB free)\n",
                     static_cast<f64>(dev_total - dev_free) / 1073741824.0,
                     static_cast<f64>(dev_total) / 1073741824.0,
                     static_cast<f64>(dev_free) / 1073741824.0);
    if (!mc.is_moe) return;
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
    const f64 decode_tps = r.generated > 0 ? r.generated / (r.decode_ms / 1000.0) : 0;
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
    cfg.expert_cache_slots = a.expert_cache_slots;
    cfg.expert_warm_mb = a.expert_warm_mb;
    cfg.expert_warm_prefetch = a.expert_warm_prefetch;

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
        std::fprintf(stderr,
                     "kraken: --expert-scan needs --prompt (or --bench) so there "
                     "are token positions to measure the routers on\n");
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
        engine.shutdown();
        ph.mark("engine.shutdown");
        delete be;
        ph.mark("delete be -> exit");
        return rc;
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
        p.sink.fn = sink_cb;
        p.sink.user = &sink;

        GenerateResult r;
        if (!engine.generate(p, &r)) return false;
        print_run_stats(stderr, engine, r, load_ms,
                        be->device_free_bytes(), be->device_total_bytes());
        return true;
    };

    int rc = 0;
    if (!a.prompt.empty()) {
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
