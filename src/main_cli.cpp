// main_cli.cpp — the kraken command line front end.
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
    int expert_l2_mb = 0;       // MoE expert second tier in host RAM (MiB), 0 = off
    int draft_tokens = 4;       // speculative decoding window
    bool cpu = false;
    bool greedy = false;
    bool chat = false;
    bool info = false;
    bool bench = false;
    bool profile = false;
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
        "  --expert-l2-mb N       pinned host-RAM tier for evicted experts (MiB, 0 = off)\n"
        "  --draft MODEL         draft model for greedy speculative decoding\n"
        "  --draft-tokens N      speculation window (default 4)\n"
        "  --info                print model and device info, then exit\n"
        "  --bench               prefill/decode benchmark on a fixed prompt\n"
        "  --profile             per-op device timeline for the last decode\n"
        "                        step(s), with the idle gap before each op\n"
        "                        (KRK_PROFILE_STEPS/FROM/OPS tune the window)\n"
        "  --debug-topk N        dump the top-N candidates and their log-probs\n"
        "                        for every token to stderr (at full precision)\n"
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
        else if (f == "--expert-l2-mb")
            a->expert_l2_mb = std::atoi(next("--expert-l2-mb"));
        else if (f == "--greedy") a->greedy = true;
        else if (f == "--chat") a->chat = true;
        else if (f == "--cpu") a->cpu = true;
        else if (f == "--info") a->info = true;
        else if (f == "--bench") a->bench = true;
        else if (f == "--profile") a->profile = true;
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

int run_bench(Engine &engine, const Args &a) {
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
    usage.report(stdout);
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
        if (ec.host_budget_bytes() > 0)
            std::printf("expert L2      %.0f MiB pinned host, %zu expert(s) demoted, "
                        "%llu promoted, %.0f MiB held\n",
                        static_cast<f64>(ec.host_budget_bytes()) / (1024.0 * 1024.0),
                        ec.host_slots(),
                        static_cast<unsigned long long>(ec.promotions()),
                        static_cast<f64>(ec.host_resident_bytes()) / (1024.0 * 1024.0));
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
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
    cfg.expert_l2_mb = a.expert_l2_mb;

    Engine engine;
    if (!engine.init(be, cfg, &err)) {
        std::fprintf(stderr, "kraken: %s\n", err.c_str());
        delete be;
        return 1;
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
            if (ec.host_budget_bytes() > 0)
                std::printf("expert L2      %.0f MiB pinned host tier "
                            "(evicted experts demote, not drop)\n",
                            static_cast<f64>(ec.host_budget_bytes()) /
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

    if (a.bench) {
        const int rc = run_bench(engine, a);
        engine.shutdown();
        delete be;
        return rc;
    }

    SinkCtx sink;
    sink.chat = a.chat;

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
        const f64 dtps = r.generated > 0 ? r.generated / (r.decode_ms / 1000.0) : 0;
        if (a.verbose)
            std::fprintf(stderr,
                         "\n[%d prompt / %d generated | prefill %.0f ms | decode %.1f tok/s]\n",
                         r.prompt_tokens, r.generated, r.prefill_ms, dtps);
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

    engine.shutdown();
    delete be;
    return rc;
}
