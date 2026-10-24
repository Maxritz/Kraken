// main_server.cpp — kraken-server entry point.
//
// Usage:
//   kraken-server --model model.gguf [--host 0.0.0.0] [--port N]
//
// The listen port is the flag, then $KRK_PORT, then 8080 -- a deployment moves
// off a busy port with an environment variable instead of a source edit.
//                 [--ctx N] [--chunk N] [--cpu] [--draft small.gguf] [-v]
//
// See server.cpp for the endpoint implementation.
#include <cstdio>
#include <mutex>
#include <string>

#include "krk/backend.hpp"
#include "krk/engine.hpp"
#include "krk/http.hpp"
#include "krk/server.hpp"

namespace {

using namespace krk;

struct Args {
    std::string model;
    std::string host = "0.0.0.0";
    int port = 8080;        // fallback when neither --port nor KRK_PORT is set
    bool port_set = false;  // distinguishes the flag from the default
    int ctx = 4096;
    int chunk = 256;
    int kv_quant = 0;  // 0 = f16 KV, 1 = int8 (EngineConfig::kv_quant)
    bool prefill_fast = false; // --prefill fast: rows>1 GEMM on the matrix cores
    int device = 0;
    int threads = 0;
    int expert_cache_mb = 0;
    int expert_cache_slots = 0;
    int expert_warm_mb = -1; // WARM expert cache (MiB): <0 auto, 0 off
    int expert_pin_mb = -1;  // pinned prefix of that tier (MiB): <0 no override
    f32 router_resident = -1.0f; // router near-tie margin toward resident experts
    int expert_warmup = -1;  // ranked WARM+VRAM warmup at load: <0 auto, 0 off
    int expert_warmup_ms = 20000; // wall-time ceiling for that phase, 0 = none
    int ram_tier_gb = 0;     // plan the host tiers for this RAM class (GiB)
    // Hybrid CPU + GPU expert compute: non-resident experts run on the host out
    // of the mapping while the device runs the resident ones (see
    // EngineConfig::hybrid_experts). Same split the CLI exposes.
    bool hybrid_experts = false;
    int hybrid_threads = 0;
    int hybrid_max_rows = 8;
    double hybrid_frac = 1.0;
    std::string draft;
    int draft_tokens = 4;
    bool cpu = false;
    bool verbose = false;
    f32 default_temp = 0.8f;
};

void usage() {
    std::printf(
        "%s %s server — OpenAI-compatible HTTP endpoint\n\n"
        "usage: kraken-server --model model.gguf [options]\n\n"
        "  --host HOST           bind address (default 0.0.0.0)\n"
        "  --port N              TCP port (default $KRK_PORT, else 8080)\n"
        "  --ctx N               KV capacity in tokens (default 4096)\n"
        "  --chunk N             prefill batch size (default 256)\n"
        "  --kv-quant f16|int8   K/V cache format (default f16): int8 is 1.8x smaller\n"
        "                        and refuses on a backend whose attention kernels do\n"
        "                        not read the layout\n"
        "  --prefill exact|fast  exact (default) keeps every prompt row's\n"
        "                        arithmetic independent of --chunk; fast takes\n"
        "                        the rows>1 GEMM to the RDNA4 matrix cores\n"
        "                        (2.2-2.4x prompt throughput on a long MoE\n"
        "                        prompt), and the generated text is then not\n"
        "                        chunk-invariant past 8 rows per chunk\n"
        "  --threads N           CPU backend threads (reserved)\n"
        "  --device N            HIP device index\n"
        "  --expert-cache-mb N   MoE expert residency budget in MiB\n"
        "  --expert-cache-slots N  cap resident (layer, expert) slots\n"
        "  --expert-warm-mb N     WARM expert cache in pageable host RAM (MiB;\n"
        "                        <0 auto, 0 off); --expert-l2-mb also accepted\n"
        "  --expert-pin-mb N      pin the first N MiB of that tier, in rank\n"
        "                         order: a promotion out of pinned memory is\n"
        "                         one DMA with no host memcpy in front of it\n"
        "                         (off by default; it is RAM that cannot be\n"
        "                         paged out)\n"
        "  --expert-warmup N      ranked expert warmup at load: -1 auto, 1 force,\n"
        "                        0 off. Stages the hot experts into WARM and\n"
        "                        promotes the top of that order into VRAM.\n"
        "  --expert-warmup-ms N   wall-time ceiling for that phase in ms (0 = none)\n"
        "  --ram-tier N           plan the host tiers for a 16/24/32/48/64/96 GiB\n"
        "                        class of machine (0 = installed RAM)\n"
        "  --hybrid-experts 1    compute non-resident experts on the host while the\n"
        "                        device runs the resident ones (frees VRAM and RAM)\n"
        "  --hybrid-threads N     host threads for that split (0 = min(hw, 8))\n"
        "  --hybrid-max-rows N    widest row count the host arm runs at (default 8)\n"
        "  --hybrid-frac F        share (0..1) of the device tier's overflow the host\n"
        "                        arm takes (resident experts always stay on GPU)\n"
        "  --draft MODEL         draft model for greedy speculative decoding\n"
        "  --draft-tokens N      speculation window (default 4)\n"
        "  --cpu                 use the scalar reference backend\n"
        "  -v                    verbose logging\n"
        "  -h, --help            this message\n",
        kEngineName, kEngineVersion);
}

// A positive port from the environment, or `fallback`. A value that is not a
// port is ignored deliberately: a typo must not make the server refuse to
// start for a reason nobody reads, and 8080 is always a valid answer.
int env_port(const char *name, int fallback) {
    const char *v = std::getenv(name);
    if (!v || !*v) return fallback;
    char *end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (!end || *end != '\0' || n <= 0 || n > 65535) return fallback;
    return static_cast<int>(n);
}

bool parse(int argc, char **argv, Args *a) {
    for (int i = 1; i < argc; i++) {
        const std::string f = argv[i];
        auto next = [&](const char *what) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "kraken-server: %s requires a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (f == "--model" || f == "-m") a->model = next("--model");
        else if (f == "--host") a->host = next("--host");
        else if (f == "--port") {
            a->port = std::atoi(next("--port"));
            a->port_set = true;
        }
        else if (f == "--ctx") a->ctx = std::atoi(next("--ctx"));
        else if (f == "--kv-quant") {
            const std::string v = next("--kv-quant");
            if (v == "int8") a->kv_quant = 1;
            else if (v == "f16") a->kv_quant = 0;
            else {
                std::fprintf(stderr, "kraken-server: --kv-quant needs f16|int8, got '%s'\n",
                             v.c_str());
                return false;
            }
        }
        else if (f == "--chunk") a->chunk = std::atoi(next("--chunk"));
        else if (f == "--threads") a->threads = std::atoi(next("--threads"));
        else if (f == "--device") a->device = std::atoi(next("--device"));
        else if (f == "--expert-cache-mb")
            a->expert_cache_mb = std::atoi(next("--expert-cache-mb"));
        else if (f == "--expert-cache-slots")
            a->expert_cache_slots = std::atoi(next("--expert-cache-slots"));
        else if (f == "--expert-warm-mb" || f == "--expert-l2-mb")
            a->expert_warm_mb = std::atoi(next("--expert-warm-mb"));
        else if (f == "--expert-pin-mb")
            a->expert_pin_mb = std::atoi(next("--expert-pin-mb"));
        else if (f == "--router-resident")
            a->router_resident = static_cast<f32>(std::atof(next("--router-resident")));
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
        else if (f == "--prefill") {
            const std::string v = next("--prefill");
            if (v == "fast") a->prefill_fast = true;
            else if (v == "exact") a->prefill_fast = false;
            else {
                std::fprintf(stderr, "kraken-server: --prefill needs exact|fast, got '%s'\n",
                             v.c_str());
                return false;
            }
        }
        else if (f == "--draft") a->draft = next("--draft");
        else if (f == "--draft-tokens") a->draft_tokens = std::atoi(next("--draft-tokens"));
        else if (f == "--cpu") a->cpu = true;
        else if (f == "-v" || f == "--verbose") a->verbose = true;
        else if (f == "-h" || f == "--help") { usage(); std::exit(0); }
        else {
            std::fprintf(stderr, "kraken-server: unknown option '%s'\n", f.c_str());
            return false;
        }
    }
    if (a->model.empty()) {
        std::fprintf(stderr, "kraken-server: --model is required\n");
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    Args a;
    if (!parse(argc, argv, &a)) return 2;
    log_set_level(a.verbose ? Log::Debug : Log::Info);

    std::string err;
    Backend *be = nullptr;
    if (a.cpu) {
        be = make_cpu_backend();
    } else {
        // Same mapping the CLI uses: the backend reads its GEMM rung switch once
        // at construction, so the env has to be set before make_hip_backend().
        if (a.prefill_fast) {
#ifdef _WIN32
            _putenv_s("KRK_GEMM_PATH", "wmma");
#else
            setenv("KRK_GEMM_PATH", "wmma", 1);
#endif
            std::fprintf(stderr,
                         "kraken-server: --prefill fast takes the rows>1 GEMM to the "
                         "matrix cores; the prefill hash is then not chunk-invariant "
                         "past 8 rows per chunk\n");
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
    cfg.threads = a.threads;
    cfg.expert_cache_mb = a.expert_cache_mb;
    cfg.expert_cache_slots = a.expert_cache_slots;
    cfg.expert_warm_mb = a.expert_warm_mb;
    cfg.expert_pin_mb = a.expert_pin_mb;
    cfg.router_tie_margin = a.router_resident;
    cfg.expert_warmup = a.expert_warmup;
    cfg.expert_warmup_ms = a.expert_warmup_ms;
    cfg.ram_tier_gb = a.ram_tier_gb;
    cfg.hybrid_experts = a.hybrid_experts;
    cfg.hybrid_threads = a.hybrid_threads;
    cfg.hybrid_max_rows = a.hybrid_max_rows;
    cfg.hybrid_frac = static_cast<f32>(a.hybrid_frac);

    Engine engine;
    if (!engine.init(be, cfg, &err)) {
        std::fprintf(stderr, "kraken-server: %s\n", err.c_str());
        // A refused startup must not free the backend: the engine holds
        // pointers into the backend's buffers and its own destructor
        // releases them through that pointer, so deleting the backend here
        // freed it and then unwound past the engine -- SIGSEGV on a path
        // whose whole job is to print one line and stop. `exit` skips it.
        std::exit(1);
    }
    if (!a.draft.empty()) {
        engine.set_draft_window(a.draft_tokens);
        if (!engine.load_draft(a.draft, &err)) {
            std::fprintf(stderr, "kraken-server: draft model: %s\n", err.c_str());
            engine.shutdown();
            delete be;
            return 1;
        }
    }

    // Flag, then KRK_PORT, then the 8080 fallback. `--port 0` still means
    // "ephemeral": the flag sets port_set, so the environment cannot turn an
    // explicit 0 into a fixed port.
    const int port = a.port_set ? a.port : env_port("KRK_PORT", a.port);

    HttpServer server;
    if (!server.listen(a.host, port, &err)) {
        std::fprintf(stderr, "kraken-server: %s\n", err.c_str());
        engine.shutdown();
        delete be;
        return 1;
    }

    std::mutex gen_mu;
    OpenAiService svc(&engine, a.default_temp, &gen_mu);

    std::printf("%s %s | %s\n", kEngineName, kEngineVersion, be->describe().c_str());
    std::printf("listening on http://%s:%d (model: %s, ctx %d)\n", a.host.c_str(),
                server.port(), engine.model().cfg().name.c_str(), a.ctx);
    std::fflush(stdout);

    server.run([&svc](const HttpRequest &req, HttpResponse &res, HttpConn &conn) {
        return svc.handle(req, res, conn);
    });

    // server.run() returns only after stop(), which nothing calls in this
    // process; keep the shape correct for future graceful shutdown.
    engine.shutdown();
    delete be;
    return 0;
}
