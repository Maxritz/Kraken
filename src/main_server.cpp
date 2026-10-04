// main_server.cpp — kraken-server entry point.
//
// Usage:
//   kraken-server --model model.gguf [--host 0.0.0.0] [--port 8080]
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
    int port = 8080;
    int ctx = 4096;
    int chunk = 256;
    int device = 0;
    int threads = 0;
    int expert_cache_mb = 0;
    int expert_cache_slots = 0;
    int expert_warm_mb = -1; // WARM expert cache (MiB): <0 auto, 0 off
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
        "  --port N              TCP port (default 8080)\n"
        "  --ctx N               KV capacity in tokens (default 4096)\n"
        "  --chunk N             prefill batch size (default 256)\n"
        "  --threads N           CPU backend threads (reserved)\n"
        "  --device N            HIP device index\n"
        "  --expert-cache-mb N   MoE expert residency budget in MiB\n"
        "  --expert-cache-slots N  cap resident (layer, expert) slots\n"
        "  --expert-warm-mb N     WARM expert cache in pageable host RAM (MiB;\n"
        "                        <0 auto, 0 off); --expert-l2-mb also accepted\n"
        "  --draft MODEL         draft model for greedy speculative decoding\n"
        "  --draft-tokens N      speculation window (default 4)\n"
        "  --cpu                 use the scalar reference backend\n"
        "  -v                    verbose logging\n"
        "  -h, --help            this message\n",
        kEngineName, kEngineVersion);
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
        else if (f == "--port") a->port = std::atoi(next("--port"));
        else if (f == "--ctx") a->ctx = std::atoi(next("--ctx"));
        else if (f == "--chunk") a->chunk = std::atoi(next("--chunk"));
        else if (f == "--threads") a->threads = std::atoi(next("--threads"));
        else if (f == "--device") a->device = std::atoi(next("--device"));
        else if (f == "--expert-cache-mb")
            a->expert_cache_mb = std::atoi(next("--expert-cache-mb"));
        else if (f == "--expert-cache-slots")
            a->expert_cache_slots = std::atoi(next("--expert-cache-slots"));
        else if (f == "--expert-warm-mb" || f == "--expert-l2-mb")
            a->expert_warm_mb = std::atoi(next("--expert-warm-mb"));
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
    cfg.expert_cache_mb = a.expert_cache_mb;
    cfg.expert_cache_slots = a.expert_cache_slots;
    cfg.expert_warm_mb = a.expert_warm_mb;

    Engine engine;
    if (!engine.init(be, cfg, &err)) {
        std::fprintf(stderr, "kraken-server: %s\n", err.c_str());
        delete be;
        return 1;
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

    HttpServer server;
    if (!server.listen(a.host, a.port, &err)) {
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
