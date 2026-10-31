// ============================================================================
//  server.hpp — the OpenAI-compatible HTTP API layer.
//
//  OpenAiService maps HttpRequest/HttpResponse onto a loaded Engine:
//
//    GET  /health               liveness + loaded model summary
//    GET  /v1/models            model metadata (OpenAI list shape)
//    POST /v1/completions       text completion (SSE when "stream": true)
//    POST /v1/chat/completions  chat completion over the ChatML template
//
//  Extracted into a header so the deterministic test suite can drive the
//  exact same request-handling code the server binary runs — the suite
//  builds a listener on an ephemeral port and issues real HTTP requests.
// ============================================================================
#ifndef KRK_SERVER_HPP
#define KRK_SERVER_HPP

#include <atomic>
#include <memory>
#include <mutex>

#include "krk/engine.hpp"
#include "krk/http.hpp"
#include "krk/json.hpp"

namespace krk {

class OpenAiService {
public:
    // `service_mutex` serializes generate() across connections; pass the same
    // mutex for every service instance sharing one Engine.
    OpenAiService(Engine *engine, f32 default_temp, std::mutex *service_mutex);

    // The handler installed in HttpServer::run().
    bool handle(const HttpRequest &req, HttpResponse &res, HttpConn &conn);

private:
    bool handle_health(HttpResponse &res);
    bool handle_models(HttpResponse &res);
    bool handle_completion(const HttpRequest &req, HttpResponse &res, HttpConn &conn);

    // The token ids the KV cache's rows hold, IN ORDER, from the last request.
    // Not just its prompt: the generated tokens are rows in the cache too, so a
    // second chat turn -- whose prompt is the first turn's prompt plus that
    // turn's answer plus a new question -- prefills only the new question
    // instead of re-running the answer through the model. Every entry is a row
    // that exists: the vector is cut to engine kv_pos, so a token the cache
    // never received can never be claimed. Cleared on any failure.
    std::vector<i32> prefix_ids_;
    // Longest common prefix length, in tokens, between `prompt` and the cache.
    i64 reusable_prefix(const std::vector<i32> &prompt) const;

    Engine *engine_;
    f32 default_temp_;
    std::mutex *mu_;
    std::atomic<u64> seq_{1};
};

} // namespace krk

#endif // KRK_SERVER_HPP
