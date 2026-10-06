// server.cpp — OpenAI-compatible request handling over a loaded Engine.
//
// The engine is single-sequence, so generation is serialized behind the
// shared mutex; connections are handled concurrently, generation is not.
// Each request rolls the KV cache back to 0 so no state leaks between calls.
#include "krk/server.hpp"
#include "krk/host_time.hpp"

#include <cstdio>
#include <ctime>

namespace krk {

namespace {

// ---------------------------------------------------------------------------
// request -> engine params
// ---------------------------------------------------------------------------

bool parse_stop_array(const JsonValue &body, std::vector<std::string> *stop) {
    const JsonValue *v = body.find("stop");
    if (!v) return true;
    if (v->is_string()) {
        if (!v->as_string().empty()) stop->push_back(v->as_string());
        return true;
    }
    if (v->is_array()) {
        for (const JsonValue &s : v->items())
            if (s.is_string() && !s.as_string().empty()) stop->push_back(s.as_string());
        return true;
    }
    return false;
}

void apply_sampling(const JsonValue &body, f32 default_temp, SampleParams *sp) {
    const f64 temp = body.get_number("temperature", default_temp);
    sp->temp = static_cast<f32>(temp);
    sp->greedy = temp <= 0.0;
    i64 top_k = body.get_int("top_k", 0);
    if (top_k <= 0) top_k = 40; // engine default
    sp->top_k = static_cast<i32>(top_k);
    f64 top_p = body.get_number("top_p", 0.95);
    if (top_p <= 0.0 || top_p > 1.0) top_p = 0.95;
    sp->top_p = static_cast<f32>(top_p);
    f64 min_p = body.get_number("min_p", 0.0);
    if (min_p < 0.0 || min_p >= 1.0) min_p = 0.0;
    sp->min_p = static_cast<f32>(min_p);
    f64 rp = body.get_number("repeat_penalty", 1.0);
    if (rp < 1.0) rp = 1.0;
    sp->repeat_penalty = static_cast<f32>(rp);
    i64 last_n = body.get_int("repeat_last_n", 64);
    if (last_n <= 0) last_n = 64;
    sp->repeat_last_n = static_cast<i32>(last_n);
    f64 seed = body.get_number("seed", 0.0);
    sp->seed = seed > 0.0 ? static_cast<u64>(seed) : 0;
}

// Flattens OpenAI `messages` into a prompt using the ChatML template
// (same shape the CLI's --chat path renders).
bool render_chat(const JsonValue &body, std::string *prompt) {
    const JsonValue *msgs = body.find("messages");
    if (!msgs || !msgs->is_array()) return false;
    std::string system, transcript;
    for (const JsonValue &m : msgs->items()) {
        std::string role, content;
        m.get_string("role", &role);
        if (!m.get_string("content", &content)) {
            // tolerate null content (some clients send it for tool stubs)
            const JsonValue *c = m.find("content");
            if (c && c->is_null()) continue;
            return false;
        }
        if (role == "system") {
            if (!system.empty()) system += "\n";
            system += content;
        } else if (role == "user") {
            transcript += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (role == "assistant") {
            transcript += "<|im_start|>assistant\n" + content + "<|im_end|>\n";
        }
        // unknown roles are skipped rather than rejected
    }
    prompt->clear();
    if (!system.empty()) *prompt += "<|im_start|>system\n" + system + "<|im_end|>\n";
    *prompt += transcript;
    *prompt += "<|im_start|>assistant\n";
    return true;
}

const char *finish_name(int finish) {
    switch (finish) {
        case FinishStop:
        case FinishEos: return "stop";
        case FinishContext: return "length";
        default: return "length";
    }
}

i64 epoch_now() { return static_cast<i64>(std::time(nullptr)); }

std::string completion_id() {
    return format("cmpl-kraken-%d", static_cast<int>(now_ms() % 100000000));
}

JsonValue error_body(const std::string &message, const char *type = "invalid_request_error") {
    JsonValue e = JsonValue::object();
    e.set("message", message);
    e.set("type", type);
    JsonValue root = JsonValue::object();
    root.set("error", std::move(e));
    return root;
}

HttpResponse json_response(int status, JsonValue body) {
    HttpResponse r;
    r.status = status;
    r.body = body.dump();
    return r;
}

std::string model_display_name(const ModelConfig &cfg) {
    return cfg.name.empty() ? std::string("kraken-local") : cfg.name;
}

// ---------------------------------------------------------------------------
// streaming
// ---------------------------------------------------------------------------

struct StreamState {
    HttpConn *conn = nullptr;
    bool chat = false;
    std::string id;
    std::string model;
    bool failed = false;
};

JsonValue chunk_event(const StreamState *st, JsonValue delta, JsonValue finish_reason) {
    JsonValue ev = JsonValue::object();
    ev.set("id", st->id);
    ev.set("object", st->chat ? "chat.completion.chunk" : "text_completion");
    ev.set("created", epoch_now());
    ev.set("model", st->model);
    JsonValue choice = JsonValue::object();
    choice.set("index", i64(0));
    choice.set("delta", std::move(delta));
    choice.set("finish_reason", std::move(finish_reason));
    JsonValue choices = JsonValue::array();
    choices.push_back(std::move(choice));
    ev.set("choices", std::move(choices));
    return ev;
}

// Emits one streaming delta per generated piece.
void stream_cb(void *user, const char *text, i32 token, bool done) {
    StreamState *st = static_cast<StreamState *>(user);
    (void)token;
    if (done || st->failed) return;
    if (!text) return;
    JsonValue delta = JsonValue::object();
    if (st->chat) delta.set("role", "assistant");
    delta.set("content", std::string(text));
    const JsonValue ev =
        chunk_event(st, std::move(delta), JsonValue(nullptr));
    if (!http_sse_event(*st->conn, ev.dump())) st->failed = true;
}

// Per-request timing. The server printed none at all before this, so a slow
// request could not be attributed from its own logs. It is opt-in with the same
// switch as the CLI (--profile / KRK_PROFILE), and the host-phase table is
// cumulative across requests, which is what a server wants: one request tells
// you little, the shape of a hundred is the number to act on.
void report_request_time(const GenerateResult &r) {
    if (!HostTime::get().on()) return;
    std::fprintf(stderr,
                 "[stats ] request   prefill %d tok in %.1f ms | decode %d tok "
                 "in %.1f ms over %lld steps\n",
                 r.prompt_tokens, r.prefill_ms, r.generated, r.decode_ms,
                 static_cast<long long>(r.decode_steps));
    HostTime::get().report(stderr);
}

} // namespace

// ---------------------------------------------------------------------------
// OpenAiService
// ---------------------------------------------------------------------------

OpenAiService::OpenAiService(Engine *engine, f32 default_temp, std::mutex *service_mutex)
    : engine_(engine), default_temp_(default_temp), mu_(service_mutex) {}

bool OpenAiService::handle(const HttpRequest &req, HttpResponse &res, HttpConn &conn) {
    if (req.method != "GET" && req.method != "POST") {
        res = json_response(405, error_body("method not allowed"));
        return true;
    }
    if (req.path == "/health") {
        if (req.method != "GET") {
            res = json_response(405, error_body("use GET"));
            return true;
        }
        return handle_health(res);
    }
    if (req.path == "/v1/models") {
        if (req.method != "GET") {
            res = json_response(405, error_body("use GET"));
            return true;
        }
        return handle_models(res);
    }
    if (req.path == "/v1/completions" || req.path == "/v1/chat/completions") {
        if (req.method != "POST") {
            res = json_response(405, error_body("use POST"));
            return true;
        }
        return handle_completion(req, res, conn);
    }
    res = json_response(404, error_body("unknown endpoint: " + req.path));
    return true;
}

bool OpenAiService::handle_health(HttpResponse &res) {
    const ModelConfig &mc = engine_->model().cfg();
    JsonValue j = JsonValue::object();
    j.set("status", "ok");
    j.set("model", mc.name.empty() ? mc.arch : mc.name);
    j.set("arch", mc.arch);
    j.set("n_ctx", engine_->kv_capacity());
    j.set("kv_pos", engine_->kv_pos());
    j.set("device", engine_->device().name);
    if (mc.is_moe) {
        j.set("moe", true);
        j.set("n_expert", static_cast<i64>(mc.n_expert));
    }
    res = json_response(200, std::move(j));
    return true;
}

bool OpenAiService::handle_models(HttpResponse &res) {
    const ModelConfig &mc = engine_->model().cfg();
    JsonValue m = JsonValue::object();
    m.set("id", model_display_name(mc));
    m.set("object", "model");
    m.set("owned_by", "kraken");
    JsonValue meta = JsonValue::object();
    meta.set("arch", mc.arch);
    meta.set("n_vocab", static_cast<i64>(mc.n_vocab));
    meta.set("n_embd", static_cast<i64>(mc.n_embd));
    meta.set("n_layer", static_cast<i64>(mc.n_layer));
    meta.set("n_head", static_cast<i64>(mc.n_head));
    meta.set("n_head_kv", static_cast<i64>(mc.n_head_kv));
    meta.set("n_ff", static_cast<i64>(mc.n_ff));
    meta.set("n_ctx_train", static_cast<i64>(mc.n_ctx_train));
    if (mc.is_moe) {
        meta.set("n_expert", static_cast<i64>(mc.n_expert));
        meta.set("n_expert_used", static_cast<i64>(mc.n_expert_used));
    }
    m.set("meta", std::move(meta));
    JsonValue data = JsonValue::array();
    data.push_back(std::move(m));
    JsonValue root = JsonValue::object();
    root.set("object", "list");
    root.set("data", std::move(data));
    res = json_response(200, std::move(root));
    return true;
}

bool OpenAiService::handle_completion(const HttpRequest &req, HttpResponse &res,
                                      HttpConn &conn) {
    JsonValue body;
    std::string jerr;
    if (!json_parse(req.body, &body, &jerr) || !body.is_object()) {
        res = json_response(400, error_body("invalid JSON body: " + jerr));
        return true;
    }
    const bool chat = req.path == "/v1/chat/completions";

    std::string prompt;
    if (chat) {
        if (!render_chat(body, &prompt)) {
            res = json_response(
                400, error_body("'messages' must be an array of {role, content} objects"));
            return true;
        }
    } else {
        bool have_prompt = body.get_string("prompt", &prompt);
        if (!have_prompt) {
            // accept a string array, first element (single-sequence engine)
            const JsonValue *p = body.find("prompt");
            if (p && p->is_array() && !p->items().empty() && p->items()[0].is_string())
                prompt = p->items()[0].as_string();
        }
        if (prompt.empty()) {
            res = json_response(400, error_body("'prompt' must be a non-empty string"));
            return true;
        }
    }

    GenerateParams p;
    p.prompt = prompt;
    apply_sampling(body, default_temp_, &p.sampler);
    i64 max_tokens = body.get_int("max_tokens", 256);
    if (max_tokens < 1) max_tokens = 1;
    if (max_tokens > 4096) max_tokens = 4096;
    p.max_tokens = static_cast<i32>(max_tokens);
    if (!parse_stop_array(body, &p.stop)) {
        res = json_response(400, error_body("'stop' must be a string or array of strings"));
        return true;
    }
    // Template markers always end the chat path.
    if (chat) {
        p.stop.push_back("<|im_end|>");
        p.stop.push_back("<|endoftext|>");
    }

    const bool stream = body.get_bool("stream", false);
    const std::string id = completion_id();
    const std::string model_name = model_display_name(engine_->model().cfg());

    GenerateResult r;
    std::lock_guard<std::mutex> lock(*mu_);
    if (!engine_->kv_rollback(0)) {
        res = json_response(500, error_body("context rollback failed", "internal_error"));
        return true;
    }

    if (stream) {
        StreamState st;
        st.conn = &conn;
        st.chat = chat;
        st.id = id;
        st.model = model_name;
        p.sink.fn = stream_cb;
        p.sink.user = &st;
        http_sse_begin(conn);
        const bool ok = engine_->generate(p, &r);
        if (!st.failed) {
            if (ok) {
                const JsonValue ev =
                    chunk_event(&st, JsonValue::object(), JsonValue(finish_name(r.finish)));
                http_sse_event(conn, ev.dump());
            } else {
                http_sse_event(conn, error_body("generation failed", "internal_error").dump());
            }
            http_sse_end(conn);
        }
        res.status = ok ? 200 : 500;
        res.content_type = "text/event-stream";
        report_request_time(r);
        return true;
    }

    const bool ok = engine_->generate(p, &r);
    if (!ok) {
        res = json_response(500, error_body("generation failed", "internal_error"));
        return true;
    }
    report_request_time(r);

    JsonValue root = JsonValue::object();
    root.set("id", id);
    root.set("object", chat ? "chat.completion" : "text_completion");
    root.set("created", epoch_now());
    root.set("model", model_name);
    JsonValue choice = JsonValue::object();
    choice.set("index", i64(0));
    if (chat) {
        JsonValue message = JsonValue::object();
        message.set("role", "assistant");
        message.set("content", r.text);
        choice.set("message", std::move(message));
    } else {
        choice.set("text", r.text);
    }
    choice.set("finish_reason", std::string(finish_name(r.finish)));
    JsonValue choices = JsonValue::array();
    choices.push_back(std::move(choice));
    root.set("choices", std::move(choices));
    JsonValue usage = JsonValue::object();
    usage.set("prompt_tokens", static_cast<i64>(r.prompt_tokens));
    usage.set("completion_tokens", static_cast<i64>(r.generated));
    usage.set("total_tokens", static_cast<i64>(r.prompt_tokens) + r.generated);
    root.set("usage", std::move(usage));
    res = json_response(200, std::move(root));
    return true;
}

} // namespace krk
