// ============================================================================
//  http.hpp — a portable HTTP/1.1 server: listener, request parsing, responses
//  and a chunked `text/event-stream` helper for OpenAI-style SSE streaming.
//
//  Thread-per-connection: the accept loop spawns a detached thread per client
//  and hands it a handler callback. The engine is NOT thread-safe for
//  concurrent generate() calls, so the server serializes generation behind a
//  mutex (see main_server.cpp); requests may still connect at any time.
// ============================================================================
#ifndef KRK_HTTP_HPP
#define KRK_HTTP_HPP

#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <thread>

#include "krk/common.hpp"

namespace krk {

struct HttpRequest {
    std::string method;
    std::string target;   // as sent, e.g. "/v1/completions?x=1"
    std::string path;     // target without the query
    std::string query;
    std::string body;

    // Header lookup, lower-case keys, e.g. http_headers("content-type").
    // Returns "" when absent. Header order and duplicates collapse to the
    // last value (correct for the headers this API consumes).
    std::string header(const std::string &lower_name) const;

    std::map<std::string, std::string> headers; // keys lower-cased
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;
    // Extra headers, e.g. {"Connection", "close"}.
    std::vector<std::pair<std::string, std::string>> extra_headers;
};

// Handler receives the parsed request and fills the response. Returning
// false abandons the connection without a response (used for protocol
// errors). For streaming endpoints the handler can use conn.sse().
using HttpHandler =
    std::function<bool(const HttpRequest &, HttpResponse &, class HttpConn &)>;

// One accepted client. write_raw/sse_* are safe to call from the handler
// while it holds the connection.
class HttpConn {
public:
    virtual ~HttpConn() = default;
    // Blocking write; false when the peer has gone.
    virtual bool write_raw(const char *data, size_t n) = 0;
};

class HttpServer {
public:
    HttpServer() = default;
    ~HttpServer() { stop(); }
    HttpServer(const HttpServer &) = delete;
    HttpServer &operator=(const HttpServer &) = delete;

    // Binds and listens. port 0 picks an ephemeral port (reported by port()).
    // host "" or "0.0.0.0" binds all interfaces.
    bool listen(const std::string &host, int port, std::string *err = nullptr);

    // The actual port after listen() (relevant when port 0 was requested).
    int port() const { return port_; }

    // Blocking accept loop. Returns when stop() is called (from any thread).
    void run(HttpHandler handler);

    // Shuts the listener down and unblocks run(). Not instant: the OS may
    // still hold the socket for a moment.
    void stop();

    bool running() const { return running_.load(); }

private:
    void serve_client(int fd, HttpHandler handler);

    int listen_fd_ = -1;
    int port_ = -1;
    std::atomic<bool> running_{false};
#ifdef _WIN32
    void *wsa_ = nullptr;
#endif
};

// ---------------------------------------------------------------------------
// SSE framing helpers
// ---------------------------------------------------------------------------

// Emits the HTTP response head for an event stream (200, chunked, SSE).
bool http_sse_begin(HttpConn &conn);
// One `data:` event followed by a blank line, framed as one HTTP chunk.
bool http_sse_event(HttpConn &conn, const std::string &data);
// `data: [DONE]` and the terminating zero chunk.
bool http_sse_end(HttpConn &conn);

// Writes a complete buffered response (head + body + framing).
bool http_write_response(HttpConn &conn, const HttpResponse &r);

// Percent-decodes a query component.
std::string http_url_decode(const std::string &s);

} // namespace krk

#endif // KRK_HTTP_HPP
