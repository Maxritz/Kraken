// http.cpp — portable HTTP/1.1 listener and SSE framing.
#include "krk/http.hpp"

#include <atomic>
#include <cstring>
#include <thread>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
constexpr socket_t kInvalidSocket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
constexpr socket_t kInvalidSocket = -1;
#endif

namespace krk {

namespace {

constexpr size_t kMaxHeaderBytes = 64 * 1024;
constexpr size_t kMaxBodyBytes = 8 * 1024 * 1024;

void close_socket(socket_t s) {
#ifdef _WIN32
    ::closesocket(s);
#else
    ::close(s);
#endif
}

// Sends everything; false on a hard error (EPIPE / ECONNRESET).
bool send_all(socket_t s, const char *data, size_t n) {
    size_t off = 0;
    while (off < n) {
#ifdef _WIN32
        const int r = ::send(s, data + off, static_cast<int>(n - off), 0);
        if (r <= 0) return false;
        off += static_cast<size_t>(r);
#else
        const ssize_t r = ::send(s, data + off, n - off, MSG_NOSIGNAL);
        if (r <= 0) {
            if (r < 0 && (errno == EINTR)) continue;
            return false;
        }
        off += static_cast<size_t>(r);
#endif
    }
    return true;
}

// Reads with a hard cap; returns false on error, true on clean EOF.
bool recv_some(socket_t s, std::string *buf) {
    char tmp[8192];
#ifdef _WIN32
    const int r = ::recv(s, tmp, sizeof(tmp), 0);
    if (r <= 0) return false;
    buf->append(tmp, static_cast<size_t>(r));
    return true;
#else
    ssize_t r;
    do {
        r = ::recv(s, tmp, sizeof(tmp), 0);
    } while (r < 0 && errno == EINTR);
    if (r <= 0) return false;
    buf->append(tmp, static_cast<size_t>(r));
    return true;
#endif
}

std::string lower(std::string s) {
    for (char &c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}

std::string trim_copy(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) --b;
    return s.substr(a, b - a);
}

const char *status_text(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default: return "OK";
    }
}

// A streaming connection bound to one socket.
class SocketConn : public HttpConn {
public:
    explicit SocketConn(socket_t s) : s_(s) {}

    bool write_raw(const char *data, size_t n) override { return send_all(s_, data, n); }
    bool write_chunk(const char *data, size_t n) {
        char head[32];
        const int h = std::snprintf(head, sizeof(head), "%zX\r\n", n);
        if (!send_all(s_, head, static_cast<size_t>(h))) return false;
        if (n && !send_all(s_, data, n)) return false;
        return send_all(s_, "\r\n", 2);
    }
    bool write_last_chunk() { return send_all(s_, "0\r\n\r\n", 5); }

private:
    socket_t s_;
};

} // namespace

// ---------------------------------------------------------------------------
// HttpRequest
// ---------------------------------------------------------------------------

std::string HttpRequest::header(const std::string &lower_name) const {
    auto it = headers.find(lower_name);
    return it == headers.end() ? std::string() : it->second;
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

bool http_sse_begin(HttpConn &conn) {
    static const char head[] = "HTTP/1.1 200 OK\r\n"
                               "Content-Type: text/event-stream\r\n"
                               "Cache-Control: no-cache\r\n"
                               "Connection: close\r\n"
                               "Transfer-Encoding: chunked\r\n"
                               "\r\n";
    return conn.write_raw(head, sizeof(head) - 1);
}

bool http_sse_event(HttpConn &conn, const std::string &data) {
    std::string ev;
    ev.reserve(data.size() + 8);
    ev += "data: ";
    ev += data;
    ev += "\n\n";
    SocketConn *sc = dynamic_cast<SocketConn *>(&conn);
    if (sc) return sc->write_chunk(ev.data(), ev.size());
    return conn.write_raw(ev.data(), ev.size());
}

bool http_sse_end(HttpConn &conn) {
    SocketConn *sc = dynamic_cast<SocketConn *>(&conn);
    if (sc) {
        const std::string done = "data: [DONE]\n\n";
        if (!sc->write_chunk(done.data(), done.size())) return false;
        return sc->write_last_chunk();
    }
    const std::string done = "data: [DONE]\n\n";
    return conn.write_raw(done.data(), done.size());
}

bool http_write_response(HttpConn &conn, const HttpResponse &r) {
    std::string head = format("HTTP/1.1 %d %s\r\n", r.status, status_text(r.status));
    head += "Content-Type: " + r.content_type + "\r\n";
    head += "Content-Length: " + std::to_string(r.body.size()) + "\r\n";
    for (const auto &kv : r.extra_headers)
        head += kv.first + ": " + kv.second + "\r\n";
    if (r.status >= 400) head += "Connection: close\r\n";
    head += "\r\n";
    if (!conn.write_raw(head.data(), head.size())) return false;
    if (!r.body.empty()) return conn.write_raw(r.body.data(), r.body.size());
    return true;
}

std::string http_url_decode(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size()) {
            const auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            const int h = hex(s[i + 1]), l = hex(s[i + 2]);
            if (h >= 0 && l >= 0) {
                out += static_cast<char>((h << 4) | l);
                i += 2;
                continue;
            }
        }
        out += s[i] == '+' ? ' ' : s[i];
    }
    return out;
}

// ---------------------------------------------------------------------------
// HttpServer
// ---------------------------------------------------------------------------

#ifdef _WIN32
namespace {
bool winsock_init(void **token) {
    WSADATA *d = new WSADATA();
    if (WSAStartup(MAKEWORD(2, 2), d) != 0) {
        delete d;
        return false;
    }
    *token = d;
    return true;
}
void winsock_free(void *token) {
    if (token) {
        WSACleanup();
        delete static_cast<WSADATA *>(token);
    }
}
} // namespace
#endif

bool HttpServer::listen(const std::string &host, int port, std::string *err) {
#ifdef _WIN32
    if (!winsock_init(&wsa_)) {
        if (err) *err = "WSAStartup failed";
        return false;
    }
#endif
    listen_fd_ = static_cast<int>(::socket(AF_INET, SOCK_STREAM, 0));
    if (listen_fd_ == static_cast<int>(kInvalidSocket)) {
        if (err) *err = "socket() failed";
        return false;
    }
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&one),
                 sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u16>(port));
    if (host.empty() || host == "0.0.0.0" || host == "*") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
            if (err) *err = "invalid host '" + host + "'";
            close_socket(listen_fd_);
            listen_fd_ = -1;
            return false;
        }
    }
    if (::bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        if (err) *err = format("bind to %s:%d failed", host.c_str(), port);
        close_socket(listen_fd_);
        listen_fd_ = -1;
        return false;
    }
    if (::listen(listen_fd_, 16) != 0) {
        if (err) *err = "listen() failed";
        close_socket(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    // discover the bound port (ephemeral case)
    sockaddr_in bound{};
    socklen_t blen = sizeof(bound);
    if (::getsockname(listen_fd_, reinterpret_cast<sockaddr *>(&bound), &blen) == 0) {
        port_ = ntohs(bound.sin_port);
    } else {
        port_ = port;
    }
    return true;
}

void HttpServer::serve_client(int fd, HttpHandler handler) {
    socket_t s = static_cast<socket_t>(fd);
    SocketConn conn(s);

    // Read until the end of the headers.
    std::string buf;
    size_t header_end = std::string::npos;
    while (true) {
        const size_t p = buf.find("\r\n\r\n");
        if (p != std::string::npos) {
            header_end = p;
            break;
        }
        if (buf.size() > kMaxHeaderBytes) {
            HttpResponse r;
            r.status = 413;
            r.body = "{\"error\":{\"message\":\"request header too large\"}}";
            http_write_response(conn, r);
            close_socket(s);
            return;
        }
        if (!recv_some(s, &buf)) {
            close_socket(s);
            return;
        }
    }

    // ---- request line + headers ------------------------------------------
    // NOTE: the loop runs to header_end + 2 because the LAST header line's
    // '\n' terminator sits at header_end + 1 (it is the first byte of the
    // "\r\n\r\n" pair's second half). Scanning only to header_end would drop
    // the final header — in practice Content-Length — and lose the body.
    HttpRequest req;
    {
        size_t line_start = 0;
        bool first = true;
        for (size_t i = 0; i < header_end + 2 && i < buf.size(); i++) {
            if (buf[i] == '\n') {
                size_t e = i;
                if (e > line_start && buf[e - 1] == '\r') --e;
                const std::string line = buf.substr(line_start, e - line_start);
                line_start = i + 1;
                if (first) {
                    first = false;
                    const size_t sp1 = line.find(' ');
                    const size_t sp2 = line.rfind(' ');
                    if (sp1 == std::string::npos || sp2 == sp1) {
                        HttpResponse r;
                        r.status = 400;
                        r.body = "{\"error\":{\"message\":\"malformed request line\"}}";
                        http_write_response(conn, r);
                        close_socket(s);
                        return;
                    }
                    req.method = line.substr(0, sp1);
                    req.target = line.substr(sp1 + 1, sp2 - sp1 - 1);
                } else {
                    const size_t colon = line.find(':');
                    if (colon != std::string::npos) {
                        const std::string key = lower(trim_copy(line.substr(0, colon)));
                        const std::string val = trim_copy(line.substr(colon + 1));
                        req.headers[key] = val;
                    }
                }
            }
        }
        const size_t q = req.target.find('?');
        if (q == std::string::npos) {
            req.path = req.target;
        } else {
            req.path = req.target.substr(0, q);
            req.query = req.target.substr(q + 1);
        }
    }

    // ---- body (Content-Length only; no chunked request bodies) -----------
    size_t body_start = header_end + 4;
    const std::string cl = req.header("content-length");
    if (!cl.empty()) {
        const long long want = std::atoll(cl.c_str());
        if (want < 0 || static_cast<size_t>(want) > kMaxBodyBytes) {
            HttpResponse r;
            r.status = 413;
            r.body = "{\"error\":{\"message\":\"request body too large\"}}";
            http_write_response(conn, r);
            close_socket(s);
            return;
        }
        while (buf.size() - body_start < static_cast<size_t>(want)) {
            if (!recv_some(s, &buf)) {
                close_socket(s);
                return;
            }
        }
        req.body = buf.substr(body_start, static_cast<size_t>(want));
    }

    // ---- dispatch ---------------------------------------------------------
    HttpResponse res;
    const bool handled = handler(req, res, conn);
    if (handled) {
        // Streaming handlers wrote the response themselves (they can tell by
        // content_type == "text/event-stream"); everything else goes through
        // the buffered writer.
        if (res.content_type != "text/event-stream") {
            http_write_response(conn, res);
        }
    } else {
        HttpResponse r;
        r.status = 400;
        r.body = "{\"error\":{\"message\":\"bad request\"}}";
        http_write_response(conn, r);
    }

    // Graceful close: half-close, drain briefly so no response bytes are lost.
    ::shutdown(s,
#ifdef _WIN32
               SD_SEND
#else
               SHUT_WR
#endif
    );
    std::string drain;
    for (int i = 0; i < 3 && recv_some(s, &drain); i++) {
    }
    close_socket(s);
}

void HttpServer::run(HttpHandler handler) {
    running_.store(true);
    while (running_.load()) {
        sockaddr_in peer{};
        socklen_t plen = sizeof(peer);
#ifdef _WIN32
        const SOCKET c = ::accept(listen_fd_, reinterpret_cast<sockaddr *>(&peer), &plen);
        const bool failed = c == INVALID_SOCKET;
#else
        const int c = static_cast<int>(::accept(listen_fd_, reinterpret_cast<sockaddr *>(&peer), &plen));
        const bool failed = c < 0;
#endif
        if (failed) {
            if (!running_.load()) break; // stop() closed the socket
            // transient error: brief backoff and continue
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        std::thread(&HttpServer::serve_client, this, static_cast<int>(c), handler)
            .detach();
    }
    running_.store(false);
}

void HttpServer::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
        if (listen_fd_ < 0) return;
    }
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_,
#ifdef _WIN32
                   SD_BOTH
#else
                   SHUT_RDWR
#endif
        );
        close_socket(listen_fd_);
        listen_fd_ = -1;
    }
#ifdef _WIN32
    winsock_free(wsa_);
    wsa_ = nullptr;
#endif
}

} // namespace krk
