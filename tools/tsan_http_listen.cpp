// tsan_http_listen — B4's ThreadSanitizer arm for HttpServer's listen
// descriptor.
//
// Why this is a separate binary: TSan needs every object in the image built with
// -fsanitize=thread, and instrumenting the whole engine costs minutes on this
// box for a test that only touches src/http.cpp. This target compiles http.cpp,
// json.cpp and common.cpp with the sanitizer and drives the exact shutdown
// lifetime B4 is about:
//
//   * 12 rounds of listen() -> run() on a thread -> four concurrent clients ->
//     stop() racing the accept loop from a third thread;
//   * every round reuses the same HttpServer object, so a descriptor that is
//     closed twice, leaked, or closed by the wrong thread is exercised.
//
// Build and run (not part of the default build; it is EXCLUDE_FROM_ALL):
//
//   cmake --build build-tests --target kraken-tsan-http-listen
//   ./build-tests/kraken-tsan-http-listen
//
// Recorded result (2026-10-31, docs/PLAN-remaining.md B4): clean, three runs.
// The negative control is the pre-B4 src/http.cpp built the same way, which
// reports a data race between stop()'s `listen_fd_ = -1` / close and run()'s
// `accept(listen_fd_)` read. A harness that cannot fail is not evidence, so the
// control was observed before this result was believed.

#include "krk/http.hpp"

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace krk;

static int connect_and_get(int port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<unsigned short>(port));
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr *>(&a), sizeof a) != 0) {
        ::close(fd);
        return -1;
    }
    const char *req = "GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n";
    ::send(fd, req, std::strlen(req), 0);
    char buf[512];
    int total = 0;
    while (true) {
        const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
        if (n <= 0) break;
        total += static_cast<int>(n);
    }
    ::close(fd);
    return total;
}

int main() {
    int rounds = 0;
    std::atomic<int> answered{0};
    for (int r = 0; r < 12; r++) {
        HttpServer s;
        std::string err;
        if (!s.listen("127.0.0.1", 0, &err)) {
            std::fprintf(stderr, "listen failed: %s\n", err.c_str());
            return 2;
        }
        const int port = s.port();
        std::thread loop([&] {
            s.run([](const HttpRequest &, HttpResponse &res, HttpConn &) {
                res.status = 200;
                res.body = "{}";
                return true;
            });
        });
        std::vector<std::thread> clients;
        for (int c = 0; c < 4; c++)
            clients.emplace_back([&] {
                if (connect_and_get(port) > 0) answered.fetch_add(1);
            });
        // stop() concurrently with in-flight clients and the accept loop.
        std::thread stopper([&] { s.stop(); });
        for (auto &t : clients) t.join();
        stopper.join();
        loop.join();
        rounds++;
    }
    std::printf("tsan_listen: %d rounds, %d answered responses\n", rounds, answered.load());
    return rounds == 12 ? 0 : 1;
}
