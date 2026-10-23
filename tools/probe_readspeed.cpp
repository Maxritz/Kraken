// probe_readspeed.cpp — the copy leg of a cold expert read, measured alone.
//
// The engine's WARM tier fills by copying an expert slice out of the file
// (ReadFile out of the page cache) into pinned host RAM. On Ornith-1.5-122B that
// leg is 64% of the run at ~3.9 GB/s, while the *other* leg -- a kernel reading
// the pinned copy back -- measured 24.3 GiB/s (tools/probe_hostread.hip). Those
// are different operations, so the gap is not a tuning bug; the question this
// probe answers is how much of the 3.9 GB/s is the single-handle, queue-depth-1
// shape of the read and how much is the machine.
//
// Ad hoc, like probe_hostread.hip: it is not a CMake target. Build it with the
// host compiler and run it against the file in question:
//
//   clang++ -O2 -std=c++17 tools/probe_readspeed.cpp -o /tmp/probe_readspeed
//   /tmp/probe_readspeed G:/More-models/Model.gguf 512 1792 1,2,4,8
//
// args: <file> <region MiB> <chunk KiB> <thread list>
//
// It reports, per thread count, the aggregate GB/s over reads of `chunk KiB`
// taken from a region that was walked sequentially first (so the numbers are
// page-cache numbers, not drive numbers, which is the comparison that matters
// for a model whose file fits in RAM).
#include <windows.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static bool read_at(HANDLE fh, size_t off, void *dst, size_t len) {
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(off & 0xffffffffull);
    ov.OffsetHigh = static_cast<DWORD>(off >> 32);
    char *p = static_cast<char *>(dst);
    size_t got = 0;
    while (got < len) {
        DWORD want = static_cast<DWORD>(len - got > 0x40000000u ? 0x40000000u : len - got);
        DWORD n = 0;
        if (!ReadFile(fh, p + got, want, &n, &ov)) return false;
        if (n == 0) return false;
        got += n;
        ov.Offset += n;
        if (ov.Offset < n) ov.OffsetHigh++;
    }
    return true;
}

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s FILE REGION_MIB CHUNK_KIB THREADS(1,2,4,8)\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    size_t region = static_cast<size_t>(std::atoll(argv[2])) * 1048576ull;
    const size_t chunk = static_cast<size_t>(std::atoll(argv[3])) * 1024ull;
    std::vector<int> threads;
    for (char *p = argv[4]; *p;) {
        char *e = nullptr;
        const long v = std::strtol(p, &e, 10);
        if (e == p) break;
        threads.push_back(static_cast<int>(v));
        p = *e ? e + 1 : e;
    }

    HANDLE probe = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (probe == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "cannot open %s (err %lu)\n", path.c_str(), GetLastError());
        return 1;
    }
    LARGE_INTEGER size{};
    GetFileSizeEx(probe, &size);
    const size_t fsize = static_cast<size_t>(size.QuadPart);
    if (region > fsize) region = fsize;
    std::printf("file %.2f GiB | region %zu MiB | chunk %zu KiB\n",
                static_cast<double>(fsize) / 1073741824.0, region / 1048576, chunk / 1024);

    // ---- pass 1: pull the region into the page cache, sequentially ----------
    {
        std::vector<char> buf(8u << 20);
        double t0 = now_s();
        for (size_t off = 0; off < region; off += buf.size()) {
            const size_t n = off + buf.size() <= region ? buf.size() : region - off;
            if (!read_at(probe, off, buf.data(), n)) {
                std::fprintf(stderr, "warm read failed at %zu\n", off);
                return 1;
            }
        }
        const double s = now_s() - t0;
        std::printf("warm  %6.2f GiB/s sequential 8 MiB reads (%.1f ms)\n",
                    static_cast<double>(region) / 1073741824.0 / s, s * 1e3);
    }

    // ---- pass 2: the same region in expert-sized chunks, T readers ----------
    for (int nt : threads) {
        if (nt < 1) continue;
        std::vector<char *> bufs(static_cast<size_t>(nt));
        for (int i = 0; i < nt; i++) bufs[static_cast<size_t>(i)] = static_cast<char *>(
            VirtualAlloc(nullptr, chunk, MEM_COMMIT, PAGE_READWRITE));
        std::vector<HANDLE> hs(static_cast<size_t>(nt), INVALID_HANDLE_VALUE);
        for (int i = 0; i < nt; i++)
            hs[static_cast<size_t>(i)] = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                                     nullptr, OPEN_EXISTING,
                                                     FILE_ATTRIBUTE_NORMAL, nullptr);
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        const size_t nchunk = region / chunk;
        auto body = [&](int tid) {
            const size_t id = static_cast<size_t>(tid);
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= nchunk) return;
                if (!read_at(hs[id], i * chunk, bufs[id], chunk)) {
                    failed = true;
                    return;
                }
            }
        };
        const double t0 = now_s();
        std::vector<std::thread> pool;
        for (int i = 0; i < nt; i++) pool.emplace_back(body, i);
        for (auto &t : pool) t.join();
        const double s = now_s() - t0;
        std::printf("%2d reader%s  %6.2f GB/s  %6.2f GiB/s  %zu x %zu KiB  %s\n", nt,
                    nt == 1 ? " " : "s", static_cast<double>(nchunk * chunk) / s / 1e9,
                    static_cast<double>(nchunk * chunk) / 1073741824.0 / s, nchunk, chunk / 1024,
                    failed ? "READ FAILED" : "");
        for (int i = 0; i < nt; i++) {
            if (hs[static_cast<size_t>(i)] != INVALID_HANDLE_VALUE)
                CloseHandle(hs[static_cast<size_t>(i)]);
            if (bufs[static_cast<size_t>(i)]) VirtualFree(bufs[static_cast<size_t>(i)], 0, MEM_RELEASE);
        }
    }
    CloseHandle(probe);
    return 0;
}
