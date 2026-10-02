// ============================================================================
//  engine_usage.hpp — which GPU engine did this process actually run on?
//
//  Windows publishes per-process, per-engine GPU utilization through the
//  "GPU Engine" performance counter set. The instance name encodes the
//  attribution, so the numbers Task Manager shows can be tied to this very
//  process:
//
//      pid_<pid>_luid_<luid>_phys_<n>_eng_<n>_engtype_<Type>
//
//  e.g. "..._engtype_Compute 0" for the compute engine, "..._engtype_3D"
//  for the graphics engine, "..._engtype_Copy" for SDMA.
//
//  A benchmark runs for a fraction of a second, so a helper thread samples
//  the counters for the duration of the run and the per-engine averages are
//  printed with the result. That makes "did this run on Compute 0 or on the
//  3D engine?" a question the tool answers about itself, with no external
//  profiler and no interpretation of the number by hand.
//
//  Everywhere but Windows this is a no-op: `report()` then says the split
//  could not be sampled.
// ============================================================================
#ifndef KRK_ENGINE_USAGE_HPP
#define KRK_ENGINE_USAGE_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define KRK_ENGINE_USAGE_WINDOWS 1
#endif

#ifdef KRK_ENGINE_USAGE_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <pdh.h>
#include <pdhmsg.h>

// Older SDK headers ship pdh.h without the message constants; the values are
// fixed by the PDH API contract, so a local definition is safe.
#ifndef PDH_MORE_DATA
#define PDH_MORE_DATA ((PDH_STATUS)0x800007D2L)
#endif
#ifndef PDH_CSTATUS_VALID_DATA
#define PDH_CSTATUS_VALID_DATA ((DWORD)0x00000000L)
#endif
#ifndef PDH_CSTATUS_NEW_DATA
#define PDH_CSTATUS_NEW_DATA ((DWORD)0x00000001L)
#endif
#endif

namespace krk {

class EngineUsage {
public:
    EngineUsage() = default;
    EngineUsage(const EngineUsage &) = delete;
    EngineUsage &operator=(const EngineUsage &) = delete;

    ~EngineUsage() { stop(); }

    // Begins sampling once every `interval_ms` until stop().
    void start(int interval_ms = 50) {
#ifdef KRK_ENGINE_USAGE_WINDOWS
        interval_ms_ = interval_ms;
        if (PdhOpenQueryW(nullptr, 0, &query_) != ERROR_SUCCESS) {
            note_ = "PdhOpenQuery failed";
            return;
        }
        running_.store(true);
        thread_ = std::thread([this] { loop(); });
#else
        (void)interval_ms;
        note_ = "engine counters are Windows-only";
#endif
    }

    // Stops the sampler. Safe to call when never started, and idempotent.
    void stop() {
#ifdef KRK_ENGINE_USAGE_WINDOWS
        if (thread_.joinable()) {
            running_.store(false);
            thread_.join();
        }
        if (query_ != nullptr) {
            PdhCloseQuery(query_);
            query_ = nullptr;
        }
        counters_.clear();
        names_.clear();
#else
        // The sampler thread is Windows-only; nothing to stop.
#endif
    }

    // Prints the per-engine split collected so far.
    void report(std::FILE *out) const {
#ifdef KRK_ENGINE_USAGE_WINDOWS
        std::fprintf(out,
                     "engine split   pid %lu, per-process GPU engine utilization "
                     "(%%), sampled every %d ms:\n",
                     static_cast<unsigned long>(GetCurrentProcessId()), interval_ms_);
#else
        std::fprintf(out, "engine split   per-process GPU engine utilization (%%):\n");
#endif
        if (stats_.empty()) {
            std::fprintf(out, "  unavailable%s%s\n",
                         note_.empty() ? "" : " \xe2\x80\x94 ", note_.c_str());
            return;
        }
        // Highest average first, so the engine that did the work leads.
        std::vector<std::pair<std::string, Acc>> rows(stats_.begin(), stats_.end());
        std::sort(rows.begin(), rows.end(),
                  [](const auto &a, const auto &b) { return a.second.sum > b.second.sum; });
        for (const auto &r : rows) {
            const double avg = r.second.sum / static_cast<double>(r.second.n);
            std::fprintf(out, "  %-18s %7.2f %% avg  %7.2f %% max  (n=%llu)\n",
                         r.first.c_str(), avg, r.second.max,
                         static_cast<unsigned long long>(r.second.n));
        }
    }

private:
    struct Acc {
        double sum = 0.0;
        double max = 0.0;
        std::uint64_t n = 0;
    };

#ifdef KRK_ENGINE_USAGE_WINDOWS
    // Finds this process's engine instances and adds a counter for each.
    // Called on every tick until it succeeds: at start-up the process may
    // not have opened a GPU queue yet, and the counter set only lists
    // instances that exist.
    bool discover(int *distinct_instances) {
        DWORD inst_size = 0, ctr_size = 0;
        // The object name is localized; the build machines here run an
        // English Windows, and the failure path is a clean "unavailable".
        PDH_STATUS st = PdhEnumObjectItemsW(nullptr, nullptr, L"GPU Engine", nullptr,
                                            &ctr_size, nullptr, &inst_size,
                                            PERF_DETAIL_WIZARD, 0);
        if (st != PDH_MORE_DATA && st != ERROR_SUCCESS) {
            note_ = "the 'GPU Engine' counter set is not available";
            return false;
        }
        std::vector<wchar_t> inst(inst_size == 0 ? 1 : inst_size, L'\0');
        std::vector<wchar_t> ctr(ctr_size == 0 ? 1 : ctr_size, L'\0');
        st = PdhEnumObjectItemsW(nullptr, nullptr, L"GPU Engine", ctr.data(),
                                 &ctr_size, inst.data(), &inst_size,
                                 PERF_DETAIL_WIZARD, 0);
        if (st != ERROR_SUCCESS) {
            note_ = "the 'GPU Engine' instance list could not be read";
            return false;
        }

        const unsigned long pid = static_cast<unsigned long>(GetCurrentProcessId());
        char prefix[64];
        std::snprintf(prefix, sizeof(prefix), "pid_%lu_", pid);
        const size_t prefix_len = std::strlen(prefix);

        int found = 0;
        for (const wchar_t *p = inst.data(); *p != L'\0'; p += std::wcslen(p) + 1) {
            char name[512];
            if (WideCharToMultiByte(CP_UTF8, 0, p, -1, name, sizeof(name), nullptr,
                                    nullptr) == 0) {
                continue;
            }
            if (std::strncmp(name, prefix, prefix_len) != 0) {
                continue;
            }
            // The engine type is the tail after the last "engtype_".
            const char *type = std::strrchr(name, '_');
            const char *eng = std::strstr(name, "engtype_");
            eng = (eng != nullptr) ? eng + 8 : ((type != nullptr) ? type + 1 : name);

            wchar_t path[640];
            std::swprintf(path, 640, L"\\GPU Engine(%ls)\\Utilization Percentage", p);
            PDH_HCOUNTER counter = nullptr;
            PDH_STATUS add = PdhAddEnglishCounterW(query_, path, 0, &counter);
            if (add != ERROR_SUCCESS) {
                add = PdhAddCounterW(query_, path, 0, &counter);
            }
            if (add != ERROR_SUCCESS) {
                continue;
            }
            counters_.push_back(counter);
            names_.emplace_back(eng);
            found++;
        }
        if (distinct_instances != nullptr) {
            *distinct_instances = found;
        }
        if (found == 0) {
            note_ = "no GPU engine instances are registered for this process";
            return false;
        }
        note_.clear();
        return true;
    }

    void loop() {
        while (running_.load()) {
            if (counters_.empty() && !discover(nullptr)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
                continue;
            }
            if (PdhCollectQueryData(query_) == ERROR_SUCCESS) {
                for (size_t i = 0; i < counters_.size(); i++) {
                    PDH_FMT_COUNTERVALUE value{};
                    if (PdhGetFormattedCounterValue(counters_[i], PDH_FMT_DOUBLE,
                                                    nullptr,
                                                    &value) != ERROR_SUCCESS) {
                        continue;
                    }
                    if (value.CStatus != PDH_CSTATUS_VALID_DATA &&
                        value.CStatus != PDH_CSTATUS_NEW_DATA) {
                        continue;
                    }
                    Acc &acc = stats_[names_[i]];
                    acc.sum += value.doubleValue;
                    acc.n++;
                    if (value.doubleValue > acc.max) {
                        acc.max = value.doubleValue;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
        }
    }

    int interval_ms_ = 50;
    std::atomic<bool> running_{false};
    std::thread thread_;
    PDH_HQUERY query_ = nullptr;
    std::vector<PDH_HCOUNTER> counters_;
    std::vector<std::string> names_;
#endif
    std::map<std::string, Acc> stats_;
    std::string note_;
};

} // namespace krk

#endif // KRK_ENGINE_USAGE_HPP
