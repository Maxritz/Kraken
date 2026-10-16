// Standalone HIP regression probe for the fused decode attention chain.
// Build with the same optimization flags as krk_hip (including fast-math).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "hip/krk_hip.hpp"
#include "hip/kernels/normalize.hpp"
#include "hip/kernels/attention.hpp"

namespace krk {
const char *hip_err_name(hipError_t e) { return hipGetErrorName(e); }
void hip_check(hipError_t e, const char *what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorName(e));
        std::exit(2);
    }
}
}
using namespace krk;

template<class T> struct Buffer {
    T *p = nullptr;
    explicit Buffer(size_t n) { KRK_HIP(hipMalloc(&p, n * sizeof(T))); }
    ~Buffer() { if (p) hipFree(p); }
    void put(const std::vector<T> &v) {
        KRK_HIP(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    std::vector<T> get(size_t n) {
        std::vector<T> v(n);
        KRK_HIP(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
        return v;
    }
};
static float sample(unsigned &s) {
    s = s * 1664525u + 1013904223u;
    return static_cast<float>((s >> 8) & 65535) / 8192.0f - 4.0f;
}
static size_t different(const std::vector<_Float16> &a,
                        const std::vector<_Float16> &b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(_Float16))) ++n;
    return n;
}
int main() {
    size_t failures = 0, cases = 0;
    for (int hd : {64, 128}) for (bool neox : {false, true})
    for (bool norm : {false, true}) {
        constexpr int heads = 8, kvheads = 2, ctx = 256;
        const size_t qn = heads * hd, kn = kvheads * hd, cn = ctx * kn;
        Buffer<_Float16> q(qn), qr(qn), k(kn), kr(kn), v(kn), kc(cn), vc(cn);
        Buffer<_Float16> kcr(cn), vcr(cn), out(qn), ref(qn), qdebug(qn), production(qn);
        Buffer<float> wq(hd), wk(hd), freq(hd / 2);
        Buffer<i64> pos(1);
        size_t out_bad = 0, key_bad = 0, val_bad = 0;
        for (unsigned seed = 1; seed <= 32; ++seed)
        for (i64 p0 : {1, 17, 63, 127, 191}) {
            unsigned s = seed;
            std::vector<_Float16> hq(qn), hk(kn), hv(kn), hkc(cn), hvc(cn);
            for (auto &x : hq) x = static_cast<_Float16>(sample(s));
            for (auto &x : hk) x = static_cast<_Float16>(sample(s));
            for (auto &x : hv) x = static_cast<_Float16>(sample(s));
            for (auto &x : hkc) x = static_cast<_Float16>(sample(s));
            for (auto &x : hvc) x = static_cast<_Float16>(sample(s));
            std::vector<float> hwq(hd), hwk(hd), hf(hd / 2);
            for (auto &x : hwq) x = 1.0f + sample(s) * 0.1f;
            for (auto &x : hwk) x = 1.0f + sample(s) * 0.1f;
            for (int i = 0; i < hd / 2; ++i)
                hf[i] = std::pow(1000000.0f, -2.0f * i / hd);
            q.put(hq); qr.put(hq); k.put(hk); kr.put(hk); v.put(hv);
            kc.put(hkc); kcr.put(hkc); vc.put(hvc); vcr.put(hvc);
            wq.put(hwq); wk.put(hwk); freq.put(hf); pos.put({p0});
            if (norm) {
                head_norm_kernel<128><<<heads, 128>>>(qr.p, wq.p, heads, hd, 1e-6f);
                head_norm_kernel<128><<<kvheads, 128>>>(kr.p, wk.p, kvheads, hd, 1e-6f);
            }
            rope_kernel<<<heads, 128>>>(qr.p, heads, hd, pos.p, freq.p, 1.0f, 1.0f, neox);
            rope_kernel<<<kvheads, 128>>>(kr.p, kvheads, hd, pos.p, freq.p, 1.0f, 1.0f, neox);
            KRK_HIP(hipMemcpy(kcr.p + p0 * kn, kr.p, kn * 2, hipMemcpyDeviceToDevice));
            KRK_HIP(hipMemcpy(vcr.p + p0 * kn, v.p, kn * 2, hipMemcpyDeviceToDevice));
            const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
            if (hd == 64)
                attention_decode_launch<2>(ref.p, qr.p, kcr.p, vcr.p, heads, kvheads,
                                          hd, pos.p, kn, 1, 0, scale);
            else
                attention_decode_launch<4>(ref.p, qr.p, kcr.p, vcr.p, heads, kvheads,
                                          hd, pos.p, kn, 1, 0, scale);
            if (hd == 64)
                attn_fused_decode_kernel<2, true><<<heads, kAttnDecThreads>>>(
                    out.p, q.p, kc.p, vc.p, k.p, v.p,
                    norm ? wq.p : nullptr, norm ? wk.p : nullptr,
                    1e-6f, neox, heads, kvheads, hd, pos.p, kn,
                    freq.p, 1.0f, scale, 1.0f, qdebug.p);
            else
                attn_fused_decode_kernel<4, true><<<heads, kAttnDecThreads>>>(
                    out.p, q.p, kc.p, vc.p, k.p, v.p,
                    norm ? wq.p : nullptr, norm ? wk.p : nullptr,
                    1e-6f, neox, heads, kvheads, hd, pos.p, kn,
                    freq.p, 1.0f, scale, 1.0f, qdebug.p);
            attn_fused_decode(production.p, q.p, kc.p, vc.p, k.p, v.p,
                              norm ? wq.p : nullptr, norm ? wk.p : nullptr,
                              1e-6f, neox, heads, kvheads, hd, pos.p, kn,
                              freq.p, 1.0f, scale, 1.0f);
            KRK_HIP(hipGetLastError());
            KRK_HIP(hipDeviceSynchronize());
            const auto reference = ref.get(qn);
            const size_t ob = different(reference, out.get(qn)) +
                              different(reference, production.get(qn));
            const size_t kb = different(kcr.get(cn), kc.get(cn));
            const size_t vb = different(vcr.get(cn), vc.get(cn));
            out_bad += ob; key_bad += kb; val_bad += vb;
            const auto expected_q = qr.get(qn), actual_q = qdebug.get(qn);
            const size_t qb = different(expected_q, actual_q);
            if (ob || qb) {
                std::printf("case hd=%d neox=%d norm=%d seed=%u pos=%lld q_diff=%zu out_diff=%zu\n",
                            hd, neox, norm, seed, static_cast<long long>(p0), qb, ob);
                for (size_t i = 0, shown = 0; i < qn && shown < 3; ++i) {
                    if (!std::memcmp(&expected_q[i], &actual_q[i], 2)) continue;
                    std::printf("  q[%zu] ref=%.9g got=%.9g\n", i,
                                static_cast<double>(expected_q[i]), static_cast<double>(actual_q[i]));
                    ++shown;
                }
            }
            failures += (ob || kb || vb || qb); ++cases;
        }
        std::printf("hd=%d neox=%d norm=%d output_diff=%zu key_diff=%zu value_diff=%zu\n",
                    hd, neox, norm, out_bad, key_bad, val_bad);
    }
    std::printf("%zu/%zu cases exact\n", cases - failures, cases);
    return failures ? 1 : 0;
}
