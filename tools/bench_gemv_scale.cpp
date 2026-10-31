// What is the ceiling on the q8_0 decode GEMV's SCALE handling, and does a
// kindi-style prebuilt scale table reach it?
//
// The engine's decode GEMV is ~86% of a CPU MoE token, so it is worth knowing
// exactly what can still come out of it. This tool answers one question with
// one changed variable: how much of the kernel is the per-(row, block pair)
// scale vector -- the f16 decode, the multiply by the activation scale, and the
// expansion into [s0 x8, s1 x8]?
//
//   V0  the shipped path, byte for byte
//   V1  the scale vector hoisted out of the loop entirely: NOT implementable
//       (the scale depends on the row and the block pair), and therefore the
//       hard upper bound on any amount of scale work ever removed
//   V2  a prebuilt f32 table [tile][pair][8 rows][2], one load and eight
//       permutes per block pair -- the shape a load-time repack would produce
//   V3  the same table in f16, so the extra traffic is 6% and not 12.5%
//
// Measured on this 2-vCPU AVX-512 host, 71.3 MB of weights (larger than the
// 54 MB L3) and n_in 4096, medians of three inside one process, interleaved.
// Two host states, because the answer depends on it:
//
//   busy host        V0 14.0-16.4 | V1 19.1-22.5 | V2 16.8-18.5 | V3 17.7-19.0 GB/s
//   quiet host       V0 22.8-24.7 | V1 26.8-27.2 | V2 24.3      | V3 25.6      GB/s
//
// So the ENTIRE scale build is worth at most 1.4x on a busy host and 1.13x on a
// quiet one; a realistic table 1.19x and 1.08x respectively. The win is real and
// it shrinks as the machine gets quieter, which is what a memory-bound kernel
// should do: uop savings only show up when the memory system is contended.
// That refutes the 49% this project previously attributed to the scale path: the
// earlier ablation removed the int-to-float convert and the multiply-add along
// with the scale build, and credited all three to the scale. What this does say
// is the opposite of what micro-optimisation wants to hear: with the scale work
// gone entirely the kernel runs at ~21 GB/s against a measured 25.5 GB/s pure
// stream read of the same bytes, so the kernel is within 1.5x of the memory
// floor and there is no large win left inside it.
#include <immintrin.h>
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>
#include "krk/common.hpp"
#include "krk/quant.hpp"

using krk::f32; using krk::i8; using krk::i64; using krk::u8; using krk::u16; using krk::u32;
static double now_s(){ timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
static u8 *W; static i64 WS,NB,NIN; static i8 *XQ; static f32 *XD, *OUT;
static f32 *TABF;   // [tile][npair][8 rows][2] f32, prebuilt
static i64 NP;

// V0: shipped scale path
__attribute__((target("avx512f,avx512bw,avx512vnni,f16c")))
static void v0(i64 n_out){
    const __m512i kl = _mm512_setr_epi32(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1);
    for(i64 o=0;o+8<=n_out;o+=8){
        __m512 acc[8]; for(int c=0;c<8;c++) acc[c]=_mm512_setzero_ps();
        for(i64 b=0;b+2<=NB;b+=2){
            const i64 p=b/2;
            const __m512i xv=_mm512_loadu_si512((const void*)(XQ+b*32));
            const f32 sx0=XD[b], sx1=XD[b+1];
            for(int c=0;c<8;c++){
                const u8* wp=W+(o+c)*WS+p*68;
                const u32 s01=(u32)wp[0]|((u32)wp[1]<<8)|((u32)wp[34]<<16)|((u32)wp[35]<<24);
                const __m128 d2=_mm_cvtph_ps(_mm_cvtsi32_si128((int)s01));
                const __m128 dd=_mm_mul_ps(d2,_mm_setr_ps(sx0,sx1,0,0));
                const __m512 sc=_mm512_permutexvar_ps(kl,_mm512_broadcast_f32x4(dd));
                const __m256i lo=_mm256_loadu_si256((const __m256i*)(wp+2));
                const __m256i hi=_mm256_loadu_si256((const __m256i*)(wp+36));
                const __m512i wv=_mm512_inserti64x4(_mm512_castsi256_si512(lo),hi,1);
                const __mmask64 ng=_mm512_cmp_epi8_mask(wv,_mm512_setzero_si512(),_MM_CMPINT_LT);
                const __m512i mag=_mm512_mask_sub_epi8(wv,ng,_mm512_setzero_si512(),wv);
                const __m512i xb=_mm512_mask_sub_epi8(xv,ng,_mm512_setzero_si512(),xv);
                const __m512i iv=_mm512_dpbusd_epi32(_mm512_setzero_si512(),mag,xb);
                acc[c]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(iv),sc,acc[c]);
            }
        }
        for(int c=0;c<8;c++) OUT[o+c]=_mm512_reduce_add_ps(acc[c]);
    }
}

// V1: scale vector hoisted out of everything -- the upper bound for any scale work
__attribute__((target("avx512f,avx512bw,avx512vnni,f16c")))
static void v1(i64 n_out){
    const __m512 sc=_mm512_set1_ps(0.0123f);
    for(i64 o=0;o+8<=n_out;o+=8){
        __m512 acc[8]; for(int c=0;c<8;c++) acc[c]=_mm512_setzero_ps();
        for(i64 b=0;b+2<=NB;b+=2){
            const i64 p=b/2;
            const __m512i xv=_mm512_loadu_si512((const void*)(XQ+b*32));
            for(int c=0;c<8;c++){
                const u8* wp=W+(o+c)*WS+p*68;
                const __m256i lo=_mm256_loadu_si256((const __m256i*)(wp+2));
                const __m256i hi=_mm256_loadu_si256((const __m256i*)(wp+36));
                const __m512i wv=_mm512_inserti64x4(_mm512_castsi256_si512(lo),hi,1);
                const __mmask64 ng=_mm512_cmp_epi8_mask(wv,_mm512_setzero_si512(),_MM_CMPINT_LT);
                const __m512i mag=_mm512_mask_sub_epi8(wv,ng,_mm512_setzero_si512(),wv);
                const __m512i xb=_mm512_mask_sub_epi8(xv,ng,_mm512_setzero_si512(),xv);
                const __m512i iv=_mm512_dpbusd_epi32(_mm512_setzero_si512(),mag,xb);
                acc[c]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(iv),sc,acc[c]);
            }
        }
        for(int c=0;c<8;c++) OUT[o+c]=_mm512_reduce_add_ps(acc[c]);
    }
}

// V2: prebuilt [pair][row][2] f32 table; one load + one broadcast-mul + 8 permutes
__attribute__((target("avx512f,avx512bw,avx512vnni,f16c")))
static void v2(i64 n_out){
    for(i64 o=0;o+8<=n_out;o+=8){
        __m512 acc[8]; for(int c=0;c<8;c++) acc[c]=_mm512_setzero_ps();
        for(i64 b=0;b+2<=NB;b+=2){
            const i64 p=b/2;
            const __m512i xv=_mm512_loadu_si512((const void*)(XQ+b*32));
            // 16 f32: [d0r0 d1r0 d0r1 d1r1 ...] for the 8 rows of THIS tile
            const f32* tb=TABF+((o>>3)*NP*16)+p*16;
            const __m512 dd=_mm512_loadu_ps(tb);
            const f32 sx0=XD[b], sx1=XD[b+1];
            const __m512 sxv=_mm512_broadcast_f32x4(_mm_setr_ps(sx0,sx1,sx0,sx1));
            const __m512 scaled=_mm512_mul_ps(dd,sxv);
            for(int c=0;c<8;c++){
                const u8* wp=W+(o+c)*WS+p*68;
                const __m512 sc=_mm512_castpd_ps(_mm512_permutexvar_pd(
                    _mm512_set1_epi64(c), _mm512_castps_pd(scaled)));
                const __m256i lo=_mm256_loadu_si256((const __m256i*)(wp+2));
                const __m256i hi=_mm256_loadu_si256((const __m256i*)(wp+36));
                const __m512i wv=_mm512_inserti64x4(_mm512_castsi256_si512(lo),hi,1);
                const __mmask64 ng=_mm512_cmp_epi8_mask(wv,_mm512_setzero_si512(),_MM_CMPINT_LT);
                const __m512i mag=_mm512_mask_sub_epi8(wv,ng,_mm512_setzero_si512(),wv);
                const __m512i xb=_mm512_mask_sub_epi8(xv,ng,_mm512_setzero_si512(),xv);
                const __m512i iv=_mm512_dpbusd_epi32(_mm512_setzero_si512(),mag,xb);
                acc[c]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(iv),sc,acc[c]);
            }
        }
        for(int c=0;c<8;c++) OUT[o+c]=_mm512_reduce_add_ps(acc[c]);
    }
}

// V3: f16 table -- 6% extra traffic instead of 12.5%, same 1 load + 8 permutes
static u16 *TABH;
__attribute__((target("avx512f,avx512bw,avx512vnni,f16c")))
static void v3(i64 n_out){
    for(i64 o=0;o+8<=n_out;o+=8){
        __m512 acc[8]; for(int c=0;c<8;c++) acc[c]=_mm512_setzero_ps();
        for(i64 b=0;b+2<=NB;b+=2){
            const i64 p=b/2;
            const __m512i xv=_mm512_loadu_si512((const void*)(XQ+b*32));
            const u16* tb=TABH+(((o>>3)*NB/2)+p)*16;
            const __m512 dd=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)tb));
            const f32 sx0=XD[b], sx1=XD[b+1];
            const __m512 sxv=_mm512_broadcast_f32x4(_mm_setr_ps(sx0,sx1,sx0,sx1));
            const __m512 scaled=_mm512_mul_ps(dd,sxv);
            for(int c=0;c<8;c++){
                const u8* wp=W+(o+c)*WS+p*68;
                const __m512 sc=_mm512_castpd_ps(_mm512_permutexvar_pd(
                    _mm512_set1_epi64(c), _mm512_castps_pd(scaled)));
                const __m256i lo=_mm256_loadu_si256((const __m256i*)(wp+2));
                const __m256i hi=_mm256_loadu_si256((const __m256i*)(wp+36));
                const __m512i wv=_mm512_inserti64x4(_mm512_castsi256_si512(lo),hi,1);
                const __mmask64 ng=_mm512_cmp_epi8_mask(wv,_mm512_setzero_si512(),_MM_CMPINT_LT);
                const __m512i mag=_mm512_mask_sub_epi8(wv,ng,_mm512_setzero_si512(),wv);
                const __m512i xb=_mm512_mask_sub_epi8(xv,ng,_mm512_setzero_si512(),xv);
                const __m512i iv=_mm512_dpbusd_epi32(_mm512_setzero_si512(),mag,xb);
                acc[c]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(iv),sc,acc[c]);
            }
        }
        for(int c=0;c<8;c++) OUT[o+c]=_mm512_reduce_add_ps(acc[c]);
    }
}

// V4: V0's algorithm with the four byte-plucks replaced by two u16 loads.
// Same scale bytes, same cache line, same arithmetic order -- only the way the
// 32-bit f16 pair is assembled changes. This is the arm that tests whether the
// win V2/V3 show is the SCALE TABLE (extra traffic, extra memory) or just the
// uop chain that assembles it (no traffic, no memory).
__attribute__((target("avx512f,avx512bw,avx512vnni,f16c")))
static void v4(i64 n_out){
    const __m512i kl = _mm512_setr_epi32(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1);
    for(i64 o=0;o+8<=n_out;o+=8){
        __m512 acc[8]; for(int c=0;c<8;c++) acc[c]=_mm512_setzero_ps();
        for(i64 b=0;b+2<=NB;b+=2){
            const i64 p=b/2;
            const __m512i xv=_mm512_loadu_si512((const void*)(XQ+b*32));
            const f32 sx0=XD[b], sx1=XD[b+1];
            for(int c=0;c<8;c++){
                const u8* wp=W+(o+c)*WS+p*68;
                const u32 s01=(u32)(*(const u16*)(wp)) |
                              ((u32)(*(const u16*)(wp+34))<<16);
                const __m128 d2=_mm_cvtph_ps(_mm_cvtsi32_si128((int)s01));
                const __m128 dd=_mm_mul_ps(d2,_mm_setr_ps(sx0,sx1,0,0));
                const __m512 sc=_mm512_permutexvar_ps(kl,_mm512_broadcast_f32x4(dd));
                const __m256i lo=_mm256_loadu_si256((const __m256i*)(wp+2));
                const __m256i hi=_mm256_loadu_si256((const __m256i*)(wp+36));
                const __m512i wv=_mm512_inserti64x4(_mm512_castsi256_si512(lo),hi,1);
                const __mmask64 ng=_mm512_cmp_epi8_mask(wv,_mm512_setzero_si512(),_MM_CMPINT_LT);
                const __m512i mag=_mm512_mask_sub_epi8(wv,ng,_mm512_setzero_si512(),wv);
                const __m512i xb=_mm512_mask_sub_epi8(xv,ng,_mm512_setzero_si512(),xv);
                const __m512i iv=_mm512_dpbusd_epi32(_mm512_setzero_si512(),mag,xb);
                acc[c]=_mm512_fmadd_ps(_mm512_cvtepi32_ps(iv),sc,acc[c]);
            }
        }
        for(int c=0;c<8;c++) OUT[o+c]=_mm512_reduce_add_ps(acc[c]);
    }
}

// V5: the kernel that actually SHIPS (dot_q8_0_q8_r1xN, the engine's decode
// path). The arms above are copies kept inside this tool; this one calls into
// quant.cpp, so a change to the shipped pluck cannot leave the tool measuring
// an older body than the one users run.
__attribute__((target("avx512f,avx512bw,avx512vnni,f16c")))
static void v5_shipped(i64 n_out){
    const int kc = krk::q8_0_gemv_width();
    for(i64 o=0;o+kc<=n_out;o+=kc)
        krk::dot_q8_0_q8_r1xN(W+o*WS, WS, XQ, XD, NIN, NIN, OUT+o, 1, kc);
}

// ---- the measurement harness ----------------------------------------------
// Every arm is run once per ROUND, in an order that rotates with the round, and
// the number reported is the MEDIAN of the rounds. Both halves matter on this
// box: the weights are 71 MB against a 54 MB L3, so the first arm of a process
// measures a cold cache (measured: V0 10.10 GB/s cold against 18.26 warm in the
// same run), and a fixed order would charge that to whichever arm ran first.
// The earlier fixed order did exactly that, which is why its V0/V2/V4 verdicts
// disagreed between runs.
static double median(std::vector<double> v){
    std::sort(v.begin(), v.end());
    const size_t n=v.size();
    return n==0 ? 0.0 : (n&1 ? v[n/2] : 0.5*(v[n/2-1]+v[n/2]));
}

int main(int argc,char**argv){
    NIN=argc>1?atoll(argv[1]):4096; const i64 n_out=argc>2?atoll(argv[2]):16384;
    NB=NIN/32; WS=NB*34;
    W=(u8*)aligned_alloc(64,(size_t)(n_out*WS));
    u32 st=1; for(i64 i=0;i<n_out*WS;i++){st=st*1664525u+1013904223u;W[i]=(u8)(st>>24);}
    for(i64 o=0;o<n_out;o++)for(i64 b=0;b<NB;b++){u16 d=(u16)(0x2800+((o*31+b*7)&0x7ff));
        W[o*WS+b*34]=(u8)(d&0xff); W[o*WS+b*34+1]=(u8)(d>>8);}
    XQ=(i8*)aligned_alloc(64,(size_t)NIN); XD=(f32*)aligned_alloc(64,(size_t)NB*4);
    for(i64 i=0;i<NIN;i++)XQ[i]=(i8)((i*37)%251-125);
    for(i64 b=0;b<NB;b++)XD[b]=0.013f+0.001f*(b%7);
    OUT=(f32*)aligned_alloc(64,(size_t)n_out*4);
    const i64 npair=NB/2; NP=npair;
    TABF=(f32*)aligned_alloc(64,(size_t)(n_out/8)*npair*16*4);
    for(i64 o=0;o+8<=n_out;o+=8) for(i64 p=0;p<npair;p++){
        f32* t=TABF+((o>>3)*npair+p)*16;
        for(int c=0;c<8;c++){ const u8* wp=W+(o+c)*WS+p*68;
            const u32 ss=(u32)wp[0]|((u32)wp[1]<<8);
            __m128 a=_mm_cvtph_ps(_mm_cvtsi32_si128((int)ss));
            t[2*c]=_mm_cvtss_f32(a);
            __m128 bb=_mm_cvtph_ps(_mm_cvtsi32_si128((int)((u32)wp[34]|((u32)wp[35]<<8))));
            t[2*c+1]=_mm_cvtss_f32(bb);
        }
    }
    TABH=(u16*)aligned_alloc(64,(size_t)(n_out/8)*npair*16*2);
    for(i64 i=0;i<(n_out/8)*npair*16;i++){
        f32 v=TABF[i]; __m128 a=_mm_set_ss(v);
        TABH[i]=(u16)_mm_extract_epi16(_mm_cvtps_ph(a,0),0);
    }
    std::printf("weights %.1f MB, n_in %lld, n_out %lld\n",
                (double)(n_out*WS)/1e6,(long long)NIN,(long long)n_out);
    std::printf("  f16 sidecar (V3) %.1f MB = +%.1f%% of the weights; "
                "f32 table (V2) %.1f MB = +%.1f%%\n",
                (double)(n_out/8)*npair*32/1e6,
                100.0*((double)(n_out/8)*npair*32/(double)(n_out*WS)),
                (double)(n_out/8)*npair*64/1e6,
                100.0*((double)(n_out/8)*npair*64/(double)(n_out*WS)));

    struct Arm { const char *nm; void (*fn)(i64); };
    const Arm arms[] = {
        {"V0 shipped",             v0},
        {"V1 scale hoisted (bound)", v1},
        {"V2 f32 table",           v2},
        {"V3 f16 table",           v3},
        {"V4 two-load pluck",      v4},
        {"V5 shipped r1xN",        v5_shipped},
    };
    const int na = sizeof(arms)/sizeof(arms[0]);
    const int rounds = argc>3 ? atoi(argv[3]) : 9;
    std::vector<std::vector<double>> gbs(static_cast<size_t>(na));
    // One untimed pass of every arm first: the tables and the weight stream are
    // cold in a fresh process, and a measurement of the cold state is not the
    // measurement any of these arms is competing on.
    for (int a=0;a<na;a++) arms[a].fn(n_out);
    for (int r=0;r<rounds;r++){
        for (int k=0;k<na;k++){
            const int a=(k+r)%na;   // rotated: no arm keeps a fixed position
            double t0=now_s(); arms[a].fn(n_out); double dt=now_s()-t0;
            gbs[static_cast<size_t>(a)].push_back((double)n_out*WS/dt/1e9);
        }
    }
    const double base = median(gbs[0]);
    std::printf("%-26s %10s %10s %8s\n","arm","median GB/s","min GB/s","vs V0");
    for (int a=0;a<na;a++){
        std::vector<double> v=gbs[static_cast<size_t>(a)];
        const double md=median(v);
        double mn=1e30; for(double x:v) if(x<mn)mn=x;
        std::printf("%-26s %10.2f %10.2f %7.2fx\n",arms[a].nm,md,mn,
                    base>0?md/base:0.0);
    }
    return 0;
}
