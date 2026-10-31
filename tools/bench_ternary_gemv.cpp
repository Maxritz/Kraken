// Does a ternary format make a CPU decode GEMV faster, and by how much?
//
// The claim this settles: "kindi is faster because it runs a 1.58-bit model, and
// q8_0 is 5.4x the bytes" -- which would predict a 5.4x decode win. It does not
// work out that way, because a ternary kernel is more taxing per MAC and the
// kernel stops being memory-bound. Three kernels, each reading its OWN natural
// format, same matrix shape, same machine:
//
//   Q0  q8_0,      1.0625 B/weight, vpdpbusd, 8 rows in flight (what ships here)
//   T0  ternary LUT, 0.25 B/weight, vpermps into 8 pair sums -- kindi's CPU
//       shape (kindi/kcpu.c rows8: rows in SIMD lanes, per-column-pair tables)
//   T1  ternary bit-plane, 0.25 B/weight, kmask -> vpmovm2b -> vpdpbusd: the
//       VNNI-native way to consume two bit-planes without a lookup table
//
// Measured on a 2-vCPU AVX-512 host, 4096 weights x 16384 rows (67.1M weights):

// Five runs, medians, on a 2-vCPU AVX-512 host:
//
//   Q0 q8_0 vpdpbusd                18.3 G-w/s   19-24 GB/s   <- memory bound
//   T0 ternary LUT (kindi shape)    17.2 G-w/s    5.6 GB/s    <- compute bound
//   T1 ternary bit-plane VNNI       36.0 G-w/s   11.5 GB/s    <- compute bound
//
// Read it off the byte rate: with 4.25x fewer bytes a ternary kernel that were
// still memory-bound would show ~4x the G-w/s. T1 shows 2x and T0 shows 0.94x --
// slower than q8_0 -- because at 0.25 B/w the loop is limited by instructions,
// not by DRAM.
//
//   * kindi's LUT shape is NOT faster than q8_0 on this CPU (17.2 vs 18.3); it
//     spends ~14 uops per 32 weights (2.3 weights/uop) against vpdpbusd's 64
//     MACs per instruction.
//   * the VNNI-native bit-plane decode is the better ternary kernel (1.5x), and
//     it is still bound by mask-register throughput, not memory.
//   * so on this box the format is worth ~2x in decode, NOT 4.25x, and only
//     with the VNNI-native kernel -- the LUT collects none of it.
//
// The byte advantage still matters enormously -- for RESIDENCY, where a model
// has to stream from disk or fit a RAM budget. It is not a free kernel win.
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
using u8=unsigned char; using i8=signed char; using u32=unsigned; using u64=unsigned long long; using i64=long long; using f32=float;
static double now_s(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

static const i64 NIN = 4096;          // weights per row
static const i64 NOUT = 16384;        // rows

// ---- Q0: q8_0 (34 B per 32 weights), 8 rows in flight
__attribute__((target("avx512f,avx512bw,avx512vnni,f16c")))
static double q0(){
    const i64 nb=NIN/32, rb=nb*34;
    u8* W=(u8*)aligned_alloc(64,(size_t)NOUT*rb);
    u32 st=1; for(i64 i=0;i<NOUT*rb;i++){st=st*1664525u+1013904223u;W[i]=(u8)(st>>24);}
    for(i64 o=0;o<NOUT;o++)for(i64 b=0;b<nb;b++){unsigned short d=(unsigned short)(0x2800+((o*31+b*7)&0x7ff));
        W[o*rb+b*34]=(u8)(d&0xff); W[o*rb+b*34+1]=(u8)(d>>8);}
    i8* xq=(i8*)aligned_alloc(64,NIN); f32* xd=(f32*)aligned_alloc(64,nb*4);
    for(i64 i=0;i<NIN;i++)xq[i]=(i8)((i*37)%251-125);
    for(i64 b=0;b<nb;b++)xd[b]=0.013f;
    f32* out=(f32*)aligned_alloc(64,NOUT*4);
    const __m512i kl=_mm512_setr_epi32(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1);
    double best=1e30;
    for(int r=0;r<3;r++){ double t0=now_s();
        for(i64 o=0;o+8<=NOUT;o+=8){
            __m512 acc[8]; for(int c=0;c<8;c++)acc[c]=_mm512_setzero_ps();
            for(i64 b=0;b+2<=nb;b+=2){
                const i64 p=b/2;
                const __m512i xv=_mm512_loadu_si512((const void*)(xq+b*32));
                const f32 sx0=xd[b],sx1=xd[b+1];
                for(int c=0;c<8;c++){
                    const u8* wp=W+(o+c)*rb+p*68;
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
            for(int c=0;c<8;c++)out[o+c]=_mm512_reduce_add_ps(acc[c]);
        }
        double dt=now_s()-t0; if(dt<best)best=dt; }
    printf("%-46s %7.2f G-w/s  %5.2f GB/s  (%5.3f B/w)\n","Q0 q8_0 vpdpbusd",
           (double)NOUT*NIN/best/1e9,(double)NOUT*rb/best/1e9,8.5/8);
    free(W);free(xq);free(xd);free(out); return best;
}

// ---- T1: ternary, 2 bit-planes per weight (128 B per 512 weights = 2 bits/w)
__attribute__((target("avx512f,avx512bw,avx512vnni,f16c")))
static void t1(){
    const i64 nb=NIN/64, rb=nb*16;          // per row: nb*16 bytes
    u8* W=(u8*)aligned_alloc(64,(size_t)NOUT*rb);
    u32 st=7; for(i64 i=0;i<NOUT*rb;i++){st=st*1664525u+1013904223u;W[i]=(u8)(st>>24);}
    i8* xq=(i8*)aligned_alloc(64,NIN);
    for(i64 i=0;i<NIN;i++)xq[i]=(i8)((i*37)%251-125);
    f32* out=(f32*)aligned_alloc(64,NOUT*4);
    double best=1e30;
    for(int r=0;r<3;r++){ double t0=now_s();
        for(i64 o=0;o+8<=NOUT;o+=8){
            __m512 acc[8]; for(int c=0;c<8;c++)acc[c]=_mm512_setzero_ps();
            for(i64 b=0;b<nb;b++){
                const __m512i xv=_mm512_loadu_si512((const void*)(xq+b*64));
                for(int c=0;c<8;c++){
                    const u8* wp=W+(o+c)*rb+b*16;
                    const u64 mp=*(const u64*)(wp), mn=*(const u64*)(wp+8);
                    const __mmask64 kp=_cvtu64_mask64(mp), kn=_cvtu64_mask64(mn);
                    const __mmask64 kb=_kor_mask64(kp,kn);
                    const __m512i mag=_mm512_movm_epi8(kb);
                    const __m512i fl=_mm512_mask_sub_epi8(xv,kn,_mm512_setzero_si512(),xv);
                    const __m512i xb=_mm512_maskz_mov_epi8(kb,fl);
                    const __m512i iv=_mm512_dpbusd_epi32(_mm512_setzero_si512(),mag,xb);
                    acc[c]=_mm512_add_ps(acc[c],_mm512_cvtepi32_ps(iv));
                }
            }
            for(int c=0;c<8;c++)out[o+c]=_mm512_reduce_add_ps(acc[c]);
        }
        double dt=now_s()-t0; if(dt<best)best=dt; }
    printf("%-46s %7.2f G-w/s  %5.2f GB/s  (%5.3f B/w)\n","T1 ternary bitplane -> vpdpbusd",
           (double)NOUT*NIN/best/1e9,(double)NOUT*rb/best/1e9,2.0/8);
    free(W);free(xq);free(out);
}

// ---- T0: ternary LUT, kindi's CPU shape: 8 rows in SIMD lanes, vpermps
__attribute__((target("avx2,fma")))
static void t0(){
    // 8 rows per block, 2 bits/weight packed 16 codes per u32 (32 weights/u32)
    const i64 ng=NIN/128, rb=ng*256;         // 256 B per 8-row block per 128 cols
    u8* W=(u8*)aligned_alloc(64,(size_t)NOUT*rb);
    u32 st=11; for(i64 i=0;i<NOUT*rb;i++){st=st*1664525u+1013904223u;W[i]=(u8)(st>>24);}
    f32* x=(f32*)aligned_alloc(64,(size_t)NIN*4);
    for(i64 i=0;i<NIN;i++)x[i]=(f32)((i*13)%97)/97.0f-0.5f;
    f32* T=(f32*)aligned_alloc(64,(size_t)(NIN/2)*8*4);   // pair sum tables, 8 per column pair
    for(i64 p=0;p<NIN/2;p++){ f32 x0=x[2*p],x1=x[2*p+1]; f32* t=T+p*8;
        t[0]=-x0-x1;t[1]=-x1;t[2]=x0-x1;t[3]=-x0;t[4]=x0;t[5]=-x0+x1;t[6]=x1;t[7]=x0+x1; }
    f32* out=(f32*)aligned_alloc(64,NOUT*4);
    const __m256i sh=_mm256_setr_epi32(0,4,8,12,16,20,24,28);
    const __m256i f=_mm256_set1_epi32(15);
    double best=1e30;
    for(int r=0;r<3;r++){ double t0_=now_s();
        for(i64 rb0=0;rb0+8<=NOUT;rb0+=8){
            __m256 acc=_mm256_setzero_ps();
            for(i64 g=0;g<ng;g++){
                const u32* pw=(const u32*)(W+rb0*rb)+g*64;
                const f32* pt=T+(size_t)g*64*8;
                __m256 s0=_mm256_setzero_ps(),s1=s0;
                for(int p=0;p<64;p+=2){
                    const __m256i c0=_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)pw[p]),sh),f);
                    const __m256i c1=_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)pw[p+1]),sh),f);
                    const __m256 v0=_mm256_permutevar8x32_ps(_mm256_load_ps(pt+8*p),c0);
                    const __m256 v1=_mm256_permutevar8x32_ps(_mm256_load_ps(pt+8*p+8),c1);
                    const __m256 z0=_mm256_castsi256_ps(_mm256_srai_epi32(_mm256_slli_epi32(c0,28),31));
                    const __m256 z1=_mm256_castsi256_ps(_mm256_srai_epi32(_mm256_slli_epi32(c1,28),31));
                    s0=_mm256_add_ps(s0,_mm256_andnot_ps(z0,v0));
                    s1=_mm256_add_ps(s1,_mm256_andnot_ps(z1,v1));
                }
                // scales: 8 rows contiguous, one ymm load
                acc=_mm256_fmadd_ps(_mm256_add_ps(s0,s1),_mm256_set1_ps(0.01f),acc);
            }
            _mm256_storeu_ps(out+rb0,acc);
        }
        double dt=now_s()-t0_; if(dt<best)best=dt; }
    // 8 rows x NIN weights per outer step
    const double gw=(double)(NOUT/8*8)*NIN;
    printf("%-46s %7.2f G-w/s  %5.2f GB/s  (%5.3f B/w)\n","T0 ternary LUT vpermps (kindi shape)",
           gw/best/1e9,(double)(NOUT/8)*rb/best/1e9,2.0/8);
    free(W);free(x);free(T);free(out);
}

int main(){
    printf("%lld weights per matrix, %lld rows\n",(long long)NIN,(long long)NOUT);
    q0(); t0(); t1(); q0(); t1();
    return 0;
}
