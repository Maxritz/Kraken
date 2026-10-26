// Register and scratch usage of the prefill WMMA instantiations.
//
// Compiled by hand with -Rpass-analysis=kernel-resource-usage, not by the
// build: the question it answers is why the full kernel costs 49% more than
// its own two halves added together (KRK_GEMM_STAGE_ONLY / _MMA_ONLY), and
// the first thing that can explain that is register pressure -- the arms each
// drop one side's live values. Nothing here runs.
#include "hip/kernels/gemm.hpp"

using namespace krk;

// The shipped prefill config, its two diagnostic arms, and the 26 kB
// two-blocks-per-CU config for comparison.
void *krk_gemm_resource_instances[] = {
    reinterpret_cast<void *>(&gemm_wmma_kernel<wmma_cfg_n2, 0,
                                               static_cast<int>(DType::Q4_K), 2>),
    reinterpret_cast<void *>(&gemm_wmma_kernel<wmma_cfg_n2, 1,
                                               static_cast<int>(DType::Q4_K), 2>),
    reinterpret_cast<void *>(&gemm_wmma_kernel<wmma_cfg_n2, 2,
                                               static_cast<int>(DType::Q4_K), 2>),
    reinterpret_cast<void *>(&gemm_wmma_kernel<wmma_cfg, 0,
                                               static_cast<int>(DType::Q4_K)>),
    reinterpret_cast<void *>(&gemm_wmma_kernel<wmma_cfg, 0,
                                               static_cast<int>(DType::Q4_K), 2>),
};
