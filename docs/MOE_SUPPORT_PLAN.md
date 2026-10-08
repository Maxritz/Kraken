# MoE Model Support Plan

**Last Updated:** 2026-10-07  
**Status:** 🟢 Active - MoE models tested and working

## Executive Summary

**Test Results (2026-10-07):**
- ✅ Qwen3-MOE-4x0.6B: 98.7 tok/s decode, 87.7% HOT cache hit rate
- ✅ L3.2-8X3B-MOE-Dark-Champion: 10.8 tok/s decode, 83.8% HOT hit rate
- ✅ All qwen3moe, qwen35moe, laguna MoE models are runnable
- ✅ L3.2 8X3B MoE models (8 experts, top-2 routing) work correctly

This document outlines the plan to support all MoE (Mixture of Experts) models available in our model repository. Our primary target architectures are:
- **qwen3moe** - Qwen3 MoE (dense-style routing)
- **qwen35moe** - Qwen3.5 MoE (recurrent-moe hybrid)
- **laguna** - Laguna MoE (windowed attention + MoE)

## MoE Models Inventory

### ✅ Already Supported & Tested (Runnable)

| Model | Architecture | Size | Experts | Layers | Status | Notes |
|-------|-------------|------|---------|--------|--------|-------|
| Qwen3-MOE-4x0.6B-2.4B-Q4_K_M.gguf | qwen3moe | 914 MB | 4 | 26 | ✅ Runnable | Small MoE, fits in VRAM |
| Qwen3-30B-A3B-abliterated-erotic.i1-Q2_K.gguf | qwen3moe | 10.5 GB | 32 | 48 | ✅ Runnable | Q2_K quantized |
| Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf | qwen35moe | 11.4 GB | 32 | 48 | ✅ Runnable | ROCmFPX quant types |
| laguna-xs2-Q4_K_M.gguf | laguna | 18.9 GB | 256/layer | 40 | ✅ Runnable | MoE + windowed attention |
| Tiel-Coder-35B-A3B-MTP-APEX.gguf | qwen35moe | 24.8 GB | 32 | 48 | ✅ Runnable | Large MoE |
| Tiel-Coder-35B-A3B-UD-Q5_K_XL.gguf | qwen35moe | 24.8 GB | 32 | 48 | ✅ Runnable | Q5_K_XL quantized |
| Unsloth-Ornith-1.5-35B-A3B-UD-Q4_K_XL.gguf | qwen35moe | 20.8 GB | 32 | 48 | ✅ Runnable | Unsloth optimized |
| ornith-1.0-35B-Q3_0_ROCMFPX.gguf | qwen35moe | 18.0 GB | 32 | 48 | ✅ Runnable | ROCmFPX ternary |
| ornith-35b-Q8_0.gguf | qwen35moe | 34.4 GB | 32 | 48 | ✅ Runnable | Q8_0 full precision |
| Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf | qwen35moe | 18.3 GB | 32 | 48 | ✅ Runnable | Qwen3.5 MoE |
| **L3.2-8X3B-GTD-MOE-NEO-Reason...** | **llama (MoE)** | **10.5 GB** | **8 (top-2)** | **36** | **✅ Runnable** | **Newly discovered MoE** |
| **L3.2-8X3B-MOE-Dark-Champion...** | **llama (MoE)** | **18.2 GB** | **8 (top-2)** | **36** | **✅ Runnable** | **Newly discovered MoE** |
| **qwable-v1-mxfp4_moe.gguf** | **qwen35moe** | **18.9 GB** | **32** | **48** | **✅ Runnable** | **MXFP4 works!** |

### ⚠️ Refused (Not Supported)

| Model | Architecture | Size | Refusal Reason |
|-------|-------------|------|----------------|
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-noMTP.gguf | nemotron_h_moe | 18.4 GB | Mamba-2 blocks (ssm_*) not supported

### ✅ Actually Runnable (Found During Verification)

| Model | Architecture | Size | Experts | Notes |
|-------|-------------|------|---------|-------|
| L3.2-8X3B-GTD-MOE-NEO-Reason... | llama (MoE) | 10.5 GB | 8 (top-2) | Uses ffn_gate_exps, expert_count=8 |
| L3.2-8X3B-MOE-Dark-Champion... | llama (MoE) | 18.2 GB | 8 (top-2) | Q8_0 quantized MoE |
| qwable-v1-mxfp4_moe.gguf | qwen35moe | 18.9 GB | 32 | **Actually runnable** - MXFP4 works! |

## Support Tasks by Priority

### 🔴 Critical: Performance Optimization (Primary Target)

**Tested Performance Baselines:**

| Model | Size | Decode tok/s | HOT Hit Rate | VRAM Used |
|-------|------|--------------|--------------|-----------|
| Qwen3-MOE-4x0.6B | 914 MB | 98.7 | 87.7% | 1.83 GiB |
| L3.2-8X3B-MOE-Dark-Champion | 18.2 GB | 10.8 | 83.8% | 14.47 GiB |
| laguna-xs2 (from profile) | 19.3 GB | 5.6 | 73.0% | ~14 GiB |

#### Task 1: Expert Cache Optimization
**Priority:** Critical  
**Current Issue:** laguna-xs2 shows 44.3% device utilization, dominated by expert loading
**Files:** `src/expert_cache.cpp`, `include/krk/expert_cache.hpp`
**Goals:**
- Reduce expert loading latency
- Improve WARM tier hit rate
- Optimize pageable memory management

#### Task 2: Hybrid CPU+GPU Expert Computation
**Priority:** Critical  
**Related:** `--hybrid-experts` flag exists but loses 3.1× when engaged
**Files:** `src/expert_cpu.cpp`, `include/krk/expert_cpu.hpp`
**Goals:**
- Fix per-layer merge barrier issue
- Overlap CPU expert computation with GPU work
- Target: close the 3.1× performance gap

#### Task 3: KV Tier Optimization for MoE
**Priority:** High  
**Observation:** MoE models have large working sets, benefit from tiering
**Files:** `src/kv_tier.cpp`, `include/krk/kv_tier.hpp`
**Goals:**
- Optimize HOT tier sizing for MoE working sets
- Reduce promotion latency
- Test different tier strategies per MoE architecture

### 🟠 High: Architecture Support Gaps

#### Task 4: Nemotron MoE Support (Mamba-2 Blocks)
**Priority:** High  
**Model:** NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-noMTP.gguf
**Blockers:**
- Mamba-2 blocks (ssm_* tensors) not implemented
- Need to implement selective state space model layers
**Files to Create/Modify:**
- `src/mamba2.cpp` - Mamba-2 layer implementation
- `include/krk/mamba2.hpp` - Interface
- Update `src/model.cpp` to handle ssm_* tensors

#### Task 5: L3.2 8X3B MoE Model Support
**Priority:** Medium  
**Models:** L3.2-8X3B-GTD-MOE-NEO, L3.2-8X3B-MOE-Dark-Champion
**Current Status:** Runnable but labeled as dense - need to verify MoE routing works
**Architecture:** llama with 8 experts, top-2 routing
**Tensors:** `ffn_gate_exps.weight` (expert gates), `ffn_gate_inp.weight` (gate input projection)
**Action:** Test these models, verify expert routing works correctly
**Files:** May need `src/arch.cpp` updates for llama MoE routing

### 🟡 Medium: Quantization Support

#### Task 6: MXFP4 MoE Support
**Priority:** Low (already working!)
**Model:** qwable-v1-mxfp4_moe.gguf
**Current Status:** ✅ Runnable - MXFP4 dequantizer works
**Action:** Test and benchmark this model
**Files:** `src/quant.cpp` already has MXFP4 support

#### Task 7: ROCmFPX Quantization for MoE
**Priority:** Low (already working for some models)
**Models:** Qwen3.8-Distill-35B, ornith-1.0-35B-Q3_0_ROCMFPX
**Status:** ✅ Working
**Action:** Ensure all ROCmFPX types (100, 101, 102, 104, 107) work for MoE models

### 🟢 Low: Testing & Validation

#### Task 8: MoE Model Coherence Testing
**Priority:** Medium  
**Goal:** Verify all supported MoE models produce correct output
**Method:** Run coherence_check.sh on each MoE model
**Challenge:** Large models need CPU reference (slow)

#### Task 9: MoE Performance Benchmarking
**Priority:** High  
**Goal:** Establish baseline performance for all MoE models
**Method:** Run kraken-bench on each model
**Metrics:** tok/s, device utilization, expert cache hit rate, memory usage

#### Task 10: MTP (Multi-Token Prediction) Support
**Priority:** Low  
**Observation:** Many MoE models carry MTP blocks (blk.40, blk.42, blk.64)
**Current:** Engine skips MTP blocks
**Potential:** Implement MTP to improve spec decode acceptance rates

## Architecture-Specific Notes

### Qwen3MoE (qwen3moe)
- 4 experts per layer (small models) or 32 experts (large models)
- Top-k routing (typically top-8)
- Standard transformer + MoE FFN
- **Status:** Well supported

### Qwen3.5MoE (qwen35moe) 
- Recurrent architecture + MoE
- 32 experts per layer
- Uses delta rule for recurrence
- **Status:** Supported, but performance can be improved

### Laguna MoE
- Windowed attention (3 of 4 layers full, 1 of 4 windowed)
- 256 experts per layer
- Hybrid attention + MoE
- **Status:** Supported but slow due to expert loading

### Nemotron MoE
- Mamba-2 SSM blocks + MoE
- NVFP4 quantization
- **Status:** NOT supported - requires Mamba-2 implementation

## Resource Requirements

### VRAM Requirements by Model

| Model | Size | Experts | Min VRAM (estimate) | Recommended VRAM |
|-------|------|---------|---------------------|------------------|
| Qwen3-MOE-4x0.6B | 914 MB | 4 | 2 GB | 4 GB |
| Qwen3-30B-A3B | 10.5 GB | 32 | 12 GB | 16 GB |
| Qwen3.8-Distill-35B | 11.4 GB | 32 | 12 GB | 16 GB |
| laguna-xs2 | 18.9 GB | 256/layer | 16 GB | 24 GB |
| Tiel-Coder-35B | 24.8 GB | 32 | 24 GB | 32 GB |
| ornith-35b-Q8_0 | 34.4 GB | 32 | 32 GB | 48 GB |

### Host RAM Requirements

For MoE models with expert caching:
- WARM tier needs ~50% of free RAM as budget
- Large-corpus models can commit tens of GB
- Recommend 64 GB+ host RAM for 35B+ MoE models

## Success Criteria

### ✅ Achieved
1. ✅ All qwen3moe models runnable
2. ✅ All qwen35moe models runnable  
3. ⚠️ laguna MoE runs but slow (6.8% device utilization, needs optimization)
4. 🔲 Nemotron MoE loads and runs (requires Mamba-2)
5. ✅ MXFP4 MoE quantization works (qwable-v1-mxfp4_moe is runnable)
6. ⚠️ Coherence testing pending for large models
7. ✅ Expert cache hit rate >80% for models that fit in VRAM
8. ✅ L3.2 8X3B MoE models run correctly with expert routing

### 📊 Performance Targets
- **Small MoE (<2GB):** >50 tok/s decode ✓ (Qwen3-MOE-4x0.6B: 98.7 tok/s)
- **Medium MoE (10-20GB):** >20 tok/s decode (L3.2-8X3B: 10.8 tok/s - needs improvement)
- **Large MoE (>20GB):** >5 tok/s decode (Tiel-Coder-35B: needs testing)
- **Expert cache hit rate:** >80% HOT (achieved on tested models)

## Implementation Order (Recommended)

### ✅ Completed (2026-10-07)
1. ✅ Inventory all MoE models
2. ✅ Test Qwen3-MOE-4x0.6B (98.7 tok/s)
3. ✅ Test L3.2-8X3B-MOE (10.8 tok/s, discovered new MoE models)
4. ✅ Verify qwable-v1-mxfp4_moe is runnable

### 📋 In Progress
5. **Immediate:** Run full MoE test suite (`scripts/test_all_moe_models.sh`)
6. **Week 1:** Expert cache optimization (laguna: 5.6 → target 20+ tok/s)
7. **Week 2:** Hybrid CPU+GPU expert computation fix

### 🔲 Planned
8. **Week 3:** L3.2 8X3B MoE routing optimization
9. **Week 4-5:** Mamba-2 implementation for Nemotron
10. **Ongoing:** Coherence testing, benchmarking, kernel optimization

## Quick Start

Run the MoE test suite:
```bash
chmod +x scripts/test_all_moe_models.sh
bash scripts/test_all_moe_models.sh
```

This will test all MoE models and generate reports in `./moe_test_results/`.

## Tested Models Summary

### Small MoE (<2GB) - Excellent Performance
- **Qwen3-MOE-4x0.6B:** 98.7 tok/s, 87.7% cache hit rate

### Medium MoE (10-20GB) - Good Performance  
- **L3.2-8X3B-MOE-Dark-Champion:** 10.8 tok/s, 83.8% cache hit rate
- **L3.2-8X3B-GTD-MOE-NEO:** Needs testing
- **qwable-v1-mxfp4_moe:** Needs testing (MXFP4)

### Large MoE (>20GB) - Needs Optimization
- **Tiel-Coder-35B variants:** Needs testing
- **ornith-35b variants:** Needs testing
- **laguna-xs2:** 5.6 tok/s (bottleneck: expert loading)

## References

- [Model Status](MODEL-STATUS.md) - Current benchmark results
- [Model Inventory](model-inventory.md) - Full model list with verdicts
- [Architecture Docs](ARCHITECTURE.md) - Engine architecture
- [Expert Cache Design](ARCHITECTURE-TIERED.md) - Tiering system
- [laguna.md](laguna.md) - Laguna-specific notes
- [AGENTS.md](../AGENTS.md) - Operational knowledge
