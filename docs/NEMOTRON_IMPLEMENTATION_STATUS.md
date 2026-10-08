# Nemotron MoE Implementation Status

## Current State
The Nemotron MoE model (NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-noMTP.gguf) has:
- Architecture recognized in arch table (nemotron_h_moe)
- Layer type detection (SSM vs expert layers)
- Partial tensor loading
- Build currently broken from brace matching issues

## Architecture Overview
Nemotron uses alternating layers:
- **Even layers (0, 2, 4, ...)**: Mamba-2 SSM blocks
  - ssm_in.weight [10304, 2688] - input projection
  - ssm_out.weight [2688, 4096] - output projection  
  - ssm_conv1d.weight - causal convolution
  - ssm_a, ssm_d, ssm_dt.bias - SSM parameters
  - ssm_norm.weight - RMS norm
- **Odd layers (1, 3, 5, ...)**: MoE expert blocks
  - ffn_gate_inp.weight - router input [128, 2688]
  - ffn_up_exps.weight - routed experts up-projection [128, 1856, 2688] (NVFP4)
  - ffn_down_exps.weight - routed experts down-projection [128, 2688, 1856] (NVFP4)
  - ffn_gate_shexp.weight - shared expert gate [128, 2688]
  - ffn_up_shexp.weight - shared expert up [2688, 3712] (NVFP4)
  - ffn_down_shexp.weight - shared expert down [3712, 2688] (NVFP4)
  - exp_probs_b.bias - router bias [128]

## What's Implemented
1. ✅ Arch table entry (nemotron_h_moe = supported)
2. ✅ Config loading (ssm_inner_size, has_d_param, nemotron_moe flag)
3. ✅ Layer type detection (nemotron_ssm flag based on ssm_in presence)
4. ✅ SSM tensor loading for even layers (ssm_in, ssm_out, ssm_conv1d, ssm_a, ssm_d, ssm_dt, ssm_norm)
5. ✅ Expert tensor loading for odd layers (exps_up, exps_down, gate_inp, shexp_*)
6. ❌ Forward pass (engine.cpp - placeholder only)
7. ❌ NVFP4 dequantizer for expert weights (needs verification)

## What Needs to Be Done

### Immediate (Blocker)
- Fix build errors (brace matching in model.cpp)
- Complete forward pass for SSM layers (engine.cpp)
- Complete forward pass for MoE layers (engine.cpp)
- Integrate with expert cache system

### NVFP4 Support
- Verify NVFP4 dequantizer works (src/quant.cpp)
- Test with Nemotron expert weights
- Add device kernel for NVFP4 if needed

### Forward Pass
The current placeholder in engine.cpp just does:
```cpp
be_->gemm(ws_xn_, ws_x_, L.ssm_in.data, L.ssm_in.type, n_embd_, mc.ssm_inner_size, n);
be_->gemm(ws_x_, ws_xn_, L.ssm_out.data, L.ssm_out.type, mc.ssm_inner_size, n_embd_, n);
```

This is incorrect - needs proper Mamba-2 forward:
1. Input projection: x @ ssm_in.T
2. Causal conv1d
3. SSM step with A, dt, D parameters
4. Output projection: state @ ssm_out.T
5. RMS norm

### MoE Forward Pass
For expert layers:
1. Router: x @ ffn_gate_inp.T → gate scores
2. Top-k selection (6 experts out of 128)
3. Expert computation: silu(gate_up @ x) * gate_down @ x
4. Combine expert outputs
5. Add shared expert output

## Files to Modify
- src/model.cpp - fix brace issues, complete tensor loading
- src/engine.cpp - implement forward pass for SSM and MoE layers
- src/quant.cpp - verify NVFP4 support
- src/expert_cache.cpp - integrate with expert cache
- src/hip/kernels/ - add device kernels for Mamba-2 and MoE if needed

## Testing
- Load Nemotron model: `kraken --model NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-noMTP.gguf --info`
- Generate tokens: `kraken --model ... --prompt "..." --max-tokens 16 --greedy`
- Profile: `kraken --model ... --profile`
- Coherence: compare with CPU reference

## Related Models
- qwable-v1-mxfp4_moe.gguf - MXFP4 MoE (needs MXFP4 dequantizer)
- Ternary models (Bonsai-27B-Q1_0, Ternary-Bonsai-2-27B-PQ2_0) - already working
- DeepSeek models - need MLA implementation (major effort)
- Gemma4 models - need multiple features (sliding window, shared-KV, softcap)
- spark2_5 models - need fused QKV support
- k2-horizon models - need routed attention values
- muse-glimmer - needs attention gate (partially supported)
- flash-next models - need head geometry bypass
ENDOFFILE
cat docs/NEMOTRON_IMPLEMENTATION_STATUS.md