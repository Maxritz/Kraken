// ============================================================================
//  mamba2.hpp — Mamba-2 block support for nemotron_h_moe architecture.
//                                                                              //  Mamba-2 blocks use a simplified selection mechanism compared to the
//  gated delta net (GDN) path. Key differences:
//    - ssm_in.weight: input projection (no gate)
//    - ssm_conv1d: causal convolution (depthwise across state dimension)
//    - ssm_a, smm_d: per-head parameters (not per-step like GDN's ssm_alpha/beta)
//    - ssm_dt.bias: timestep projection bias
//    - ssm_norm: RMS norm before output projection
//    - ssm_out.weight: output projection
//                                                                              //  Nemotron uses alternating SSM + expert layers, not hybrid attention.
// ============================================================================

#ifndef KRK_MAMBA2_HPP
#define KRK_MAMBA2_HPP

#include "krk/model.hpp"

namespace krk {

// Mamba-2 block configuration (read from GGUF metadata)
struct Mamba2Config {
    i32 d_state = 128;         // state size (nemotron: 128)
    i32 dt_rank = 64;          // timestep projection rank (nemotron: 64)
    i32 n_group = 8;           // state groups (nemotron: 8)
    i32 d_conv = 4;            // convolution kernel size (nemotron: 4)
    i32 inner_size = 4096;     // inner dimension for ssm_in projection
    bool has_d_param = true;   // whether ssm_d is present (nemotron has it)
};

// Load Mamba-2 configuration from GGUF metadata
bool load_mamba2_config(const Gguf &gguf, ModelConfig &cfg, std::string &err);

// Resolve Mamba-2 tensor names for a given layer
struct Mamba2Tensors {
    QuantTensor ssm_in;        // [inner_size, n_embd] - input projection
    QuantTensor ssm_out;       // [n_embd, inner_size] - output projection
    QuantTensor ssm_conv1d;    // [d_conv * conv_dim, d_conv] - depthwise conv
    f32 *ssm_a = nullptr;      // [dt_rank] - selection parameter A
    f32 *ssm_d = nullptr;      // [dt_rank] - selection parameter D (if present)
    f32 *ssm_dt = nullptr;     // [dt_rank] - timestep bias
    f32 *ssm_norm = nullptr;   // [inner_size] - RMS norm weights
};

// Load Mamba-2 tensors for a specific layer
bool load_mamba2_tensors(const Gguf &gguf, i32 layer, ModelConfig &cfg,
                         Mamba2Tensors &tensors, std::string &err);

// Mamba-2 block forward pass (simplified selection mechanism)
// Unlike GDN, Mamba-2 uses:
//   1. Input projection: x @ ssm_in.T
//   2. Causal conv: conv1d(x, ssm_conv1d)
//   3. SSM step: simplified state update with A, dt, D parameters
//   4. Output projection: state @ ssm_out.T
//   5. RMS norm
struct Mamba2State {
    std::vector<f32> hidden;   // [batch, seq, inner_size] hidden state
    std::vector<f32> conv_state; // [batch, conv_dim, d_conv-1] conv state
};

// Initialize Mamba-2 state for a sequence
bool init_mamba2_state(i32 batch, i32 seq_len, i32 inner_size,
                       i32 conv_dim, i32 d_conv, Mamba2State &state);

// Run Mamba-2 forward pass for one step
bool mamba2_step(const f32 *x, i32 batch, i32 n_embd, i32 inner_size,
                 const Mamba2Tensors &tensors, const ModelConfig &cfg,
                 Mamba2State &state, f32 *out, std::string &err);

// Cleanup Mamba-2 state
void cleanup_mamba2_state(Mamba2State &state);

} // namespace krk

#endif // KRK_MAMBA2_HPP
