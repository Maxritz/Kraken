# DeepSeek-V4 quant type #26 is I32, not a weight quant

Type 26 in this build is `DType::I32` = plain `int32`, widened to f32 by a
straight cast. It is **not** a quantization with blocks and codebooks, so
dequantizing it is not the gap and never was. The decoder is already here:

    src/quant.cpp: dequant_row for DType::I32 reads the backing i32* and writes
    static_cast<f32>(w[i]) for each element — a 1:1 widen, no block walk.

What that means for the real file:

- The DeepSeek-V4-Flash GGUF's type-26 tensors are the **integer hash-route
  tables** (`ffn_gate_tid2eid`-class data), host-side routing data looked up as
  integers, never multiplied as a matmul weight. That is the same class of
  tensor the loader already treats as an int32 table in `src/model.cpp` (the
  `int32 table ... is missing` / `expected an int32 table (ggml type 26)` / 
  `... is an int32 table (ggml type 26), not a matmul weight` paths).
- So "uses quantization type #26, which this build cannot dequantize" is stale
  as the reason the file cannot run. The widen already exists. The refusal is
  the **architecture** — the V4 shared-latent MLA + grouped output attention
  graph and the V4-only extras (compressed attention, Sinkhorn hyper-connections,
  tid2eid hash routing, compress ratios) — not the tensor's numeric type.

Honest state, in one line:

- `deepseek4` is still `ArchSupport::No` with `attention_implemented = false` in
  `src/arch.cpp`, and the loader/engine refuse it for the V4 attention shape and
  the V4 extras by name. Type #26 is not the blocker; it is I32, and the loader
  already reads int32 tables of it.
