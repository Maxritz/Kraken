# GGUF type 26 is I32, not a weight quant — and that is already decoded here

Type 26 in this build = `DType::I32`, a plain `int32` tensor widened to f32 by a
straight cast (`src/quant.cpp`, `dequant_row` for `DType::I32`). There is no block
to walk, because the i32 values ARE the data. The only question is sign/width,
which is already answered by a 1:1 widen.

That is why the DeepSeek-V4-Flash file's type-26 tensors are not a dequantization
gap. They are the **integer hash-route tables** (`ffn_gate_tid2eid`-class data),
host-side routing tables looked up as integers — not matmul weights. The loader
already has the int32-table path for exactly that class of tensor (`src/model.cpp`:
`int32 table ... is missing`, `expected an int32 table (ggml type 26)`,
`... is an int32 table (ggml type 26), not a matmul weight`).

So the stale line in the refusal text — "uses quantization type #26, which this
build cannot dequantize" — is not the reason the file cannot run. The file is
still refused, but for the **architecture**: the V4 shared-latent MLA + grouped
output attention graph and the V4-only extras (compressed attention, Sinkhorn
hyper-connections, tid2eid hash routing, compress ratios), with
`attention_implemented = false` on the `deepseek4` arch row in `src/arch.cpp`.
Type #26 is already handled; the blocker is the attention computation those
tensors imply, not the tensor's numeric type.
