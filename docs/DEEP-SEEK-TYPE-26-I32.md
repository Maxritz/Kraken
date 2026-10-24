// I32 (ggml type 26): plain int32, widened. Not a quantization, so there
// is no block to walk -- the values ARE the data, and the only question
// is the sign and the width.
if (t == DType::I32) {
    const i32 *w = static_cast<const i32 *>(src);
    for (i64 i = 0; i < n; i++) out[i] = static_cast<f32>(w[i]);
    return;
}
