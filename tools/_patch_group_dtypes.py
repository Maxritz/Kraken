"""Per-matrix weight types in a fused group.

The GDN group is inert without this: Qwen3.5-0.8B's `attn_qkv` is Q5_K while
`ssm_alpha`/`ssm_beta` are Q4_K, and `gemm_group` carries ONE `wt` for the whole
group -- so the four projections that share an activation could not ride one
launch. Real checkpoints mix formats across projections routinely (this one
does), which is why the fused group has been reachable only for models that
happen to quantise every projection of a group identically.

The kernel's weight addressing is already per matrix (`w_row_bytes[m]`); the
only thing shared was the dequant dispatch tag `wt`, so this is `FusedLayer
.wt[8]` plus a nullable `wts[]` on the launcher. Nothing changes for a caller
that passes one type.

Per-file line endings, asserted anchors, /tmp backups.
"""
import io
import shutil

CR = "\r\n"
FILES = [
    "src/hip/kernels/fused.hpp",
    "include/krk/backend.hpp",
    "src/hip/backend_hip.hip",
    "src/engine.cpp",
]


def patch(path, edits):
    s = io.open(path, encoding="utf-8", newline="").read()
    crlf = s.count(CR)
    e = CR if crlf > (s.count("\n") - crlf) else "\n"
    print("  %s: %s" % (path, "CRLF" if e == CR else "LF"))
    for old, new in edits:
        old = old.replace("\n", e)
        new = new.replace("\n", e)
        n = s.count(old)
        assert n == 1, "%s: anchor found %d times: %r" % (path, n, old[:80])
        s = s.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="").write(s)
    print("patched", path)


for f in FILES:
    shutil.copyfile(f, "/tmp/dtbak_" + f.replace("/", "_"))
print("backed up %d files to /tmp" % len(FILES))

# ---- fused.hpp --------------------------------------------------------------
patch("src/hip/kernels/fused.hpp", [
    ("""struct FusedLayer {
    int count;
    const u8 *w[8];""",
     """struct FusedLayer {
    int count;
    // One weight type per matrix. A checkpoint mixes formats across the
    // projections of a group routinely (Qwen3.5-0.8B: attn_qkv Q5_K, ssm_alpha
    // and ssm_beta Q4_K), and the weight addressing was always per matrix --
    // `w_row_bytes[m]` -- so the dequant dispatch tag was the only thing that
    // had to be shared to keep a mixed group from being expressible.
    int wt[8];
    const u8 *w[8];"""),
    ("""    fused_layer_gemv_kernel(const FusedLayer fl, int wt) {""",
     """    fused_layer_gemv_kernel(const FusedLayer fl) {"""),
    ("""    for (int c = lane + nch0; c < nch1; c += 32)
        part += dot_chunks(wt, wrow, fl.in[m], c, c + 1);""",
     """    for (int c = lane + nch0; c < nch1; c += 32)
        part += dot_chunks(fl.wt[m], wrow, fl.in[m], c, c + 1);"""),
    ("""    fused_gate_silu_kernel(const u8 *__restrict__ wg, const u8 *__restrict__ wu,
                           int wt, i64 n_out, i64 n_in, size_t w_row_bytes,
                           const _Float16 *__restrict__ x,
                           _Float16 *__restrict__ out) {""",
     """    fused_gate_silu_kernel(const u8 *__restrict__ wg, const u8 *__restrict__ wu,
                           int wtg, int wtu, i64 n_out, i64 n_in,
                           size_t w_row_bytes, const _Float16 *__restrict__ x,
                           _Float16 *__restrict__ out) {"""),
    ("""    for (int c = lane; c < nch; c += 32) {
        g += dot_chunks(wt, grow, x, c, c + 1);
        u += dot_chunks(wt, urow, x, c, c + 1);
    }""",
     """    for (int c = lane; c < nch; c += 32) {
        g += dot_chunks(wtg, grow, x, c, c + 1);
        u += dot_chunks(wtu, urow, x, c, c + 1);
    }"""),
    ("""                               int cu_count, int wt, int blk_target = 0,
                               int splits_out[8] = nullptr,
                               hipStream_t st = nullptr, bool silu = false) {""",
     """                               int cu_count, int wt, int blk_target = 0,
                               int splits_out[8] = nullptr,
                               hipStream_t st = nullptr, bool silu = false,
                               const int wts[] = nullptr) {"""),
    ("""    for (int m = 0; m < n_mat; m++) {
        fl.w[m] = w[m];""",
     """    for (int m = 0; m < n_mat; m++) {
        fl.wt[m] = wts ? wts[m] : wt;
        fl.w[m] = w[m];"""),
    ("""        fused_gate_silu_kernel<256><<<static_cast<unsigned>((n_out[0] + 7) / 8),
                                      256, 0, st>>>(
            w[0], w[1], wt, n_out[0], n_in[0], w_row_bytes[0], in[0], out[0]);""",
     """        fused_gate_silu_kernel<256><<<static_cast<unsigned>((n_out[0] + 7) / 8),
                                      256, 0, st>>>(
            w[0], w[1], fl.wt[0], fl.wt[1], n_out[0], n_in[0],
            w_row_bytes[0], in[0], out[0]);"""),
    ("""    fused_layer_gemv_kernel<256>
        <<<static_cast<unsigned>(total), 256, 0, st>>>(fl, wt);""",
     """    fused_layer_gemv_kernel<256>
        <<<static_cast<unsigned>(total), 256, 0, st>>>(fl);"""),
])

# ---- backend.hpp ------------------------------------------------------------
patch("include/krk/backend.hpp", [(
    """    virtual bool gemm_group(void *const out[], const void *x,
                            const void *const w[], DType wt,
                            const i64 n_out[], i64 n_in, int n_mat,
                            i64 rows, bool silu = false,
                            bool *silu_fused = nullptr,
                            bool row_parallel_only = false) {
        (void)silu;
        (void)row_parallel_only;""",
    """    // `wts` gives each matrix its own weight type, for a group whose
    // projections are not all quantised the same way (a real checkpoint will
    // have attn_qkv at one format and ssm_alpha/ssm_beta at another). Null
    // means every matrix uses `wt`.
    virtual bool gemm_group(void *const out[], const void *x,
                            const void *const w[], DType wt,
                            const i64 n_out[], i64 n_in, int n_mat,
                            i64 rows, bool silu = false,
                            bool *silu_fused = nullptr,
                            bool row_parallel_only = false,
                            const DType *wts = nullptr) {
        (void)silu;
        (void)row_parallel_only;
        (void)wts;""")])

# ---- backend_hip.hip --------------------------------------------------------
patch("src/hip/backend_hip.hip", [
    ("""                    i64 rows, bool silu = false,
                    bool *silu_fused = nullptr,
                    bool row_parallel_only = false) override {""",
     """                    i64 rows, bool silu = false,
                    bool *silu_fused = nullptr,
                    bool row_parallel_only = false,
                    const DType *wts = nullptr) override {"""),
    ("""            wp[m] = static_cast<const u8 *>(w[m]);
            in[m] = static_cast<const _Float16 *>(x);
            op[m] = static_cast<_Float16 *>(out[m]);
            nin[m] = n_in;
            rb[m] = dtype_row_bytes(wt, n_in);""",
     """            wp[m] = static_cast<const u8 *>(w[m]);
            in[m] = static_cast<const _Float16 *>(x);
            op[m] = static_cast<_Float16 *>(out[m]);
            nin[m] = n_in;
            wtm[m] = static_cast<int>(wts ? wts[m] : wt);
            rb[m] = dtype_row_bytes(static_cast<DType>(wtm[m]), n_in);"""),
    ("""        const u8 *wp[8];
        const _Float16 *in[8];
        _Float16 *op[8];
        i64 nin[8];
        size_t rb[8];""",
     """        const u8 *wp[8];
        const _Float16 *in[8];
        _Float16 *op[8];
        i64 nin[8];
        size_t rb[8];
        int wtm[8];"""),
    ("""            row_parallel_only ? 1 : 0, nullptr, nullptr,
            fuse_silu_ && silu);""",
     """            row_parallel_only ? 1 : 0, nullptr, nullptr,
            fuse_silu_ && silu, wts ? wtm : nullptr);"""),
])

# ---- engine.cpp: hand the group its per-matrix types ------------------------
patch("src/engine.cpp", [(
    """    const bool gdn_onelaunch =
        gdn_group_ && !L.h_attn_in && !L.h_attn_gate &&
        L.wqkv.type == L.wqkv_gate.type && L.wqkv.type == L.ssm_alpha.type &&
        L.wqkv.type == L.ssm_beta.type;
    if (gdn_onelaunch) {
        void *gout[4] = {ws_qkv_, ws_z_, ws_ssm_, ws_beta_};
        const void *gw[4] = {L.wqkv.data, L.wqkv_gate.data, L.ssm_alpha.data,
                             L.ssm_beta.data};
        const i64 gn_out[4] = {cdim, vdim, mc.ssm_dt_rank, mc.ssm_dt_rank};
        be_->gemm_group(gout, ws_xn_, gw, L.wqkv.type, gn_out, n_embd_, 4, n,
                        /*silu=*/false, /*silu_fused=*/nullptr,
                        /*row_parallel_only=*/true);""",
    """    const bool gdn_onelaunch = gdn_group_ && !L.h_attn_in && !L.h_attn_gate;
    if (gdn_onelaunch) {
        void *gout[4] = {ws_qkv_, ws_z_, ws_ssm_, ws_beta_};
        const void *gw[4] = {L.wqkv.data, L.wqkv_gate.data, L.ssm_alpha.data,
                             L.ssm_beta.data};
        const DType gtypes[4] = {L.wqkv.type, L.wqkv_gate.type,
                                 L.ssm_alpha.type, L.ssm_beta.type};
        const i64 gn_out[4] = {cdim, vdim, mc.ssm_dt_rank, mc.ssm_dt_rank};
        be_->gemm_group(gout, ws_xn_, gw, L.wqkv.type, gn_out, n_embd_, 4, n,
                        /*silu=*/false, /*silu_fused=*/nullptr,
                        /*row_parallel_only=*/true, gtypes);""")])
