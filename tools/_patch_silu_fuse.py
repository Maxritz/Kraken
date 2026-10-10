"""Fold silu_mul into the gate/up group launch.

Written as a file, not a shell heredoc: the payload carries C++ string literals
and quotes that a heredoc mis-parses (AGENTS.md, "Patching files safely"). The
line ending is taken from each file rather than assumed -- fused.hpp is LF on
disk and engine.cpp is CRLF, and an anchor built with the wrong one matches
nothing. Anchors are asserted, and the four files are backed up to /tmp first so
a failed verification can be reverted without git.
"""
import io
import shutil

CR = "\r\n"


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


FILES = [
    "src/hip/kernels/fused.hpp",
    "include/krk/backend.hpp",
    "src/hip/backend_hip.hip",
    "src/engine.cpp",
]
for f in FILES:
    shutil.copyfile(f, "/tmp/bak_" + f.replace("/", "_"))
print("backed up %d files to /tmp" % len(FILES))

# ---- fused.hpp: the gate/up + silu kernel and the launcher's silu branch ----
patch("src/hip/kernels/fused.hpp", [
    ("// Launch the fused GEMM phase. `n_mat` matrices share the launch",
     """// ----------------------------------------------------------------- gate/up --
//
// The FFN's gate and up projections share one activation and are followed by
// silu_mul, which reads the SAME row index of both. One warp per output row
// computes both dots, so the activation is applied to registers and the two
// launches that would produce the operands disappear into the launch that
// already reads them.
//
// Bit-identical to gemm_group() (two gemv rows) + silu_mul_kernel(), and by the
// same argument the residual epilogue uses: each dot keeps gemv_kernel's
// lane-strided chunk order and wave reduce, so the f32 partial is the same
// number; both are rounded to f16 before the activation, which is exactly what
// the separate silu_mul_kernel reads back from memory; and the activation and
// the product are computed in f32 and rounded once, as that kernel does.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
    fused_gate_silu_kernel(const u8 *__restrict__ wg, const u8 *__restrict__ wu,
                           int wt, i64 n_out, i64 n_in, size_t w_row_bytes,
                           const _Float16 *__restrict__ x,
                           _Float16 *__restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const i64 row = static_cast<i64>(blockIdx.x) * (THREADS / 32) +
                    static_cast<i64>(threadIdx.x >> 5);
    if (row >= n_out) return;

    const int nch = static_cast<int>(n_in / 32);
    const u8 *grow = wg + static_cast<size_t>(row) * w_row_bytes;
    const u8 *urow = wu + static_cast<size_t>(row) * w_row_bytes;

    f32 g = 0.0f, u = 0.0f;
    for (int c = lane; c < nch; c += 32) {
        g += dot_chunks(wt, grow, x, c, c + 1);
        u += dot_chunks(wt, urow, x, c, c + 1);
    }
    g = d_wave_reduce_sum(g);
    u = d_wave_reduce_sum(u);
    if (lane == 0) {
        const f32 hg = static_cast<f32>(static_cast<_Float16>(g));
        const f32 hu = static_cast<f32>(static_cast<_Float16>(u));
        out[row] = static_cast<_Float16>(d_silu(hg) * hu);
    }
}

// Launch the fused GEMM phase. `n_mat` matrices share the launch"""),
    ("""inline void fused_layer_launch(const u8 *const w[],
                               const _Float16 *const in[],
                               _Float16 *const out[],
                               const i64 n_out[], const i64 n_in[],
                               const size_t w_row_bytes[],
                               int n_mat,
                               f32 *part, i64 part_cap,
                               int cu_count, int wt, int blk_target = 0,
                               int splits_out[8] = nullptr,
                               hipStream_t st = nullptr) {
    if (n_mat <= 0) return;
    if (n_mat > 8) n_mat = 8;""",
     """// The return value is the SILU answer and nothing else: true when `silu` was
// asked for and the pair kernel ran (out[0] then holds silu(gate)*up and the
// caller must NOT issue silu_mul()), false for every other outcome, including
// the ordinary group launch.
inline bool fused_layer_launch(const u8 *const w[],
                               const _Float16 *const in[],
                               _Float16 *const out[],
                               const i64 n_out[], const i64 n_in[],
                               const size_t w_row_bytes[],
                               int n_mat,
                               f32 *part, i64 part_cap,
                               int cu_count, int wt, int blk_target = 0,
                               int splits_out[8] = nullptr,
                               hipStream_t st = nullptr, bool silu = false) {
    if (n_mat <= 0) return false;
    if (n_mat > 8) n_mat = 8;"""),
    ("""    fl.part = part;
    if (splits_out)
        for (int m = 0; m < n_mat; m++) splits_out[m] = fl.splits[m];
    fused_layer_gemv_kernel<256>
        <<<static_cast<unsigned>(total), 256, 0, st>>>(fl, wt);""",
     """    fl.part = part;
    if (splits_out)
        for (int m = 0; m < n_mat; m++) splits_out[m] = fl.splits[m];
    // gate/up followed by the activation: both matrices read the same
    // activation and neither needs split-K, so one warp can compute the same
    // row of both and apply silu in the epilogue -- one launch where the
    // separate path runs three (gate, up, silu_mul).
    if (silu && n_mat == 2 && n_out[0] == n_out[1] &&
        w_row_bytes[0] == w_row_bytes[1] && in[0] == in[1] &&
        fl.splits[0] == 1 && fl.splits[1] == 1) {
        fused_gate_silu_kernel<256><<<static_cast<unsigned>((n_out[0] + 7) / 8),
                                      256, 0, st>>>(
            w[0], w[1], wt, n_out[0], n_in[0], w_row_bytes[0], in[0], out[0]);
        return true;
    }
    fused_layer_gemv_kernel<256>
        <<<static_cast<unsigned>(total), 256, 0, st>>>(fl, wt);"""),
    ("""                part + fl.part_off[m], fl.splits[m], fl.n_out[m],
                fl.n_out[m], out[m]);
#endif
}""",
     """                part + fl.part_off[m], fl.splits[m], fl.n_out[m],
                fl.n_out[m], out[m]);
#endif
    return false;
}"""),
])

# ---- backend.hpp: the silu request and its answer ---------------------------
patch("include/krk/backend.hpp", [(
    """    virtual bool gemm_group(void *const out[], const void *x,
                            const void *const w[], DType wt,
                            const i64 n_out[], i64 n_in, int n_mat,
                            i64 rows) {
        for (int m = 0; m < n_mat; m++)
            gemm(out[m], x, w[m], wt, n_out[m], n_in, rows);
        return false;
    }""",
    """    // `silu` asks for silu_mul() of matrices 0 and 1 to be folded into the
    // group's epilogue: out[0] then receives silu(out[0]) * out[1] and the
    // caller must not issue that activation itself. *silu_fused reports whether
    // that happened -- the group can still have fused without it (a shape or a
    // split-K decision it cannot express), which is why the return value alone
    // is not the answer. The default does the per-matrix gemms and leaves the
    // activation to the caller, so a backend without the folded form behaves
    // exactly as one that never had it.
    virtual bool gemm_group(void *const out[], const void *x,
                            const void *const w[], DType wt,
                            const i64 n_out[], i64 n_in, int n_mat,
                            i64 rows, bool silu = false,
                            bool *silu_fused = nullptr) {
        (void)silu;
        if (silu_fused) *silu_fused = false;
        for (int m = 0; m < n_mat; m++)
            gemm(out[m], x, w[m], wt, n_out[m], n_in, rows);
        return false;
    }""")])

# ---- backend_hip.hip: switch, member, override, pass-through ----------------
patch("src/hip/backend_hip.hip", [
    ("""        const char *age = std::getenv("KRK_ACC_GEMM");
        acc_gemm_ = !age || std::strcmp(age, "0") != 0;""",
     """        const char *age = std::getenv("KRK_ACC_GEMM");
        acc_gemm_ = !age || std::strcmp(age, "0") != 0;
        // Fused gate/up + silu_mul (kernels/fused.hpp, fused_gate_silu_kernel):
        // the FFN's two projections and the activation between them ride one
        // launch on the decode row. KRK_FUSED_SILU=0 restores the separate
        // silu_mul() call (A/B and fallback switch).
        const char *fse = std::getenv("KRK_FUSED_SILU");
        fuse_silu_ = !fse || std::strcmp(fse, "0") != 0;"""),
    ("    bool acc_gemm_ = true;",
     """    bool acc_gemm_ = true;
    // Fold silu_mul into the gate/up group's epilogue (KRK_FUSED_SILU=0
    // restores the separate activation launch).
    bool fuse_silu_ = true;"""),
    ("""    bool gemm_group(void *const out[], const void *x,
                    const void *const w[], DType wt,
                    const i64 n_out[], i64 n_in, int n_mat,
                    i64 rows) override {
        KRK_TIME_OP("gemm_group");""",
     """    bool gemm_group(void *const out[], const void *x,
                    const void *const w[], DType wt,
                    const i64 n_out[], i64 n_in, int n_mat,
                    i64 rows, bool silu = false,
                    bool *silu_fused = nullptr) override {
        KRK_TIME_OP("gemm_group");
        if (silu_fused) *silu_fused = false;"""),
    ("""        fused_layer_launch(wp, in, op, n_out, nin, rb, n_mat,
                           split_dev_, split_cap_ / sizeof(f32),
                           caps_.cu_count, static_cast<int>(wt));
        return true;
    }""",
     """        const bool silu_done = fused_layer_launch(
            wp, in, op, n_out, nin, rb, n_mat, split_dev_,
            split_cap_ / sizeof(f32), caps_.cu_count, static_cast<int>(wt), 0,
            nullptr, nullptr, fuse_silu_ && silu);
        if (silu_done && silu_fused) *silu_fused = true;
        return true;
    }"""),
])

# ---- engine.cpp: dense_ffn asks for the folded activation -------------------
patch("src/engine.cpp", [(
    """    if (L.wgate.type == L.wup.type) {
        void *gu[2] = {ws_gate_, ws_up_};
        const void *wgu[2] = {L.wgate.data, L.wup.data};
        const i64 ngu[2] = {n_ff_, n_ff_};
        be_->gemm_group(gu, xn, wgu, L.wgate.type, ngu, n_embd_, 2, n);
    } else {
        be_->gemm(ws_gate_, xn, L.wgate.data, L.wgate.type, n_ff_, n_embd_, n);
        be_->gemm(ws_up_, xn, L.wup.data, L.wup.type, n_ff_, n_embd_, n);
    }
    be_->silu_mul(ws_gate_, ws_gate_, ws_up_, n * n_ff_);""",
    """    // The activation rides the fused launch's epilogue where the backend can
    // express it (one launch for gate, up and silu instead of three); the flags
    // say which half happened, so the fallback issues exactly the calls the
    // folded path skipped. The folded form leaves ws_up_ untouched, and nothing
    // reads it after this point either way.
    bool silu_done = false;
    if (L.wgate.type == L.wup.type) {
        void *gu[2] = {ws_gate_, ws_up_};
        const void *wgu[2] = {L.wgate.data, L.wup.data};
        const i64 ngu[2] = {n_ff_, n_ff_};
        be_->gemm_group(gu, xn, wgu, L.wgate.type, ngu, n_embd_, 2, n, true,
                        &silu_done);
    } else {
        be_->gemm(ws_gate_, xn, L.wgate.data, L.wgate.type, n_ff_, n_embd_, n);
        be_->gemm(ws_up_, xn, L.wup.data, L.wup.type, n_ff_, n_embd_, n);
    }
    if (!silu_done) be_->silu_mul(ws_gate_, ws_gate_, ws_up_, n * n_ff_);""")])
