"""Group the GDN layer's four shared-activation projections into one launch.

Qwen3.5-0.8B's recurrent layers issue four `gemm(gemv)` calls per layer that all
read the attention-norm output and have no dependency between them ([q|k|v], the
z gate, ssm_alpha and ssm_beta) -- the condition `gemm_group` was built for, and
54 of that model's 451 ops per decode step.

Split-K is pinned OFF for this group (`row_parallel_only`): the group's usual
top-up fills the device for a matrix with too few row-blocks, and ssm_alpha /
ssm_beta are dt_rank rows -- it would buy parallelism with a reduce launch per
split matrix, which is the launch this removes, and regroup the fp32 partials
while doing it. With splits at 1 each row keeps gemv_kernel's accumulation
order, so the group is bit-identical to the four launches it replaces, and
stdout md5 is the falsifier.

Per-file line endings, asserted anchors, /tmp backups (AGENTS.md, "Patching
files safely").
"""
import io
import shutil

CR = "\r\n"
FILES = ["include/krk/backend.hpp", "src/hip/backend_hip.hip", "src/engine.cpp"]


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
    shutil.copyfile(f, "/tmp/gdnbak_" + f.replace("/", "_"))
print("backed up %d files to /tmp" % len(FILES))

# ---- backend.hpp ------------------------------------------------------------
patch("include/krk/backend.hpp", [(
    """    virtual bool gemm_group(void *const out[], const void *x,
                            const void *const w[], DType wt,
                            const i64 n_out[], i64 n_in, int n_mat,
                            i64 rows, bool silu = false,
                            bool *silu_fused = nullptr) {
        (void)silu;
        if (silu_fused) *silu_fused = false;""",
    """    // `row_parallel_only` pins every matrix in the group to one row-parallel
    // pass (no split-K top-up). A caller that groups matrices too small to fill
    // the device -- a recurrent layer's dt_rank-wide projections, say -- wants
    // the launch count down, and the top-up would spend a reduce launch per
    // split matrix to buy parallelism it does not need; it also regroups the
    // fp32 partials, so a group with splits at 1 is bit-identical to the
    // separate launches while a split one is not.
    virtual bool gemm_group(void *const out[], const void *x,
                            const void *const w[], DType wt,
                            const i64 n_out[], i64 n_in, int n_mat,
                            i64 rows, bool silu = false,
                            bool *silu_fused = nullptr,
                            bool row_parallel_only = false) {
        (void)silu;
        (void)row_parallel_only;
        if (silu_fused) *silu_fused = false;""")])

# ---- backend_hip.hip --------------------------------------------------------
patch("src/hip/backend_hip.hip", [
    ("""    bool gemm_group(void *const out[], const void *x,
                    const void *const w[], DType wt,
                    const i64 n_out[], i64 n_in, int n_mat,
                    i64 rows, bool silu = false,
                    bool *silu_fused = nullptr) override {
        KRK_TIME_OP("gemm_group");
        if (silu_fused) *silu_fused = false;""",
     """    bool gemm_group(void *const out[], const void *x,
                    const void *const w[], DType wt,
                    const i64 n_out[], i64 n_in, int n_mat,
                    i64 rows, bool silu = false,
                    bool *silu_fused = nullptr,
                    bool row_parallel_only = false) override {
        KRK_TIME_OP("gemm_group");
        if (silu_fused) *silu_fused = false;"""),
    ("""        const bool silu_done = fused_layer_launch(
            wp, in, op, n_out, nin, rb, n_mat, split_dev_,
            split_cap_ / sizeof(f32), caps_.cu_count, static_cast<int>(wt), 0,
            nullptr, nullptr, fuse_silu_ && silu);""",
     """        // blk_target 1 means "one row-parallel pass per matrix": the split
        // policy is `row_blocks < blk_target`, and no matrix has fewer than one
        // row-block.
        const bool silu_done = fused_layer_launch(
            wp, in, op, n_out, nin, rb, n_mat, split_dev_,
            split_cap_ / sizeof(f32), caps_.cu_count, static_cast<int>(wt),
            row_parallel_only ? 1 : 0, nullptr, nullptr,
            fuse_silu_ && silu);"""),
])

# ---- engine.cpp: the group, and the two gemms it replaces -------------------
patch("src/engine.cpp", [
    ("""    // 1. One projection for [q | k | v] and one for the z gate. Both read the
    //    same attention-norm output, so they are independent. Folded ones read
    //    the transformed copy; ssm_alpha/ssm_beta further down read ws_xn_
    //    primal, which is exactly why the copy is a copy.
    const void *xn = ws_xn_;
    if (L.h_attn_in || L.h_attn_gate) {
        had_xform(ws_had_, ws_xn_, n, n_embd_, false, false);
        xn = ws_had_;
    }
    be_->gemm(ws_qkv_, L.h_attn_in ? xn : ws_xn_, L.wqkv.data, L.wqkv.type,
              cdim, n_embd_, n);
    be_->gemm(ws_z_, L.h_attn_gate ? xn : ws_xn_, L.wqkv_gate.data,
              L.wqkv_gate.type, vdim, n_embd_, n);""",
     """    // 1. FOUR projections read the attention-norm output here, not two: the
    //    [q | k | v] block and the z gate, and (below, in step 4) ssm_alpha and
    //    ssm_beta. None of them depends on another, and none of them reads
    //    anything the conv, the L2 norms or the gate chain writes, so they are
    //    one group launch -- 4 launches per layer become 1, which is 54 of
    //    Qwen3.5-0.8B's 451 ops in a decode step.
    //
    //    Row-parallel only, deliberately: ssm_alpha/ssm_beta are dt_rank rows,
    //    so the group's usual split-K top-up would fill the device for them and
    //    pay a reduce launch per split matrix for the privilege -- buying
    //    parallelism with exactly the launch this removes. With the splits
    //    pinned at 1 every row keeps gemv_kernel's accumulation order, so the
    //    group is BIT-IDENTICAL to the four launches it replaces; the stdout
    //    md5 is the check, because a regrouping would show up as different
    //    greedy text.
    //
    //    A folded q/k/v or z reads the hadamard copy (ws_had_) while alpha and
    //    beta read ws_xn_ primal, and a group shares one activation, so the
    //    group covers the unfolded case -- which is also the only case where
    //    all four read the same buffer. A dtype mix falls back too: the group
    //    carries one weight type.
    const void *xn = ws_xn_;
    if (L.h_attn_in || L.h_attn_gate) {
        had_xform(ws_had_, ws_xn_, n, n_embd_, false, false);
        xn = ws_had_;
    }
    static const bool gdn_group_ = [] {
        const char *e = std::getenv("KRK_GDN_GROUP");
        return !e || std::strcmp(e, "0") != 0;
    }();
    const bool gdn_onelaunch =
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
                        /*row_parallel_only=*/true);
    } else {
        be_->gemm(ws_qkv_, L.h_attn_in ? xn : ws_xn_, L.wqkv.data, L.wqkv.type,
                  cdim, n_embd_, n);
        be_->gemm(ws_z_, L.h_attn_gate ? xn : ws_xn_, L.wqkv_gate.data,
                  L.wqkv_gate.type, vdim, n_embd_, n);
    }"""),
    ("""    be_->gemm(ws_ssm_, ws_xn_, L.ssm_alpha.data, L.ssm_alpha.type, mc.ssm_dt_rank,
              n_embd_, n);
    be_->add_bias_cols(ws_ssm_, L.ssm_dt, n, mc.ssm_dt_rank);""",
     """    // ssm_alpha came out of the group launch above when it ran.
    if (!gdn_onelaunch) {
        be_->gemm(ws_ssm_, ws_xn_, L.ssm_alpha.data, L.ssm_alpha.type,
                  mc.ssm_dt_rank, n_embd_, n);
    }
    be_->add_bias_cols(ws_ssm_, L.ssm_dt, n, mc.ssm_dt_rank);"""),
    ("""    be_->gemm(ws_beta_, ws_xn_, L.ssm_beta.data, L.ssm_beta.type, mc.ssm_dt_rank,
              n_embd_, n);
    be_->sigmoid_act(ws_beta_, static_cast<i64>(n) * mc.ssm_dt_rank);""",
     """    if (!gdn_onelaunch) {
        be_->gemm(ws_beta_, ws_xn_, L.ssm_beta.data, L.ssm_beta.type,
                  mc.ssm_dt_rank, n_embd_, n);
    }
    be_->sigmoid_act(ws_beta_, static_cast<i64>(n) * mc.ssm_dt_rank);"""),
])
