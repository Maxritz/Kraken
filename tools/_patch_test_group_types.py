"""Add the gemm_group per-matrix-type regression test to tests/test_kraken.cpp.

The defect it guards: gemm_group's per-matrix fallback (which is what a
multi-row / prefill call takes, since the fused path is decode-only) passed the
group's single `wt` to every matrix. On Qwen3.5-0.8B -- whose GDN layers group
attn_qkv (Q5_K) with ssm_alpha/ssm_beta (Q4_K) -- that decoded two Q4_K matrices
as Q5_K during the prompt: KRK_DUMP gdn.delta read -nan(ind) against
-1.66893e-06 from the separate-launch arm, while the op count and the timing
looked healthy.

The test lives in test_gdn_ops rather than in a new group so the suite's SHAPE is
unchanged (a new group is a shape change the baselines must be re-recorded for;
an added check is a count note). It uses the CPU backend, like every group in
this suite -- the suite never opens the device -- so its reference is
`Backend::gemm_group`'s default implementation, the very code that was wrong.

Written as a file, not a heredoc: the payload carries C++ string literals and
line-ending-sensitive anchors, and this repo has a long note on what a heredoc
does to those. Per-file line endings, asserted anchor, /tmp backup.
"""
import io
import shutil

CR = "\r\n"
PATH = "tests/test_kraken.cpp"

NEW_FN = """// gemm_group must honour EACH MATRIX's own weight type -- on the fused path and
// on the per-matrix fallback, which is what a multi-row (prefill) call takes,
// because the fused kernel expresses the decode row only.
//
// This is a regression test for a real defect. The fallback used to pass the
// group's single `wt` to every matrix. Measured on Qwen3.5-0.8B, whose GDN
// layers group attn_qkv (Q5_K) with ssm_alpha and ssm_beta (Q4_K): the two Q4_K
// matrices were decoded as Q5_K during the prompt, KRK_DUMP's gdn.delta read
// -nan(ind) against -1.66893e-06 from the separate-launch arm, and the op count
// and the timing looked healthy the whole way.
//
// The reference is therefore the SAME backend's per-matrix gemm() with each
// matrix's own type, never the group against itself -- a group compared with
// itself agrees with any bug it carries. Three formats with three different
// geometries (F16 at 2 bytes a value, F32 at 4, Q8_0 at 34 bytes per 32 values)
// make a misread unambiguous, and the last check proves the misread is visible
// at all, so the comparison above cannot pass by being blind.
static void test_gdn_group_matrix_types() {
    Backend *cpu = make_cpu_backend();
    const i64 n_in = 64, n_out = 8; // 64 = two Q8_0 blocks per row

    CHECK(dtype_row_bytes(DType::F16, n_in) != dtype_row_bytes(DType::F32, n_in),
          "F16 and F32 rows are different lengths");
    CHECK(dtype_row_bytes(DType::F32, n_in) != dtype_row_bytes(DType::Q8_0, n_in),
          "F32 and Q8_0 rows are different lengths");

    // Every buffer gets slack: the sensitivity check at the end reads one
    // format's bytes as another's, and that must be wrong WITHOUT running off
    // the allocation.
    std::vector<u8> w16(static_cast<size_t>(n_out * n_in * 2) + 512, 0);
    std::vector<u8> w32(static_cast<size_t>(n_out * n_in * 4) + 512, 0);
    std::vector<u8> w8(static_cast<size_t>(n_out * (n_in / 32) * 34) + 512, 0);
    for (i64 r = 0; r < n_out; r++) {
        for (i64 c = 0; c < n_in; c++) {
            const f32 v = static_cast<f32>((r * 5 + c * 3) % 29) - 14.0f;
            const u16 h = fp32_to_fp16(v);
            std::memcpy(&w16[static_cast<size_t>(r * n_in + c) * 2], &h, 2);
            std::memcpy(&w32[static_cast<size_t>(r * n_in + c) * 4], &v, 4);
        }
        for (i64 b = 0; b < n_in / 32; b++) {
            u8 *blk = &w8[static_cast<size_t>(r * (n_in / 32) + b) * 34];
            const u16 h = fp32_to_fp16(0.25f);
            blk[0] = static_cast<u8>(h & 0xFFu);
            blk[1] = static_cast<u8>(h >> 8);
            for (int i = 0; i < 32; i++)
                blk[2 + i] =
                    static_cast<u8>(static_cast<i8>((i * 7 + r * 3) % 61 - 30));
        }
    }

    const void *w[3] = {w16.data(), w32.data(), w8.data()};
    const DType types[3] = {DType::F16, DType::F32, DType::Q8_0};
    const i64 n_out_arr[3] = {n_out, n_out, n_out};
    const i64 shapes[2] = {4, 1}; // prefill (the fallback) and the decode row

    for (int s = 0; s < 2; s++) {
        const i64 rows = shapes[s];
        const size_t n_el = static_cast<size_t>(rows * n_out);
        std::vector<f32> x(static_cast<size_t>(rows * n_in));
        for (size_t i = 0; i < x.size(); i++)
            x[i] = static_cast<f32>(static_cast<int>(i % 19) - 9) * 0.125f;

        std::vector<f32> got[3];
        void *out[3];
        for (int m = 0; m < 3; m++) {
            got[m].assign(n_el, -12345.0f);
            out[m] = got[m].data();
        }
        // The group's own `wt` names the F16 matrix on purpose, so a fallback
        // that leans on it misreads the other two.
        cpu->gemm_group(out, x.data(), w, DType::F16, n_out_arr, n_in, 3, rows,
                        /*silu=*/false, /*silu_fused=*/nullptr,
                        /*row_parallel_only=*/false, types);

        for (int m = 0; m < 3; m++) {
            std::vector<f32> ref(n_el, 0.0f);
            cpu->gemm(ref.data(), x.data(), w[m], types[m], n_out, n_in, rows);
            int bad = 0;
            for (size_t i = 0; i < n_el; i++)
                if (got[m][i] != ref[i]) bad++;
            CHECK(bad == 0,
                  rows == 1
                      ? "gemm_group's decode row matches per-matrix gemm() with "
                        "each matrix's own type"
                      : "gemm_group's multi-row fallback matches per-matrix "
                        "gemm() with each matrix's own type");
        }
    }

    // Sensitivity: reading the F32 matrix with the F16 type -- the shape of the
    // real bug -- has to produce something different, or the checks above would
    // pass whether or not the fallback honours per-matrix types.
    {
        const i64 rows = 4;
        const size_t n_el = static_cast<size_t>(rows * n_out);
        std::vector<f32> x(static_cast<size_t>(rows * n_in), 0.25f);
        std::vector<f32> as_f32(n_el, 0.0f), as_f16(n_el, 0.0f);
        cpu->gemm(as_f32.data(), x.data(), w32.data(), DType::F32, n_out, n_in,
                  rows);
        cpu->gemm(as_f16.data(), x.data(), w32.data(), DType::F16, n_out, n_in,
                  rows);
        int diff = 0;
        for (size_t i = 0; i < n_el; i++)
            if (as_f32[i] != as_f16[i]) diff++;
        CHECK(diff > 0,
              "an F32 matrix read as F16 differs, so a type mix-up is visible");
    }
}

static void test_gdn_ops() {
    test_gdn_group_matrix_types();
"""


def main():
    s = io.open(PATH, encoding="utf-8", newline="").read()
    crlf = s.count(CR)
    assert crlf > s.count("\n") - crlf, "expected CRLF on disk"
    shutil.copyfile(PATH, "/tmp/bak_test_kraken.cpp")
    old = "static void test_gdn_ops() {"
    assert s.count(old) == 1, "test_gdn_ops definition not found"
    new = NEW_FN.replace("\n", CR).rstrip(CR)
    s = s.replace(old, new)
    io.open(PATH, "w", encoding="utf-8", newline="").write(s)
    print("patched", PATH)


main()
