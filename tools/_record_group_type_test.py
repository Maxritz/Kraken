"""Record the mixed-dtype gemm_group regression test.

Numbers are from this session's runs (/tmp): the group count, the suite total,
the reintroduced-bug run, and the baseline re-record.
"""
import io

REPORT_SECTION = """

---

## 10. The regression test for section 9's bug

`tests/test_kraken.cpp`, group `test_gdn_ops`:
**`test_gdn_group_matrix_types`**. It runs three projections through one
`gemm_group` call -- F16 (2 bytes a value), F32 (4) and Q8_0 (34 bytes per 32
values), so three different bytes-per-row -- at **rows 4** (the per-matrix
fallback, which is what prefill takes) and **rows 1**, and compares each output
against *that same backend's* per-matrix `gemm()` with the matrix's own type.
The group's own `wt` deliberately names the F16 matrix, so a fallback that leans
on it misreads the other two.

The reference is never the group against itself: a group compared with itself
agrees with any bug it carries. That is the whole point of the case -- the
section-9 defect left the op count, the timing and the fused decode path all
looking healthy, and only a comparison against the *other* implementation
exposed it.

It lives in `test_gdn_ops` rather than in a group of its own so the suite's
shape is unchanged: a new group is a shape change the baselines must be
re-recorded for, an added check is a count note. It uses the CPU backend, as
every group in this suite does -- the suite never opens the device -- so its
reference is `Backend::gemm_group`'s default implementation, the code that was
wrong.

### The constructed failure

Reintroducing the defect (the default passing the group's `wt` to every matrix)
and rebuilding:

| build | result |
|---|---|
| fix in place | `[group test_gdn_ops] 138/138 checks` |
| bug reintroduced | `FAIL test_kraken.cpp:2639` x4 -- **134/138 checks**, and the four are exactly the two matrices whose type differs from the group's, at both row counts |

So the case is not an assertion that happens to hold: it is one that fails, in
the right place, on the defect it exists for. (The device-side half of the same
defect -- the HIP override's fallback -- is not reachable from this suite, since
no group opens the device; that half is pinned by the `KRK_DUMP`/md5 evidence in
section 9.)

### Suite and baselines

- full suite: **2760/2760 checks, rc=0** (2751 + the 9 this case adds)
- `tests/kraken-tests-baseline.json` re-recorded deliberately
  (`KRK_TEST_BASELINE_RECORD=1`, which refuses a run that is not green), then
  verified plainly: `verdict: no regression, ... 0 arm(s) newly broken`
"""

AGENTS_BULLET = """- **`test_gdn_ops` carries a mixed-dtype `gemm_group` case, and it fails on the
  defect it was written for.** `test_gdn_group_matrix_types` runs three
  projections (F16, F32, Q8_0 -- three different bytes per row) through
  `gemm_group` with an explicit `wts[]`, at rows 4 (the per-matrix fallback, the
  shape prefill takes) and rows 1, and compares each against that same backend's
  per-matrix `gemm()` with its own type; a fourth check proves the misread is
  visible at all. Reintroducing the bug (the fallback handing the group's single
  `wt` to every matrix) fails it **134/138 with four FAILs**, on exactly the two
  matrices whose type differs from the group's -- which is why the reference is
  never the group against itself. The device-side half is unreachable from the
  suite (no group opens the device) and is pinned by `KRK_DUMP` + md5 instead.
"""


def main():
    s = io.open("sherlock_report.md", encoding="utf-8", newline="").read()
    if not s.endswith("\n"):
        s += "\n"
    io.open("sherlock_report.md", "w", encoding="utf-8", newline="").write(
        s + REPORT_SECTION.lstrip("\n"))
    print("report section 10 appended")

    a = io.open("AGENTS.md", encoding="utf-8", newline="").read()

    old_wm = ("  missing dequantizers landed, 2316 -> 2329 when Q2_0's geometry changed, and\n"
              "  2329 -> 2751 when the KV-tier per-layer geometry and the sparse-KV fixture\n"
              "  tests landed. Read the count off the run (`2751/2751 checks passed`) rather\n"
              "  than trusting the number written here, which is only the last one seen.")
    new_wm = ("  missing dequantizers landed, 2316 -> 2329 when Q2_0's geometry changed,\n"
              "  2329 -> 2751 when the KV-tier per-layer geometry and the sparse-KV fixture\n"
              "  tests landed, and 2751 -> 2760 when the mixed-dtype `gemm_group` case below\n"
              "  landed. Read the count off the run (`2760/2760 checks passed`) rather than\n"
              "  trusting the number written here, which is only the last one seen.")
    assert a.count(old_wm) == 1, "watermark anchor"
    a = a.replace(old_wm, new_wm)

    anchor = "- **`env A=0,B=0 cmd` is ONE variable, not two.**"
    assert a.count(anchor) == 1, "bullet anchor"
    a = a.replace(anchor, AGENTS_BULLET + anchor)
    io.open("AGENTS.md", "w", encoding="utf-8", newline="").write(a)
    print("AGENTS.md watermark updated and bullet inserted")

    try:
        log = io.open("sherlock-kraken-session2.log", encoding="utf-8", newline="").read()
    except FileNotFoundError:
        print("no session log")
        return
    if not log.endswith("\n"):
        log += "\n"
    io.open("sherlock-kraken-session2.log", "w", encoding="utf-8", newline="").write(
        log + """
================================================================================
H. Regression test: mixed-dtype gemm_group vs its own per-matrix fallback
================================================================================

H1. tests/test_kraken.cpp, group test_gdn_ops -> test_gdn_group_matrix_types.
    Three matrices through one gemm_group call with an explicit wts[]:
    F16 (2 B/value), F32 (4 B/value), Q8_0 (34 B per 32 values). The group's own
    wt names the F16 matrix. Compared at rows 4 (the per-matrix fallback, the
    prefill shape) and rows 1, against the SAME backend's per-matrix gemm() with
    each matrix's own type -- never the group against itself.

    $ ./build-hip/kraken-tests.exe --run-group test_gdn_ops
      [group test_gdn_ops] 138/138 checks

H2. Constructed failure. Reintroduced the defect in the base-class default
    (gemm(out[m], x, w[m], wt, ...) instead of wts ? wts[m] : wt), rebuilt:

      FAIL tests/test_kraken.cpp:2639  gemm_group's multi-row fallback matches
           per-matrix gemm() with each matrix's own type     (x2)
      FAIL tests/test_kraken.cpp:2639  gemm_group's decode row matches
           per-matrix gemm() with each matrix's own type     (x2)
      [group test_gdn_ops] 134/138 checks

    Four FAILs, on exactly the two matrices whose type differs from the group's,
    at both row counts -- the defect signature. Restored, rebuilt, 138/138.

H3. Suite and baselines on the final tree:
      kraken-tests                2760/2760 checks passed, rc=0
      KRK_TEST_BASELINE_RECORD=1  suite baseline re-recorded (2760/2760)
      plain re-run                rc=0, "no regression, ... 0 newly broken"
""")
    print("session log section H appended")


main()
