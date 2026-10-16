#!/usr/bin/env python3
"""bench: measure the burst release and gate the shared mark."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "tools/bench_ops.cpp"
raw = p.read_text(newline="")
eol = "\r\n" if "\r\n" in raw else "\n"
s = raw.replace("\r\n", "\n")

pairs = []

# 1. the burst primitive, next to the single-cell round trip it replaces
pairs.append((
    """    for (int i = 0; i < kCells; i++) be->release_pooled(cells[i], kCell);
    be->release(src);
    be->release(sink);
    const auto st = be->expert_pool_stats();""",
    """    // The shape the engine actually drives at a constrained budget: an eviction
    // returns an expert's THREE slices in one call. Both arms below hand back the
    // same three cells with a reader queued on the default stream, so they are
    // one call apart, and the counters say how many records each issued -- the
    // mechanism, not just a time.
    constexpr int kBurst = 3;
    size_t blen[kBurst];
    for (int i = 0; i < kBurst; i++) blen[i] = kCell;
    void *bc[kBurst];
    be->set_expert_pool_budget(kCell * (kCells + kBurst));
    for (int i = 0; i < kBurst; i++) {
        bc[i] = be->alloc_pooled(kCell);
        be->fill0(bc[i], kCell);
    }
    const auto marks0 = be->expert_pool_stats();
    const double burst_ms = time_calls(be, n, [&] {
        be->copy_act(sink, bc[0], 64);
        be->release_pooled_batch(bc, blen, kBurst);
        for (int i = 0; i < kBurst; i++) bc[i] = be->alloc_pooled(kCell);
    });
    const auto burst_st = be->expert_pool_stats();
    const double sep_ms = time_calls(be, n, [&] {
        be->copy_act(sink, bc[0], 64);
        for (int i = 0; i < kBurst; i++) be->release_pooled(bc[i], kCell);
        for (int i = 0; i < kBurst; i++) bc[i] = be->alloc_pooled(kCell);
    });
    const auto sep_st = be->expert_pool_stats();
    for (int i = 0; i < kBurst; i++) be->release_pooled(bc[i], kCell);

    for (int i = 0; i < kCells; i++) be->release_pooled(cells[i], kCell);
    be->release(src);
    be->release(sink);
    const auto st = be->expert_pool_stats();"""))

# 2. print it
pairs.append((
    """    std::printf(
        "pool round trip (%d cells of %d KiB, budget %d KiB): %.2f us ready | "
        "%.2f us with a queued reader | alloc %llu reuse %llu trim %llu\\n",
        kCells, static_cast<int>(kCell / 1024),
        static_cast<int>(kCell * kCells / 1024), ready_ms * 1000.0,
        reader_ms * 1000.0, static_cast<unsigned long long>(st.allocations),
        static_cast<unsigned long long>(st.reuses),
        static_cast<unsigned long long>(st.trims));""",
    """    std::printf(
        "pool round trip (%d cells of %d KiB, budget %d KiB): %.2f us ready | "
        "%.2f us with a queued reader | alloc %llu reuse %llu trim %llu\\n",
        kCells, static_cast<int>(kCell / 1024),
        static_cast<int>(kCell * kCells / 1024), ready_ms * 1000.0,
        reader_ms * 1000.0, static_cast<unsigned long long>(st.allocations),
        static_cast<unsigned long long>(st.reuses),
        static_cast<unsigned long long>(st.trims));
    // Sequenced, so this ranks the two arms and does not time the machine: the
    // verdict on the engine comes from the interleaved A/B, and the counter
    // ratio below is the part that is a fact either way.
    std::printf(
        "pool burst (%d slices of %d KiB, a queued reader): %.2f us one mark | "
        "%.2f us one record per slice | %llu vs %llu mark(s) for %llu slice(s)\\n",
        kBurst, static_cast<int>(kCell / 1024), burst_ms * 1000.0,
        sep_ms * 1000.0,
        static_cast<unsigned long long>(burst_st.release_marks -
                                        marks0.release_marks),
        static_cast<unsigned long long>(sep_st.release_marks -
                                        burst_st.release_marks),
        static_cast<unsigned long long>(sep_st.release_slices -
                                        marks0.release_slices));"""))

# 3. the gate: one mark per burst, and every cell of the burst keeps it
pairs.append((
    """    void *staged = be->alloc_pooled(1024);
    void *dst[] = {staged};""",
    """    {   // The burst's shared mark has to be honored by EVERY writer that reuses
        // one of its cells, not only the first one handed out. Three cells are
        // read on the default stream, released in ONE call, then each handed back
        // out and zeroed before those reads can have completed: every reader must
        // still see its own bytes. This is the burst form of the single-cell check
        // above, and the counter check is the sharing the whole change is for.
        void *bc[3] = {}, *bo[3] = {};
        size_t bl[3] = {1024, 1024, 1024};
        std::vector<u16> bgot[3] = {std::vector<u16>(512), std::vector<u16>(512),
                                    std::vector<u16>(512)};
        for (int i = 0; i < 3; i++) {
            bc[i] = be->alloc_pooled(1024);
            bo[i] = be->alloc(1024);
            be->copy_act(bc[i], source, 512);
            be->copy_act(bo[i], bc[i], 512);
        }
        const auto b0 = be->expert_pool_stats();
        be->release_pooled_batch(bc, bl, 3);
        const auto b1 = be->expert_pool_stats();
        check(b1.release_marks - b0.release_marks == 1 &&
                  b1.release_slices - b0.release_slices == 3,
              "a three-slice burst shares one release mark");
        bool reused[3] = {false, false, false};
        for (int i = 0; i < 3; i++) {
            void *q = be->alloc_pooled(1024);
            for (int j = 0; j < 3; j++)
                if (q == bc[j]) reused[j] = true;
            be->fill0(q, 1024);
            be->release_pooled(q, 1024);
        }
        check(reused[0] && reused[1] && reused[2],
              "every cell of a burst is handed back out");
        for (int i = 0; i < 3; i++) be->download(bgot[i].data(), bo[i], 1024);
        be->sync();
        for (int i = 0; i < 3; i++) be->release(bo[i]);
        check(bgot[0] == input && bgot[1] == input && bgot[2] == input,
              "every cell of a burst keeps its mark: no reader is overwritten");
    }
    void *staged = be->alloc_pooled(1024);
    void *dst[] = {staged};"""))

for old, new in pairs:
    n = s.count(old)
    if n != 1:
        sys.exit(f"anchor found {n} times: {old[:70]!r}")
    s = s.replace(old, new)

p.write_text(s.replace("\n", eol), newline="")
print(f"tools/bench_ops.cpp: {len(pairs)} replacement(s)")
