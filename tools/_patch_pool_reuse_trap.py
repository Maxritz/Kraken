import sys

p = 'tools/bench_ops.cpp'
raw = open(p, 'rb').read().decode('utf-8')
assert raw.count('\n') == raw.count('\r\n'), 'expected CRLF throughout'

def crlf(t):
    return t.replace('\n', '\r\n')

TRAP = '''    {   // A cell reused while its old reader is still QUEUED must not be
        // overwritten under that reader. The pool hands a released cell back out
        // carrying the mark its readers were released under, and every write path
        // is supposed to wait for it. The pinned-DMA arm a pinned WARM promotion
        // takes (`upload_pinned_batch`) issues its copy on a private NON-BLOCKING
        // stream and did not, so the DMA lands on bytes a queued expert read is
        // still reading. The reads here are 256 copies of one cell queued BEFORE
        // its release (so its mark covers them) and then a different pattern is
        // written into the very same cell.
        constexpr size_t kTrap = 4u << 20;  // 4 MiB: ~200 us of pinned DMA
        constexpr int kReaders = 256;        // ~1 GiB of reads: ~2-4 ms queued
        constexpr int kOuts = 8;
        const size_t elems = kTrap / 2;      // f16 elements
        std::vector<u16> pat_a(elems, fp32_to_fp16(1.0f));
        std::vector<u16> pat_b(elems, fp32_to_fp16(-1.0f));
        be->set_expert_pool_budget(64u << 20);
        void *trap_cell = be->alloc_pooled(kTrap);
        be->upload(trap_cell, pat_a.data(), kTrap);
        void *outs[kOuts] = {};
        for (int i = 0; i < kOuts; i++) outs[i] = be->alloc(kTrap);
        void *pinned = be->alloc_host(kTrap);
        std::memcpy(pinned, pat_b.data(), kTrap);
        for (int r = 0; r < kReaders; r++)
            be->copy_act(outs[r % kOuts], trap_cell, static_cast<i64>(elems));
        be->release_pooled(trap_cell, kTrap);
        void *reused = be->alloc_pooled(kTrap);
        const auto pb0 = be->pinned_dma_batches();
        void *t_dst[] = {reused};
        const void *t_src[] = {pinned};
        const size_t t_len[] = {kTrap};
        be->upload_pinned_batch(t_dst, t_src, t_len, 1);
        be->sync();
        std::vector<u16> tgot(elems);
        int mixed = 0;
        for (int i = 0; i < kOuts; i++) {
            be->download(tgot.data(), outs[i], kTrap);
            if (tgot != pat_a) ++mixed;
        }
        check(reused == trap_cell,
              "the trap's cell is the one handed back out (a fresh cell tests nothing)");
        check(be->pinned_dma_batches() == pb0 + 1,
              "the trap took the pinned DMA arm (a fallback path would pass vacuously)");
        check(mixed == 0,
              "a cell reused under a queued reader is not overwritten by the next DMA");
        for (int i = 0; i < kOuts; i++) be->release(outs[i]);
        be->release_host(pinned);
        be->release_pooled(reused, kTrap);
        be->set_expert_pool_budget(4096);
    }
'''

anchor = crlf('''    be->release(source); be->release(output);
    be->set_expert_pool_budget(0);
''')
assert raw.count(anchor) == 1, 'release anchor'
raw = raw.replace(anchor, crlf(TRAP) + anchor)

a2 = crlf('''    bool floor = false;
    bool h2d = false;
''')
assert raw.count(a2) == 1, 'decl anchor'
raw = raw.replace(a2, crlf('''    bool pool_test = false;
''') + a2)

a3 = crlf('''        else if (a == "--sweep") sweep = true;
''')
assert raw.count(a3) == 1, 'parse anchor'
raw = raw.replace(a3, a3 + crlf('''        else if (a == "--pool") pool_test = true;
'''))

a4 = crlf('''    if (h2d_noise_arm) return h2d_noise(be, h2d_noise_mib);
''')
assert raw.count(a4) == 1, 'dispatch anchor'
raw = raw.replace(a4, a4 + crlf('''    // The expert pool's own traps, runnable alone: the full gate also drives a
    // CLI speculation arm, which is minutes when all this needs is the device.
    if (pool_test) {
        std::printf("\\nexpert-pool (device):\\n");
        const i64 bad = gate_expert_pool(be);
        delete be;
        if (bad) {
            std::fprintf(stderr, "\\nEXPERT-POOL FAILED: %lld check(s)\\n",
                         static_cast<long long>(bad));
            return 1;
        }
        return 0;
    }
'''))

open(p, 'wb').write(raw.encode('utf-8'))
print('patched', p)
