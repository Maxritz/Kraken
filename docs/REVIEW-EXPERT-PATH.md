# Expert path code review: flow, defects, fixes

Scope: the MoE expert path end to end -- `Engine::moe_ffn` -> `ExpertCache` ->
`Backend` pool/copies -- plus the host RAM policy that sizes the WARM tier.
Every claim below was read in the code and, where it is a performance claim,
measured on `Qwen3.8-Flash-Next-REAP-320-YMQ-M-TI.gguf` (67 GB, 320 experts, 48
layers, top-10, `qwen4exp`) on an RX 9070 XT with 95.9 GiB host RAM.

## 1. Flow of one routed expert

```
Engine::moe_ffn(layer, n)                       src/engine.cpp ~6440
  router (device) -> moe_sel_ [n,k], moe_wt_
  ├─ hybrid split?  (OFF by default)            cfg.hybrid_experts == false
  ├─ prefetch_layer(warm_set)                   one batched file read for misses
  ├─ stage_batch / stage_ahead (opt-in)         DMA ahead of use
  └─ for each expert e in the routed set:
       acquire(src, layer, e)                   src/expert_cache.cpp ~1050
         ├─ owner_bytes(need) -> pool budget
         ├─ HOT  hit          -> return &s.m            (no work)
         ├─ staged          -> finish_stage()  (wait ticket)
         ├─ WARM hit        -> promote_to_vram()        [D1, D4]
         ├─ make_vram_room(need)                       [D5]
         ├─ alloc_pooled() x3                          [D2, D3, D6]
         ├─ upload from WARM or mapping                [D7]
         └─ slots_.emplace(key, s); return &s.m        [D4, D4b]
       gemm up/gate -> silu -> gemm down -> scatter
```

Eviction (`retire`) drops the device cells and keeps the WARM copy; `drop_host`
returns the RAM copy to the backend's deferred-free list; `forget_if_dead` erases
a slot that is in neither tier.

## 2. Defects found, with status

Severity: **S1** wrong answer or crash, **S2** silent resource/accounting
corruption, **S3** performance, **S4** measurement blind spot.

| # | Sev | Where | What it did | Status |
|---|-----|-------|-------------|--------|
| D1 | S1 | `acquire`, WARM hit | `promote_to_vram` failure fell through to the cold reload with the WARM slot still in the map. `emplace()` on an existing key is a no-op, so the function returned the OLD entry -- whose `ResidentExpert` is exactly what the failed promotion never built. Every GEMM then read null weight pointers, and the cells the reload allocated leaked. **This is the constrained-card path**, i.e. the one this engine exists for. | **fixed**: the stale slot is dropped (`drop_host` + `forget_if_dead`) before the reload, and `!ins.second` is now an explicit refusal that releases the cells |
| D2 | S1 | `acquire` cold path | `alloc_pooled` returns `nullptr` on a full card (by design); the answer was passed to `upload()` -> `hipMemcpy(nullptr)` -> `hipErrorInvalidValue` five tokens into a decode, reported against the upload line. | **fixed**: retry with `force_reclaim()`, then refuse the expert instead of the context |
| D3 | S2 | `acquire` cold path | A failed load still did `bytes_ += need; vram_slots_++` and inserted a slot with null slices, so the ledger claimed residency that did not exist -- which then justified evicting nothing. | **fixed** |
| D4 | S1 | `promote_to_vram` | Three cells allocated one by one; a refusal on slice 2 released nothing on the path that returns `false` from the *caller* (the WARM copy survived, cells were released -- verified correct, not a defect). | verified |
| D4b | S2 | `clear()` | `host_pinned_bytes_` was not reset while `host_bytes_` was. After a clear the pin prefix believes it is spent, so no later WARM admission is ever pinned. | **fixed** |
| D5 | S2 | `make_vram_room` | Decides only on `bytes_ + need > budget_`. The budget is a plan made before the KV cache and workspaces exist, so on a full card it evicts nothing and the *allocation* fails instead (measured: budget 7624 MiB, 7444 resident, 3202 free beside a 3261 floor). | **mitigated** by D2's reclaim loop; the clean fix is to test the live `free - floor` sample here (open) |
| D6 | S1 | `backend_hip.hip` `alloc_pooled` | `std::abort()` when `committed > budget - rounded`. With slabs this is a legitimate state (slab slack), and on a full pool it is the steady state. | **fixed** (abort applies to the per-cell allocator only; the arena refuses instead) |
| D7 | S1 | `engine.cpp` host policy | `host_stage_room = live - reserve - min(mapped, ram_total)`, i.e. the whole corpus charged against RAM the process does not own. Every model bigger than the ceiling got WARM = 0 -- and the "tier is disabled" warning was gated on `host_budget > 0`, so it printed nothing. | **fixed**; measured +46% on REAP-320, 2.2x on REAP-288 |
| D8 | S1 | `ExpertCache::stage_ahead` | Same-size-class eviction picked the target layer's own experts (one layer == one size class), so the pass evicted the set it was completing. | **fixed** (lease the target layer) |
| D9 | S1 | `backend_hip.hip` release path | `expert_arena()` gated BOTH the growth allocator and whether released cells were cached, so switching the allocator silently disabled reuse. Caught by the bench's `expert-lease`/`expert-pool` checks. | **fixed** (two decisions, two code paths) |
| D10 | S1 | `pool_slab_take_` reuse | A reused cell did not increment its slab's `live`, so a slab could be freed under an in-flight copy (`hipErrorInvalidValue` from the ring's `hipMemcpyAsync`). | **fixed** |

## 3. Verified correct (the review's negative results)

- `prefetch_layer` releases every host buffer on each error path and charges
  `host_bytes_`/`loads_`/`bytes_loaded_` only at commit, so a batch that fails
  halfway neither leaks nor inflates the residency report.
- `retire` keeps the WARM copy by construction (it only drops device cells), the
  pin counters follow the same `set_pinned` path, and `forget_if_dead` erases
  only slots in neither tier. No double free found in `clear()`/destructor.
- `warm_admit`'s only refusal is an expert larger than the whole tier, which is
  the documented policy; the per-admission ledger (`SharedHostRamBudget`) is what
  actually protects the machine, and it fails closed when RAM is unmeasurable.
- `moe_ffn` null-checks `acquire()` (`if (!re) continue;`), so a refused expert is
  skipped rather than dereferenced -- the D1 nulls were reachable only because
  `acquire()` returned a *non-null* pointer to an empty expert.
- The hybrid CPU arm is **off by default** (`EngineConfig::hybrid_experts =
  false`, both CLIs). A default run's host work is the copy path, not CPU expert
  compute; `--hybrid-experts 1` is the only way to get the latter.

## 4. Why it runs at 2-3 tok/s on this file (measured, not inferred)

Per token: ~240 expert promotions of ~2.2 MiB = ~520 MiB of RAM->VRAM traffic, of
which ~390 MiB also comes off disk on the first touch. On a pageable WARM tier
every promotion is a host `memcpy` into the ring plus a DMA (143 us per expert of
enqueue), so the host is the copy engine and the GPU waits on it.

| configuration | ms/step | note |
|---|---|---|
| WARM disabled (the shipped default before this review) | 206.8 | 387 MiB read per token |
| WARM as the policy intends | 141.3 | 27 MiB read per token |
| WARM + `KRK_HOST_PIN_MB=16000` (DMA promotions) | 107-127 | xfer 880 -> 400 ms per 16 tokens |

The remaining 110-140 ms is the copy bandwidth itself, which is why scheduling
tricks (next-layer prefetch, `-9%`) and allocator tricks (slab arena, alloc row
5.8x cheaper but step -13.8%) do not move it: they rearrange a cost that is
already saturating the link. Only fewer bytes per token change this number.

## 5. Open items, ranked by expected effect

1. **Bytes per expert per token.** The routed set is 42.2 GiB against a 7.4 GiB
   device budget; every token re-promotes 520 MiB. The levers that actually
   change the numerator: an expert-usage rank (`.krakenexperts.json`, the writer
   and reader both exist) to keep the *hot* 7 GiB resident, and a per-layer
   batched DMA (one `upload_paged_batch` of 3k slices instead of k batches of 3)
   to remove per-call enqueue cost.
2. **Live-sample eviction (D5).** `make_vram_room` should test
   `device_free_bytes() - vram_floor_bytes(total)` so a miss evicts before the
   allocation fails, instead of after.
3. **Surface a skipped expert.** `pool_refusals_` exists and is never printed;
   a run that silently drops an expert's contribution has no witness.
4. **Make the pinned prefix automatic when `large_bar == 0`**, bounded to
   min(WARM/4, 8 GiB). Measured +12%, but it changes host commitment for every
   model, so it wants a 16 GB-class validation first.
5. **Instrument the cold path.** With WARM off, the acquire-loop `upload()` is
   outside every timer, so a run that moved 387 MiB/token reported `read 0.0 ms`.

## 6. Reproduction

```
KRK_ARCH_ALLOW_EXPERIMENTAL=1 ./build-hip/kraken.exe \
  -m H:/OLLAMA-Models/GGUF/Qwen3.8-Flash-Next-REAP-320-YMQ-M-TI.gguf \
  -p "The capital of France is" -n 16 --greedy
```
Reports to read: `[stats ] cache`, `[stats ] expert-path`, `[stats ] transfer`,
`[stats ] experts`. A refused expert prints one warning naming the layer; the run
continues by design (`docs/AUDIT-REAP-DECODE.md` §2-§3 has the reso
resolution).
