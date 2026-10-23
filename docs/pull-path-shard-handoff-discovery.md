# Shard-aware HIP pull/read handoff — discovery notes

This note records the shard-aware weight-pull handoff investigation and the
implemented path. The backend gets each shard mapping and resolves read requests
against the owning mapping; see the implementation in `include/krk/backend.hpp`,
`src/model.cpp`, and `src/hip/backend_hip.hip`.

## 1. What the rename from `src/h2d_probe.cpp` to `src/hip/h2d_probe.cpp` actually is

- GGUF parsing is already the shard-aware side of this. `src/gguf.cpp` / `include/krk/gguf.hpp`
  load every sibling by name, merge the tensor directory, and keep per-shard mappings
  alive (`GgufShard { file, data_off, split_no, n_tensors }`).
- Each `GgufTensor` carries `u32 shard`, the shard index its `data` pointer belongs to.
  That is the source of truth for "which file does this pointer live in."
- `Gguf::data_base()` / `data_section_off()` deliberately mean the **metadata shard**
  (shard 0), not "every tensor's base." That is why a single-base pointer subtraction
  is wrong for a non-zero shard.

## 2. Previous handoff contract (the single-base assumption)

Before the shard-aware patch, `include/krk/backend.hpp` exposed one optional
`set_weight_pull(path, map_base, map_size, data_off)` source. `src/model.cpp`
passed the metadata shard through that hook, and the HIP backend accepted only
pointers contained by that mapping. A pointer in another shard could not be
resolved to its owning path/offset and therefore lost the file-read fast path.

The current implementation adds `WeightSource` and `set_weight_sources()`;
`Model::load()` passes the path and mapping for every loaded shard, and the HIP
backend resolves the source by integer address-range containment before pull or
WARM reads. `set_weight_pull()` remains as a compatibility wrapper for one
source. A real split GGUF remains untested in this checkout.

## 3. Why the old single-base assumption failed

On a split set, shard 0 holds metadata and its own tensors. Other shards have
independent mappings, data offsets, and tensor entries. The old backend only
stored shard 0, so pointers from a different shard could not be translated to
that shard's file path and offset. The GGUF loader already merged metadata and
kept every shard mapping alive; the gap was the backend's single-source handoff.

## 4. The applied implementation

The change makes pointer-to-offset resolution shard-aware while keeping a
single-source compatibility wrapper. The real model load smoke test covered a
single-file model; a real multi-file GGUF is not available in this worktree and
remains an explicit verification gap.

### Applied — backend shard table with range containment

- `WeightSource` provides the path, map base, map size and data-section offset.
- The HIP backend finds the owning source by integer range containment and derives the file offset from that map base.
- `set_weight_pull()` remains a single-source compatibility wrapper.
- The existing per-worker `FileReader` design stays in place; workers reopen their handle when a tensor belongs to another shard.
- Batched WARM reads group requests by source shard and sort by file offset within that shard.

## 5. Verification results

### 5.1 Build check

`sh scripts/build_check.sh build-name-hpc` passed after rebuilding the six
requested binaries, with all six fingerprints verified on the second run.

### 5.2 The loader's verdict stays the same and the read path handles split files

- A single-file GGUF loads and generates with the HIP backend: the local SmolLM2 smoke test completed successfully.
- Split-file pointers are routed through the owning-source table in code, but a real split GGUF fixture was not available, so split reads are not yet end-to-end verified.
- The inventory scanner was smoke-tested on both local GGUF fixtures; it records the metadata-based verdict and does not infer architecture from filenames. See `docs/QWEN-FAMILY-DISAMBIG.md` for the family distinction.

### 5.3 Tests and baseline

The latest full suite run reports 3723/3724 checks passed, with one failure in
`test_moe_schema_and_laziness`; the baseline comparison identifies it as one
newly broken group. It also reports one new group and three baseline groups
missing from the current test runner. Focused load-verdict,
GGUF roundtrip, expert-budget and architecture-table groups pass. The test failure
reproduces on the earlier `build-name` binary as well; it is not evidence about a
split read. No assertion or baseline was weakened.

### 5.4 No regressions on the real-file shapes already proven

The single-file HIP smoke test passed on the local 135M GGUF. The full large-model
coherence checks could not be rerun here; see the final verification summary for
remaining blockers.

## 6. Relevant implementation locations

- Backend contract: `include/krk/backend.hpp` (`WeightSource`, `set_weight_sources`).
- Loader handoff: `src/model.cpp`.
- HIP pull and WARM readers: `src/hip/backend_hip.hip`.
- GGUF shard metadata and mappings: `include/krk/gguf.hpp`, `src/gguf.cpp`.
- Local inventory and loader verdict tool: `reports/kraken_inventory.md`, `tools/kraken_inventory.log`, and `tools/inspect_gguf.cpp`.
