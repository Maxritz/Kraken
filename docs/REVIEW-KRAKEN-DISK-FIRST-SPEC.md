# Review notes on KRAKEN_DISK_FIRST_FULL_PATCH_SPEC.md

Read: 2026-10-23
Source spec: `KRAKEN_DISK_FIRST_FULL_PATCH_SPEC.md` (user-provided; original local path is not included in this project).

## Status
Read and treated as review guidance, not as a patch to apply verbatim: the spec's claims about the current source and local model corpus needed reconciliation with this checkout.

## Shortlist of what the spec gets right
- Prefer speed without silently changing quant arithmetic or token choices.
- Keep the corpus on disk; do not page-fault the whole file into RAM at startup.
- Sparse, coalesced, shard-aware reads; bounded HOT/WARM/COLD working sets.
- Separate correctness gates from throughput claims; run long-prompt benchmarks, not only 51–55-token ones.
- Refuse incomplete architectures with named blockers; do not flip a table entry to Yes because a loader parsed tensors.
- Per-model inventory from GGUF metadata, not filenames, with shard/sidecar identity handled separately.

## What needs reconciliation before treating as a patch plan
- §1 troubleshoot table: good instincts, but a few "current local state" claims need a HEAD check — especially the `qwen4exp` Yes/refused wording against this tree (here the current source already has `qwen4exp` as `ArchSupport::Yes` with a whole-graph fixture note in docs/AUDIT-qwen4exp-support.md, while some other docs here still say refused).
- §5 shard-aware pull: the *direction* is correct (route reads through all shards' mappings, not just the metadata shard's handle). The spec's specific current-source narrative is partly imprecise against this tree:
  - the current merger merges split shards into one tensor directory, and `gguf_.file()` at load is the first/merged mapping, i.e. the metadata shard for a split set, so a single-path pull will indeed miss tensors in the split.* shards — that part of the diagnosis is directionally right;
  - but the earlier Laguna-S refusal was dominated by arch-feature gaps (attn_gate, exp_probs_b, window, hybrid head-count), not solely by the single-path pull, so "Laguna-S failure mode" is an imprecise framing.
- §2 benchmark matrix is stricter and better than current scripts, but the spec does not quote current local numbers for this checkout, so P0/P2 passages that require a fresh matrix are not yet evidenced by the document itself.
- §14 model list is the spec author's list, not necessarily the actual shoulder-surfing collection here; de-duplicate by canonical path and strong file identity, not display names.

## Changes applied from the review
- Added shard-aware weight-source routing for dense pull reads and expert WARM reads.
- Kept the single-source backend hook as a compatibility wrapper.
- Replaced the machine-specific inventory script assumptions with a checkout-relative scanner that invokes the current text-based `kraken-inspect` CLI.
- Added a local inventory artifact and kept absent requested filenames distinct from discovered files.

## Still open
- Test the shard-aware path end to end with a real split GGUF; none was available in the worktree.
- Reconcile the older broad architecture notes when their model corpus is available again.
- Run long-prompt performance/coherence checks against representative full-size models before making any speed or correctness claim for those models.
