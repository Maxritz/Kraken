Aftermath note: user's RTX 4070 SUPER result and the DeepEP/EPLB/PD framing.

What the user reported
- RTX 4070 SUPER, 12 GB VRAM, 32 GB RAM.
- Strata 0.2 is an app now; its model list includes a new Qwen 3.8 Flash-Next file called Gyro-S.
- Same 6 jobs as every model on the board.
- A 58 GB file that finally fits 12 + 32 GB, "except half of it lives on your SSD."
- Result: 21.3 tok/s, zero loops, and "for this file 32 GB of RAM is not enough."

How to read that against the DeepEP/EPLB/PD discussion
- The user's result is the real single-device boundary the DeepSeek serving papers are mostly not written for. DeepEP/EPLB/PD are multi-rank, multi-GPU ideas; on one card the relevant question is tiering and residency, not all-to-all dispatch.
- "Half of it lives on your SSD" is COLD/spill behavior, and "32 GB of RAM is not enough" is a hard working-set boundary at that run's shape. That is exactly the regime where a single-card engine needs staged loads, promotion/eviction policy, and reader overlap, not a new dispatch fabric.
- The 21.3 tok/s number is believable as a single-device result in that regime; the honest follow-up is whether the slow part is SSD/COLD reads, promotion/eviction churn, or router/plan latency, and that is measurable with existing per-phase stats rather than by importing a multi-GPU serving stack.

What I actually added here
- A scoping note that DeepSeek V2-Lite MLA is the realistic first target because the lite layout already runs, and that a first diff should be a CPU-oracle whole-graph test for that layout, not a V4 graph port: docs/DEEP-SEEK-V2-LITE-MLA-NOTE.md
- A note that DeepEP/EPLB/PD are framing, not a copyable mechanism on this box, and that the portable bits are prefill/decode phase awareness and a residency/load-balance view: docs/DEEP-EP-EPLB-PD-HONEST-FRAMEWORK.md
- A short paper-compare doc already on disk from the earlier turn: docs/three-moe-prefetch-papers.md

What remains undecided
- Whether to treat the user's RTX result as a Strata-vs-kraken comparison task, a kraken tiering/budget task, or just a data point. That is a scope decision, not something to infer from the papers alone.
- Whether to chase V4 at all, given it is still refused by name here for the MLA/compressed-attention gap plus the type-#26 tensor class. The honest position is that it stays refused until a staged gate passes; DeepEP/EPLB/PD do not change that.
