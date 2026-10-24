DeepEP / EPLB / PD disaggregation — what transfers, what does not

Short frame
- The user's result on a 12 GB RTX 4070 SUPER with 32 GB RAM is the right data point to take seriously: a 58 GB file that finally fits 12 + 32 GB "except half of it lives on your SSD", 21.3 tok/s, zero loops, and "32 GB of RAM is not enough". That is a real single-device boundary, and it is the regime where tier/prefetch ideas matter.
- The user also said: Strata 0.2 is an app now, with a new Qwen 3.8 Flash-Next file called Gyro-S, and they ran "the same 6 jobs as every model on my board." That is a comparison they own; I am not re-deriving it here.

What DeepEP/EPLB/PD are, in one line each
- DeepEP: expert-parallel all-to-all routing/dispatch and expert weight exchange across ranks, built around multi-GPU bandwidth (NVLink/RDMA-style).
- EPLB: load balancing for expert-parallel prefill/decode, including slow/mean replica placement metrics.
- PD disaggregation: prefill and decode run on separate device pools so prefill-heavy work does not starve decode latency.

What does NOT transfer 1:1 to kraken on this machine
- There is no multi-rank all-to-all here. kraken's present DeepSeek-relevant path that already runs is w4exp-style single-device MoE (named in the arch table, runnable). The single-card design has no NVLink fabric and no prefill/decode pool split at the infrastructure level, so the CUDA-centric exchange recipes are not copyable as-is and would not help a single device anyway.
- EPLB's load-balance numbers are about balancing across replicas/ranks, not about a single device's expert-cache residency. On one card the comparable control knob is expert-tier residency (HOT/WARM/COLD) plus route/plan latency, which is the actual measured bottleneck here.
- PD disaggregation as a deployment model is not kraken today: it is not a multi-pool serving runtime.

What CAN be borrowed as a design idea
- Separating prefill and decode intent is still a real idea even on one card, because prefill calls a different expert-union and row-count pattern than decode. kraken already has the tier machinery; the useful adaptation is making the prefetch/staging policy phase-aware (prefill vs decode), not copying an all-to-all dispatch.
- Load-balance style analysis can be repurposed: single-card "balance" becomes "how much of a layer's routed set is resident vs promoted per step," which is measurable from the existing stats.
- A local analogue of PD-style disaggregation would mean: do not let prefill-only traffic drive decode-oriented residency choices, and vice versa. That is a policy question the tier could track, not a new server architecture.

What is out of scope on this box
- Any DeepEP all-to-all kernel recipe, because there is no multi-device NVLink path; copying CUDA-centric exchange code would not run on gfx1201 without a HIP rewrite, and would not help a single device anyway.
- EPLB metrics as if they were speedups: they measure cross-rank balance, not single-card expert latency.
- PD disaggregation as a deployment model: kraken is not currently a multi-pool serving runtime.

Honest read on the user's RTX result
- "Half of it lives on your SSD" is exactly the COLD tier behavior kraken already has. The honest frame is: on a 12 GB card with 32 GB RAM, a 58 GB file is not going to be fully resident, so the system is doing staged loads and SSD spillover by necessity; whether the path is good depends on promotion/Eviction/reader overlap, not on a new all-to-all.
- "32 GB of RAM is not enough" is a concrete boundary: it says the file's working set still exceeds the combined VRAM+RAM tiering budget at the run's shape, which is the kind of data that should drive budget/scan/warmup policy, not a CUDA dispatch redesign.
- The 21.3 tok/s figure is believable as a single-device number in that regime; the important question is whether the path is CPU/SSD-limited, promotion-limited, or router/plan-limited, and that is measurable with the existing host-time/expert-path stats.

Bottom line
Take the framing, not the mechanism: prefill/decode phase awareness and a residency/load-balance view are portable ideas. The CUDA/multi-GPU all-to-all and pool-splitting machinery is not a direct inspiration for kraken on this machine, and the local evidence here is that the real single-card gap is expert-tier residency plus route/plan latency plus SSD spillover behavior, not cross-rank dispatch.
