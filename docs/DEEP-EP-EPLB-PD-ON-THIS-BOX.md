Point: comparison of DeepEP/EPLB/PD disaggregation and kraken's current single-card expert-tier path, on this machine (gfx1201, 15.9 GiB VRAM, 96 GB host RAM).

What DeepEP/EPLB/PD are
- DeepEP: expert-parallel all-to-all routing/dispatch + expert weight exchange across ranks, with NVLink/RDMA in mind.
- EPLB: load balancing for expert-parallel prefill/decode, including slow/mean replica placement metrics (their reported 1.045-1.090 slowest/mean).
- PD disaggregation: prefill and decode run on separate device pools so a prefill-heavy step does not starve decode latency.

Why they do not transfer 1:1 to kraken
- kraken's present DeepSeek-relevant path is not expert-parallel across cards. The local target family that already runs is w4exp-style single-device MoE (named in the arch table, runnable). The single-card design has no multi-rank all-to-all, no NVLink fabric, and no prefill/decode pool split at the infrastructure level.
- Their load-balance numbers are about balancing across replicas/ranks, not about a single device's expert-cache residency. On one card the comparable control knob is expert-tier residency (HOT/WARM/COLD) plus route/plan latency, which is the actual measured bottleneck here.

What can be borrowed as a design idea
- Separating prefill and decode intent is still a real idea, even on one card, because prefill calls a different expert-union and row-count pattern than decode. kraken already has the tier machinery; the useful adaptation is making the prefetch/staging policy aware of prefill-vs-decode phase, not copying an all-to-all dispatch.
- Load-balance style analysis can still be repurposed: single-card "balance" becomes "how much of a layer's routed set is resident vs promoted per step," which is measurable from the existing stats.
- PD-style disaggregation is not implementable as server/infra disaggregation here, but a local version could mean: do not promote decode-oriented residency choices using prefill-only traffic, and vice versa. That is a policy question the tier could track.

What is out of scope on this box
- Any DeepEP all-to-all kernel recipe, because there is no multi-device NVLink path; copying CUDA-centric exchange code would not run on gfx1201 without a HIP rewrite, and would not help a single device anyway.
- EPLB metrics as if they were speedups: they measure cross-rank balance, not single-card expert latency.
- PD disaggregation as a deployment model: kraken is not currently a multi-pool serving runtime.

Bottom line
Take the framing, not the mechanism: prefill/decode phase awareness and a residency/load-balance view are portable ideas. The CUDA/Multi-GPU all-to-all and pool-splitting machinery is not a direct inspiration for kraken on this machine, and the local evidence here is that the real single-card gap is expert-tier residency plus route/plan latency, not cross-rank dispatch.
