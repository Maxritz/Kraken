# Review — `kraken_compressed_expert_tiered_residency_spec.md` (rev 1.0)

**Reviewed:** `C:\Users\rr\Downloads\kraken_compressed_expert_tiered_residency_spec.md`,
2511 lines, rev 1.0 dated 2026-10-04.

**Verdict.** The evidence audit is honest and the roadmap ordering is right, but
about half the document specifies machinery Kraken already has, and the part
that is genuinely new (§35's logical/physical split, §27's budget arithmetic) is
smaller than the part that does not apply. Two of the four research pillars —
expert-only quantisation and no-FP16-materialisation — are the input format
rather than an implementation task. The ternary pillar is blocked by a
measurement of ours, not by effort.

Read §35 and §27 as the deliverable; read §8/§12/§13/§15/§25 as a research
programme that should not start yet.

---

## 1. Arithmetic audit of §30

Every figure in the Qwen3-30B-A3B table was recomputed. Params per expert
`2048 x 768 x 3 = 4,718,592` and the routed bank `x 128 x 48 = 28,991,029,248`
are exact. Bits-per-weight rows:

| Representation | spec | recomputed | |
|---|---:|---:|---|
| BF16 | 54.0 GiB | 54.00 GiB | ok |
| W8 + FP16 scales, G=128 | 27.42 GiB | 27.42 GiB | ok |
| W6 + FP16 scales, G=128 | 20.67 GiB | 20.67 GiB | ok |
| W4 + FP16 scales, G=128 | 13.76 GiB | **13.92 GiB** | ~1% low |
| ternary at 1.796 bpw | 6.06 GiB | 6.06 GiB | ok |

The W4 row is the only slip and it is immaterial, but the row that matters for
Kraken is one the table does not have: see §3.1 below.

---

## 2. What Kraken already implements

| spec | Kraken | status |
|---|---|---|
| §26 direct representation; I2 no FP16 on the miss path | experts live in the GGUF mapping already quantised (Q4_K/Q5_K/Q6_K); `QuantTensor` holds packed device bytes and kernels dequantise in-register | **already true by construction** — the prohibited `FP16 RAM → CPU quantise → repack` path does not exist |
| §16 warm RAM tier, bounded pinned pool | `ExpertCache` second tier is byte-budgeted page-locked host memory (`host_budget_`); only that many bytes are ever pinned | present (Correction 5 already policy) |
| §39 weighted LRU; §18 slot states | LFU + aging + **pins** (`kPinThreshold=24`, decay window scaled to slot count). Pins are the mechanism the spec lacks: they encode "long-serving favourite, do not sacrifice to a cold token" | **ahead of** the spec's "improved" tier, lands near "predictive" minus the predictor |
| §44 VRAM full / RAM full | `make_vram_room` / `make_host_room`, `demote_to_host` / `promote_to_vram`; a demoted expert keeps its counter, pin and load order | present |
| §49 telemetry | `loads/evictions/hits/host_hits/demotions/promotions/acquires/decays/pinned_slots` all exist and are printed by `main_cli` | counters present, no aggregation |
| §43 exactness vocabulary | the verdict system is built on exactly this distinction (`ArchSupport::No` refuses rather than decodes nonsense) | present |
| §27 GPU budget from free memory | `Engine::configure_expert_cache`: `vram_free * 0.9`, clamped to the whole routed bank when it fits | present for VRAM only |

So §26, §58 Correction 5, most of §18/§39/§44 and the VRAM half of §27 are
already shipped. Ticking them off in an appendix checklist would misrepresent
where the remaining work is.

---

## 3. Where the spec's premise does not match our measurements

### 3.1 Expert-only W4 is already the input format

Phase 1 asks for "offline W4 packing" and §30 prices it against BF16. Kraken
never has BF16 experts: the GGUFs we load already store
`blk.N.ffn_{gate,up,down}_exps.weight` in **Q4_K, 4.5 bpw**, with the dense trunk
in Q5_K/Q6_K. The comparison that decides whether to build a new store is not
4.125 vs 16.0 bpw, it is 4.125 vs 4.5:

```
Q4_K   4.5   bpw -> 15.19 GiB
W4 G128 4.125 bpw -> 13.92 GiB
delta            ->  1.27 GiB  (9.1% of the W4 store)
```

A 9% storage win for a compiler, a manifest format, a new file layout and a new
grouped GEMM. That is a bad first trade, and it is invisible from the spec
because the spec is written against a BF16 source that Kraken's inputs already
left behind. The place 9% is worth having is a model that does not otherwise
fit; nothing on this machine is in that position yet.

### 3.2 The residency thesis is a decode thesis; our bottleneck is prefill

`ExpertCache`'s own header states the mechanism: top-k routing touches every
expert within a few dozen tokens. That is true at batch 1. At prefill it is not
a tendency but an identity — 256 experts, top-10, thousands of tokens, so the
*required* set per layer approaches the entire bank, and every placement policy
in §19/§39/§20 degenerates to "everything is hot". A cache cannot help a
workload whose working set is the whole model.

Consequence for the roadmap: §20's prediction levels, §38's horizons and §51's
hit-rate metrics are decode-window metrics and must be *measured* in the decode
window. They must not be validated against TTFT, where our attention kernel is
95% of device time and expert residency is a rounding error. The spec's
acceptance criteria ("measurable reduction in expert storage and H2D traffic",
§53) are satisfiable while end-to-end prefill gets no faster at all.

This also reorders the goal. If the object is "run a 30B-A3B on 16 GiB", the
first lever is **bytes per expert** (3.1, ternary) and the second is **grouped
prefill execution**, not the NVMe tier.

### 3.3 BITCOS currently selects five-trit for every tensor we own

Two independent reasons, and the spec supplies the first itself (§58
Corrections 1–3, §11).

1. BITCOS is lossless **w.r.t. an already-ternary tensor**. Every non-ternary
   checkpoint would have to be ternarised first — a CAT-Q-like calibration
   pipeline, which by §11 is an EXPERIMENTAL compiler mode and by §54 Phase 4 is
   three phases away.
2. We measured the quantity that makes the choice: `tools/probe_zero_blocks.cpp`
   decodes every block with the engine's own decoder and counts all-zero blocks.

   | model | type | blocks | all-zero blocks |
   |---|---|---:|---:|
   | Qwen3-MoE-4x0.6B | Q4_K | 4,129,678 | **0 (0.000%)** |
   | Qwen3-MoE-4x0.6B | Q6_K | 656,519 | **0 (0.000%)** |
   | probe_TQ2_0 | TQ2_0 | 408,576 | **0 (0.000%)** |

   Element-wise zeros are plentiful (0.01% Q4_K, 2.68% Q6_K, 88–98% TQ2_0) but
   a presence bitmap is block- or page-granular. At 0.000% block density the
   mask is all-ones and BITCOS degenerates to `mask + sign` — strictly worse than
   five-trit packing by exactly the mask it added.

So §10's rule is not wrong, it simply answers "five-trit" for every tensor we
have. The rule is worth keeping in the compiler as a measured branch; it is not
worth building the BITCOS encoder for.

Note also that the ternary path *is* already partly real here: TQ1_0/TQ2_0 decode
end to end (`DType` 34/35, host + device), verified against ggml's own reference
encoder. But the one real ternary file that loads (`probe_TQ2_0.gguf`) generates
garbage text, so **fixing that outranks adding a second ternary storage format** —
a new format on top of an unverified path is two unknowns.

---

## 4. Adoptable now, in this order

### A. Size the host tier and turn it on (spec §27, §28)

`--expert-l2-mb` defaults to **0**, i.e. the measured ~1.4x-decode second tier is
shipped but off, and when a user does enable it they must guess a number.
§27/§28's subtraction model is the right shape for that default:

```
warm = min( profile_target, total_expert_bytes() )
profile_target = RAM_total - OS_headroom - KV_host - pinned_IO - reserve
```

Clamping to `total_expert_bytes()` is what makes this safe: when the whole bank
fits in the warm target the tier never evicts, so enabling it is lossless and
cannot regress anything. Cheapest item in this document; measure-only.

### B. Split the logical cache object from the physical transfer (§35)

This is the sharpest idea in the document for our code, and the one with no
Kraken equivalent. Today a re-acquire after eviction issues three small reads
per expert (gate/up/down are three separate tensors, non-adjacent in the file)
through a per-thread `FileReader`. But expert slices inside
`ffn_gate_exps.weight` are at a fixed stride, so a *run* of consecutive expert
ids is one contiguous extent. A miss should fetch `[e, e+k)` in one aligned read
and land the neighbours directly in the warm tier, with `k` chosen to make an
extent ~4–16 MiB from the measured per-expert bytes.

This is the same insight as the AGENTS.md weight-pull note (small pages, large
transfers) applied one level down, and it composes with A: coalesced extents
only matter once there is a tier to land them in.

### C. Compute the L2 budget from free memory (§27)

Kraken does this for VRAM and not for host. Same model, plus a safety margin so
an allocation cannot force emergency eviction.

### D. Aggregate the counters into a residency report (spec §49, §51; Phase 0)

Every counter exists and none of them are presented as a profile. Per §51 the
reporting that matters is the *split* — compute / H2D / staging / scheduling —
and AGENTS.md is explicit that the wall time of the thing being changed must be
instrumented directly rather than inferred from thread-summed stage timers. This
is the spec's Phase 0 and it is correctly placed first: without it, A and B are
unmeasurable.

### E. Vocabulary (§43)

`complete-expert availability`, `routing-faithful execution`, `numerical
equivalence` are the right three names for what Kraken already guarantees and
refuses. Cheap, and it makes the guarantees legible in docs.

---

## 5. Defer or skip

| spec | why |
|---|---|
| §8 compiler, §12 sensitivity, §13 allocator, §15 calibration, §42 Q2/Q3 gates | an offline toolchain plus a calibration corpus and a perplexity/task harness Kraken does not have. Highest cost in the document; §55 correctly says do not start here |
| §22 CPU fallback | Fiddler's premise is activation ≪ expert bytes, which fails in the prefill window that dominates our workload |
| §17 NVMe tier, §45 crash recovery, §47 `storage/` tree | pays only when the routed bank exceeds RAM (>70 GiB of experts on this machine). One model class away, not now, and it needs async I/O plumbing that does not exist |
| §25 AMD BITCOS kernel, §24 `KRK_T3_5TRIT` / `KRK_T3_BITCOS` | blocked on 3.3 and on the open `probe_TQ2_0` correctness bug |
| §47 module layout | Kraken is a flat `src/` + `include/krk/`; a 30-file `moe/` tree for work the engine does inline is churn |
| §30/§31 profile tables as constants | fine as documentation of intent; must not be hard-coded (§28 says so itself) |

---

## 6. One gap the spec does not cover

The document is written as if the expert bank were the only large object. On the
model that motivated this round — Laguna-S, 73 GB across three shards — the dense
trunk and the 100352x3072 token embedding are a large fraction of the bytes, and
Kraken has a **known** limitation there: the `read()`-based weight pull is
inactive on split sets, because it is installed against the metadata shard whose
data section is empty (see STATUS.md §10a). Every tensor therefore falls back to
direct `hipMemcpy` from a cold mapping, which per AGENTS.md costs one soft fault
per 4 KiB page (~0.4 us each) and is the bottleneck rather than the transfer.

A first touch of a 73 GB model is exactly the workload where that difference is
minutes versus hours. §35's insight applies verbatim to dense weights, and the
non-MoE weight-residency policy already on the list is the same fix.

---

## 7. Bottom line

- Adopt: **A** (host tier default), **B** (coalesced extents), **C** (L2 budget
  arithmetic), **D** (residency report), **E** (vocabulary).
- Keep as a written rule: §10's density threshold, §35's granularity split,
  §43's three exactness terms, §58's seven corrections.
- Do not start: the compiler/allocation/calibration programme, the NVMe tier,
  BITCOS.
- Correct the premises: §30's W4-vs-BF16 comparison is against a precision
  Kraken never loads (§3.1); the residency metrics are decode metrics and must
  not be validated on TTFT (§3.2); BITCOS as specified selects five-trit for
  every tensor we own (§3.3).

The single highest-value sentence in the document is §35's *"the logical cache
unit is an expert; the physical I/O unit may contain multiple adjacent expert
objects"* — that is a change to `ExpertCache`, not a new subsystem.
