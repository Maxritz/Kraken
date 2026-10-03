# Papers to implement

Every paper that has been sent, what it says (from the abstract, v1 of each —
none of these have been read past the abstract yet, so nothing here claims
more than that), and the concrete thing it would change in *this* codebase.
Ordered by how directly it applies to kraken today, not by date.

The short version of the ordering argument:

1. **Tail-Replay** repairs a *documented refusal* of ours (no partial rewind on
   recurrent models) and makes prefix caching work for Qwen3.5 — 25 of the 44
   runnable files in the collection.
2. **RadixMLP** turns a cost we already pay (shared prefixes re-computed per
   request) into a data-structure problem, and it fits the gather/scatter ops
   the MoE path already has.
3. **Feather + CHT** decides *what to batch*; without it, prefix reuse in a
   local single-user engine is a smaller win than in a datacenter — but the
   chunked hash tree is a cheap, useful index for our own prefix cache.
4. **Self-Indexing Attention** and **DeepSeek-V4.1-Flash KV compression** are the
   long-context memory story: sparse retrieval plus low-bit KV.
5. **REA** is context management for chat, a serving-layer feature, not a kernel.
6. **YOCO**, **SPLASH**, **VitaLLM** are architecture/hardware designs: read,
   track, and only act if a model of that shape appears in the folders.
7. **CodeTD** is a *quality* feature (hallucination detection) that needs
   attention maps exported; useful for a code assistant, not for the engine.

---

## 1. Tail-Replay — prefix caching for hybrid (GDN) models

**Paper.** *Tail-Replay: Escaping the Curse of Linear Attention in Prefix Caching
for Hybrid LLMs* — arXiv [2608.30310](https://arxiv.org/abs/2608.30310)
(Liu, Qi, Wu, Liu, Chen; 31 Aug 2026).

**What it says.** Hybrid models interleave full attention (whose KV cache is
token-addressable) with linear attention (whose recurrent state cannot be rolled
back to an arbitrary prefix). Existing prefix caching for these models stores
recurrent-state checkpoints, so reuse only works at checkpoint boundaries.
Tail-Replay observes that a gated recurrent state is a *lossy compression* of the
prefix, so the state at a matched prefix can be reconstructed by replaying a short
recent suffix. It caches the full-attention KV, stores no recurrent checkpoints,
and on a hit replays 5–10% of the prefix. Quality retained: 92.8–99.9% on
LongBench/RULER; time-to-first-token 9.1–14.3× faster than full prefill at 32K.

**Why it matters here.** kraken implements exactly this family (gated delta net
hybrids: qwen35, qwen35moe) and *refuses the thing this paper makes possible*.
`Engine::kv_rollback` warns "a recurrent model cannot rewind to position N — only
a full rewind rebuilds the delta-rule state", and the server's per-request
isolation falls back to replaying the whole prompt. Tail-Replay is the principled
version of that fallback: keep the token-addressable KV for the attention layers,
and rebuild the delta state from a bounded suffix instead of from zero.

**Where it would land.** `Engine::kv_rollback` (the refusal) and the server's
prefix-reuse path; the state arrays are already separated per recurrent layer
(`rec_state_span_`, `recurrent_index(i32)`), so a "replay last S tokens into the
state" routine is a small addition to the existing chunked prefill. What is
missing is the *cache*: a token-level prefix index and a store of the attention
KV plus the committed token ids.

**Status.** Not started; documented here as the first thing to build.

## 2. RadixMLP — deduplicate shared prefixes inside a forward pass

**Paper.** *RadixMLP — Intra-batch Deduplication for Causal Transformers* —
arXiv [2601.15013](https://arxiv.org/abs/2601.15013) (Feil, Lipp; 21 Jan 2026).

**What it says.** Batch inference re-computes identical MLP/LayerNorm/embedding
activations for every copy of a shared prefix. RadixMLP maps the batch onto a
prefix trie, computes the shared segments once, and scatters results back at
attention boundaries (the one stage that is not position-wise). Stateless, inside
one forward pass. On MS MARCO v1.1 with Qwen3 0.6B–8B: 1.44–1.59× end-to-end, up
to 5× on longer shared prefixes.

**Why it matters here.** The position-wise stages of our prefill (embedding,
RMSNorm, the FFN/MoE FFN, the output head) are exactly the ones that get no
benefit from prefix caching today, and our MoE path already has
`gather_rows`/`scatter_axpy_rows` — the same primitive this needs. With a
per-request chat template and a fixed system prompt, every turn re-runs the same
prefix.

**Where it would land.** `forward_core`'s per-layer loop, between attention
calls: a "segment plan" (row gather by prefix trie) applied to the position-wise
ops. Combined with Tail-Replay the two cover different halves of the problem —
Tail-Replay avoids recomputing *attention* over a cached prefix, RadixMLP avoids
recomputing the *position-wise* work even when the batch is heterogeneous.

**Status.** Not started. Needs a prefix trie; the gather/scatter ops exist.

## 3. Feather + CHT — what to batch, and a cheap prefix index

**Paper.** *Requests of a Feather Must Flock Together: Batch Size vs. Prefix
Homogeneity in LLM Inference* — arXiv
[2605.06046](https://arxiv.org/abs/2605.06046) (Rathi, Preeti, Vutukuru;
7 May 2026).

**What it says.** With prefix-sharing workloads, smaller *prefix-homogeneous*
batches decode faster than larger heterogeneous ones (better KV cache locality).
Existing schedulers only maximize prefix reuse to save memory, and detect shared
prefixes with radix-tree traversals whose CPU cost is comparable to GPU
execution. Feather is an RL-learned scheduler for the batch-size/homogeneity
tradeoff, and the Chunked Hash Tree (CHT) is a lightweight prefix index that
avoids tree traversals. 2–10× end-to-end in vLLM/SGLang versus existing
schedulers.

**Why it matters here.** kraken-server is the multi-request surface. The RL half
is not interesting for a workstation with one user and a handful of requests; the
**CHT** is: it is the index our own prefix cache (item 1) needs, and it is
deliberately cheap to maintain. Their finding that *homogeneity beats batch size*
is also a useful prior for how to schedule the server's queue.

**Where it would land.** A new prefix index owned by the server; the scheduler
chooses, per step, between requests that share a prefix. Not urgent until there
is real multi-request traffic.

**Status.** Deferred; the CHT design is worth copying when the prefix cache lands.

## 4. Self-Indexing Attention — sparse long-context with a 1-bit index

**Paper.** *Self-Indexing Attention for Compression-Compatible Sparse Long-Context
LLM Inference* — arXiv [2609.13205](https://arxiv.org/abs/2609.13205) (Yang et
al.; 16 Aug 2026).

**What it says.** A training-free framework built on one shared transform-domain
sign-magnitude representation: the key *signs* become a reusable 1-bit token
index for both grouped prefill selection and decode retrieval, and the same
representation stays compatible with external KV-cache compression (no separate
indexer metadata). At 5% density it stays close to dense attention on LongBench
and RULER, with up to 6.1× prefill and 10.3× decode attention-operator speedups.
Verified against TurboQuant and DeepSeekV4-Flash.

**Why it matters here.** Our attention op is a materialized softmax: every decode
step reads the whole KV cache. A 1-bit sign index would let a decode step compute
exact attention over a K-sized candidate set instead of all keys — the same shape
of change as sliding-window attention but data-dependent instead of positional,
and it composes with the KV-compression work below.

**Where it would land.** A new `attention_sparse` op (CPU reference + HIP),
plus an index built where K is written (`kv_append`). This is a large,
self-contained feature; it should follow the KV-compression decision, because the
index is defined on the transformed key representation.

**Status.** Not started; recorded as the long-context direction.

## 5. DeepSeek-V4.1-Flash — KV cache compression limits

**Paper.** *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression* —
arXiv [2609.19969](https://arxiv.org/abs/2609.19969) (DeepSeek-AI; 17 Sep 2026).

**What it says.** (Abstract page read; the compression recipe is the substance of
the paper, which has not been read in full.) The title states the claim: a
V4.1-Flash model trained to tolerate aggressive KV-cache compression.

**Why it matters here.** KV is the resource that decides context length on a
16 GiB card: for the 0.8B qwen35 the cache is 6 MiB at ctx 128 and grows linearly
(12 MiB in the CPU backend's f32 layout). Low-bit KV would multiply usable
context — and the paper's model is in the folders (`DeepSeek-V4-Flash-*`), so it
is also a customer for the LLaDA/MLA work.

**Where it would land.** The cache dtype is already a backend parameter
(`act_type`, f16 on GPU) — the step is a *block* KV format (per-head scales, e.g.
Q8_0-style) plus attention kernels that dequantize on the fly. That is the same
machinery the int8 DP4A GEMM work already built for weights
(`gemm_dp4a.hpp`), applied to K instead.

**Status.** Not started; needs the full paper read before designing.

## 6. REA — role-aware context management for chat

**Paper.** *Role-aware Heuristic Episodic Attention for Conversational LLMs* —
arXiv [2610.00958](https://arxiv.org/abs/2610.00958) (Hong et al.; 1 Oct 2026).

**What it says.** Multi-turn conversations decay in three ways (attention
pollution, dilution, drift). REA keeps instructions in a dedicated prefix and
represents episodic turns with per-turn policies: raw text, compressed, or
omitted. On Long-MT-Bench+: 6.32 → 7.36 judge score and 2.91× lower latency.

**Why it matters here.** This is prompt-side engineering, not kernel work, and it
is the cheapest big quality win available to the server: a chat session with a
stable instruction prefix and a bounded episodic window. It overlaps with our
prefix cache (stable prefix = reusable KV) and with speculative decoding (a
smaller, cached prefix means less prefill to redo).

**Where it would land.** `kraken-server`'s request handling and the ChatML
wrapper in the CLI: an instruction block that is never evicted, plus a per-turn
policy for what to keep. No kernel changes.

**Status.** Not started; no dependency on the engine work.

## 7. YOCO — decoder-decoder with one KV cache

**Paper.** *You Only Cache Once: Decoder-Decoder Architectures for Language
Models* — arXiv [2405.05254](https://arxiv.org/abs/2405.05254) (Sun et al.;
May 2024).

**What it says.** A self-decoder encodes global KV once; a cross-decoder reuses
that cache via cross-attention. Memory drops sharply, prefill can exit early
without changing the output, and it extends to 1M context with near-perfect
needle retrieval.

**Why it matters here.** It is an architecture, not a technique: it applies only
if a YOCO-shaped model shows up in the folders. The *early-exit* observation is
transferable in spirit to any two-stage stack (and to our chunked prefill), but
there is nothing to implement against the models we have.

**Status.** Reference only.

## 8. SPLASH — switching attention parallel layouts while serving

**Paper.** *SPLASH: Switching Parallel Layouts of Attention with Seamless Handoff
for LLM Serving* — arXiv [2609.37626](https://arxiv.org/abs/2609.37626) (Liu et
al.; 29 Sep 2026).

**What it says.** Attention layout (tensor/data/context parallel) should change
with the load, but engines fix it at launch because switching means draining.
SPLASH reuses most of the existing state and hands off at a batch boundary;
median switch overhead under 0.51% of a step. Its "decoupled ownership
parallelism" keeps each request's cache on one owner while sharding weights.
1.3–1.73× over fixed layouts on B200s.

**Why it matters here.** kraken is a single-GPU engine: there is no layout to
switch. The one transferable idea is the decoupling itself — where a request's KV
lives versus how weights are sharded — which is exactly the distinction the
expert cache already makes for MoE weights (weights stream, caches stay put).
**Status.** Not applicable today; noted for a future multi-device backend.

## 9. VitaLLM — mixed-precision edge accelerator

**Paper.** *VitaLLM: A Versatile and Tiny Accelerator for Mixed-Precision LLM
Inference on Edge Devices* — arXiv
[2605.00320](https://arxiv.org/abs/2605.00320) (Lin, Chang; 1 May 2026, ISCAS).

**What it says.** A 16 nm accelerator for ternary-weight LLMs: a multiplier-free
TINT core for ternary projections, a radix-4 Booth core shared between INT8
attention and ternary work, predictive sparse attention with a leading-one
surrogate and comparison-free top-K, verified by a silicon prototype
(72.46 tok/s decode on BitNet b1.58 3B within 0.214 mm²).

**Why it matters here.** It is a *hardware* design — we cannot implement it — but
two of its ideas are implementable in software on RDNA4: the **leading-one
surrogate for top-K selection** (a cheaper score than the full product, useful
for our router's top-k over 256 experts) and **radix-4 Booth** as an alternative
INT8 datapath, which is relevant because gfx12 has no `sdot4` (our DP4A port is
emulated — see docs/TODO.md). The ternary-BitNet direction also matches the
`ternary`/`Q1_0` files in the folders, which we currently refuse.

**Status.** Ideas noted; nothing to implement until a BitNet/ternary model is a
target.

## 10. CodeTD — detecting hallucinations from attention topology

**Paper.** *CodeTD: Topology of Attention Detects Hallucinations in Code LLMs* —
arXiv [2609.07779](https://arxiv.org/abs/2609.07779) (Voronkova et al.; 7 Sep
2026, EMNLP).

**What it says.** Pre-execution correctness assessment for generated code using
topological data analysis of attention maps: prompt-generation mismatch is
quantified from topological patterns, transferable across benchmarks, 5 languages
and 10 code LLMs up to 34B.

**Why it matters here.** It needs attention maps, which kraken throws away after
each layer. A `--dump-attn` capability plus this analysis would give a code
assistant a "this answer looks hallucinated" signal before running the code. That
is a product feature on top of the engine, and it depends on being able to export
attention cheaply (a debug path, not the hot path).

**Status.** Deferred; would need attention-map export first.

## 11. Self-Indexing Attention's sibling work — HPC-Ops exact Top-K

**Paper.** *Sample-Guided Exact Top-K Selection for Long-Context Sparse
Attention* — arXiv [2609.08450](https://arxiv.org/abs/2609.08450) (Liu et al.;
8 Sep 2026).

**What it says.** Sparse attention's exact Top-K stage must process score rows
whose length grows with context, and production radix selectors need a
full-row pass before they can refine. This work uses a fixed-stride sample to
propose a coarse row-local boundary, certifies it with the mandatory full-row
pass, then refines exactly in FP32 — sampling controls common-path work but never
correctness. 1.29–1.75× over the fastest verified exact baseline; implemented in
Tencent's HPC-Ops.

**Why it matters here.** Two uses: the attention top-K of the sparse path above,
and *our own* MoE router, which currently does a full exp/softmax over 256
experts and then an O(k·ne) selection per token on the host. A sampled-boundary
select with an exact certification pass is a direct fit for that loop, and the
"sampling never changes the answer" property is what makes it acceptable in a
router.

**Status.** Not started; small, self-contained, testable against the current
exhaustive selection.

## 12. Multilinguality in hybrid attention models — layer ordering

**Paper.** *Multilinguality in Hybrid Attention LLMs* — arXiv
[2609.35378](https://arxiv.org/abs/2609.35378) (Bandarkar et al.; 28 Sep 2026).

**What it says.** A study of how interleaving full attention with recurrence
affects multilinguality: cross-lingual representations develop patterns tied to
the *ordering* of the two layer kinds, with a pronounced alignment spike around
the first full-attention layer; alternative orderings outperform the standard one
in distillation, learning up to 2.5× faster.

**Why it matters here.** No code: we do not train. It matters as a correctness
prior — it says the layer ordering inside a hybrid stack is load-bearing, so any
place where kraken *derives* the ordering instead of reading it is a risk. That
is exactly what our loader does when `attention.recurrent_layers` is absent: it
falls back to "every Nth layer attends". This paper is the argument for treating
that fallback as a guess to be logged (and for trusting the explicit array when
the file has one).

**Status.** Applied as a logging/validation argument; the loader already prefers
the explicit array.
