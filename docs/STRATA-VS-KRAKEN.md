# Strata and kraken: where the designs already agree, and where they don't

Sources for everything on the Strata side: `Maxritz/Strata-rocm` `README.md`
(fetched, quoted below). The `docs/paper/Strata-Paper.pdf` text was **not**
extracted, so nothing here claims to represent the paper's internals beyond what
its README states. Everything on the kraken side is measured on this machine
(RX 9070 XT, gfx1201, 15.92 GiB VRAM) during the session that landed the
nemotron_h_moe expert-tier fixes.

## What the README actually claims

| Claim | README wording |
|---|---|
| Target | "Run a 125-billion-parameter AI model on a normal gaming PC … one NVIDIA card (12-24 GB) + 64 GB of RAM" |
| Model | Qwen3.8-Flash-Next, sharded. "Shard 1 is the part of the model that gets loaded when it starts: its experts go into your **RAM**, the rest onto your graphics card (the second shard, a 29 GB lookup table, stays on the SSD)" |
| Speed, 12 GB card | Q2_0: 90 tok/s short chat, 67 tok/s at 128K context, 1,310 tok/s prompt |
| AMD/ROCm | "A **ROCm/HIP port** runs the same models on AMD cards (`gfx1201` RDNA4, `gfx1031` RDNA2)"; on **RX 9070 XT (16 GB)**: pinned experts 34.2 GB host RAM → 14-17 tok/s answers, 380-580 tok/s prompt; `--expert-ram-gb 22` bounded → **23.2 GB** host RAM → 11-14 tok/s, "Output is coherent and byte-identical to the pinned mode" |
| Speculation | short-chat numbers are "speculative decoding" |
| Not yet supported | `qwen35moe` (Qwen3.5-35B-A3B and friends) is listed as "new architecture — see `docs/MODEL_SUPPORT.md`" |

Two things worth noting immediately. First, Strata's ROCm target is the *same
GPU as this box*, so its AMD row is comparable hardware. Second, its bounded
mode is an LRU ring over a fixed RAM expert pool that is **byte-identical** to
the pinned mode — the same invariant kraken enforces with
`scripts/coherence_check.sh` (device arm vs `--cpu` arm, diffed).

## The four mechanisms, mapped

| Strata mechanism | kraken's counterpart | Measured here |
|---|---|---|
| GPU holds the part used for every word **plus the hottest experts** | HOT tier: VRAM expert budget, LFU+aging, pins | 2074 experts resident of 2944 in the routed set; budget 11102 MiB |
| RAM holds all experts | WARM tier: pageable host pages, read-through on miss, eviction is free (the file is canonical) | 15758 MiB capacity, 2944 held (100%), 0 rejects |
| SSD holds a large lookup table; a few rows read per word | weights are read from the mapping/file on demand (`Backend::read_host`, batched `read_host_batch`) | **0.00 MiB/token** once tiering worked; was 918.63 MiB/token when it didn't |
| CPU computes the few experts the GPU doesn't have, concurrently with the GPU | `CpuExpertPool` / `--hybrid-experts` | **−3.1×** when it engages (per-layer merge barrier; see AGENTS.md "Known open items" and docs/test-results.md §9) |
| A small in-model helper guesses the next words; several guesses verified at once | draft/MTP path (`--draft`, `spec_steps`, device top-k verification) | was a 2.7× net loss, caused by a call-site pulling the whole logits row per proposal; both call sites now use `fetch_logits(true)` / `topk_argmax` |

Kraken already runs `qwen35moe`, which Strata's README lists as an unsupported
architecture — and kraken runs it with a CPU scalar reference arm that must
match the GPU token-for-token. That is the one axis where this engine is ahead,
and it is a correctness axis, not a speed one.

## The honest comparison to the AMD numbers

They are **not** comparable, and nothing here should be read as one:

- Strata's 14-17 tok/s is a **125B** model with ~33 GiB of experts, complete and
  coherent, on the same card.
- kraken's 88-92 tok/s is a **30B-A3B** nemotron whose 23 mamba-2 blocks
  currently contribute **nothing** to the residual (the SSD scan is
  unimplemented), so its output is degraded and its speed is a *lower bound on
  cost*, not a useful throughput figure. Its routed set is 15.4 GiB against a
  10.8 GiB device budget (70%), with the remainder served from RAM.

What the fix did prove is that this engine's tier design behaves as specified
when the bookkeeping is right: 92% HOT / 8% WARM / 0% COLD per token, zero file
reads, and the device ceiling respected at 2074/2074 slots.

## What is worth taking from Strata

1. **An explicit, bounded expert-RAM knob with the trade documented.** Strata's
   `--expert-ram-gb` is what lets a 12 GB / 48 GB-RAM machine run at all, and the
   README states both arms (34.2 GB at 14-17 tok/s vs 23.2 GB at 11-14 tok/s,
   byte-identical output). Kraken's WARM budget is automatic (half of free RAM);
   an explicit cap plus a published trade table would make the same promise.
2. **An SSD tier that is a design input, not a fallback.** For a 24,576-expert
   corpus the WARM tier cannot hold the routed set, and AGENTS.md already records
   the read-path ceiling that matters then (593 MiB/s at queue depth 1 against
   3400 MiB/s sequential). Strata treats the second shard as permanently
   SSD-resident and reads rows per word.
3. **Byte-identical output across residency modes.** Strata's README asserts it;
   kraken can prove it per change with `coherence_check.sh`.
4. **Re-examine the CPU-overlap path — but measure, don't copy.** The user's
   description has the CPU computing the experts the GPU lacks *concurrently*;
   this repo measured that arm losing 3.1× because a per-layer merge barrier
   forbids the overlap the paper's ceiling assumes. Either the barrier can be
   removed (a real design change) or the claim does not transfer to this
   architecture. That question is open and is the highest-value one on this list.

## Not verified here

- The paper's own text. Only the README was read.
- Strata's expert tiering internals, its speculative scheme, and its SSD row
  layout: not inspected.
- Any kraken throughput number for a *complete* nemotron run: the mamba-2
  forward is still missing.
