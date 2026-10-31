# The MTP drafter: what the artifact is, what runs today, and what the port needs

Status: **loads and validates, does not run.** `--mtp <file>` on an HF-named
`mtp.*` head set now loads all 31 tensors, checks every shape against the trunk
and against the head's own other tensors, splits the fused expert bank into two
zero-copy views, and reports the geometry (src/mtp_head.cpp,
docs/MTP-HEAD-SCHEMA.md). The block's forward is still not implemented, so it
cannot propose a token. This file remains the port spec for that half.

## What already exists in kraken

- **The sidecar path**: `--mtp FILE` loads a nextn block from a GGUF of its own
  ([model.cpp:2651](../src/model.cpp#L2651)), `--mtp-probe` runs it for one step,
  `--mtp-spec` decodes with it as a drafter. The block shape it expects is
  `blk.<n>.attn_{q,k,v,output}.weight` + `blk.<n>.ffn_*`, i.e. a llama-style
  transformer block with the trunk's width.
- **The verify loop** (`generate_speculative`) with the greedy prefix rule and the
  sampled Leviathan rule from `src/spec_sampling.cpp`. Nothing about it needs to
  change for a new proposer: it takes drafts and answers how many were accepted.
- **The recipe prefix caches** (`KV` rollback, `truncate`) that a rejected suffix
  needs.

So the drafter slot, the verification, and the acceptance accounting are all
present. The missing piece is one block forward.

## What the artifact is

`C:\Strata-HIP-data\mtp\mtp-q2_0.gguf` — 889,014,272 bytes, GGUF v3, 31 tensors,
`general.architecture = "qwen4exp-mtp"`, produced from
`Qwen/Qwen3.8-Flash-Next` BF16 `mtp.*` tensors (checksum in
`strata.mtp.source_sha256`). Formats: 20 BF16 (projections and router), 9 F32
(norms), 2 `q2_0` (the 512 routed experts, 637.5 MiB of the 810 MiB payload).
`mtp-inventory.md` beside it has the per-tensor table; the BF16 original is
`mtp-qwen4-mtp-bf16.gguf` (5.214 GB).

`kraken-inspect` reads it today and says `31 tensors, 810.3 MiB, qwen4exp-mtp —
unknown, verdict runnable` with the line *"carries multi-token-prediction head
tensors, which this engine does not use"*. That line is now backed by the loader:
`--mtp` on the REAP-288 file prints

```
carries 31 `mtp.*` tensor(s): the same head this trunk's own block would be, in
HF names and in a shape this loader does not run (hyper-connection mixer, QSA
indexer attention, its own routed MoE) ...
```

## The graph, from the reference that already runs it

Strata's `include/strata/core/mtp.hpp` documents the block as transcribed from
vLLM 0.30.0's `qwen4_exp` MTP layer and validated by its own probe (greedy draft
acceptance 0.89 / 0.86 / 0.85 at steps 1-3):

```
e = fc_embedding(rms(embed(tok)) * (1 + w))          # 2560 -> 2560
h = fc_hidden(per-stream rms_10240(R) * (1 + w))     # 10240 -> 2560
R = h + e
  -> attention hyper-connection (4 lanes)
  -> QSA attention, its own K/V
  -> MLP hyper-connection
  -> MoE (512 experts, top-10, shared expert)
  -> R += y * 2 * sigmoid(inject / hc)
  -> final mixer  -> the TRUNK's head
```

Three details are load-bearing and each is a way to get a head that loads and
drafts garbage:

1. **The pairing.** Cell `i` pairs the trunk's final multi-stream residual at
   position `i` with the token at position `i+1`, **at rope position `i`**
   (vLLM's convention). Its output predicts the token at position `i+2`, and its
   own residual feeds the next draft step. Off-by-one here produces a plausible
   drafter with near-zero acceptance.
2. **The hidden norm is per stream, opt-in in the reference.** `mtp.pre_fc_norm_hidden`
   is 10240 wide: either one RMS over all four lanes, or (llama.cpp's qwen4exp
   MTP graph, Strata's `--mtp-hnorm`) four separate RMSes. The two differ in the
   first float of every draft; the reference carries the switch, so the port must
   too, and the acceptance rate decides which one this artifact wants.
3. **The head is the trunk's.** There is no `output.weight` in the 31 tensors and
   no `output_norm`: the block ends in its own final mixer and reuses the model's
   LM head (Strata's `bind(..., const NativeHead* head, ...)`). Its vocabulary is
   therefore the trunk's — no `d2t` map, unlike the EAGLE head.

## Why it does not need the trunk's expert set

The block carries **its own router and its own 512 experts**, so it drafts
independently of what the trunk has: the REAP-288 file routes 288 experts top-10,
the head routes 512 top-10, and that is not a mismatch — every draft is verified
by the trunk, so a head from a differently pruned checkpoint costs *acceptance*,
never correctness. What it does constrain is geometry: hidden 2560, four
hyper-connection lanes (10240), expert FF 640, shared FF 640, and the trunk's
vocabulary and head. All of those match the REAP-288 file.

## Sizes and where they can live

- BF16 block as shipped: 5.214 GB; the q2_0 repack: 0.889 GB (experts 637.5 MiB,
  projections 172.7 MiB, norms 173 KiB).
- Strata keeps **all 512 experts resident in VRAM at 708 MB** for its Q8_0
  projections, which is the design point: a drafter that pages experts would
  compete with the trunk's own tier for the same bytes and lose the reason it
  exists (a draft must be cheaper than the token it proposes).
- On this card (15.9 GiB, expert pool 4 GB at the measured shape) 0.9 GB of
  drafter is affordable; at the default auto budget it is not free and the pool
  re-clamp must see it.

## Port plan (in order, each stage gated)

1. **Loader**: parse `mtp.*` into an `MtpBlock`-shaped struct of its own, with the
   shapes checked against the trunk (hidden, lanes, experts, FF, head sharing).
   Refuse a file whose geometry disagrees, naming the tensor.
2. **Dense prefix**, host f32: `e = fc_embedding(rms(embed)*(1+w))`,
   `h = fc_hidden(rms_10240(R)*(1+w))`, `R = h + e`, in both `hnorm` modes.
   Gate: finite, stable under `R` scaling, and the two modes differ exactly as
   the reference says.
3. **The block** on the device: hyper-connection attention, dense QSA attention
   with its own KV, hyper-connection MLP, MoE through a resident 512-expert bank,
   final mixer. Gate: it reproduces the trunk's own logits for a cell whose
   prediction is already known — i.e. run the block on the prompt's last cells
   and require its argmax to match the trunk's next token at a rate well above
   chance before any speculative decode is enabled.
4. **Proposer**: `--mtp-spec` with the acceptance rate reported per step. Gate:
   acceptance ≥ ~0.6 at step 1 on greedy text, and **wall time below plain
   decode** on the same prompt — a drafter that is accepted but costs more than
   the tokens it saves is a regression with a good story.

Stage 3 is the whole project; stages 1 and 2 are a day of the same session that
wrote this file.

## Traps already known

- **A drafter that loads is not a drafter that drafts.** The only cheap oracle is
  the verify loop's acceptance rate; a graph wired with a wrong pairing or a
  wrong norm produces a head that runs, drafts, and is rejected every time.
  Budget a gate that *fails* (acceptance ≈ 0) for a deliberately wrong pairing,
  or the measurement proves nothing.
- **The trunk's head is shared, not copied.** A port that projects into its own
  vocabulary silently drafts tokens the trunk can never accept.
- **The BLAS-shaped temptation**: the block's dense projections are 2560x2560 and
  the MoE is 512x640 — on the host this is milliseconds per draft and immediately
  dominates. The block belongs on the device next to the trunk's, not beside the
  EAGLE head's host path.
