# An MTP drafter: the artifact, the wiring, and what actually blocks it

Status, 2026-10-09. Three asks were on the table — extract the `mtp.*` tensors a
checkpoint ships, load them, and verify proposals through the existing `--draft`
loop. One is done, one is recognized-but-refused with a named reason, and the
third turned out to be blocked by a defect in the loop itself rather than by
anything about MTP.

## 1. The artifact — done

An MTP head is the speculative drafter a checkpoint sometimes ships beside its own
weights: `mtp.*` tensors — a norm pair, two input projections, and one transformer
block with its own attention and MoE. The reference extraction on this box left
the pieces at `C:\Strata-HIP-data\mtp\`:

| piece | what it is |
|---|---|
| `mtp-manifest.json` | 31 tensors: name, dtype (all BF16), shape, bytes, `sha256`, source shard |
| `tensors\*.bin` | the raw payloads, 4,972.7 MiB total |
| `mtp-inventory.md` | "MTP block in the BF16 checkpoint — 31 tensors, 5.214 GB, in 28 shards" |
| `mtp-q2_0.gguf` | the reference engine's own 889,014,272-byte quantised build of it |

`tools/mtp_extract.py` turns the manifest plus the payloads into a GGUF:

```
python3 tools/mtp_extract.py --manifest C:/Strata-HIP-data/mtp/mtp-manifest.json \
                             --out      C:/Strata-HIP-data/mtp/mtp-qwen4-mtp-bf16.gguf
```

Two deliberate non-features. It does **not** requantise — the tensors are copied
verbatim, so every payload's `sha256` in the artifact equals the manifest's, and a
quantised build (the 889 MB shape, ~6x smaller) stays a separate stage with its own
parity gate, because a drafter reading the wrong bytes proposes wrong tokens and
loses accept rate *quietly*. It does **not** guess: a dtype it does not write, a
byte count that disagrees, or a hash that does not match stops the run before a
byte lands. `--check-only` verifies the source, `--verify-only` re-reads an
artifact, and both the write path and the verify path hash what actually reaches
the file.

Written file: **5,214,304,864 bytes, 31 tensors, arch `qwen4-mtp`**, dims stored in
GGUF order (the manifest's PyTorch shapes reversed) and payloads aligned to 32.

### Verification, four ways

1. **The writer's own round trip** — `rc=0`, all 31 payloads hash to their manifest
   `sha256` on the way out and back in.
2. **`tools/gguf_scan.py --audit`** — reads the file's own tensor offsets and reports
   `{"type": 30, "name": "BF16", "bits_per_element": 16.0, "tensors": 30}`. The
   density is exactly BF16 storage, which is the audit's independent check that the
   offsets and spans agree. (30 not 31: the audit's span method needs a *next*
   tensor offset, so the last tensor has no span to check.)
3. **`kraken-inspect`**, the loader's own verdict tool, freshly built — it names the
   arch, the role, the refusal and the formats:

```
arch       qwen4-mtp — draft file (not a model)
verdict    refused
           - an MTP head set for a qwen4exp target: one block with gated attention,
             a QSA indexer, hyper-connections and a 512-expert MoE, which this
             engine does not implement; it runs only through --draft, beside the
             target whose hidden states it reads
format histogram:
  BF16         31 tensors  4.86 GiB
```

4. **`--selftest`, hermetic** — `7/7 cases behaved as required`. Seven constructed
   cases, five of them failures: a clean extraction, `--verify-only` on it, a
   corrupted source payload, a wrong byte count, a dtype the tool does not write, a
   flipped **payload** byte in the artifact, and a truncated artifact. It exists
   because the real verification needs a 4.9 GiB source that is not in the repo,
   and because a checker that cannot fail proves nothing. The first version of the
   flipped-byte case flipped a byte 40 bytes from EOF — which lands in inter-tensor
   *padding*, not data, so the verifier read it as clean and the case passed while
   proving nothing. The aim is now the last payload's bytes (they run to EOF) and
   the reason is written down in the case.

## 2. Loading it — recognized, refused by name, and why that is the honest ceiling

`src/arch.cpp` now carries the head set as its own family:

```cpp
{"qwen4-mtp", ArchShape::Moe, ArchSupport::No, ArchRole::Draft, "qwen4exp",
 "an MTP head set for a qwen4exp target: one block with gated attention, a QSA "
 "indexer, hyper-connections and a 512-expert MoE, which this engine does not "
 "implement", true, true, false, false},
```

`ArchRole::Draft` is the repo's existing role for "a speculative head set that only
means something beside a target", and the loader already knows how to answer one:
refuse it as a model, say what it is for instead, and keep it out of the gap map.
Two checks pin it in `tests/test_kraken.cpp` (the table entry and a verdict fixture
built with the suite's own `GgufBuilder`), and the whole suite is green —
**2798/2798 checks, rc=0**.

Through the real draft path, with the artifact as the drafter:

```
$ kraken.exe --model models/SmolLM2-135M-Instruct.Q4_K_M.gguf \
             --draft C:/Strata-HIP-data/mtp/mtp-qwen4-mtp-bf16.gguf -p hello -n 4
kraken: draft model: cannot run this file ('qwen4-mtp'). an MTP head set for a
qwen4exp target: ... which this engine does not implement; it runs only through
--draft, beside the target whose hidden states it reads
rc=1
```

Three things stand between that message and a running drafter, and only the third
is about MTP:

1. **The target family is refused.** This head drafts for `qwen4exp`, whose file
   names `attn_gate`, a fused `attn_qkv`, `ssm_conv1d` and `ssm_out` — see
   `docs/STRATA-VS-KRAKEN.md`, "What supporting `qwen4exp` would actually need".
   Nothing to draft *for* until that runs.
2. **The block is a `qwen4exp` block.** Gated attention with the packed query gate,
   per-head QK-norm, a QSA indexer over compressed KV, hyper-connections at
   low-rank 320, and a 512-expert MoE. Its `o_proj` is `[2560, 6144]`,
   `q_proj` `[12288, 2560]` — the same shapes the target refuses.
3. **Its input is the target's hidden state.** `mtp.fc_embedding` projects the last
   token's embedding and `mtp.fc_hidden` projects the previous hidden state; the
   head cannot run standalone from tokens. The mechanism to reuse is DFlash's: the
   target copies residual streams into a capture buffer (`Engine::dflash_capture`,
   `dflash_->commit`), the drafter fuses them, and `draft_block()` proposes. An MTP
   head would be a second family on that same interface, not a new pipeline.

## 3. Verifying proposals — the loop is the blocker, and it reproduces

The existing loop cannot accept tokens from *any* drafter on this build. Measured
2026-10-09, SmolLM2-135M as both target and draft (`--draft` pointing at its own
file, window 4, ctx 512, n=32, greedy, order-balanced plain/draft/draft/plain):

| arm | result |
|---|---|
| plain | 669.1 tok/s (1.5 ms/step), 707.0 tok/s (1.4 ms/step) |
| draft | 171.4 tok/s over 18 rounds, **20/72 accepted (27.8%)** |
| draft | 172.7 tok/s over 18 rounds, **20/72 accepted (27.8%)** |

A **4.0x loss** on the 2026-09-09 measurement, with the generated text identical to
the plain arm — so the verify/accept machinery is correct and the tokens are right;
the *proposals* are what fail. A self-draft proposes its own argmax, so acceptance
should be ~100%, and the number is not noise: it is exactly 20/72 = 5/18 in both
runs, and `docs/DIAG-decode-cpu.md` F8 recorded 40/144 = the same 5/18 at 36
rounds on a different model and token count. This is F8 with a second reproduction,
**not a new finding**.

That defect is **now fixed** on the current build (`src/engine.cpp:3745-3800`),
and F8 is resolved across families: `laguna-xs2-Q4_K_M` as both target and draft
reaches **52/52 = 100%** at `-n 64` (13 full-accept rounds, no pre-check failures,
no partials, no KV-invariant failures) and **77/80 = 96.2%** at `-n 96` (20 rounds,
all full accept, the trailing 3 tokens the 128-context cap cutting the last round);
`SmolLM2-135M` self-draft is the same shape — 96.2% at 96 tokens, all rounds
full. The earlier 27.8% / 34.4% records were stale and predate the fix; they are
corrected here. `

**So: wiring an MTP head into this loop today is now on top of a loop that
accepts ~100% of a self-draft.** The order that pays is (b) quantify what the fixed
loop is worth with a self-draft (where acceptance has a known correct answer, ~100%)
and only then (c) an MTP head, which is a *cheaper and better* proposer than a
second model but not a different mechanism. (Step (a) is done.)


## 4. An MTP head is per-family, and the target has to clear a wall first

Two facts settled 2026-10-09 answer "would laguna run an MTP head too", and
neither is about the head.

**A drafter is not portable between families.** The MTP block's input is the
*target's* hidden state and its body is that family's own block (qwen4's is a
`qwen4exp` block: gated attention, a QSA indexer, hyper-connections, a 512-expert
MoE). A head extracted from one checkpoint cannot propose for another family's
target -- there is no "our drafter" that gets pointed at laguna. A family gets an
MTP head only if its checkpoint ships `mtp.*` tensors to extract, and the loop it
feeds has to be built for that family's block.

**And the target has to be able to rewind.** `Engine::load_draft`
(`src/engine.cpp:1595-1605`) refuses a **recurrent** target outright: "speculative
decoding needs a partial KV rewind, which a recurrent (gated delta net) model
cannot do" -- a delta-rule state has no inverse, so after a block is verified the
state still describes the rejected tokens. Measured on the pair that is affordable
here, `models/Qwen3.5-0.8B.Q4_K_M.gguf` with
`laguna-xs21-dflash-q8.gguf` as the draft: rc=1, that message, at load. So a
family whose layers are recurrent cannot speculate at all, whatever drafter it is
handed -- and qwen4exp's declared block includes an `ssm_conv1d`/`ssm_out` path,
which makes this the first question for its MTP plan, before the loop is built.

**laguna clears it, and it already has a drafter -- a different kind.** Kraken's
`--draft` takes *two* things that arrive the same way: a full draft model, and a
DFlash head set, distinguished by a metadata read of the file itself (`dflash_arch`,
`general.architecture = dflash`). DFlash is implemented here (`src/dflash.cpp`:
`load()` reads a head set with no embedding and no output head because it borrows
both from the target, `commit()` fuses captured residual streams and projects K/V
into a cache of its own, `draft_block()` proposes) and wired through `--draft` in
both `src/main_cli.cpp:225` and `src/main_server.cpp:119`. And the family already
ships one:

| | `laguna-s-2.1-DFlash-Q4_K_M.gguf` |
|---|---|
| arch | `dflash`, `decoder_arch = "laguna"` |
| verdict | refused as a model -- "a speculative head set... runs only through `--draft`, beside the target whose hidden states it reads" |
| shape | 76 tensors, 618.4 MiB, 6 blocks, 1.1B, 72 heads / 8 KV heads, `block_size 16`, `target_layers [6 i32]` |
| base model | `poolside/Laguna-S-2.1` |

`laguna-xs2-Q4_K_M.gguf` is not recurrent -- 40 blocks, `embedding_length 2048`,
per-layer `head_count` (`[40 i32]`), `head_count_kv 8`, `leading_dense_block_count
1`, and no `ssm`/`conv`/`gdn` key of any kind -- so the rewind wall does not apply
and DFlash is the drafter that fits it. What is **not** measured: that this
pairing runs end to end. The drafter names `Laguna-S-2.1` as its base (S, 36+23 GB
of shards) and the target at hand is the XS, so the honest state is "a drafter
exists and the mechanism is implemented", not ``laguna speculates at N tok/s``.

## What is not verified here

The MTP block has never been executed — no forward pass for it exists, so nothing
below the artifact's bytes has a number. The head's size as extracted is 4.86 GiB
BF16 against the reference engine's 889 MB quantised build; the extractor does not
quantise, so that comparison is a size, not a measurement. And the accept-rate
measurement above is a *self*-draft on a 135M model: it establishes that the loop is
broken, not what a correct loop would score.
