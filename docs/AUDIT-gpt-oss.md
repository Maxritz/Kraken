# gpt-oss-120b: what it took to load, and what is still wrong

`H:/OLLAMA-Models/GGUF/gpt-oss-120b-Q8_0.gguf` (59.0 GiB, 36 layers, 2880 embd,
64 query heads of 64 dims over 8 KV heads, 128 experts top-4, a 128-token window
on alternating layers, YaRN 32x over 4096). The file name says Q8_0; its routed
expert banks are **MXFP4** (ggml type 39), which this engine already dequantized.

Status at the end of this pass: **loads, streams its experts and decodes**, the
trunk arithmetic is verified against an independent reference layer by layer,
and the text is still wrong. Two defects are open and both are named below.

## The reference is cheap, and that is the finding worth keeping

llama.cpp on this box runs the same file at **7.6 tok/s decode, 4.96 tok/s
prompt** (CPU, 16 threads, page cache), loading in **1.0 s** because it mmaps and
faults in only what it touches. It answers `The capital of France is` with
` Paris.` That is one command and it turns a "garbage out" report into a
diff:

```bash
export PATH="/g/ROCM10RT-gfx1201/bin:$PATH"
H:/LLAMA-bins/rocm10/llama-completion.exe \
  -m H:/OLLAMA-Models/GGUF/gpt-oss-120b-Q8_0.gguf -no-cnv --no-jinja \
  -p "The capital of France is" -n 16 --temp 0 --no-warmup < /dev/null
```

`--verbose-prompt` prints the prompt's token ids, which is how the tokenizer was
cleared: kraken gives `[976, 9029, 328, 10128, 382]`, llama.cpp gives
`976 'The', 9029 ' capital', 328 ' of', 10128 ' France', 382 ' is'` â€” the same
five ids, so the tokenizer was never the problem.

## Defects found and fixed

Each was measured, not guessed.

1. **Windowed layers rotated at base 10000.** `rope.freq_base_swa` was read with
   a hard default of 10000 while this file declares one base (`150000`) for the
   whole stack. The default is now the model's own `rope.freq_base`, and a file
   that declares no `_swa` rope keys at all builds **no** windowed table, so
   `inv_freq_at()` returns the full-layer table -- YaRN included. (llama.cpp
   states the same thing in one line: `openai-moe.cpp` sets
   `rope_freq_base_train_swa = rope_freq_base_train`.)
2. **The window pattern was laguna's.** `is_swa()` was `layer % 4 != 0` -- full
   layer first, 3 of 4 windowed. gpt-oss alternates the other way round
   (`load_swa_pattern(ml, 2)` -> `il % 2 < 1`, windowed on the even layers). The
   pattern is now arch-declared (`ArchSpec::swa_period` / `swa_window_first`)
   and `CLAUDE`-level `ModelConfig::swa_phase` carries the half-cycle. Reading
   the pattern off the wrong end gives every layer the wrong context with no
   shape anywhere complaining.
3. **`attn_output.bias` was never loaded.** One f32 per output channel, added to
   the wo result before the residual. New `LayerWeights::wo_bias`, applied with
   `add_bias_cols` right after the projection.
4. **Per-expert FFN biases were never loaded.**
   `ffn_{gate,up,down}_exps.bias` are `[n_ff, n_expert]` / `[n_embd, n_expert]`
   tables, one row per expert. They are DEVICE-resident tables indexed by expert
   and added after each projection, *before* the activation -- which means a
   layer carrying them cannot take the fused gate/silu pair, whose epilogue has
   already applied the activation. That exclusion is explicit in the dispatch.
5. **`ffn_gate_inp.bias` was read as the wrong kind of bias.** laguna's
   `exp_probs_b.bias` is added to the router *probabilities*; llama.cpp's
   `ffn_gate_inp.bias` is folded into the router *logits*, before the gating
   function (`build_moe_ffn` folds it into `logits`). The two select different
   experts and mix them with different weights. The loader now reads the tensor
   name as the convention (`ModelConfig::router_bias_pre`) and the routing loop
   adds the bias to the logits copy that both the softmax and the top-k read.
6. **The top-k weights were renormalized when the model does not.** This is the
   big one. llama.cpp's `expert_weights_norm` defaults to **false** and its
   openai-moe loader never sets it, so gpt-oss mixes experts with the raw
   softmax probabilities: with 128 experts, the top four hold **13%** of the
   mass. kraken normalized unconditionally, which multiplies every layer's
   expert contribution by **7.6** and makes the residual stream grow ~1.2x per
   layer instead of holding. Measured at layer 0: python (raw probs) gives
   `moe.sum` rms **0.0640**, kraken gave **0.840** -- the ratio is 1/0.131, the
   top-4 mass. `ModelConfig::expert_w_norm` now takes the arch default when the
   file is silent, and gpt-oss declares 0.
7. **The expert activation was the wrong function.** gpt-oss's FFN is
   `ggml_swiglu_oai`, not `silu(gate) * up`:

   ```
   x = min(gate, 7)                      one-sided clamp
   y = clamp(up, -7, 7)                  two-sided clamp
   out = x / (1 + exp(-1.702 * x)) * (y + 1)
   ```

   New `Backend::swiglu_oai` with a HIP kernel and the scalar CPU arm, taken by
   any layer of a file whose arch declares `swiglu_oai` (the fused pair is
   excluded for the same reason the biases exclude it).

## Verification, and the reference harness that produced it

The engine's stage dump plus a numpy implementation of the same layer is what
localized each of these. `KRK_DUMP=<path> KRK_DUMP_FULL=1` writes one summary
line per stage per layer (peak, rms, non-finite count) and `KRK_DUMP=<path>`
alone writes the last row's values.

- **MoE, layer 0, last prompt row: rel-l2 0.0005** against a numpy reference that
  dequantizes the MXFP4 expert banks itself, redoes the router (F32 weights +
  logits bias), the clamps, the biases and the weighted sum. Routed experts were
  `[89, 102, 40, 53]`. This is the measurement that says the expert path,
  the MXFP4 dequant, the bias layout (`e * width + j`) and swiglu-oai are right.
- **Attention, layer 0, position 0: rel-l2 0.00019.** At position 0 the rope is
  the identity and there is one visible key, so the whole formula is
  `v * 1/(1 + exp(sink - q.k*scale))`. Reproduced from the dumped q/k/v and the
  file's sink row. That cleared the sink arithmetic: it is the denominator-only
  term the reference builds (`ggml_soft_max_add_sinks`).
- **Rope: kraken's table is ggml's.** Recovered the per-pair angle from the
  pre-rope and post-rope q dumps (`theta = atan2(x*pj - y*pi, x*pi + y*pj)`) and
  compared against `rope_yarn_corr_dims` + `rope_yarn_ramp`. kraken matches; the
  first numpy attempt did not because the ramp is indexed by the **pair** index
  (`i`), not by the channel index (`2i`) -- `rope_yarn_ramp(i0)` divides by 2
  internally, which lands on `i`.
- **Prompt tokens match the oracle exactly** (above), so the detokenizer/tokenizer
  is out of the picture.

Suite after the changes: **3050/3050**, `build_check.sh` **0 failed**, ninja
rc=0 with 0 `error:` lines.

## The two defects that remain

### 1. Windowed multi-row prefill does not write its output  (hard bug)

`attn.out` for every **even** layer (the windowed ones) of a multi-row prefill
holds **64 non-zero values out of 4096** and the *same bytes* for two different
prompts -- 4032/4096 identical between a 2-token and a 5-token run. Odd layers
(full attention, which take the bit-exact prefill) are fresh: 0-1/4096 identical
and fully populated. So the buffer is stale: the tiled `attention_kernel` either
does not run or writes elsewhere.

Discriminators already measured:

- position 0 through the **decode** kernel is exact (2e-4, above);
- laguna's windowed layers produce full output -- but laguna carries **no
  `attn_sinks`**, so `sink + windowed + multi-row prefill` is a combination this
  engine has never executed before, and gpt-oss is the only file that has it;
- `KRK_SWA_OFF=1` (new A/B arm: every layer runs as full attention, which is the
  same computation below the window length) switches those layers to the exact
  prefill and the output becomes fresh.

`KRK_SWA_OFF=1` is therefore a usable workaround for prompts shorter than 128
tokens, and the arm that proves the model math is otherwise sound.

### 2. The answer is fluent and still wrong

With `KRK_SWA_OFF=1`, `The capital of France is` decodes ` the capital of the`
(the oracle says ` Paris.`). The residual stream still grows from rms 2.0 at
layer 0 to **1323** at layer 35 -- ~1.2x per layer -- so one systematic
amplification is still in the trunk even though layer 0's MoE matches to 5e-4
and its attention to 2e-4. Open, and the next thing to chase with the same
harness: run the numpy layer-0 reference one layer deeper (it now agrees, so a
layer *k* disagreement is a defect in a specific layer's code path, most likely
a per-layer one rather than the arithmetic).

## Reproducing the whole investigation

```bash
# oracle text + token ids
# (above, with --verbose-prompt)

# kraken, both prefill arms
./build-hip/kraken.exe -m H:/OLLAMA-Models/GGUF/gpt-oss-120b-Q8_0.gguf \
  -p "The capital of France is" -n 24 --ctx 512 --greedy --expert-warmup 1
KRK_SWA_OFF=1 ./build-hip/kraken.exe ... (same line)

# stage dumps
KRK_DUMP=/tmp/d.txt KRK_DUMP_FULL=1 ./build-hip/kraken.exe ...
KRK_DUMP=/tmp/d.txt KRK_DUMP_MOE=1 ./build-hip/kraken.exe ...   # per-expert legs
```

Two smaller tools were written for this and are worth keeping as scripts: a
GGUF **tensor** lister (name, dims, type -- `tools/gguf_meta.py` prints keys
only) and a dequant/forward reference for one MoE layer + the final head.

## State after the swiglu-oai / top-k-norm pass (2026-10-20)

The residual no longer runs away from layer 0: `l_out` rms is **2.05 at L00,
2.03 at L01, 2.08 at L02, 2.16 at L03, 2.45 at L04, 2.58 at L05**, then it
compounds (L09 27, L19 84, L29 234, L35 892). Layer 0's MoE is still the
reference-matching 5e-4 above, so the trunk is right where it was checked.

**The remaining defect is now a single, measured line.** Dump the last prompt row
of `attn.out` (`KRK_DUMP=<f> KRK_DUMP_ROW=<n>`, no `KRK_DUMP_FULL`) and count the
non-zeros: every **odd** layer fills all 4096 of the row, and every **even**
(windowed) layer fills **64 of 4096** â€” one value per head, at index `h*hd`, so
each head's slice carries channel 0 and 63 zeros:

```
attn.out L00 n=4096 nonzero=64     indices 0, 64, 128, ... 4032
attn.out L01 n=4096 nonzero=4096
```

`KRK_SWA_OFF=1` (every layer as full attention) restores 4096 non-zeros on the
even layers, which places the fault in the **tiled `attention_kernel` write-out**
(`src/hip/kernels/attention.hpp`, the `nov`/`kMaxOv` loop at the end) rather than
in the window arithmetic, the rope table or the sinks: the value it does write at
`h*hd` equals the exact kernel's channel 0 to the printed digits. At a 5-token
prompt the window covers the whole prefix, so the two kernels run the same
computation and must agree â€” they do not, and only the cheap one is wrong.

That is why the text is still ` the capital of the` in *both* arms: mixed, the
even layers are 63/64 zero; with `KRK_SWA_OFF=1` the growth in the odd layers is
still fed by the same trunk, so the answer does not change.

## Where the throughput stands (same build, RX 9070 XT, ctx 512, 16 tokens)

`--expert-warmup 1`, 5-token prompt: **load 12.6 s** (2.12 GiB of trunk),
**decode 11.7 tok/s** (85.7 ms/step), prefill 2.7 tok/s on a 5-token prompt (the
cold half dominates a prompt that short). The step is transfers, not math:

| | |
|---|---|
| cache | 2698 acquires, **HOT 74.0%**, WARM 26.0%, COLD 12.4% |
| promoted | 701 experts, 8838 MiB in 1227.8 ms = **7.0 GB/s** (1751 us/expert) |
| decode split | read 406 | room 21 | alloc 0.2 | xfer 395 ms over 15 steps |
| per token | ~584 MiB promoted, ~263 MiB read from the file |

The routed set is 56.7 GiB and the device budget is 10.6 GiB, so the pool churns
by construction (the loader warns about exactly this). Two levers, both measured
elsewhere in this repo: `--expert-scan` ranking (Tiel: 1850 -> 671 promotions,
same text) and a WARM budget that holds the whole routed set, which here would
remove the 12.4% COLD reads entirely. The device line reports **no large BAR**
(`hipDeviceAttributeIsLargeBar` = 256 MiB aperture on this box), so 7 GB/s is the
WDDM staged path, not the card's copy engine.

## The MoE is not right either, and the two backends agree with each other

The layer-0 "5e-4 against a numpy reference" recorded above does not survive a
correct reference. `tools/moe_ref.py` now reads the same file with the same
routing, the same clamps and the same biases, and it disagrees with the engine:

| layer | numpy (correct read) | device | `--cpu` |
|---|---|---|---|
| L00 | 0.0222 | 0.0691 | 0.0970 |
| L06 | 0.5509 | 7.506 | 8.205 |
| L35 | 3.209 | 31.48 | 29.94 |

The routing is **exactly** right (`KRK_DUMP_ROUTE=1`, new: one line per (layer,
token) with the selected experts, their mix weights and their logits — the
engine's L00 t=0 is `[63,106,31,80]`, which is what the reference picks), so the
input vector, the router weights, the logit-side bias, the softmax and the
no-renormalise rule are all correct. What is wrong is the expert arithmetic: a
3x overshoot at L00 that grows to 14x by L06 with an unrelated direction.

The two backends **agree with each other** (7.5 against 8.2 at L06), and they
dequantise separately in different code (`src/quant.cpp` vs
`src/hip/kernels/dequant.hpp`), so the fault is in something they share: the
expert slice handed to the GEMM, or the activation/bias order in `moe_ffn`.
Reading the *wrong* expert bytes is ruled out as the whole story — a wrong bank
region still gives 0.02-0.03 rms, because every expert in a bank has the same
magnitude (measured: stride 17 vs 18 vs one block shifted all land at
0.022-0.027 at L00). So it is the arithmetic, not the addressing.

`KRK_DUMP_MOE=1` bounds the activation: `[moeact] L06 e=6 act_peak=6.6`,
`e=17 act_peak=8.2`, i.e. inside swiglu-oai's own ±56 envelope, so the
activation is not saturating either. The next probe is the per-expert `down`
leg against the reference for the same expert, one expert at a time.

## Throughput on the 24 GB RAM / 12 GB VRAM target

Measured on this box (RX 9070 XT, 16 GiB, ctx 512, 24 tokens, warm):

| arm | decode | per step | promotions | bytes/token |
|---|---|---|---|---|
| auto WARM (34.5 GiB) | **15.7 tok/s** | 63.7 ms | 705, 8888 MiB @ 7.5 GB/s | 386 MiB/step |
| `--expert-warm-mb 60000` | 15.0 tok/s | 66.6 ms | 728, 9178 MiB @ 7.0 GB/s | 398 MiB/step |

Raising the RAM tier by 60% buys **nothing**, and that is the whole finding:
the step is not bound by COLD reads (8.7% -> 6.6%, 4.2 GiB -> 3.2 GiB read), it
is bound by **promotions**. One decode step touches 4 experts x 36 layers =
144 expert reads of 12.6 MiB, and whatever the tier layout, the bytes that are
not already in VRAM must cross the bus:

    0.5-0.6 GB/token.  60 tok/s needs 35 GB/s.  Measured: 7.5 GB/s.

On a 24 GB RAM / 12 GB card the cacheable fraction of a 56.7 GiB routed set is
~35-40%, so promotions per token go *up*, not down: ~1 GB/token, i.e. 6-9 tok/s.
That is the arithmetic, not the tuning: an MXFP4 120B MoE cannot hit 60 tok/s on
that budget at any H2D rate a consumer PCIe link provides. The models that do
hit 60+ on 24/12 are the ones whose **routed set fits** (gpt-oss-20b is 12.8 GiB
total, so it lives entirely in VRAM). The two levers that move the 120B are
fewer bytes per token (residency-aware routing among near-equal experts) and a
faster bus, in that order.

## The bus does 27 GB/s and the promotion path uses 7.2

`kraken-bench --h2d` (new) times the copy shapes the expert path actually
issues, on this box:

| arm | per call | rate |
|---|---|---|
| pageable, 1 x 12.6 MB | 0.933 ms | 14.2 GB/s |
| pinned, 1 x 12.6 MB | 0.668 ms | **19.8 GB/s** |
| pinned, 3 x 4.4 MB (one expert's three slices) | 0.669 ms | **19.8 GB/s** |
| pinned, 1 x 128 MB | 4.936 ms | **27.2 GB/s** |
| pinned, 36 x 4.4 MB (a step's worth) | 5.869 ms | **27.0 GB/s** |

The decode path reports **7.0-7.5 GB/s and 1750 us per expert**, so promotions run
at a quarter of what the link does. The arithmetic closes exactly: `warm_fill`
allocates the WARM tier with `alloc_host_pageable_budgeted`, so every promotion
is a **host memcpy of 12.6 MB into the bounded pinned ring** (~1.1 ms at pageable
RAM speed) **plus** a 0.67 ms DMA. The DMA is not the cost; the copy in front of
it is.

The fix is one tier policy, not a kernel: admit the ranked hot prefix of WARM into **pinned** memory and promote it with `upload_pinned_batch`, the API's stated "source is ALREADY pinned, skip the ring" path. **Landed** as `--expert-pin-mb N` (`KRK_HOST_PIN_MB`), with a matching deferred free on `drop_host` (`release_host_pinned_deferred`), and the COLD-miss promote path switched over too, so a miss that was just read into a pinned WARM buffer is also one DMA.

Measured (gpt-oss-120b Q8_0, ctx 512, 24 tokens, greedy, `--expert-warmup 1`, this box's RX 9070 XT; interleaved order-balanced pairs, text identical in every arm -- `9f8be8ea52`):

| arm | decode | ms/step | verdict |
|---|---|---|---|
| `--expert-warm-mb 11000`, unpinned | 16.4 | 60.9 | - |
| `--expert-warm-mb 11000 --expert-pin-mb 8192` | **20.5** | **48.8** | **+25.0%**, 6/6 pairs, MAD 2.0% |
| `--expert-warm-mb 11000 --expert-pin-mb 4096` | 16.3 | 61.5 | NO DIFFERENCE |
| auto 34.5 GiB tier, `--expert-pin-mb 12288` | 20.3 -> **23.9** | 49.1 -> **41.7** | **+18.3%**, 5/5, MAD 6.1% |

Where the 7.6 -> 22.7 GB/s shows up: the unpinned arm's `transfer` line reads 6.6-7.6 GB/s at 1623-1867 us/expert, the pin-8192 arm's copy-stream line reads **22.7 GB/s**, and the decode-phase promotion time falls 298.9 -> 124.1 ms at the auto tier. On the 24 GB / 12 GB target shape that is 16.3 -> 20.5 tok/s, with the remaining gap the 283 MB/token an 11 GiB tier cannot cover.

Two things the measurement corrected:

- **A pin budget smaller than the promotion working set is worth zero.** 4 GiB halved the copy time (`xfer` 1020 -> 521 ms) and moved the step by nothing: the unpinned half is still on the critical path. 8 GiB covered all 7098 MiB the run promoted, and that is where the gain is. Size the pin to promotion BYTES, not to a fraction of RAM.
- **The `transfer` GB/s is an enqueue rate on the pinned path.** Its timer wraps `upload_pinned_batch`, which returns while the copy is still in flight, so a fully pinned arm printed `7098 MiB in 107 ms = 64.6 GB/s` on a link that does 27. The new `[stats ] pinned` line is event-timed on the copy stream instead, and it is the one to quote. It reads 6.1-24.9 GB/s depending on what else the device is doing, because the copy shares memory bandwidth with the kernels -- the step time, not this line, is the cost.

## Routing to what is resident: -20% bytes/token

The tier now prefers an expert it already holds over a near-equal one
(`--router-resident P`, `KRK_ROUTER_TIE_MARGIN`): a candidate within `P` of the
top score is taken if it is in VRAM, then if it is in WARM. The index is what
filled those tiers, so this reads the index's answer live rather than a stored
rank that evictions have invalidated. Measured on the same 11 GiB WARM / 8 GiB
pin arm, 24 tokens, greedy, interleaved order-balanced pairs, identical text:

| margin | picks moved | promotions | bytes/token | decode |
|---|---|---|---|---|
| off | 0 | 563 | 283.65 MiB | 20.3-21.4 tok/s |
| 0.02 | 2.6% | 537 | 269.47 MiB | 21.6 |
| **0.10** | **11.4%** | **462** | **226.92 MiB (-20.0%)** | **23.7-24.9 (+16.7%)** |
| 0.20 | 27.8% | 349 | 168.09 MiB (-40.7%) | 26.7 |

The byte counters are exact integers and were identical across five runs of an
arm (563 / 283.65 five times), which is why one run per arm is enough for the
byte claim and the repetitions are there for the rate. On `Qwen3.8-Distill-35B-A3B`
-- where nothing is COLD, so the win has to come from promotions between tiers --
0.10 gives 917 -> 742 promotions and 30.3 -> 32.7 tok/s, and 0.20 measures
**+14.5%** over five order-balanced pairs with identical text.

What this changes is the route, so it can change the text; the arms above printed
the same tokens, which says the near-ties are genuinely near and is not a promise
for every prompt. `scripts/kraken_router_bytes_ab.sh` is the harness: it reports
the byte counters, which the kernel A/B refuses to report because it will not
compare two arms that print different text.

## Residency, ranked: the routing really is this concentrated

`--expert-scan --expert-stub` on a 42-token prompt writes
`<model>.krakenexperts.json` (36 positions, 5184 picks, 13,219,200 bytes per
expert). Cumulative coverage of the picks, best-first:

| resident | experts | coverage | bytes/token | ms/step @ 27 GB/s | ceiling |
|---|---|---|---|---|---|
| 8 GB | 605 | 84.8% | 290 MB | 10.7 | 93 tok/s |
| 10 GB | 756 | 90.6% | 179 MB | 6.6 | 150 |
| 12 GB | 907 | **93.8%** | 117 MB | 4.3 | 230 |
| 16 GB | 1210 | 99.7% | 6 MB | 0.2 | - |

So the 12 GB the target box has covers 94% of the picks, and that is why the
grouping matters: an *unranked* fill of the same 10.3 GiB of VRAM measured 81.6%
HOT, and the ranked one 84.9-85.0%. Measured decode, interleaved and
order-balanced (2 pairs, 24 tokens, greedy):

| arm | decode | per step | HOT | COLD |
|---|---|---|---|---|
| even fill (no index) | 15.6 / 16.5 tok/s | 64.0 / 60.7 ms | 81.6% | 8.7% |
| ranked fill | **19.5 / 19.3 tok/s** | 51.4 / 51.7 ms | 84.9% | 5.0% |

**+18-19%** from the index alone, and it is the same text (greedy, same prompt).
