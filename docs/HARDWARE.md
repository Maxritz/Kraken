# Hardware notes

Fleet priority: the **CPU backend is the oracle and always validated first**;
then the **RX 6700 XT (`gfx1031`, RDNA2)**, the primary GPU target on which
correctness and bandwidth-bound decode are judged; then the **RX 9070 XT
(`gfx1201`, RDNA4)**, the secondary target that inherits every validated
decision before its own paths are exercised.

What changes between the two cards this engine targets, and where the code
notices.

| | RX 6700 XT | RX 9070 XT |
|---|---|---|
| target triple | `gfx1031` | `gfx1201` |
| generation | RDNA2 | RDNA4 |
| wavefront | 32 (RDNA2 runs wave32 in practice) | 32 (mandatory) |
| matrix cores | **none** | dense WMMA, `f16`/`bf16`/`i8`/`fp8` |
| fp16 rate | 2x fp32 (packed math) | 2x fp32 + WMMA |
| `v_dot2_f32_f16` | yes | **no** (superseded by WMMA) |
| LDS per workgroup | 64 KiB | 64 KiB |
| VRAM | 12 GB | 16 GB |

The two cards therefore want different primitives, not the same primitive at
different sizes. The build produces one code path per target and the loader
picks it; there is no runtime branch in the inner loops.

---

## RDNA2 (`gfx1031`) — there is no WMMA, so stop pretending there is

The temptation is to emulate WMMA with `__hfma2`. That is a mistake: `half2`
accumulation over a 4096-wide K dimension loses so much precision that greedy
decoding visibly degrades. The correct primitive is `v_dot2_f32_f16`, which
computes `a0*b0 + a1*b1 + c` with **f32 accumulation** at the packed fp16 rate.
That is exactly the shape a GEMV wants.

`src/hip/kernels/gemm.hpp` wraps it in `dev_dot2`, guarded by `KRK_HAS_DOT2`
(defined for RDNA2 and RDNA3 only, because RDNA4 dropped the instruction). The
SIMT GEMM's inner loop is one `dev_dot2` per two K values per output pair.

Decode on this card is bandwidth-bound: ~12 GB of weights read per token, so the
ceiling is roughly `VRAM bandwidth / model size`. The GEMV kernel is built for
precisely that — one block per output channel, weights streamed once, dequantized
straight into registers, one reduction at the end. It never writes a dequantized
row anywhere.

## RDNA3 (`gfx11xx`) — WMMA with matrix replication

If you port from an RDNA4 example you will get wrong numbers, because RDNA3
mirrors fragments across the two lane groups instead of splitting K. See
`docs/RESEARCH.md` §1.

## RDNA4 (`gfx1200`/`gfx1201`) — WMMA with the K-split layout

* The builtin **must** carry the `_gfx12` suffix. The unsuffixed name does not
  exist on this target and the compile fails.
* Fragments are 8 fp16 per lane, not 16.
* The two lane groups cover *different* K positions, interleaved in groups of
  four.
* fp8/bf8 inputs are available and are the natural next step for the prefill
  path: `wmma_f32_16x16x16_fp8_fp8_w32_gfx12` doubles the arithmetic per
  instruction, and RDNA4 adds a 16x16x32 shape. Nothing in the tiling
  prevents it — BK is already 64 and chunk-aligned.

---

## Probing the card

`kraken --info` prints what the engine believes:

```
KRAKEN 0.1.0 | AMD Radeon RX 9070 XT (gfx1201, RDNA4, 64 CU, wave32, WMMA gfx12)
```

If the `gfx` string does not match the `--offload-arch` you built for, the
kernel selection is wrong — rebuild. `DeviceCaps::family` is derived from the
device string and drives the *logging*; the actual kernel choice is made at
compile time by the `__gfx*__` macros of each offload pass, which is why a
mismatch is a build problem, not a runtime one.

---

## Known limits worth knowing before benchmarking

* **Decode on a 32-head model launches 32 attention blocks**, one per head. On a
  40- or 64-CU card that is poor occupancy. A split-K attention (several blocks
  cooperating per head, combining partial online-softmaxes) is the standard fix
  and is not implemented in v0.1.
* **Sampling happens on the host.** Logits for a 150k vocab are 300 KiB of fp16
  per token across PCIe. A device-side top-k/top-p sampler removes that; the
  kernel structure is trivial but it is not written yet.
* **Prefill block tile is 32x64.** For a batch of 32 tokens that is one M-tile;
  a larger BM would help long prompts on the 9070 XT's wider array.
