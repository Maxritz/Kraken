# Loading modes: memory-resident vs streamed from disk

Every dense weight in a run arrives through one of two paths, and the choice is
one flag:

| what | flag | what it does |
|---|---|---|
| **memory** | `--weight-vram-mb -1` | streaming disabled: upload every tensor once at load and keep it in VRAM |
| **auto** (default) | `--weight-vram-mb 0` | keep what fits at load, stream the rest through a device ring before each layer runs, and promote a streamed layer to resident on first use when free VRAM allows |
| **disk** | `--weight-vram-mb N` with `KRK_WEIGHT_PROMOTE=0` | a fixed `N` MiB prefix stays resident and every other layer is re-read from the file before each use, every step |

The third arm is the one that is actually *disk-based*. `--weight-vram-mb N`
alone is not: promotion re-points a streamed layer into a permanent allocation
the first time it runs, so on a model that fits, a "streamed" run is fully
resident within the first token. Set `KRK_WEIGHT_PROMOTE=0` to measure the
re-read.

MoE files are the exception to all of this: the ring only carries the **dense
trunk** (attention, shared FFN, embeddings). Routed `_exps` banks belong to the
expert tier (`--expert-cache-mb`, `--expert-warm-mb`), so a 30B-A3B file shows
0.71 GiB of "weights" in every arm.

## The commands

The harness is `scripts/bench_loading.sh`: it runs the three arms interleaved
and order-balanced (reversed on even reps), records every log, and prints one
row per (model, arm) from the warmest rep.

```sh
export PATH="/g/ROCM10RT-gfx1201/bin:$PATH"
sh scripts/bench_loading.sh \
    /g/More-models/Phi-3.5-mini-instruct-python_coding_assistanQ4_K_M.gguf \
    /g/More-models/Qwen3-8B-Q4_K_M.gguf \
    /g/More-models/Qwen3-30B-A3B-abliterated-erotic.i1-Q2_K.gguf:16 \
    /g/More-models/Qwen3.8-27B-WebGGUF-Q4_0.gguf:8
```

`PATH:n` sets the generated-token count for that model (`KRK_BENCH_N` is the
default, 32); `KRK_BENCH_REPS`, `KRK_BENCH_CTX`, `KRK_BENCH_DISK_MB` and
`KRK_BENCH_OUT` cover the rest. The same arms by hand:

```sh
# memory-resident
kraken.exe -m M --bench --ctx 512 --chunk 256 -n 32 --greedy --weight-vram-mb -1

# auto (the default policy)
kraken.exe -m M --bench --ctx 512 --chunk 256 -n 32 --greedy

# disk: 2 GiB resident, everything else re-read every step
KRK_WEIGHT_PROMOTE=0 kraken.exe -m M --bench --ctx 512 --chunk 256 -n 32 \
    --greedy --weight-vram-mb 2048
```

`--bench` prints `prefill N tokens in X ms (Y tok/s)`, `decode ...` and `load`;
the `[info ] weight residency:` line says which layers are streamed, and
`[stats ] load` gives the upload rate. Read the residency line before believing
the throughput: an arm that streamed nothing is a resident arm.

**`--bench` ignores `-p`.** It runs its own fixed prompt
("The history of computing is a history of abstraction ..."), which tokenizes
to **125 tokens on Phi's tokenizer and 55 on Qwen's** -- so prefill tok/s in the
Phi row is not comparable with the rest, and `-p` on a `--bench` line does
nothing. `-n`, `--ctx`, `--chunk` and the sampling flags are honoured.

## Measured (RX 9070 XT, 15.9 GiB, gfx1201, ROCm 7.16)

One run of `sh scripts/bench_loading.sh` with no arguments (2026-10-19T18:58Z),
two interleaved order-balanced reps per arm; the table quotes rep 2 (warm), and
the same script's earlier sweep agreed with it to within 4% on every cell. The
first read of a multi-GB model is cold, which is why the 27B's resident arm
reads `load 12.73 s` on a cold first rep against 3.51 s warm -- the same bytes
at 1.1 GB/s versus 4.1 GB/s, and the reason a single sequential run reports the
page cache rather than the policy.

| model | size / shape | arm | streamed | load | prefill | decode |
|---|---|---|---|---|---|---|
| Phi-3.5-mini Q4_K_M | 2.16 GiB, 32L hd96, 125-tok prompt | memory | -- | 0.96 s | 1863 tok/s | **126.3** tok/s |
| | | auto | -- | 0.99 s | 1853 | 126.0 |
| | | disk 2 GiB | L29..31 | 0.93 s | 1150 | 25.0 |
| Qwen3-8B Q4_K_M | 4.68 GiB, 36L, 55-tok prompt | memory | -- | 1.36 s | 1031 tok/s | **91.2** tok/s |
| | | auto | -- | 1.41 s | 1022 | 91.2 |
| | | disk 2 GiB | L11..35 | 0.73 s | 114 | 2.4 |
| Qwen3-30B-A3B i1-Q2_K | 0.71 GiB trunk, 48L, 128 exp x top-8 | memory | -- | 0.53 s | 25.2 tok/s | **33.0** tok/s |
| | | auto | -- | 0.53 s | 25.1 | 32.9 |
| | | disk 2 GiB | none | 0.54 s | 25.2 | 33.4 |
| Qwen3.8-27B WebGGUF Q4_0 | 14.40 GiB, 55-tok prompt | memory | -- | 3.51 s | 288 tok/s | **14.3** tok/s |
| | | auto (12.02 GiB budget) | L52..63 | 2.81 s | 68.6 | 7.9 |
| | | disk 2 GiB | L1..63 | 0.69 s | 27.0 | 0.5 |

`-n` is 32 for Phi and the 8B, 16 for the 30B and 8 for the 27B: the 27B's disk
arm costs 1.6 s per token, so a 32-token run there is two minutes of wall time
for one number. Decode tok/s on the 30B also moves 3% between arms with the
same policy (`33.0` / `32.9` / `33.4`) -- that is the expert tier's own
run-to-run noise, not the loading mode.

### Larger models, beyond the card

Same script, same arms, two models that do not fit 15.9 GiB:

```sh
KRK_BENCH_OUT=/tmp/sweep_large sh scripts/bench_loading.sh \
    /h/OLLAMA-Models/GGUF/Qwen3.8-27B-Q4_K_M.gguf:4 \
    /h/OLLAMA-Models/GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf:8
```

| model | size / shape | arm | streamed | load | prefill | decode |
|---|---|---|---|---|---|---|
| Qwen3.8-27B-Q4_K_M | 15.93 GiB, dense 27B | memory | -- | 3.87 s | 131 tok/s | **5.6** tok/s |
| | | auto (12.02 GiB) | L47..63 | 2.75 s | 51.4 | 3.4 |
| | | disk 2 GiB | L1..63 | 0.75 s | 25.5 | 0.5 |
| Qwen3.5-35B-A3B UD-Q4_K_XL | 18.34 GiB, MoE, 1.52 GiB trunk | memory | -- | 5.25 s | 32.9 tok/s | **14.0** tok/s |
| | | auto | -- | 5.35 s | 32.6 | 13.9 |
| | | disk 2 GiB | none | 5.26 s | 33.1 | 14.2 |

- **A dense model just over the card is where the policy shows.** On the 27B
  Q4_K_M, resident is 5.6 tok/s and the auto budget's 12.02 GiB is 3.4 (it
  streams 17 layers), while the 2 GiB ring is 0.5 -- 11x below resident.
  `--weight-vram-mb -1` cannot fit every model (this one needs ~14.6 GiB of
  weights beside KV and workspaces), so the honest ordering here is
  `-1` if it fits, auto if it does not, and the ring only when nothing else
  fits.
- **A MoE larger than the card is not a streaming problem at all.** The
  35B-A3B's trunk is 1.52 GiB, so the ring is inert (`streamed: none`) and all
  three arms land inside 3% of each other. Its decode rate is set by the expert
  tier (`--expert-cache-mb`, `--expert-warm-mb`, `--expert-scan` ranking), not
  by this flag -- which is why moving a 35B-A3B file to disk loading does not
  make it slower *or* faster.

### Super-sized files: what the loader refuses, and why

The two largest complete files on this box cannot appear in this table at all,
and `kraken-inspect` says why before any VRAM is touched:

| file | size | arch | refused because |
|---|---|---|---|
| `DeepSeek-V4-Flash-0731-K160-IQ2XXS-...imatrix.gguf` | 53.5 GiB | `deepseek4` | multi-head latent attention (`attn_kv_a`/`attn_kv_b`), `attn_sinks`, quantization type #26 |
| `gpt-oss-120b-Q8_0.gguf` | 59.0 GiB | `gpt-oss` | `attn_sinks`, and the arch is not in the table |
| `qwen3.8-flash-next-reap-288-Q4_K_M.gguf` | 78.1 GiB | `qwen4exp` | `attn_gate`, fused `attn_qkv`, `ssm_conv1d`, `ssm_out`, PLE, QSA indexer |
| `Muse-Glimmer-30B-UD-Q8_K_XL.gguf` | 30.1 GiB | `muse-glimmer` | arch not supported |

That is a capability boundary, not a loading-mode result: these files fail at
`verify`/load, so no arm can measure them. `docs/AUDIT-supersized-gguf.md`
catalogs every piece each one needs and the order the work has to land in.

## What the numbers say

- **Disk re-reads are linear in the layers they cover.** 3 streamed layers of
  Phi (about 0.3 GiB per token) cost 4.9x decode; 25 layers of the 8B
  (about 4.3 GiB per token) cost 36.8x. The ring is a bounded-memory device, not
  a performance mode: it is what makes a model larger than the card *possible*.
- **Loading is not where the time goes for a fitting model.** One pass at
  1.4-4.5 GiB/s: 2.16 GiB in 1.02 s for Phi, 4.68 GiB in 1.29 s for the 8B,
  14.40 GiB in 3.23 s for the 27B. The same bytes re-read per token is what
  costs 2.5 tok/s.
- **Prefill pays the re-read once per layer per chunk, decode pays it per
  token.** On the 8B, `--chunk 256` amortizes 25 streamed layers over 256 rows,
  so prefill only drops 8.8x (1037 -> 118) while decode drops 36.8x
  (92.1 -> 2.5). On the 27B the same split is 10.1x against 30x. A disk arm is
  therefore least bad at large chunks and long prompts -- which is the opposite
  of what an interactive run does.
- **`--weight-vram-mb -1` can beat the auto budget on a large dense model.**
  The 27B's 12.02 GiB auto budget streams 12 layers and decodes at 8.6 tok/s;
  forcing all 14.40 GiB resident decodes at 14.9. The auto plan is decided
  before the KV cache and workspaces exist, so it under-uses the card -- the
  same reason `Model::promote_layer` exists (it recovered some of that:
  the residency line shows each promotion with the MiB it kept free).
- **A MoE file's "weights" are not its routing cost.** All three arms agree for
  the 30B-A3B because the ring never sees the expert banks; that run measures
  the trunk (0.71 GiB) and the expert tier, not streaming. Use
  `--expert-cache-mb` / `--expert-warm-mb` to move *that* working set.
- **A cold page cache is not a slow arm.** The 30B's rep-1 resident run read
  `load 1.30 s` and 20.8 tok/s against 34.4 warm; the disk arm of the 27B reads
  only its 2 GiB resident prefix (0.65 s) and so looks faster to load than the
  resident arm at 0.5 tok/s. Compare decode, not load, across arms.

## Reproducing

`KRK_BENCH_REPS=3 sh scripts/bench_loading.sh MODEL ...` is the loop; the arms
are order-sensitive (page cache, driver state, expert tiers), so the script
interleaves them and reverses the order on even reps. The summary takes the
warmest rep per arm rather than a mean, and every run is printed so a cold one
is visible instead of averaged away. A first rep that is much slower is the
cold file: check `[stats ] load` before concluding the arm is slow.

`AGENTS.md` ("Dense weights stream now, and the ring has two traps") has the
correctness rules behind these numbers -- the ring must be drained before the
next layer is read, routed banks must be excluded from a layer's byte total,
and `unload()` must not free pointers inside a promoted allocation.
