Scope note: DeepSeek V2-Lite MLA as the first concrete DeepSeek diff, not V4 shared-latent.

What "DeepSeek MLA" means here
- DeepSeek-V2/V2-Lite use multi-head latent attention: a single compressed KV latent plus a decoupled rope key, with per-head value and a grouped output projection. That is the lite layout kraken already names.
- DeepSeek-V4 adds the shared-latent / compressed-attention / hash-routing / indexer variants. Those are a different graph and are out of scope for a first diff.

What kraken already has, per the tree I read
- An MLA data model: MlaLiteGeometry / MlaLiteWeights, plus a loader path that reads the lite tensors and a CPU oracle in tests/test_kraken.cpp for the lite layout.
- An arch table entry that already says the family has an MLA implementation here, with the loader accepting V2-lite MLA tensors and refusing the split/shared layouts by name at init.
- A test group shape in tests/kraken-tests-baseline.json for MLA lite plus the absorb/backend/loader groups, and a group table in tests/test_kraken.cpp that now matches the baseline shape (79 real groups, ordering verified, test_qwen4exp_whole_graph restored).

What the first concrete diff should be
- Not a new arch family yet. The shortest credible first step is a CPU-oracle-based whole-graph test group for a small V2-Lite MLA model, using the existing lite loader/oracle, anchored on logits at a few positions within f16 tolerance.
- If that gate is green, the promotion story is "MLA lite loads and generates the right row", which is the same shape as the qwen4exp whole-graph gate. That is the thing to add before any V4 work.

What is explicitly out of scope
- V4 shared-latent attention, CSA/HCA compression, the indexer/top-k cache, hash routing, and the V4 hyper-connection/Sinkhorn shape. Those are separate loader/arch/engine gaps and should not be folded into a "MLA support" claim.
- Any claim that DeepSeek-V4 runs. It is still refused by name here for the MLA/compressed-attention gap plus the type-#26 tensor class; that refusal is the honest state.

Bottom line
- DeepSeek-V2-Lite MLA is the realistic first target because the lite layout already runs. A first diff should be a CPU-oracle whole-graph test for that layout, not a V4 graph port. DeepEP/EPLB/PD disaggregation is a multi-rank serving idea; on this single-card box it is an inspiration for prefill/decode phase awareness and a residency/load-balance view, not a copyable dispatch mechanism.
