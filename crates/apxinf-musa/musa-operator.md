# MUSA Operator Catalog

This catalog is for model authors deciding what PI0.5's hot path needs on MUSA
and what is still missing. It is the MUSA counterpart of
[`crates/apxinf-cuda-new/cuda-operator.md`](../apxinf-cuda-new/cuda-operator.md),
and it follows the rule stated there: a model that needs a semantic the L3 layer
does not expose **must record an operator gap rather than infer support from the
legacy implementation**.

**Every entry below is a gap.** No MUSA implementation of any of these semantics
exists, so the registry is empty and every call resolves to `deferred`. A deferred
semantic produces no value: the numerical truth for the stages it feeds comes from
`python/apxinf_ref`, the device-independent eager reference runtime, and a
stage-probe comparison against the MUSA engine does not cover those stages. Why
nothing here falls back to a host implementation is argued once, in the
[crate README](README.md#why-there-is-no-fallback-implementation-here).

Each `l3-operator` comment is checked against the Rust catalog by
`src/tests/mod.rs`: every semantic appears exactly once, and every entry carries
its fallback and its exit criterion. The test checks presence, not truth --
reviewers still have to check that the contracts and exit criteria are accurate.

## Reading an entry

| Column | Meaning |
|---|---|
| Semantic | The Rust `Semantic` variant and its `l3-operator` marker (the `Semantic::name()` string) |
| Math | The mathematical meaning, written out rather than named |
| Contract | Shape, dtype and layout of the operands |
| Probe stage | The stage of `apxinf.pi05.stage-probe.v1` whose value it feeds |
| Frequency | How often it runs |
| Fallback | What runs when no MUSA candidate exists |
| Exit criterion | The safe API that retires the gap, and the replay that proves it |

## Available MUSA operators

_None._ The registry is empty; every semantic below resolves to `deferred`.

## Gap table

<!-- l3-operator:layer_norm -->
### `layer_norm`

| Field | Value |
|---|---|
| Math | `(x - mean(x)) / sqrt(var(x) + eps) * weight + bias` |
| Contract | [tokens, 1152] bf16 in and out; weight and bias [1152] bf16; eps 1e-6 |
| Probe stage | `vision_layer_{i}` |
| Frequency | twice per vision layer |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | norm::layer_* for bf16 at 1152 wide; retired when vision_layer_{i} reports a MUSA number within the bf16 threshold of the reference. |

<!-- l3-operator:rms_norm -->
### `rms_norm`

| Field | Value |
|---|---|
| Math | `x * rsqrt(mean(x**2) + eps) * (1 + weight)` |
| Contract | [tokens, 2048] bf16 in and out; weight [2048] f32; eps 1e-6 |
| Probe stage | `prefix_v_layer{0,depth-1}` |
| Frequency | twice per language layer |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | norm::rms_* at 2048 wide; retired when the prefix K/V stages agree with the reference. |

<!-- l3-operator:adaptive_rms_norm -->
### `adaptive_rms_norm`

| Field | Value |
|---|---|
| Math | `mod = dense(cond); scale, shift, gate = chunk(mod, 3); (x * rsqrt(mean(x**2)+eps)) * (1+scale) + shift, and gate` |
| Contract | [tokens, 1024] bf16; cond [1, 1024] f32; dense weight [3072, 1024] f32; two outputs |
| Probe stage | `denoise_step_{s}` |
| Frequency | twice per action layer per step |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | norm::adaptive_rms_* with the f32 conditioning contract; retired when the denoise stages agree with the reference. Note the conditioning is f32 upstream and bf16 in the engine today, so the two disagree by construction until that is reconciled. |

<!-- l3-operator:gemm -->
### `gemm`

| Field | Value |
|---|---|
| Math | `y = x @ W + b` |
| Contract | [tokens, K] bf16 x [K, N] bf16, optionally plus bias [N] |
| Probe stage | `vision_layer_{i}, prefix_v_layer*, denoise_step_{s}` |
| Frequency | four to seven times per layer |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | ops::gemm and ops::gemm_bias already publish this semantic on CUDA; the MUSA question is the provider, not the contract. |

<!-- l3-operator:gemm_geglu -->
### `gemm_geglu`

| Field | Value |
|---|---|
| Math | `gelu(x @ W_gate) * (x @ W_up), with gate and up packed into one weight` |
| Contract | [tokens, K] bf16 x [K, 2N] bf16; gate columns first |
| Probe stage | `prefix_v_layer*, denoise_step_{s}` |
| Frequency | once per language and action layer |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | ops::gemm_geglu already publishes this semantic on CUDA; packing stays internal. |

<!-- l3-operator:split_qkv_bias -->
### `split_qkv_bias`

| Field | Value |
|---|---|
| Math | `split a [tokens, 3 * heads * head_dim] projection into q, k, v and add each bias` |
| Contract | [tokens, 3456] bf16 -> three [tokens, 1152] bf16; 16 heads, head_dim 72 |
| Probe stage | `vision_layer_{i}` |
| Frequency | once per vision layer |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | attention::split_qkv_bias_*; retired when vision_layer_{i} is reachable on MUSA. |

<!-- l3-operator:split_qkv_apply_rope -->
### `split_qkv_apply_rope`

| Field | Value |
|---|---|
| Math | `split as above, then rotate q and k by position` |
| Contract | [tokens, 2304] bf16 -> q [tokens, 2048], k/v [tokens, 256]; 8 heads, 1 kv head, head_dim 256 |
| Probe stage | `prefix_v_layer{0,depth-1}` |
| Frequency | once per language layer |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | rope::split_qkv_apply_*; retired when the prefix K/V stages agree with the reference. |

<!-- l3-operator:apply_query_write_kv -->
### `apply_query_write_kv`

| Field | Value |
|---|---|
| Math | `rotate q by position and write the rotated k and v into the prefix cache at the offset` |
| Contract | q/k/v [tokens, 256] bf16; cache [prefix + horizon, 256] bf16 |
| Probe stage | `denoise_step_{s}` |
| Frequency | once per action layer per step |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | rope::apply_q_write_kv_*; the cache rows written must exclude the reserved tail that the probe currently signs. |

<!-- l3-operator:mha -->
### `mha`

| Field | Value |
|---|---|
| Math | `softmax(q @ k^T / sqrt(d)) @ v, bidirectional, within each window` |
| Contract | q [windows, heads, patches_per_view, 72]; 16 heads |
| Probe stage | `vision_layer_{i}` |
| Frequency | once per vision layer |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | attention::mha_*; retired when vision_layer_{i} is reachable on MUSA. |

<!-- l3-operator:mqa -->
### `mqa`

| Field | Value |
|---|---|
| Math | `softmax(q @ k^T / sqrt(d)) @ v with one shared kv head, causal, over prefix plus suffix` |
| Contract | q [tokens, 8, 256]; k/v [tokens, 1, 256] |
| Probe stage | `prefix_v_layer{0,depth-1}, denoise_step_{s}` |
| Frequency | once per language and action layer |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | attention::mqa_*; retired when the K/V and denoise stages agree with the reference. |

<!-- l3-operator:bias_residual -->
### `bias_residual`

| Field | Value |
|---|---|
| Math | `y = x @ W + b + residual` |
| Contract | [tokens, K] bf16; bias [N]; residual [tokens, N] |
| Probe stage | `vision_layer_{i}, prefix_v_layer{0,depth-1}` |
| Frequency | twice per vision layer, once per language layer |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | fused::bias_residual_*; retired when the enclosing layer is reachable on MUSA. |

<!-- l3-operator:bias_residual_rms_norm -->
### `bias_residual_rms_norm`

| Field | Value |
|---|---|
| Math | `h = x @ W + b + residual; y = h * rsqrt(mean(h**2) + eps) * (1 + weight); two outputs` |
| Contract | [tokens, 2048] bf16; the normalised output feeds the next layer's QKV |
| Probe stage | `prefix_v_layer{0,depth-1}` |
| Frequency | once per language layer |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | fused::bias_residual_rms_*; the second output must be kept, it is the next layer's input. |

<!-- l3-operator:bias_residual_layer_norm -->
### `bias_residual_layer_norm`

| Field | Value |
|---|---|
| Math | `h = x @ W + b + residual; y = LayerNorm(h); two outputs` |
| Contract | [tokens, 1152] bf16 |
| Probe stage | `vision_layer_{i}` |
| Frequency | once per vision layer |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | fused::bias_residual_layer_*; retired when vision_layer_{i} is reachable on MUSA. |

<!-- l3-operator:adaptive_gate_residual_rms_norm -->
### `adaptive_gate_residual_rms_norm`

| Field | Value |
|---|---|
| Math | `h = x @ W + residual * gate; y = (h * rsqrt(mean(h**2)+eps)) * (1 + next_scale) + next_shift; two outputs` |
| Contract | [tokens, 1024] bf16; gate, scale and shift from the f32 conditioning |
| Probe stage | `denoise_step_{s}` |
| Frequency | twice per action layer per step |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | fused::adaptive_gate_residual_rms_*; the second output is the next layer's normalised input and must not be recomputed differently. |

<!-- l3-operator:bias_gelu -->
### `bias_gelu`

| Field | Value |
|---|---|
| Math | `gelu(x @ W + b) with the tanh approximation` |
| Contract | [tokens, 1152] -> [tokens, 4304] bf16 |
| Probe stage | `vision_layer_{i}` |
| Frequency | once per vision layer |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | activation::bias_gelu_*; the tanh approximation is part of the contract, not an implementation detail. |

<!-- l3-operator:add_position -->
### `add_position`

| Field | Value |
|---|---|
| Math | `y = x @ W + b + position_table[window]` |
| Contract | [views * 256, 1152] bf16; position table [256, 1152] |
| Probe stage | `vision_patch_embed` |
| Frequency | once per request |
| Importance | Boundary |
| Fallback | `ReferenceComparison` |
| Exit criterion | embedding::add_position_*; retired when vision_patch_embed is reachable on MUSA. |

<!-- l3-operator:embedding_lookup -->
### `embedding_lookup`

| Field | Value |
|---|---|
| Math | `y = table[token_ids] * sqrt(width)` |
| Contract | ids [tokens] u32; table [257152, 2048] bf16; the sqrt scale is part of the semantic |
| Probe stage | `prefix_v_layer{0,depth-1}` |
| Frequency | once per request |
| Importance | Boundary |
| Fallback | `ReferenceComparison` |
| Exit criterion | embedding::lookup_*; the tied embed_tokens/lm_head weight needs to be resolved at load, not here. |

<!-- l3-operator:concat_rows -->
### `concat_rows`

| Field | Value |
|---|---|
| Math | `rows of A followed by rows of B` |
| Contract | two [tokens, 2048] bf16 -> [tokens_a + tokens_b, 2048] |
| Probe stage | `prefix_v_layer{0,depth-1}` |
| Frequency | once per request |
| Importance | Boundary |
| Fallback | `ReferenceComparison` |
| Exit criterion | elementwise::concat_rows_*; a pure data movement with no arithmetic to get wrong. |

<!-- l3-operator:euler_update -->
### `euler_update`

| Field | Value |
|---|---|
| Math | `x <- x + dt * v` |
| Contract | [horizon, 32] f32; dt is a host scalar |
| Probe stage | `denoise_step_{s}` |
| Frequency | once per denoising step |
| Importance | High |
| Fallback | `ReferenceComparison` |
| Exit criterion | elementwise::euler_update_*; dt must be -flow_start_time / num_flow_steps, not -1 / num_flow_steps. |

## Resolution

`src/resolve.rs` walks the same admission chain the CUDA layer walks
(`registry.cu:229-275`): device features, contract, alignment, graph safety,
determinism, workspace budget. With an empty registry it stops at the first step
and reports `no_candidate_registered`. Running the binary prints the full
per-semantic report:

```sh
cargo run --manifest-path crates/apxinf-musa/Cargo.toml --bin operator-gaps -- --token-count 10
```

The report is `apxinf.musa.operator-resolution.v1`. It shares stage names with the
stage probe, so "this semantic is deferred" and "this stage has no MUSA number"
are the same statement in both documents.
