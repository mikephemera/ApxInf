//! The operator catalog: one entry per semantic the PI0.5 hot path needs.
//!
//! The list is derived from the engine, not invented. Each entry names the
//! operation as the Blocks compose it
//! (`crates/apxinf-model/src/pi05/model/blocks/bf16.rs`) and the stage of
//! `apxinf.pi05.stage-probe.v1` it feeds, so
//! that "this semantic is deferred" and "this stage has no MUSA number" are the
//! same statement.
//!
//! # Why every entry is a gap
//!
//! `crates/apxinf-cuda-new/cuda-operator.md:76-82` states the rule this catalog
//! obeys: a model that needs a semantic the L3 layer does not expose "must
//! record an operator gap rather than infer support from the legacy
//! implementation". What `apxinf-cuda-new` publishes as L3 is `ops::gemm`,
//! `ops::gemm_bias`, `ops::gemm_bias_gelu`, `ops::gemm_geglu`, and -- since the
//! attention adapter landed -- `ops::attention`, `ops::kv_cache_attention` and
//! `ops::segmented_attention` (`crates/apxinf-cuda-new/src/ops/mod.rs:19-22`).
//! Note that `cuda-operator.md:64-82` still lists only the four GEMM operators:
//! the attention entries postdate that file, so read the Rust surface rather
//! than the catalog file for what is published. PI0.5 still needs normalisation,
//! RoPE, embeddings, elementwise updates and six fused compositions, so most of
//! what is below remains a gap even on CUDA -- and on MUSA, where nothing has
//! been ported, all of it is.
//!
//! # The fallback column
//!
//! `doc/adding-new-kernels.md:163` asks each gap to record its fallback as
//! "existing composition, ordinary cuBLAS, or none". Here the answer is a fourth
//! kind: [`Fallback::ReferenceComparison`]. No implementation stands in for the
//! missing operator at run time; instead the stage is compared against
//! `python/apxinf_ref`, and the comparison says plainly that no MUSA number
//! exists for it. That is the only fallback that cannot become a silent
//! correctness claim, because it never produces a number at all.

use crate::spec::{DType, ScaleKind, Spec};

/// A distinct mathematical operation, as opposed to a distinct call site.
///
/// The language tower's QKV projection and the vision tower's are the same
/// semantic with different `Spec`s; the vision tower's is a different semantic
/// because it adds a bias the language tower does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Semantic {
    /// LayerNorm, as SigLIP uses it.
    LayerNorm,
    /// RMSNorm with the `1 + weight` convention, as Gemma uses it.
    RmsNorm,
    /// RMSNorm whose scale and shift come from a conditioning vector, plus the
    /// gate the same projection produces.
    AdaptiveRmsNorm,
    /// Plain dense projection.
    Gemm,
    /// Packed gate/up projection with `gelu(gate) * up` folded in.
    GemmGeglu,
    /// QKV projection followed by splitting into three tensors, with bias.
    SplitQkvBias,
    /// QKV projection followed by splitting and rotary position application.
    SplitQkvApplyRope,
    /// Rotary position application to the query, and a write of the key and
    /// value into a prefix cache.
    ApplyQueryWriteKv,
    /// Bidirectional multi-head attention over windows.
    MultiHeadAttention,
    /// Causal multi-query attention.
    MultiQueryAttention,
    /// Projection plus bias plus residual add.
    BiasResidual,
    /// Projection, bias, residual add, and the normalisation of the sum -- the
    /// second output of which feeds the next layer.
    BiasResidualRmsNorm,
    /// The LayerNorm variant of the above, used by the vision tower.
    BiasResidualLayerNorm,
    /// Projection, then an adaptive gate applied to the residual, then an
    /// adaptive normalisation; the second output feeds the next layer.
    AdaptiveGateResidualRmsNorm,
    /// Projection, bias, and the tanh GELU.
    BiasGelu,
    /// Add a fixed position table to a projection output, per window.
    AddPosition,
    /// Token embedding lookup.
    EmbeddingLookup,
    /// Concatenate two row blocks into one.
    ConcatRows,
    /// One explicit Euler step of the flow-matching solver.
    EulerUpdate,
}

impl Semantic {
    /// The name the catalog document uses, and the `<!-- l3-operator:... -->` marker.
    pub fn name(self) -> &'static str {
        match self {
            Self::LayerNorm => "layer_norm",
            Self::RmsNorm => "rms_norm",
            Self::AdaptiveRmsNorm => "adaptive_rms_norm",
            Self::Gemm => "gemm",
            Self::GemmGeglu => "gemm_geglu",
            Self::SplitQkvBias => "split_qkv_bias",
            Self::SplitQkvApplyRope => "split_qkv_apply_rope",
            Self::ApplyQueryWriteKv => "apply_query_write_kv",
            Self::MultiHeadAttention => "mha",
            Self::MultiQueryAttention => "mqa",
            Self::BiasResidual => "bias_residual",
            Self::BiasResidualRmsNorm => "bias_residual_rms_norm",
            Self::BiasResidualLayerNorm => "bias_residual_layer_norm",
            Self::AdaptiveGateResidualRmsNorm => "adaptive_gate_residual_rms_norm",
            Self::BiasGelu => "bias_gelu",
            Self::AddPosition => "add_position",
            Self::EmbeddingLookup => "embedding_lookup",
            Self::ConcatRows => "concat_rows",
            Self::EulerUpdate => "euler_update",
        }
    }

    /// The safe CUDA interface that already implements this semantic.
    ///
    /// Recorded because `doc/adding-new-kernels.md:186` asks the port to resolve
    /// coverage against maintained interfaces before inventing anything. It is a
    /// pointer, not a claim that the same code can run on MUSA: the CUDA crate's
    /// kernels are `.cu`, and `crates/apxinf-cuda-new/cuda-operator.md:82`
    /// forbids reading support off a legacy implementation.
    pub fn legacy_reference(self) -> &'static str {
        match self {
            Self::LayerNorm => "kernels::norm::layer_bf16",
            Self::RmsNorm => "kernels::norm::rms_bf16",
            Self::AdaptiveRmsNorm => "kernels::norm::adaptive_rms_bf16",
            Self::Gemm => "ops::gemm (published L3)",
            Self::GemmGeglu => "ops::gemm_geglu (published L3)",
            Self::SplitQkvBias => "kernels::attention::split_qkv_bias_bf16",
            Self::SplitQkvApplyRope => "kernels::rope::split_qkv_apply_bf16",
            Self::ApplyQueryWriteKv => "kernels::rope::apply_q_write_kv_bf16",
            Self::MultiHeadAttention => "ops::attention (published L3)",
            Self::MultiQueryAttention => "ops::kv_cache_attention (published L3)",
            Self::BiasResidual => "kernels::fused::bias_residual_bf16",
            Self::BiasResidualRmsNorm => "kernels::fused::bias_residual_rms_bf16",
            Self::BiasResidualLayerNorm => "kernels::fused::bias_residual_layer_bf16",
            Self::AdaptiveGateResidualRmsNorm => {
                "kernels::fused::adaptive_gate_residual_rms_bf16"
            }
            Self::BiasGelu => "kernels::activation::bias_gelu_bf16",
            Self::AddPosition => "kernels::embedding::add_position_bf16",
            Self::EmbeddingLookup => "kernels::embedding::lookup_bf16",
            Self::ConcatRows => "kernels::elementwise::concat_rows_bf16",
            Self::EulerUpdate => "kernels::elementwise::euler_update_bf16",
        }
    }
}

/// How important this semantic is to the port's latency, per
/// `doc/adding-new-kernels.md:159`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Importance {
    /// Called per layer, per denoising step, or otherwise in the innermost loop.
    High,
    /// Called once or twice per request.
    Medium,
    /// Called at a boundary.
    Boundary,
}

impl Importance {
    pub fn name(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Medium => "medium",
            Self::Boundary => "boundary",
        }
    }
}

/// What happens to a call for a semantic with no MUSA implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// Compare the stage against `python/apxinf_ref` and report that no MUSA
    /// number exists. Produces no value, so it cannot be mistaken for one.
    ReferenceComparison,
    /// An existing safe interface produces the right answer more slowly.
    ExistingComposition,
    /// No path at all; the semantic must be implemented before the stage runs.
    None,
}

impl Fallback {
    pub fn name(self) -> &'static str {
        match self {
            Self::ReferenceComparison => "reference_comparison",
            Self::ExistingComposition => "existing_composition",
            Self::None => "none",
        }
    }
}

/// One row of the gap table.
#[derive(Clone, Copy, Debug)]
pub struct OperatorGap {
    pub semantic: Semantic,
    /// The mathematical meaning, written out rather than named.
    pub math: &'static str,
    /// Shape, dtype and layout of the operands.
    pub contract: &'static str,
    /// The stage of `apxinf.pi05.stage-probe.v1` whose value this feeds.
    pub stage: &'static str,
    /// How often it runs.
    pub frequency: &'static str,
    pub importance: Importance,
    pub fallback: Fallback,
    /// `doc/adding-new-kernels.md:198` -- for anything standing in for a device
    /// implementation, the safe API that will replace it and the replay that
    /// retires it.
    pub exit_criterion: &'static str,
}

/// Every semantic the PI0.5 hot path needs, in the order the executors call them.
pub const SEMANTICS: &[OperatorGap] = &[
    OperatorGap {
        semantic: Semantic::LayerNorm,
        math: "(x - mean(x)) / sqrt(var(x) + eps) * weight + bias",
        contract: "[tokens, 1152] bf16 in and out; weight and bias [1152] bf16; eps 1e-6",
        stage: "vision_layer_{i}",
        frequency: "twice per vision layer",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "norm::layer_* for bf16 at 1152 wide; retired when vision_layer_{i} reports a MUSA number within the bf16 threshold of the reference.",
    },
    OperatorGap {
        semantic: Semantic::RmsNorm,
        math: "x * rsqrt(mean(x**2) + eps) * (1 + weight)",
        contract: "[tokens, 2048] bf16 in and out; weight [2048] f32; eps 1e-6",
        stage: "prefix_v_layer{0,depth-1}",
        frequency: "twice per language layer",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "norm::rms_* at 2048 wide; retired when the prefix K/V stages agree with the reference.",
    },
    OperatorGap {
        semantic: Semantic::AdaptiveRmsNorm,
        math: "mod = dense(cond); scale, shift, gate = chunk(mod, 3); (x * rsqrt(mean(x**2)+eps)) * (1+scale) + shift, and gate",
        contract: "[tokens, 1024] bf16; cond [1, 1024] f32; dense weight [3072, 1024] f32; two outputs",
        stage: "denoise_step_{s}",
        frequency: "twice per action layer per step",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "norm::adaptive_rms_* with the f32 conditioning contract; retired when the denoise stages agree with the reference. Note the conditioning is f32 upstream and bf16 in the engine today, so the two disagree by construction until that is reconciled.",
    },
    OperatorGap {
        semantic: Semantic::Gemm,
        math: "y = x @ W + b",
        contract: "[tokens, K] bf16 x [K, N] bf16, optionally plus bias [N]",
        stage: "vision_layer_{i}, prefix_v_layer*, denoise_step_{s}",
        frequency: "four to seven times per layer",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "ops::gemm and ops::gemm_bias already publish this semantic on CUDA; the MUSA question is the provider, not the contract.",
    },
    OperatorGap {
        semantic: Semantic::GemmGeglu,
        math: "gelu(x @ W_gate) * (x @ W_up), with gate and up packed into one weight",
        contract: "[tokens, K] bf16 x [K, 2N] bf16; gate columns first",
        stage: "prefix_v_layer*, denoise_step_{s}",
        frequency: "once per language and action layer",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "ops::gemm_geglu already publishes this semantic on CUDA; packing stays internal.",
    },
    OperatorGap {
        semantic: Semantic::SplitQkvBias,
        math: "split a [tokens, 3 * heads * head_dim] projection into q, k, v and add each bias",
        contract: "[tokens, 3456] bf16 -> three [tokens, 1152] bf16; 16 heads, head_dim 72",
        stage: "vision_layer_{i}",
        frequency: "once per vision layer",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "attention::split_qkv_bias_*; retired when vision_layer_{i} is reachable on MUSA.",
    },
    OperatorGap {
        semantic: Semantic::SplitQkvApplyRope,
        math: "split as above, then rotate q and k by position",
        contract: "[tokens, 2304] bf16 -> q [tokens, 2048], k/v [tokens, 256]; 8 heads, 1 kv head, head_dim 256",
        stage: "prefix_v_layer{0,depth-1}",
        frequency: "once per language layer",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "rope::split_qkv_apply_*; retired when the prefix K/V stages agree with the reference.",
    },
    OperatorGap {
        semantic: Semantic::ApplyQueryWriteKv,
        math: "rotate q by position and write the rotated k and v into the prefix cache at the offset",
        contract: "q/k/v [tokens, 256] bf16; cache [prefix + horizon, 256] bf16",
        stage: "denoise_step_{s}",
        frequency: "once per action layer per step",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "rope::apply_q_write_kv_*; the cache rows written must exclude the reserved tail that the probe currently signs.",
    },
    OperatorGap {
        semantic: Semantic::MultiHeadAttention,
        math: "softmax(q @ k^T / sqrt(d)) @ v, bidirectional, within each window",
        contract: "q [windows, heads, patches_per_view, 72]; 16 heads",
        stage: "vision_layer_{i}",
        frequency: "once per vision layer",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "attention::mha_*; retired when vision_layer_{i} is reachable on MUSA.",
    },
    OperatorGap {
        semantic: Semantic::MultiQueryAttention,
        math: "softmax(q @ k^T / sqrt(d)) @ v with one shared kv head, causal, over prefix plus suffix",
        contract: "q [tokens, 8, 256]; k/v [tokens, 1, 256]",
        stage: "prefix_v_layer{0,depth-1}, denoise_step_{s}",
        frequency: "once per language and action layer",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "attention::mqa_*; retired when the K/V and denoise stages agree with the reference.",
    },
    OperatorGap {
        semantic: Semantic::BiasResidual,
        math: "y = x @ W + b + residual",
        contract: "[tokens, K] bf16; bias [N]; residual [tokens, N]",
        stage: "vision_layer_{i}, prefix_v_layer{0,depth-1}",
        frequency: "twice per vision layer, once per language layer",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "fused::bias_residual_*; retired when the enclosing layer is reachable on MUSA.",
    },
    OperatorGap {
        semantic: Semantic::BiasResidualRmsNorm,
        math: "h = x @ W + b + residual; y = h * rsqrt(mean(h**2) + eps) * (1 + weight); two outputs",
        contract: "[tokens, 2048] bf16; the normalised output feeds the next layer's QKV",
        stage: "prefix_v_layer{0,depth-1}",
        frequency: "once per language layer",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "fused::bias_residual_rms_*; the second output must be kept, it is the next layer's input.",
    },
    OperatorGap {
        semantic: Semantic::BiasResidualLayerNorm,
        math: "h = x @ W + b + residual; y = LayerNorm(h); two outputs",
        contract: "[tokens, 1152] bf16",
        stage: "vision_layer_{i}",
        frequency: "once per vision layer",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "fused::bias_residual_layer_*; retired when vision_layer_{i} is reachable on MUSA.",
    },
    OperatorGap {
        semantic: Semantic::AdaptiveGateResidualRmsNorm,
        math: "h = x @ W + residual * gate; y = (h * rsqrt(mean(h**2)+eps)) * (1 + next_scale) + next_shift; two outputs",
        contract: "[tokens, 1024] bf16; gate, scale and shift from the f32 conditioning",
        stage: "denoise_step_{s}",
        frequency: "twice per action layer per step",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "fused::adaptive_gate_residual_rms_*; the second output is the next layer's normalised input and must not be recomputed differently.",
    },
    OperatorGap {
        semantic: Semantic::BiasGelu,
        math: "gelu(x @ W + b) with the tanh approximation",
        contract: "[tokens, 1152] -> [tokens, 4304] bf16",
        stage: "vision_layer_{i}",
        frequency: "once per vision layer",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "activation::bias_gelu_*; the tanh approximation is part of the contract, not an implementation detail.",
    },
    OperatorGap {
        semantic: Semantic::AddPosition,
        math: "y = x @ W + b + position_table[window]",
        contract: "[views * 256, 1152] bf16; position table [256, 1152]",
        stage: "vision_patch_embed",
        frequency: "once per request",
        importance: Importance::Boundary,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "embedding::add_position_*; retired when vision_patch_embed is reachable on MUSA.",
    },
    OperatorGap {
        semantic: Semantic::EmbeddingLookup,
        math: "y = table[token_ids] * sqrt(width)",
        contract: "ids [tokens] u32; table [257152, 2048] bf16; the sqrt scale is part of the semantic",
        stage: "prefix_v_layer{0,depth-1}",
        frequency: "once per request",
        importance: Importance::Boundary,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "embedding::lookup_*; the tied embed_tokens/lm_head weight needs to be resolved at load, not here.",
    },
    OperatorGap {
        semantic: Semantic::ConcatRows,
        math: "rows of A followed by rows of B",
        contract: "two [tokens, 2048] bf16 -> [tokens_a + tokens_b, 2048]",
        stage: "prefix_v_layer{0,depth-1}",
        frequency: "once per request",
        importance: Importance::Boundary,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "elementwise::concat_rows_*; a pure data movement with no arithmetic to get wrong.",
    },
    OperatorGap {
        semantic: Semantic::EulerUpdate,
        math: "x <- x + dt * v",
        contract: "[horizon, 32] f32; dt is a host scalar",
        stage: "denoise_step_{s}",
        frequency: "once per denoising step",
        importance: Importance::High,
        fallback: Fallback::ReferenceComparison,
        exit_criterion: "elementwise::euler_update_*; dt must be -flow_start_time / num_flow_steps, not -1 / num_flow_steps.",
    },
];

/// Look up a gap by semantic.
pub fn gap(semantic: Semantic) -> Option<&'static OperatorGap> {
    SEMANTICS.iter().find(|entry| entry.semantic == semantic)
}

/// The shape a `Spec` takes for this semantic on PI0.5's two-view profile.
///
/// Providing it here keeps candidate `supports` predicates honest: a candidate
/// that only handles a wide, aligned, unquantised contraction can say so against
/// a real spec rather than a hand-written one in a test.
pub fn reference_spec(semantic: Semantic, token_count: usize) -> Spec {
    const PATCHES_PER_VIEW: usize = 256;
    const PATCH_TOKENS: usize = 2 * PATCHES_PER_VIEW;
    let prefix = PATCH_TOKENS + token_count;
    match semantic {
        Semantic::LayerNorm | Semantic::BiasResidualLayerNorm | Semantic::BiasGelu => {
            Spec::dense(PATCH_TOKENS, 1152, 1152, DType::Bf16)
        }
        Semantic::AddPosition => Spec::dense(PATCH_TOKENS, 588, 1152, DType::Bf16),
        Semantic::RmsNorm | Semantic::BiasResidualRmsNorm | Semantic::BiasResidual => {
            Spec::dense(prefix, 2048, 2048, DType::Bf16)
        }
        Semantic::GemmGeglu => Spec::dense(prefix, 2048, 32768, DType::Bf16),
        Semantic::SplitQkvBias => Spec::dense(PATCH_TOKENS, 1152, 3456, DType::Bf16),
        Semantic::SplitQkvApplyRope => Spec::dense(prefix, 2048, 2304, DType::Bf16),
        Semantic::ApplyQueryWriteKv => Spec::dense(10, 1024, 768, DType::Bf16),
        Semantic::MultiHeadAttention => {
            let mut spec = Spec::dense(PATCH_TOKENS, 72, 1152, DType::Bf16);
            spec.windows = 2;
            spec
        }
        Semantic::MultiQueryAttention => Spec::dense(prefix, 256, 2048, DType::Bf16),
        Semantic::AdaptiveRmsNorm | Semantic::AdaptiveGateResidualRmsNorm => {
            Spec::dense(10, 1024, 1024, DType::Bf16)
        }
        Semantic::EmbeddingLookup => Spec::dense(token_count, 2048, 257152, DType::Bf16),
        Semantic::ConcatRows => Spec::dense(prefix, 2048, 2048, DType::Bf16),
        Semantic::EulerUpdate => {
            let mut spec = Spec::dense(10, 32, 32, DType::F32);
            spec.accumulation = DType::F32;
            spec
        }
        Semantic::Gemm => Spec::dense(prefix, 2048, 2048, DType::Bf16),
    }
}

/// Scale kinds this catalog expects to see once a quantized MUSA path exists.
///
/// Declared now so the `Spec` shape does not have to change later; no candidate
/// consumes them yet.
pub const QUANTIZED_SCALE_KINDS: &[ScaleKind] = &[ScaleKind::Unit, ScaleKind::RowChannel];
