//! NVTX names for the native BF16 PI0.5 execution path.
//!
//! The names are intentionally private to the model implementation. Stage,
//! layer, and flow-step ranges carry the dynamic indices; operator ranges are
//! short stable names nested inside those ranges so Nsight can group them
//! without allocating a long string for every kernel launch.

pub(super) const CAPTURE: &str = "pi05.bf16.graph_capture";
pub(super) const PREPROCESS: &str = "pi05.bf16.preprocess";
pub(super) const VISION: &str = "pi05.bf16.vision";
pub(super) const VISION_PATCH_EMBED: &str = "pi05.bf16.vision.patch_embed";
pub(super) const VISION_POST_NORM: &str = "pi05.bf16.vision.post_norm";
pub(super) const VISION_PROJECTOR: &str = "pi05.bf16.vision.projector";
pub(super) const PREFIX: &str = "pi05.bf16.prefix";
pub(super) const PREFIX_EMBED: &str = "pi05.bf16.prefix.token_embed";
pub(super) const PREFIX_CONCAT: &str = "pi05.bf16.prefix.concat";
pub(super) const MODULATION_CONDITIONING: &str = "pi05.bf16.denoise.conditioning";
pub(super) const MODULATION_PROJECTION: &str = "pi05.bf16.denoise.modulation_projection";
pub(super) const ACTION_INPUT_GEMM: &str = "pi05.bf16.denoise.action_in.gemm";
pub(super) const ACTION_INPUT_BIAS: &str = "pi05.bf16.denoise.action_in.bias";
pub(super) const ACTION_OUTPUT_GEMM: &str = "pi05.bf16.denoise.action_out.gemm";
pub(super) const ACTION_OUTPUT_BIAS: &str = "pi05.bf16.denoise.action_out.bias";
pub(super) const EULER: &str = "pi05.bf16.denoise.euler_update";

pub(super) const OP_NORM: &str = "op.norm";
pub(super) const OP_PATCH_GEMM: &str = "op.patch_gemm";
pub(super) const OP_POSITION: &str = "op.position_embedding";
pub(super) const OP_QKV_GEMM: &str = "op.qkv_gemm";
pub(super) const OP_QKV_SPLIT_ROPE: &str = "op.qkv_split_rope";
pub(super) const OP_QKV_ROPE_CACHE: &str = "op.qkv_rope_cache_write";
pub(super) const OP_ATTENTION: &str = "op.attention";
pub(super) const OP_OUTPUT_GEMM: &str = "op.output_gemm";
pub(super) const OP_RESIDUAL_NORM: &str = "op.residual_norm";
pub(super) const OP_GATE_UP_GEGLU: &str = "op.gate_up_geglu";
pub(super) const OP_DOWN_GEMM: &str = "op.down_gemm";
pub(super) const OP_MLP_IN_GEMM: &str = "op.mlp_in_gemm";
pub(super) const OP_ACTIVATION: &str = "op.activation";
pub(super) const OP_RESIDUAL: &str = "op.residual";

pub(super) fn vision_layer(index: usize) -> String {
    format!("pi05.bf16.vision.layer_{index:02}")
}

pub(super) fn prefix_layer(index: usize) -> String {
    format!("pi05.bf16.prefix.layer_{index:02}")
}

pub(super) fn modulation_step(index: usize) -> String {
    format!("pi05.bf16.denoise.prepare_modulation.step_{index:02}")
}

pub(super) fn denoise_step(index: usize) -> String {
    format!("pi05.bf16.denoise.step_{index:02}")
}

pub(super) fn action_layer(index: usize) -> String {
    format!("pi05.bf16.denoise.action.layer_{index:02}")
}
