//! Pre-rendered NVTX range names for the π0.5 inference path.
//!
//! These label the eager (non-CUDA-graph) path so an nsys timeline can be
//! attributed by pipeline stage and transformer layer. The graph path bakes
//! the whole pipeline into one `cudaGraphLaunch`, so host-side labels inside
//! the stage functions only fire at capture time, never at replay; see the
//! module docs on `pi05_bench`.
//!
//! Names follow the reference FlashRT trace (`Pi05.<stage>`, `key=value` fields,
//! `layer_NN` zero-padded to two digits) so the two timelines can be compared
//! side by side. Layer names carry their tensor shape in parentheses: `x=` for a
//! single activation, `q=`/`kv=` for the action expert's attention operands.
//!
//! Two label families live here. [`Bf16TraceNames`] pre-renders the labels that
//! are fixed once the config is known and that only the BF16 backbone emits —
//! the vision tower and its 27 layers. The free functions serve the rest: the
//! request-dependent labels (encoder/prefix/decoder depend on the prefix length,
//! so they are built on the way in) plus the two stage labels the precision-generic
//! schedule emits. A `format!` per layer costs ~100 ns against a pass that runs a
//! thousand times longer, and it keeps the shape in the label truthful.

use super::Pi05Config;

/// Range names the BF16 vision tower emits, fixed once the config is known.
#[derive(Clone)]
pub(crate) struct Bf16TraceNames {
    /// `Pi05.vision images=(...) patches=(...) layers=27 hidden=1152`
    pub vision: String,
    /// `Pi05.vision.layer_00 x=(512,1152)`, one per vision block.
    pub vision_layers: Vec<String>,
}

impl Bf16TraceNames {
    pub fn new(config: &Pi05Config) -> Self {
        let vision_rows = config.num_views * config.patches_per_view();
        let vision_layers = (0..config.vision_depth)
            .map(|layer| {
                format!(
                    "Pi05.vision.layer_{layer:02} x=({vision_rows},{})",
                    config.vision_width
                )
            })
            .collect();
        Self {
            vision: format!(
                "Pi05.vision images=({},{},{},{}) patches=({},{}) layers={} hidden={}",
                config.num_views,
                3,
                config.image_size,
                config.image_size,
                vision_rows,
                3 * config.patch_size * config.patch_size,
                config.vision_depth,
                config.vision_width,
            ),
            vision_layers,
        }
    }
}

/// `Pi05.pipeline.full_forward precision=bf16 seq=522 steps=10` — the outermost
/// range. Emitted by the precision-generic schedule, so `precision` is passed in.
pub(crate) fn full_forward(config: &Pi05Config, precision: &str, prefix_tokens: usize) -> String {
    format!(
        "Pi05.pipeline.full_forward precision={precision} seq={prefix_tokens} steps={}",
        config.num_flow_steps
    )
}

/// `Pi05.pipeline.denoise steps=10` — the flow-matching loop.
pub(crate) fn denoise(config: &Pi05Config) -> String {
    format!("Pi05.pipeline.denoise steps={}", config.num_flow_steps)
}

/// `Pi05.encoder seq=10 hidden=2048` — the token embedding + prefix concat.
pub(crate) fn embed(config: &Pi05Config, token_count: usize) -> String {
    format!(
        "Pi05.encoder seq={token_count} hidden={}",
        config.language.width
    )
}

/// `Pi05.encoder.prefix prefix=522 layers=18 hidden=2048` — the language prefill.
pub(crate) fn prefix(config: &Pi05Config, prefix_tokens: usize) -> String {
    format!(
        "Pi05.encoder.prefix prefix={prefix_tokens} layers={} hidden={}",
        config.language.depth, config.language.width
    )
}

/// `Pi05.encoder.layer_00 x=(522,2048)` — one language prefill block.
pub(crate) fn encoder_layer(config: &Pi05Config, layer: usize, prefix_tokens: usize) -> String {
    format!(
        "Pi05.encoder.layer_{layer:02} x=({prefix_tokens},{})",
        config.language.width
    )
}

/// `Pi05.decoder.step_00 x=(10,1024) enc=(522,2048)` — one flow-matching step.
///
/// `x` is the step's working width after the action-in projection; `enc` is the
/// language prefix its attention reads, not the 32-wide noisy action input.
pub(crate) fn decoder_step(config: &Pi05Config, step: usize, prefix_tokens: usize) -> String {
    format!(
        "Pi05.decoder.step_{step:02} x=({},{}) enc=({prefix_tokens},{})",
        config.action_horizon, config.action_expert.width, config.language.width,
    )
}

/// `Pi05.decoder.step_00.layer_00 q=(10,1024) kv=(522,2048)` — one action block.
pub(crate) fn decoder_layer(
    config: &Pi05Config,
    step: usize,
    layer: usize,
    prefix_tokens: usize,
) -> String {
    format!(
        "Pi05.decoder.step_{step:02}.layer_{layer:02} q=({},{}) kv=({prefix_tokens},{})",
        config.action_horizon, config.action_expert.width, config.language.width,
    )
}
