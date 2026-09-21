//! Per-stage numerical probe for PI0.5, in the precision the target needs.
//!
//! This example intentionally bypasses the unified `AutoModel`/`infer` frontend
//! to reach the per-precision models and the stage-level kernel signatures
//! directly, which the frontend does not (and should not) expose. See
//! `pi05_auto_smoke` for the abstraction-level entry point.
//!
//! It emits `apxinf.pi05.stage-probe.v1`, the same document
//! `python/apxinf_ref` emits, so the two can be subtracted stage by stage and a
//! numerical disagreement localised to a layer instead of an action.
//!
//! Three precisions are reachable, because the interesting comparison differs by
//! target: BF16 needs no calibration artifacts and is the shortest chain on
//! Orin, FP8 is the Thor path, and W8A8 is what `resolve_precision` picks
//! automatically on SM80-family parts.
//!
//! # Why this file exists next to `pi05_integrity_probe`
//!
//! `pi05_integrity_probe` is the older, FP8-only, positional-argument form of
//! this probe. It emits the same document, so a comparison against it is still
//! meaningful, but it carries two defects that were found by building the
//! reference runtime and are fixed only here:
//!
//! * its `prefix_v_layer*` signature covers `prefix_rows + action_horizon` rows
//!   and `cache::reserve_prefix_bf16`
//!   (`crates/apxinf-cuda/src/kernels/cache.rs`) writes only `prefix_rows`, so
//!   those two stages fold uninitialized device memory into the result and are
//!   not reproducible even between two runs. This file signs only written rows
//!   (`device_row_signature`);
//! * it hard-codes `dt = -1.0 / num_flow_steps` where `denoise_all_steps` uses
//!   `-flow_start_time / num_flow_steps`
//!   (`crates/apxinf-model/src/pi05/model/mod.rs`). The two agree while
//!   `flow_start_time` is 1.0 and diverge silently otherwise.
//!
//! Keeping this probe in its own file rather than editing that one is
//! deliberate: the older file is also maintained on the main branch, and a
//! shared file that both branches rewrite is a merge conflict waiting to
//! happen. Prefer this one. See `doc/pi05-reference-runtime.md`.
//!
//! For the same reason this file does not chase that one's stage names. The
//! PI0.5 refactor renamed its `vision_patch_embed` key to
//! `vision_patch_embed_fp8_static`; this probe keeps the bare name the reference
//! runtime emits, and `compare.py` matches the two by stripping the precision
//! suffix and reporting the pair under `aliased_stages`. That keeps the naming
//! migration on the branch that owns it.

use std::path::PathBuf;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Tensor};
use apxinf_cuda::{CudaBackend, CudaBuffer};
use apxinf_model::pi05::{
    build_bf16_model, build_fp8_static_model, build_int8_dynamic_model, checkpoint_identity,
    upload_time_embeddings_bf16, upload_time_embeddings_fp8_static,
    upload_time_embeddings_int8_dynamic, vision_layer_bf16, vision_layer_fp8_static,
    vision_layer_int8_dynamic, vision_patch_embed_bf16, vision_patch_embed_fp8_static,
    vision_patch_embed_int8_dynamic, vision_qkv_packed_from_env, Bf16Model, Bf16PrefixKvCache,
    Bf16Weights, Fp8StaticActivationScales, Fp8StaticCalibration, Fp8StaticModel,
    Fp8StaticPrefixKvCache, Fp8StaticWeights, Int8DynamicModel, Int8DynamicPrefixKvCache,
    Int8DynamicWeights, Pi05Config, Pi05Weights,
};

fn signature(values: &[f32]) -> serde_json::Value {
    let elements = values.len();
    let sample_count = elements.min(256);
    let mut sum = 0.0f64;
    let mut abs_checksum = 0.0f64;
    let mut square_sum = 0.0f64;
    let mut max_abs = 0.0f64;
    for &value in values {
        let value = f64::from(value);
        sum += value;
        abs_checksum += value.abs();
        square_sum += value * value;
        max_abs = max_abs.max(value.abs());
    }
    let sample = (0..sample_count)
        .map(|index| {
            let source = if sample_count == 1 {
                0
            } else {
                index * (elements - 1) / (sample_count - 1)
            };
            values[source]
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "elements": elements,
        "sum": sum,
        "abs_checksum": abs_checksum,
        "l2": square_sum.sqrt(),
        "max_abs": max_abs,
        "sample": sample,
    })
}

fn device_signature(
    backend: &CudaBackend,
    tensor: &Tensor,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let values = backend.to_cpu(tensor)?.to_f32_vec()?;
    Ok(signature(&values))
}

/// Signature over the first `rows` rows of a matrix.
///
/// `prefix_forward` allocates its K/V cache with `prefix_rows + action_horizon`
/// rows and `cache::reserve_prefix_bf16` writes only the first `prefix_rows`
/// (`crates/apxinf-cuda/src/kernels/cache.rs:111`). Signing the whole buffer
/// would fold uninitialized device memory into the result, which would make the
/// stage irreproducible even between two runs of this probe.
fn device_row_signature(
    backend: &CudaBackend,
    tensor: &Tensor,
    rows: usize,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let dims = tensor.shape().dims().to_vec();
    if dims.len() != 2 || rows > dims[0] {
        return Err(format!(
            "expected a [rows, cols] matrix with at least {rows} rows, got {dims:?}"
        )
        .into());
    }
    let values = backend.to_cpu(tensor)?.to_f32_vec()?;
    Ok(signature(&values[..rows * dims[1]]))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Precision {
    Bf16,
    Fp8,
    Int8,
}

impl Precision {
    fn parse(name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match name {
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            "int8" | "w8a8" => Ok(Self::Int8),
            other => Err(format!("unknown precision {other:?}: expected bf16, fp8 or int8").into()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
            Self::Int8 => "int8_w8a8",
        }
    }

    /// The dtype `encode_vision` accepts for this precision.
    fn patch_dtype(self) -> DType {
        match self {
            Self::Bf16 | Self::Int8 => DType::BF16,
            Self::Fp8 => DType::F16,
        }
    }
}

/// The per-precision model plus the state the probe drives it with.
///
/// Every arm exposes the same layer-level calls, so the probing loop below is
/// written once. This mirrors `pi05_bench`'s `Bench` enum, which exists for the
/// same reason on the inference surface.
///
/// The device weights stay here rather than being read back off the model: the
/// stage-level functions take them directly, and the model does not expose the
/// blocks it was built from.
enum ProbeRuntime {
    Bf16 {
        model: Bf16Model,
        weights: Arc<Bf16Weights>,
        time_embeddings: Vec<Tensor>,
    },
    Fp8 {
        model: Fp8StaticModel,
        weights: Arc<Fp8StaticWeights>,
        scales: Arc<Fp8StaticActivationScales>,
        time_embeddings: Vec<Tensor>,
    },
    Int8 {
        model: Int8DynamicModel,
        weights: Arc<Int8DynamicWeights>,
        time_embeddings: Vec<Tensor>,
    },
}

/// The three precisions return differently-typed K/V caches with the same content.
enum ProbePrefix {
    Bf16(Bf16PrefixKvCache),
    Fp8(Fp8StaticPrefixKvCache),
    Int8(Int8DynamicPrefixKvCache),
}

impl ProbePrefix {
    fn values(&self) -> &[Tensor] {
        match self {
            Self::Bf16(cache) => &cache.values,
            Self::Fp8(cache) => &cache.values,
            Self::Int8(cache) => &cache.values,
        }
    }

    fn tokens(&self) -> usize {
        match self {
            Self::Bf16(cache) => cache.tokens,
            Self::Fp8(cache) => cache.tokens,
            Self::Int8(cache) => cache.tokens,
        }
    }
}

struct ProbeContext {
    backend: Arc<CudaBackend>,
    config: Arc<Pi05Config>,
}

impl ProbeRuntime {
    fn vision_patch_embed(
        &self,
        ctx: &ProbeContext,
        patches: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let patches_per_view = ctx.config.patches_per_view();
        Ok(match self {
            Self::Bf16 { weights, .. } => vision_patch_embed_bf16(
                ctx.backend.context(),
                &weights.patch_embedding,
                &weights.position_embedding,
                patches,
                patches_per_view,
            )?,
            Self::Fp8 { weights, scales, .. } => vision_patch_embed_fp8_static(
                ctx.backend.context(),
                &weights.patch_embedding,
                &weights.position_embedding,
                patches,
                patches_per_view,
                scales.vision_patch_input,
            )?,
            Self::Int8 { weights, .. } => vision_patch_embed_int8_dynamic(
                ctx.backend.context(),
                &weights.patch_embedding,
                &weights.position_embedding,
                patches,
                patches_per_view,
            )?,
        })
    }

    fn vision_layer(
        &self,
        ctx: &ProbeContext,
        index: usize,
        hidden: &Tensor,
        packed_qkv: bool,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let config = &ctx.config;
        Ok(match self {
            Self::Bf16 { weights, .. } => vision_layer_bf16(
                ctx.backend.context(),
                &weights.vision_layers[index],
                hidden,
                config.patches_per_view(),
                config.vision_heads,
                config.vision_head_dim,
                config.layer_norm_eps,
            )?,
            Self::Fp8 { weights, scales, .. } => vision_layer_fp8_static(
                ctx.backend.context(),
                &weights.vision_layers[index],
                scales.vision_layers[index],
                hidden,
                config.patches_per_view(),
                config.vision_heads,
                config.vision_head_dim,
                packed_qkv,
                config.layer_norm_eps,
            )?,
            Self::Int8 { weights, .. } => vision_layer_int8_dynamic(
                ctx.backend.context(),
                &weights.vision_layers[index],
                hidden,
                config.patches_per_view(),
                config.vision_heads,
                config.vision_head_dim,
                config.layer_norm_eps,
            )?,
        })
    }

    fn encode_vision(
        &self,
        patches: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        Ok(match self {
            Self::Bf16 { model, .. } => model.encode_vision(patches)?,
            Self::Fp8 { model, .. } => model.encode_vision(patches)?,
            Self::Int8 { model, .. } => model.encode_vision(patches)?,
        })
    }

    fn embed_prefix(
        &self,
        vision_tokens: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        Ok(match self {
            Self::Bf16 { model, .. } => {
                model.embed_prefix(vision_tokens, token_ids, token_count)?
            }
            Self::Fp8 { model, .. } => {
                model.embed_prefix(vision_tokens, token_ids, token_count)?
            }
            Self::Int8 { model, .. } => {
                model.embed_prefix(vision_tokens, token_ids, token_count)?
            }
        })
    }

    fn prefix_forward(
        &self,
        prefix: &Tensor,
    ) -> Result<ProbePrefix, Box<dyn std::error::Error>> {
        Ok(match self {
            Self::Bf16 { model, .. } => ProbePrefix::Bf16(model.prefix_forward(prefix)?),
            Self::Fp8 { model, .. } => ProbePrefix::Fp8(model.prefix_forward(prefix)?),
            Self::Int8 { model, .. } => ProbePrefix::Int8(model.prefix_forward(prefix)?),
        })
    }

    fn denoise_step(
        &self,
        state: &Tensor,
        time_embedding: &Tensor,
        prefix: &ProbePrefix,
        dt: f32,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        Ok(match (self, prefix) {
            (Self::Bf16 { model, .. }, ProbePrefix::Bf16(cache)) => {
                model.denoise_step(state, time_embedding, cache, dt)?
            }
            (Self::Fp8 { model, .. }, ProbePrefix::Fp8(cache)) => {
                model.denoise_step(state, time_embedding, cache, dt)?
            }
            (Self::Int8 { model, .. }, ProbePrefix::Int8(cache)) => {
                model.denoise_step(state, time_embedding, cache, dt)?
            }
            _ => {
                return Err("probe runtime and prefix cache disagree on precision".into());
            }
        })
    }

    fn time_embeddings(&self) -> &[Tensor] {
        match self {
            Self::Bf16 { time_embeddings, .. } => time_embeddings,
            Self::Fp8 { time_embeddings, .. } => time_embeddings,
            Self::Int8 { time_embeddings, .. } => time_embeddings,
        }
    }
}

/// Everything the probe needs from the command line.
struct Arguments {
    checkpoint: PathBuf,
    precision: Precision,
    calibration: Option<PathBuf>,
    tactics: Option<PathBuf>,
    token_count: usize,
    views: Option<usize>,
}

fn parse_arguments(arguments: &[String]) -> Result<Arguments, Box<dyn std::error::Error>> {
    if arguments.len() < 2 {
        return Err(format!(
            "usage: {} <checkpoint> [--precision bf16|fp8|int8] [--calibration <json>] \
             [--tactics <json>] [--token-count <n>] [--views <n>]",
            arguments
                .first()
                .map(String::as_str)
                .unwrap_or("pi05_stage_probe")
        )
        .into());
    }

    let mut precision = Precision::Bf16;
    let mut calibration: Option<PathBuf> = None;
    let mut tactics: Option<PathBuf> = None;
    let mut token_count = 10usize;
    let mut views: Option<usize> = None;

    let mut index = 2;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--precision" => {
                index += 1;
                precision = Precision::parse(arguments.get(index).ok_or("--precision needs a value")?)?;
            }
            "--calibration" => {
                index += 1;
                calibration = Some(
                    arguments
                        .get(index)
                        .ok_or("--calibration needs a path")?
                        .into(),
                );
            }
            "--tactics" => {
                index += 1;
                tactics = Some(arguments.get(index).ok_or("--tactics needs a path")?.into());
            }
            "--token-count" => {
                index += 1;
                token_count = arguments
                    .get(index)
                    .ok_or("--token-count needs a value")?
                    .parse()?;
            }
            "--views" => {
                index += 1;
                views = Some(arguments.get(index).ok_or("--views needs a value")?.parse()?);
            }
            other => return Err(format!("unrecognized argument {other:?}").into()),
        }
        index += 1;
    }

    Ok(Arguments {
        checkpoint: PathBuf::from(&arguments[1]),
        precision,
        calibration,
        tactics,
        token_count,
        views,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    let Arguments {
        checkpoint: checkpoint_dir,
        precision,
        calibration: calibration_path,
        tactics: tactics_path,
        token_count,
        views,
    } = parse_arguments(&arguments)?;

    let mut config = Pi05Config::thor_two_view();
    if let Some(views) = views {
        config.num_views = views;
    }
    let config = Arc::new(config);

    let checkpoint_dir = checkpoint_dir.as_path();
    let checkpoint = checkpoint_identity(checkpoint_dir)?;
    let backend = Arc::new(CudaBackend::new(0)?);

    if let Some(path) = &tactics_path {
        let tuning = apxinf_cuda::tuning::TuningDb::from_json_file(path)?;
        apxinf_cuda::kernels::gemm::install_tuning_db(backend.context(), &tuning)?;
    }

    eprintln!("loading π0.5 checkpoint for {}...", precision.name());
    let host_weights = Pi05Weights::from_safetensors(&config, checkpoint_dir)?;

    let ctx = ProbeContext {
        backend: backend.clone(),
        config: config.clone(),
    };

    let probe = match precision {
        Precision::Bf16 => {
            if calibration_path.is_some() {
                eprintln!("note: --calibration is not used by the bf16 path");
            }
            let weights = Arc::new(Bf16Weights::from_host(
                &host_weights,
                &*backend,
                config.language_dual_geglu_shape_possible(),
            )?);
            drop(host_weights);
            let time_embeddings = upload_time_embeddings_bf16(&config, &*backend)?;
            let model = build_bf16_model(backend.clone(), config.clone(), weights.clone())?;
            ProbeRuntime::Bf16 {
                model,
                weights,
                time_embeddings,
            }
        }
        Precision::Fp8 => {
            let path = calibration_path
                .as_ref()
                .ok_or("the fp8 path requires --calibration <json>")?;
            let calibration = Fp8StaticCalibration::from_json_file(path, &config, &checkpoint)?;
            let scales = Arc::new(Fp8StaticActivationScales::from_calibration(
                &config, &calibration,
            )?);
            let weights = Arc::new(Fp8StaticWeights::from_host(
                &host_weights,
                &*backend,
                config.language_dual_geglu_shape_possible(),
            )?);
            drop(host_weights);
            let time_embeddings = upload_time_embeddings_fp8_static(&config, &*backend)?;
            let model = build_fp8_static_model(
                backend.clone(),
                config.clone(),
                weights.clone(),
                scales.clone(),
            )?;
            ProbeRuntime::Fp8 {
                model,
                weights,
                scales,
                time_embeddings,
            }
        }
        Precision::Int8 => {
            if calibration_path.is_some() {
                eprintln!("note: --calibration is not used by the int8 path");
            }
            let weights = Arc::new(Int8DynamicWeights::from_host(&host_weights, &*backend)?);
            drop(host_weights);
            let time_embeddings = upload_time_embeddings_int8_dynamic(&config, &*backend)?;
            let model = build_int8_dynamic_model(backend.clone(), config.clone(), weights.clone())?;
            ProbeRuntime::Int8 {
                model,
                weights,
                time_embeddings,
            }
        }
    };

    let patch_tokens = config.num_views * config.patches_per_view();
    let patch_width = 3 * config.patch_size * config.patch_size;
    let patches = backend.to_device(&Tensor::zeros(
        vec![patch_tokens, patch_width],
        precision.patch_dtype(),
    ))?;
    let noise = backend.to_device(&Tensor::zeros(
        vec![config.action_horizon, config.action_dim],
        precision.patch_dtype(),
    ))?;
    let token_ids =
        CudaBuffer::alloc_zeros(token_count * 4, backend.device_id()).map_err(std::io::Error::other)?;

    let mut signatures = serde_json::Map::new();

    eprintln!("probing patch embedding and each vision layer...");
    let packed_vision_qkv = if precision == Precision::Fp8 {
        vision_qkv_packed_from_env()?
    } else {
        false
    };
    let mut vision_hidden = probe.vision_patch_embed(&ctx, &patches)?;
    signatures.insert(
        "vision_patch_embed".into(),
        device_signature(&backend, &vision_hidden)?,
    );
    for index in 0..config.vision_depth {
        vision_hidden = probe.vision_layer(&ctx, index, &vision_hidden, packed_vision_qkv)?;
        signatures.insert(
            format!("vision_layer_{index}"),
            device_signature(&backend, &vision_hidden)?,
        );
    }

    eprintln!("probing vision projection...");
    let vision = probe.encode_vision(&patches)?;
    signatures.insert(
        "vision_projected".into(),
        device_signature(&backend, &vision)?,
    );

    eprintln!("probing language prefix K/V...");
    let prefix_input = probe.embed_prefix(&vision, &token_ids, token_count)?;
    let prefix = probe.prefix_forward(&prefix_input)?;
    let prefix_tokens = prefix.tokens();
    let values = prefix.values();
    signatures.insert(
        "prefix_v_layer0".into(),
        device_row_signature(&backend, &values[0], prefix_tokens)?,
    );
    signatures.insert(
        format!("prefix_v_layer{}", config.language.depth - 1),
        device_row_signature(&backend, &values[config.language.depth - 1], prefix_tokens)?,
    );

    eprintln!("probing ten denoising steps...");
    let mut state = noise;
    // The engine's own step size; note this is `-flow_start_time / steps`, which
    // the probe previously hard-coded as `-1.0 / steps` and would have diverged
    // from the runtime had `flow_start_time` ever moved off 1.0.
    let dt = -config.flow_start_time / config.num_flow_steps as f32;
    let time_embeddings = probe.time_embeddings().to_vec();
    for (step, embedding) in time_embeddings.iter().enumerate() {
        state = probe.denoise_step(&state, embedding, &prefix, dt)?;
        signatures.insert(
            format!("denoise_step_{step}"),
            device_signature(&backend, &state)?,
        );
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "apxinf.pi05.stage-probe.v1",
            "precision": precision.name(),
            "token_count": token_count,
            "intermediate_signatures": signatures,
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    //! The signature, pinned against the same numbers `python/apxinf_ref` asserts.
    //!
    //! The reference carries the matching expectations for its own implementation
    //! of the algorithm (`python/apxinf_ref/tests/test_contract.py`). Both sides
    //! assert against the *same literals*, so the two documents are guaranteed to
    //! be comparable without having to run both -- which matters because the
    //! engine side of the comparison only runs on a CUDA host, and a mismatch
    //! discovered there costs a whole run.

    fn sign(values: &[f32]) -> serde_json::Value {
        super::signature(values)
    }

    #[test]
    fn signature_matches_the_python_side_element_for_element() {
        let value = sign(&[1.0, -2.0, 3.5, 0.0]);
        assert_eq!(value["elements"], 4);
        assert_eq!(value["sum"], 2.5);
        assert_eq!(value["abs_checksum"], 6.5);
        assert_eq!(value["l2"], 4.153311931459037_f64);
        assert_eq!(value["max_abs"], 3.5);
        assert_eq!(value["sample"], serde_json::json!([1.0, -2.0, 3.5, 0.0]));

        // A real stage carries more elements than the sample limit, which is
        // where the grid stops being the identity: `i * (elements - 1) /
        // (sample_count - 1)`, integer division, both ends kept. That is the
        // branch every vision and denoising stage takes, and `compare` reads the
        // samples positionally -- so two implementations whose grids drift see
        // values from different positions and report a difference that is not one.
        let values: Vec<f32> = (0..1000).map(|v| v as f32).collect();
        let value = sign(&values);
        assert_eq!(value["elements"], 1000);
        assert_eq!(value["sum"], 499500.0);
        assert_eq!(value["l2"], 18243.72494859534_f64);
        assert_eq!(value["max_abs"], 999.0);
        let sample = value["sample"].as_array().expect("sample array");
        assert_eq!(sample.len(), 256);
        assert_eq!(sample[0], 0.0);
        assert_eq!(sample[1], 3.0); // 1 * 999 / 255, floored
        assert_eq!(sample[2], 7.0);
        assert_eq!(sample[254], 995.0);
        assert_eq!(sample[255], 999.0);
    }
}
