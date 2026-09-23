#!/usr/bin/env python3
"""Generate a deterministic Pi0.5 PyTorch eager reference.

The model implementation is the vendored ``PI0Pytorch`` snapshot under
``python/apxinf_ref/_vendor``.  This command deliberately drives the canonical
tensor boundary directly: images are already normalized ``NCHW`` tensors,
token ids are integer ids, and noise is the action-space input to the Euler
loop.  That keeps a CUDA capture and a MUSA replay on exactly the same input
bytes and avoids mixing tokenizer or camera preprocessing into a kernel
comparison.

Examples::

    # Produce the CUDA reference on a CUDA host.
    python python/apxinf_ref/generate_pi05_torch_gold.py \
        --checkpoint /path/to/pi05_libero_pytorch \
        --device cuda:0 \
        --output devlocal/pi05-torch-gold/cuda

    # On a MUSA host, replay the exact fixture saved by the CUDA run.
    python python/apxinf_ref/generate_pi05_torch_gold.py \
        --checkpoint /path/to/pi05_libero_pytorch \
        --device musa:0 \
        --input devlocal/pi05-torch-gold/cuda/gold.npz \
        --output devlocal/pi05-torch-gold/musa

The output directory contains ``gold.npz`` and ``manifest.json``.  The NPZ
stores the three canonical inputs, the final raw action chunk, and sampled
intermediate tensors.  The manifest stores the full checkpoint identity,
device/runtime information, shapes, and scalar signatures for every captured
stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPO_ROOT / "python" / "apxinf_ref"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))


SCHEMA = "apxinf.pi05.torch-eager-gold.v1"
DEFAULT_CHECKPOINT = Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
DEFAULT_OUTPUT = REPO_ROOT / "devlocal" / "pi05-torch-gold" / "run"
DEFAULT_VIEWS = 2
DEFAULT_TOKEN_COUNT = 10
DEFAULT_SAMPLE_COUNT = 256
VISION_PREFIX = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
VISION_EMBEDDINGS = VISION_PREFIX + ".embeddings"
VISION_LAYERS = VISION_PREFIX + ".encoder.layers"
VISION_PROJECTOR = "paligemma_with_expert.paligemma.model.multi_modal_projector"
PREFIX_LAYERS = "paligemma_with_expert.paligemma.model.language_model.layers"
DECODER_LAYERS = "paligemma_with_expert.gemma_expert.model.layers"


def _bootstrap_torch(device_spec: str):
    """Import torchada before torch so the same file can run on MUSA and CUDA."""

    if device_spec.split(":", 1)[0].strip().lower() == "musa":
        try:
            import torchada  # noqa: F401  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "device 'musa' requires torchada; install the MUSA torchada extra"
            ) from error
    import torch

    return torch


torch = None


def _device(spec: str):
    device = torch.device(spec)
    if device.type == "cpu":
        return device
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device but torch.cuda.is_available() is false")
        count = torch.cuda.device_count()
    elif device.type == "musa":
        musa = getattr(torch, "musa", None)
        if musa is None or not musa.is_available():
            raise RuntimeError("requested MUSA device but torch.musa.is_available() is false")
        count = musa.device_count()
    else:
        raise ValueError(f"unsupported device {spec!r}; choose cpu, cuda:N, or musa:N")
    index = 0 if device.index is None else device.index
    if index < 0 or index >= count:
        raise RuntimeError(f"requested {device}, but this host has {count} {device.type} device(s)")
    return torch.device(device.type, index)


def _synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "musa":
        torch.musa.synchronize(device)


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_checkpoint(path: Path) -> tuple[Path, Path, dict[str, Any]]:
    root = path.expanduser().resolve()
    if root.is_dir():
        weights = root / "model.safetensors"
        config_path = root / "config.json"
    else:
        weights = root
        config_path = root.parent / "config.json"
        root = root.parent
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint weights not found: {weights}")
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config not found: {config_path}")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint config is not a JSON object: {config_path}")
    return root, weights, config


def _config_value(config: dict[str, Any], *names: str, default: Any) -> Any:
    for name in names:
        if name in config:
            return config[name]
    return default


def _load_fixture(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("images", "token_ids", "noise")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"fixture {path} is missing arrays: {', '.join(missing)}")
        arrays = {name: np.asarray(data[name]) for name in required}
    if arrays["images"].ndim != 4 or arrays["images"].shape[1] != 3:
        raise ValueError(f"images must have shape [views,3,height,width], got {arrays['images'].shape}")
    if arrays["images"].shape[2:] != (224, 224):
        raise ValueError(f"images must be 224x224 canonical tensors, got {arrays['images'].shape}")
    if arrays["token_ids"].ndim != 1:
        raise ValueError(f"token_ids must have shape [tokens], got {arrays['token_ids'].shape}")
    if arrays["noise"].ndim != 2 or arrays["noise"].shape[1] != 32:
        raise ValueError(f"noise must have shape [horizon,32], got {arrays['noise'].shape}")
    if not np.isfinite(arrays["images"]).all() or not np.isfinite(arrays["noise"]).all():
        raise ValueError(f"fixture {path} contains non-finite image/noise values")
    if arrays["images"].dtype != np.float32:
        arrays["images"] = arrays["images"].astype(np.float32)
    if arrays["token_ids"].dtype != np.int64:
        arrays["token_ids"] = arrays["token_ids"].astype(np.int64)
    if arrays["noise"].dtype != np.float32:
        arrays["noise"] = arrays["noise"].astype(np.float32)
    return arrays


def _make_fixture(*, seed: int, views: int, token_count: int, horizon: int) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    return {
        "images": generator.uniform(-1.0, 1.0, (views, 3, 224, 224)).astype(np.float32),
        "token_ids": generator.integers(0, 257152, token_count, dtype=np.int64),
        "noise": generator.standard_normal((horizon, 32), dtype=np.float32),
    }


def _validate_fixture(arrays: dict[str, np.ndarray], *, max_token_len: int, views: int | None, horizon: int) -> None:
    if views is not None and arrays["images"].shape[0] != views:
        raise ValueError(f"fixture has {arrays['images'].shape[0]} views but --views={views}")
    if arrays["token_ids"].size == 0 or arrays["token_ids"].size > max_token_len:
        raise ValueError(f"fixture token count must be in 1..{max_token_len}, got {arrays['token_ids'].size}")
    if arrays["noise"].shape != (horizon, 32):
        raise ValueError(f"fixture noise shape must be {(horizon, 32)}, got {arrays['noise'].shape}")


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _signature(tensor, *, sample_count: int) -> dict[str, Any]:
    values = tensor.detach().float().cpu().numpy().reshape(-1)
    if values.size == 0:
        return {"elements": 0, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "sum": 0.0,
                "abs_sum": 0.0, "l2": 0.0, "max_abs": 0.0, "sample": []}
    count = min(sample_count, values.size)
    if count == 1:
        indices = np.array([0], dtype=np.int64)
    else:
        indices = (np.arange(count, dtype=np.int64) * (values.size - 1)) // (count - 1)
    return {
        "elements": int(values.size),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sum": float(np.sum(values, dtype=np.float64)),
        "abs_sum": float(np.sum(np.abs(values), dtype=np.float64)),
        "l2": float(np.sqrt(np.sum(np.square(values, dtype=np.float64), dtype=np.float64))),
        "max_abs": float(np.max(np.abs(values))),
        "sample": values[indices].astype(np.float32).tolist(),
    }


class _StageCollector:
    def __init__(self, *, sample_count: int):
        self.sample_count = sample_count
        self.step: int | None = None
        self.calls: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def hook(self, name: str):
        def collect(_module, _inputs, output):
            tensor = _first_tensor(output)
            if tensor is None:
                return
            stage = name
            if name.startswith("Pi05.decoder.layer_"):
                if self.step is None:
                    return
                stage = name.replace("Pi05.decoder.", f"Pi05.decoder.step_{self.step:02d}.")
            self.calls.setdefault(stage, []).append(_signature(tensor, sample_count=self.sample_count))

        return collect


def _register_stages(model, collector: _StageCollector):
    handles = []
    modules = dict(model.named_modules())
    if VISION_EMBEDDINGS in modules:
        handles.append(modules[VISION_EMBEDDINGS].register_forward_hook(collector.hook("Pi05.vision.patch_embed")))
    for index in range(64):
        name = f"{VISION_LAYERS}.{index}"
        if name not in modules:
            break
        handles.append(modules[name].register_forward_hook(collector.hook(f"Pi05.vision.layer_{index:02d}")))
    if VISION_PROJECTOR in modules:
        handles.append(modules[VISION_PROJECTOR].register_forward_hook(collector.hook("Pi05.vision.projected")))
    for index in range(64):
        name = f"{PREFIX_LAYERS}.{index}"
        if name not in modules:
            break
        handles.append(modules[name].register_forward_hook(collector.hook(f"Pi05.prefix.layer_{index:02d}")))
    for index in range(64):
        name = f"{DECODER_LAYERS}.{index}"
        if name not in modules:
            break
        handles.append(modules[name].register_forward_hook(collector.hook(f"Pi05.decoder.layer_{index:02d}")))
    return handles


def _pin_eager(model) -> None:
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"


def _load_model(weights: Path, config: dict[str, Any], device, precision: str):
    from _vendor.openpi_pi0_pytorch.transformers_overlay import install_transformers_overlay

    install_transformers_overlay()
    from _vendor.openpi_pi0_pytorch.pi0_config import Pi0Config
    from _vendor.openpi_pi0_pytorch.pi0_pytorch import PI0Pytorch
    from safetensors.torch import load_file
    checkpoint_precision = str(config.get("precision", "bfloat16"))
    dtype_name = checkpoint_precision if precision == "checkpoint" else precision
    if dtype_name not in ("bfloat16", "float32"):
        raise ValueError(f"unsupported checkpoint precision {dtype_name!r}")
    model_config = Pi0Config(
        dtype=dtype_name,
        paligemma_variant=str(config.get("paligemma_variant", "gemma_2b")),
        action_expert_variant=str(config.get("action_expert_variant", "gemma_300m")),
        action_dim=int(_config_value(config, "action_dim", "max_action_dim", default=32)),
        action_horizon=int(_config_value(config, "action_horizon", "chunk_size", default=10)),
        max_token_len=int(_config_value(config, "max_token_len", "tokenizer_max_length", default=200)),
        pi05=True,
        pytorch_compile_mode=None,
    )
    model = PI0Pytorch(model_config)
    state = load_file(str(weights), device="cpu")
    alias = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    target = "paligemma_with_expert.paligemma.lm_head.weight"
    if alias not in state and target in state:
        state[alias] = state[target]
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {alias}
    missing = sorted(set(missing) - allowed_missing)
    unexpected = sorted(unexpected)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint/model state mismatch; refusing to produce a gold file. "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    del state
    model.to(device=device)
    model.eval()
    _pin_eager(model)
    return model, model_config, dtype_name


def _run(model, model_config, arrays: dict[str, np.ndarray], device, *, num_steps: int, sample_count: int):
    from _vendor.openpi_pi0_pytorch.pi0_pytorch import make_att_2d_masks

    images = torch.from_numpy(arrays["images"]).to(device=device)
    token_ids = torch.from_numpy(arrays["token_ids"]).to(device=device)
    state = torch.zeros((1, model_config.action_dim), dtype=torch.float32, device=device)
    noise = torch.from_numpy(arrays["noise"]).to(device=device)
    image_list = [images[index : index + 1] for index in range(images.shape[0])]
    image_masks = [torch.ones((1,), dtype=torch.bool, device=device) for _ in image_list]
    lang_tokens = token_ids.unsqueeze(0)
    lang_masks = torch.ones((1, token_ids.numel()), dtype=torch.bool, device=device)

    collector = _StageCollector(sample_count=sample_count)
    handles = _register_stages(model, collector)
    try:
        with torch.inference_mode():
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                image_list, image_masks, lang_tokens, lang_masks
            )
            prefix_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_positions = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_mask_4d = model._prepare_attention_masks_4d(prefix_mask)
            language = model.paligemma_with_expert.language_model if hasattr(model.paligemma_with_expert, "language_model") else model.paligemma_with_expert.paligemma.language_model
            _, past_key_values = model.paligemma_with_expert.forward(
                attention_mask=prefix_mask_4d,
                position_ids=prefix_positions,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            del language

            dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            current = noise
            for step in range(num_steps):
                collector.step = step
                velocity = model.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    current.unsqueeze(0),
                    time.reshape(1),
                ).squeeze(0)
                current = current + dt * velocity
                time = time + dt
            raw_actions = current
    finally:
        for handle in handles:
            handle.remove()
    _synchronize(device)
    return raw_actions.detach().float().cpu().numpy(), collector.calls


def _write_output(output: Path, *, arrays, raw_actions, stages, manifest, force: bool) -> None:
    if output.exists():
        if not force and (output.is_file() or any(output.iterdir())):
            raise FileExistsError(f"output already exists: {output}; pass --force to replace it")
        if output.is_file():
            output.unlink()
        elif force:
            for child in output.iterdir():
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child)
                else:
                    child.unlink()
    output.mkdir(parents=True, exist_ok=True)
    npz_arrays = dict(arrays)
    npz_arrays["raw_actions"] = raw_actions
    stage_keys = []
    for index, (stage, calls) in enumerate(stages.items()):
        for call_index, record in enumerate(calls):
            key = f"stage_{index:04d}_call_{call_index:02d}"
            npz_arrays[key] = np.asarray(record["sample"], dtype=np.float32)
            stage_keys.append({"stage": stage, "call": call_index, "array": key})
    manifest["stage_arrays"] = stage_keys
    np.savez_compressed(output / "gold.npz", **npz_arrays)
    manifest["files"] = {
        "gold.npz": {"bytes": (output / "gold.npz").stat().st_size, "sha256": _sha256(output / "gold.npz")}
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("APXINF_PI05_CHECKPOINT", DEFAULT_CHECKPOINT)))
    parser.add_argument("--device", default="musa:0", help="cpu, cuda:N, or musa:N (default: musa:0)")
    parser.add_argument("--input", type=Path, help="existing gold.npz fixture to replay; outputs in it are ignored")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0, help="seed for a newly generated canonical fixture")
    parser.add_argument("--views", type=int, default=DEFAULT_VIEWS, help="views for a new fixture (default: 2)")
    parser.add_argument("--token-count", type=int, default=DEFAULT_TOKEN_COUNT)
    parser.add_argument("--num-flow-steps", type=int, default=10)
    parser.add_argument("--precision", choices=("checkpoint", "bfloat16", "float32"), default="checkpoint")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    global torch
    torch = _bootstrap_torch(args.device)
    if args.sample_count <= 0 or args.sample_count > 4096:
        raise SystemExit("--sample-count must be in 1..4096")
    if args.num_flow_steps <= 0:
        raise SystemExit("--num-flow-steps must be positive")
    device = _device(args.device)
    checkpoint_root, weights, config = _resolve_checkpoint(args.checkpoint)
    checkpoint_sha256 = _sha256(weights)
    action_horizon = int(_config_value(config, "action_horizon", "chunk_size", default=10))
    max_token_len = int(_config_value(config, "max_token_len", "tokenizer_max_length", default=200))
    if args.input is None:
        if args.views <= 0 or args.token_count <= 0:
            raise SystemExit("--views and --token-count must be positive")
        arrays = _make_fixture(seed=args.seed, views=args.views, token_count=args.token_count, horizon=action_horizon)
        fixture_source = {"kind": "generated", "seed": args.seed}
    else:
        arrays = _load_fixture(args.input.expanduser().resolve())
        fixture_source = {"kind": "npz", "path": str(args.input.expanduser().resolve()), "sha256": _sha256(args.input.expanduser().resolve())}
    _validate_fixture(arrays, max_token_len=max_token_len, views=None if args.input else args.views, horizon=action_horizon)

    model, model_config, dtype_name = _load_model(weights, config, device, args.precision)
    raw_actions, stages = _run(model, model_config, arrays, device, num_steps=args.num_flow_steps, sample_count=args.sample_count)
    _synchronize(device)
    manifest = {
        "schema": SCHEMA,
        "implementation": "vendored_openpi_pi0_pytorch",
        "execution": "torch_eager",
        "model": "pi05",
        "checkpoint": {"root": str(checkpoint_root), "weights": str(weights), "sha256": checkpoint_sha256, "config": config},
        "runtime": {"device": str(device), "device_type": device.type, "torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(), "precision": dtype_name},
        "fixture": fixture_source,
        "inputs": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
        "outputs": {"raw_actions": {"shape": list(raw_actions.shape), "dtype": str(raw_actions.dtype)}},
        "sampling": {"num_flow_steps": args.num_flow_steps, "flow_start_time": 1.0, "euler_time": "float32 time starts at 1 and subtracts 1/num_flow_steps"},
        "model_config": {"action_dim": model_config.action_dim, "action_horizon": model_config.action_horizon, "max_token_len": model_config.max_token_len, "num_views": int(arrays["images"].shape[0]), "paligemma_variant": model_config.paligemma_variant, "action_expert_variant": model_config.action_expert_variant},
        "stages": stages,
    }
    _write_output(args.output.expanduser().resolve(), arrays=arrays, raw_actions=raw_actions, stages=stages, manifest=manifest, force=args.force)
    print(json.dumps({"schema": SCHEMA, "device": str(device), "output": str(args.output.expanduser().resolve()), "stages": len(stages), "raw_actions_shape": list(raw_actions.shape)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
