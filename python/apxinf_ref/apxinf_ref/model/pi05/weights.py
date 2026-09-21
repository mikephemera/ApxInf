"""Checkpoint loading for the reference runtime.

The tensor names are the OpenPI PyTorch export's own names -- the same tree the
engine reads, rooted at ``paligemma_with_expert``
(``crates/apxinf-model/src/pi05/weights/host.rs:16``). Nothing is
renamed or reshaped here: the engine transposes ``[out, in]`` into its row-major
GEMM layout and folds Gemma's ``1 + gamma`` into the consuming weights, and both
of those are *engine* representation choices. The reference deliberately keeps
the checkpoint's own orientation so that a mismatch caused by the folding is
visible in the comparison instead of being reproduced on both sides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping

from ...device import torch_module

__all__ = ["ROOT", "Pi05Weights", "load_weights", "materialize_precision"]

#: ``crates/apxinf-model/src/pi05/weights/host.rs:16`` -- the root every
#: PaliGemma tensor sits under.
ROOT = "paligemma_with_expert"

_WEIGHT_FILES = ("model.safetensors",)


class Pi05Weights:
    """A loaded checkpoint, addressing tensors by their export names."""

    def __init__(self, tensors: Mapping[str, "object"], source: Path) -> None:
        self._tensors = dict(tensors)
        self.source = source

    def __len__(self) -> int:
        return len(self._tensors)

    def __contains__(self, name: str) -> bool:
        return name in self._tensors

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensors)

    def get(self, name: str):
        """Return the named tensor, or raise naming the closest candidates."""
        try:
            return self._tensors[name]
        except KeyError:
            tail = name.rsplit(".", 1)[-1]
            near = sorted(k for k in self._tensors if k.endswith("." + tail))[:6]
            hint = f" Did you mean one of: {', '.join(near)}?" if near else ""
            raise KeyError(f"checkpoint has no tensor {name!r}.{hint}") from None

    def linear(self, prefix: str) -> tuple["object", "object | None"]:
        """``(weight, bias)`` for a ``nn.Linear``-style tensor pair."""
        bias = self._tensors.get(prefix + ".bias")
        return self.get(prefix + ".weight"), bias

    def to(self, *, dtype=None, device=None) -> "Pi05Weights":
        """Return a copy with every tensor cast and/or moved.

        Casting is done once, up front, so that the forward pass never mixes
        dtypes by accident: the engine uploads BF16 weights and the reference
        must see the same values the engine sees.
        """
        moved = {}
        for name, tensor in self._tensors.items():
            if dtype is not None:
                tensor = tensor.to(dtype)
            if device is not None:
                tensor = tensor.to(device)
            moved[name] = tensor
        return Pi05Weights(moved, self.source)

#: ``gemma_pytorch.py:71-78`` ``params_to_keep_float32``, matched as substrings
#: against parameter names that do *not* carry the ``paligemma_with_expert.``
#: root, exactly as ``named_parameters()`` reports them.
#:
#: The substring matching has a consequence worth stating plainly: "input_layernorm"
#: and "model.norm" also match the action expert's adaRMS `dense` projections, so
#: those stay float32 too, while the *vision* LayerNorm weights are not matched by
#: anything and stay bfloat16. `dump_vendor_dtypes.py` in the private workspace
#: records the resulting split so this list is not taken on faith.
_UPSTREAM_KEEP_FLOAT32 = (
    "vision_tower.vision_model.embeddings.patch_embedding.weight",
    "vision_tower.vision_model.embeddings.patch_embedding.bias",
    "vision_tower.vision_model.embeddings.position_embedding.weight",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
)

#: The four PI0.5 projection modules live on ``PI0Pytorch`` rather than on
#: ``PaliGemmaWithExpertModel``, so ``to_bfloat16_for_selected_params`` never
#: reaches them and they are float32 under every precision setting. They are
#: what carries the timestep condition into adaRMS, and because ``F.linear``
#: refuses mixed dtypes rather than promoting, that float32-ness is load-bearing:
#: ``adarms_cond`` must be float32 for the model to run at all.
_PI05_FLOAT32_PREFIXES = (
    "action_in_proj",
    "action_out_proj",
    "time_mlp_in",
    "time_mlp_out",
)


def materialize_precision(weights: "Pi05Weights", precision: str) -> "Pi05Weights":
    """Cast every tensor the way the upstream port's loader would.

    ``precision`` is the checkpoint's own field: ``bfloat16`` or ``float32``.
    This is not a free choice -- it is what ``to_bfloat16_for_selected_params``
    computes, and matching it is the difference between an assembled model that
    reproduces the checkpoint's reference outputs and one that merely looks
    like it.
    """
    torch = torch_module()
    if precision == "float32":
        return weights.to(dtype=torch.float32)
    if precision != "bfloat16":
        raise ValueError(f"unsupported precision {precision!r}: expected bfloat16 or float32")

    tensors = {}
    for name, tensor in weights._tensors.items():
        local = name[len(ROOT) + 1 :] if name.startswith(ROOT + ".") else name
        if name.startswith(ROOT + "."):
            keep_float32 = any(selector in local for selector in _UPSTREAM_KEEP_FLOAT32)
            tensors[name] = tensor.to(torch.float32 if keep_float32 else torch.bfloat16)
        elif name.startswith(_PI05_FLOAT32_PREFIXES):
            tensors[name] = tensor.to(torch.float32)
        else:
            raise KeyError(
                f"checkpoint tensor {name!r} is under neither {ROOT} nor a known "
                "PI0.5 projection module; the precision policy would have to guess."
            )
    return Pi05Weights(tensors, weights.source)


def _normalize_lerobot_prefix(tensors: dict) -> dict:
    """Port of ``crates/apxinf-model/src/pi05/weights/host.rs:485``
    ``normalize_lerobot_prefix``.

    A LeRobot export wraps the whole tree in ``model.``. The engine strips it
    only when no canonical key is present *and* at least one wrapped key is;
    the reference does the same so both sides index the same tree.
    """
    canonical = f"{ROOT}."
    wrapped = f"model.{ROOT}."
    if any(name.startswith(canonical) for name in tensors) or not any(
        name.startswith(wrapped) for name in tensors
    ):
        return tensors
    return {
        (name[len("model.") :] if name.startswith("model.") else name): tensor
        for name, tensor in tensors.items()
    }


def load_weights(path: str | Path, *, device=None, dtype=None, verbose: bool = False) -> Pi05Weights:
    """Load a checkpoint directory (or a safetensors file) into a ``Pi05Weights``.

    Only the file is read here; casting happens at ``to()`` time so that the
    caller decides whether the comparison runs in the checkpoint's own BF16 or
    in FP32.
    """
    root = Path(path)
    if root.is_dir():
        candidates = [root / name for name in _WEIGHT_FILES]
        source = next((c for c in candidates if c.is_file()), None)
        if source is None:
            present = sorted(p.name for p in root.iterdir())[:12]
            raise FileNotFoundError(
                f"{root} has no {' or '.join(_WEIGHT_FILES)}. Present: {present}"
            )
    else:
        source = root
        if not source.is_file():
            raise FileNotFoundError(f"no such checkpoint file: {source}")

    from safetensors import safe_open

    tensors: dict = {}
    with safe_open(str(source), framework="pt") as handle:
        metadata = handle.metadata() or {}
        for name in handle.keys():
            tensors[name] = handle.get_tensor(name)

    # The export declares the tied embedding in metadata instead of storing a
    # second copy: ``embed_tokens.weight`` is ``lm_head.weight``.
    for alias, target in metadata.items():
        if alias not in tensors and target in tensors:
            tensors[alias] = tensors[target]

    tensors = _normalize_lerobot_prefix(tensors)

    if verbose:
        print(f"[apxinf_ref] loaded {len(tensors)} tensors from {source}")

    weights = Pi05Weights(tensors, source)
    if device is not None or dtype is not None:
        weights = weights.to(dtype=dtype, device=device)
    return weights
