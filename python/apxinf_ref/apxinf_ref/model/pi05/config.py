"""Pi0.5 configuration for the reference runtime.

The defaults mirror ``crates/apxinf-model/src/pi05/config.rs`` (``Pi05Config``,
``GEMMA_2B``, ``GEMMA_300M``, ``THOR_TWO_VIEW``) rather than the upstream OpenPI
dataclass, because the reference runtime exists to be subtracted from ApxInf's
own stage probe: the two sides must agree on every dimension before a single
number can be compared. Where the checkpoint's ``config.json`` states a value,
it wins.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

__all__ = ["GemmaVariant", "Pi05Config", "load_config"]


@dataclasses.dataclass(frozen=True)
class GemmaVariant:
    """Decoder dimensions, matching ``config.rs`` ``GemmaVariantConfig``."""

    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


#: ``config.rs`` ``GEMMA_2B`` -- the PaliGemma language tower.
GEMMA_2B = GemmaVariant(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256)

#: ``config.rs`` ``GEMMA_300M`` -- the action expert.
GEMMA_300M = GemmaVariant(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)

_VARIANTS = {"gemma_2b": GEMMA_2B, "gemma_300m": GEMMA_300M}


@dataclasses.dataclass(frozen=True)
class Pi05Config:
    """Every dimension and scalar the reference forward pass reads."""

    language: GemmaVariant = GEMMA_2B
    action_expert: GemmaVariant = GEMMA_300M

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200
    num_flow_steps: int = 10
    #: Flow-matching start time. ``denoise_all_steps`` in the engine steps
    #: ``dt = -flow_start_time / num_flow_steps``.
    flow_start_time: float = 1.0
    num_views: int = 3

    image_size: int = 224
    patch_size: int = 14
    vision_width: int = 1152
    vision_depth: int = 27
    vision_mlp_dim: int = 4304
    vision_heads: int = 16
    vision_head_dim: int = 72

    vocab_size: int = 257152
    rms_norm_eps: float = 1e-6
    layer_norm_eps: float = 1e-6
    rope_theta: float = 1e4
    time_min_period: float = 4e-3
    time_max_period: float = 4.0

    #: ``precision`` from the checkpoint, for logging and dtype selection.
    precision: str = "bfloat16"

    @property
    def patches_per_view(self) -> int:
        """``config.rs`` ``patches_per_view``: (image_size / patch_size) ** 2."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def patch_width(self) -> int:
        return 3 * self.patch_size * self.patch_size

    @property
    def patch_tokens(self) -> int:
        return self.num_views * self.patches_per_view

    @property
    def max_prefix_len(self) -> int:
        return self.patch_tokens + self.max_token_len

    def language_dual_geglu_shape_possible(self) -> bool:
        """``config.rs:190`` -- whether the packed gate/up shape applies.

        Only the two-view profile reaches a token count where
        ``m > patch_tokens and m - patch_tokens <= max_token_len`` holds for the
        engine's dual-GeGLU packing, and the engine feeds that condition into
        ``Bf16Weights::from_host``. The reference does not pack, but
        the flag is recorded so a mismatch in the comparison is attributable.
        """
        return self.patch_tokens == 512


def _variant_from_name(name: str) -> GemmaVariant:
    try:
        return _VARIANTS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown Gemma variant {name!r}: expected one of {', '.join(sorted(_VARIANTS))}"
        ) from error


def from_mapping(document: Mapping[str, Any], *, num_views: int | None = None) -> Pi05Config:
    """Build a config from a checkpoint ``config.json``.

    The LeRobot/OpenPI aliases accepted here are the same ones
    ``config.rs:from_json_str`` accepts (``max_action_dim``, ``chunk_size``,
    ``tokenizer_max_length``, ``num_inference_steps``).
    """
    known = {field.name for field in dataclasses.fields(Pi05Config)}
    values: dict[str, Any] = {}

    aliases = {
        "max_action_dim": "action_dim",
        "chunk_size": "action_horizon",
        "tokenizer_max_length": "max_token_len",
        "num_inference_steps": "num_flow_steps",
    }
    for key, value in document.items():
        target = aliases.get(key, key)
        if target in known:
            values[target] = value

    if "paligemma_variant" in document:
        values["language"] = _variant_from_name(str(document["paligemma_variant"]))
    if "action_expert_variant" in document:
        values["action_expert"] = _variant_from_name(str(document["action_expert_variant"]))
    if num_views is not None:
        values["num_views"] = num_views

    config = Pi05Config(**values)
    config.validate()
    return config


def load_config(path: str | Path, *, num_views: int | None = None) -> Pi05Config:
    """Load a checkpoint directory's ``config.json``, or accept a JSON file."""
    root = Path(path)
    document_path = root / "config.json" if root.is_dir() else root
    if not document_path.is_file():
        raise FileNotFoundError(
            f"no config.json at {document_path}. The reference runtime reads the "
            "checkpoint's own config rather than guessing dimensions."
        )
    document = json.loads(document_path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{document_path} does not contain a JSON object")
    return from_mapping(document, num_views=num_views)


def _validate(self: Pi05Config) -> None:
    if self.language.depth != self.action_expert.depth:
        raise ValueError(
            "language and action-expert depth must match: the engine's prefix "
            "loop runs both towers' layers together (config.rs validate())."
        )
    for field in ("num_heads", "num_kv_heads", "head_dim"):
        if getattr(self.language, field) != getattr(self.action_expert, field):
            raise ValueError(f"language and action-expert {field} must match")
    if self.image_size % self.patch_size != 0:
        raise ValueError(
            f"image_size {self.image_size} is not a multiple of patch_size {self.patch_size}"
        )
    if self.action_dim % 1 or self.action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {self.action_dim}")


Pi05Config.validate = _validate  # type: ignore[attr-defined]
