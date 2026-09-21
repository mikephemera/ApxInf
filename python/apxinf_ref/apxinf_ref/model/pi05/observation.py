"""Observation in, actions out: the boundary the stage probe deliberately skips.

The stage probe takes patches and token ids directly, because that is what the
engine's own probe emits. This module is the other half -- it turns an
observation into exactly those inputs, and turns the model's raw action chunk
back into the seven values LIBERO executes.

Every step here is a choice that moves the numbers, so each one is written down
where it happens rather than left to a caller's default. Two of them are worth
reading before the code:

**The camera frames are rotated, and the resize is the pinned one.**
``scripts/libero_observation.libero_images`` owns the 180-degree rotation and
``apxinf.processors.ResizeWithPad`` owns the letterbox; both are reused rather
than restated, for the reasons their own module docstrings record. LIBERO's
demonstrations need the *same* rotation as the live simulator, because LIBERO
stores frames as robosuite rendered them (see ``model/pi05/libero.py``).

**The state width comes from the statistics it will be normalised against**, not
from a convention: ``finger_joints_for_state_width`` derives it, and a
disagreement is an error rather than a prompt the checkpoint never saw.

Differences from FlashRT's MUSA gold producer, recorded rather than reconciled --
they are two projects' conventions and neither is being changed here:

* it resizes with a plain PIL bilinear call where this uses the pinned letterbox
  ``ResizeWithPad`` (``processors/resize.py`` records that the two resizes
  disagree by up to 52/255 at 256 -> 224, enough to change a prompt's tokens);
* it normalises the state from ``ee_pos`` and the raw quaternion where this uses
  LIBERO's own axis-angle convention;
* its ``unnormalize_actions`` clips the raw chunk to ``[-1, 1]`` first; this
  reuses ``apxinf.processors.Unnormalizer``, which does not clip, because the
  clip is a property of that project's pipeline rather than of the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ...device import torch_module
from . import preprocess
from .tokenizer import PromptTokenizer

__all__ = [
    "ModelInputs",
    "PREPROCESSED_VIEW_NAMES",
    "VIEW_NAMES",
    "Observation",
    "assemble",
    "final_actions",
    "read_norm_stats",
]


#: The two camera views, in the order everything that stacks them uses: the
#: observation, the patch layout, and the arrays a capture stores. One tuple
#: because "which one is the base camera" is a fact that is obvious in three
#: places at once and wrong in one of them.
VIEW_NAMES = ("base_image", "wrist_image")

#: The same views as the normalized images, which is what a capture keeps to
#: localise a divergence to the resize or to the patch layout.
PREPROCESSED_VIEW_NAMES = ("preprocessed_base_rgb", "preprocessed_wrist_rgb")


def _import_apxinf(module: str):
    """A module from ``apxinf``, or a message saying what the observation path needs.

    Imported here rather than at the top of the file so that the stage probe --
    which takes patches and token ids directly -- keeps working on a host where
    ``apxinf`` is not installed.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as error:  # pragma: no cover - depends on the host
        raise ImportError(
            f"the observation path needs `{module}` from the `apxinf` package (it "
            "lives at python/apxinf in this repository; `pip install -e "
            "python/apxinf`). The stage probe does not need it."
        ) from error


@dataclass(frozen=True)
class Observation:
    """Two camera frames, proprioception and a task, before any conversion."""

    base_image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray
    prompt: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def oriented(self) -> "Observation":
        """Rotate both frames into the policy's orientation.

        The frames are stacked by the same function the live simulator path uses,
        so the HDF5 and the simulator cannot drift apart in orientation without
        this line changing too.
        """
        images = preprocess.orient_libero_images(self.base_image, self.wrist_image)
        return Observation(
            base_image=images[0],
            wrist_image=images[1],
            state=self.state,
            prompt=self.prompt,
            provenance=self.provenance,
        )

    def images(self) -> np.ndarray:
        """``[views, H, W, 3]`` in the policy's orientation, resized to the model."""
        return np.stack([self.base_image, self.wrist_image])


@dataclass(frozen=True)
class ModelInputs:
    """What the model consumes, plus the facts a comparison has to agree on.

    Three of the fields are here because a capture has to record them and
    ``assemble`` is the only place they exist:

    * ``formatted_prompt`` is the text the tokenizer was actually given, not the
      task string it was built from;
    * ``tokenizer_digest`` is the digest of the file that produced the token ids.
      Two hosts whose tokenizer files differ produce different ids and their
      stage signatures are not comparable, which no number in the document
      reveals;
    * ``preprocessed_rgb`` is the normalized image, the last step before the
      patch layout, so a capture that stores it can tell a resize divergence
      from a layout one. It stays on the host: nothing in the forward pass reads
      it.
    """

    patches: Any
    token_ids: Any
    noise: Any
    token_count: int
    state_width: int
    image_size: int
    source_image_size: tuple[int, int]
    prompt: str
    formatted_prompt: str
    tokenizer_digest: str
    preprocessed_rgb: Any
    provenance: dict[str, Any]


def read_norm_stats(path: str | Path) -> dict:
    """Read a ``norm_stats.json``, unwrapping the optional ``norm_stats`` envelope.

    Delegated to ``apxinf`` rather than restated: the envelope handling and the
    error type are the ones the policy loader uses, and two readers of one file
    that disagree about its shape is exactly the failure this avoids.
    """
    return dict(_import_apxinf("apxinf.checkpoints.norm_stats").read_norm_stats(Path(path)))


def _prompt_tokenizer(tokenizer_path: str | Path, max_token_len: int) -> PromptTokenizer:
    """The tokenizer the prompt goes through, with the state spliced in."""
    return PromptTokenizer(tokenizer_path, max_token_len=max_token_len, discrete_state=True)


def assemble(
    observation: Observation,
    *,
    config,
    tokenizer_path,
    norm_stats: Mapping[str, Any],
    device,
    seed: int,
) -> ModelInputs:
    """An observation already in the policy's frame -> the model's three inputs.

    The rotation is deliberately *not* applied here. Which sources need it
    depends on how they were captured: LIBERO's HDF5 stores frames as robosuite
    rendered them and needs it (see ``model/pi05/libero.py``), while a captured
    bundle holds frames someone already prepared. Applying it twice would be
    invisible in the numbers and wrong, so the caller says so explicitly with
    :meth:`Observation.oriented`.

    Resizing *is* applied here, and it is safe for both: ``ResizeWithPad``
    returns an already-square frame of the target size untouched, bit for bit.

    ``norm_stats`` is the parsed statistics document: its ``state`` entry decides
    both the normalisation and the joint count, and its ``actions`` entry is what
    :func:`final_actions` later uses. Both come from the same file on purpose.
    """
    torch = torch_module()
    stats = norm_stats["state"]
    state_width = len(np.asarray(stats["q01"]).reshape(-1))

    source = tuple(int(size) for size in observation.base_image.shape[:2])
    # One view at a time: `ResizeWithPad` is a single-image processor, and the
    # policy pipeline that owns it stacks views the same way (`ImageStack`).
    resized = np.stack(
        [
            preprocess.resize_with_pad(view, config.image_size)
            for view in observation.images()
        ]
    )
    # The normalized image is what the model consumes, so it is computed once
    # here and handed to the patch conversion rather than recomputed inside it.
    pixels = preprocess.normalized_pixels(
        np.asarray(resized), image_size=config.image_size
    )
    patches = preprocess.patches_from_pixels(
        torch.from_numpy(pixels), patch_size=config.patch_size
    )

    state = np.asarray(observation.state, dtype=np.float32).reshape(-1)
    if state.size != state_width:
        raise ValueError(
            f"this checkpoint's `state` statistics are {state_width} wide but the "
            f"observation carries {state.size} values. The joint count follows the "
            "statistics (see preprocess.finger_joints_for_state_width); a mismatch "
            "would discretise into a prompt the checkpoint was never trained on."
        )

    discrete_state = preprocess.normalize_state(state, stats)
    tokenizer = _prompt_tokenizer(tokenizer_path, config.max_token_len)
    token_ids = tokenizer(observation.prompt, discrete_state)
    token_ids = torch.from_numpy(np.asarray(token_ids, dtype=np.int64)).to(device)

    noise = preprocess.sample_noise(
        (config.action_horizon, config.action_dim), device, seed=seed
    )

    return ModelInputs(
        patches=patches.to(device),
        token_ids=token_ids,
        noise=noise,
        token_count=int(token_ids.shape[-1]),
        state_width=state_width,
        image_size=int(config.image_size),
        source_image_size=source,
        prompt=observation.prompt,
        # The same builder the tokenizer itself calls, so the recorded text is
        # the text that was tokenised rather than a second derivation of it.
        formatted_prompt=tokenizer.text(observation.prompt, discrete_state),
        tokenizer_digest=tokenizer.digest,
        preprocessed_rgb=pixels,
        provenance=dict(observation.provenance),
    )


def final_actions(raw_actions, *, norm_stats: Mapping[str, Any], action_dim: int | None = None):
    """The model's raw ``[horizon, action_dim]`` chunk as LIBERO's 7 values.

    The unnormalizer is ``apxinf.processors.Unnormalizer`` -- the same object the
    policy pipeline uses -- and it passes a wider array's tail through unchanged,
    so slicing after it is the same statement as the pipeline's ``Trim`` followed
    by ``Unnormalize``.
    """
    unnormalizer_type = _import_apxinf("apxinf.processors.normalize").Unnormalizer

    raw = np.asarray(raw_actions, dtype=np.float32)
    unnormalizer = unnormalizer_type(
        q01=np.asarray(norm_stats["actions"]["q01"], dtype=np.float32),
        q99=np.asarray(norm_stats["actions"]["q99"], dtype=np.float32),
    )
    width = unnormalizer.width if action_dim is None else int(action_dim)
    return unnormalizer(raw)[..., :width].astype(np.float32, copy=False)
