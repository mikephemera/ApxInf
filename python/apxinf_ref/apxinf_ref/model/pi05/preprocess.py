"""Observation to model boundary: images, proprioception, and the noise draw.

Three conventions are pinned here rather than left to the caller, because each
one silently changes the numbers if it is chosen differently, and a stage
comparison between two runs that chose differently measures the choice:

**Resize.** `python/apxinf` ships two letterbox resizes and documents that they
disagree by up to ~52/255 on random content at 256 -> 224 — enough to change
which FAST action tokens pi0-FAST emits (`processors/resize.py:8-31`). PI0.5 uses
the PIL one (`ResizeWithPad`), which is what OpenPI's own preprocessing uses and
what `Pi05Policy.default_pipelines` selects. This module reuses that class
directly rather than reimplementing it, so the two cannot drift.

**Camera orientation and proprioception.** LIBERO's frames come back rotated 180
degrees relative to what the policy expects and its gripper is reported twice
with mirrored signs. Both conversions live in `scripts/libero_observation.py`
and are reused here for the same reason. How many of those joints the state
carries is *not* a convention, though: it is the width of the checkpoint's own
normalization statistics, and `finger_joints_for_state_width` reads it from
there.

**Noise.** The engine draws `[horizon, action_dim]` standard normal values and
the two implementations must see the *same* ones. Round-tripping through
bfloat16 before freezing is what makes that true: the noise is an input to the
model, and a value that differs in the last bit of bfloat16 is a different
input, not a rounding difference in the output.

The module imports `apxinf` and `scripts.libero_observation` lazily, so that the
stage-probe path — which takes patches and token ids directly — keeps working
with neither on the path.
"""

from __future__ import annotations

from ...device import torch_module

__all__ = [
    "PIL_RESIZE",
    "finger_joints_for_state_width",
    "libero_state_vector",
    "normalize_state",
    "normalized_pixels",
    "orient_libero_images",
    "patches_from_pixels",
    "patches_from_rgb_u8",
    "sample_noise",
    "try_import_apxinf_processors",
    "try_import_libero_observation",
]

#: Name the resize path this runtime is pinned to, for error messages and docs.
PIL_RESIZE = "apxinf.processors.ResizeWithPad (PIL BILINEAR, antialiased)"


def try_import_apxinf_processors():
    """Import `apxinf.processors`, or say exactly what is missing."""
    try:
        import apxinf.processors as processors  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - depends on the host
        raise ImportError(
            "the observation path needs the `apxinf` Python package for its "
            "pinned resize and prompt conventions "
            "(it lives at python/apxinf in this repository; add it to "
            "PYTHONPATH or `pip install -e python/apxinf`). The stage-probe "
            "path does not need it."
        ) from error
    return processors


def try_import_libero_observation():
    """Import `scripts.libero_observation`, or say exactly what is missing."""
    try:
        from scripts import libero_observation  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - depends on the host
        raise ImportError(
            "LIBERO frame orientation and proprioception live in "
            "scripts/libero_observation.py; run from the repository root, or "
            "add it to PYTHONPATH."
        ) from error
    return libero_observation


def orient_libero_images(base, wrist):
    """Rotate LIBERO's camera frames into the policy's orientation.

    The conversion is a 180-degree rotation (both axes flipped), not a
    transpose and not a single-axis flip; `libero_observation.py:36-40` owns it
    and is reused rather than restated.
    """
    return try_import_libero_observation().libero_images(base, wrist)


def finger_joints_for_state_width(state_width: int) -> int:
    """How many of LIBERO's mirrored finger joints a state vector of this width carries.

    The width is a *checkpoint* property -- it is the width of the checkpoint's own
    ``state`` normalization statistics -- so it is read from there rather than
    assumed. `scripts/eval_libero.py` already owns this rule for the live
    simulator path and is imported rather than restated, the same way the resize
    and the prompt builder are.
    """
    try:
        from scripts.eval_libero import state_finger_joints  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - depends on the host
        raise ImportError(
            "the state-width rule lives in scripts/eval_libero.py; run from the "
            "repository root, or add it to PYTHONPATH."
        ) from error
    return state_finger_joints({"state_dim": int(state_width)})


def libero_state_vector(observation, *, state_width=None, finger_joints=None):
    """LIBERO proprioception as the vector a PI0.5 checkpoint consumes.

    The width is not a detail and not a free choice. The checkpoint's LIBERO
    statistics carry ``state`` at width 8 and ``actions`` at width 7
    (``assets/physical-intelligence/libero/norm_stats.json``), and the state
    reaches the model only through the discretised prompt. Getting it wrong fails
    in one of two ways depending on which statistics are in hand, and only one of
    them is loud: against this checkpoint's 8-wide ``state`` statistics a
    collapsed 7-value vector raises a broadcast error, but against 7-wide
    statistics it would normalise cleanly into a different number of prompt bins,
    produce different token ids, and change the whole rollout with no error at
    all. Pass the width of the statistics you are about to normalise against
    (``state_width``) and the joint count follows; pass ``finger_joints`` only to
    override it deliberately.

    The vector is ``eef_pos(3) + axis_angle(3) + gripper(finger_joints)``; the
    joint values are taken as they are rather than summed or averaged, which would
    be a plausible-looking bug that shifts the gripper coordinate.
    """
    if finger_joints is None:
        if state_width is None:
            raise ValueError(
                "libero_state_vector needs either state_width (the width of the "
                "checkpoint's `state` normalization statistics) or an explicit "
                "finger_joints; guessing would produce a prompt the checkpoint was "
                "not trained on."
            )
        finger_joints = finger_joints_for_state_width(state_width)
    return try_import_libero_observation().libero_state(
        observation, finger_joints=finger_joints
    )


def normalize_state(state, norm_stats, *, use_quantiles: bool = True):
    """Normalise proprioception with the checkpoint's own statistics.

    PI0.5 conditions only through the discretised prompt, so this matters for
    the prompt string rather than for a tensor the model consumes directly --
    which is why an out-of-range value must reach `discretize_state` rather than
    being clamped here.
    """
    import numpy as np

    values = np.asarray(state, dtype=np.float32)
    if use_quantiles:
        low = np.asarray(norm_stats["q01"], dtype=np.float32)
        high = np.asarray(norm_stats["q99"], dtype=np.float32)
    else:
        low = np.asarray(norm_stats["mean"], dtype=np.float32)
        high = np.asarray(norm_stats["std"], dtype=np.float32)
    return ((values - low) / (high - low + 1e-6) * 2.0 - 1.0).astype(np.float32)


def resize_with_pad(images, size: int):
    """The pinned letterbox resize. See the module docstring."""
    processors = try_import_apxinf_processors()
    return processors.ResizeWithPad(int(size))(images)


def normalized_pixels(rgb, *, image_size: int):
    """``[views, H, W, 3]`` uint8 (or float in [-1, 1]) -> ``[views, 3, H, W]`` float32.

    The normalization is the engine's: ``(value / 255) * 2 - 1``. It is exact in
    float32 for every uint8 value, so the two implementations cannot disagree
    here for a reason that matters.

    This is the boundary the model actually consumes, which is why it is a
    function of its own rather than three lines inside the patch conversion: the
    normalized image is also what a capture keeps alongside the raw frames, so
    that a divergence can be localised to the resize or to the patch layout
    instead of being attributed to "preprocessing" as a whole.
    """
    import numpy as np

    array = np.asarray(rgb)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected [views, H, W, 3], got {array.shape}")
    if array.shape[1] != image_size or array.shape[2] != image_size:
        raise ValueError(
            f"expected {image_size}x{image_size} frames, got "
            f"{array.shape[1]}x{array.shape[2]}; resize first "
            f"({PIL_RESIZE})"
        )

    if array.dtype == np.uint8:
        normalized = array.astype(np.float32) / 255.0 * 2.0 - 1.0
    else:
        normalized = array.astype(np.float32)

    # NHWC -> NCHW, the layout both the patch conversion and the model use.
    return np.ascontiguousarray(normalized.transpose(0, 3, 1, 2))


def patches_from_pixels(pixels, *, patch_size: int = 14):
    """``[views, 3, H, W]`` -> ``[views * patches_per_view, 3 * patch ** 2]``.

    The engine's `rgb_u8_to_patches_bf16_kernel` layout, which
    `Pi05.pixels_from_patches` inverts: per view the patches are row-major over
    ``(patch_y, patch_x)``, and within a patch the values run
    ``(channel, dy, dx)``. Both directions are exercised against each other by
    the package tests, because this is the one boundary where a transposed axis
    produces output that still looks like a plausible image.

    ``patch_size`` is taken as an argument and checked against the frame rather
    than assumed, and every caller takes it from the same ``config.patch_size``
    the model uses -- which is what keeps the two halves of the round trip from
    being told different sizes. A hard-coded 14 would not fail loudly on a
    checkpoint whose patch is a different size: 224 divides by 14 and by 16
    alike, so the grid would simply come out the wrong shape.
    """
    torch = torch_module()
    views, channels, height, width = pixels.shape
    if channels != 3:
        raise ValueError(f"expected 3 channels, got {channels}")
    if height != width:
        raise ValueError(f"expected square images, got {height}x{width}")

    patch = int(patch_size)
    if patch <= 0 or height % patch != 0:
        raise ValueError(f"image size {height} is not a multiple of the patch size {patch}")
    side = height // patch
    grid = pixels.reshape(views, 3, side, patch, side, patch)
    grid = grid.permute(0, 2, 4, 1, 3, 5)
    return grid.reshape(views * side * side, channels * patch * patch)


def patches_from_rgb_u8(rgb, *, image_size: int, patch_size: int = 14):
    """``[views, H, W, 3]`` uint8 (or float in [-1, 1]) -> engine patch layout.

    The two steps it composes, each of which has exactly one implementation:
    :func:`normalized_pixels` for the engine's normalization, and
    :func:`patches_from_pixels` for the layout.
    """
    torch = torch_module()

    pixels = normalized_pixels(rgb, image_size=image_size)
    return patches_from_pixels(torch.from_numpy(pixels), patch_size=patch_size)


def sample_noise(shape, device, *, seed: int, dtype="bfloat16"):
    """A noise draw the other implementation can be made to see identically.

    The draw is rounded through bfloat16 and back before it is returned. The
    engine receives bf16 noise, so a float32 draw would hand the two
    implementations inputs that differ in the last bit of bfloat16 -- and a
    different input is not a rounding difference in the output, it is a
    different question.
    """
    torch = torch_module()
    target = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    return values.to(target).to(torch.float32).to(device)
