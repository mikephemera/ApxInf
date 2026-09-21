"""The assembled Pi0.5 forward pass.

Shape of the computation, following the upstream port:

* ``embed_prefix`` runs the SigLIP tower, embeds the language tokens, and
  concatenates the two into the prefix sequence.
* ``prefix_forward`` pushes that sequence through the PaliGemma language tower and
  keeps every layer's K/V.
* ``denoise_step`` embeds the current action state and timestep, runs the Gemma
  action expert over the suffix while it attends to the stored prefix, and
  projects back to action space.
* ``sample_actions`` is the Euler loop that ties the two together.

Two conventions differ from the engine by construction. They are named here
because the stage probe will show them as small constant offsets rather than as
errors, and a reader who does not know about them will chase them:

* **Time embedding dtype.** The upstream reference computes the sinusoidal
  embedding in float64, narrows it to float32, and runs ``time_mlp`` in float32,
  so ``adarms_cond`` reaching adaRMS is float32 throughout. The engine stores the
  same embedding as bf16 (``upload_time_embeddings_bf16``).
* **Step times.** Each step's time is computed directly as
  ``flow_start_time * (1 - step / num_flow_steps)`` -- the expression the engine
  uses -- rather than accumulated with ``time += dt`` in float32 the way the
  upstream inference loop does. The two agree to about an ulp and then drift.
"""

from __future__ import annotations

from ...device import assert_tensor_device, torch_module
from . import nn_ops
from .gemma import ActionExpert, AttentionMask, LanguageTower, make_att_2d_masks
from .vision import VisionTower

__all__ = ["Pi05", "PrefixCache"]


def _activation_dtype(config) -> "object":
    torch = torch_module()
    return torch.bfloat16 if config.precision == "bfloat16" else torch.float32


class PrefixCache:
    """The language tower's K/V, one pair per layer."""

    __slots__ = ("keys", "values", "length")

    def __init__(self, keys, values) -> None:
        self.keys = keys
        self.values = values
        self.length = keys[0].shape[2]


def _vision_stage_names(config) -> list:
    return (
        ["vision_patch_embed"]
        + [f"vision_layer_{index}" for index in range(config.vision_depth)]
        + ["vision_projected"]
    )


class Pi05:
    """Assembled eager Pi0.5, driven from the same boundaries as the engine."""

    def __init__(self, weights, config, device, *, dtype=None) -> None:
        self.weights = weights
        self.config = config
        self.device = device
        self.dtype = dtype if dtype is not None else _activation_dtype(config)
        self.vision = VisionTower(weights, config, dtype=self.dtype)
        self.language = LanguageTower(weights, config, dtype=self.dtype)
        self.expert = ActionExpert(weights, config, dtype=self.dtype)

    # -- inputs ------------------------------------------------------------

    def pixels_from_patches(self, patches):
        """``[views * patches_per_view, 3 * patch ** 2]`` -> ``[views, 3, H, W]``.

        The inverse of the engine's ``rgb_u8_to_patches_bf16_kernel`` layout:
        within a view the patches are row-major and each patch is laid out
        ``[channel, dy, dx]``.
        """
        torch = torch_module()
        config = self.config
        side = config.image_size // config.patch_size
        patch = config.patch_size
        expected = (config.patch_tokens, config.patch_width)
        if tuple(patches.shape) != expected:
            raise ValueError(f"patches must have shape {expected}, got {tuple(patches.shape)}")
        per_view = patches.reshape(config.num_views, side, side, 3, patch, patch)
        return per_view.permute(0, 3, 1, 4, 2, 5).reshape(
            config.num_views, 3, config.image_size, config.image_size
        )

    def time_embedding(self, step: int):
        """Conditioning for flow step ``step``, float32 like the upstream port."""
        torch = torch_module()
        config = self.config
        time = torch.full(
            (1,),
            config.flow_start_time * (1.0 - step / config.num_flow_steps),
            dtype=torch.float32,
            device=self.device,
        )
        return nn_ops.sinusoidal_time_embedding(
            time,
            config.action_expert.width,
            config.time_min_period,
            config.time_max_period,
            self.device,
        ).to(torch.float32)

    def conditioning(self, time_value):
        """``elementwise`` -> ``time_mlp``: the adaRMS condition for a flow time."""
        return self.expert.time_mlp(time_value)

    # -- prefix ------------------------------------------------------------

    def embed_prefix(self, patches, token_ids, *, on_stage=None):
        """Vision tower plus token embeddings, concatenated along the sequence.

        Each view runs through the tower on its own, which is what the upstream
        port does: one image, one batch element. The engine instead batches all
        views into a single GEMM. Both are correct and they do not round
        identically, so the reference follows the upstream.
        """
        torch = torch_module()
        config = self.config

        # Every input crosses a device boundary here, and a tensor that arrived
        # on the wrong one would move silently inside the first matmul. Refuse
        # rather than let a run measure the host it happened to land on.
        assert_tensor_device(patches, self.device, what="the patch tensor")
        assert_tensor_device(token_ids, self.device, what="the token ids")

        per_view = {name: [] for name in _vision_stage_names(config)}
        for view in range(config.num_views):
            start = view * config.patches_per_view
            single = patches[start : start + config.patches_per_view]
            stages = {}

            def record(name, tensor, _stages=stages):
                _stages[name] = tensor

            self.vision.forward(single, on_stage=record)
            for name, tensor in stages.items():
                per_view[name].append(tensor)

        # The tower runs one view at a time, so each captured tensor carries a
        # leading batch axis; flattening it reproduces the engine's
        # `[views * patches_per_view, width]` layout exactly.
        stacked = {
            name: torch.cat(parts, dim=0).reshape(-1, parts[0].shape[-1])
            for name, parts in per_view.items()
        }
        if on_stage is not None:
            for name, tensor in stacked.items():
                on_stage(name, tensor)

        vision_tokens = stacked["vision_projected"].unsqueeze(0)
        language_tokens = self.language.embed_tokens(token_ids)
        return torch.cat([vision_tokens, language_tokens], dim=1)

    def prefix_forward(self, prefix_embeddings, *, on_stage=None):
        """Run the language tower over the prefix and return its K/V.

        The mask is the all-zero ``att_masks`` case: every prefix token attends to
        every earlier prefix token.
        """
        torch = torch_module()
        length = prefix_embeddings.shape[1]
        pad_masks = torch.ones(1, length, dtype=torch.bool, device=self.device)
        att_masks = torch.zeros(1, length, dtype=torch.bool, device=self.device)
        mask_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        _, keys, values = self.language.forward(
            prefix_embeddings, AttentionMask(mask_2d), position_ids
        )
        cache = PrefixCache(keys, values)
        if on_stage is not None:
            for layer in (0, self.config.language.depth - 1):
                on_stage(f"prefix_v_layer{layer}", cache.values[layer])
        return cache

    # -- denoising ---------------------------------------------------------

    def denoise_step(self, cache, state, time_value):
        """One flow-matching step: suffix through the expert, projected to actions."""
        torch = torch_module()
        config = self.config
        position_offset = cache.length

        assert_tensor_device(state, self.device, what="the action state")

        projected = self.expert.action_in_proj(state)
        hidden = projected.unsqueeze(0).to(self.dtype)

        horizon = state.shape[-2]
        pad_masks = torch.ones(1, horizon, dtype=torch.bool, device=self.device)
        att_masks = torch.zeros(1, horizon, dtype=torch.bool, device=self.device)
        att_masks[0, 0] = True  # the suffix does not attend across its own steps

        suffix_2d = make_att_2d_masks(pad_masks, att_masks)
        prefix_pad = torch.ones(1, position_offset, dtype=torch.bool, device=self.device)
        prefix_2d = prefix_pad[:, None, :].expand(1, horizon, position_offset)
        full_2d = torch.cat([prefix_2d, suffix_2d], dim=2)

        attention = AttentionMask(full_2d)

        prefix_offsets = torch.sum(prefix_pad, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(pad_masks, dim=1) - 1

        hidden = self.expert.forward(
            hidden, attention, position_ids, self.conditioning(time_value), cache.keys, cache.values
        )
        hidden = hidden[:, -config.action_horizon :].to(torch.float32)
        return self.expert.action_out_proj(hidden).squeeze(0)

    def sample_actions(self, cache, noise, *, on_stage=None):
        """The Euler loop: ten steps from the flow start time down to zero."""
        config = self.config
        dt = -config.flow_start_time / config.num_flow_steps
        state = noise
        for step in range(config.num_flow_steps):
            velocity = self.denoise_step(cache, state, self.time_embedding(step))
            state = nn_ops.euler_update(state, velocity, dt)
            if on_stage is not None:
                on_stage(f"denoise_step_{step}", state)
        return state
