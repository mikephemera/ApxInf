"""The upstream snapshot behind the same interface as the assembled runtime.

This exists for one job: proving the assembled implementation is the same model.
It runs once, against the assembled version, on identical inputs; after that the
assembled path is what produces reference numbers and this one stays as the
arbitration anchor.

The snapshot is driven through its own ``PI0Pytorch``, so what is compared is the
assembled code and not a second transcription of the upstream logic.

One upstream default is deliberately overridden. ``transformers`` gives every
tower ``sdpa`` attention, and the upstream inference path pins only the language
tower and the action expert to ``eager`` -- the vision tower keeps SDPA, whose
numerics depend on which device and which backend it lands on. The reference
fixes attention to ``eager`` everywhere so a stage signature means the same thing
on CPU, CUDA and MUSA; ``pin_vision_attention`` applies the same to the snapshot
so the two are comparable.
"""

from __future__ import annotations

from pathlib import Path

from ..device import torch_module

__all__ = ["VendorPi05"]

_VISION_STAGES = ("vision_patch_embed", "vision_layer_{index}", "vision_projected")


class VendorPi05:
    """Adapter presenting the upstream snapshot the way ``probe.run`` expects."""

    def __init__(
        self,
        weights_path,
        config,
        device,
        *,
        pin_vision_attention: bool = True,
        load_weights: bool = True,
    ) -> None:
        torch = torch_module()
        from ..vendor.pi05.transformers_overlay import install_transformers_overlay

        install_transformers_overlay()

        from ..vendor.pi05.pi0_config import Pi0Config
        from ..vendor.pi05.pi0_pytorch import PI0Pytorch

        self.config = config
        self.device = device

        self.model = PI0Pytorch(
            Pi0Config(
                pi05=True,
                action_dim=config.action_dim,
                action_horizon=config.action_horizon,
                max_token_len=config.max_token_len,
                # The upstream compile path is irrelevant to a reference run and
                # would make the first call into a tracing pass.
                pytorch_compile_mode=None,
            )
        )
        if load_weights:
            self._load(weights_path)
        self.model.to(device)
        self.model.eval()

        if pin_vision_attention:
            vision = self.model.paligemma_with_expert.paligemma.model.vision_tower.vision_model
            vision.config._attn_implementation = "eager"
            for layer in vision.encoder.layers:
                layer.self_attn.config._attn_implementation = "eager"

    def _load(self, weights_path) -> None:
        from safetensors.torch import load_file

        root = Path(weights_path)
        source = root / "model.safetensors" if root.is_dir() else root
        state = load_file(str(source))
        # The checkpoint's own names already match PI0Pytorch's parameter tree;
        # only the tied embedding alias has to be supplied.
        state.setdefault(
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
            state["paligemma_with_expert.paligemma.lm_head.weight"],
        )
        self.model.load_state_dict(state, strict=False)

    # -- inputs ------------------------------------------------------------

    def _pixels_from_patches(self, patches):
        config = self.config
        side = config.image_size // config.patch_size
        patch = config.patch_size
        per_view = patches.reshape(config.num_views, side, side, 3, patch, patch)
        return per_view.permute(0, 3, 1, 4, 2, 5).reshape(
            config.num_views, 3, config.image_size, config.image_size
        )

    def time_embedding(self, step: int, *, accumulate: bool = False):
        """The flow time for a step.

        ``accumulate`` reproduces the upstream inference loop's ``time += dt`` in
        float32 rather than recomputing the value directly; the two drift apart by
        an ulp or so, and the comparison needs to be able to say which was used.
        """
        torch = torch_module()
        config = self.config
        dt = -config.flow_start_time / config.num_flow_steps
        if accumulate:
            time = torch.tensor(config.flow_start_time, dtype=torch.float32, device=self.device)
            for _ in range(step):
                time = time + dt
            return time
        return torch.tensor(
            config.flow_start_time * (1.0 - step / config.num_flow_steps),
            dtype=torch.float32,
            device=self.device,
        )

    # -- prefix ------------------------------------------------------------

    def embed_prefix(self, patches, token_ids, *, on_stage=None):
        torch = torch_module()
        config = self.config
        pi0 = self.model
        vision = pi0.paligemma_with_expert.paligemma.model.vision_tower.vision_model
        projector = pi0.paligemma_with_expert.paligemma.model.multi_modal_projector

        pixels = self._pixels_from_patches(patches).to(self.device)
        names = (
            ["vision_patch_embed"]
            + [f"vision_layer_{index}" for index in range(config.vision_depth)]
            + ["vision_projected"]
        )
        per_view = {name: [] for name in names}

        with torch.no_grad():
            for view in range(config.num_views):
                single = pixels[view : view + 1]
                hidden = vision.embeddings(single.to(dtype=torch.float32))
                per_view["vision_patch_embed"].append(hidden)
                hidden = hidden.to(torch.bfloat16)
                for index, layer in enumerate(vision.encoder.layers):
                    hidden = layer(hidden, None, output_attentions=False)[0]
                    per_view[f"vision_layer_{index}"].append(hidden)
                per_view["vision_projected"].append(projector(vision.post_layernorm(hidden)))

            if on_stage is not None:
                for name, parts in per_view.items():
                    # View-major flat order, the engine's `[views * patches, width]`.
                    on_stage(name, torch.cat(parts, dim=0).reshape(-1, parts[0].shape[-1]))

            # Along the sequence axis: every view of one image, then the tokens.
            vision_tokens = torch.cat(per_view["vision_projected"], dim=1)
            language_tokens = pi0.paligemma_with_expert.embed_language_tokens(
                token_ids.to(self.device).unsqueeze(0)
            )
            language_tokens = language_tokens * (language_tokens.shape[-1] ** 0.5)
            return torch.cat([vision_tokens, language_tokens], dim=1)

    def prefix_forward(self, prefix_embeddings, *, on_stage=None):
        torch = torch_module()
        from ..vendor.pi05.pi0_pytorch import make_att_2d_masks

        pi0 = self.model
        length = prefix_embeddings.shape[1]
        pad_masks = torch.ones(1, length, dtype=torch.bool, device=self.device)
        att_masks = torch.zeros(1, length, dtype=torch.bool, device=self.device)
        mask_4d = pi0._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        language_model = pi0.paligemma_with_expert.paligemma.language_model
        language_model.config._attn_implementation = "eager"
        with torch.no_grad():
            output = language_model.forward(
                inputs_embeds=prefix_embeddings,
                attention_mask=mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=True,
            )
        cache = _VendorCache(output.past_key_values, pi0, self.device)
        if on_stage is not None:
            for layer in (0, self.config.language.depth - 1):
                on_stage(f"prefix_v_layer{layer}", cache.value(layer))
        return cache

    # -- denoising ---------------------------------------------------------

    def denoise_step(self, cache, state, time_value):
        torch = torch_module()
        pi0 = self.model
        pi0.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        # PI0.5 does not feed proprioception to the suffix, so the snapshot's
        # `state` argument is inert; `x_t` is the action state being denoised.
        unused_state = torch.zeros(
            1, self.config.action_dim, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            velocity = pi0.denoise_step(
                unused_state,
                cache.pad_masks,
                cache.past_key_values,
                # The snapshot reads `x_t` as [batch, horizon, dim].
                state.unsqueeze(0),
                time_value.reshape(1),
            )
        return velocity.squeeze(0)

    def sample_actions(self, cache, noise, *, on_stage=None):
        """The Euler loop, with times taken from ``time_embedding``."""
        torch = torch_module()
        config = self.config
        dt = -config.flow_start_time / config.num_flow_steps
        state = noise.to(self.device)
        for step in range(config.num_flow_steps):
            velocity = self.denoise_step(cache, state, self.time_embedding(step))
            state = state + dt * velocity
            if on_stage is not None:
                on_stage(f"denoise_step_{step}", state)
        return state


class _VendorCache:
    """Thin view over the snapshot's ``DynamicCache``."""

    __slots__ = ("past_key_values", "pad_masks", "_device", "_length")

    def __init__(self, past_key_values, pi0, device) -> None:
        torch = torch_module()
        self.past_key_values = past_key_values
        self._device = device
        self._length = int(past_key_values[0][0].shape[2])
        self.pad_masks = torch.ones(1, self._length, dtype=torch.bool, device=device)

    @property
    def length(self) -> int:
        return self._length

    def value(self, layer: int):
        return self.past_key_values[layer][1]

    def key(self, layer: int):
        return self.past_key_values[layer][0]
