"""SigLIP vision tower, assembled.

Stage names match ``pi05_stage_probe.rs`` exactly --
``vision_patch_embed``, ``vision_layer_{i}``, ``vision_projected`` -- so the
probe can emit a signature at each boundary without a translation table.

The dtype flow is copied from the upstream port rather than chosen here, and it
is not uniform: the checkpoint keeps ``patch_embedding`` and
``position_embedding`` in float32 (``to_bfloat16_for_selected_params`` lists
them), so patch embedding and the position add run in float32 and the result is
narrowed to bfloat16 only on entry to the encoder. The encoder's LayerNorm
weights, by contrast, are *not* in that list and stay bfloat16. Running the
whole tower in one dtype would look tidier and would silently produce different
numbers from the checkpoint's own reference.
"""

from __future__ import annotations

from ...device import torch_module
from . import nn_ops

__all__ = ["VisionTower"]


class VisionTower:
    """The 27-layer SigLIP tower plus its projector.

    Weights are read from the export's own names under
    ``paligemma_with_expert.paligemma.model.vision_tower.vision_model``.
    """

    def __init__(self, weights, config, *, dtype=None) -> None:
        self.config = config
        self._w = weights
        self.dtype = dtype
        self._prefix = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"

    # -- weight access -----------------------------------------------------

    def _vision(self, suffix: str):
        return self._w.get(f"{self._prefix}.{suffix}")

    def _projector(self, suffix: str):
        return self._w.get(f"paligemma_with_expert.paligemma.model.multi_modal_projector.linear.{suffix}")

    # -- stages ------------------------------------------------------------

    def patch_embed(self, patches):
        """``vision_patch_embed``: patch projection plus position embedding.

        ``patches`` is ``[views * patches_per_view, 3 * patch_size ** 2]``, the
        same flat layout the engine's ``rgb_u8_to_patches_bf16_kernel`` and the
        probe use: per view, patches ordered row-major, each patch laid out
        ``[channel, dy, dx]``. The projection is expressed as a single GEMM
        against the ``[hidden, 3 * patch_size ** 2]`` reshape of conv weights,
        which is the engine's own decomposition.
        """
        torch = torch_module()
        config = self.config
        weight = self._vision("embeddings.patch_embedding.weight")
        bias = self._vision("embeddings.patch_embedding.bias")
        flat_weight = weight.reshape(weight.shape[0], -1)

        projected = nn_ops.linear(patches, flat_weight, bias)
        positions = self._vision("embeddings.position_embedding.weight")
        return projected + positions[None, :, :]

    def layer(self, index: int, hidden):
        """``vision_layer_{index}``: one SigLIP encoder layer.

        Self-attention here is bidirectional (``is_causal=False``) and carries no
        mask, which is why the engine's vision path calls ``attention.mha_bf16``
        rather than the language tower's MQA.
        """
        torch = torch_module()
        config = self.config
        prefix = f"encoder.layers.{index}"

        residual = hidden
        normed = nn_ops.layer_norm(
            hidden,
            self._vision(f"{prefix}.layer_norm1.weight"),
            self._vision(f"{prefix}.layer_norm1.bias"),
            config.layer_norm_eps,
        )

        query = nn_ops.linear(
            normed,
            self._vision(f"{prefix}.self_attn.q_proj.weight"),
            self._vision(f"{prefix}.self_attn.q_proj.bias"),
        )
        key = nn_ops.linear(
            normed,
            self._vision(f"{prefix}.self_attn.k_proj.weight"),
            self._vision(f"{prefix}.self_attn.k_proj.bias"),
        )
        value = nn_ops.linear(
            normed,
            self._vision(f"{prefix}.self_attn.v_proj.weight"),
            self._vision(f"{prefix}.self_attn.v_proj.bias"),
        )

        batch, seq, _ = normed.shape
        heads, head_dim = config.vision_heads, config.vision_head_dim
        query = query.view(batch, seq, heads, head_dim).transpose(1, 2)
        key = key.view(batch, seq, heads, head_dim).transpose(1, 2)
        value = value.view(batch, seq, heads, head_dim).transpose(1, 2)

        attended = nn_ops.eager_attention(
            query, key, value, None, head_dim**-0.5, num_kv_groups=1
        )
        attended = attended.reshape(batch, seq, heads * head_dim)
        hidden = residual + nn_ops.linear(
            attended,
            self._vision(f"{prefix}.self_attn.out_proj.weight"),
            self._vision(f"{prefix}.self_attn.out_proj.bias"),
        )

        residual = hidden
        normed = nn_ops.layer_norm(
            hidden,
            self._vision(f"{prefix}.layer_norm2.weight"),
            self._vision(f"{prefix}.layer_norm2.bias"),
            config.layer_norm_eps,
        )
        activated = nn_ops.gelu_tanh(
            nn_ops.linear(
                normed,
                self._vision(f"{prefix}.mlp.fc1.weight"),
                self._vision(f"{prefix}.mlp.fc1.bias"),
            )
        )
        projected = nn_ops.linear(
            activated,
            self._vision(f"{prefix}.mlp.fc2.weight"),
            self._vision(f"{prefix}.mlp.fc2.bias"),
        )
        return residual + projected

    def post_layernorm(self, hidden):
        return nn_ops.layer_norm(
            hidden,
            self._vision("post_layernorm.weight"),
            self._vision("post_layernorm.bias"),
            self.config.layer_norm_eps,
        )

    def project(self, hidden):
        """``vision_projected``: the multi-modal projector, a plain linear.

        The export carries only ``linear.weight``/``linear.bias`` -- there is no
        activation in the PyTorch port's projector, whatever the JAX variant's
        ``projector_hidden_act`` suggests.
        """
        return nn_ops.linear(
            hidden, self._projector("weight"), self._projector("bias")
        )

    def forward(self, patches, *, on_stage=None):
        """Run the tower, optionally reporting each intermediate as it is made.

        ``on_stage(name, tensor)`` is called for ``vision_patch_embed``,
        ``vision_layer_{i}`` and ``vision_projected``. The reference runtime has
        no hook machinery beyond this callback -- the probe passes one in.
        """
        torch = torch_module()
        config = self.config

        hidden = self.patch_embed(patches)
        if on_stage is not None:
            on_stage("vision_patch_embed", hidden)

        # The upstream port narrows here, on entry to the encoder, because the
        # encoder's first projection is bfloat16. `self.dtype` is that dtype
        # unless the run asked for float32, which widens weights with activations.
        hidden = hidden.to(self.dtype if self.dtype is not None else torch.bfloat16)
        for index in range(config.vision_depth):
            hidden = self.layer(index, hidden)
            if on_stage is not None:
                on_stage(f"vision_layer_{index}", hidden)

        hidden = self.post_layernorm(hidden)
        projected = self.project(hidden)
        if on_stage is not None:
            on_stage("vision_projected", projected)
        return projected
