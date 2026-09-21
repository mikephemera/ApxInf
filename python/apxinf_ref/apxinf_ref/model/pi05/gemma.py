"""The two Gemma towers: PaliGemma's language model and PI0.5's action expert.

Both are ordinary Gemma decoders whose attention reads a shared K/V cache; the
differences are that the language tower uses a plain RMSNorm while the action
expert uses adaRMS (``cond`` supplies scale, shift and gate through a three-way
projection), and that the language tower's last layer is deliberately truncated.

That truncation is the one piece of non-obvious structure here. The engine
passes ``compute_tail = index + 1 < depth``
(``bf16_runtime.rs:291``, ``runtime.rs:397``, ``int8_runtime.rs:247``) and the
executor, when it is false, returns the layer's *input* unchanged alongside the
K/V it just wrote (``bf16_executor.rs:47``, ``fp8_executor.rs:109``) -- no
attention, no output projection, no MLP. It saves work without changing the
result, because layer 17's hidden state is never read; only its K/V is. The
reference reproduces it so that a per-layer comparison sees structure rather
than a fictitious disagreement.
"""

from __future__ import annotations

from ...device import torch_module
from . import nn_ops

__all__ = ["AttentionMask", "ActionExpert", "LanguageTower", "make_att_2d_masks"]

#: ``pi0_pytorch.py:_prepare_attention_masks_4d`` -- the additive mask value.
MASKED_ATTENTION_BIAS = -2.3819763e38

_ROOT = "paligemma_with_expert"


def make_att_2d_masks(pad_masks, att_masks):
    """Prefix-LM mask construction, copied from big_vision via the upstream port.

    Tokens attend to valid input tokens whose cumulative ``att_masks`` value is
    less than or equal to their own, intersected with the padding mask.
    """
    torch = torch_module()
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def prepare_attention_masks_4d(att_2d_masks):
    torch = torch_module()
    return torch.where(att_2d_masks[:, None, :, :], 0.0, MASKED_ATTENTION_BIAS)


class AttentionMask:
    """The additive 4-D mask handed to every layer of one tower pass."""

    def __init__(self, mask_2d) -> None:
        self.mask_2d = mask_2d
        self.mask_4d = prepare_attention_masks_4d(mask_2d)

    __slots__ = ("mask_2d", "mask_4d")


class _GemmaLayer:
    """Shared decoder-layer body; ``adaptive`` selects the normalisation."""

    def __init__(self, weights, weights_prefix: str, config, *, adaptive: bool) -> None:
        self.weights = weights
        self.prefix = weights_prefix
        self.config = config
        self.adaptive = adaptive

    def _w(self, suffix: str):
        return self.weights.get(f"{self.prefix}.{suffix}")

    def norm(self, hidden, cond, eps, which: str):
        """``which`` is ``input_layernorm`` or ``post_attention_layernorm``."""
        if self.adaptive:
            return nn_ops.adaptive_rms_norm(
                hidden,
                self._w(f"{which}.dense.weight"),
                self._w(f"{which}.dense.bias"),
                cond,
                eps,
            )
        return nn_ops.rms_norm(hidden, self._w(f"{which}.weight"), eps), None

    def project_qkv(self, hidden):
        torch = torch_module()
        config = self.config
        batch, seq, _ = hidden.shape

        query = nn_ops.linear(hidden, self._w("self_attn.q_proj.weight"))
        key = nn_ops.linear(hidden, self._w("self_attn.k_proj.weight"))
        value = nn_ops.linear(hidden, self._w("self_attn.v_proj.weight"))

        query = query.view(batch, seq, config.num_heads, config.head_dim).transpose(1, 2)
        key = key.view(batch, seq, config.num_kv_heads, config.head_dim).transpose(1, 2)
        value = value.view(batch, seq, config.num_kv_heads, config.head_dim).transpose(1, 2)
        return query, key, value

    def attend(self, query, key, value, attention: AttentionMask, cos, sin, *, prefix_kv=None):
        """Rotate, optionally extend the cache, then attend.

        With ``prefix_kv`` the layer is an action-expert layer attending over the
        stored prefix plus its own suffix; ``use_cache`` is false there, so the
        cache is concatenated rather than extended in place.
        """
        torch = torch_module()
        query, key = nn_ops.apply_rotary(query, key, cos, sin)
        if prefix_kv is not None:
            cached_key, cached_value = prefix_kv
            key = torch.cat([cached_key, key], dim=2)
            value = torch.cat([cached_value, value], dim=2)
        attended = nn_ops.eager_attention(
            query,
            key,
            value,
            attention.mask_4d,
            self.config.head_dim**-0.5,
            num_kv_groups=self.config.num_heads // self.config.num_kv_heads,
        )
        batch, seq = query.shape[0], query.shape[2]
        return attended.reshape(batch, seq, self.config.num_heads * self.config.head_dim)

    def project_output(self, attended):
        return nn_ops.linear(attended, self._w("self_attn.o_proj.weight"))

    def mlp(self, hidden):
        gate_up = nn_ops.linear(hidden, self._w("mlp.gate_proj.weight"))
        up = nn_ops.linear(hidden, self._w("mlp.up_proj.weight"))
        hidden = nn_ops.gelu_tanh(gate_up) * up
        return nn_ops.linear(hidden, self._w("mlp.down_proj.weight"))


class LanguageTower:
    """PaliGemma's Gemma-2B language model, used only to produce the prefix K/V."""

    def __init__(self, weights, config, *, dtype=None) -> None:
        self.weights = weights
        self.config = config
        self.dtype = dtype
        self.layers = [
            _GemmaLayer(
                weights,
                f"{_ROOT}.paligemma.model.language_model.layers.{index}",
                config.language,
                adaptive=False,
            )
            for index in range(config.language.depth)
        ]
        self._final_norm = f"{_ROOT}.paligemma.model.language_model.norm"

    def embed_tokens(self, token_ids):
        """``embedding.lookup``: token embedding, scaled by ``sqrt(width)``.

        ``embed_tokens.weight`` is tied to ``lm_head.weight`` in the export's
        metadata rather than stored twice; ``weights.py`` materialises the alias.
        """
        torch = torch_module()
        table = self.weights.get(f"{_ROOT}.paligemma.model.language_model.embed_tokens.weight")
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        embedded = torch.nn.functional.embedding(token_ids, table)
        return embedded * (embedded.shape[-1] ** 0.5)

    def forward(self, hidden, attention: AttentionMask, position_ids, *, compute_tail_per_layer=True):
        """Run every language layer, returning the cache and the final hidden.

        ``compute_tail_per_layer`` mirrors the engine's ``index + 1 < depth``:
        when it is on, the last layer is truncated to a K/V write.
        """
        torch = torch_module()
        config = self.config.language
        cos, sin = nn_ops.rotary_cos_sin(position_ids, config.head_dim, self.config.rope_theta, hidden.dtype)

        keys, values = [], []
        depth = config.depth
        for index, layer in enumerate(self.layers):
            # The engine's rule is `index + 1 < depth`; disabling the shortcut
            # computes every layer in full.
            compute_tail = (index + 1 < depth) or not compute_tail_per_layer
            normed, _ = layer.norm(hidden, None, self.config.rms_norm_eps, "input_layernorm")
            query, key, value = layer.project_qkv(normed)
            query, key = nn_ops.apply_rotary(query, key, cos, sin)
            keys.append(key)
            values.append(value)

            if not compute_tail:
                # ``bf16_executor.rs:47`` -- the input passes through untouched.
                continue

            attended = nn_ops.eager_attention(
                query,
                key,
                value,
                attention.mask_4d,
                config.head_dim**-0.5,
                num_kv_groups=config.num_heads // config.num_kv_heads,
            )
            batch, seq = query.shape[0], query.shape[2]
            attended = attended.reshape(batch, seq, config.num_heads * config.head_dim)
            hidden = hidden + layer.project_output(attended)
            residual = hidden
            normed, _ = layer.norm(
                hidden, None, self.config.rms_norm_eps, "post_attention_layernorm"
            )
            hidden = residual + layer.mlp(normed)

        hidden = nn_ops.rms_norm(
            hidden, self.weights.get(f"{self._final_norm}.weight"), self.config.rms_norm_eps
        )
        return hidden, keys, values


class ActionExpert:
    """The Gemma-300M action expert: adaRMS layers attending over the prefix."""

    def __init__(self, weights, config, *, dtype=None) -> None:
        self.weights = weights
        self.config = config
        self.dtype = dtype
        self.layers = [
            _GemmaLayer(
                weights,
                f"{_ROOT}.gemma_expert.model.layers.{index}",
                config.action_expert,
                adaptive=True,
            )
            for index in range(config.action_expert.depth)
        ]
        self._final_norm = f"{_ROOT}.gemma_expert.model.norm"

    def action_in_proj(self, noisy_actions):
        return nn_ops.linear(
            noisy_actions,
            self.weights.get("action_in_proj.weight"),
            self.weights.get("action_in_proj.bias"),
        )

    def action_out_proj(self, hidden):
        return nn_ops.linear(
            hidden,
            self.weights.get("action_out_proj.weight"),
            self.weights.get("action_out_proj.bias"),
        )

    def time_mlp(self, time_embedding):
        """``time_mlp_in -> silu -> time_mlp_out -> silu``: the adaRMS condition.

        Note the second SiLU: the upstream port applies it to the module output,
        not just between the two projections.
        """
        torch = torch_module()
        hidden = nn_ops.linear(
            time_embedding,
            self.weights.get("time_mlp_in.weight"),
            self.weights.get("time_mlp_in.bias"),
        )
        hidden = torch.nn.functional.silu(hidden)
        hidden = nn_ops.linear(
            hidden,
            self.weights.get("time_mlp_out.weight"),
            self.weights.get("time_mlp_out.bias"),
        )
        return torch.nn.functional.silu(hidden)

    def forward(self, hidden, attention: AttentionMask, position_ids, cond, prefix_keys, prefix_values):
        """Run the expert over the suffix, attending to the prefix and itself."""
        torch = torch_module()
        config = self.config.action_expert
        cos, sin = nn_ops.rotary_cos_sin(position_ids, config.head_dim, self.config.rope_theta, hidden.dtype)

        for index, layer in enumerate(self.layers):
            residual = hidden
            normed, gate = layer.norm(
                hidden, cond, self.config.rms_norm_eps, "input_layernorm"
            )
            query, key, value = layer.project_qkv(normed)
            attended = layer.attend(
                query,
                key,
                value,
                attention,
                cos,
                sin,
                prefix_kv=(prefix_keys[index], prefix_values[index]),
            )
            hidden = residual + layer.project_output(attended) * gate

            residual = hidden
            normed, gate = layer.norm(
                hidden, cond, self.config.rms_norm_eps, "post_attention_layernorm"
            )
            hidden = residual + layer.mlp(normed) * gate

        return nn_ops.adaptive_rms_norm(
            hidden,
            self.weights.get(f"{self._final_norm}.dense.weight"),
            self.weights.get(f"{self._final_norm}.dense.bias"),
            cond,
            self.config.rms_norm_eps,
        )[0]
