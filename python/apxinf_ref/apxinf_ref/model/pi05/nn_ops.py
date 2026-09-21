"""Eager torch primitives, named to line up with ApxInf's safe CUDA kernels.

These are ordinary functions, not a plug-in layer: there is no registry, no
candidate list and no native slot. Their only organising idea is that each one
should be recognisable as the reference semantics of exactly one operation the
engine calls, so that reading this file beside
``crates/apxinf-model/src/pi05/model/blocks/{bf16,fp8_static,int8_dynamic}.rs``
shows which
number on the engine side corresponds to which expression here.

The names follow the engine's ``kernels::{norm,gemm,rope,attention,fused,
activation,embedding,elementwise}`` grouping. The dtypes are *not* forced:
callers pass whatever the checkpoint and the surrounding graph imply, because
the upstream port's ``to_bfloat16_for_selected_params`` leaves a specific set of
parameters in float32 and torch's promotion rules are part of the observed
numerics.
"""

from __future__ import annotations

import math

from ...device import torch_module

__all__ = [
    "adaptive_rms_norm",
    "apply_rotary",
    "eager_attention",
    "euler_update",
    "gelu_tanh",
    "geglu",
    "layer_norm",
    "linear",
    "repeat_kv",
    "rms_norm",
    "rotary_cos_sin",
    "sinusoidal_time_embedding",
]


def linear(x, weight, bias=None):
    """``gemm``: ``x @ weight.T + bias``, the row-major layout the export uses."""
    torch = torch_module()
    return torch.nn.functional.linear(x, weight, bias)


def rms_norm(x, weight, eps: float):
    """``norm.rms``: Gemma RMSNorm, including the ``1 + weight`` convention.

    The upstream order matters and is reproduced exactly: the variance is
    accumulated in float32, the reciprocal square root is applied while ``x``
    is still in its own dtype (so the product is float32), the scale is applied
    in float32, and only then is the result narrowed back.
    """
    torch = torch_module()
    dtype = x.dtype
    variance = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    normed = normed * (1.0 + weight.float())
    return normed.to(dtype)


def adaptive_rms_norm(x, dense_weight, dense_bias, cond, eps: float):
    """``norm.adaptive_rms``: adaRMS, returning ``(normed, gate)``.

    The action expert has no RMSNorm scale of its own -- ``cond`` supplies
    scale, shift and gate through a three-way projection. Note that the projection
    runs at float32 in the upstream port (its weights are in the
    keep-float32 set), so ``cond`` is promoted; that promotion is part of the
    reference numerics, not an implementation detail.
    """
    torch = torch_module()
    dtype = x.dtype
    variance = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)

    modulation = torch.nn.functional.linear(cond, dense_weight, dense_bias)
    if len(x.shape) == 3:  # [batch, seq, features]
        modulation = modulation.unsqueeze(1)
    scale, shift, gate = torch.chunk(modulation, 3, dim=-1)

    normed = normed * (1 + scale.to(torch.float32)) + shift.to(torch.float32)
    return normed.to(dtype), gate.to(dtype)


def layer_norm(x, weight, bias, eps: float):
    """``norm.layer``: SigLIP's LayerNorm."""
    torch = torch_module()
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def gelu_tanh(x):
    """``activation.bias_gelu`` / ``activation.geglu``: the tanh GELU variant.

    ``gelu_pytorch_tanh`` is what both towers configure, and transformers maps it
    to ``F.gelu(..., approximate="tanh")``.
    """
    torch = torch_module()
    return torch.nn.functional.gelu(x, approximate="tanh")


def geglu(gate_up, intermediate_size: int):
    """``gemm.bf16_geglu_fused`` / ``activation.geglu``: split, then ``gelu(g) * u``.

    ``gate_up`` is the packed ``[.., 2 * intermediate_size]`` projection with the
    gate half first, which is the order the export's ``gate_proj``/``up_proj``
    pair concatenates to.
    """
    gate, up = gate_up.split(intermediate_size, dim=-1)
    return gelu_tanh(gate) * up


def repeat_kv(hidden_states, n_rep: int):
    """``attention.mqa``: broadcast one KV head across the query heads."""
    torch = torch_module()
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return expanded.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def eager_attention(query, key, value, attention_mask, scaling: float, num_kv_groups: int):
    """``attention.mqa`` / ``attention.mha``: the eager attention both towers use.

    Softmax is taken in float32 and narrowed back to the query dtype before the
    value projection -- cheap to get wrong and impossible to see in the final
    output.
    """
    torch = torch_module()
    key_states = repeat_kv(key, num_kv_groups)
    value_states = repeat_kv(value, num_kv_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous()


def rotary_cos_sin(position_ids, head_dim: int, theta: float, dtype):
    """``rope.split_qkv_apply`` / ``rope.apply_q_write_kv``: the cos/sin tables.

    Frequencies are built and applied in float32 (transformers wraps the
    computation in ``autocast(enabled=False)``) and only then narrowed to the
    activation dtype.
    """
    torch = torch_module()
    inverse_frequency = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(position_ids.device)
            / head_dim
        )
    )
    inverse_frequency_expanded = (
        inverse_frequency[None, :, None]
        .float()
        .expand(position_ids.shape[0], -1, 1)
        .to(position_ids.device)
    )
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inverse_frequency_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    # `attention_scaling` is 1.0 for the default RoPE type.
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    torch = torch_module()
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(query, key, cos, sin, unsqueeze_dim: int = 1):
    """``rope.split_qkv_apply``: rotate query and key in place of a cache write."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    query_embed = (query * cos) + (_rotate_half(query) * sin)
    key_embed = (key * cos) + (_rotate_half(key) * sin)
    return query_embed, key_embed


def sinusoidal_time_embedding(time, dimension: int, min_period: float, max_period: float, device):
    """``elementwise`` time embedding, matching ``math.rs`` ``sinusoidal_time_embedding``.

    Computed in float64 and concatenated as ``[sin, cos]``; the caller narrows.
    """
    torch = torch_module()
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("the time tensor is expected to be of shape (batch_size,)")

    # float64 everywhere: get_safe_dtype(torch.float64, ...) never narrows it.
    dtype = torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def euler_update(state, velocity, dt: float):
    """``elementwise.euler_update``: one flow-matching step ``x += dt * v``."""
    return state + dt * velocity
