"""The ``apxinf.pi05.stage-probe.v1`` document.

This is the interface between the two halves of the comparison: the engine's
``crates/apxinf-model/examples/pi05_stage_probe.rs`` emits this document and
so does this module, and ``compare.py`` subtracts them stage by stage.

The signature algorithm is transcribed from ``fn signature`` in
``crates/apxinf-model/examples/pi05_stage_probe.rs`` rather than re-derived,
including the details that are easy to "clean up" and wrong to:

* ``elements`` is the flat element count, not ``numel`` of a shaped view;
* ``sum``, ``abs_checksum``, ``l2`` and ``max_abs`` accumulate in float64 after
  widening each float32, and ``l2`` is the norm ``sqrt(sum(v**2))`` -- *not* an
  RMS, so it grows with the tensor's size;
* the accumulation is **sequential**, in memory order. NumPy's ``sum`` uses
  pairwise summation and lands on a different last bit, so this module uses
  ``cumsum`` and takes the final prefix, which is the same running total the Rust
  loop produces;
* ``sample`` holds ``min(elements, 256)`` values at indices
  ``i * (elements - 1) // (sample_count - 1)`` -- integer division, both
  endpoints always included -- with a special case for a single element.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .device import torch_module

__all__ = ["SCHEMA", "ProbeResult", "collect", "run", "signature"]

SCHEMA = "apxinf.pi05.stage-probe.v1"

#: ``fn signature`` in ``examples/pi05_stage_probe.rs`` -- at most this many values.
SAMPLE_LIMIT = 256


def signature(values) -> dict:
    """The six-field signature of a flat float32 sequence."""
    import numpy as np

    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    elements = int(flat.size)
    if elements == 0:
        raise ValueError("a stage signature over zero elements is not meaningful")

    wide = flat.astype(np.float64)
    # Sequential accumulation, matching the Rust loop element for element.
    total = float(np.cumsum(wide)[-1])
    absolute = float(np.cumsum(np.abs(wide))[-1])
    square_sum = float(np.cumsum(wide * wide)[-1])
    max_abs = float(np.abs(wide).max())

    sample_count = min(elements, SAMPLE_LIMIT)
    if sample_count == 1:
        indices = np.zeros(1, dtype=np.int64)
    else:
        indices = (
            np.arange(sample_count, dtype=np.int64) * (elements - 1) // (sample_count - 1)
        )

    return {
        "elements": elements,
        "sum": total,
        "abs_checksum": absolute,
        "l2": float(np.sqrt(square_sum)),
        "max_abs": max_abs,
        "sample": [float(value) for value in flat[indices]],
    }


def _to_f32(device_tensor) -> "Any":
    """Match Rust's ``to_cpu(tensor)?.to_f32_vec()``: widen to float32, in memory order."""
    import numpy as np

    torch = torch_module()
    with torch.no_grad():
        tensor = device_tensor.detach()
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        return np.ascontiguousarray(tensor.cpu().numpy()).reshape(-1)


class ProbeResult:
    """A completed stage probe: the document, the raw tensors, and the actions.

    ``actions`` is what the model finally produced, and exists for the callers
    that want an action chunk rather than a signature -- ``infer`` emits the same
    document as ``probe`` with the actions attached, so one comparison tool reads
    both. It stays ``None`` for a caller that only wants the stages.
    """

    __slots__ = ("action", "document", "tensors")

    def __init__(self, document: dict, tensors: Mapping[str, Any], action=None) -> None:
        self.document = document
        self.tensors = dict(tensors)
        self.action = action

    @property
    def signatures(self) -> dict:
        return self.document["intermediate_signatures"]


def collect(
    stage_tensors: Mapping[str, Any],
    *,
    token_count: int,
    on_signature: Callable[[str, dict], None] | None = None,
) -> ProbeResult:
    """Build the probe document from tensors that were captured per stage."""
    signatures = {}
    for name in sorted(stage_tensors):
        value = signature(_to_f32(stage_tensors[name]))
        signatures[name] = value
        if on_signature is not None:
            on_signature(name, value)
    document = {
        "schema": SCHEMA,
        "token_count": int(token_count),
        "intermediate_signatures": signatures,
    }
    return ProbeResult(document, stage_tensors)


def run(model, *, patches, token_ids, noise) -> ProbeResult:
    """Drive ``model`` through the three phase boundaries and capture every stage.

    ``model`` is anything exposing the assembled runtime's interface --
    ``model.pi05.Pi05`` or ``model.vendor_pi05.VendorPi05`` -- so the anchor
    comparison is a change of engine and nothing else.
    """
    stages: dict = {}

    def on_stage(name: str, tensor) -> None:
        stages[name] = tensor

    embeddings = model.embed_prefix(patches, token_ids, on_stage=on_stage)
    cache = model.prefix_forward(embeddings, on_stage=on_stage)
    action = model.sample_actions(cache, noise, on_stage=on_stage)
    result = collect(stages, token_count=int(token_ids.shape[-1]))
    result.action = action
    return result
