"""Which model, at which precision, on which device.

Split out of the command line so that ``probe`` and ``infer`` build the same
engine the same way. The two commands differ in what they feed the model and what
they do with the result, not in what the model is -- and a difference in engine
construction between them would be invisible in both outputs.

``reference`` and ``vendor`` are the two engines. ``vendor`` is the untouched
upstream snapshot behind the same interface, kept so the assembled implementation
can be shown to be the same model; it follows the checkpoint's own precision and
cannot be widened, which is why ``--precision`` is refused for it rather than
quietly ignored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import device as device_module

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

__all__ = ["ENGINES", "PRECISIONS", "build_engine", "resolve_precision"]

ENGINES = ("assembled", "vendor")

#: ``checkpoint`` follows the checkpoint's own precision, which is the definition
#: of a reference run. ``float32`` widens weights and activations together and
#: exists to answer one question: is a divergence the bfloat16 arithmetic, or the
#: algorithm? Widening the activations without widening the weights would raise
#: from ``F.linear`` rather than answer it.
PRECISIONS = ("checkpoint", "float32")


def resolve_precision(requested: str, config) -> str:
    """The width a run actually computes in."""
    if requested not in PRECISIONS:
        raise ValueError(f"unknown precision {requested!r}: expected one of {', '.join(PRECISIONS)}")
    return config.precision if requested == "checkpoint" else "float32"


def build_engine(*, engine: str, checkpoint: "Path | str", config, device, precision: str):
    """Construct the model named by ``engine`` at ``precision`` on ``device``."""
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}: expected one of {', '.join(ENGINES)}")

    if engine == "vendor":
        from .model.vendor_pi05 import VendorPi05

        if precision != config.precision:
            raise ValueError(
                f"the vendor snapshot follows the checkpoint's precision "
                f"({config.precision}); --precision is only available for the "
                "assembled engine"
            )
        return VendorPi05(checkpoint, config, device)

    from .model.pi05.model import Pi05
    from .model.pi05.weights import load_weights, materialize_precision

    weights = materialize_precision(load_weights(checkpoint), precision).to(device=device)
    dtype = device_module.torch_module().float32 if precision == "float32" else None
    return Pi05(weights, config, device, dtype=dtype)
