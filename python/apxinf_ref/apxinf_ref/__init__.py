"""Device-independent eager Pi0.5 reference runtime.

This package is the ground-truth oracle that the ApxInf CUDA and MUSA engines
are measured against. It runs the model as plain eager torch on any device
torch (and, on Moore Threads hardware, ``torchada``) can address, and emits the
``apxinf.pi05.stage-probe.v1`` per-stage numeric signature that
``crates/apxinf-model/examples/pi05_stage_probe.rs`` also emits. Subtracting
the two documents stage by stage localises numerical loss.

Deliberately *not* here: any operator-replacement machinery. This runtime has no
operator registry, no candidate slots and no native directory. ApxInf's operator
coverage is classified through the Operator Gap Table in
``doc/adding-new-kernels.md`` and lands in the engine, not in this oracle. The
two are independent: this package answers "what is the right number", the engine
answers "which implementation produced it".
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
