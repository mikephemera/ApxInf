"""Device resolution for the reference runtime.

One source tree serves three devices -- ``cpu``, ``cuda`` (Orin/Thor) and
``musa`` (Moore Threads M1000). Two rules make that work and both are easy to
get wrong:

``torchada`` is imported before any other torch-shaped API is touched. It is
what rewrites the torch namespace for Moore Threads hardware; a process that
imports a CUDA-shaped symbol first can end up holding the un-rewritten version.

Accelerator detection never goes through ``torch.cuda.is_available()``.
``torchada`` deliberately leaves that function alone -- it still returns
``False`` on MUSA -- so the native ``torch.musa`` signal is the only honest
one. This mirrors ``FlashRT_musa/flash_rt/hardware/__init__.py:detect_arch``.

Nothing here falls back silently. A request for a device that is not present
raises, and a tensor sitting on the wrong device raises with both device types
named; a runtime that quietly moved the work to the CPU would make every
downstream comparison meaningless.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = [
    "ACCELERATORS",
    "DEVICE_CHOICES",
    "available_accelerators",
    "describe_devices",
    "manual_seed_all",
    "resolve",
    "resolve_device_string",
    "synchronize",
    "torch_module",
]

#: Device families this runtime knows how to resolve.
DEVICE_CHOICES = ("cpu", "cuda", "musa")

#: The two accelerator families, in the order they are probed. MUSA is probed
#: first because ``torch.cuda.is_available()`` is not a reliable negative on a
#: MUSA host and a MUSA device must never be mistaken for an NVIDIA one.
ACCELERATORS = ("musa", "cuda")

_TORCH: Any = None
_TORCHADA: Any = None
_TORCHADA_ERROR: str | None = None


def torch_module() -> Any:
    """Import ``torchada`` (when present) and then ``torch``, once.

    The ordering is the whole point of this function: ``torchada`` has to see
    the torch namespace before it is handed out, so callers must obtain torch
    through here rather than importing it themselves.
    """
    global _TORCH, _TORCHADA, _TORCHADA_ERROR
    if _TORCH is not None:
        return _TORCH

    if _TORCHADA is None and _TORCHADA_ERROR is None:
        try:
            import torchada  # type: ignore[import-not-found]
        except ImportError as error:  # pragma: no cover - depends on host
            _TORCHADA_ERROR = str(error)
        else:
            _TORCHADA = torchada

    import torch

    _TORCH = torch
    return _TORCH


def _accelerator_is_available(torch: Any, name: str) -> bool:
    if name == "musa":
        musa = getattr(torch, "musa", None)
        if musa is None or not callable(getattr(musa, "is_available", None)):
            return False
        return bool(musa.is_available())
    if name == "cuda":
        return bool(torch.cuda.is_available())
    return False


def available_accelerators() -> tuple[str, ...]:
    """Return the accelerator families that are actually usable right now."""
    torch = torch_module()
    return tuple(name for name in ACCELERATORS if _accelerator_is_available(torch, name))


def _device_name(torch: Any, family: str, index: int) -> str:
    if family == "musa":
        musa = getattr(torch, "musa", None)
        if musa is not None and callable(getattr(musa, "get_device_name", None)):
            return str(musa.get_device_name(index))
        return f"musa:{index}"
    if family == "cuda":
        return str(torch.cuda.get_device_name(index))
    return "cpu"


def describe_devices() -> str:
    """A one-line-per-device summary, used by ``--list-devices`` and errors."""
    torch = torch_module()
    lines = []
    for name in ACCELERATORS:
        if _accelerator_is_available(torch, name):
            try:
                lines.append(f"{name}: available ({_device_name(torch, name, 0)})")
            except Exception as error:  # pragma: no cover - driver-dependent
                lines.append(f"{name}: available (device name unavailable: {error})")
        else:
            lines.append(f"{name}: not available")
    torchada_state = "imported" if _TORCHADA is not None else f"absent ({_TORCHADA_ERROR})"
    lines.append(f"torchada: {torchada_state}")
    lines.append(f"torch: {torch.__version__} ({os.path.dirname(torch.__file__)})")
    return "\n".join(lines)


def resolve_device_string(spec: str, *, index: int | None = None) -> str:
    """Turn a ``--device`` value into a torch device string, validating it.

    ``index`` overrides any index already spelled into ``spec`` (``cuda:1``).
    """
    family, _, spelled_index = str(spec).partition(":")
    family = family.strip().lower()
    if family not in DEVICE_CHOICES:
        raise ValueError(
            f"unknown device {spec!r}: expected one of {', '.join(DEVICE_CHOICES)} "
            f"(optionally with an index, e.g. 'cuda:1')"
        )

    if index is None and spelled_index:
        try:
            index = int(spelled_index)
        except ValueError as error:
            raise ValueError(f"device index in {spec!r} is not an integer") from error
    if index is None:
        index = 0

    torch = torch_module()

    if family == "cpu":
        return "cpu"
    if not _accelerator_is_available(torch, family):
        raise RuntimeError(
            f"requested device {spec!r} is not available on this host.\n"
            f"{describe_devices()}\n"
            "Refusing to fall back to another device: a reference run that "
            "silently changed device would invalidate every comparison made "
            "against it."
        )
    if family == "musa" and _TORCHADA is None:
        raise RuntimeError(
            "device 'musa' requires the 'torchada' package, which is not "
            f"importable ({_TORCHADA_ERROR}). Install it with "
            "`pip install torchada`, or use the 'musa' extra of this package."
        )
    return f"{family}:{index}"


def resolve(spec: str, *, index: int | None = None) -> "torch.device":
    """Resolve ``spec`` to a ``torch.device``, or raise explaining why not."""
    torch = torch_module()
    return torch.device(resolve_device_string(spec, index=index))


def synchronize(device: "torch.device") -> None:
    """Block until ``device`` has finished its queued work."""
    torch = torch_module()
    if device.type == "musa":
        torch.musa.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def manual_seed_all(device: "torch.device", seed: int) -> None:
    """Seed every generator that can contribute to a run on ``device``."""
    torch = torch_module()
    torch.manual_seed(seed)
    if device.type == "musa":
        torch.musa.manual_seed_all(seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def assert_tensor_device(tensor: Any, device: "torch.device", *, what: str) -> None:
    """Fail loudly when a tensor is not on the device the run asked for."""
    if tensor.device.type != device.type or (
        device.index is not None
        and tensor.device.index is not None
        and tensor.device.index != device.index
    ):
        raise RuntimeError(
            f"{what} is on {tensor.device} but this run was asked to use "
            f"{device}. Refusing to continue: a silently misplaced tensor "
            "makes every downstream stage signature meaningless."
        )
