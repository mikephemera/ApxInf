"""Where a run's artifacts come from.

Three artifacts decide whether two runs are comparable at all -- the checkpoint,
the SentencePiece model and the normalization statistics -- so each resolves the
same way and says so: an explicit argument, then the environment variable this
repository already uses for that artifact, then the layout OpenPI's downloader
produces. Nothing is guessed, and a missing artifact is an error naming every
place that was tried rather than a quiet fall back to a different file. Two runs
that used different artifacts are not comparable, and the difference would be
invisible in the numbers.

The environment variable names are the ones already in use elsewhere in this
repository rather than new ones:

* ``APXINF_PI05_CHECKPOINT`` -- ``crates/apxinf-py/tests/conftest.py``;
* ``APXINF_TOKENIZER`` -- ``apxinf.checkpoints.layout.resolve_tokenizer``, which
  is imported rather than restated because it already encodes the search order;
* ``APXINF_NORM_STATS`` -- the reference package's own test fixture.

Only :func:`resolve_checkpoint` is importable without ``apxinf``: the stage probe
takes patches and token ids directly, and that path has to keep working on a host
where neither ``apxinf`` nor a tokenizer is installed.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_TOKENIZER",
    "ENV_CHECKPOINT",
    "ENV_NORM_STATS",
    "ENV_TOKENIZER",
    "WEIGHTS_NAME",
    "resolve_checkpoint",
    "resolve_norm_stats",
    "resolve_tokenizer",
]

ENV_CHECKPOINT = "APXINF_PI05_CHECKPOINT"
ENV_TOKENIZER = "APXINF_TOKENIZER"
ENV_NORM_STATS = "APXINF_NORM_STATS"

#: Where OpenPI's downloader puts its assets, which is where this project's
#: development machines have them. A default, not a contract: every resolver
#: accepts an explicit path and an environment variable first.
CACHE_ROOT = Path.home() / ".cache" / "openpi"
DEFAULT_CHECKPOINT = CACHE_ROOT / "openpi-assets/checkpoints/pi05_libero_pytorch"
DEFAULT_TOKENIZER = CACHE_ROOT / "big_vision/paligemma_tokenizer.model"

#: The weights file every loader here reads. Its presence is what makes a
#: directory a checkpoint rather than just a directory.
WEIGHTS_NAME = "model.safetensors"


def _first_file(candidates) -> Path | None:
    """The first candidate that is an existing file, ignoring ``None`` slots."""
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.is_file():
            return path
    return None


def _not_found(candidates, what: str, how_to: str) -> FileNotFoundError:
    tried = "\n".join(f"  {candidate}" for candidate in candidates if candidate is not None)
    return FileNotFoundError(f"no {what} found; tried:\n{tried}\n{how_to}")


def try_import_layout():
    """``apxinf.checkpoints.layout``, or a message naming what is missing."""
    try:
        from apxinf.checkpoints import layout  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - depends on the host
        raise ImportError(
            "resolving the tokenizer and the normalization statistics needs the "
            "`apxinf` package (it lives at python/apxinf in this repository; "
            "`pip install -e python/apxinf`). The stage probe does not need it: "
            "it takes patches and token ids directly."
        ) from error
    return layout


def resolve_checkpoint(explicit: str | Path | None = None) -> Path:
    """The PI0.5 checkpoint directory: argument, then env, then the cache default.

    ``APXINF_PI05_CHECKPOINT`` is the same variable ``crates/apxinf-py``'s
    conftest reads, so one setting serves both sides of a comparison.
    """
    directories = [explicit, os.environ.get(ENV_CHECKPOINT), DEFAULT_CHECKPOINT]
    weights = _first_file(
        Path(directory) / WEIGHTS_NAME for directory in directories if directory is not None
    )
    if weights is None:
        raise _not_found(
            directories,
            "PI0.5 checkpoint",
            f"A checkpoint is a directory holding {WEIGHTS_NAME}. Pass --checkpoint, "
            f"set {ENV_CHECKPOINT}, or place one at {DEFAULT_CHECKPOINT}.",
        )
    return weights.parent


def resolve_tokenizer(checkpoint: str | Path, explicit: str | Path | None = None) -> Path:
    """The SentencePiece model: argument, env, this project's cache, then the layout.

    The checkpoint-directory candidates and the error message are delegated to
    ``apxinf`` rather than restated -- that function already owns the convention
    and already explains why both sides of a comparison must use the same file.
    What is added here is ``DEFAULT_TOKENIZER``: the one PaliGemma tokenizer
    serves the whole family, so this project keeps a single copy under the OpenPI
    cache instead of duplicating it into every checkpoint directory, and no
    checkpoint ships one.
    """
    found = _first_file([explicit, os.environ.get(ENV_TOKENIZER), DEFAULT_TOKENIZER])
    if found is not None:
        return found
    return try_import_layout().resolve_tokenizer(Path(checkpoint), explicit)


def resolve_norm_stats(checkpoint: str | Path, explicit: str | Path | None = None) -> Path:
    """The normalization statistics: argument, then env, then the checkpoint's layout.

    The checkpoint's own copy wins over any global default, because the
    statistics and the weights have to be the same export's -- and unlike the
    tokenizer, a norm_stats.json is routinely stored inside the checkpoint.
    """
    tried = [
        explicit,
        os.environ.get(ENV_NORM_STATS),
        # OpenPI's downloaded layout. Tried directly rather than through
        # detect_checkpoint because it is unambiguous and needs no sniffing.
        Path(checkpoint) / "assets/physical-intelligence/libero/norm_stats.json",
        Path(checkpoint) / "norm_stats.json",
    ]
    found = _first_file(tried)
    if found is not None:
        return found

    detected = _detect_norm_stats(checkpoint)
    if detected is not None:
        return detected
    raise _not_found(
        tried,
        f"normalization statistics for {checkpoint}",
        f"Pass --norm-stats or set {ENV_NORM_STATS}. LIBERO's are published with "
        "the checkpoint under assets/physical-intelligence/libero/norm_stats.json.",
    )


def _detect_norm_stats(checkpoint: str | Path) -> Path | None:
    """Ask ``apxinf``'s layout detector, treating an unknown layout as "not found".

    ``detect_checkpoint`` raises for a directory it cannot identify, and the
    exception types it can raise are not worth enumerating here: this is the last
    resort, reached only after every convention in :func:`resolve_norm_stats` has
    already failed, so an unidentified layout means the same thing as a missing
    file.
    """
    layout = try_import_layout()
    try:
        detected = layout.detect_checkpoint(Path(checkpoint)).norm_stats
    except Exception:  # noqa: BLE001 - see the docstring; any failure means "absent"
        return None
    if detected is None:
        return None
    path = Path(detected)
    return path if path.is_file() else None
