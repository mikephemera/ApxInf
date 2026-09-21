"""Make the in-tree packages importable without an install, and locate artifacts.

The fixtures here do not restate where the checkpoint, the tokenizer and the
normalization statistics live: they call the same resolvers the CLI calls
(:mod:`apxinf_ref.paths`), so a test and a run cannot disagree about which file
they used. LIBERO is the exception -- nothing in the package reads it but
``infer --libero-root`` -- so its root follows the convention the runbook
documents: ``APXINF_LIBERO_ROOT``, then the same cache the other three live in.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parents[1]
_REPO = _HERE.parents[1]

sys.path.insert(0, str(_HERE))
# `apxinf_ref` reuses `apxinf`'s resize, prompt and discretisation conventions
# rather than restating them, so the observation-path tests need it importable.
# It is a sibling package in the same repository, not an installed dependency.
if (_REPO / "python" / "apxinf").is_dir():
    sys.path.insert(0, str(_REPO / "python" / "apxinf"))
if (_REPO / "scripts").is_dir():
    sys.path.insert(0, str(_REPO))

from apxinf_ref import paths  # noqa: E402 - needs the paths above

DEFAULT_LIBERO_ROOT = paths.CACHE_ROOT / "libero_10"


@pytest.fixture(scope="session")
def checkpoint_path() -> Path:
    """The PI0.5 checkpoint, or a skip that says how to provide one."""
    try:
        return paths.resolve_checkpoint()
    except FileNotFoundError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="session")
def tokenizer_path(checkpoint_path) -> Path:
    """PI0.5's SentencePiece model, or a skip that says how to provide it.

    The checkpoint does not ship it; `paths.resolve_tokenizer` documents where it
    looks. A comparison run whose two sides used different files is not
    comparable, which is why this resolves through the same code a run does.
    """
    try:
        return paths.resolve_tokenizer(checkpoint_path)
    except ImportError as error:
        pytest.skip(str(error))
    except RuntimeError as error:
        # `apxinf.checkpoints.layout.CheckpointError`, raised once every
        # candidate location has been tried.
        pytest.skip(str(error))


@pytest.fixture(scope="session")
def norm_stats_path(checkpoint_path) -> Path:
    """The LIBERO normalization statistics the checkpoint is normalised against."""
    try:
        return paths.resolve_norm_stats(checkpoint_path)
    except (FileNotFoundError, ImportError) as error:
        pytest.skip(str(error))


@pytest.fixture(scope="session")
def libero_root() -> Path:
    """A LIBERO dataset root, or a skip that says how to provide one."""
    root = Path(os.environ.get("APXINF_LIBERO_ROOT", DEFAULT_LIBERO_ROOT)).expanduser()
    if not root.is_dir() or not any(root.glob("*.hdf5")):
        pytest.skip(
            f"no LIBERO dataset at {root}; set APXINF_LIBERO_ROOT to a directory "
            "holding the *.hdf5 demonstrations"
        )
    return root
