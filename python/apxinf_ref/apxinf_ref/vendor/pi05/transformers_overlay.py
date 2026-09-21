"""Install the local Transformers model replacements used by Pi0.5.

OpenPI patches a few Transformers classes for the PaliGemma/Gemma checkpoint.
Loading those files from this package keeps the MUSA runtime independent of an
OpenPI checkout while preserving the upstream model implementation.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import TypedDict


def install_transformers_overlay() -> None:
    """Load the bundled replacement modules into Transformers' module names."""
    import transformers

    if not hasattr(transformers.utils, "LossKwargs"):
        transformers.utils.LossKwargs = TypedDict("LossKwargs", {}, total=False)

    root = Path(__file__).parent / "models_pytorch" / "transformers_replace" / "models"
    modules = (
        ("transformers.models.gemma.configuration_gemma", "gemma/configuration_gemma.py"),
        ("transformers.models.gemma.modeling_gemma", "gemma/modeling_gemma.py"),
        ("transformers.models.siglip.modeling_siglip", "siglip/modeling_siglip.py"),
        ("transformers.models.paligemma.modeling_paligemma", "paligemma/modeling_paligemma.py"),
        ("transformers.models.siglip.check", "siglip/check.py"),
    )
    for module_name, relative_path in modules:
        path = root / relative_path
        if not path.is_file():
            raise ImportError(f"bundled Pi0.5 Transformers module is missing: {path}")
        parent_name, attribute = module_name.rsplit(".", 1)
        parent = importlib.import_module(parent_name)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load bundled Transformers module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        setattr(parent, attribute, module)

    # The bundled check intentionally accepts this local overlay regardless of
    # whether a distro has added a patch marker of its own.
    from transformers.models.siglip import check
    check.check_whether_transformers_replace_is_installed_correctly = lambda: True
