"""Install the pinned Transformers replacements required by vendored Pi0.5."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import TypedDict


def install_transformers_overlay() -> None:
    import transformers

    if not hasattr(transformers.utils, "LossKwargs"):
        transformers.utils.LossKwargs = TypedDict("LossKwargs", {}, total=False)

    # Upstream nests these one level deeper, under ``models/``.  This repository
    # ignores every directory called ``models/`` (.gitignore), which would drop
    # any replacement added there without a word, so the family directories sit
    # directly under ``transformers_replace/``.  Nothing in the layout below
    # carries meaning to Python: a module's parent comes from the dotted name
    # handed to ``spec_from_file_location``, never from where the file lives.
    root = Path(__file__).parent / "transformers_replace"
    # Order is load-bearing.  ``modeling_paligemma`` imports GemmaModel and
    # SiglipVisionModel through ``..gemma``/``..siglip``, which resolve out of
    # ``sys.modules``.  Load paligemma first and it would bind to the *unpatched*
    # transformers modules instead of failing -- gemma and siglip have to be in
    # place before it runs.
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
            raise ImportError(f"vendored Transformers replacement is missing: {path}")
        parent_name, attribute = module_name.rsplit(".", 1)
        parent = importlib.import_module(parent_name)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load vendored Transformers replacement: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        setattr(parent, attribute, module)

    from transformers.models.siglip import check

    check.check_whether_transformers_replace_is_installed_correctly = lambda: True
