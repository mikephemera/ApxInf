# Standalone Pi0.5 runtime vendor

This directory is `apxinf_ref`'s package-local copy of the JAX-free PyTorch Pi0.5
implementation captured from `training/_vendor/openpi_pi0_pytorch` (OpenPI
PyTorch port, 2026-04-25). The model, preprocessing, and configuration files
use only relative imports and are safe to ship in the wheel.

The required Gemma, PaliGemma, and SigLIP Transformers replacements are bundled
under `models_pytorch/transformers_replace`. `transformers_overlay.py` installs
them into Transformers' normal module names at model import time. This is a
runtime compatibility layer, not a dependency on an OpenPI checkout. The
runtime is tested against Transformers 4.53.2; use the `apxinf-ref[musa]` extra
(or an equivalent manually pinned installation).

The reference runtime constructs `PI0Pytorch` with `pytorch_compile_mode=None`.
This deliberately uses eager BF16 execution: it claims no FP8, FP4 or CUDA-graph
kernels on any device, which is what makes it the same model to compare an
engine against. Keep this copy synchronized with the training vendor when
updating the OpenPI snapshot, and preserve the upstream OpenPI/Gemma license
notices in source distributions.
