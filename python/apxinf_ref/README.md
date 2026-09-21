# apxinf-ref — the Pi0.5 reference runtime

An eager, device-independent Pi0.5 that answers one question: **what is the right
number?** It runs the model as ordinary torch on CPU, CUDA (Orin/Thor) or MUSA
(M1000) and emits the same per-stage numeric signature the engine's own probe
emits, so the two documents can be subtracted stage by stage and a disagreement
can be localised to a layer instead of a final action.

It is deliberately *not* an operator-replacement framework. There is no operator
registry, no candidate list and no native directory here. ApxInf classifies
operator coverage through the Operator Gap Table in `doc/adding-new-kernels.md`
and implements replacements in the engine; this package only decides what the
correct output is. Keeping those separate is what lets the reference be trusted:
nothing in it exists to make a particular kernel look good.

## Install

```sh
# MUSA host: the wheels are cp310-aarch64 only.
pyenv install 3.10.21
~/.pyenv/versions/3.10.21/bin/python -m venv .venv
.venv/bin/pip install \
  /path/to/torch-2.9.0-cp310-cp310-linux_aarch64.whl \
  /path/to/torch_musa-2.9.0-cp310-cp310-linux_aarch64.whl \
  /path/to/torchvision-0.22.1+e98278b-cp310-cp310-linux_aarch64.whl
.venv/bin/pip install -e python/apxinf_ref[musa]

# CUDA host: no torchada, no MUSA wheels.
pip install -e python/apxinf_ref
```

The stage probe depends only on torch, transformers, sentencepiece, numpy and
Pillow. There is no CUDA-specific dependency, which is what makes one source tree
serve both accelerators.

`infer` needs two more things, both of which live in this repository and neither
of which the stage probe needs:

```sh
pip install -e python/apxinf            # the pinned resize, prompt and norm stats
pip install -e 'python/apxinf_ref[libero]'   # h5py, to read LIBERO demonstrations
```

## Use

Documents go to **stdout** and the receipt goes to stderr, so a redirect is the
file and `|` is a pipe. `probe` and `infer` write no file themselves: `--capture`
(a bundle) and `compare --json` (the comparison table) are the only flags here
that touch the disk.

```sh
# What can this host run on?
python -m apxinf_ref devices

# A stage probe, matching the engine's fixture (zeros) and view count.
python -m apxinf_ref probe --device musa > probe-musa.json

# Self-validation with a non-zero fixture; not usable against the engine.
python -m apxinf_ref probe --device cpu --fixture random > self.json

# Run the reference on a real observation, and print the stage probe for it.
python -m apxinf_ref infer --device musa --libero-root ~/.cache/openpi/libero_10 \
    --seed 0 > infer-musa.json

# Capture that observation as a bundle. Do this where the observation is -- on
# the host with the camera, the dataset or the simulator. The bundle carries
# this run's probe as its own probe.json, so no redirect is needed.
python -m apxinf_ref infer --device cuda --libero-root <root> --seed 0 \
    --capture orin-bundle

# Replay the bundle here, so both hosts see the same frames and the same frozen
# noise, and subtract the two documents stage by stage. The reference document
# is the one inside the bundle.
python -m apxinf_ref infer --device musa --bundle orin-bundle > replay.json
python -m apxinf_ref compare orin-bundle/probe.json replay.json --thresholds bf16

# Subtract any two probes. An `infer` document is a probe, so this works on it too.
python -m apxinf_ref compare reference.json candidate.json --thresholds bf16
```

`infer` prints the stage-probe schema with the action chunk and the provenance
attached, so `compare` reads it unchanged: the same per-stage subtraction works
against an engine run on the same observation. What it adds is which frame, which
rotation, which resize, which statistics and which noise -- the facts that decide
whether two runs are comparable and that no number in the document reveals.

`--libero-root` reads LIBERO's own HDF5 (see `model/pi05/libero.py` for the two
conventions that ride with it: the frames are stored as robosuite rendered them
and are rotated on the way in, and the prompt comes from the dataset's
`problem_info`). `--capture` writes what this run observed out as an
`apxinf.pi05.bundle.v1` directory -- frames, state, prompt, the frozen noise and
the answer -- so `--bundle` can hand another host exactly the same inputs.

A replay refuses a bundle whose frozen noise is not the draw its seed derives:
noise is an input, so a mismatch is a different question rather than a rounding
difference, and that is deliberately the only hard stop. Everything else a bundle
records -- the three artifact digests, the token ids, and the thresholds the
producer was willing to call agreement -- is reported in the document and changes
no exit code. A container whose manifest does not declare `apxinf.pi05.bundle.v1`
is refused outright, which includes the older `io.npz` bundle: that format
belonged to another project, and a reader that best-effort parses it is a reader
that silently compares two different questions.

`--num-views 2` is the default because it is the profile the engine's stage probe
runs (`Pi05Config::thor_two_view`), and because it is the only view count for
which the engine's dual-GeGLU packing applies. Comparing against a probe taken
at a different view count is meaningless, so the two sides must agree on it.

## What the comparison can and cannot see

The probe keeps six numbers and 256 sampled values per stage — not the tensors.
That is enough to say *which* stage diverged and roughly how much. It is not
enough to say why. A stage that fails should be re-run with tensors kept (the
private workspace has scripts that do this).

The threshold set is copied from
`crates/apxinf-model/examples/pi05_bench.rs:89-110` so a verdict means the
same thing on both sides: per-dtype minimum cosine, maximum relative L2 and, for
INT8, a maximum absolute error.

### Stages that cannot be compared as they stand

`prefix_v_layer0` and `prefix_v_layer17` come back **structural**, not pass or
fail, when the candidate came from `examples/pi05_integrity_probe.rs`, which signs
K/V rows the engine never wrote. `examples/pi05_stage_probe.rs` is the probe to
run; the defect and its diagnosis are in [PI0.5 Reference
Runtime](../../doc/pi05-reference-runtime.md#the-probes-two-known-defects).

## Determinism

Six things are pinned, because without them a comparison measures noise:

1. **The fixture is zeros** by default — images, token ids and diffusion noise —
   matching the engine's integrity probe.
`crates/apxinf-model/examples/pi05_bench.rs:573-574` rejects any
   reference that declares anything else. `--fixture random` exists for
   self-validation only.
2. **Attention is eager everywhere.** `transformers` defaults every tower to
   `sdpa`, and the upstream Pi0.5 inference path pins only the language tower and
   the action expert to `eager` — the vision tower keeps SDPA, whose kernel
   depends on the device and the backend selected for it. A reference has to mean
   the same thing on every device, so this runtime fixes eager everywhere. The
   vendor anchor applies the same pinning when it compares.
3. **Step times are computed, not accumulated.** Each flow step's time is
   `flow_start_time * (1 - step / num_flow_steps)`, the expression the engine
   uses, rather than the upstream loop's repeated `time += dt` in float32. The
   two agree to about an ulp and then drift; the reference does not inherit the
   drift.
4. **The observation path pins its conventions in code**, not in a note.
   `preprocess.py` reuses `apxinf.processors.ResizeWithPad` (the PIL resize
   PI0.5's pipeline selects, not the no-antialias one pi0-FAST uses — they
   disagree by up to ~52/255 at 256→224), `scripts.libero_observation`'s 180°
   camera rotation, and `apxinf.processors.tokenize`'s prompt template and state
   discretiser. Reusing rather than restating is the point: those files record
   ulp-level traps that a reimplementation walks into. The **state width** is
   derived from the checkpoint's own statistics rather than assumed — this
   checkpoint's are 8 wide, not the 7 a collapsed gripper would give, and the
   difference changes the prompt's tokens.
5. **Noise is rounded through bfloat16 before it is frozen**
   (`preprocess.sample_noise`). The engine receives bf16 noise; a float32 draw
   would hand the two implementations different *inputs*, which is a different
   question rather than a rounding difference.
6. **The tokenizer file is pinned by digest.** `PromptTokenizer` records the
   SHA256 of the model it loaded, because two runs that used different tokenizer
   files produced different token ids and their results are not comparable. The
   digest the LIBERO regression record pins is asserted in
   `tests/test_e2e.py`, against a real replay.

Same seed twice produces a byte-identical document.

## The measured floors

Two numbers bound what a comparison can conclude, and both are the same size,
which is the useful part: **same-device implementation floor** (this runtime
against the upstream snapshot, CPU) worst cosine 0.999748, and **cross-device
floor** (CPU against the M1000, same implementation) worst cosine 0.999712.

Read them before reading a comparison: the floor is not uniform across stages,
so a stage that misses a threshold may be measuring the precision rather than the
port -- on the deep vision layers the BF16 gate in `pi05_bench.rs` (cosine >=
0.999) sits barely above the floor itself.

The per-stage table, the mechanism behind it and the three perturbations it was
measured under are in [PI0.5 Reference
Runtime](../../doc/pi05-reference-runtime.md#the-measured-floors).

## How it lines up with the engine

The probe's stage names are the engine's own and are not translated:
`vision_patch_embed`, `vision_layer_{0..26}`, `vision_projected`,
`prefix_v_layer{0,17}`, `denoise_step_{0..9}`. The signature algorithm is
transcribed from `fn signature` in `crates/apxinf-model/examples/pi05_stage_probe.rs`,
including the parts that look like accidents and are not:

* `sum`, `abs_checksum`, `l2` and `max_abs` accumulate in float64 **in memory
  order**. Widening float32 to float64 makes that accumulation exact for any
  tensor of realistic magnitude, so the order rarely shows — but the Rust loop is
  a running total, and `cumsum` reproduces it exactly rather than approximately;
* `l2` is `sqrt(sum(v**2))`, a norm that grows with tensor size, not an RMS;
* `sample` takes `min(elements, 256)` values at `i * (elements - 1) //
  (sample_count - 1)`, integer division, both endpoints always present.

### Three pieces of model structure that must not be "cleaned up"

* **The last language layer is truncated.** The engine passes
  `compute_tail = index + 1 < depth`
  (`crates/apxinf-model/src/pi05/model/blocks/bf16.rs:356`) and the executor,
  when it is false, returns the layer's input alongside the K/V it just wrote —
  no attention, no output projection, no MLP
  (`crates/apxinf-model/src/pi05/model/blocks/bf16.rs:47`). It saves
  work without changing any result, and `prefix_v_layer17` can therefore only
  ever validate the post-RoPE K/V of that layer, never its attention stack.
* **Action layers reuse the previous layer's normalisation.** A layer's
  `next_normalized` is the next layer's `attention_normalized`
  (`crates/apxinf-model/src/pi05/model/blocks/bf16.rs:20`, consumed at `:475`),
  so the normalisation is not recomputed.
* **The checkpoint's float32 parameters are not a rounding detail.**
  `to_bfloat16_for_selected_params` matches parameter names as *substrings*, and
  `"input_layernorm"` / `"model.norm"` also match the action expert's adaRMS
  `dense` projections — so those are float32 while the vision tower's LayerNorm
  weights are bfloat16. `F.linear` refuses mixed dtypes rather than promoting,
  which is why `adarms_cond` must be float32 for the model to run at all.
  `materialize_precision` reproduces this split.

## Three differences from the engine that the probe will show

All three are representation choices on the engine side, not errors:

* the engine stores the sinusoidal time embedding as **bf16**
  (`upload_time_embeddings_bf16`); upstream computes it in float64, narrows to
  float32 and keeps `time_mlp` in float32;
* the engine computes the embedding's phase as `TAU / period` (`math.rs`), upstream
  as `(1 / period) * 2 * pi`. Algebraically equal, differently associated in
  float64.
* the engine folds Gemma's `1 + gamma` into the consuming projections
  (`crates/apxinf-model/src/pi05/weights/host.rs:192-207`) and rounds the
  scale to bfloat16 on the way (`add_one`,
  `crates/apxinf-model/src/pi05/weights/host.rs:556`); the upstream reference
  keeps the scale in
  float32 and multiplies the activation. **Measured cost: up to 1.0e-02 relative
  L2 on the prefix K/V**, zero on the vision tower (which has no `1 + gamma`).
  See `devlocal/pi05-ref-runtime/reports/anchor-fidelity.md`.

## Layout

```
apxinf_ref/
  device.py              torchada ordering, native MUSA detection, no silent fallback
  model/pi05/
    config.py            defaults mirroring crates/apxinf-model/src/pi05/config.rs
    weights.py           checkpoint loading + the upstream precision policy
    preprocess.py        images, proprioception and noise, with the pinned conventions
    tokenizer.py         prompt construction and SentencePiece
    libero.py            one demonstration frame out of LIBERO's HDF5
    observation.py       observation -> model inputs, and raw actions -> LIBERO's
    nn_ops.py            eager primitives, named after the engine's kernels
    vision.py            SigLIP tower
    gemma.py             PaliGemma language tower and the action expert
    model.py             prefix, denoising loop, Euler integration
  model/vendor_pi05.py   the upstream snapshot behind the same interface
  engines.py             which model, at which precision, on which device
  paths.py               checkpoint / tokenizer / norm-stats resolution
  probe.py               apxinf.pi05.stage-probe.v1
  infer.py               a real observation through the same forward pass
  sources.py             where that observation came from
  bundle.py              apxinf.pi05.bundle.v1: the capture container
  compare.py             stage-by-stage comparison
  cli.py                 python -m apxinf_ref
  vendor/pi05/           upstream snapshot, untouched (see its VENDOR.md)
```
