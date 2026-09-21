# PI0.5 Reference Runtime

`python/apxinf_ref` is a device-independent, eager PyTorch PI0.5 that runs on
CPU, CUDA and MUSA from one source tree and emits the same per-stage numeric
signature the engine's own probe emits
(`crates/apxinf-model/examples/pi05_stage_probe.rs`). Subtracting the two
documents stage by stage answers the question the task-success-rate numbers
cannot: **which layer lost the precision.**

It exists for two targets at once — pinning the CUDA engine's numerical baseline
on Orin/Thor, and establishing one for MUSA before the engine is ported — which
is why it is device-independent rather than a CUDA script.

The stage probe takes patches and token ids directly, which is what the engine's
own probe emits. Putting a real observation in front of that is what `infer` is
for: it reads a LIBERO demonstration or a captured bundle, builds the model's
inputs, runs the same forward pass, and writes the same probe document with the
action chunk and the provenance attached.

Every convention on that path is reused rather than restated -- `apxinf`'s pinned
resize, prompt template and normalization, and `libero_observation`'s 180-degree
camera rotation -- and the noise is rounded through bfloat16 before it is frozen.
Those conventions are not cosmetic: the two resizes `apxinf` ships disagree by up
to 52/255 at 256→224, which is enough to change the emitted action tokens, and
LIBERO's HDF5 stores frames in the renderer's orientation rather than the
policy's.

```sh
python -m apxinf_ref devices                      # what can this host run on
python -m apxinf_ref probe --device musa --out musa.json
python -m apxinf_ref infer --device musa --libero-root <libero-root> --out infer.json
python -m apxinf_ref infer --device cuda --libero-root <root> --capture bundle --out orin.json
python -m apxinf_ref infer --device musa --bundle bundle --out replay.json
python -m apxinf_ref compare reference.json candidate.json --thresholds bf16
```

Read [`python/apxinf_ref/README.md`](../python/apxinf_ref/README.md) for the
interface, the determinism rules and the measured floors.

## What it is not

It is **not** an operator-replacement framework, and it has no operator registry,
candidate list or native slot: the reference only decides what the correct output
is, and nothing in it exists to make a particular kernel look good. ApxInf
classifies operator coverage through the Operator Gap Table in [Adding New
Kernels](adding-new-kernels.md) and implements replacements in the engine.

The MUSA side of that framework is
[`crates/apxinf-musa/`](../crates/apxinf-musa/README.md). No MUSA candidate is
registered yet, so every semantic there resolves to `deferred` and produces no
value at all — the honest state of a port that has not started. A deferred
semantic is the fourth answer to the Fallback column
([Adding New Kernels](adding-new-kernels.md) §3): the stage is compared against
the reference and the comparison says plainly that no MUSA number exists for it.

## Capture and replay

A stage probe takes patches and token ids directly, so a probe taken on one host
says nothing about a probe taken on another unless both were handed the same
inputs. A **bundle** is that input: one observation, frozen where it was made and
replayed wherever the port needs a comparison. `infer --capture` writes one and
`infer --bundle` reads it, which is what lets an Orin capture be subtracted
stage by stage against a MUSA replay rather than scored on one final cosine.

```text
Orin (CUDA host)                              MUSA (M1000)
────────────────                              ────────────
apxinf_ref infer --device cuda \              apxinf_ref infer --device musa \
    --libero-root <root> --seed S \              --bundle <dir> --out musa.json
    --capture <dir> --out orin.json                       │
        │                                                 │
        ├── <dir>/manifest.json   the contract            compare orin.json musa.json
        ├── <dir>/inputs.npz      inputs and the answer   --thresholds bf16
        ├── <dir>/probe.json      Orin's own stages         → subtracted stage by stage
        └── <dir>/run.log         a human-readable receipt
```

The producer is the **eager reference runtime**, not the engine. A bundle's
`gold_actions` are therefore what "the right number" means for that observation,
and a replay measures the other host against the reference rather than against a
particular build. The engine's own numbers stay where they already live:
`crates/apxinf-model/examples/pi05_stage_probe.rs` emits the same document schema
from the other side, and comparing *that* against a reference probe is the
engine-versus-reference question -- a different axis, which a bundle does not
serve.

`--capture` and `--out` are independent, and deliberately so: a capture that
recorded a probe but no bundle, or a bundle with no standalone probe, would each
be a way to lose half the result. Storing the producer's own probe inside the
bundle is the load-bearing part of the design: without it a replay can only be
scored on the final action chunk.

### What a bundle holds

Frames are stored **oriented but not resized**. The rotation is a property of the
observation -- LIBERO stores frames as the renderer drew them and the policy wants
them the other way up -- while the resize is a deterministic function of the frame
and the pinned conventions. Storing the pre-resize frame makes the replay re-run
the letterbox on its own hardware, so the preprocessing path is part of what the
comparison exercises instead of being assumed away.

| name | shape | dtype | why it is here |
|---|---|---|---|
| `base_image`, `wrist_image` | `(H, W, 3)` | `uint8` | the inputs, in the policy's orientation |
| `state` | `(8,)` | `float32` | proprioception, before normalisation |
| `prompt`, `formatted_prompt` | `(1,)` | `str` | the task, and the template's output |
| `noise` | `(10, 32)` | `float32` | frozen before it was used; an **input** |
| `token_ids` | `(46,)` | `int64` | lets a replay check that the same prompt discretised into the same ids |
| `gold_actions` | `(10, 7)` | `float32` | what the producer's run returned, in LIBERO's units |
| `gold_raw_actions` | `(10, 32)` | `float32` | the chunk before unnormalisation |
| `preprocessed_base_rgb`, `preprocessed_wrist_rgb` | `(1, 3, 224, 224)` | `float32` | optional; localises a divergence to the resize or the patch layout |

The array holding the answer is named `gold_actions`, and that name is fixed by
the format. Naming it after the device that produced it is a mistake worth not
repeating: a reader that looks for one name finds it on one host and silently
finds nothing on another.

The manifest is the contract. Beyond the schema it records who produced the
bundle and how (`producer`, with the engine, device, precision and torch
version), what the observation was (`source`, including `oriented`, the resize,
the view count and both image sizes), the seed, the token count and the state
width, a digest per artifact, the thresholds the producer was willing to call
agreement, a digest per array and a digest per file.

```json
{
  "schema": "apxinf.pi05.bundle.v1",
  "generated_at": "...",
  "producer": {"package": "apxinf-ref", "version": "...", "engine": "assembled",
               "device": "cuda:0", "precision": "bfloat16", "torch": "..."},
  "source": {"kind": "libero_hdf5", "sample": {}, "oriented": true,
             "resize": "apxinf.processors.ResizeWithPad (PIL BILINEAR, antialiased)",
             "num_views": 2, "image_size": 224, "source_image_size": [128, 128]},
  "seed": 0, "token_count": 46, "state_width": 8,
  "artifacts": {
    "checkpoint":  {"path": "...", "weights": "model.safetensors", "sha256": "..."},
    "tokenizer":   {"path": "...", "sha256": "..."},
    "norm_stats":  {"path": "...", "sha256": "..."}
  },
  "thresholds": {"stage_cosine": 0.999, "final_actions_cosine": 0.999},
  "arrays": {"inputs.npz": {"<name>": {"shape": [], "dtype": "", "sha256": ""}}},
  "files":  {"inputs.npz": {"bytes": 0, "sha256": ""},
             "probe.json": {"bytes": 0, "sha256": ""}}
}
```

`run.log` is written for a person and is deliberately **not** in `files`: it is a
receipt someone may append to, and a digest a receipt can invalidate would refuse
a bundle that is perfectly intact.

The writer keeps three rules, all of them the same rule this codebase applies
elsewhere -- refuse rather than produce something that looks fine. It is
**atomic**, written into a sibling temporary directory and renamed into place, so
a half-written bundle is never a bundle. A **non-empty destination is refused**
unless `--force`, because silently merging two captures is how a directory ends up
with one host's frames and another's actions. And a schema it does not know is
**refused rather than best-effort parsed** -- which includes the older `io.npz`
container, whose format belongs to another project.

### What a replay refuses, and what it only reports

`read_bundle` verifies the digests in `files` and refuses a bundle that does not
match them. A file truncated in transit has to be detectable, and the reader is
the only place it can be detected: a replay of a damaged bundle produces stage
signatures that look exactly like an answer.

A replay also refuses a bundle captured at another **view count, image size or
token count**. Those are not facts about a host but about the question, and the
prefix length and every stage downstream follow them; letting one through would
surface much later as a shape error or, worse, as a cosine that looks like a
precision difference.

Everything else is recorded and changes no exit code:

* **the artifact digests.** The document gains a `bundle.artifacts_match` block
  listing the checkpoint, the tokenizer and the normalization statistics with
  whether the local file matched and both digests. Two hosts using different
  tokenizer files produce different token ids and their stage signatures are not
  comparable -- but that is a fact for the reader, not grounds for a refusal.
* **the token ids**, reported the same way. This is the same judgement: an
  artifact mismatch is a statement about the two hosts' *choices*, while the
  noise check below is a statement about the *input*.
* **the gold cosine**, reported alongside the thresholds the bundle recorded.
  `thresholds` is recorded, never enforced: a bundle states what its producer was
  willing to call agreement, and what a reader does with that is the reader's
  decision.

The one hard stop is the **noise**, and it is checked element by element. Noise is
an input: a replay under a different draw answers a different question while
looking exactly like the same one. `--seed` therefore defaults to the one the
bundle recorded rather than to zero, so the documented replay command replays the
question that was actually asked.

### Closing the loop without the other host

The exchange can be verified on one machine. Capture a bundle, immediately replay
the bundle this code just wrote, and compare: same device, same implementation,
same inputs.

The loop is a test, so that is the check to run: the contract tests finish in
seconds, and the end-to-end capture-and-replay is marked `slow` and takes about
two minutes. The dataset is found through `APXINF_LIBERO_ROOT`, falling back to
`~/.cache/openpi/libero_10`.

```sh
.venv/bin/pytest python/apxinf_ref/tests              # contracts + the loop
.venv/bin/pytest python/apxinf_ref/tests -m "not slow"   # contracts, seconds

# The same loop by hand, which is what the slow test runs:
.venv/bin/python -m apxinf_ref infer --device musa \
    --libero-root ~/.cache/openpi/libero_10 --seed 0 \
    --capture devlocal/pi05-ref-runtime/results/self-bundle \
    --out devlocal/pi05-ref-runtime/results/self-infer.json
.venv/bin/python -m apxinf_ref infer --device musa \
    --bundle devlocal/pi05-ref-runtime/results/self-bundle \
    --out devlocal/pi05-ref-runtime/results/self-replay.json
.venv/bin/python -m apxinf_ref compare \
    devlocal/pi05-ref-runtime/results/self-infer.json \
    devlocal/pi05-ref-runtime/results/self-replay.json --thresholds bf16
```

Anything but bit-identical means capture and replay are not in fact sharing one set
of conventions, which is the only thing the design is for. `compare`'s
`bitwise_equal` is computed over the 256-value sample grid, so the whole-tensor
statement is the four scalars alongside it: every stage must have
`bitwise_equal: true` *and* all four `scalar_drift` values at exactly zero. The
stronger and simpler form is the one the test asserts: the two
`intermediate_signatures` maps must be equal, stage for stage.

Two signals come free and are worth reading: the frames in `inputs.npz` should be
at their **original size**, not at the model's, which is what shows the resize is
really re-run; and the replay document's `gold_cosine` should be `1.0`, because
the `gold_actions` it is measured against are the ones this code just computed.
It lands on `1.0` to within an ulp of float64 -- the measure is a dot product
divided by the product of two norms, not a bitwise comparison.

**The cross-device number needs the other host.** Capture there, copy the
directory over, replay here -- and read the result against [the measured
floors](#the-measured-floors) rather than against zero.

## The measured floors

Two numbers bound what a comparison can conclude, both measured on the M1000 host
and recorded in full in `devlocal/pi05-ref-runtime/reports/anchor-fidelity.md`:

- the **same-device implementation floor** -- this implementation against the
  upstream snapshot, on CPU -- is worst-cosine **0.999748**;
- the **cross-device floor** -- CPU against the M1000, same implementation -- is
  worst-cosine **0.999712**.

They are the same size, and both come from one mechanism: patch embedding
expressed as a `Conv2d` upstream and as a GEMM over the flattened patch in the
engine, differing by ~4e-07 relative in float32, with every bfloat16 rounding
decision downstream landing on a neighbouring value.

The floor is not uniform, and that is what makes the tool usable. Three unrelated
perturbations -- re-implementing the model, moving it to another device, and
changing its arithmetic width -- land in the same range at the same stages, so
past `vision_layer_10` the tower's own bfloat16 rounding dominates.

| stage group | floor (relative L2) | what a disagreement means |
|---|---|---|
| `vision_patch_embed` | ~4e-07 | a real defect, immediately |
| `denoise_step_*` | 2e-04 to 5e-03 | a real defect in the action expert or the solver |
| `vision_layer_{10..26}`, `prefix_v_layer*` | 1e-02 to 2e-02 | almost nothing — this is the BLAS, not the port |

`vision_patch_embed` is bit-identical between bf16 and fp32 (that path runs in
float32 either way), which is why it stays informative. The BF16 gate in
`pi05_bench.rs` is cosine >= 0.999 and the floor alone is 0.9997, so on the deep
vision layers a perfect port and a slightly wrong one are 0.0007 apart. Those
stages should be reported and not relied on.

## Using it in a port

[Model Execution Wiring](model-execution-wiring.md) requires every ledger row to
resolve to "a safe device call, a named correctness scaffold that is still being
replaced, or an explicit blocker". The reference runtime is what makes the
distinction checkable rather than asserted: a stage whose engine value agrees
with the reference is a device call, and a stage that disagrees is a scaffold or
a defect. Read [the measured floors](#the-measured-floors) before reading a
comparison.

## The probe's two known defects

Both were found by building the reference and are fixed in
`examples/pi05_stage_probe.rs` (`--precision bf16|fp8|int8`), which is the probe
to run. `examples/pi05_integrity_probe.rs` is the older FP8-only form of the same
document and still has them:

1. **`prefix_v_layer*` signed uninitialized memory.** `prefix_forward` allocates
   its K/V cache with `prefix_rows + action_horizon` rows and
   `cache::reserve_prefix_bf16`
   (`crates/apxinf-cuda/src/kernels/cache.rs:111`) copies only the first
   `prefix_rows`, but the probe signed the whole buffer. Those signatures were
   not reproducible even between two runs of the engine. The probe now signs only
   the written rows; `compare.py` reports the two stages as `structural`, with
   the diagnosis, against an older document.
2. **The step size did not match the runtime's.** The probe used
   `dt = -1.0 / num_flow_steps`; `denoise_all_steps` uses
   `-flow_start_time / num_flow_steps` (`runtime.rs:598`). They agree while
   `flow_start_time` is 1.0 and diverge silently otherwise.

## Running it against the engine

**Scope note (2026-09-20).** The engine comparison is *not* part of this round's
acceptance criteria; it is deferred to an environment with NVIDIA hardware. The
development machine is aarch64 MUSA — no NVIDIA device, no CUDA toolkit, and no
Orin/Thor reachable from it — so this is a missing task environment rather than
missing implementation work.

What this round delivers instead is everything the comparison needs to be
trustworthy on its first run: the runbook below, the two implementations of the
stage signature pinned to the same literals on both sides (the engine side only
runs on CUDA, so a mismatch there costs a whole run), and the measured floors
that make a result attributable.

The comparison needs a CUDA host. The procedure is written down here rather than
kept in a script, because it is four commands and two of them are the whole
comparison:

Three things must match or the comparison is meaningless rather than noisy: the
**token count** (the engine defaults to 10, and `infer` will not match it unless
you pass the same count), the **view count** (both default to the two-view
`thor_two_view` profile), and the **checkpoint** (a different export is a
different model). `compare` refuses a token-count mismatch outright rather than
reporting it.

One measured fact about that checkpoint: the copies of `pi05_libero_pytorch` on
the M1000 development host hash to `d648dc6c…`, while
`doc/pi05-cuda-regression.md` pins `21b87117…` for the record it was taken
against. They are not the same file, so a comparison made here is not a
reproduction of that record. Nothing in the probe fingerprints the checkpoint --
`--checkpoint`, `APXINF_PI05_CHECKPOINT` and the path recorded in an `infer`
document are what say which one was used.

```sh
# On the CUDA host (Orin/Thor), with the repository checked out, the Rust
# toolchain, the checkpoint, and `pip install -e python/apxinf_ref` in place.

# 1. the engine, on the same fixture the reference uses. The probe writes its
#    document to stdout and its progress to stderr, so the redirect is the file.
cargo run --release -p apxinf-model --features cuda --example pi05_stage_probe -- \
    <checkpoint-dir> --precision bf16 --token-count 10 > engine-bf16.json

# 2. the reference, same device class
python -m apxinf_ref probe --engine assembled --device cuda --fixture zeros \
    --checkpoint <checkpoint-dir> --token-count 10 --out reference-bf16.json

# 3. subtract, stage by stage
python -m apxinf_ref compare reference-bf16.json engine-bf16.json \
    --thresholds bf16 --json phase5-bf16-compare.json

# 4. (optional) the same comparison on a real LIBERO observation, which also
#    produces an action chunk. Both sides must be given the same observation and
#    the same seed, and `infer` records both in its document.
python -m apxinf_ref infer --device cuda --libero-root <libero-root> --seed 0 \
    --out infer-cuda.json
```

Read the result against [the measured floors](#the-measured-floors), not against
zero. Three attributions are already accounted for and should be subtracted
before blaming a kernel:

* the engine folds Gemma's `1 + gamma` into the consuming projections and rounds
  the scale to bfloat16 on the way, which costs up to **1.0e-02 relative L2 on
  `prefix_v_layer17`** on its own (measured on CPU without CUDA; see
  `devlocal/pi05-ref-runtime/scripts/measure_weight_fold_cost.py`). The axis that
  fold runs along has no unit test of its own -- the two that pinned it were
  dropped when this round's test volume was cut -- so an orientation bug that
  produces a plausible matrix is caught by this comparison or not at all;
* the engine stores the time embedding as bfloat16 where the reference keeps it
  in float32, and computes its phase as `TAU / period` against the reference's
  `(1 / period) * 2 * pi`; together these put a floor under how well the
  `denoise_step_*` sequence can ever agree;
* the deep vision layers are the third row of [the floors
  table](#the-measured-floors): a disagreement there is the BLAS, not the port.

`prefix_v_layer{0,17}` come back **structural** rather than pass or fail when the
candidate came from `examples/pi05_integrity_probe.rs`, which signs
uninitialized K/V rows. `examples/pi05_stage_probe.rs` is the probe to run.

## Where the evidence lives

Generated captures, comparison JSON and the one-off debugging scripts are in the
ignored `devlocal/pi05-ref-runtime/` (see [AGENTS.md](../AGENTS.md)). The
findings that a maintainer needs are in
`devlocal/pi05-ref-runtime/reports/anchor-fidelity.md`: the fidelity proof that
the assembled implementation is the same model as the upstream snapshot, the
amplification measurement, the two probe defects, and the measured floors.
