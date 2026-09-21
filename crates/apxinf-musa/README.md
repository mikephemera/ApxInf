# apxinf-musa

The MUSA operator layer: where "which implementation of this operation runs on
MUSA" is decided and recorded.

This round it records one thing: **nothing runs on MUSA yet.** The candidate
registry is empty, every semantic resolves to `deferred`, and a deferred semantic
produces no value at all. The numerical truth for the stages it would have fed
comes from `python/apxinf_ref`, the device-independent eager reference runtime.

## Why there is no fallback implementation here

`doc/model-execution-wiring.md:84-88` is explicit that on an accelerator target a
steady-state host scaffold "is not deliverable optimization debt: it remains
unfinished implementation", and `doc/adding-new-kernels.md:45` requires that
unsupported input "must never silently produce an incorrect result".

A host-side reimplementation of the missing semantics would fail both: it would
be a second transcription of the model to keep correct, and every stage it served
would produce a plausible number that says nothing about MUSA. The deferred path
has no way to return a tensor, which is the strongest available guarantee.

## Layout

```
musa-operator.md    the operator catalog: one entry per semantic, every one a gap
src/
  catalog.rs        the machine-readable catalog, and the reference Spec per semantic
  spec.rs           Spec (what makes two calls the same operation) and Policy
  registry.rs       candidate descriptors and the per-semantic registry
  resolve.rs        the admission chain, and the resolution report
  bin/operator-gaps.rs   prints apxinf.musa.operator-resolution.v1
  tests/            the four promises: doc/catalog agreement, the empty registry,
                    the deferred resolution, and the probe's stage names
```

`src/spec.rs` keeps scale *values* out of `Spec` on purpose. `Spec` decides
whether a candidate is admissible, so it holds shapes, dtypes and scale *kinds*;
the numbers live in the bindings. The CUDA layer draws the same line
(`crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h:37`), and for the same
reason: a tuning cache keyed on a calibration value fragments into one entry per
calibration.

## Checking it

The crate is **not a workspace member**, following `crates/apxinf-cuda-new`. It
is developed against its own entry point and swapped in when it is ready, rather
than replacing a working backend in the same build. That means
`cargo check --workspace` does not cover it; use:

```sh
./crates/apxinf-musa/check.sh
```

which runs the tests and then the report, and fails if any semantic resolves to a
native candidate.

The doc gate earns its keep: on the first compile it caught the catalog document
carrying `Semantic` variant names (`LayerNorm`) where `Semantic::name()` returns
snake_case (`layer_norm`), which would have left every prose entry unfindable
from the code.

## What fills a gap

`doc/adding-new-kernels.md` is the workflow (§4, "Record the selected backend"
and the paragraph after it), and `Exit criterion` in `musa-operator.md` records
per semantic what retires it. The short version:
implement the safe Rust operator, register a candidate for the semantic with its
`supports`/`alignment_satisfied`/`workspace_bytes` predicates, then replay it
against the stage probe and confirm the affected stages move inside the
threshold. A returned implementation is not accepted because it builds or passes
a synthetic test (`doc/porting-workflow.md` §2, "A declared fallback without
replay evidence is a kernel gap").
