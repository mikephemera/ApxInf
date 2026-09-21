//! The candidate registry.
//!
//! A candidate is one way to compute one semantic. The descriptor mirrors the
//! CUDA layer's `Implementation`
//! (`crates/apxinf-cuda-new/native/adapters/gemm/internal.h:63-79`) where the
//! field has a meaning without a C++ ABI, so that a MUSA candidate added later
//! is described the same way a CUDA one is -- and so that the admission
//! questions (`graph_safe`, `deterministic`, alignment, workspace) are asked in
//! the same order and with the same names.
//!
//! The mirror is deliberately not complete. `Implementation` also carries
//! `fallback` (`internal.h:71`), which the CUDA layer enforces as a hard
//! invariant -- a semantic must register exactly one fallback candidate
//! (`candidates.cpp:231-239`, mirrored for attention in
//! `adapters/attention/candidates.cpp`). [`Candidate`] has no such field because
//! this round registers nothing at all: with an empty registry there is no
//! fallback to designate, and adding the field before the first candidate would
//! record a promise with no way to keep it.
//!
//! **No candidate is registered in this round, and that is a deliberate state
//! rather than an unfinished one.** The registry being empty is what
//! [`crate::resolve`] reports on; a stand-in candidate that produced
//! approximately-right numbers would turn every downstream comparison into a
//! measurement of the stand-in.

use crate::spec::Spec;

/// One implementation of one semantic.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    /// Which provider the implementation belongs to; providers are grouped so a
    /// whole family can be excluded at once.
    pub provider_id: u32,
    /// Distinguishes implementations within a provider.
    pub implementation_id: u32,
    /// The name reported in errors and in the resolution report.
    pub name: &'static str,
    /// One-bit-per-feature mask the device must satisfy.
    pub required_device_features: u64,
    /// Safe to run inside a captured graph: no allocation, no synchronisation,
    /// no host callback.
    pub graph_safe: bool,
    /// Bit-identical between two runs with the same inputs.
    pub deterministic: bool,
    /// Whether this candidate can compute this exact spec.
    pub supports: fn(&Spec) -> bool,
    /// Whether the caller's pointers satisfy this candidate's alignment rules.
    pub alignment_satisfied: fn(&Spec) -> bool,
    /// Workspace the candidate needs, in bytes.
    pub workspace_bytes: fn(&Spec) -> usize,
}

/// A per-semantic candidate list.
///
/// Held as a plain slice per semantic, the way the CUDA layer holds one
/// `apxinf::framework::Registry<Implementation>`
/// (`crates/apxinf-cuda-new/native/framework/registry.h:11`) per `Semantic`, its
/// entries at `crates/apxinf-cuda-new/native/adapters/gemm/candidates.cpp:151`.
/// There is no dynamic registration API here or there: the set of
/// implementations is a property of the build, and letting it be mutated at run
/// time is how "this candidate silently did not run" becomes possible.
pub struct Registry {
    semantic: crate::catalog::Semantic,
    candidates: &'static [Candidate],
}

impl Registry {
    pub const fn new(semantic: crate::catalog::Semantic, candidates: &'static [Candidate]) -> Self {
        Self {
            semantic,
            candidates,
        }
    }

    pub fn semantic(&self) -> crate::catalog::Semantic {
        self.semantic
    }

    pub fn candidates(&self) -> &'static [Candidate] {
        self.candidates
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }
}

/// The candidates registered for a semantic.
///
/// Every arm returns the empty slice today. The match is written out rather than
/// collapsed to `&[]` so that filling one in is a local edit, and so that
/// deleting an arm is visibly a decision rather than a cleanup.
pub fn registry(semantic: crate::catalog::Semantic) -> Registry {
    use crate::catalog::Semantic::*;
    match semantic {
        LayerNorm
        | RmsNorm
        | AdaptiveRmsNorm
        | Gemm
        | GemmGeglu
        | SplitQkvBias
        | SplitQkvApplyRope
        | ApplyQueryWriteKv
        | MultiHeadAttention
        | MultiQueryAttention
        | BiasResidual
        | BiasResidualRmsNorm
        | BiasResidualLayerNorm
        | AdaptiveGateResidualRmsNorm
        | BiasGelu
        | AddPosition
        | EmbeddingLookup
        | ConcatRows
        | EulerUpdate => Registry::new(semantic, &[]),
    }
}

/// How many candidates are registered across the whole catalog.
pub fn total_registered() -> usize {
    crate::catalog::SEMANTICS
        .iter()
        .map(|gap| registry(gap.semantic).len())
        .sum()
}
