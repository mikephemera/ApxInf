//! MUSA operator layer for ApxInf.
//!
//! # What this is
//!
//! The place where "which implementation of this operation runs on MUSA" is
//! decided and recorded. It follows the shape ApxInf already uses for CUDA
//! operators -- a public L3 semantic, an internal candidate registry, and a
//! resolution step that admits or rejects each candidate against a `Spec` and a
//! `Policy` (`crates/apxinf-cuda-new/native/adapters/gemm/internal.h:63-79`,
//! whose `Implementation` descriptor this mirrors).
//!
//! # What this is not
//!
//! There are no kernels here, and that is the point of the first round. Every
//! semantic resolves to [`Resolution::Deferred`], naming the reason, and the
//! numerical truth for those stages comes from `python/apxinf_ref` -- the
//! device-independent eager reference runtime -- through the
//! `apxinf.pi05.stage-probe.v1` comparison.
//!
//! A host-side reimplementation of the missing semantics was considered and
//! rejected. `doc/model-execution-wiring.md:118-119` is explicit that on an
//! accelerator target a steady-state host scaffold "remains unfinished
//! implementation", and a second
//! transcription of the model would be a second thing to keep correct. The
//! reference runtime already answers "what is the right number"; this crate
//! answers "which implementation produced it", and keeping those separate stops
//! either from quietly papering over the other.
//!
//! # Reading the catalog
//!
//! [`catalog::SEMANTICS`] is the machine-readable form and
//! [`crate::docs::OPERATOR_DOC`] is the prose one; a test checks they agree, the
//! same way `crates/apxinf-cuda-new/src/ops/tests/operator_doc.rs` does for the
//! CUDA catalog.

pub mod catalog;
#[cfg(test)]
mod tests;
pub mod registry;
pub mod resolve;
pub mod spec;

pub use catalog::{OperatorGap, SEMANTICS};
pub use registry::{Candidate, Registry};
pub use resolve::{Resolution, resolve};
pub use spec::{Policy, Spec};

/// The catalog document, embedded so the doc gate needs no filesystem access.
pub mod docs {
    /// `musa-operator.md`, the human-readable operator catalog.
    pub const OPERATOR_DOC: &str = include_str!("../musa-operator.md");
}
