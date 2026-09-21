//! Which candidate serves a call, and what happens when none does.
//!
//! The admission chain is the CUDA layer's
//! (`crates/apxinf-cuda-new/native/adapters/gemm/candidates.cpp:250-295`; the
//! attention adapter walks the same one at
//! `adapters/attention/candidates.cpp:200-230`), in its
//! order: device features, then the contract, then alignment, then the policy's
//! graph-safety and determinism requirements, then the workspace budget. Each
//! rejection is kept with its reason rather than collapsed into a bool, because
//! "no candidate matched the shape" and "the shape matched but no candidate can
//! be captured" call for different work.
//!
//! When the chain admits nothing, the result is [`Resolution::Deferred`] and
//! **no value is produced**. That is the whole design: `doc/adding-new-kernels.md:45`
//! requires that unsupported input "must never silently produce an incorrect
//! result", and the strongest way to honour that is for the deferred path to
//! have no way to return a tensor.

use crate::catalog::{self, Semantic};
use crate::registry::{Candidate, registry};
use crate::spec::{Policy, Spec};

/// Why a call could not be served.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeferredReason {
    /// The registry holds nothing for this semantic. The expected state for
    /// every semantic in this round.
    NoCandidateRegistered,
    /// Candidates exist but none accepts this spec.
    ContractMismatch,
    /// A candidate accepted the spec but the caller's alignment does not satisfy
    /// it.
    AlignmentMismatch,
    /// A candidate accepted the spec but the policy rejects it -- not graph
    /// safe, not deterministic, or over the workspace budget.
    PolicyRejected,
    /// The device lacks a feature every candidate requires.
    UnsupportedDevice,
}

impl DeferredReason {
    pub fn name(&self) -> &'static str {
        match self {
            Self::NoCandidateRegistered => "no_candidate_registered",
            Self::ContractMismatch => "contract_mismatch",
            Self::AlignmentMismatch => "alignment_mismatch",
            Self::PolicyRejected => "policy_rejected",
            Self::UnsupportedDevice => "unsupported_device",
        }
    }
}

/// The outcome of resolving one call.
#[derive(Clone, Debug)]
pub enum Resolution {
    /// A candidate will compute it.
    Native {
        candidate: &'static Candidate,
        workspace_bytes: usize,
    },
    /// Nothing will compute it here. The stage's truth comes from
    /// `python/apxinf_ref`; see [`crate::docs::OPERATOR_DOC`].
    Deferred {
        semantic: Semantic,
        reason: DeferredReason,
        /// `(candidate name, why it was rejected)` for every candidate that got
        /// far enough to be rejected for a reason more specific than "there are
        /// none".
        rejected: Vec<(&'static str, &'static str)>,
    },
}

impl Resolution {
    pub fn is_native(&self) -> bool {
        matches!(self, Self::Native { .. })
    }

    pub fn is_deferred(&self) -> bool {
        matches!(self, Self::Deferred { .. })
    }

    pub fn reason(&self) -> Option<&DeferredReason> {
        match self {
            Self::Native { .. } => None,
            Self::Deferred { reason, .. } => Some(reason),
        }
    }
}

/// The device capabilities this build can target.
///
/// A placeholder with one field, because nothing consumes it yet. It exists so
/// that adding MUSA feature detection later changes a struct rather than a
/// function signature.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceCaps {
    pub features: u64,
}

/// Resolve a call against the registry, the spec and the policy.
pub fn resolve(semantic: Semantic, spec: &Spec, policy: &Policy, device: &DeviceCaps) -> Resolution {
    let registry = registry(semantic);
    let mut rejected: Vec<(&'static str, &'static str)> = Vec::new();
    let mut saw_contract_match = false;
    let mut saw_alignment_match = false;
    let mut saw_device_match = false;

    for candidate in registry.candidates() {
        if candidate.required_device_features & !device.features != 0 {
            rejected.push((candidate.name, "device lacks a required feature"));
            continue;
        }
        saw_device_match = true;

        if !(candidate.supports)(spec) {
            rejected.push((candidate.name, "contract mismatch"));
            continue;
        }
        saw_contract_match = true;

        if !(candidate.alignment_satisfied)(spec) {
            rejected.push((candidate.name, "alignment mismatch"));
            continue;
        }
        saw_alignment_match = true;

        if policy.graph_safe && !candidate.graph_safe {
            rejected.push((candidate.name, "not graph safe"));
            continue;
        }
        if policy.deterministic && !candidate.deterministic {
            rejected.push((candidate.name, "not deterministic"));
            continue;
        }
        let required = (candidate.workspace_bytes)(spec);
        if required > policy.workspace_limit {
            rejected.push((candidate.name, "workspace budget exceeded"));
            continue;
        }

        return Resolution::Native {
            candidate,
            workspace_bytes: required,
        };
    }

    let reason = if registry.is_empty() {
        DeferredReason::NoCandidateRegistered
    } else if !saw_device_match {
        DeferredReason::UnsupportedDevice
    } else if !saw_contract_match {
        DeferredReason::ContractMismatch
    } else if !saw_alignment_match {
        DeferredReason::AlignmentMismatch
    } else {
        DeferredReason::PolicyRejected
    };

    Resolution::Deferred {
        semantic,
        reason,
        rejected,
    }
}

/// Resolve every semantic in the catalog and emit the resolution report.
///
/// The report is the machine-readable half of the deliverable: it says, per
/// semantic, that no MUSA implementation exists and therefore that no MUSA
/// number exists for the stages that semantic feeds. `python/apxinf_ref`'s
/// `compare` reads the same stage names, so the two documents name the same
/// holes.
pub fn report(token_count: usize, policy: &Policy, device: &DeviceCaps) -> serde_json::Value {
    let entries: Vec<serde_json::Value> = catalog::SEMANTICS
        .iter()
        .map(|gap| {
            let spec = catalog::reference_spec(gap.semantic, token_count);
            let resolution = resolve(gap.semantic, &spec, policy, device);
            let (status, reason, rejected) = match &resolution {
                Resolution::Native { candidate, .. } => {
                    (String::from("native"), String::new(), serde_json::json!([candidate.name]))
                }
                Resolution::Deferred {
                    reason, rejected, ..
                } => (
                    String::from("deferred"),
                    String::from(reason.name()),
                    serde_json::Value::Array(
                        rejected
                            .iter()
                            .map(|(name, why)| serde_json::json!({ "candidate": name, "rejected": why }))
                            .collect(),
                    ),
                ),
            };
            serde_json::json!({
                "semantic": gap.semantic.name(),
                "status": status,
                "reason": reason,
                "rejected": rejected,
                "spec": spec.key(),
                "stage": gap.stage,
                "math": gap.math,
                "contract": gap.contract,
                "frequency": gap.frequency,
                "importance": gap.importance.name(),
                "fallback": gap.fallback.name(),
                "exit_criterion": gap.exit_criterion,
                "legacy_reference": gap.semantic.legacy_reference(),
            })
        })
        .collect();

    let registered = crate::registry::total_registered();
    let native = entries
        .iter()
        .filter(|entry| entry["status"] == "native")
        .count();

    serde_json::json!({
        "schema": "apxinf.musa.operator-resolution.v1",
        "token_count": token_count,
        "candidates_registered": registered,
        "semantics_total": entries.len(),
        "semantics_native": native,
        "semantics_deferred": entries.len() - native,
        "note": "Deferred semantics produce no value. Their stages have no MUSA number; the numerical truth for them comes from python/apxinf_ref, and a stage-probe comparison against this engine does not cover them.",
        "semantics": entries,
    })
}
