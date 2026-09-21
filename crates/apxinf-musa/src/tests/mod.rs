//! The promises this crate makes, held up against the code.
//!
//! The catalog is an empty registry's claim about coverage: which semantics
//! PI0.5's hot path needs on MUSA, and the fact that none of them runs there
//! yet. Four tests keep that claim honest -- the prose and the Rust catalog list
//! the same semantics (and each entry carries its fallback and exit criterion),
//! the document still says the registry is empty, every semantic still resolves
//! to a deferred candidate with the report counting nothing native, and the
//! stage names are the ones the engine's probe emits.

use crate::catalog::{SEMANTICS, reference_spec};
use crate::docs::OPERATOR_DOC;
use crate::registry::total_registered;
use crate::resolve::{DeferredReason, report, resolve};
use crate::spec::Policy;

const OPEN: &str = "<!-- l3-operator:";

/// The document split at its markers: one `(name, body)` per documented semantic.
fn sections(document: &str) -> Vec<(&str, &str)> {
    let mut found = Vec::new();
    let mut rest = document;
    while let Some(start) = rest.find(OPEN) {
        let after = &rest[start + OPEN.len()..];
        let Some(end) = after.find(" -->") else {
            break;
        };
        let body = &after[end..];
        let body_end = body.find(OPEN).unwrap_or(body.len());
        found.push((&after[..end], &body[..body_end]));
        rest = &body[body_end..];
    }
    found
}

#[test]
fn every_semantic_appears_exactly_once() {
    // A semantic that exists in code and not in prose is the failure that
    // matters most for a document like this one: a reader concludes the port
    // covers something it does not. The check runs both ways, so a prose entry
    // with no `Semantic` behind it fails too.
    let documented = sections(OPERATOR_DOC);
    let found: Vec<&str> = documented.iter().map(|(name, _)| *name).collect();
    let expected: Vec<&str> = SEMANTICS.iter().map(|gap| gap.semantic.name()).collect();

    for name in &expected {
        let count = found.iter().filter(|marker| *marker == name).count();
        assert_eq!(
            count, 1,
            "semantic {name:?} appears {count} times in musa-operator.md, expected exactly once"
        );
    }
    for marker in &found {
        assert!(
            expected.contains(marker),
            "musa-operator.md documents {marker:?}, which is not a Semantic in the catalog"
        );
    }

    // `doc/adding-new-kernels.md` §4 requires a fallback per row and an exit
    // criterion for anything standing in for a device implementation, and the
    // gap table is where a reader looks for them.
    for (name, body) in &documented {
        for column in ["| Fallback |", "| Exit criterion |"] {
            assert!(
                body.contains(column),
                "the {name:?} entry in musa-operator.md has no {column} row"
            );
        }
    }
}

#[test]
fn the_document_says_the_registry_is_empty() {
    // The catalog is a claim about coverage, and the claim has to match the
    // registry. If a candidate is ever registered, this test fails and the
    // document has to say so.
    assert_eq!(
        total_registered(),
        0usize,
        "a MUSA candidate is registered but musa-operator.md still says there are none"
    );
    assert!(
        OPERATOR_DOC.contains("_None._"),
        "musa-operator.md no longer states that no MUSA operators are available"
    );
}

#[test]
fn every_semantic_is_deferred_and_the_report_counts_nothing_native() {
    let policy = Policy::default();
    let device = Default::default();

    for gap in SEMANTICS {
        let spec = reference_spec(gap.semantic, 10);
        // The resolver only ever hands the spec to a candidate's predicates, and
        // there are no candidates -- so an empty spec would pass silently while
        // making the catalog's own shape a fiction.
        assert!(
            spec.rows > 0 && spec.columns > 0 && spec.inner > 0 && !spec.key().is_empty(),
            "{} produced an empty reference spec",
            gap.semantic.name()
        );

        let resolution = resolve(gap.semantic, &spec, &policy, &device);
        assert!(
            resolution.is_deferred(),
            "{} resolved to a native candidate, but no MUSA kernel exists",
            gap.semantic.name()
        );
        assert_eq!(
            resolution.reason(),
            Some(&DeferredReason::NoCandidateRegistered),
            "{} was deferred for the wrong reason",
            gap.semantic.name()
        );
    }

    let value = report(10, &policy, &device);
    assert_eq!(
        value["schema"].as_str(),
        Some("apxinf.musa.operator-resolution.v1")
    );
    // Counted through `as_u64` rather than compared against a `Value`: there is
    // no `PartialEq<usize> for Value`, so `value["x"] == SEMANTICS.len()` does
    // not compile.
    assert_eq!(value["candidates_registered"].as_u64(), Some(0));
    assert_eq!(value["semantics_native"].as_u64(), Some(0));
    assert_eq!(
        value["semantics_deferred"].as_u64(),
        Some(SEMANTICS.len() as u64),
        "every semantic must be reported as deferred"
    );
    assert_eq!(value["semantics_total"].as_u64(), Some(SEMANTICS.len() as u64));
    assert_eq!(
        value["semantics"].as_array().map(Vec::len),
        Some(SEMANTICS.len())
    );
}

#[test]
fn stage_names_match_the_stage_probe() {
    // The catalog's value depends on its stage names being the ones the probe
    // emits, because "this semantic is deferred" and "this stage has no MUSA
    // number" have to be the same statement.
    const KNOWN: &[&str] = &[
        "vision_patch_embed",
        "vision_layer_{i}",
        "vision_projected",
        "prefix_v_layer{0,depth-1}",
        "prefix_v_layer*",
        "denoise_step_{s}",
    ];
    for gap in SEMANTICS {
        let stages: Vec<&str> = gap.stage.split(", ").collect();
        for stage in stages {
            assert!(
                KNOWN.contains(&stage),
                "{} names stage {stage:?}, which is not one the probe emits",
                gap.semantic.name()
            );
        }
    }
}
