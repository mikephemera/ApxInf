"""Diff two ``apxinf.pi05.stage-probe.v1`` documents.

The stage probe keeps six numbers and 256 sampled values per stage, not the
tensors themselves, so this is a comparison of summaries: whole-tensor scalars
(``sum``, ``abs_checksum``, ``l2``, ``max_abs``) plus a fixed grid of samples.
That is enough to localise a divergence to a stage, which is the question the
tool exists to answer; it is not enough to explain one, so a stage that fails
should be re-run with tensors kept.

Thresholds come from ``pi05_bench.rs:87-108`` so that a stage verdict means the
same thing here as it does in the engine's own bench: the same per-dtype cosine
floor, relative-L2 ceiling and, for INT8, absolute ceiling.

The bench's *other* gate -- ``EAGER_GRAPH_MIN_COSINE`` and its per-dtype
``eager_graph_max_abs`` -- is deliberately not here. That one compares two runs
of a single implementation, eager against a captured graph, and the bench knows
that is what it is looking at because it made both runs. This module is handed
two documents of unknown provenance: it cannot check the premise, and a gate the
caller asserts but the tool cannot verify is worse than no gate at all. What a
comparison needs in order to make that claim now travels with the data -- a
bundle's manifest records the producer and the device -- so the evidence has a
home it did not have before.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["THRESHOLDS", "compare_documents", "render_table", "stage_metrics"]

#: ``pi05_bench.rs:87-108`` ``Dtype::thresholds``.
THRESHOLDS = {
    "bf16": {
        "min_cosine": 0.999,
        "max_relative_l2": 0.05,
        "max_abs": None,
    },
    "fp8": {
        "min_cosine": 0.997,
        "max_relative_l2": 0.10,
        "max_abs": None,
    },
    "int8": {
        "min_cosine": 0.995,
        "max_relative_l2": 0.10,
        "max_abs": 0.125,
    },
}


def _finite(values) -> bool:
    import math

    return all(math.isfinite(float(value)) for value in values)


def stage_metrics(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict:
    """Sample-based error metrics, in the vocabulary of ``ErrorMetrics::measure``
    (``crates/apxinf-model/examples/pi05_bench.rs``).

    The engine's signature is ``measure(actual, expected)`` and it is called as
    ``measure(&captured, &expected)`` -- so ``expected`` is the *reference* and
    ``actual`` is the candidate under test. The names here are kept so that the
    two stay obviously the same function: ``expected`` is the reference, and
    ``relative_l2`` is normalised by the reference's norm, not the candidate's.
    Normalising by the candidate would be a plausible-looking metric that
    changes a verdict near the threshold.
    """
    import math

    expected = [float(value) for value in reference["sample"]]
    actual = [float(value) for value in candidate["sample"]]
    if len(expected) != len(actual):
        raise ValueError("sample arrays have different lengths")

    if not _finite(expected) or not _finite(actual):
        raise ValueError("stage comparison contains a non-finite value")

    n = len(expected)
    squared_error = sum((a - e) ** 2 for a, e in zip(actual, expected))
    expected_squared = sum(e * e for e in expected)
    actual_squared = sum(a * a for a in actual)
    denominator = math.sqrt(actual_squared) * math.sqrt(expected_squared)
    if denominator == 0.0:
        cosine = 1.0 if actual_squared == expected_squared else 0.0
    else:
        cosine = sum(a * e for a, e in zip(actual, expected)) / denominator

    return {
        "cosine": cosine,
        "max_abs": max(abs(a - e) for a, e in zip(actual, expected)),
        "mean_abs": sum(abs(a - e) for a, e in zip(actual, expected)) / n,
        "rmse": math.sqrt(squared_error / n),
        "relative_l2": (
            math.sqrt(squared_error / expected_squared) if expected_squared else (
                0.0 if squared_error == 0.0 else float("inf")
            )
        ),
        "bitwise_equal": all(a == e for a, e in zip(actual, expected)),
    }


def _scalar_drift(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict:
    """Relative drift of the whole-tensor scalars, which cover every element.

    These four are the only fields computed over every element rather than over
    the 256-value sample grid, which is why a self-comparison reads them: all
    four at zero plus a bitwise-equal sample grid is as close to "the same
    numbers" as a probe document can state.

    Normalised by the *reference*, the convention ``stage_metrics`` was corrected
    to for the same reason it states: the engine's
    ``ErrorMetrics::measure(actual, expected)`` names the reference ``expected``,
    and normalising by the candidate is a plausible-looking metric that changes
    near the threshold. It moves no verdict here -- this is the diagnostic path --
    but two conventions in one file is how the wrong one survives.
    """
    drift = {}
    for field in ("sum", "abs_checksum", "l2", "max_abs"):
        expected, actual = float(reference[field]), float(candidate[field])
        if expected == 0.0:
            drift[field] = 0.0 if actual == 0.0 else float("inf")
        else:
            drift[field] = abs(actual - expected) / abs(expected)
    return drift


def _structural_diagnosis(name: str, reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    a, b = int(reference["elements"]), int(candidate["elements"])
    if a == b:
        return ""
    if name.startswith("prefix_v_layer"):
        tail = abs(a - b)
        return (
            f"element counts differ by {tail} (reference {a}, candidate {b}). This is the "
            "engine's reserved K/V tail: `prefix_forward` allocates "
            "`prefix_rows + action_horizon` rows and `cache::reserve_prefix_bf16` "
            "(`crates/apxinf-cuda/src/kernels/cache.rs:111`) copies only the first "
            "`prefix_rows`, so the probe's signature covers uninitialized device memory. "
            "The sample grids of the two documents therefore do not line up and this "
            "stage cannot be compared until the engine dumps only written rows."
        )
    return (
        f"element counts differ (reference {a}, candidate {b}); the two sample grids do "
        "not line up and the stage was not compared numerically."
    )


def compare_documents(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    thresholds: str = "bf16",
    report: str | None = None,
) -> dict:
    """Compare two probe documents stage by stage."""
    if reference.get("schema") != candidate.get("schema"):
        raise ValueError(
            f"probe schemas differ: {reference.get('schema')!r} vs {candidate.get('schema')!r}"
        )
    if reference.get("schema") != "apxinf.pi05.stage-probe.v1":
        raise ValueError(f"unsupported probe schema {reference.get('schema')!r}")

    # The token count changes the prefix length, hence every downstream stage.
    # Two documents taken at different counts are different questions, not a
    # noisy reading of the same one, so this refuses rather than reporting the
    # pair and letting a reader decide.
    reference_tokens = reference.get("token_count")
    candidate_tokens = candidate.get("token_count")
    if (
        reference_tokens is not None
        and candidate_tokens is not None
        and reference_tokens != candidate_tokens
    ):
        raise ValueError(
            f"token counts differ: reference {reference_tokens}, candidate "
            f"{candidate_tokens}. The prefix length, and therefore every stage, "
            "depends on it; re-run one side so both use the same count."
        )

    limits = THRESHOLDS[thresholds]
    reference_stages = reference["intermediate_signatures"]
    candidate_stages = candidate["intermediate_signatures"]

    stages = {}
    missing_in_candidate = sorted(set(reference_stages) - set(candidate_stages))
    missing_in_reference = sorted(set(candidate_stages) - set(reference_stages))

    for name in sorted(set(reference_stages) & set(candidate_stages)):
        a, b = reference_stages[name], candidate_stages[name]
        entry: dict = {
            "elements_reference": int(a["elements"]),
            "elements_candidate": int(b["elements"]),
        }
        if int(a["elements"]) != int(b["elements"]):
            entry["status"] = "structural"
            entry["diagnosis"] = _structural_diagnosis(name, a, b)
            stages[name] = entry
            continue

        metrics = stage_metrics(a, b)
        drift = _scalar_drift(a, b)
        passed = (
            metrics["cosine"] >= limits["min_cosine"]
            and metrics["relative_l2"] <= limits["max_relative_l2"]
            and (limits["max_abs"] is None or metrics["max_abs"] <= limits["max_abs"])
        )
        entry.update(metrics)
        entry["scalar_drift"] = drift
        entry["status"] = "pass" if passed else "fail"
        stages[name] = entry

    failures = [name for name, entry in stages.items() if entry["status"] == "fail"]
    structural = [name for name, entry in stages.items() if entry["status"] == "structural"]
    worst = sorted(
        (name for name, entry in stages.items() if entry["status"] != "structural"),
        key=lambda name: stages[name]["cosine"],
    )[:5]

    return {
        "schema": "apxinf.pi05.stage-compare.v1",
        "report": report,
        "thresholds": {"name": thresholds, **limits},
        "passed": not failures and not missing_in_candidate and not missing_in_reference,
        "token_count_reference": reference.get("token_count"),
        "token_count_candidate": candidate.get("token_count"),
        "stages": stages,
        "failures": failures,
        "structural_stages": structural,
        "missing_in_candidate": missing_in_candidate,
        "missing_in_reference": missing_in_reference,
        "weakest_stages": worst,
    }


def render_table(comparison: Mapping[str, Any], reference_name: str, candidate_name: str) -> str:
    """A human-readable rendering of one comparison."""
    lines = []
    limits = comparison["thresholds"]
    lines.append(f"reference: {reference_name}")
    lines.append(f"candidate: {candidate_name}")
    lines.append(
        f"thresholds ({limits['name']}): cosine >= {limits['min_cosine']}, "
        f"relative_l2 <= {limits['max_relative_l2']}"
        + (f", max_abs <= {limits['max_abs']}" if limits["max_abs"] is not None else "")
    )
    lines.append("")
    lines.append(f"{'stage':<26} {'elements':>10} {'cosine':>12} {'rel_l2':>10} {'max_abs':>11}  status")
    lines.append("-" * 82)
    for name, entry in comparison["stages"].items():
        if entry["status"] == "structural":
            lines.append(
                f"{name:<26} {'-':>10} {'-':>12} {'-':>10} {'-':>11}  structural"
            )
            continue
        lines.append(
            f"{name:<26} {entry['elements_reference']:>10} "
            f"{entry['cosine']:>12.9f} {entry['relative_l2']:>10.3e} "
            f"{entry['max_abs']:>11.4e}  {entry['status']}"
        )

    for name in comparison["structural_stages"]:
        lines.append("")
        lines.append(f"[structural] {name}")
        lines.append("  " + comparison["stages"][name]["diagnosis"])

    if comparison["missing_in_candidate"]:
        lines.append("")
        lines.append(f"missing in candidate: {', '.join(comparison['missing_in_candidate'])}")
    if comparison["missing_in_reference"]:
        lines.append("")
        lines.append(f"missing in reference: {', '.join(comparison['missing_in_reference'])}")

    lines.append("")
    if comparison["passed"]:
        lines.append("VERDICT: every comparable stage is within tolerance.")
    else:
        lines.append(
            f"VERDICT: {len(comparison['failures'])} stage(s) out of tolerance"
            + (
                f"; weakest: {', '.join(comparison['weakest_stages'])}"
                if comparison["weakest_stages"]
                else ""
            )
        )
    return "\n".join(lines)
