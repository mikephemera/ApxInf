"""What an artifact is: the fixture, the stored encoding, and the gate it passes.

This is the half of the tool that needs no device stack -- nothing here imports
torch at module scope, so the writer and its checker load and are tested on a
host with no GPU toolchain.  A test enforces that; :mod:`pi05_gold.capture` is
the half that drives a model.

An artifact is three files: ``inputs.npz`` (the replay fixture -- canonical
inputs, state and flow-matching scalars, and the only file ``--input`` reads),
``debug.npz`` (the reference tensors), and ``manifest.json`` (what they are,
where they came from, which checks they passed).

Writer and reader have to agree on what the stored bytes mean or the manifest's
statistics mean nothing.  :func:`encode_tensor` stores bfloat16 as its raw
``uint16`` bit pattern, since numpy has no bfloat16 type, and everything else
natively; :func:`auditor_values` widens those bits back to float32, keeps a bool
array bool, and narrows the rest to float32 before summing.  :func:`summarize`
is that reader, and it is the very function the generator writes the manifest
with, so a published artifact cannot disagree with its own manifest.

Nothing is published unless it verifies: :func:`write_output` stages the files
beside the destination, reads them back off disk and re-checks every digest and
statistic there, then renames the directory into place -- so a failed run leaves
no half-verified gold behind, and ``--force`` cannot destroy the previous
artifact before its replacement has passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
from pathlib import Path
from typing import Any

import numpy as np


BFLOAT16_BITS, NATIVE = "bfloat16-bits", "native"
EXIT_REFUSED = 3

DEFAULT_VIEWS, DEFAULT_TOKEN_COUNT = 2, 10
DEFAULT_NUM_FLOW_STEPS, DEFAULT_FLOW_START_TIME = 10, 1.0
DEFAULT_ACTION_DIM = 32


class GoldVerificationError(RuntimeError):
    """The artifact failed a check; nothing was published."""


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Digest of a file, read in chunks so a multi-gigabyte checkpoint stays cheap."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device) -> None:
    """Drain the device queue; a no-op on the host."""

    if device.type == "cpu":
        return
    import torch

    module = getattr(torch, device.type, None)
    if module is not None:
        module.synchronize(device)


def host_read(tensor):
    """Move a tensor to the host: the only device->host path in this package.

    The explicit synchronisation is load-bearing.  Without it MUSA hands back a
    buffer the asynchronous copy has not finished writing, or one a later copy
    has already reused, and a run records a value belonging to a different tensor
    (a 522-element all-False mask whose ``sum`` came out as 10, the size of
    another tensor in the same run).  Cloning gives the caller host memory the
    runtime cannot recycle.  It lives here because the statistics this module
    writes describe the bytes this read returns, and a read that raced is exactly
    what the gate below exists to catch.
    """

    if tensor.device.type == "cpu":
        return tensor.detach()
    synchronize(tensor.device)
    host = tensor.detach().to("cpu")
    synchronize(tensor.device)
    return host.contiguous().clone()


def _bf16_bits_to_float32(array: np.ndarray) -> np.ndarray:
    """bfloat16 is the top half of float32, so widening the bits is exact."""

    return (array.astype(np.uint32) << 16).view(np.float32)


def encode_tensor(tensor) -> tuple[np.ndarray, dict[str, str]]:
    """Store a tensor bit-faithfully, as ``(array, {"dtype", "encoding"})``.

    The read goes through :func:`host_read`, so being handed a device tensor is
    safe rather than merely discouraged.  torch is imported here rather than at
    module scope: this module also carries the pure-numpy reader side, which has
    no business needing a device stack.
    """

    import torch

    cpu = host_read(tensor)
    if cpu.dtype == torch.bfloat16:
        return cpu.view(torch.uint16).numpy(), {"dtype": "torch.bfloat16", "encoding": BFLOAT16_BITS}
    native = {torch.float32: "torch.float32", torch.int64: "torch.int64", torch.bool: "torch.bool"}.get(cpu.dtype)
    if native is None:
        raise ValueError(f"no fixture encoding for dtype {cpu.dtype}")
    return cpu.numpy(), {"dtype": native, "encoding": NATIVE}


def decode_tensor(array: np.ndarray, dtype: str, encoding: str) -> np.ndarray:
    """Inverse of :func:`encode_tensor` for a reader that wants the values back."""

    if encoding == NATIVE:
        return array
    if encoding == BFLOAT16_BITS:
        if dtype != "torch.bfloat16":
            raise ValueError(f"encoding {BFLOAT16_BITS!r} does not match dtype {dtype!r}")
        if array.dtype != np.uint16:
            raise ValueError(f"bfloat16 bit patterns must be uint16, got {array.dtype}")
        return _bf16_bits_to_float32(array)
    raise ValueError(f"unknown tensor encoding {encoding!r}")


def auditor_values(array: np.ndarray, *, dtype: str, encoding: str) -> np.ndarray:
    """The value ladder a reader applies to a stored array before summing it.

    bf16 bit patterns widen to float32, a bool array stays bool, and everything
    else narrows to float32 -- which is what makes an int64 array's statistics
    comparable at all.
    """

    if encoding == BFLOAT16_BITS:
        if dtype != "torch.bfloat16":
            raise ValueError(f"encoding {BFLOAT16_BITS!r} does not match dtype {dtype!r}")
        return _bf16_bits_to_float32(array)
    if dtype == "torch.bool":
        return array.astype(np.bool_, copy=False)
    return array.astype(np.float32, copy=False)


def array_digest(array: np.ndarray) -> str:
    """Identity of the bytes an array occupies in its npz file."""

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def tensor_digest(tensor) -> str:
    """Identity of the bytes ``tensor`` would occupy in ``debug.npz``."""

    return array_digest(encode_tensor(tensor)[0])


def sample_indices(size: int, count: int) -> np.ndarray:
    """Evenly spaced element indices: the contract for every decimated sample."""

    if count == 1:
        return np.array([0], dtype=np.int64)
    return (np.arange(count, dtype=np.int64) * (size - 1)) // (count - 1)


def summarize(values: np.ndarray, *, shape, dtype: str, sample_count: int = 0) -> dict[str, Any]:
    """Scalar summary of one stored array, plus a decimated sample when one is asked.

    ``values`` is the flat array a reader sees after :func:`auditor_values`.  An
    external verifier recomputes exactly these expressions from the array in
    ``debug.npz`` -- float64 accumulation for ``sum``/``abs_sum``, float64
    squares for ``l2``, the stored precision for ``max_abs`` -- which is what
    makes the manifest checkable against the artifact.
    """

    flat = np.asarray(values).reshape(-1)
    record: dict[str, Any] = {"elements": int(flat.size), "shape": list(shape), "dtype": dtype}
    if flat.size == 0:
        return {**record, "sum": 0.0, "abs_sum": 0.0, "l2": 0.0, "max_abs": 0.0, "sample": []}
    record.update(
        sum=float(np.sum(flat, dtype=np.float64)),
        abs_sum=float(np.sum(np.abs(flat), dtype=np.float64)),
        l2=float(np.sqrt(np.sum(np.square(flat, dtype=np.float64), dtype=np.float64))),
        max_abs=float(np.max(np.abs(flat))),
    )
    if sample_count <= 0:
        return record
    count = min(sample_count, flat.size)
    return {**record, "sample": flat[sample_indices(flat.size, count)].astype(np.float32).tolist()}


def statistics_from_array(array: np.ndarray, *, dtype: str, encoding: str) -> dict[str, Any]:
    """The statistics the manifest has to carry for a stored array."""

    return summarize(auditor_values(array, dtype=dtype, encoding=encoding).reshape(-1),
                     shape=array.shape, dtype=dtype)


def make_fixture(*, seed: int, views: int, token_count: int, horizon: int,
                 action_dim: int = DEFAULT_ACTION_DIM,
                 num_flow_steps: int = DEFAULT_NUM_FLOW_STEPS) -> dict[str, np.ndarray]:
    """A fresh canonical fixture, reproducible from its seed alone.

    A fixture is data, and the same bytes have to reach the CUDA capture and the
    MUSA replay whichever host wrote them, so everything here is numpy.
    """

    if num_flow_steps <= 0:
        raise ValueError(f"num_flow_steps must be positive, got {num_flow_steps}")
    generator = np.random.default_rng(seed)
    return {
        "images": generator.uniform(-1.0, 1.0, (views, 3, 224, 224)).astype(np.float32),
        "token_ids": generator.integers(0, 257152, token_count, dtype=np.int64),
        "noise": generator.standard_normal((horizon, action_dim), dtype=np.float32),
        # PI0.5 ignores the state in embed_suffix, but sample_actions reads it.
        "state": np.zeros((action_dim,), dtype=np.float32),
        "num_flow_steps": np.asarray(num_flow_steps, dtype=np.int64),
        "dt": np.asarray(-1.0 / num_flow_steps, dtype=np.float32),
        "flow_start_time": np.asarray(DEFAULT_FLOW_START_TIME, dtype=np.float32),
    }


def load_fixture(path: Path) -> dict[str, np.ndarray]:
    """Read a fixture back, rejecting anything the model could not consume."""

    with np.load(path, allow_pickle=False) as data:
        required = ("images", "token_ids", "noise")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"fixture {path} is missing arrays: {', '.join(missing)}")
        arrays = {name: np.asarray(data[name]) for name in required}
        # Optional so artifacts written before the two-file split still replay.
        for name in ("state", "num_flow_steps", "dt", "flow_start_time"):
            if name in data:
                arrays[name] = np.asarray(data[name])
    if arrays["images"].ndim != 4 or arrays["images"].shape[1] != 3:
        raise ValueError(f"images must have shape [views,3,height,width], got {arrays['images'].shape}")
    if arrays["images"].shape[2:] != (224, 224):
        raise ValueError(f"images must be 224x224 canonical tensors, got {arrays['images'].shape}")
    if arrays["token_ids"].ndim != 1:
        raise ValueError(f"token_ids must have shape [tokens], got {arrays['token_ids'].shape}")
    if arrays["noise"].ndim != 2 or arrays["noise"].shape[1] != DEFAULT_ACTION_DIM:
        raise ValueError(f"noise must have shape [horizon,{DEFAULT_ACTION_DIM}], got {arrays['noise'].shape}")
    if not np.isfinite(arrays["images"]).all() or not np.isfinite(arrays["noise"]).all():
        raise ValueError(f"fixture {path} contains non-finite image/noise values")
    for name, dtype in (("images", np.float32), ("noise", np.float32), ("state", np.float32), ("token_ids", np.int64)):
        if name in arrays and arrays[name].dtype != dtype:
            arrays[name] = arrays[name].astype(dtype)
    return arrays


def flow_steps(arrays: dict[str, np.ndarray], override: int | None) -> int:
    """Euler steps: the command line wins, then the fixture, then the default."""

    if override is not None:
        return override
    recorded = arrays.get("num_flow_steps")
    return DEFAULT_NUM_FLOW_STEPS if recorded is None else int(np.asarray(recorded).reshape(-1)[0])


def validate_fixture(arrays: dict[str, np.ndarray], *, max_token_len: int, views: int | None,
                     horizon: int, action_dim: int) -> None:
    """Check the fixture against the checkpoint's own limits before loading a model."""

    if views is not None and arrays["images"].shape[0] != views:
        raise ValueError(f"fixture has {arrays['images'].shape[0]} views but --views={views}")
    if arrays["token_ids"].size == 0 or arrays["token_ids"].size > max_token_len:
        raise ValueError(f"fixture token count must be in 1..{max_token_len}, got {arrays['token_ids'].size}")
    if arrays["noise"].shape != (horizon, DEFAULT_ACTION_DIM):
        raise ValueError(f"fixture noise shape must be {(horizon, DEFAULT_ACTION_DIM)}, got {arrays['noise'].shape}")
    if "state" in arrays and arrays["state"].shape != (action_dim,):
        raise ValueError(f"fixture state shape must be {(action_dim,)}, got {arrays['state'].shape}")


def resolve_inputs(args, action_dim: int, action_horizon: int):
    """Build the fixture and the flow-matching schedule, from disk or from a seed."""

    if args.input is not None:
        path = args.input.expanduser().resolve()
        arrays = load_fixture(path)
        num_steps = flow_steps(arrays, args.num_flow_steps)
        if num_steps <= 0:
            raise SystemExit(f"fixture {path} records a non-positive num_flow_steps: {num_steps}")
        source = {"kind": "npz", "path": str(path), "sha256": sha256(path), "num_flow_steps": num_steps}
    else:
        if args.views <= 0 or args.token_count <= 0:
            raise SystemExit("--views and --token-count must be positive")
        num_steps = args.num_flow_steps if args.num_flow_steps is not None else DEFAULT_NUM_FLOW_STEPS
        arrays = make_fixture(seed=args.seed, views=args.views, token_count=args.token_count,
                              horizon=action_horizon, action_dim=action_dim, num_flow_steps=num_steps)
        source = {"kind": "generated", "seed": args.seed, "views": args.views,
                  "token_count": args.token_count, "num_flow_steps": num_steps}
    if args.num_flow_steps is None:
        arrays["num_flow_steps"] = np.asarray(num_steps, dtype=np.int64)
        arrays["dt"] = np.asarray(-1.0 / num_steps, dtype=np.float32)
        arrays["flow_start_time"] = np.asarray(DEFAULT_FLOW_START_TIME, dtype=np.float32)
    return arrays, num_steps, source


def describe_inputs(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    """The manifest's summary of the fixture it is shipping."""

    return {name: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
            for name, value in arrays.items()}


def att_2d_masks(pad_masks: np.ndarray, att_masks: np.ndarray) -> np.ndarray:
    """The vendored ``make_att_2d_masks`` (``pi0_pytorch.py``), for checking only.

    Kept in numpy so a stored mask can be recomputed from the stored inputs
    without a device.
    """

    cumulative = np.cumsum(att_masks, axis=1)
    return (cumulative[:, None, :] <= cumulative[:, :, None]) & (pad_masks[:, None, :] & pad_masks[:, :, None])


def content_violations(debug: dict[str, np.ndarray], meta: dict[str, dict[str, Any]]) -> list[str]:
    """Structural checks that catch a self-consistent but wrong capture.

    Statistics can only ever agree with the array they came from, so a capture
    that read the wrong bytes passes them.  These are the properties of this
    generator's own boundary: it hands the model all-ones image and language
    masks and pi05 builds ones-only suffix masks, so every pad mask is true; the
    prefix attention mask is all-false and each suffix mask is ``[1, 0, ...]``;
    and every attention mask is the vendor's algebra over those inputs.  They are
    what caught the runs whose mask came back with a handful of stray ``False``
    entries while its own statistics still agreed with it.
    """

    def values(key: str) -> np.ndarray:
        return auditor_values(debug[key], dtype=meta[key]["dtype"], encoding=meta[key]["encoding"])

    # A key whose manifest entry is missing is reported by the caller; skip it here.
    present = {key for key in debug if key in meta}
    violations: list[str] = []
    for key in sorted(present):
        if meta[key]["dtype"] != "torch.bool":
            continue
        # A bool array is one byte per element and only 0/1 are legal; numpy
        # neither normalises on write nor validates on read, so a stray byte
        # survives the round trip and still compares equal as a value.
        stray = np.flatnonzero(debug[key].view(np.uint8).reshape(-1) > 1)
        if stray.size:
            violations.append(f"{key}: {stray.size} bool byte(s) outside 0/1, first at flat index {int(stray[0])}")
    for key in sorted(present):
        if key.endswith("pad_masks") and not values(key).all():
            violations.append(f"{key}: {int((~values(key).astype(bool)).sum())} element(s) False; this fixture asks "
                              "for every image, token and action to be present")
    if "prefix.att_masks" in present and values("prefix.att_masks").any():
        violations.append("prefix.att_masks: pi05 attends to the whole prefix, so the mask must be all False")
    if "prefix.pad_masks" in present and "prefix.position_ids" in present:
        expected = np.cumsum(values("prefix.pad_masks").astype(np.int64), axis=1) - 1
        if not np.array_equal(expected, values("prefix.position_ids").astype(np.int64)):
            violations.append("prefix.position_ids: not cumsum(prefix.pad_masks, dim=1) - 1")
    if {"prefix.pad_masks", "prefix.att_masks", "prefix.att_2d_masks"} <= present:
        expected = att_2d_masks(values("prefix.pad_masks"), values("prefix.att_masks"))
        differing = int(np.count_nonzero(expected != values("prefix.att_2d_masks").astype(bool)))
        if differing:
            violations.append(f"prefix.att_2d_masks: {differing} of {expected.size} entries disagree with the mask "
                              "algebra over prefix.pad_masks and prefix.att_masks")
    steps = sorted({key.split(".")[1] for key in present
                    if key.startswith("step.") and key.endswith(".att_2d_masks")})
    for step in steps:
        suffix_att = values(f"step.{step}.suffix_att_masks")
        if not np.all(suffix_att[..., 0] == 1) or np.any(suffix_att[..., 1:]):
            violations.append(f"step.{step}.suffix_att_masks: pi05 marks the first suffix column and nothing else")
        suffix_pad = values(f"step.{step}.suffix_pad_masks")
        prefix_pad = values("prefix.pad_masks")
        expected = np.concatenate([
            np.broadcast_to(prefix_pad[:, None, :], (suffix_pad.shape[0], suffix_pad.shape[1], prefix_pad.shape[1])),
            att_2d_masks(suffix_pad, suffix_att),
        ], axis=2)
        got = values(f"step.{step}.att_2d_masks").astype(bool)
        differing = int(np.count_nonzero(expected != got))
        if differing:
            violations.append(f"step.{step}.att_2d_masks: {differing} of {got.size} entries disagree with the mask "
                              "algebra over this step's own inputs")
    if "raw_actions" in present:
        actions = np.asarray(debug["raw_actions"])
        if not np.isfinite(actions).all():
            violations.append(f"raw_actions: {int(np.count_nonzero(~np.isfinite(actions)))} non-finite value(s)")
        if actions.ndim != 2 or actions.shape[-1] != DEFAULT_ACTION_DIM:
            violations.append(f"raw_actions: expected a [horizon, {DEFAULT_ACTION_DIM}] array, got {list(actions.shape)}")
    return violations


def verify_staged(staging: Path, *, manifest: dict[str, Any], inputs: dict[str, np.ndarray],
                  debug: dict[str, np.ndarray]) -> list[str]:
    """Re-read the staged files the way a consumer would, and check the manifest against them.

    Reading the artifact back off disk rather than trusting what is still in
    memory is what makes this cover the write path too.
    """

    violations: list[str] = []
    with np.load(staging / "inputs.npz", allow_pickle=False) as document:
        staged_inputs = {name: document[name] for name in document.files}
    for name in sorted(set(staged_inputs) ^ set(inputs)):
        violations.append(f"inputs.npz[{name}]: present in only one of the artifact and the fixture")
    for name, array in sorted(inputs.items()):
        staged = staged_inputs.get(name)
        if staged is None:
            continue
        if (staged.dtype != array.dtype or staged.shape != np.asarray(array).shape
                or array_digest(staged) != array_digest(np.asarray(array))):
            violations.append(f"inputs.npz[{name}]: staged {staged.dtype} {list(staged.shape)} is not what was written")
    try:
        # The artifact has to stay replayable by this tool's own --input path.
        load_fixture(staging / "inputs.npz")
    except (FileNotFoundError, ValueError) as error:
        violations.append(f"inputs.npz: not replayable through --input: {error}")

    with np.load(staging / "debug.npz", allow_pickle=False) as document:
        staged_debug = {name: document[name] for name in document.files}
    for name in sorted(set(staged_debug) ^ set(debug)):
        violations.append(f"debug.npz[{name}]: present in only one of the artifact and the capture")
    meta = manifest["debug"]
    for key in sorted(set(meta) ^ set(staged_debug)):
        violations.append(f"{key}: manifest['debug'] and debug.npz disagree about which arrays exist")
    for key in sorted(staged_debug):
        description = meta.get(key)
        if description is None:
            continue
        array = staged_debug[key]
        if array_digest(array) != description.get("sha256"):
            violations.append(f"{key}: staged bytes differ from the digest the manifest recorded")
            continue
        recorded = manifest["statistics"].get(key)
        if recorded is None:
            continue
        recomputed = statistics_from_array(array, dtype=description["dtype"], encoding=description["encoding"])
        for field in ("elements", "shape", "sum", "abs_sum", "l2", "max_abs"):
            if recomputed[field] != recorded[field]:
                violations.append(f"{key}.{field}: manifest {recorded[field]!r} != array {recomputed[field]!r}")
    for key in sorted(manifest["statistics"]):
        if key not in staged_debug:
            violations.append(f"{key}: manifest statistics without an array in debug.npz")
    for item in manifest.get("stage_arrays", []):
        expected = np.asarray(manifest["stages"][item["stage"]][item["call"]]["sample"], dtype=np.float32)
        if not np.array_equal(staged_debug[item["array"]], expected):
            violations.append(f"{item['array']}: staged sample does not match manifest['stages']")
    return violations + content_violations(staged_debug, meta)


def write_output(output: Path, *, inputs: dict[str, np.ndarray], debug: dict[str, np.ndarray],
                 manifest: dict[str, Any], force: bool) -> None:
    """Write to a staging directory, verify it there, then publish it.

    Nothing appears at ``output`` until the staged bytes have been read back and
    checked against the manifest, so a failed run leaves no half-verified gold
    behind -- and, because ``--force`` only clears the old directory once its
    replacement has passed, it cannot destroy the previous one either.
    """

    if output.is_file():
        raise ValueError(f"output path is a file, not a directory: {output}")
    _refuse_locked_output(output, force)
    staging = output.with_name(f"{output.name}.partial-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    np.savez_compressed(staging / "inputs.npz", **inputs)
    # debug.npz holds large float and bfloat16 payloads that gzip barely touches,
    # so it is written uncompressed to keep the writer's wall clock down.
    np.savez(staging / "debug.npz", **debug)
    violations = verify_staged(staging, manifest=manifest, inputs=inputs, debug=debug)
    if violations:
        (staging / "verify-failure.json").write_text(
            json.dumps({"violations": violations}, indent=2, sort_keys=True) + "\n")
        raise GoldVerificationError(
            f"refusing to publish a gold artifact: {len(violations)} check(s) failed on the staged files in "
            f"{staging}\n  " + "\n  ".join(violations[:10])
            + ("" if len(violations) <= 10 else f"\n  ... and {len(violations) - 10} more")
            + f"\n{output} was left untouched; the full list is in {staging / 'verify-failure.json'}"
        )
    manifest["files"] = {
        name: {"bytes": (staging / name).stat().st_size, "sha256": sha256(staging / name)}
        for name in ("inputs.npz", "debug.npz")
    }
    manifest.setdefault("checks", {}).update({"written_arrays_match_manifest": True,
                                              "content_invariants_hold": True})
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # A manifest that does not parse back is a broken artifact, not a gold one.
    json.loads((staging / "manifest.json").read_text())
    _clear_output(output, force)
    os.replace(staging, output)


def _refuse_locked_output(output: Path, force: bool) -> None:
    """Fail early when a published artifact is already there and ``--force`` was not given."""

    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"output already exists: {output}; pass --force to replace it")


def _clear_output(output: Path, force: bool) -> None:
    """Drop an existing output directory -- only once its replacement has been verified."""

    _refuse_locked_output(output, force)
    if output.exists():
        shutil.rmtree(output)


def timing_report(timed_seconds: list[float], *, warmup_seconds: list[float],
                  recording_seconds: float | None = None) -> dict[str, Any]:
    """Timing block for the un-instrumented path, with its basis spelled out.

    A single reading measures the first call's kernel compilation and allocator
    growth as much as the model, so warmup runs are discarded and percentiles
    taken over the timed ones -- nearest-rank p50/p95, following
    ``scripts/bench_pi05.py``, with mean and population standard deviation beside
    them.  ``recording_seconds`` is the instrumented run that produced the
    artifact and is reported separately, because the stage hooks cost real time
    and folding it in would describe this script rather than the model.  The
    block is printed but kept out of the manifest, so two runs over one fixture
    still produce byte-identical manifest bytes.
    """

    if not timed_seconds:
        raise ValueError("timing needs at least one timed run")
    ordered = sorted(timed_seconds)
    return {
        "basis": f"un-instrumented sample_actions: {len(warmup_seconds)} warmup run(s) discarded, then "
                 f"{len(ordered)} timed run(s); percentiles are nearest-rank over the timed runs",
        "instrumented": False,
        "warmup_runs": len(warmup_seconds),
        "warmup_seconds": [round(value, 4) for value in warmup_seconds],
        "timed_runs": len(ordered),
        "timed_seconds": [round(value, 4) for value in timed_seconds],
        "p50": round(statistics.median(ordered), 4),
        "p95": round(_nearest_rank(ordered, 0.95), 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
        "mean": round(statistics.fmean(ordered), 4),
        "std": round(statistics.pstdev(ordered), 4) if len(ordered) > 1 else 0.0,
        "samples": len(ordered),
        "recording_run_seconds": None if recording_seconds is None else round(recording_seconds, 4),
    }


def format_timing(block: dict[str, Any]) -> str:
    """The human-readable timing block: milliseconds per call, and the Hz that is.

    Each reading prints both ways -- ``min 1316 ms (0.76 Hz)`` -- because the
    same number is a per-chunk latency on one side of this project and a control
    frequency on the other, one reading per line so the number to compare is
    found by eye.  The instrumented recording run is not printed; it goes out
    with the JSON record underneath, which is the copy to parse.
    """

    header = (f"timing: un-instrumented sample_actions, "
              f"{block['warmup_runs']} warmup + {block['timed_runs']} timed runs")
    readings = [f"  {name}   {block[name] * 1000:.0f} ms   ({1.0 / block[name]:.2f} Hz)"
                for name in ("min", "p50", "max")]
    return "\n".join([header, *readings])


def _nearest_rank(ordered: list[float], quantile: float) -> float:
    return ordered[min(len(ordered) - 1, int(quantile * (len(ordered) - 1) + 0.5))]
