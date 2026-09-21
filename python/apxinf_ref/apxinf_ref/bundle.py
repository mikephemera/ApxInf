"""The capture container: ``apxinf.pi05.bundle.v1``.

A bundle is one observation, frozen where it was made and replayable anywhere.
It exists because the stage probe takes patches and token ids directly: a probe
taken on one host says nothing about a probe taken on another unless both were
handed the same inputs. The bundle is that input.

This module is the single place the format is defined. It replaces the read half
that used to live in ``model/pi05/observation.py`` -- a module about turning an
observation into model inputs, which is not the same job as reading a directory.
The format it replaces (the ``io.npz`` container the FlashRT gold producer
wrote) is deliberately **not** supported: this project owns both ends of the
exchange now, and a reader that best-effort parses a foreign container is a
reader that silently compares two different questions.

**Frames are stored oriented but not resized.** The rotation is a property of
the observation -- LIBERO stores frames as the renderer drew them and the policy
wants them the other way up -- so it is resolved at capture time and recorded in
``manifest["source"]["oriented"]`` rather than left for the reader to infer from
which kind of source produced them. The resize is a deterministic function of
the frame and the pinned conventions, so it is *not* resolved here: the replay
re-runs the letterbox on its own hardware, which puts the preprocessing path
inside what the comparison exercises instead of assuming it away.

Three rules the writer keeps, all of them the same rule this codebase applies
elsewhere -- refuse rather than produce something that looks fine:

* **atomic** -- written into a sibling temporary directory and renamed into
  place, so a half-written bundle is never a bundle;
* **a non-empty destination is refused** unless ``force=True``, because silently
  merging two captures is how a directory ends up with one host's frames and
  another's actions;
* **a schema this code does not know is refused**, not best-effort parsed.

``inputs.npz`` records a sha256 per array and ``manifest.json`` a sha256 per
file, and :func:`read_bundle` verifies the file digests: a bundle truncated in
transit is detectable rather than merely wrong. The reader is the only place it
can be detected, so the reader is where it happens.

The two levels are not redundant. The file digest is what makes a corrupted
container loud. The array digests say what the container was *supposed* to hold,
which is what lets a reader localise a difference by hand -- and unlike a
container's bytes they are stable across writers: ``np.savez`` writes an array's
own bytes and stamps a fixed 1980 date into every zip header, so the same arrays
produce the same file byte for byte.

The array holding the answer is named ``gold_actions`` and that name is fixed by
this format. Naming it after the device that produced it is a mistake worth not
repeating: a reader that looks for one name finds it on one host and silently
finds nothing on another.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import __version__
from .device import torch_module
from .model.pi05.observation import (
    PREPROCESSED_VIEW_NAMES,
    VIEW_NAMES,
    ModelInputs,
    Observation,
)
from .model.pi05.tokenizer import sha256_of

if TYPE_CHECKING:  # pragma: no cover - typing only
    # `sources` reads bundles back, so importing it for real would close a cycle.
    # Nothing here needs the class at run time: it is only ever read.
    from .sources import Source

__all__ = [
    "BUNDLE_SCHEMA",
    "INPUTS_NAME",
    "LOG_NAME",
    "MANIFEST_NAME",
    "PROBE_NAME",
    "Bundle",
    "read_bundle",
    "replay_report",
    "write_bundle",
]

#: The container this module reads and writes, and the only one it will.
BUNDLE_SCHEMA = "apxinf.pi05.bundle.v1"

INPUTS_NAME = "inputs.npz"
PROBE_NAME = "probe.json"
MANIFEST_NAME = "manifest.json"
LOG_NAME = "run.log"

#: What a replay cannot do without. ``noise`` is on the list because it is an
#: input rather than a rounding difference: a replay under a different draw
#: answers a different question while looking like the same one.
REQUIRED_ARRAYS = ("base_image", "wrist_image", "state", "prompt", "noise")

#: What a reader may find and does not require. ``preprocessed_*`` are written
#: to localise a divergence to the resize or the patch layout by hand; nothing
#: in the package consumes them yet, which is why they are optional rather than
#: part of what a replay reads.
OPTIONAL_ARRAYS = (
    "formatted_prompt",
    "token_ids",
    "gold_actions",
    "gold_raw_actions",
    "preprocessed_base_rgb",
    "preprocessed_wrist_rgb",
)

@dataclass(frozen=True)
class Bundle:
    """A capture, as read back from disk.

    ``observation`` holds the frames in the orientation the producer recorded,
    not resized. ``probe`` is the producer's own stage-probe document, which is
    what lets a replay be subtracted stage by stage instead of scored on one
    final cosine; it is ``None`` only for a bundle written without one.
    """

    observation: Observation
    noise: np.ndarray
    manifest: dict
    probe: dict | None
    gold_actions: np.ndarray | None
    gold_raw_actions: np.ndarray | None
    token_ids: np.ndarray | None
    formatted_prompt: str | None

    @property
    def oriented(self) -> bool:
        """Whether the frames on disk are already in the policy's orientation."""
        return bool((self.manifest.get("source") or {}).get("oriented"))


def read_bundle(path: str | Path) -> Bundle:
    """Read a bundle directory (or its ``inputs.npz``) into a :class:`Bundle`."""
    directory, inputs = _locate(path)
    manifest = _read_manifest(directory)
    _verify_files(directory, manifest)

    with np.load(inputs, allow_pickle=False) as arrays:
        present = set(arrays.files)
        missing = [name for name in REQUIRED_ARRAYS if name not in present]
        if missing:
            raise ValueError(
                f"{inputs} is missing {', '.join(missing)}; it is not a "
                f"{BUNDLE_SCHEMA} bundle. Every bundle carries at least "
                f"{', '.join(REQUIRED_ARRAYS)}."
            )
        observation = Observation(
            base_image=np.asarray(arrays["base_image"]),
            wrist_image=np.asarray(arrays["wrist_image"]),
            state=np.asarray(arrays["state"], dtype=np.float32).reshape(-1),
            prompt=_as_text(arrays["prompt"]),
            provenance={"bundle": directory.name},
        )
        noise = np.asarray(arrays["noise"], dtype=np.float32)
        gold = _optional(arrays, "gold_actions")
        raw_gold = _optional(arrays, "gold_raw_actions")
        token_ids = _optional(arrays, "token_ids")
        formatted = (
            _as_text(arrays["formatted_prompt"])
            if "formatted_prompt" in present
            else None
        )

    probe = _read_optional_json(directory / PROBE_NAME)
    return Bundle(
        observation=observation,
        noise=noise,
        manifest=manifest,
        probe=probe,
        gold_actions=gold,
        gold_raw_actions=raw_gold,
        token_ids=token_ids,
        formatted_prompt=formatted,
    )


def write_bundle(
    directory: str | Path,
    *,
    observation: Observation,
    inputs: ModelInputs,
    document: Mapping[str, Any],
    actions: Any,
    raw_actions: Any,
    source: "Source",
    artifacts: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    engine: str,
    device: Any,
    precision: str,
    seed: int,
    receipt: Sequence[str] = (),
    force: bool = False,
) -> Path:
    """Write a capture, or refuse and say why.

    ``document`` is the producer's stage-probe document -- the very one written
    to ``--out`` -- and it is stored as ``probe.json``: a capture that recorded a
    probe but no bundle, or a bundle with no standalone probe, would each be a
    way to lose half the result.

    ``engine``, ``device``, ``precision`` and ``seed`` are not derivable from
    anything else here; they are what the manifest's ``producer`` block needs and
    only the caller knows them.

    The rename into place is atomic when the destination does not exist. When
    ``force=True`` replaces an existing directory it is not: the old directory is
    removed first, so a failure between the two leaves no bundle rather than a
    mixed one. Nothing is ever written *into* the destination, which is the
    property that matters.
    """
    target = Path(directory).expanduser()
    _refuse_a_destination_in_use(target, force=force)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f"{target.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    generated_at = _now()

    arrays = _bundle_arrays(
        observation=observation, inputs=inputs, actions=actions, raw_actions=raw_actions
    )
    replaced: Path | None = None
    try:
        staging.mkdir()
        np.savez(staging / INPUTS_NAME, **dict(sorted(arrays.items())))
        _write_json(staging / PROBE_NAME, document)
        (staging / LOG_NAME).write_text(_log(generated_at, receipt))
        _write_json(
            staging / MANIFEST_NAME,
            _manifest(
                generated_at=generated_at,
                arrays=arrays,
                staging=staging,
                observation=observation,
                inputs=inputs,
                source=source,
                artifacts=artifacts,
                thresholds=thresholds,
                engine=engine,
                device=device,
                precision=precision,
                seed=seed,
            ),
        )
        if target.exists():
            # Renamed aside rather than deleted: `os.replace` is rename(2) and
            # cannot put a directory over a non-empty one, and doing this in two
            # steps means a failure between them leaves the previous capture on
            # disk under a temporary name rather than gone.
            replaced = target.parent / f".{target.name}.old-{os.getpid()}-{secrets.token_hex(4)}"
            os.replace(target, replaced)
        os.rename(staging, target)
    except BaseException:
        # Nothing was ever written *into* the destination, so a failure here
        # leaves either the previous capture or no capture, never a mixture.
        shutil.rmtree(staging, ignore_errors=True)
        if replaced is not None and not target.exists():
            os.replace(replaced, target)
        raise
    if replaced is not None:
        shutil.rmtree(replaced, ignore_errors=True)
    return target


def replay_report(
    manifest: Mapping[str, Any],
    *,
    local_artifacts: Mapping[str, Any],
    token_ids: Any | None = None,
) -> dict:
    """What a replay can say about the bundle it was handed, recorded not judged.

    Every entry here is a fact for the reader. A mismatch means the two runs are
    not comparable -- different tokenizer files produce different token ids, and
    their stage signatures are then two different questions -- but that is a
    statement about the pair, not grounds for refusing to run.
    """
    recorded = dict(manifest.get("artifacts") or {})
    report: dict[str, Any] = {"artifacts_match": {}}
    for name in sorted(set(recorded) | set(local_artifacts)):
        theirs = recorded.get(name) or {}
        ours = local_artifacts.get(name) or {}
        producer_digest = theirs.get("sha256")
        local_digest = ours.get("sha256")
        report["artifacts_match"][name] = {
            "recorded": producer_digest,
            "local": local_digest,
            "matched": bool(
                producer_digest and local_digest and producer_digest == local_digest
            ),
        }

    frozen = manifest.get("token_count")
    if token_ids is not None:
        ours = int(np.asarray(token_ids).reshape(-1).size)
        report["token_ids_match"] = None if frozen is None else int(frozen) == ours
        report["token_count"] = {"recorded": frozen, "local": ours}
    report["thresholds"] = dict(manifest.get("thresholds") or {})
    return report


def _locate(path: str | Path) -> tuple[Path, Path]:
    """The bundle directory and its ``inputs.npz``, from either spelling."""
    target = Path(path).expanduser()
    if target.is_dir():
        return target, target / INPUTS_NAME
    if not target.is_file():
        raise FileNotFoundError(f"no bundle at {target}: expected a directory or an {INPUTS_NAME}")
    return target.parent, target


def _read_manifest(directory: Path) -> dict:
    """The manifest, or a refusal naming what this directory actually is."""
    path = directory / MANIFEST_NAME
    if not path.is_file():
        raise ValueError(
            f"{directory} has no {MANIFEST_NAME}; it is not a {BUNDLE_SCHEMA} "
            "bundle. A directory holding only io.npz is the FlashRT gold "
            "standard's format, which this reader does not accept: the bundle "
            "format is defined here and nowhere else."
        )
    manifest = json.loads(path.read_text())
    schema = manifest.get("schema")
    if schema != BUNDLE_SCHEMA:
        raise ValueError(
            f"{path} declares schema {schema!r}; this reader knows {BUNDLE_SCHEMA!r}. "
            "Refusing rather than best-effort parsing: a container whose layout is "
            "not the one this code was written against would produce numbers that "
            "look like an answer."
        )
    return manifest


def _verify_files(directory: Path, manifest: Mapping[str, Any]) -> None:
    """Insist the bundle is the one the manifest describes, byte for byte."""
    files = manifest.get("files")
    if not files:
        raise ValueError(
            f"{directory / MANIFEST_NAME} records no digests, so nothing about this "
            "bundle can be checked. A bundle whose files are not digested cannot be "
            "told apart from one that lost half its bytes in transit."
        )
    for name, recorded in sorted(files.items()):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} is listed in {MANIFEST_NAME} but is not in {directory}"
            )
        actual = sha256_of(path)
        expected = (recorded or {}).get("sha256")
        if actual != expected:
            size = path.stat().st_size
            recorded_size = (recorded or {}).get("bytes")
            # The two failures read differently on purpose: a short file was
            # truncated in transit, a same-length file that hashes differently
            # was edited, and neither is worth replaying.
            because = (
                "it is shorter than the manifest records, so the copy is truncated"
                if recorded_size is not None and size < int(recorded_size)
                else "its bytes differ from the recorded ones, so this copy was modified"
            )
            raise ValueError(
                f"{name} does not match {MANIFEST_NAME}: {because} (recorded sha256 "
                f"{expected}, this copy hashes to {actual}; {size} bytes on disk, "
                f"{recorded_size} recorded). Copy the bundle again rather than "
                "replaying this one."
            )


def _bundle_arrays(
    *, observation: Observation, inputs: ModelInputs, actions: Any, raw_actions: Any
) -> dict[str, np.ndarray]:
    """The arrays a capture stores, in the layout the format documents."""
    frames = _as_frames(observation)
    arrays = {
        VIEW_NAMES[0]: frames[0],
        VIEW_NAMES[1]: frames[1],
        "state": np.asarray(observation.state, dtype=np.float32).reshape(-1),
        "prompt": np.asarray([observation.prompt]),
        "formatted_prompt": np.asarray([inputs.formatted_prompt]),
        "noise": _to_numpy(inputs.noise, np.float32),
        "token_ids": _to_numpy(inputs.token_ids, np.int64).reshape(-1),
        "gold_actions": np.asarray(actions, dtype=np.float32),
        "gold_raw_actions": np.asarray(raw_actions, dtype=np.float32),
    }
    pixels = getattr(inputs, "preprocessed_rgb", None)
    if pixels is not None:
        pixels = np.asarray(pixels, dtype=np.float32)
        if pixels.ndim != 4 or pixels.shape[0] != 2:
            raise ValueError(
                f"preprocessed_rgb must be [views, 3, H, W] with two views, got "
                f"{pixels.shape}"
            )
        for index, name in enumerate(PREPROCESSED_VIEW_NAMES):
            arrays[name] = pixels[index : index + 1]
    return arrays


def _as_frames(observation: Observation) -> tuple[np.ndarray, np.ndarray]:
    """The two camera frames as stored, or a refusal naming what arrived."""
    frames = []
    for name, frame in zip(
        VIEW_NAMES, (observation.base_image, observation.wrist_image), strict=True
    ):
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"{name} must be [H, W, 3], got {array.shape}")
        if array.dtype != np.uint8:
            raise ValueError(
                f"{name} is {array.dtype}, not uint8. A bundle stores the raw "
                "frames the renderer produced; storing a normalised or resized "
                "frame would move the resize out of what the replay exercises."
            )
        frames.append(array)
    return frames[0], frames[1]


def _manifest(
    *,
    generated_at: str,
    arrays: Mapping[str, np.ndarray],
    staging: Path,
    observation: Observation,
    inputs: ModelInputs,
    source: "Source",
    artifacts: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    engine: str,
    device: Any,
    precision: str,
    seed: int,
) -> dict:
    """The manifest, with the file digests of everything already written."""
    return {
        "schema": BUNDLE_SCHEMA,
        "generated_at": generated_at,
        "producer": {
            "package": "apxinf-ref",
            "version": __version__,
            "engine": str(engine),
            "device": str(device),
            "precision": str(precision),
            "torch": str(torch_module().__version__),
        },
        "source": {
            "kind": str(source.kind),
            "sample": (source.provenance or {}).get("sample"),
            "oriented": bool(source.oriented),
            "resize": str(source.resize),
            "num_views": int(np.asarray(observation.images()).shape[0]),
            "image_size": int(inputs.image_size),
            "source_image_size": [int(size) for size in inputs.source_image_size],
        },
        "seed": int(seed),
        "token_count": int(inputs.token_count),
        "state_width": int(inputs.state_width),
        "artifacts": {name: dict(entry) for name, entry in artifacts.items()},
        "thresholds": dict(thresholds),
        "arrays": {INPUTS_NAME: {name: _array_record(array) for name, array in arrays.items()}},
        # `run.log` is deliberately absent: it is a receipt a person may append
        # to, and a digest that a receipt can invalidate would refuse a bundle
        # that is perfectly intact.
        "files": {
            name: _file_record(staging / name) for name in (INPUTS_NAME, PROBE_NAME)
        },
    }


def _array_record(array: np.ndarray) -> dict:
    return {
        "shape": [int(size) for size in array.shape],
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }


def _file_record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256_of(path)}


def _refuse_a_destination_in_use(target: Path, *, force: bool) -> None:
    if not target.exists():
        return
    if not target.is_dir():
        raise FileExistsError(f"{target} is not a directory; refusing to replace it")
    if any(target.iterdir()) and not force:
        raise FileExistsError(
            f"{target} is not empty; refusing to write a capture over it. Silently "
            "merging two captures is how a directory ends up with one host's frames "
            "and another's actions. Pass --force to replace it."
        )


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Sorted keys, so two runs of the same capture produce the same bytes."""
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _read_optional_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _log(generated_at: str, receipt: Sequence[str]) -> str:
    lines = [f"PI0.5 reference capture, written {generated_at}", ""]
    lines.extend(receipt)
    return "\n".join(lines) + "\n"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_numpy(values: Any, dtype) -> np.ndarray:
    """A tensor on whatever device, as a CPU array -- and it may be an array already."""
    if hasattr(values, "detach"):
        values = values.detach().to("cpu").numpy()
    return np.asarray(values, dtype=dtype)


def _as_text(array: np.ndarray) -> str:
    value = np.asarray(array).reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _optional(arrays, name: str) -> np.ndarray | None:
    if name not in arrays.files:
        return None
    dtype = np.int64 if name == "token_ids" else np.float32
    return np.asarray(arrays[name], dtype=dtype)
