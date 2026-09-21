"""Where an ``infer`` run's observation came from.

There are two sources today -- a LIBERO demonstration and a captured bundle --
and each carries conventions the other does not: one rotates its frames and
resolves a prompt out of the dataset, the other is handed frames someone else
prepared and a prompt someone else chose. Those conventions used to be spelled
out in two private functions inside ``infer.py``, and capture would have been a
third spelling of them. Three writers of the same conventions is how the capture
path and the replay path drift apart, which is what this module exists to
prevent: one :class:`Source`, produced one way per kind of input.

``oriented`` is the fact worth pausing on. Today "were these frames rotated?" is
answered by looking at which kind of source produced them, which only works
while both sources are local. Once a bundle crosses a machine boundary the
question is about *another host's* behaviour, and the only honest answer is the
one the producer wrote down -- so a bundle's answer comes from its manifest and
not from this module's opinion. It records what the producer did; the
observation handed out is in the policy's orientation either way, because a
consumer whose bundle says the frames still need rotating rotates them here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bundle import read_bundle
from .model.pi05 import preprocess
from .model.pi05.observation import Observation

__all__ = ["Source", "from_bundle", "from_libero"]


@dataclass(frozen=True)
class Source:
    """An observation, plus the conventions that produced it.

    ``extras`` carries what only some sources have: a bundle's frozen noise and
    its recorded answer, and the producer's own probe. ``provenance`` is what the
    infer document records about this run -- the part a reader needs to decide
    whether two documents are comparable, and which no number in the document
    reveals.
    """

    kind: str
    observation: Observation
    oriented: bool
    resize: str
    extras: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


def from_libero(root: str | Path, *, seed: int) -> Source:
    """One deterministic demonstration frame out of LIBERO's HDF5.

    The rotation is applied here and nowhere else: LIBERO stores its frames as
    robosuite rendered them and the policy wants them the other way up, so
    ``Observation.oriented`` -- which owns that conversion, including its reuse
    of ``scripts/libero_observation`` -- is the single place it happens.
    """
    from .model.pi05 import libero

    sample = libero.load_sample(root, seed=seed)
    return Source(
        kind="libero_hdf5",
        observation=sample.as_observation().oriented(),
        oriented=True,
        resize=preprocess.PIL_RESIZE,
        extras={},
        provenance={
            "kind": "libero_hdf5",
            "root": Path(root).expanduser().name,
            "sample": sample.as_json(),
            "runtime": None,
            "resize": preprocess.PIL_RESIZE,
        },
    )


def from_bundle(path: str | Path) -> Source:
    """A capture, read back for replay.

    ``resize`` names the *local* resize, not the producer's: the frames are
    stored before the letterbox precisely so that the replay re-runs it here, on
    this device, with the other device's arithmetic. The producer's own resize
    identity stays where it belongs, in the bundle's manifest.
    """
    bundle = read_bundle(path)
    observation = bundle.observation
    if not bundle.oriented:
        observation = observation.oriented()
    manifest = bundle.manifest
    return Source(
        kind="bundle",
        observation=observation,
        oriented=bundle.oriented,
        resize=preprocess.PIL_RESIZE,
        extras={
            "frozen_noise": bundle.noise,
            "gold_actions": bundle.gold_actions,
            "gold_raw_actions": bundle.gold_raw_actions,
            "token_ids": bundle.token_ids,
            "manifest": manifest,
            "probe": bundle.probe,
        },
        provenance={
            "kind": "bundle",
            "bundle": Path(path).expanduser().name,
            "runtime": (manifest.get("producer") or {}).get("engine"),
            "sample": (manifest.get("source") or {}).get("sample"),
            "resize": preprocess.PIL_RESIZE,
            "producer_resize": (manifest.get("source") or {}).get("resize"),
            "recorded_oriented": bundle.oriented,
        },
    )
