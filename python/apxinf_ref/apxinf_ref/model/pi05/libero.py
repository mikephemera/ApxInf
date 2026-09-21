"""Read one frame of a LIBERO demonstration out of its HDF5 dataset.

LIBERO's demonstrations are the only real observations this project can score a
reference run against without standing up the simulator, and the HDF5 they ship
is a verbatim recording rather than a prepared input — so this module's whole job
is to say what the file holds and hand it on unchanged.

Three things about that file are load-bearing and none of them are guessed:

**Frame orientation.** ``create_dataset.py`` stores ``obs["agentview_image"]``
and ``obs["robot0_e_in_hand_image"]`` exactly as robosuite rendered them
(``scripts/create_dataset.py:220-221`` at the pinned LIBERO revision), which is
the OpenGL convention: LIBERO's *own* code flips them before a human looks at one
(``libero/utils/video_utils.py:36``, ``benchmark_scripts/render_single_task.py:33``).
A checkpoint trained on openpi's LIBERO data expects the flipped frame, which is
why the returned images are passed through
``preprocess.orient_libero_images`` -- the same 180-degree rotation the live
simulator path uses -- and not written out raw. Feeding them unflipped produces a
plausible action chunk for an upside-down scene, which is the kind of error that
shows up as a bad success rate and nothing else.

**Proprioception.** The stored ``ee_states`` is ``ee_pos`` followed by
``quat2axisangle(ee_quat)`` (``scripts/create_dataset.py:206-207``), which is
exactly the layout ``scripts/libero_observation.libero_state`` builds for the live
simulator: ``eef_pos(3) + axis_angle(3) + both gripper joints(2)``. The two
sources therefore agree by construction, and
``tests/test_contract.py::test_the_two_libero_sources_build_the_same_state_vector``
pins that they do. Collapsing the gripper to one joint is what a *different* consumer
wants; see ``preprocess.finger_joints_for_state_width``.

**The prompt** is a JSON blob in the dataset-level ``problem_info`` attribute, not
a per-demo string. Falls back to a filename derived task, because the LIBERO
distribution names each file after the instruction it demonstrates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["LiberoHdf5Sample", "load_sample"]

#: Camera datasets, in the order they are tried. ``agentview`` is the fixed
#: third-person camera, ``eye_in_hand`` the wrist camera.
_BASE_CAMERA_NAMES = ("agentview_rgb", "agentview_image")
_WRIST_CAMERA_NAMES = (
    "eye_in_hand_rgb",
    "robot0_eye_in_hand_image",
    "robot0_eye_in_hand_rgb",
)

#: Proprioception layouts, in the order they are tried: ``(position, orientation,
#: gripper)``. The orientation entry is always axis-angle, whether the file stores
#: it as ``ee_ori`` or as a quaternion in ``robot0_eef_quat``. The HDF5 files the
#: LIBERO distribution ships use the first.
_STATE_LAYOUTS = (
    ("ee_pos", "ee_ori", "gripper_states"),
    ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
    ("ee_pos", "ee_ori", "robot0_gripper_qpos"),
)

SUFFIXES = (".hdf5", ".h5")

#: ``data.attrs["problem_info"]`` -- a JSON string carrying the instruction.
_PROBLEM_INFO = "problem_info"


@dataclass(frozen=True)
class LiberoHdf5Sample:
    """One demonstration frame, as stored, plus where it came from."""

    base_image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray
    prompt: str
    #: Host-independent receipt: a relative path, the demo group and the frame
    #: index. Absolute paths are not recorded, so a capture can be published.
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        """The sample's identifying facts, for a run document."""
        return {
            "provenance": dict(self.provenance),
            "prompt": self.prompt,
            "state": [float(value) for value in self.state],
            "state_dim": int(self.state.size),
            "image_shape": list(self.base_image.shape),
            "wrist_image_shape": list(self.wrist_image.shape),
        }

    def as_observation(self):
        """This sample as an :class:`~.observation.Observation`.

        Not yet rotated: the frames are handed over exactly as the file stores
        them, and the caller decides. See the module docstring for why they do
        need rotating, and ``observation.assemble`` for why that step is the
        caller's to take.
        """
        from .observation import Observation

        return Observation(
            base_image=self.base_image,
            wrist_image=self.wrist_image,
            state=self.state,
            prompt=self.prompt,
            provenance=dict(self.provenance),
        )


def _require_h5py():
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - depends on the host
        raise ImportError(
            "reading LIBERO demonstrations needs `h5py`; install the 'libero' "
            "extra (`pip install -e 'python/apxinf_ref[libero]'`). A pre-captured "
            "--bundle needs neither h5py nor this module."
        ) from error
    return h5py


def hdf5_files(root: str | Path) -> list[Path]:
    """Every LIBERO HDF5 under ``root``, sorted, so a seed picks the same one twice."""
    directory = Path(root).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"no LIBERO directory at {directory}")
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"no LIBERO HDF5 files under {directory}; expected *.hdf5 or *.h5 "
            "(the distribution ships one file per task)"
        )
    return files


def _first_dataset(group, names) -> tuple[Any, str]:
    for name in names:
        if name in group:
            return group[name], name
    raise ValueError(
        f"refused: the HDF5 observation group has none of {', '.join(names)}"
    )


def _state_layout(observation) -> tuple[Any, Any, Any, str]:
    for position, orientation, gripper in _STATE_LAYOUTS:
        if all(name in observation for name in (position, orientation, gripper)):
            return (
                observation[position],
                observation[orientation],
                observation[gripper],
                f"{position}+{orientation}+{gripper}",
            )
    tried = "; ".join("+".join(layout) for layout in _STATE_LAYOUTS)
    raise ValueError(f"refused: the HDF5 observation group matches no state layout; tried {tried}")


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode_text(value.item())
    return str(value)


def _prompt_from_attributes(attributes) -> str | None:
    """``problem_info`` first, then any attribute literally named for the task."""
    for key, raw in attributes.items():
        text = _decode_text(raw).strip()
        if key == "language_instruction" and text:
            return text
        if key != _PROBLEM_INFO or not text.startswith("{"):
            continue
        try:
            metadata = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"refused: {_PROBLEM_INFO} is not valid JSON") from error
        instruction = str(metadata.get("language_instruction", "")).strip()
        if instruction:
            return instruction
    return None


#: The scene-name prefixes LIBERO puts in front of the instruction, as
#: ``<suite>_SCENE<n>_<instruction>_demo.hdf5``.
_SUITE_PREFIXES = ("KITCHEN", "LIVING", "STUDY", "OFFICE")


def _prompt_from_filename(path: Path) -> str:
    """``KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5`` -> the task.

    The distribution names every file after the instruction it demonstrates,
    which is the only task source left for a file whose attributes were stripped.
    """
    stem = path.stem
    for marker in ("_demo", "_demos"):
        if stem.endswith(marker):
            stem = stem[: -len(marker)]

    head, separator, rest = stem.partition("_")
    if separator and head.upper() in _SUITE_PREFIXES:
        # Drop the SCENE<n> segment too, but never all of it: a file named only
        # `KITCHEN_SCENE8_demo.hdf5` still has to yield something.
        _, _, after_scene = rest.partition("_")
        stem = after_scene or rest
    return stem.replace("_", " ").strip()


def _shape_of(dataset, path: Path, name: str, rank: int) -> tuple[int, ...]:
    if dataset.ndim != rank:
        raise ValueError(
            f"refused: {path.name}:{name} must have rank {rank}, got shape {dataset.shape}"
        )
    return tuple(int(size) for size in dataset.shape)


def load_sample(root: str | Path, *, seed: int = 0) -> LiberoHdf5Sample:
    """One deterministic demonstration frame from the HDF5 dataset under ``root``.

    The selection is a reservoir sample over ``(file, demo, frame)`` in sorted
    order under ``numpy.random.default_rng(seed)``, the FlashRT producer's
    scheme, so the same seed names the same frame without holding an index of
    every frame in memory.
    """
    h5py = _require_h5py()
    files = hdf5_files(root)
    root_path = Path(root).expanduser()

    rng = np.random.default_rng(int(seed))
    seen = 0
    selected: tuple[Path, str, int, int] | None = None
    for path in files:
        with h5py.File(path, "r") as handle:
            data = handle.get("data")
            if data is None:
                raise ValueError(f"refused: {path.name} has no `data` group")
            demos = sorted(name for name in data.keys() if str(name).startswith("demo_"))
            if not demos:
                raise ValueError(f"refused: {path.name} has no data/demo_* groups")
            for demo_name in demos:
                observation = data[demo_name].get("obs")
                if observation is None:
                    raise ValueError(f"refused: {path.name}:{demo_name} has no obs group")
                base, _ = _first_dataset(observation, _BASE_CAMERA_NAMES)
                for frame in range(int(base.shape[0])):
                    if int(rng.integers(seen + 1)) == 0:
                        selected = (path, demo_name, frame, seen)
                    seen += 1

    if selected is None:
        raise RuntimeError(f"no demonstration frames under {root_path}")
    path, demo_name, frame, global_index = selected

    with h5py.File(path, "r") as handle:
        data = handle["data"]
        demo = data[demo_name]
        observation = demo["obs"]
        base, base_name = _first_dataset(observation, _BASE_CAMERA_NAMES)
        wrist, wrist_name = _first_dataset(observation, _WRIST_CAMERA_NAMES)
        position, orientation, gripper, layout = _state_layout(observation)

        _shape_of(base, path, base_name, 4)
        _shape_of(wrist, path, wrist_name, 4)
        for name, dataset in (
            (base_name, base),
            (wrist_name, wrist),
            ("position", position),
            ("orientation", orientation),
            ("gripper", gripper),
        ):
            if dataset.shape[0] != base.shape[0]:
                raise ValueError(
                    f"refused: {path.name}:{demo_name} has inconsistent frame counts "
                    f"({name} has {dataset.shape[0]}, the base camera has {base.shape[0]})"
                )

        # The 8-value vector `scripts/libero_observation.libero_state` builds for
        # the live simulator, from the fields the file already stores in exactly
        # that form. See the module docstring.
        state = np.concatenate(
            (
                np.asarray(position[frame], dtype=np.float32).reshape(-1)[:3],
                np.asarray(orientation[frame], dtype=np.float32).reshape(-1)[:3],
                np.asarray(gripper[frame], dtype=np.float32).reshape(-1)[:2],
            )
        ).astype(np.float32)
        if state.size != 8:
            raise ValueError(
                f"refused: expected LIBERO's 8-value state, built {state.size} from "
                f"{layout} in {path.name}:{demo_name}"
            )
        if not np.isfinite(state).all():
            raise ValueError(f"refused: non-finite state in {path.name}:{demo_name}:{frame}")

        prompt = _prompt_from_attributes(data.attrs) or _prompt_from_attributes(demo.attrs)
        prompt = prompt or _prompt_from_filename(path)

        try:
            receipt = str(path.relative_to(root_path))
        except ValueError:
            receipt = path.name

        return LiberoHdf5Sample(
            base_image=np.ascontiguousarray(np.asarray(base[frame])),
            wrist_image=np.ascontiguousarray(np.asarray(wrist[frame])),
            state=state,
            prompt=prompt,
            provenance={
                "file": receipt,
                "demo": demo_name,
                "frame_index": int(frame),
                "global_index": int(global_index),
                "state_layout": layout,
                "cameras": [base_name, wrist_name],
            },
        )
