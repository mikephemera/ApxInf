"""The promises another component, another host or the engine depends on.

Each test below holds up one promise the documentation makes: a container rule,
a refusal, a resolution order, a literal shared with the Rust side. These are
the things the far side of an exchange is entitled to rely on. Behaviour that is
only internal to a helper is deliberately not here -- ``test_e2e.py`` is the
exchange these are the contract for.

Nothing in this file loads a model, so it stays in the seconds range.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from apxinf_ref import bundle as bundle_module
from apxinf_ref import device as device_module
from apxinf_ref import infer as infer_module
from apxinf_ref import paths
from apxinf_ref import probe, sources
from apxinf_ref.compare import THRESHOLDS, compare_documents, stage_metrics
from apxinf_ref.model.pi05 import preprocess
from apxinf_ref.model.pi05.observation import ModelInputs, Observation

RESIZE = "apxinf.processors.ResizeWithPad (PIL BILINEAR, antialiased)"


# --- fixtures built by hand, so the reader's contract is what is exercised ---


def an_observation(size=256):
    rng = np.random.default_rng(0)
    return Observation(
        base_image=rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
        wrist_image=rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
        state=np.linspace(-1.0, 1.0, 8, dtype=np.float32),
        prompt="put both moka pots on the stove",
    )


def some_inputs(observation):
    rng = np.random.default_rng(1)
    return ModelInputs(
        patches=rng.normal(size=(512, 588)).astype(np.float32),
        token_ids=np.arange(46, dtype=np.int64),
        noise=rng.normal(size=(10, 32)).astype(np.float32),
        token_count=46,
        state_width=8,
        image_size=224,
        source_image_size=(256, 256),
        prompt=observation.prompt,
        formatted_prompt="Task: put both moka pots on the stove, State: 1 2;\nAction: ",
        tokenizer_digest="a" * 64,
        preprocessed_rgb=rng.normal(size=(2, 3, 224, 224)).astype(np.float32),
        provenance={},
    )


def a_source(observation, oriented=True):
    return sources.Source(
        kind="libero_hdf5",
        observation=observation,
        oriented=oriented,
        resize=RESIZE,
        extras={},
        provenance={"kind": "libero_hdf5", "sample": {"file": "demo.hdf5"}},
    )


def write(directory, observation=None, **overrides):
    sample = observation if observation is not None else an_observation()
    values = dict(
        observation=sample,
        inputs=some_inputs(sample),
        document={
            "schema": probe.SCHEMA,
            "token_count": 46,
            "intermediate_signatures": {"vision_patch_embed": {"elements": 4}},
        },
        actions=np.zeros((10, 7), dtype=np.float32),
        raw_actions=np.zeros((10, 32), dtype=np.float32),
        source=a_source(sample),
        artifacts={
            "checkpoint": {
                "path": "/ckpt",
                "weights": "model.safetensors",
                "sha256": "b" * 64,
            }
        },
        thresholds={"stage_cosine": 0.999},
        engine="assembled",
        device="cpu",
        precision="bfloat16",
        seed=0,
        receipt=["engine=assembled device=cpu"],
    )
    values.update(overrides)
    return bundle_module.write_bundle(directory, **values)


def a_container(directory, *, oriented=True, **manifest_overrides):
    """A container written by hand, with the manifest a foreign producer wrote."""
    directory.mkdir(parents=True, exist_ok=True)
    sample = an_observation()
    inputs = directory / bundle_module.INPUTS_NAME
    np.savez(
        inputs,
        base_image=sample.base_image,
        wrist_image=sample.wrist_image,
        state=sample.state,
        prompt=np.asarray([sample.prompt]),
        formatted_prompt=np.asarray(["Task: put both moka pots on the stove;\nAction: "]),
        noise=np.arange(320, dtype=np.float32).reshape(10, 32),
        token_ids=np.arange(46, dtype=np.int64),
        gold_actions=np.zeros((10, 7), dtype=np.float32),
    )
    manifest = {
        "schema": bundle_module.BUNDLE_SCHEMA,
        "producer": {"package": "apxinf-ref", "engine": "assembled", "device": "musa:0"},
        "source": {
            "kind": "libero_hdf5",
            "sample": {"file": "demo.hdf5"},
            "oriented": oriented,
            "resize": RESIZE,
            "num_views": 2,
            "image_size": 224,
            "source_image_size": [256, 256],
        },
        "seed": 424242,
        "token_count": 46,
        "state_width": 8,
        "files": {
            bundle_module.INPUTS_NAME: {
                "bytes": inputs.stat().st_size,
                "sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
            }
        },
    }
    manifest.update(manifest_overrides)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


# --- the container ---------------------------------------------------------


def test_a_round_trip_gives_back_what_was_stored(tmp_path):
    sample = an_observation()
    inputs = some_inputs(sample)
    written = write(tmp_path / "bundle", observation=sample, inputs=inputs)

    bundle = bundle_module.read_bundle(written)
    assert bundle.manifest["schema"] == bundle_module.BUNDLE_SCHEMA
    assert np.array_equal(bundle.observation.base_image, sample.base_image)
    assert np.array_equal(bundle.observation.wrist_image, sample.wrist_image)
    assert np.array_equal(bundle.observation.state, sample.state)
    assert bundle.observation.prompt == sample.prompt
    assert np.array_equal(bundle.noise, inputs.noise)
    assert np.array_equal(bundle.token_ids, inputs.token_ids)
    assert bundle.formatted_prompt == inputs.formatted_prompt
    assert bundle.oriented is True
    # The producer's own stage probe travels with the capture: without it a
    # replay could only be scored on one final cosine.
    assert bundle.probe["intermediate_signatures"]["vision_patch_embed"]["elements"] == 4
    # The frames are stored before the resize, so the replay re-runs the letterbox.
    assert bundle.observation.base_image.shape == (256, 256, 3)


def test_the_manifest_names_its_files_and_digests_them(tmp_path):
    written = write(tmp_path / "bundle")
    manifest = json.loads((written / "manifest.json").read_text())

    assert sorted(manifest["files"]) == [
        bundle_module.INPUTS_NAME,
        bundle_module.PROBE_NAME,
    ]
    for name, record in manifest["files"].items():
        assert record["bytes"] == (written / name).stat().st_size
        assert len(record["sha256"]) == 64
    stored = manifest["arrays"][bundle_module.INPUTS_NAME]
    assert sorted(stored) == [
        "base_image",
        "formatted_prompt",
        "gold_actions",
        "gold_raw_actions",
        "noise",
        "preprocessed_base_rgb",
        "preprocessed_wrist_rgb",
        "prompt",
        "state",
        "token_ids",
        "wrist_image",
    ]
    assert stored["base_image"]["dtype"] == "uint8"
    assert stored["noise"]["shape"] == [10, 32]
    assert stored["preprocessed_wrist_rgb"]["shape"] == [1, 3, 224, 224]


def test_a_truncated_bundle_is_refused(tmp_path):
    """A damaged container has to be refused here: a replay of one produces stage
    signatures that look exactly like an answer."""
    written = write(tmp_path / "bundle")
    path = written / bundle_module.INPUTS_NAME
    path.write_bytes(path.read_bytes()[:-16])

    with pytest.raises(ValueError, match="truncated"):
        bundle_module.read_bundle(written)


def test_an_edited_bundle_is_refused_and_says_so(tmp_path):
    """Same length, different bytes: not a truncated copy but a changed one."""
    written = write(tmp_path / "bundle")
    path = written / bundle_module.INPUTS_NAME
    content = bytearray(path.read_bytes())
    content[len(content) // 2] ^= 0x01
    path.write_bytes(bytes(content))

    with pytest.raises(ValueError, match="modified"):
        bundle_module.read_bundle(written)


def test_a_directory_with_no_manifest_is_refused(tmp_path):
    """The FlashRT gold standard's container, which this reader does not accept."""
    directory = tmp_path / "legacy"
    directory.mkdir()
    np.savez(directory / "io.npz", image=np.zeros((4, 4, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="FlashRT"):
        bundle_module.read_bundle(directory)


def test_an_unknown_schema_is_refused(tmp_path):
    written = write(tmp_path / "bundle")
    path = written / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema"] = "apxinf.pi05.bundle.v2"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="bundle.v2"):
        bundle_module.read_bundle(written)


def test_a_non_empty_destination_is_refused(tmp_path):
    target = tmp_path / "bundle"
    target.mkdir()
    (target / "something").write_text("from an earlier experiment")

    with pytest.raises(FileExistsError, match="not empty"):
        write(target)
    assert (target / "something").read_text() == "from an earlier experiment"


def test_force_replaces_a_capture_and_leaves_nothing_behind(tmp_path):
    target = tmp_path / "bundle"
    write(target)
    first = json.loads((target / "manifest.json").read_text())["generated_at"]

    write(target, force=True, receipt=["replaced"])
    assert json.loads((target / "manifest.json").read_text())["generated_at"] >= first
    assert (target / bundle_module.LOG_NAME).read_text().endswith("replaced\n")
    assert [entry.name for entry in tmp_path.iterdir()] == ["bundle"]


def test_a_failed_write_leaves_the_previous_capture_alone(tmp_path):
    """Nothing is ever written *into* the destination, so a failure cannot mix two."""
    target = tmp_path / "bundle"
    write(target, receipt=["the first capture"])
    marker = (target / bundle_module.LOG_NAME).read_text()

    with pytest.raises(TypeError):
        # A document json cannot serialise: this fails after inputs.npz and
        # probe.json are already in the staging directory.
        write(target, document={"signature": np.zeros(2)}, force=True)

    assert (target / bundle_module.LOG_NAME).read_text() == marker
    assert [entry.name for entry in tmp_path.iterdir()] == ["bundle"]


def test_artifact_digests_and_token_ids_are_reported_not_gated():
    """Only the noise is a hard stop; an artifact mismatch is a fact for the reader."""
    manifest = {
        "artifacts": {
            "tokenizer": {"path": "/theirs", "sha256": "1" * 64},
            "norm_stats": {"path": "/both", "sha256": "2" * 64},
        },
        "thresholds": {"stage_cosine": 0.999},
        "token_count": 46,
    }
    local = {
        "tokenizer": {"path": "/ours", "sha256": "9" * 64},
        "norm_stats": {"path": "/both", "sha256": "2" * 64},
    }

    report = bundle_module.replay_report(
        manifest, local_artifacts=local, token_ids=np.arange(46)
    )

    assert report["artifacts_match"]["tokenizer"] == {
        "recorded": "1" * 64,
        "local": "9" * 64,
        "matched": False,
    }
    assert report["artifacts_match"]["norm_stats"]["matched"] is True
    assert report["token_ids_match"] is True
    assert report["thresholds"] == {"stage_cosine": 0.999}


def test_frames_the_manifest_says_are_not_oriented_are_rotated(tmp_path):
    """The rotation follows the manifest, so a foreign producer can say otherwise.

    Every bundle this code writes records ``oriented: true``, so this branch
    exists only for another host's capture -- and a missing or doubled rotation
    would be invisible in a same-host replay, which rotates on both sides.
    """
    directory = a_container(tmp_path / "bundle", oriented=False)

    source = sources.from_bundle(directory)

    stored = bundle_module.read_bundle(directory).observation
    rotated = preprocess.orient_libero_images(stored.base_image, stored.wrist_image)

    assert source.oriented is False
    assert not np.array_equal(source.observation.base_image, stored.base_image)
    assert np.array_equal(source.observation.base_image, rotated[0])
    assert source.resize == preprocess.PIL_RESIZE


# --- the replay's pre-flight refusals --------------------------------------


class a_config:
    """Only the fields the pre-flight check reads."""

    def __init__(self, num_views=2, image_size=224):
        self.num_views = num_views
        self.image_size = image_size


def a_bundle_source(**manifest_overrides):
    """A bundle source whose manifest asks a question someone else chose."""
    manifest = {
        "seed": 424242,
        "token_count": 46,
        "source": {"num_views": 2, "image_size": 224},
    }
    for key, value in manifest_overrides.items():
        if key in manifest["source"]:
            manifest["source"][key] = value
        else:
            manifest[key] = value
    return sources.Source(
        kind="bundle",
        observation=an_observation(),
        oriented=True,
        resize=RESIZE,
        extras={"manifest": manifest},
        provenance={"kind": "bundle"},
    )


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"num_views": 3}, "view count 3"),
        ({"image_size": 448}, "image size 448"),
        ({"token_count": 21}, "token count 21"),
    ],
)
def test_a_replay_refuses_a_bundle_that_asks_another_question(overrides, expected):
    """These three are facts about the question, not about a host.

    The prefix length and every stage downstream follow them, so letting one
    through would surface much later as a shape error or, worse, as a cosine that
    looks like a precision difference.
    """
    with pytest.raises(SystemExit, match=expected):
        infer_module._refuse_a_bundle_that_asks_another_question(
            a_bundle_source(**overrides),
            config=a_config(),
            inputs=some_inputs(an_observation()),
        )


# --- the stage signature, shared with the Rust probe ------------------------


def test_the_signature_matches_the_rust_side_element_for_element():
    """The same literals `pi05_stage_probe.rs` asserts in its own test.

    Pinning both implementations to one set of literals is what makes the two
    documents comparable, and it is checkable here: the engine side only runs on
    a CUDA host, so a mismatch found there costs a whole run.
    """
    value = probe.signature(np.array([1.0, -2.0, 3.5, 0.0], dtype=np.float32))
    assert value["elements"] == 4
    assert value["sum"] == 2.5
    assert value["abs_checksum"] == 6.5
    assert value["l2"] == 4.153311931459037
    assert value["max_abs"] == 3.5
    assert value["sample"] == [1.0, -2.0, 3.5, 0.0]

    # A real stage has more elements than the sample limit, where the grid's
    # integer division is no longer the identity.
    value = probe.signature(np.arange(1000, dtype=np.float32))
    assert value["elements"] == 1000
    assert value["sum"] == 499500.0
    assert value["l2"] == 18243.72494859534
    assert value["max_abs"] == 999.0
    sample = value["sample"]
    assert len(sample) == 256
    assert (sample[0], sample[1], sample[2], sample[254], sample[255]) == (
        0.0,
        3.0,
        7.0,
        995.0,
        999.0,
    )


def test_the_probe_document_is_stamped_and_ordered():
    import torch

    stages = {"denoise_step_1": torch.zeros(2), "denoise_step_0": torch.ones(2)}
    result = probe.collect(stages, token_count=7)

    assert result.document["schema"] == probe.SCHEMA
    assert result.document["token_count"] == 7
    # Sorted so that two runs of the same implementation are byte-identical; the
    # engine gets the same ordering from its BTreeMap.
    assert list(result.signatures) == ["denoise_step_0", "denoise_step_1"]


# --- devices ---------------------------------------------------------------


def test_an_unavailable_accelerator_is_refused_not_downgraded():
    """A runtime that quietly moved the work to the CPU would make every
    downstream comparison meaningless."""
    available = device_module.available_accelerators()
    for family in ("cuda", "musa"):
        if family in available:
            continue
        with pytest.raises(RuntimeError, match="not available on this host"):
            device_module.resolve_device_string(family)


def test_the_model_refuses_a_misplaced_input_before_doing_any_work():
    """The guarantee is only real if the model actually asks for it.

    The failure it prevents -- a tensor quietly carried across a device inside
    the first matmul -- is invisible in the numbers. The assertion runs before
    any weight is touched, so a stub instance is enough to observe it.
    """
    torch = pytest.importorskip("torch")

    from apxinf_ref.model.pi05.config import Pi05Config
    from apxinf_ref.model.pi05.model import Pi05

    model = Pi05.__new__(Pi05)
    model.config = Pi05Config(num_views=2)
    model.device = torch.device("meta")
    with pytest.raises(RuntimeError, match="the patch tensor"):
        model.embed_prefix(torch.zeros(512, 588), torch.zeros(10, dtype=torch.long))


# --- artifact resolution ---------------------------------------------------


def checkpoint_at(root, name="ckpt"):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / paths.WEIGHTS_NAME).write_bytes(b"")
    return directory


def test_the_resolution_order_is_argument_then_environment_then_cache(tmp_path, monkeypatch):
    """Each step pins an *order*: the failure mode is not a crash, it is a run
    quietly measuring a different file from the other side of a comparison.

    ``APXINF_PI05_CHECKPOINT`` is the variable ``crates/apxinf-py``'s conftest
    reads, so one setting serves both sides.
    """
    explicit = checkpoint_at(tmp_path, "explicit")
    from_env = checkpoint_at(tmp_path, "from-env")
    default = checkpoint_at(tmp_path, "default")
    monkeypatch.setenv(paths.ENV_CHECKPOINT, str(from_env))
    monkeypatch.setattr(paths, "DEFAULT_CHECKPOINT", default)

    assert paths.resolve_checkpoint(explicit) == explicit
    assert paths.resolve_checkpoint() == from_env
    monkeypatch.delenv(paths.ENV_CHECKPOINT)
    assert paths.resolve_checkpoint() == default


def test_norm_stats_prefer_the_checkpoints_own_copy(tmp_path, monkeypatch):
    """The statistics and the weights have to be the same export's."""
    checkpoint = checkpoint_at(tmp_path)
    inside = checkpoint / "assets/physical-intelligence/libero"
    inside.mkdir(parents=True)
    own = inside / "norm_stats.json"
    own.write_text("{}")
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text("{}")

    monkeypatch.setenv(paths.ENV_NORM_STATS, str(elsewhere))
    # The env var is an explicit request, so it still outranks the convention...
    assert paths.resolve_norm_stats(checkpoint) == elsewhere
    # ...but the checkpoint's own copy is what is found with no request made.
    monkeypatch.delenv(paths.ENV_NORM_STATS)
    assert paths.resolve_norm_stats(checkpoint) == own


# --- the comparison --------------------------------------------------------


def a_stage(values, **overrides):
    entry = {
        "elements": len(values),
        "sum": float(sum(values)),
        "abs_checksum": float(sum(abs(v) for v in values)),
        "l2": float(np.sqrt(sum(v * v for v in values))),
        "max_abs": max((abs(v) for v in values), default=0.0),
        "sample": list(values),
    }
    entry.update(overrides)
    return entry


def a_document(stages, token_count=10):
    return {
        "schema": probe.SCHEMA,
        "token_count": token_count,
        "intermediate_signatures": stages,
    }


def test_thresholds_mirror_the_bench():
    """A verdict has to mean the same thing on both sides of the comparison."""
    assert THRESHOLDS["bf16"]["min_cosine"] == 0.999
    assert THRESHOLDS["bf16"]["max_relative_l2"] == 0.05
    assert THRESHOLDS["fp8"]["min_cosine"] == 0.997
    assert THRESHOLDS["fp8"]["max_relative_l2"] == 0.10
    assert THRESHOLDS["int8"]["max_abs"] == 0.125


def test_relative_l2_is_normalised_by_the_reference_not_the_candidate():
    """The two denominators differ, and picking the wrong one moves the verdict.

    Normalising by the candidate is the mistake this pins: it is not obviously
    wrong on paper, and it is not visible in the metric's name. The mirror image
    follows the reference the other way, so neither assertion passes by accident
    of which side happens to be larger.
    """
    reference = a_stage([1.0, 0.0])
    candidate = a_stage([1.0, 3.0])
    assert stage_metrics(reference, candidate)["relative_l2"] == pytest.approx(3.0)
    assert stage_metrics(candidate, reference)["relative_l2"] == pytest.approx(
        np.sqrt(9.0 / 10.0)
    )


def test_a_token_count_mismatch_is_refused_rather_than_reported():
    # The two documents describe different questions, so there is no verdict to
    # report at all.
    reference = a_document({"vision_layer_0": a_stage([1.0])}, token_count=10)
    candidate = a_document({"vision_layer_0": a_stage([1.0])}, token_count=21)
    with pytest.raises(ValueError, match="token counts differ"):
        compare_documents(reference, candidate)


# --- the observation path --------------------------------------------------


def test_the_two_libero_sources_build_the_same_state_vector():
    """The HDF5 and the live simulator must agree on the 8-value vector.

    ``create_dataset.py`` stores ``ee_ori`` as ``quat2axisangle(ee_quat)``, so
    ``eef_pos + ee_ori + gripper`` is the same vector
    ``scripts/libero_observation.libero_state`` builds from a simulator
    observation with both finger joints. If either convention moved, one of the
    two data sources would be scoring prompts the checkpoint never saw.
    """
    pytest.importorskip("h5py")
    from scripts.libero_observation import libero_state, quat_to_axis_angle

    position = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    quaternion = np.array([0.0, 0.0, 0.5, 0.8660254], dtype=np.float64)
    gripper = np.array([0.01, -0.01], dtype=np.float64)

    from_simulator = libero_state(
        {
            "robot0_eef_pos": position,
            "robot0_eef_quat": quaternion,
            "robot0_gripper_qpos": gripper,
        },
        finger_joints=2,
    )
    from_hdf5 = np.concatenate(
        [position, quat_to_axis_angle(quaternion), gripper]
    ).astype(np.float32)

    assert from_simulator.shape == from_hdf5.shape == (8,)
    assert from_simulator == pytest.approx(from_hdf5, abs=1e-6)
