"""The exchange, end to end, on one host.

Capture a bundle from a real LIBERO observation, replay it, and subtract the two
documents stage by stage -- same device, same implementation, same inputs.
Anything but bit-identical means capture and replay are not in fact sharing one
set of conventions, which is the only thing the design is for.

Everything here needs the checkpoint and a LIBERO dataset, and the capture runs
the model twice: about two minutes, much of it the full-file hash of the 7 GB
checkpoint that each run deliberately recomputes. `pytest -m "not slow"` leaves
this file out; the documented self-loop runs it.
"""

from __future__ import annotations

import contextlib
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from apxinf_ref import bundle as bundle_module
from apxinf_ref import cli
from apxinf_ref import compare, device as device_module
from apxinf_ref.model.pi05 import preprocess

pytestmark = pytest.mark.slow

# The seed the one capture is made under. Deliberately not 0: the replay borrows
# the seed the bundle recorded, so a regression to "always zero" would be
# invisible if the capture had used the default.
SEED = 7

# The probe's stage names at this checkpoint's depths. Derived rather than
# counted, so a depth change fails here instead of silently changing a number.
VISION_DEPTH = 27
LANGUAGE_DEPTH = 18
ACTION_DEPTH = 10
STAGES = (
    ["vision_patch_embed"]
    + [f"vision_layer_{index}" for index in range(VISION_DEPTH)]
    + ["vision_projected", "prefix_v_layer0", f"prefix_v_layer{LANGUAGE_DEPTH - 1}"]
    + [f"denoise_step_{step}" for step in range(ACTION_DEPTH)]
)

# What `python/apxinf_ref/README.md` says is asserted in the tests, and what
# `doc/pi05-cuda-regression.md` pins for the LIBERO record.
TOKENIZER_DIGEST = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"


@pytest.fixture(scope="module", autouse=True)
def the_observation_path_is_importable():
    """`infer` reads an observation through two packages of this repository.

    On a host without them the import error would be an error rather than a
    skip, so it is turned into one here.
    """
    for module in ("apxinf.processors", "scripts.libero_observation"):
        pytest.importorskip(module)


def a_device() -> str:
    available = device_module.available_accelerators()
    return available[0] if available else "cpu"


def run(argv) -> None:
    """A documented command, in process. A refusal raises rather than returns."""
    assert cli.main(argv) == 0


def printed(argv) -> str:
    """Run a documented command and catch the document it prints.

    The commands write nothing, so stdout is where a run's document exists -- and
    catching it here rather than with ``capsys`` is not a preference: ``capsys``
    is function-scoped and this file's capture is shared by a module-scoped
    fixture, which could not depend on it.

    Everything the command prints went to stderr, so what comes back is the
    document and nothing else; a stray ``print`` would surface downstream as a
    parse error rather than as a quiet extra line in a file.
    """
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        run(argv)
    return stream.getvalue()


@pytest.fixture(scope="module")
def exchange(tmp_path_factory, checkpoint_path, tokenizer_path, norm_stats_path, libero_root):
    """One capture and its replay, shared by every test in this file."""
    directory = tmp_path_factory.mktemp("exchange")
    bundle = directory / "bundle"
    device = a_device()

    captured = printed(
        [
            "infer",
            "--device",
            device,
            "--libero-root",
            str(libero_root),
            "--seed",
            str(SEED),
            "--capture",
            str(bundle),
        ]
    )
    # No --seed: the replay takes the one the bundle recorded.
    replayed = printed(["infer", "--device", device, "--bundle", str(bundle)])

    return SimpleNamespace(
        bundle=bundle,
        # The capture's own stdout, held as text so it can be compared to the
        # bundle's probe.json byte for byte.
        captured=captured,
        replay=json.loads(replayed),
        # The capture has no other copy: the command wrote no document of its
        # own, so the bundle's probe.json *is* what it printed.
        document=json.loads((bundle / bundle_module.PROBE_NAME).read_text()),
    )


def test_a_capture_replays_bit_identically_on_the_same_host(exchange):
    captured = exchange.document
    replayed = exchange.replay

    assert sorted(captured["intermediate_signatures"]) == sorted(STAGES)
    assert captured["intermediate_signatures"] == replayed["intermediate_signatures"]

    # The producer's own probe travels inside the capture byte for byte: it is
    # what makes a replay scoreable stage by stage rather than on one cosine.
    assert (exchange.bundle / bundle_module.PROBE_NAME).read_text() == exchange.captured

    stored = bundle_module.read_bundle(exchange.bundle)
    # Frames are stored before the resize, so the replay re-runs the letterbox
    # here rather than trusting the producer's.
    assert stored.observation.base_image.shape == (128, 128, 3)
    assert captured["image_size"] == 224
    assert captured["resize"] == preprocess.PIL_RESIZE

    # Noise is rounded through bfloat16 before it is frozen: the engine receives
    # bf16 noise, so a float32 draw would hand the two implementations different
    # inputs, which is a different question rather than a rounding difference.
    noise = stored.noise
    assert np.array_equal(noise, torch.tensor(noise).to(torch.bfloat16).float().numpy())

    # The bundle is scored against the answer it carried, so this is 1.0 to
    # within an ulp of float64 -- the measure is a dot product divided by the
    # product of two norms, not a bitwise comparison.
    assert replayed["gold_cosine"] == pytest.approx(1.0, rel=1e-12)

    report = replayed["bundle"]
    assert all(entry["matched"] for entry in report["artifacts_match"].values())
    assert report["artifacts_match"]["tokenizer"]["local"] == TOKENIZER_DIGEST
    assert report["token_ids_match"] is True

    # And the two documents subtract to nothing at all.
    comparison = compare.compare_documents(captured, replayed)
    assert comparison["passed"], comparison["failures"]


def test_a_replay_under_another_seed_is_refused(exchange):
    """The one hard stop: noise is an input, so a different draw is a different
    question -- and it would look exactly like the same one."""
    with pytest.raises(SystemExit, match="frozen noise"):
        cli.main(
            [
                "infer",
                "--device",
                a_device(),
                "--bundle",
                str(exchange.bundle),
                "--seed",
                str(SEED + 1),
            ]
        )
