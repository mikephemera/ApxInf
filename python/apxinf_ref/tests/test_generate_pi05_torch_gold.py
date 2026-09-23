from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_pi05_torch_gold",
    ROOT / "generate_pi05_torch_gold.py",
)
assert SPEC is not None and SPEC.loader is not None
gold = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gold)


def test_fixture_generation_is_cpu_seeded_and_reproducible():
    first = gold._make_fixture(seed=17, views=2, token_count=10, horizon=10)
    second = gold._make_fixture(seed=17, views=2, token_count=10, horizon=10)

    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert first["images"].shape == (2, 3, 224, 224)
    assert first["images"].dtype == np.float32
    assert first["token_ids"].shape == (10,)
    assert first["token_ids"].dtype == np.int64
    assert first["noise"].shape == (10, 32)
    assert first["noise"].dtype == np.float32
    assert np.all(first["images"] >= -1.0)
    assert np.all(first["images"] <= 1.0)


def test_fixture_loader_rejects_wrong_shape(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, images=np.zeros((2, 224, 224, 3), dtype=np.float32), token_ids=np.zeros(10, dtype=np.int64), noise=np.zeros((10, 32), dtype=np.float32))

    try:
        gold._load_fixture(path)
    except ValueError as error:
        assert "images must have shape" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("invalid fixture was accepted")


def test_output_contains_npz_and_manifest(tmp_path):
    arrays = gold._make_fixture(seed=3, views=2, token_count=4, horizon=2)
    stages = {
        "Pi05.vision.layer_00": [
            {
                "elements": 2,
                "shape": [1, 2],
                "dtype": "torch.bfloat16",
                "sum": 1.0,
                "abs_sum": 3.0,
                "l2": 2.236,
                "max_abs": 2.0,
                "sample": [1.0, -2.0],
            }
        ]
    }
    manifest = {"schema": gold.SCHEMA, "stages": stages}

    gold._write_output(
        tmp_path,
        arrays=arrays,
        raw_actions=np.zeros((2, 32), dtype=np.float32),
        stages=stages,
        manifest=manifest,
        force=False,
    )

    with np.load(tmp_path / "gold.npz", allow_pickle=False) as document:
        assert document["images"].shape == (2, 3, 224, 224)
        assert document["raw_actions"].shape == (2, 32)
        assert document["stage_0000_call_00"].shape == (2,)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["schema"] == gold.SCHEMA
    assert saved["files"]["gold.npz"]["sha256"]
