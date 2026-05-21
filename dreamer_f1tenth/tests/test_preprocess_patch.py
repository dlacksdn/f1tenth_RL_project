"""A20 — WorldModel.preprocess image-key KeyError fix (decision #14 fork-patch).

vendor/dreamerv3-torch/models.py:182 was `obs["image"] = obs["image"] / 255.0`
(unconditional). Vector-only F1Tenth obs (lidar/state, no image) raised KeyError.
The fork-patch wraps it in `if "image" in obs:`.

This test binds the real `WorldModel.preprocess` to a dummy holding a minimal
`_config` and exercises the patched code path with vector-only obs over a batch
that exceeds the A20 spec (100 train + 10 eval steps), asserting no KeyError and
correct downstream key construction (`cont`).
"""
import types

import numpy as np
import pytest
import torch

import models  # vendor/dreamerv3-torch (on path via conftest)


def _dummy_wm():
    """Object exposing only what WorldModel.preprocess touches."""
    cfg = types.SimpleNamespace(device="cpu", discount=0.997)
    return types.SimpleNamespace(_config=cfg)


def _vector_only_obs(batch, length):
    """F1Tenth obs dict: lidar (1080,) + state (5,), no image key."""
    return {
        "lidar": np.random.rand(batch, length, 1080).astype(np.float32),
        "state": np.random.rand(batch, length, 5).astype(np.float32),
        "is_first": np.zeros((batch, length), dtype=bool),
        "is_terminal": np.zeros((batch, length), dtype=bool),
    }


def test_a20_preprocess_vector_only_no_keyerror():
    wm = _dummy_wm()
    preprocess = models.WorldModel.preprocess.__get__(wm)

    # A20: 100 train-shaped + 10 eval-shaped vector-only preprocess calls, no error.
    for _ in range(100):
        obs = _vector_only_obs(batch=8, length=64)  # train batch shape
        out = preprocess(obs)
        assert "image" not in out  # patch must not synthesize an image key
        assert "cont" in out and out["cont"].shape == (8, 64, 1)
        assert out["lidar"].dtype == torch.float32

    for _ in range(10):
        obs = _vector_only_obs(batch=1, length=1)  # eval/rollout shape
        out = preprocess(obs)
        assert "image" not in out
        assert "cont" in out


def test_a20_preprocess_still_divides_image_when_present():
    """Regression: vision path unchanged — image still divided by 255 when present."""
    wm = _dummy_wm()
    preprocess = models.WorldModel.preprocess.__get__(wm)

    obs = {
        "image": (np.ones((2, 3, 4, 4, 3), dtype=np.float32) * 255.0),
        "is_first": np.zeros((2, 3), dtype=bool),
        "is_terminal": np.zeros((2, 3), dtype=bool),
    }
    out = preprocess(obs)
    assert torch.allclose(out["image"], torch.ones_like(out["image"]))
