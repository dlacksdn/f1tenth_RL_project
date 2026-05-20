"""A1, A2 — gymnasium API + obs dict shape."""
import numpy as np
import pytest

from dreamer_f1tenth.envs import F110GymnasiumWrapper


@pytest.fixture(scope="module")
def env():
    e = F110GymnasiumWrapper(trackname="Oschersleben")
    yield e
    e.close()


def test_a1_reset_signature(env):
    out = env.reset(seed=0)
    assert isinstance(out, tuple) and len(out) == 2, "reset must return (obs, info)"
    obs, info = out
    assert isinstance(obs, dict)
    assert isinstance(info, dict)


def test_a1_step_signature(env):
    env.reset(seed=0)
    out = env.step(np.zeros(2, dtype=np.float32))
    assert isinstance(out, tuple) and len(out) == 5, "step must return 5-tuple"
    obs, reward, terminated, truncated, info = out
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_a2_obs_dict_shape(env):
    obs, _ = env.reset(seed=0)
    # Required keys.
    for key in ("lidar", "state", "is_first", "is_terminal", "is_last"):
        assert key in obs, f"missing obs key {key!r}"
    # lidar: (1080,) float32, ∈ [0, 1]
    assert obs["lidar"].shape == (1080,)
    assert obs["lidar"].dtype == np.float32
    assert obs["lidar"].min() >= 0.0
    assert obs["lidar"].max() <= 1.0
    # state: (5,) float32
    assert obs["state"].shape == (5,)
    assert obs["state"].dtype == np.float32
    # boolean flags
    assert isinstance(obs["is_first"], bool) and obs["is_first"] is True
    assert isinstance(obs["is_terminal"], bool) and obs["is_terminal"] is False
    assert isinstance(obs["is_last"], bool) and obs["is_last"] is False

    # After one step, is_first must flip to False.
    obs2, *_ = env.step(np.zeros(2, dtype=np.float32))
    assert obs2["is_first"] is False
