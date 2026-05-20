"""A3 — 1 episode no-error smoke test (GapFollower policy on Oschersleben)."""
import numpy as np
import pytest

from dreamer_f1tenth.envs import F110GymnasiumWrapper
from pkg.drivers import GapFollower


@pytest.mark.parametrize("trackname", ["Oschersleben"])
def test_a3_smoke_one_episode(trackname):
    env = F110GymnasiumWrapper(trackname=trackname, max_episode_steps=2000)
    try:
        obs, info = env.reset(seed=0)
        gf = GapFollower()
        # raw LiDAR is what GF needs — but our obs is normalized.
        # Reconstruct raw via env._raw_obs (test-only access).
        steps = 0
        while True:
            raw_scan = env._raw_obs["scans"][0]
            speed, steer = gf.process_lidar(raw_scan)
            obs, r, term, trunc, info = env.step(np.array([steer, speed], dtype=np.float32))
            steps += 1
            assert obs["lidar"].shape == (1080,)
            if term or trunc:
                break
        assert steps > 0
    finally:
        env.close()
