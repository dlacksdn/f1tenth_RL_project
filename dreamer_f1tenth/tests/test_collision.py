"""A4 — collision → terminated=True, info['cause']='collision'.

Force a collision by holding maximum left steering at high speed; the car
quickly drives into a wall on Oschersleben from default pose.
"""
import numpy as np

from dreamer_f1tenth.envs import F110GymnasiumWrapper
from dreamer_f1tenth.envs.f1tenth_env import S_MIN, V_MAX


def test_a4_collision_terminates():
    env = F110GymnasiumWrapper(trackname="Oschersleben", max_episode_steps=2000,
                               ignore_first_collision=True)
    try:
        env.reset(seed=0)
        action = np.array([S_MIN, V_MAX], dtype=np.float32)  # hard-left, max speed
        terminated = False
        cause = None
        for _ in range(2000):
            obs, r, terminated, truncated, info = env.step(action)
            if terminated:
                cause = info["cause"]
                assert obs["is_terminal"] is True
                break
            if truncated:
                break
        assert terminated, "expected a collision-induced terminated=True"
        assert cause == "collision", f"expected cause='collision', got {cause!r}"
    finally:
        env.close()
