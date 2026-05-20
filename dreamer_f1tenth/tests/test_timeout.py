"""A5 — max_episode_steps 도달 시 truncated=True ∧ terminated=False.

Use a small max_episode_steps for fast test. Action = zero (steer=0, speed=0)
keeps the car stationary → no collision, no lap → must trigger timeout.
"""
import numpy as np

from dreamer_f1tenth.envs import F110GymnasiumWrapper


def test_a5_timeout_truncates():
    LIMIT = 50  # fast test; semantics identical to 9000.
    env = F110GymnasiumWrapper(trackname="Oschersleben", max_episode_steps=LIMIT,
                               ignore_first_collision=True)
    try:
        env.reset(seed=0)
        action = np.zeros(2, dtype=np.float32)
        terminated = False
        truncated = False
        steps = 0
        for _ in range(LIMIT * 2):
            obs, r, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        assert not terminated, "stationary policy should not terminate"
        assert truncated, f"expected truncated=True at step {LIMIT}, got steps={steps}"
        assert info["cause"] == "timeout"
        assert info["env_step"] == LIMIT
        assert steps == LIMIT
    finally:
        env.close()
