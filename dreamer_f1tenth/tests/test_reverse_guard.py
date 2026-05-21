"""A18 — reverse_guard: 강제 후진 액션 1.1초 → terminated ∧ cause=='reverse'.

reverse_guard (v3 §3 1-4, decision #8/#24): centerline tangent · world-frame
velocity < 0 (후진) 가 REVERSE_COUNTER_LIMIT(=50 env step = 1s @ action_repeat=2)
연속 지속 시 terminated, cause='reverse'. dot ≥ 0 (전진·정지) 이면 counter reset.

map_easy3(개방 구조, 97.6% free; implementation/002)에서 후진 → 벽 충돌 전에
reverse 트리거. counter reset은 전진 1초로 검증.
"""
import numpy as np

from dreamer_f1tenth.envs import F110GymnasiumWrapper
from dreamer_f1tenth.envs.f1tenth_env import REVERSE_COUNTER_LIMIT


def test_a18_forced_reverse_terminates():
    env = F110GymnasiumWrapper(trackname="map_easy3", ignore_first_collision=True)
    try:
        env.reset(seed=0)
        action = np.array([0.0, -3.0], dtype=np.float32)  # steer 0, speed -3 (후진)
        terminated = truncated = False
        info = {}
        steps = 0
        # 1.1s = 55 env step. 여유로 60까지 돌리되 종료 시 break.
        for _ in range(60):
            obs, r, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        assert terminated, f"forced reverse should terminate; steps={steps}, info={info}"
        assert info["cause"] == "reverse", (
            f"expected cause='reverse', got {info['cause']!r} "
            f"(collision_raw={info.get('collision_raw')}, steps={steps})"
        )
        # 1.1초(=55 env step) 이내 종료, counter는 정확히 LIMIT 도달.
        assert steps <= 55, f"reverse must trigger within 1.1s (55 step), got {steps}"
        assert info["reverse_counter"] == REVERSE_COUNTER_LIMIT, (
            f"counter at termination should be {REVERSE_COUNTER_LIMIT}, "
            f"got {info['reverse_counter']}"
        )
        print(
            f"[A18] reverse terminated at env_step={steps}, "
            f"reverse_counter={info['reverse_counter']}, cause={info['cause']!r}"
        )
    finally:
        env.close()


def test_a18_forward_resets_counter():
    """전진 1초 → counter=0 유지, reverse로 종료되지 않음 (정지·전진은 후진 아님)."""
    env = F110GymnasiumWrapper(trackname="map_easy3", ignore_first_collision=True)
    try:
        env.reset(seed=0)
        action = np.array([0.0, 3.0], dtype=np.float32)  # steer 0, speed +3 (전진)
        max_counter = 0
        reverse_terminated = False
        for _ in range(REVERSE_COUNTER_LIMIT):
            obs, r, terminated, truncated, info = env.step(action)
            max_counter = max(max_counter, info["reverse_counter"])
            if terminated and info["cause"] == "reverse":
                reverse_terminated = True
            if terminated or truncated:
                break
        assert not reverse_terminated, "forward driving must not trigger reverse guard"
        assert max_counter == 0, (
            f"reverse_counter must stay 0 while moving forward, peaked at {max_counter}"
        )
        print(f"[A18-reset] forward 1s: max reverse_counter={max_counter} (expect 0)")
    finally:
        env.close()
