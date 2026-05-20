"""A6 (v3 §2 line 163): base_classes.py:488 dynamic patch 후 코너링 시 |vel_y|.max > 0.05 검증.

시나리오: 직선 가속 (1초) → mild 코너링 (1초).
- dynamic_models.py:152 임계 `abs(x[3]) < 0.5` → vel > 0.5 m/s면 dynamic mode 활성
  (v_switch=7.319는 accl_constraints의 가속 제한일 뿐, kinematic 전환 임계가 아님).
- 직선 phase: slip_angle ≈ 0 → vel_y ≈ 0
- 코너 phase: slip_angle ≠ 0 → |vel_y| > 0.05

참조: v3 §1-C #27, planning/005 line 226-227. base_classes.py:488 `linear_vels_y=0`
→ `agent.state[3] * np.sin(agent.state[6])` (vel * sin(slip_angle)).
"""
import numpy as np

from dreamer_f1tenth.envs import F110GymnasiumWrapper


def test_a6_dynamic_vel_y_nonzero_on_cornering():
    env = F110GymnasiumWrapper("Oschersleben", action_repeat=2)
    _obs, _info = env.reset()

    # Phase 1: 직선 가속 (steer=0, speed=10) — 1초 = 50 env step (action_repeat=2 @ 0.01s)
    straight_vel_y = []
    for _ in range(50):
        _o, _r, term, _t, info = env.step(np.array([0.0, 10.0], dtype=np.float32))
        straight_vel_y.append(float(env._raw_obs["linear_vels_y"][0]))
        assert not term, f"unexpected terminated in straight phase: {info}"
    final_speed_straight = float(env._raw_obs["linear_vels_x"][0])
    straight_max = max(abs(v) for v in straight_vel_y)

    # Phase 2: mild 코너링 (steer=0.15 rad, speed=10) — 1초 = 50 env step
    corner_vel_y = []
    corner_slip = []
    for _ in range(50):
        _o, _r, term, trunc, _i = env.step(np.array([0.15, 10.0], dtype=np.float32))
        corner_vel_y.append(float(env._raw_obs["linear_vels_y"][0]))
        # raw state[6] = slip_angle for diagnostics
        agent_state = env._env.sim.agents[0].state
        corner_slip.append(float(agent_state[6]))
        if term or trunc:
            break

    assert len(corner_vel_y) > 0, "no corner samples collected"
    corner_max = max(abs(v) for v in corner_vel_y)
    slip_max = max(abs(s) for s in corner_slip)

    print(
        f"\n[A6] straight: speed_end={final_speed_straight:.3f}m/s "
        f"(dynamic threshold=0.5), |vel_y|.max={straight_max:.4f}"
    )
    print(
        f"[A6] corner:   |vel_y|.max={corner_max:.4f}, |slip|.max={slip_max:.4f}rad "
        f"(samples={len(corner_vel_y)})"
    )

    assert final_speed_straight > 0.5, (
        f"speed didn't exceed dynamic threshold 0.5 (got {final_speed_straight:.3f})."
    )
    assert corner_max > 0.05, (
        f"|vel_y|.max={corner_max:.4f} <= 0.05 — base_classes.py:488 patch ineffective "
        "(check vehicle_dynamics_st state[6]=slip_angle)."
    )

    env.close()
