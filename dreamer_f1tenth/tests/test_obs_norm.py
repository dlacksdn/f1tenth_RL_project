"""A_norm — state 정규화 saturation 검증 (v3 결정 #15).

기준 (Oschersleben, GapFollower policy):
  - vel_x / 20  ∈  [-0.25, +1.05]   at 99%-ile (저속 후진 -5 m/s ~ +20 m/s 여유)
  - |vel_y / 5|  ≤  1.0   at 99%-ile  (Phase 1-3 patch 전엔 항상 0 → 자동 만족)
  - |ang_vel_z / (2π)|  ≤  1.0   at 99%-ile

v3 사양은 100 ep × 9000 env step. CI/로컬 회귀용으로는 default 3 ep × 1500 step (~수십초)
로 동작하고, `RUN_FULL_OBS_NORM=1` env var로 100 ep × 9000 step 풀 모드 활성.
"""
import os

import numpy as np
import pytest

from dreamer_f1tenth.envs import F110GymnasiumWrapper
from pkg.drivers import GapFollower


def _collect(trackname, n_episodes, max_steps):
    env = F110GymnasiumWrapper(trackname=trackname, max_episode_steps=max_steps,
                               ignore_first_collision=True)
    vel_x_norm = []
    vel_y_norm = []
    ang_z_norm = []
    try:
        for ep in range(n_episodes):
            env.reset(seed=ep)
            gf = GapFollower()
            for _ in range(max_steps):
                raw_scan = env._raw_obs["scans"][0]
                speed, steer = gf.process_lidar(raw_scan)
                obs, _r, term, trunc, _i = env.step(np.array([steer, speed], dtype=np.float32))
                # state = [vel_x/20, vel_y/5, ang_vel_z/(2π), prev_steer/0.4189, prev_speed/20]
                # (ang_vel_z scale은 implementation/005 §2-1에서 /π → /(2π)로 갱신)
                vel_x_norm.append(float(obs["state"][0]))
                vel_y_norm.append(float(obs["state"][1]))
                ang_z_norm.append(float(obs["state"][2]))
                if term or trunc:
                    break
    finally:
        env.close()
    return (np.asarray(vel_x_norm), np.asarray(vel_y_norm), np.asarray(ang_z_norm))


def _q(arr, q):
    return float(np.quantile(arr, q))


@pytest.mark.parametrize("trackname", ["Oschersleben"])
def test_a_norm_quick(trackname):
    full = os.environ.get("RUN_FULL_OBS_NORM") == "1"
    # Quick: 5 ep × 9000 step ≈ 15K samples (Oschersleben GF는 ~1500 step에 lap_complete).
    # Full: v3 명세 그대로 100 ep × 9000 step.
    n_ep = 100 if full else 5
    max_steps = 9000

    vx, vy, az = _collect(trackname, n_ep, max_steps)
    assert vx.size > 0, "no samples collected"

    # 99%-ile bounds.
    vx_lo, vx_hi = _q(vx, 0.005), _q(vx, 0.995)
    vy_abs_hi = float(np.quantile(np.abs(vy), 0.99))
    az_abs_hi = float(np.quantile(np.abs(az), 0.99))

    # Tolerances per v3 결정 #15.
    assert -0.25 <= vx_lo, f"vel_x/20 99% lower {vx_lo:.3f} below -0.25"
    assert vx_hi <= 1.05, f"vel_x/20 99% upper {vx_hi:.3f} above 1.05"
    assert vy_abs_hi <= 1.0, f"|vel_y/5| 99% {vy_abs_hi:.3f} above 1.0"
    assert az_abs_hi <= 1.0, f"|ang_vel_z/(2π)| 99% {az_abs_hi:.3f} above 1.0"

    # Diagnostic print (visible with -s).
    print(f"\n[A_norm/{trackname}] N={vx.size}  "
          f"vel_x/20 0.5%={vx_lo:.3f} 99.5%={vx_hi:.3f}  "
          f"|vel_y/5| 99%={vy_abs_hi:.3f}  |ang_z/(2π)| 99%={az_abs_hi:.3f}")
