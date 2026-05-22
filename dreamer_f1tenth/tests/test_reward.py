"""A_reward — Phase 4 reward + arclength lap 판정 (planning/005 §4-3·4-4 + 009 결정 A).

검증 항목:
  - progress reward: env-step당 arclength 증분(m), step-cap clip(0, 0.5), 후진=0.
  - reward = 분리 component 합 (A17: progress/collision/reverse/diverged/lap).
  - R_lap: 새 high-water lap마다 1회 (Map=25 / Osch=100), 경계 왕복 farming 방지.
  - lap_complete: arclength wrap 기반 2-lap (f110 lap_count 미사용, 009 §2).
  - collision/diverged 페널티 = -10.
  - windowed closest-point: 전역 argmin 점프 없이 국소 이동 (009 §2).
"""
import numpy as np
import pytest

from dreamer_f1tenth.envs import F110GymnasiumWrapper
from dreamer_f1tenth.envs.f1tenth_env import (
    PROGRESS_CAP, PENALTY_TERMINAL, ALPHA_PROGRESS, LAP_TARGET, S_MIN, V_MAX,
)
from pkg.drivers import GapFollower


def _gf_step(env, gf):
    scan = env._raw_obs["scans"][0]
    speed, steer = gf.process_lidar(scan)
    return env.step(np.array([steer, speed], dtype=np.float32))


def test_reward_components_sum_and_progress_capped():
    """reward == 분리 component 합. progress ∈ [0, cap]. total_arclen 누적."""
    env = F110GymnasiumWrapper(trackname="Oschersleben", max_episode_steps=2000,
                               ignore_first_collision=True)
    try:
        env.reset(seed=0)
        gf = GapFollower()
        prev_total = 0.0
        n = 0
        for _ in range(800):
            obs, r, term, trunc, info = _gf_step(env, gf)
            # component 분리 합 = reward
            comp = (info["reward_progress"] + info["reward_collision"]
                    + info["reward_reverse"] + info["reward_diverged"]
                    + info["reward_lap"])
            assert abs(r - comp) < 1e-6, f"reward {r} != component sum {comp}"
            # progress step-cap (α=1.0 이므로 동일 스케일)
            assert -1e-9 <= info["reward_progress"] <= ALPHA_PROGRESS * PROGRESS_CAP + 1e-6, (
                f"progress {info['reward_progress']} out of [0, {PROGRESS_CAP}]"
            )
            # 전진 정책 → total_arclen 단조 증가 경향 (감소해도 wrap 보정 내 소폭)
            assert info["total_arclen"] >= prev_total - PROGRESS_CAP - 1e-6
            prev_total = info["total_arclen"]
            n += 1
            if term or trunc:
                break
        assert n > 100, "GF가 충분히 진행해야 검증 의미 있음"
        assert prev_total > 5.0, f"전진 정책인데 누적 진행이 너무 작음: {prev_total}"
    finally:
        env.close()


def test_arclength_lap_complete_and_R_lap():
    """Oschersleben GF: arclength wrap으로 2-lap → cause='lap_complete'.
    R_lap 보너스는 lap당 정확히 1회, 값=R_lap(=100)."""
    env = F110GymnasiumWrapper(trackname="Oschersleben", max_episode_steps=6000,
                               ignore_first_collision=True)
    try:
        env.reset(seed=0)
        gf = GapFollower()
        lap_bonus_count = 0
        lap_bonus_vals = []
        cause = None
        for _ in range(6000):
            obs, r, term, trunc, info = _gf_step(env, gf)
            if info["reward_lap"] > 0:
                lap_bonus_count += 1
                lap_bonus_vals.append(info["reward_lap"])
            if term or trunc:
                cause = info["cause"]
                break
        assert cause == "lap_complete", f"expected lap_complete, got {cause!r}"
        assert info["lap_count_arc"] >= LAP_TARGET, info["lap_count_arc"]
        # lap 보너스 = LAP_TARGET회, 각 R_lap(=100)
        assert lap_bonus_count == LAP_TARGET, f"lap bonus {lap_bonus_count} != {LAP_TARGET}"
        assert all(abs(v - env.R_lap) < 1e-6 for v in lap_bonus_vals), lap_bonus_vals
        # 마지막 step에 lap_complete 보너스 포함, is_last
        assert obs["is_last"] is True
        assert info["reward_lap"] == pytest.approx(env.R_lap)
    finally:
        env.close()


def test_progress_sum_approx_arclength():
    """progress 보상 합(α=1) ≈ 실제 누적 주행거리 (step-cap·wrap 오차 내)."""
    env = F110GymnasiumWrapper(trackname="Oschersleben", max_episode_steps=4000,
                               ignore_first_collision=True)
    try:
        env.reset(seed=0)
        gf = GapFollower()
        prog_sum = 0.0
        for _ in range(4000):
            obs, r, term, trunc, info = _gf_step(env, gf)
            prog_sum += info["reward_progress"]
            if term or trunc:
                break
        # total_arclen은 부호 포함 누적, prog_sum은 순방향 clip 합 → 근사 일치
        rel_err = abs(prog_sum - info["total_arclen"]) / max(info["total_arclen"], 1.0)
        assert rel_err < 0.05, (
            f"progress_sum={prog_sum:.2f} vs total_arclen={info['total_arclen']:.2f} "
            f"(rel_err={rel_err:.3f})"
        )
    finally:
        env.close()


def test_collision_penalty_applied():
    """collision 종료 step에 reward_collision = -10, 다른 페널티 component는 0."""
    env = F110GymnasiumWrapper(trackname="Oschersleben", max_episode_steps=2000,
                               ignore_first_collision=True)
    try:
        env.reset(seed=0)
        action = np.array([S_MIN, V_MAX], dtype=np.float32)  # hard-left → wall
        for _ in range(2000):
            obs, r, term, trunc, info = env.step(action)
            if term:
                assert info["cause"] == "collision"
                assert info["reward_collision"] == pytest.approx(PENALTY_TERMINAL)
                assert info["reward_reverse"] == 0.0
                assert info["reward_diverged"] == 0.0
                assert info["reward_lap"] == 0.0
                # 종료 step reward = progress + 충돌 페널티
                assert r == pytest.approx(info["reward_progress"] + PENALTY_TERMINAL)
                break
        else:
            pytest.fail("collision 미발생")
    finally:
        env.close()


def test_windowed_closest_no_global_jump():
    """windowed closest_idx는 step간 국소 이동만 (전역 argmin 점프 없음, 009 §2)."""
    env = F110GymnasiumWrapper(trackname="map_easy3", max_episode_steps=1500,
                               ignore_first_collision=True)
    try:
        env.reset(seed=0)
        gf = GapFollower()
        n_pts = len(env._centerline_xy)
        prev_idx = env._closest_idx
        max_window = env._fwd_window + env._back_window
        for _ in range(1200):
            obs, r, term, trunc, info = _gf_step(env, gf)
            idx = info["closest_idx"]
            # 순환 인덱스 거리 (wrap 고려)
            d = abs(idx - prev_idx)
            d = min(d, n_pts - d)
            assert d <= max_window, (
                f"closest_idx 점프 {prev_idx}->{idx} (d={d} > window {max_window})"
            )
            prev_idx = idx
            if term or trunc:
                break
    finally:
        env.close()
