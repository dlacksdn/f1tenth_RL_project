"""발산 가드 검증 (notes/smoke_findings #4 RESOLVED).

f110 ST dynamics가 수치 발산해 raw obs가 inf/huge가 되어도, wrapper가
(1) obs를 항상 finite·bounded로 정화하고 (2) 'diverged' 종료를 내야 한다.
이게 깨지면 replay buffer가 오염돼 인코더 overflow→NaN logit으로 학습이 죽는다.
"""
import numpy as np
import pytest

from dreamer_f1tenth.envs import F110GymnasiumWrapper
from dreamer_f1tenth.envs.f1tenth_env import NUM_BEAMS, _VEL_DIVERGE, _STATE_CLIP


@pytest.fixture
def env():
    e = F110GymnasiumWrapper(trackname="map_easy3", ignore_first_collision=True)
    e.reset(seed=0)
    yield e
    e.close()


def _fake_raw(base, **overrides):
    """현재 raw를 복사해 일부 필드를 오염시킨다."""
    raw = {k: np.array(v, dtype=np.float64, copy=True) for k, v in base.items()}
    for k, v in overrides.items():
        raw[k] = np.array(v, dtype=np.float64)
    return raw


def test_build_obs_sanitizes_inf_nan(env):
    base = env._raw_obs
    # vel_x = +inf, vel_y = nan, scan에 inf 섞기
    bad_scan = np.array(base["scans"][0], dtype=np.float64, copy=True)
    bad_scan[::100] = np.inf
    bad_scan[1] = np.nan
    raw = _fake_raw(base, linear_vels_x=[np.inf], linear_vels_y=[np.nan],
                    ang_vels_z=[1e40], scans=[bad_scan])
    obs = env._build_obs(raw)
    assert np.isfinite(obs["lidar"]).all(), "lidar에 비유한 값 잔존"
    assert np.isfinite(obs["state"]).all(), "state에 비유한 값 잔존"
    assert obs["lidar"].shape == (NUM_BEAMS,)
    assert (obs["lidar"] >= 0).all() and (obs["lidar"] <= 1).all()
    assert (np.abs(obs["state"]) <= _STATE_CLIP + 1e-6).all(), "state clip 위반"


def test_step_terminates_on_divergence(env):
    # raw를 발산값으로 바꾼 뒤 zero-action step → 'diverged' 최우선 종료.
    # step()은 내부에서 self._env.step을 호출하므로, 그 결과를 발산값으로 가로채기 위해
    # _raw_obs를 직접 오염시키는 대신 base env.step을 monkeypatch.
    base = {k: np.array(v, dtype=np.float64, copy=True)
            for k, v in env._raw_obs.items()}
    base["linear_vels_x"] = np.array([_VEL_DIVERGE * 10.0])

    def fake_step(action_2d):
        return base, 0.0, False, {}

    env._env.step = fake_step  # type: ignore
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], np.float32))
    assert terminated is True
    assert info["cause"] == "diverged"
    assert np.isfinite(obs["state"]).all()
    assert np.isfinite(obs["lidar"]).all()
