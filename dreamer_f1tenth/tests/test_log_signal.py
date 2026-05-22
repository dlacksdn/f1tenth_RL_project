"""log_ 진단 신호 도달 테스트 (020 §1 B-1, 경로①).

검증 대상:
1. env reset/step obs가 동일한 log_ 키 집합을 갖는다(simulate cache 키 일관성 함정).
2. per-lap lap_time_s 산출이 Δenv_step × DT_WRAP(=SIM_TIMESTEP×action_repeat)와 일치.
3. ★ 핵심: tools.simulate → save_episodes(npz)에 log_lap_time_s가 보존된다
   (tools.py:205 save가 :213-219 log_ pop보다 앞 → npz엔 남음). 학습 중(dreamer.py)
   eval/train 신호 수집이 조용히 실패하지 않음을 보장.
4. log_ 키가 agent(encoder) 입력 및 sample_episodes 배치에서 strip된다(world model 무영향).
"""
import os
import glob
from collections import OrderedDict

import numpy as np
import pytest

import tools  # vendor/dreamerv3-torch (conftest sys.path)
from parallel import Damy

from dreamer_f1tenth.envs.f1tenth_env import (
    F110GymnasiumWrapper,
    SIM_TIMESTEP,
    DEFAULT_ACTION_REPEAT,
)

_LOG_KEYS = (
    "log_lap_time_s",
    "log_reward_progress",
    "log_reward_collision",
    "log_reward_reverse",
    "log_reward_diverged",
    "log_reward_lap",
    "log_lap_count_arc",
    "log_completed",
)


# ---------------------------------------------------------------------------
# 1·2. 실제 env: log_ 키 일관성 + lap_time 산출식
# ---------------------------------------------------------------------------
def test_env_log_keys_consistent_reset_and_step():
    env = F110GymnasiumWrapper(trackname="map_easy3")
    obs_reset, _ = env.reset(seed=0)
    # observation_space는 5-key 유지(log_ 미포함, 020 §1 함정).
    assert set(env.observation_space.spaces.keys()) == {
        "lidar", "state", "is_first", "is_terminal", "is_last"
    }
    # reset obs에 모든 log_ 키가 기본 0.0으로 존재(is_first transition 키 일관성).
    for k in _LOG_KEYS:
        assert k in obs_reset, f"reset obs missing {k!r}"
        assert np.asarray(obs_reset[k]).dtype == np.float32
        assert float(obs_reset[k]) == 0.0
    obs_step, _r, _term, _trunc, _info = env.step(np.zeros(2, dtype=np.float32))
    # step obs도 동일 키 집합 → 전 transition 동일 키.
    assert set(obs_reset.keys()) == set(obs_step.keys())
    for k in _LOG_KEYS:
        assert k in obs_step
    env.close()


def test_lap_time_derivation_matches_dt_wrap():
    env = F110GymnasiumWrapper(trackname="map_easy3")
    # DT_WRAP는 하드코딩 0.02가 아니라 SIM_TIMESTEP×action_repeat로 도출(020 §3-4).
    assert env._dt_wrap == pytest.approx(SIM_TIMESTEP * DEFAULT_ACTION_REPEAT)
    assert env._dt_wrap == pytest.approx(0.02)
    env.close()


# ---------------------------------------------------------------------------
# 3. ★ 도달: simulate → npz 보존 (dummy env, 경량)
# ---------------------------------------------------------------------------
class _DummyLogEnv:
    """dreamerv3-internal 컨벤션의 최소 env. 3 step 후 done, 마지막 step에서
    log_lap_time_s에 알려진 값을 주입한다(완주 lap 모사)."""

    def __init__(self, known_lap_time, ep_len=3):
        self.id = "dummy-ep-0"
        self._known = float(known_lap_time)
        self._ep_len = ep_len
        self._t = 0

    def _obs(self, is_first, is_last, lap_time):
        o = OrderedDict([
            ("lidar", np.zeros(1080, np.float32)),
            ("state", np.zeros(5, np.float32)),
            ("is_first", bool(is_first)),
            ("is_terminal", False),
            ("is_last", bool(is_last)),
        ])
        for k in _LOG_KEYS:
            o[k] = np.float32(0.0)
        o["log_lap_time_s"] = np.float32(lap_time)
        return o

    def reset(self):
        self._t = 0
        return self._obs(is_first=True, is_last=False, lap_time=0.0)

    def step(self, action):
        self._t += 1
        done = self._t >= self._ep_len
        lap_time = self._known if done else 0.0
        obs = self._obs(is_first=False, is_last=done, lap_time=lap_time)
        info = {"discount": np.array(1.0, dtype=np.float32)}
        return obs, 1.0, done, info

    def close(self):
        pass


class _FakeLogger:
    def __init__(self):
        self.step = 0

    def scalar(self, *a, **k):
        pass

    def write(self, *a, **k):
        pass

    def video(self, *a, **k):
        pass


def _dummy_agent(obs, done, state):
    # simulate가 log_ strip 후 stacked obs를 넘긴다 → log_ 키가 없어야 함(검증).
    assert all("log_" not in k for k in obs), "log_ key leaked into agent input"
    n = len(next(iter(obs.values())))
    return np.zeros((n, 2), dtype=np.float32), None


def test_log_lap_time_reaches_npz(tmp_path):
    KNOWN = 12.34
    env = Damy(_DummyLogEnv(known_lap_time=KNOWN))
    cache = {}
    logger = _FakeLogger()
    tools.simulate(
        _dummy_agent, [env], cache, str(tmp_path), logger,
        is_eval=True, episodes=1,
    )
    npzs = glob.glob(os.path.join(str(tmp_path), "*.npz"))
    assert len(npzs) == 1, f"expected 1 saved episode npz, got {npzs}"
    data = np.load(npzs[0])
    # ★ log_lap_time_s가 npz에 보존됐는가(B-1 도달의 핵심).
    assert "log_lap_time_s" in data.files, (
        f"log_lap_time_s lost before npz save; keys={data.files}"
    )
    arr = data["log_lap_time_s"]
    assert float(arr.max()) == pytest.approx(KNOWN, abs=1e-3), (
        f"injected lap_time {KNOWN} not preserved; arr={arr}"
    )
    # reward component 채널도 함께 보존(A-5 경로 동일).
    assert "log_reward_progress" in data.files


def test_log_keys_stripped_from_sample_episodes():
    """sample_episodes가 log_ 키를 strip → world model 배치 입력 무영향(tools.py:347/360)."""
    ep = {
        "lidar": np.zeros((5, 1080), np.float32),
        "state": np.zeros((5, 5), np.float32),
        "is_first": np.array([True, False, False, False, False]),
        "is_terminal": np.zeros(5, bool),
        "is_last": np.array([False, False, False, False, True]),
        "action": np.zeros((5, 2), np.float32),
        "reward": np.ones(5, np.float32),
        "discount": np.ones(5, np.float32),
        "log_lap_time_s": np.array([0, 0, 0, 0, 12.34], np.float32),
    }
    gen = tools.sample_episodes({"e": ep}, length=3)
    batch = next(gen)
    assert all("log_" not in k for k in batch), f"log_ leaked: {list(batch)}"
