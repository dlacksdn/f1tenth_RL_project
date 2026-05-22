"""A-1 snapshot 유틸 테스트 (020 §2, 019 §2-5).

- bin 경계 분류((0,T] n등분 우상향 폐구간).
- partial(inference-only) state_dict: _wm.* + _task_behavior.actor.* 만, value/_slow_value/
  _world_model(B-2 중복)/optim 제외 + reload 동작.
- per-lap lap_time 수집, diversity bin-best 교체/옛 파일 정리, interval 저장, threshold 조회.
"""
import glob
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

import snapshot_utils as su  # vendor/dreamerv3-torch (conftest sys.path)


# ---------------------------------------------------------------------------
# bin 분류
# ---------------------------------------------------------------------------
def test_resolve_track_value():
    bw = {"map_easy3": 1.0, "oschersleben": 10.0}
    assert su.resolve_track_value(bw, "map_easy3") == 1.0
    assert su.resolve_track_value(bw, "Oschersleben") == 10.0  # 대문자 → lower 매칭
    assert su.resolve_track_value(bw, "unknown") is None
    assert su.resolve_track_value(10.0, "x") == 10.0           # 스칼라 그대로
    assert su.resolve_track_value(None, "x") is None


def test_lap_time_bin_1s_width_map_easy3():
    W, MAX = 1.0, 20.0  # map_easy3 실측 8~13s: (7,8]→8, (8,9]→9, (9,10]→10
    assert su.lap_time_bin(8.06, W, MAX) == 9    # ceil(8.06)=9
    assert su.lap_time_bin(8.0, W, MAX) == 8     # 경계 우상향 폐구간
    assert su.lap_time_bin(10.28, W, MAX) == 11  # ceil(10.28)=11
    assert su.lap_time_bin(20.0, W, MAX) == 20
    assert su.lap_time_bin(20.1, W, MAX) is None


def test_lap_time_bin_10s_fixed_width():
    W, MAX = 10.0, 110.0  # (0,10]→10,(10,20]→20,...,(100,110]→110 (상한 label)
    assert su.lap_time_bin(0.0, W, MAX) is None
    assert su.lap_time_bin(-1.0, W, MAX) is None
    assert su.lap_time_bin(0.1, W, MAX) == 10
    assert su.lap_time_bin(10.0, W, MAX) == 10     # 경계 우상향 폐구간
    assert su.lap_time_bin(10.01, W, MAX) == 20
    assert su.lap_time_bin(12.3, W, MAX) == 20
    assert su.lap_time_bin(90.0, W, MAX) == 90
    assert su.lap_time_bin(90.1, W, MAX) == 100
    assert su.lap_time_bin(105.0, W, MAX) == 110
    assert su.lap_time_bin(110.0, W, MAX) == 110
    assert su.lap_time_bin(110.1, W, MAX) is None  # lap_max 초과 → 미저장


# ---------------------------------------------------------------------------
# partial state_dict
# ---------------------------------------------------------------------------
class _DummyAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self._wm = nn.Linear(2, 2)
        tb = nn.Module()
        tb.actor = nn.Linear(2, 2)
        tb.value = nn.Linear(2, 2)
        tb._slow_value = nn.Linear(2, 2)
        tb._world_model = self._wm  # B-2: world model 공유(_wm.* 와 동일 텐서)
        self._task_behavior = tb


def test_inference_state_dict_whitelist():
    agent = _DummyAgent()
    sd = su.inference_state_dict(agent)
    keys = set(sd.keys())
    # 포함: _wm.* + _task_behavior.actor.*
    assert any(k.startswith("_wm.") for k in keys)
    assert any(k.startswith("_task_behavior.actor.") for k in keys)
    # 제외: value / _slow_value / _world_model(B-2 중복) — 추론 불요
    assert not any(k.startswith("_task_behavior.value.") for k in keys)
    assert not any(k.startswith("_task_behavior._slow_value.") for k in keys)
    assert not any(k.startswith("_task_behavior._world_model.") for k in keys)


def test_save_inference_only_reload(tmp_path):
    agent = _DummyAgent()
    path = tmp_path / "policy.pt"
    su.save_inference_only(agent, path)
    assert path.exists()
    ckpt = torch.load(str(path))
    assert "agent_state_dict" in ckpt
    sd = ckpt["agent_state_dict"]
    full = agent.state_dict()
    for k, v in sd.items():
        assert torch.equal(v, full[k])  # reload 값 일치
    # optim 키 부재(애초에 저장 안 함)
    assert "optims_state_dict" not in ckpt


# ---------------------------------------------------------------------------
# lap_time 수집 / diversity / interval
# ---------------------------------------------------------------------------
def test_collect_lap_times_from_episode():
    ep = {"log_lap_time_s": np.array([0, 0, 12.3, 0, 11.8], np.float32)}
    laps = su.collect_lap_times_from_episode(ep)
    assert laps == pytest.approx([12.3, 11.8], abs=1e-3)
    assert su.collect_lap_times_from_episode({"reward": np.zeros(3)}) == []


def test_update_diversity_snapshots_10s_bin_and_best(tmp_path):
    agent = _DummyAgent()
    bins, best = {}, {}
    W, MAX = 10.0, 110.0
    # 8.0 → bin label 10, 동시에 global best
    su.update_diversity_snapshots(agent, [8.0], bins, best, tmp_path,
                                  bin_width=W, lap_max=MAX, step_k=10)
    assert 10 in bins and bins[10]["lap_time"] == 8.0
    assert best["lap_time"] == 8.0
    f1 = bins[10]["path"]
    assert os.path.exists(f1) and os.path.exists(best["path"])
    # 같은 bin 더 느린 8.5 무시(bin·best 둘 다 유지)
    su.update_diversity_snapshots(agent, [8.5], bins, best, tmp_path,
                                  bin_width=W, lap_max=MAX, step_k=20)
    assert bins[10]["lap_time"] == 8.0 and best["lap_time"] == 8.0
    # 더 빠른 7.0 → bin10 교체 + best 갱신 + 옛 파일 삭제
    old_best = best["path"]
    su.update_diversity_snapshots(agent, [7.0], bins, best, tmp_path,
                                  bin_width=W, lap_max=MAX, step_k=30)
    assert bins[10]["lap_time"] == 7.0 and best["lap_time"] == 7.0
    assert not os.path.exists(f1) and not os.path.exists(old_best)
    # 느린 95.0 → bin label 100, best는 7.0 유지(갱신 X)
    su.update_diversity_snapshots(agent, [95.0], bins, best, tmp_path,
                                  bin_width=W, lap_max=MAX, step_k=40)
    assert 100 in bins and bins[100]["lap_time"] == 95.0
    assert best["lap_time"] == 7.0
    # lap_max 초과(110.1)는 미저장
    su.update_diversity_snapshots(agent, [110.1], bins, best, tmp_path,
                                  bin_width=W, lap_max=MAX, step_k=50)
    assert set(bins.keys()) == {10, 100}


def test_save_interval_snapshot(tmp_path):
    items = {"agent_state_dict": {"x": torch.zeros(2)}}
    p = su.save_interval_snapshot(items, tmp_path, step_k=130, keep=True)
    assert p.name == "step_130k.pt"
    assert p.exists()
    assert su.save_interval_snapshot(items, tmp_path, step_k=131, keep=False) is None
