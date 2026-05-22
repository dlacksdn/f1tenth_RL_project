"""A-4 평가 게이트 집계·판정 순수함수 테스트 (019 §5-3, 020 §5(6)).

실모델/시뮬레이터 불요 — 합성 episode info 시퀀스(완주/충돌/timeout 혼합)로
aggregate_episodes / evaluate_gate / resolve_gates 로직만 검증.
"""
import pathlib
import sys

import pytest

# scripts/ 를 import 경로에 추가 (eval_gate는 vendor도 sys.path에 넣지만,
# 순수 함수만 쓰므로 numpy 외 무거운 의존성 없이 import 된다).
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import eval_gate as eg  # noqa: E402


# ---------------------------------------------------------------------------
# is_completed
# ---------------------------------------------------------------------------
def test_is_completed():
    assert eg.is_completed("lap_complete") is True
    assert eg.is_completed("collision") is False
    assert eg.is_completed("timeout") is False
    assert eg.is_completed(None) is False


# ---------------------------------------------------------------------------
# aggregate_episodes
# ---------------------------------------------------------------------------
def _ep(cause, laps=()):
    return {"cause": cause, "lap_times": list(laps)}


def test_aggregate_basic_completion_rate():
    # 8 완주 / 2 미완주 → 0.80
    eps = [_ep("lap_complete", [9.0, 10.0]) for _ in range(8)]
    eps += [_ep("collision"), _ep("timeout")]
    agg = eg.aggregate_episodes(eps)
    assert agg["n_episodes"] == 10
    assert agg["n_completed"] == 8
    assert agg["completion_rate"] == pytest.approx(0.80)
    assert agg["cause_counts"]["lap_complete"] == 8
    assert agg["cause_counts"]["collision"] == 1
    assert agg["cause_counts"]["timeout"] == 1


def test_aggregate_lap_population_median_best():
    # 완주 ep만 lap 모집단에 기여. 미완주 ep의 lap_times는 무시.
    eps = [
        _ep("lap_complete", [8.0, 12.0]),
        _ep("lap_complete", [10.0, 10.0]),
        _ep("collision", [99.0]),  # 미완주 → 모집단 제외
        _ep("timeout"),
    ]
    agg = eg.aggregate_episodes(eps)
    # 모집단 = [8,12,10,10] → median=10.0, best=8.0, n_laps=4
    assert agg["n_laps"] == 4
    assert agg["lap_median"] == pytest.approx(10.0)
    assert agg["lap_best"] == pytest.approx(8.0)


def test_aggregate_no_completion():
    eps = [_ep("collision"), _ep("timeout"), _ep("diverged")]
    agg = eg.aggregate_episodes(eps)
    assert agg["completion_rate"] == 0.0
    assert agg["n_laps"] == 0
    assert agg["lap_median"] is None
    assert agg["lap_best"] is None


def test_aggregate_zero_lap_times_ignored():
    # log_lap_time_s==0 step 값(미완주 lap)은 모집단에서 제외.
    eps = [_ep("lap_complete", [0.0, 9.5])]
    agg = eg.aggregate_episodes(eps)
    assert agg["n_laps"] == 1
    assert agg["lap_best"] == pytest.approx(9.5)


# ---------------------------------------------------------------------------
# evaluate_gate
# ---------------------------------------------------------------------------
def test_a11_completion_only_pass():
    # A11: 완주율 0.80 ≥ 0.80 → PASS. lap median 게이트 없음(느려도 무관).
    eps = [_ep("lap_complete", [18.0, 19.0]) for _ in range(16)]
    eps += [_ep("collision") for _ in range(4)]
    agg = eg.aggregate_episodes(eps)
    res = eg.evaluate_gate("A11", agg)
    assert res["passed"] is True
    # completion만 체크, lap_median/best 체크 없음(completion-only).
    metrics = {c["metric"] for c in res["checks"]}
    assert metrics == {"completion_rate"}


def test_a11_completion_only_fail_below_threshold():
    eps = [_ep("lap_complete", [9.0, 9.0]) for _ in range(15)]
    eps += [_ep("collision") for _ in range(5)]  # 0.75 < 0.80
    agg = eg.aggregate_episodes(eps)
    res = eg.evaluate_gate("A11", agg)
    assert res["passed"] is False


def test_a16_relaxed_threshold():
    # 완주율 0.75 → A11(0.80) FAIL, A16(0.70) PASS.
    eps = [_ep("lap_complete", [9.0, 9.0]) for _ in range(15)]
    eps += [_ep("timeout") for _ in range(5)]
    agg = eg.aggregate_episodes(eps)
    assert eg.evaluate_gate("A11", agg)["passed"] is False
    assert eg.evaluate_gate("A16", agg)["passed"] is True


def test_a12_completion_only():
    eps = [_ep("lap_complete", [105.0, 108.0]) for _ in range(17)]
    eps += [_ep("collision") for _ in range(3)]  # 0.85 ≥ 0.80
    agg = eg.aggregate_episodes(eps)
    res = eg.evaluate_gate("A12", agg)
    assert res["passed"] is True
    assert {c["metric"] for c in res["checks"]} == {"completion_rate"}


def test_a13_lap_time_gate_pass():
    # A13: median ≤ 120 ∧ best ≤ 110. 완주율은 A13 미적용.
    eps = [_ep("lap_complete", [108.0, 118.0]) for _ in range(5)]
    agg = eg.aggregate_episodes(eps)
    res = eg.evaluate_gate("A13", agg)
    # median=113, best=108 → median≤120 OK, best≤110 OK
    assert res["passed"] is True
    assert {c["metric"] for c in res["checks"]} == {"lap_median", "lap_best"}


def test_a13_fail_best_too_slow():
    # best=115 > 110 → FAIL (median=120는 경계 통과).
    eps = [_ep("lap_complete", [115.0, 120.0]) for _ in range(5)]
    agg = eg.aggregate_episodes(eps)
    res = eg.evaluate_gate("A13", agg)
    assert res["passed"] is False
    best_check = next(c for c in res["checks"] if c["metric"] == "lap_best")
    assert best_check["ok"] is False


def test_a13_fail_no_completion():
    # 완주 0 → lap_median/best None → A13 FAIL (None은 통과 못 함).
    eps = [_ep("collision") for _ in range(5)]
    agg = eg.aggregate_episodes(eps)
    res = eg.evaluate_gate("A13", agg)
    assert res["passed"] is False


def test_gate_boundary_inclusive():
    # 임계 경계: 완주율 정확히 0.80 → A11 PASS (>= 포함).
    eps = [_ep("lap_complete", [9.0]) for _ in range(4)]
    eps += [_ep("collision")]  # 0.80
    agg = eg.aggregate_episodes(eps)
    assert eg.evaluate_gate("A11", agg)["passed"] is True


# ---------------------------------------------------------------------------
# resolve_gates
# ---------------------------------------------------------------------------
def test_resolve_gates_defaults():
    assert eg.resolve_gates("f1tenth_map_easy3", None) == ["A11"]
    assert eg.resolve_gates("f1tenth_Oschersleben", None) == ["A12", "A13"]


def test_resolve_gates_explicit():
    assert eg.resolve_gates("f1tenth_map_easy3", "A11,A16") == ["A11", "A16"]


def test_resolve_gates_unknown():
    with pytest.raises(ValueError):
        eg.resolve_gates("f1tenth_map_easy3", "A99")


def test_resolve_gates_task_mismatch():
    # A13은 Oschersleben 전용 → map_easy3 task에 지정 시 오류.
    with pytest.raises(ValueError):
        eg.resolve_gates("f1tenth_map_easy3", "A13")
