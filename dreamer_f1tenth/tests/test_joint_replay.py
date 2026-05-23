"""A-2 joint replay 유틸 테스트 (019 §3-3, 020 §3-2).

stage2_utils.joint_episode_generator / make_joint_dataset:
- ratio=0/1/0.3 에서 두 풀(Stage1=old / 현 트랙=new) 샘플 비율이 시드 고정 하 통계적으로 맞는지.
- 재현성(같은 seed → 같은 시퀀스).
- make_joint_dataset 가 tools.sample_episodes/from_generator 와 호환되는지(실 episode dict).
"""
import threading

import numpy as np
import pytest

import stage2_utils as s2  # vendor/dreamerv3-torch (conftest sys.path)


def _run_with_timeout(fn, timeout=10.0):
    """fn()을 daemon 스레드에서 실행. timeout 내 미완료면 hang(무한 루프)으로 판정.

    tools.sample_episodes는 모든 에피소드 len<2면 무한 루프(tools.py:339-341)라
    강제 종료 불가 → daemon 스레드로 돌려 timeout 후 TimeoutError. (남은 스레드는
    pytest 종료 시 daemon이라 함께 정리됨.)
    """
    result = {}

    def target():
        try:
            result["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - 스레드 예외 전파용
            result["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(
            f"make_joint_dataset이 {timeout}s 내 미반환 → hang(무한 루프) 검출"
        )
    if "error" in result:
        raise result["error"]
    return result["value"]


def _const_gen(label):
    """매번 같은 라벨 dict 를 yield 하는 무한 generator(풀 식별용)."""
    while True:
        yield {"src": label}


def _sample_ratio(ratio, seed=0, n=10000):
    gen = s2.joint_episode_generator(_const_gen("old"), _const_gen("new"), ratio, seed)
    labels = [next(gen)["src"] for _ in range(n)]
    return labels.count("old") / n


def test_ratio_zero_all_new():
    # ratio=0 → rng.rand() < 0 절대 거짓 → 전부 new (Stage1 미사용)
    assert _sample_ratio(0.0) == 0.0


def test_ratio_one_all_old():
    # ratio=1 → rng.rand() < 1 항상 참 → 전부 old (Stage1)
    assert _sample_ratio(1.0) == 1.0


def test_ratio_03_statistical():
    # ratio=0.3 → old 비율 ≈ 0.3 (n=10000, 시드 고정, 허용오차 ±0.02)
    old_frac = _sample_ratio(0.3, seed=0, n=10000)
    assert abs(old_frac - 0.3) < 0.02


def test_ratio_05_statistical():
    old_frac = _sample_ratio(0.5, seed=0, n=10000)
    assert abs(old_frac - 0.5) < 0.02


def test_reproducible_with_seed():
    # 같은 seed → 동일 시퀀스(재현성)
    g1 = s2.joint_episode_generator(_const_gen("old"), _const_gen("new"), 0.3, seed=42)
    g2 = s2.joint_episode_generator(_const_gen("old"), _const_gen("new"), 0.3, seed=42)
    seq1 = [next(g1)["src"] for _ in range(200)]
    seq2 = [next(g2)["src"] for _ in range(200)]
    assert seq1 == seq2
    # 다른 seed → 시퀀스 다름(거의 확실)
    g3 = s2.joint_episode_generator(_const_gen("old"), _const_gen("new"), 0.3, seed=7)
    seq3 = [next(g3)["src"] for _ in range(200)]
    assert seq1 != seq3


# ---------------------------------------------------------------------------
# make_joint_dataset: 실제 episode dict + tools 경로 호환
# ---------------------------------------------------------------------------
class _Cfg:
    batch_length = 8
    batch_size = 4
    joint_replay_ratio = 0.5
    seed = 0


def _make_episodes(n_eps, length, key_val, feat=3):
    """sample_episodes 가 소비할 수 있는 episodes dict({id: {field: ndarray[T,...]}})."""
    eps = {}
    for i in range(n_eps):
        eps[f"ep{i}"] = {
            "obs": np.full((length, feat), float(key_val), dtype=np.float32),
            "action": np.zeros((length, 2), dtype=np.float32),
            "is_first": np.zeros((length,), dtype=bool),
            # log_ 키는 sample_episodes 가 strip → joint element 키 일관성 확인용
            "log_lap_time_s": np.full((length,), 9.9, dtype=np.float32),
        }
    return eps


def test_make_joint_dataset_compatible_batch():
    new_eps = _make_episodes(3, 20, key_val=1.0)
    old_eps = _make_episodes(3, 20, key_val=0.0)
    cfg = _Cfg()
    dataset = s2.make_joint_dataset(new_eps, old_eps, cfg)
    batch = next(dataset)
    # from_generator 가 batch_size 만큼 stack
    assert batch["obs"].shape == (cfg.batch_size, cfg.batch_length, 3)
    assert batch["action"].shape[0] == cfg.batch_size
    # log_ 키는 sample_episodes 에서 strip → batch 에 없어야 함(키 일관성)
    assert "log_lap_time_s" not in batch
    # 여러 배치 연속 생성 가능(generator 무한)
    batch2 = next(dataset)
    assert batch2["obs"].shape == (cfg.batch_size, cfg.batch_length, 3)


# ---------------------------------------------------------------------------
# hang 가드(027 권장안 b): 빈/len1 train_eps → hang 없이 gen_old fallback
# ---------------------------------------------------------------------------
def test_empty_train_eps_no_hang():
    """train_eps 비어있음(Stage2 첫 train 시점) → 가드 없으면 gen_new=sample_episodes(빈)
    이 hang. 가드로 gen_old fallback해 timeout 내 정상 배치 반환해야."""
    new_eps = {}  # 신규 빈 traindir
    old_eps = _make_episodes(3, 20, key_val=0.0)  # Stage1 풀(항상 len≥2)
    cfg = _Cfg()
    cfg.joint_replay_ratio = 0.3  # 운영값

    def build():
        dataset = s2.make_joint_dataset(new_eps, old_eps, cfg)
        return next(dataset)

    batch = _run_with_timeout(build, timeout=10.0)
    assert batch["obs"].shape == (cfg.batch_size, cfg.batch_length, 3)
    # 전부 gen_old(Stage1, key_val=0.0)에서 나와야 함
    assert np.allclose(batch["obs"], 0.0)


def test_len1_train_eps_no_hang():
    """모든 train_eps가 len==1(fresh actor 즉시 충돌) → sample_episodes total<2 무한 루프.
    가드로 gen_old fallback해 hang 없이 연속 배치 반환."""
    new_eps = _make_episodes(3, 1, key_val=1.0)  # length=1 → total<2 hang 유발
    old_eps = _make_episodes(3, 20, key_val=0.0)
    cfg = _Cfg()
    cfg.joint_replay_ratio = 0.3

    def build():
        dataset = s2.make_joint_dataset(new_eps, old_eps, cfg)
        return [next(dataset) for _ in range(5)]

    batches = _run_with_timeout(build, timeout=10.0)
    for b in batches:
        assert b["obs"].shape == (cfg.batch_size, cfg.batch_length, 3)
        assert np.allclose(b["obs"], 0.0)  # 전부 gen_old fallback


def test_guard_recovers_after_maturity():
    """train_eps 미성숙(빈) 동안 gen_old fallback, len≥2 에피소드가 추가되면(같은 dict ref)
    의도된 ratio로 자동 복귀. dreamer.py가 simulate cache(train_eps) ref를 그대로
    make_joint_dataset에 넘기는 실제 배선을 모사."""
    new_episodes = {}  # 미성숙
    g = s2.joint_episode_generator(
        _const_gen("old"), _const_gen("new"), 0.3, seed=0, new_episodes=new_episodes
    )
    first = [next(g)["src"] for _ in range(100)]
    assert all(x == "old" for x in first)  # 미성숙 → 전부 old fallback

    # train_eps 성숙: len≥2 에피소드 in-place 추가(simulate cache 갱신 모사)
    new_episodes["ep0"] = {"obs": np.zeros((5, 3), dtype=np.float32)}
    after = [next(g)["src"] for _ in range(3000)]
    assert "new" in after  # 이제 new 등장 → ratio 복귀
    new_frac = after.count("new") / len(after)
    assert new_frac > 0.5  # ratio=0.3 → new≈0.7 기대(보수적 하한)


def test_guard_disabled_when_none():
    """new_episodes=None(기존 호출 경로) → 가드 비활성, 순수 ratio 동작(하위호환·회귀 0)."""
    g = s2.joint_episode_generator(
        _const_gen("old"), _const_gen("new"), 0.3, seed=0, new_episodes=None
    )
    labels = [next(g)["src"] for _ in range(10000)]
    assert abs(labels.count("old") / 10000 - 0.3) < 0.02
