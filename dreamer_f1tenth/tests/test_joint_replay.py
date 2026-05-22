"""A-2 joint replay 유틸 테스트 (019 §3-3, 020 §3-2).

stage2_utils.joint_episode_generator / make_joint_dataset:
- ratio=0/1/0.3 에서 두 풀(Stage1=old / 현 트랙=new) 샘플 비율이 시드 고정 하 통계적으로 맞는지.
- 재현성(같은 seed → 같은 시퀀스).
- make_joint_dataset 가 tools.sample_episodes/from_generator 와 호환되는지(실 episode dict).
"""
import numpy as np
import pytest

import stage2_utils as s2  # vendor/dreamerv3-torch (conftest sys.path)


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
