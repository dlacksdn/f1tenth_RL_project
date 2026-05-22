"""A-2 Stage 2 fine-tune 유틸 (019 §3, 020 §3 검수 확정).

순수 헬퍼만 모아 dreamer.py가 호출(dead config 0 배선) + 무거운 deps 없이 단위 테스트.

- warm-load: Stage1 ckpt에서 world model weights(_wm.*)만 추출(actor/critic/optim fresh).
- joint replay: Stage1 episodes와 현 트랙 episodes를 ratio로 섞는 generator + dataset.

설계 근거:
- #21(017 §2 확정): world model **weights만** warm, actor/critic + 모든 optimizer는 fresh.
- compile=False(configs.yaml:209)라 state_dict 키에 `_orig_mod.` prefix 없음 → `_wm.` 직접 매칭(020 §3-1).
- B-2 중복키(020 §3-5): models.py:226 `self._world_model = world_model` 때문에 world model이
  `_wm.*` + `_task_behavior._world_model.*` 두 경로로 중복(동일 Parameter 공유). warm-load는
  `_wm.*`만 load(strict=False)해도 공유 텐서라 동시 갱신 → 정상. `_task_behavior._world_model.*`가
  load_state_dict의 missing_keys로 뜨는 것은 정상.
"""
import numpy as np

import tools


def extract_warm_state(agent_state_dict):
    """agent.state_dict()에서 world model weights(_wm.*)만 추출.

    actor/critic weights·optimizer state는 제외(fresh). #21 해석(017 §2).
    `_wm.` 하위가 encoder/dynamics/decoder/reward·cont heads 전체 커버(models.py:38-88, 020 §3-1 #5).
    """
    return {k: v for k, v in agent_state_dict.items() if k.startswith("_wm.")}


def joint_episode_generator(gen_old, gen_new, ratio, seed=0):
    """두 episode generator를 ratio로 섞는다.

    매 yield마다 rng.rand() < ratio면 gen_old(Stage1), 아니면 gen_new(현 트랙).
    seed 고정으로 재현 가능(통계적 비율 = ratio).
    """
    rng = np.random.RandomState(seed)
    while True:
        if rng.rand() < ratio:
            yield next(gen_old)
        else:
            yield next(gen_new)


def make_joint_dataset(episodes, stage1_episodes, config):
    """현 트랙 episodes + Stage1 episodes를 joint_replay_ratio로 섞은 batch dataset.

    make_dataset(dreamer.py:147)과 동일 인터페이스(from_generator(*, batch_size) 반환).
    sample_episodes가 `log_` 키를 strip(tools.py:344,357)하므로 두 풀 element 키 집합 동일.
    """
    gen_new = tools.sample_episodes(episodes, config.batch_length)
    gen_old = tools.sample_episodes(stage1_episodes, config.batch_length)
    gen = joint_episode_generator(
        gen_old, gen_new, config.joint_replay_ratio, config.seed
    )
    return tools.from_generator(gen, config.batch_size)
