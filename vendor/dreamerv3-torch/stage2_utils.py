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


def joint_episode_generator(gen_old, gen_new, ratio, seed=0, new_episodes=None):
    """두 episode generator를 ratio로 섞는다.

    매 yield마다 rng.rand() < ratio면 gen_old(Stage1), 아니면 gen_new(현 트랙).
    seed 고정으로 재현 가능(통계적 비율 = ratio).

    hang 가드(027 권장안 b): gen_new 분기로 결정됐어도 new_episodes(=train_eps)에
    len≥2 에피소드가 1개도 없으면(미성숙) 그 yield를 gen_old로 우회한다. gen_new가
    참조하는 tools.sample_episodes는 모든 에피소드 len<2면 무한 루프(tools.py:339-341)
    이므로, Stage2 첫 train 시점(빈/len1 train_eps)에 hang을 차단한다. gen_old(Stage1
    풀)는 항상 len≥2라 안전. train_eps가 성숙(len≥2 1개+)하면 의도된 ratio로 자동 복귀.
    new_episodes=None(기존 호출 경로)이면 가드 비활성 → 하위호환·회귀 0.
    """
    rng = np.random.RandomState(seed)
    while True:
        use_old = rng.rand() < ratio
        if not use_old and new_episodes is not None:
            has_valid = any(
                len(next(iter(ep.values()))) >= 2 for ep in new_episodes.values()
            )
            if not has_valid:
                use_old = True  # train_eps 미성숙 → 안전한 gen_old로 우회
        yield next(gen_old) if use_old else next(gen_new)


def make_joint_dataset(episodes, stage1_episodes, config):
    """현 트랙 episodes + Stage1 episodes를 joint_replay_ratio로 섞은 batch dataset.

    make_dataset(dreamer.py:147)과 동일 인터페이스(from_generator(*, batch_size) 반환).
    sample_episodes가 `log_` 키를 strip(tools.py:344,357)하므로 두 풀 element 키 집합 동일.
    """
    gen_new = tools.sample_episodes(episodes, config.batch_length)
    gen_old = tools.sample_episodes(stage1_episodes, config.batch_length)
    gen = joint_episode_generator(
        gen_old, gen_new, config.joint_replay_ratio, config.seed,
        new_episodes=episodes,
    )
    return tools.from_generator(gen, config.batch_size)
