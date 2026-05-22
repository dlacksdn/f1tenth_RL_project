"""A-2 warm-load 유틸 테스트 (019 §3-3, 020 §3-1/§3-5).

stage2_utils.extract_warm_state: Stage1 ckpt에서 world model weights(_wm.*)만 추출.
- _wm.* 만 로드, actor/critic 파라미터는 초기값 유지.
- optimizer state는 추출 대상 아님(빈 상태 = fresh).
- B-2 중복키: _task_behavior._world_model.* 는 _wm.* 가 아니므로 추출 안 됨(공유 텐서,
  strict=False missing 정상). 실모델 불요 — 소형 nn.Module로 CPU 단위 테스트.
"""
import torch
import torch.nn as nn

import stage2_utils as s2  # vendor/dreamerv3-torch (conftest sys.path)


class TinyWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 3)
        self.dynamics = nn.Linear(3, 3)


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(3, 2)


class TinyBehavior(nn.Module):
    """ActorCritic 모사: actor/value + B-2 처럼 world model 공유 참조."""

    def __init__(self, world_model):
        super().__init__()
        self.actor = nn.Linear(3, 2)
        self.value = nn.Linear(3, 1)
        self._world_model = world_model  # models.py:226 공유 참조(B-2 중복키 원인)


class TinyAgent(nn.Module):
    """Dreamer 모사: _wm + _task_behavior(_world_model 공유)."""

    def __init__(self):
        super().__init__()
        self._wm = TinyWorldModel()
        self._task_behavior = TinyBehavior(self._wm)


def test_extract_warm_state_only_wm_keys():
    agent = TinyAgent()
    wm_state = s2.extract_warm_state(agent.state_dict())
    # 모든 키가 _wm. prefix
    assert wm_state, "extract 결과 비어있으면 안 됨"
    assert all(k.startswith("_wm.") for k in wm_state)
    # encoder/dynamics 둘 다 포함(world model 전체 커버)
    assert any("encoder" in k for k in wm_state)
    assert any("dynamics" in k for k in wm_state)


def test_extract_warm_state_excludes_actor_critic_and_shared():
    agent = TinyAgent()
    wm_state = s2.extract_warm_state(agent.state_dict())
    # actor/critic weights 제외
    assert not any(k.startswith("_task_behavior.actor") for k in wm_state)
    assert not any(k.startswith("_task_behavior.value") for k in wm_state)
    # B-2 중복키: _task_behavior._world_model.* 는 _wm. prefix 아니므로 제외
    assert not any(k.startswith("_task_behavior._world_model") for k in wm_state)


def test_warm_load_preserves_actor_and_is_partial():
    # Stage1(source): 학습된 것처럼 모든 파라미터를 상수로 채움
    source = TinyAgent()
    with torch.no_grad():
        for p in source.parameters():
            p.fill_(7.0)
    ckpt = {"agent_state_dict": source.state_dict()}

    # Stage2(target): fresh 초기화 상태. actor 초기값 스냅샷
    target = TinyAgent()
    actor_before = target._task_behavior.actor.weight.detach().clone()

    wm_state = s2.extract_warm_state(ckpt["agent_state_dict"])
    missing, unexpected = target.load_state_dict(wm_state, strict=False)

    # world model 은 source 값(7.0)으로 덮임 (공유 텐서라 _task_behavior._world_model 도 동시 갱신)
    assert torch.allclose(
        target._wm.encoder.weight, torch.full_like(target._wm.encoder.weight, 7.0)
    )
    assert torch.allclose(
        target._task_behavior._world_model.encoder.weight,
        torch.full_like(target._wm.encoder.weight, 7.0),
    )
    # actor 는 fresh 초기값 유지(warm 대상 아님)
    assert torch.allclose(target._task_behavior.actor.weight, actor_before)

    # unexpected 없음(추출이 _wm.* 만), missing 에 actor/value/_world_model 존재(strict=False 정상)
    assert unexpected == []
    assert any(k.startswith("_task_behavior.actor") for k in missing)
    # B-2: _task_behavior._world_model.* 가 missing 으로 뜨는 것은 정상(공유 텐서)
    assert any(k.startswith("_task_behavior._world_model") for k in missing)


def test_warm_load_excludes_optimizer_state():
    # warm-load 는 agent_state_dict 의 _wm.* 만 본다 — optims_state_dict(별도 키)는 건드리지 않음.
    # ckpt 에 optim 이 따로 있어도 추출 결과엔 어떤 optim 키도 들어가지 않음(fresh optim 보장).
    source = TinyAgent()
    ckpt = {
        "agent_state_dict": source.state_dict(),
        "optims_state_dict": {"model_opt": {"state": {"dummy": 1}}},
    }
    wm_state = s2.extract_warm_state(ckpt["agent_state_dict"])
    assert not any("opt" in k for k in wm_state)
