#!/usr/bin/env python3
"""충돌(collision)-only 데이터 수집 하니스 (planning/005 §4~§6).

속도캡 봉우리 정책을 stochastic rollout 하여 **충돌로 끝난 episode만** dreamer npz로
저장한다(완주/diverged/reverse/timeout 폐기). 각 transition은 ``tools.simulate``
(tools.py:150-199) 정렬을 100% 미러하여 기존 train_eps npz와 동일 키/형식을 유지하며,
Diffuser용 갭 보완으로 ``pose``(T,3 world x/y/theta)와 ``v_max``(스칼라/tier)를 추가한다.

핵심 설계(전부 1차 소스 검증):
  - stochastic   : config.eval_state_mean=False (latent sample, dreamer.py:96 skip) +
                   eval_policy=partial(agent, training=True) (action sample, dreamer.py:105/110/113).
  - transition   : reset transition{obs.copy()+reward0+discount1, action 없음} →
                   step transition{step후 obs.copy()+a(dict: action+logprob)+reward+discount}.
                   add_to_cache가 action/logprob 첫 등장 시 0-패딩 → action[0]=logprob[0]=0.
  - pose(T,3)    : f1tenth_env._LOG_KEYS의 log_pose_x/y/theta(env에 3키 추가)를 obs에서
                   읽어 transition마다 (3,)로 주입 → npz는 길이 T의 log_pose_* 보존.
                   별도 pose 키도 직접 (3,)로 추가(로더 편의). env.unwrapped._raw_obs는
                   어댑터 F1Tenth(gym.Wrapper 미상속)에서 끊기므로 log_ 채널이 견고한 경로.
  - v_max        : tier 값을 transition마다 동일 주입(P3 역정규화용).
  - 충돌 필터    : info['cause']=='collision' ep만 save_episodes; 그 외 폐기(카운트 로그).
  - 동시 수집    : config.device='cpu'. 독립 프로세스 4개(다른 --ckpt/--v_max/--out) 동시 가능.

사용:
  cd /home/dlacksdn/f1tenth_RL_project && source .venv/bin/activate
  python scripts/collect_crash_data.py \
      --ckpt runs/cap10_oschersleben/step_45k.pt --v_max 10 \
      --episodes 200 --out runs/crash_data/cap10

tier 정책(--ckpt / --v_max):
  cap-5  runs/cap5_oschersleben/step_25k.pt   5
  cap-10 runs/cap10_oschersleben/step_45k.pt  10
  cap-15 runs/cap15_oschersleben/step_105k.pt 15
  cap-20 runs/stage2_oschersleben/policy_best_lap16.6s_step85k.pt 20
"""
import argparse
import functools
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VENDOR = PROJECT_ROOT / "vendor" / "dreamerv3-torch"
SCRIPTS = PROJECT_ROOT / "scripts"
# eval_gate(build_config/load_agent) + vendor(tools/dreamer) 재사용.
for p in (str(SCRIPTS), str(VENDOR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# pose 추출 (1순위: obs의 log_pose_* / 폴백: .env 체인 → F1Tenth._env._raw_obs)
# ---------------------------------------------------------------------------
def _pose_from_obs(obs):
    """obs(log_pose_x/y/theta 보유)에서 world pose (3,) float32. env 수정으로 항상 존재."""
    if "log_pose_x" in obs and "log_pose_y" in obs and "log_pose_theta" in obs:
        return np.array(
            [float(obs["log_pose_x"]), float(obs["log_pose_y"]), float(obs["log_pose_theta"])],
            dtype=np.float32,
        )
    return None


def _pose_from_chain(env):
    """폴백: make_env 체인을 .env로 내려가 F110GymnasiumWrapper._raw_obs에서 pose 추출.

    env.unwrapped._raw_obs 가 끊기는 이유(1차 소스): old-gym 0.18 Wrapper.unwrapped 는
    self.env.unwrapped 재귀인데, 어댑터 F1Tenth(envs/f1tenth.py:46)는 gym.Wrapper 미상속
    plain class라 .env/.unwrapped/__getattr__ 부재 → AttributeError. F1Tenth._env 가
    F110GymnasiumWrapper(=_raw_obs 보유, step/reset마다 갱신).
    """
    node = env
    while not hasattr(node, "_env") and hasattr(node, "env"):
        node = node.env  # UUID→SelectAction→TimeLimit→NormalizeActions(.env 보유)
    inner = getattr(node, "_env", node)  # node == F1Tenth 어댑터; ._env == F110GymnasiumWrapper
    raw = inner._raw_obs
    return np.array(
        [float(raw["poses_x"][0]), float(raw["poses_y"][0]), float(raw["poses_theta"][0])],
        dtype=np.float32,
    )


def _pose(env, obs):
    p = _pose_from_obs(obs)
    if p is not None:
        return p
    return _pose_from_chain(env)


# ---------------------------------------------------------------------------
# collect_episode — tools.simulate(L150-199) 정렬 미러
# ---------------------------------------------------------------------------
def collect_episode(agent, env, v_max):
    """단일 env 1 episode stochastic rollout → 전 transition을 cache에 누적.

    반환: (cache, ep_id, cause, length).
      - cache  : {ep_id: {key: [t0, t1, ...]}} (save_episodes에 그대로 전달 가능).
      - ep_id  : env.id (UUID 래퍼, reset마다 재생성; cache 키 = 파일명 prefix).
      - cause  : info['cause'] (collision/diverged/reverse/lap_complete/timeout/None).
      - length : len(cache[ep_id]['reward']) = T(=1 reset + step수).

    정렬(tools.simulate 미러):
      reset transition (L156-163): obs.copy() + reward=0.0 + discount=1.0 (action 없음).
      step  transition (L189-199): step후 obs.copy() + a(dict update) + reward + discount.
      add_to_cache(L256-268)가 action/logprob 첫 등장 시 reset 위치에 0-패딩.
    """
    import torch
    import tools

    cache = {}

    # --- reset: ep_id는 reset 직후 확정(UUID 래퍼가 reset마다 재생성) ---
    obs = env.reset()
    ep_id = env.id
    agent_state = None
    is_first = True

    # reset transition (simulate L156-163 미러).
    t = {k: tools.convert(v) for k, v in obs.items()}
    t["reward"] = 0.0
    t["discount"] = 1.0
    t["pose"] = _pose(env, obs)          # reset 시점 pose (T축 첫 행)
    t["v_max"] = np.float32(v_max)
    tools.add_to_cache(cache, ep_id, t)

    cause = None
    while True:
        # agent 호출 규약 = simulate L167-168 / run_episode L183-186 미러:
        # log_ 키 제외 batch + done=[is_first](첫 step만 True).
        obs_batch = {k: np.stack([obs[k]]) for k in obs if "log_" not in k}
        done = np.array([is_first])
        with torch.no_grad():
            action, agent_state = agent(obs_batch, done, agent_state)
        is_first = False

        # simulate L169-173 미러: agent는 dict({"action","logprob"}) 반환 → 배치 첫 원소.
        if isinstance(action, dict):
            a = {k: np.array(action[k][0].detach().cpu()) for k in action}
        else:
            a = np.array(action[0].detach().cpu())

        # env.step: SelectAction(key="action")이 a["action"] 추출; 어댑터가
        # gymnasium 5-tuple을 4-tuple(obs,reward,done,info)로 변환 반환.
        obs, reward, done, info = env.step(a)

        # step transition (simulate L191-199 미러).
        transition = {k: tools.convert(v) for k, v in obs.items()}
        if isinstance(a, dict):
            transition.update(a)            # action + logprob 둘 다 (train_eps와 동일)
        else:
            transition["action"] = a
        transition["reward"] = reward
        transition["discount"] = info.get("discount", np.array(1 - float(done)))
        transition["pose"] = _pose(env, obs)      # step후 pose
        transition["v_max"] = np.float32(v_max)
        tools.add_to_cache(cache, ep_id, transition)
        # → 첫 step에서 action/logprob는 reset에 없던 키라 add_to_cache가 0-패딩
        #   (action[0]=logprob[0]=0). pose/v_max는 reset에도 넣어 패딩 없이 정상 길이.

        if done:
            cause = info.get("cause")
            break

    length = len(cache[ep_id]["reward"])   # T = 1(reset) + step수 (save_episodes 기준)
    return cache, ep_id, cause, length


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    import tools  # noqa: F401  (sys.path 확인 겸; collect_episode 내에서도 import)
    from eval_gate import build_config, load_agent

    ap = argparse.ArgumentParser(
        description="충돌-only 데이터 수집 (stochastic rollout, planning/005)"
    )
    ap.add_argument("--ckpt", required=True, help="봉우리 정책 ckpt 경로(상대=PROJECT_ROOT 기준)")
    ap.add_argument("--v_max", type=float, default=20.0,
                    help="tier 속도캡(m/s). action space high + pose 역정규화용. "
                         "cap-5→5/cap-10→10/cap-15→15/cap-20→20. 미지정=20(Dreamer 기본).")
    ap.add_argument("--task", default="f1tenth_Oschersleben",
                    choices=["f1tenth_map_easy3", "f1tenth_Oschersleben"])
    ap.add_argument("--episodes", type=int, default=200,
                    help="rollout 시도 횟수(충돌 ep만 저장되므로 실제 저장수 ≤ 이 값)")
    ap.add_argument("--out", required=True,
                    help="충돌 npz 저장 디렉터리(mkdir -p). tier별 분리 권장.")
    ap.add_argument("--save-complete", action="store_true", default=False,
                    help="저속 tier(cap-5/10) 전용: collision뿐 아니라 lap_complete ep도 저장. "
                         "미지정(기본 False)=기존 충돌-only 동작 100% 불변. "
                         "diverged/reverse/timeout/None은 여전히 폐기.")
    ap.add_argument("--max-env-steps", type=int, default=None,
                    help="지정 시 build_config 후 config.time_limit 오버라이드(env-step 단위). "
                         "배회 ep 조기 truncate용. 미지정(기본 None)=config.time_limit 유지. "
                         "완주 ep가 안 끊기게 cap-5/10은 9000 권장(=180s @action_repeat).")
    args = ap.parse_args()

    ckpt_path = pathlib.Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")

    out_dir = pathlib.Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- config: build_config 반환 후 stochastic/v_max/device 오버라이드 ---
    config = build_config(args.task)        # eval_state_mean=True, device=cpu, precision=32
    config.eval_state_mean = False          # latent stochastic (_policy L96 분기 skip)
    config.v_max = args.v_max               # action space high (load_agent의 make_env 전 확정 필수)
    config.device = "cpu"                    # 학습 GPU 무경쟁, 4 tier 동시
    if args.max_env_steps is not None:      # 배회 ep 조기 truncate(완주 ep는 안 끊기게 9000 권장)
        config.time_limit = args.max_env_steps  # make_env 전 확정 필수(TimeLimit 래퍼가 읽음)

    print(f"[collect] task={args.task} ckpt={ckpt_path} v_max={args.v_max} "
          f"episodes={args.episodes} out={out_dir}", flush=True)
    print(f"[collect] eval_state_mean={config.eval_state_mean} device={config.device} "
          f"(stochastic latent+action)", flush=True)
    print(f"[collect] save_complete={args.save_complete} "
          f"max_env_steps={args.max_env_steps} time_limit={config.time_limit}", flush=True)

    import tools
    agent, env = load_agent(config, ckpt_path)  # 내부 make_env → config.num_actions 세팅

    # ★ stochastic policy 호출 규약.
    # Dreamer.__call__(dreamer.py:60-86)은 training=True 시 self._train(next(self._dataset))
    # 를 돌리는데(L62-69) 수집에는 dataset=None → 'NoneType is not an iterator' 크래시.
    # _policy(dreamer.py:88-123)를 직접 호출하면 train 루프를 우회하면서 stochastic 분기
    # (eval_state_mean=False → latent sample L96 skip; training=True → action sample
    # L105 not-training skip → L108/L110 또는 L113 sample. expl_until=0이라 _should_expl는
    # 항상 True지만 expl_behavior='greedy'=task_behavior라 결과 동일)만 취한다.
    # __call__과 동일한 (obs, reset, state) 시그니처로 감싼다(reset 인자는 _policy 미사용).
    def collect_policy(obs, reset, state=None):
        return agent._policy(obs, state, training=True)

    n_collision = 0   # 저장
    n_complete = 0    # 완주 폐기
    n_other = {}      # diverged/reverse/timeout/None 등 폐기 카운트

    try:
        for i in range(args.episodes):
            cache, ep_id, cause, length = collect_episode(collect_policy, env, args.v_max)
            saved = False
            if cause == "collision":
                tools.save_episodes(out_dir, {ep_id: cache[ep_id]})  # {ep_id}-{length}.npz
                n_collision += 1
                saved = True
            elif cause == "lap_complete":
                n_complete += 1
                # --save-complete 시 완주 ep도 저장(저속 tier=느린완주라 BC위험 없음 + 다양성).
                # 미지정 시 기존대로 폐기(동작 불변).
                if args.save_complete:
                    tools.save_episodes(out_dir, {ep_id: cache[ep_id]})  # {ep_id}-{length}.npz
                    saved = True
            else:
                key = str(cause)
                n_other[key] = n_other.get(key, 0) + 1
            print(f"[collect] ep {i + 1}/{args.episodes}: cause={cause} len={length} "
                  f"saved={saved} (collision={n_collision} complete={n_complete})", flush=True)
            del cache  # ep마다 새 cache → 메모리 누수 방지
    finally:
        try:
            env.close()
        except Exception:
            pass

    n_saved = n_collision + (n_complete if args.save_complete else 0)
    print("\n========== collect_crash_data 결과 ==========", flush=True)
    print(f"시도 ep        : {args.episodes}", flush=True)
    print(f"save_complete  : {args.save_complete}  (max_env_steps={args.max_env_steps})", flush=True)
    print(f"충돌 저장      : {n_collision}  → {out_dir}", flush=True)
    print(f"완주 {'저장' if args.save_complete else '폐기'}      : {n_complete}"
          f"{'  → ' + str(out_dir) if args.save_complete else ''}", flush=True)
    print(f"총 저장        : {n_saved}", flush=True)
    print(f"기타 폐기      : {n_other}", flush=True)
    print(f"수집률(저장)   : {n_saved / args.episodes:.3f}" if args.episodes else "-",
          flush=True)
    print("=============================================\n", flush=True)


if __name__ == "__main__":
    main()
