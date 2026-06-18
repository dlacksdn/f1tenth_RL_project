#!/usr/bin/env python3
"""학습 중인 모델로 차가 굴러가는 걸 실시간 창으로 본다 (Phase 5).

설계 원칙
- ★ 환경/차체/맵/물리 무수정: make_env 체인을 그대로 쓰고 내부 F110Env.render()를
  호출만 한다. RenderDamy는 step thunk 실행 뒤 render()를 끼울 뿐 물리에 개입 안 함.
- ★ 진행 중 학습 무방해: latest.pt를 /tmp로 스냅샷 복사 후 로드(쓰는 중 partial read
  방지) + CPU 추론(학습 GPU와 경쟁 안 함). simulate는 /tmp에만 episode 저장.

사용:
  cd /home/dlacksdn/f1tenth_RL_project
  source .venv/bin/activate
  python scripts/watch_drive.py --logdir runs/stage1_map_easy3 --episodes 3
옵션: --mode human|human_fast (기본 human), --task <override>, --ckpt <path>
"""
import argparse
import collections
import functools
import os
import pathlib
import shutil
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VENDOR = PROJECT_ROOT / "vendor" / "dreamerv3-torch"
sys.path.insert(0, str(VENDOR))

import numpy as np  # noqa: E402
import ruamel.yaml as yaml  # noqa: E402
import torch  # noqa: E402

import tools  # noqa: E402
from dreamer import Dreamer, make_env  # noqa: E402
from parallel import Damy  # noqa: E402
from f110_gym.envs.f110_env import F110Env  # noqa: E402


def build_config(task_override=None):
    """dreamer.py __main__ 의 config 조립을 복제 (defaults + f1tenth)."""
    raw = yaml.YAML(typ="safe").load((VENDOR / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    defaults = {}
    for name in ["defaults", "f1tenth"]:
        recursive_update(defaults, raw[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser.parse_args([])

    # eval/시각화 강제 오버라이드 (학습 무방해)
    config.device = "cpu"            # 학습 GPU와 경쟁 안 함
    config.precision = 32            # CPU에서 fp16 회피
    config.envs = 1
    config.parallel = False
    if task_override:
        config.task = task_override
    return config


def find_f110(env):
    """make_env 체인을 .env / ._env 로 따라 내려가 내부 F110Env 를 찾는다."""
    e, seen = env, set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, F110Env):
            return e
        e = getattr(e, "env", None) or getattr(e, "_env", None)
    raise RuntimeError("체인에서 F110Env 를 못 찾음 (render 대상 없음)")


class WindowClosed(Exception):
    """사용자가 렌더 창을 닫았을 때 (f110 rendering 이 예외를 던짐) 관람을 끝낸다."""


class RenderDamy:
    """Damy 를 감싸 step thunk 실행 직후 내부 F110Env.render() 를 끼운다.

    물리/관측/보상에는 일절 개입하지 않는다 — 출력(렌더)과 에피소드 요약만 추가."""

    def __init__(self, damy, f110, mode, v_max=20.0):
        self._damy = damy
        self._f110 = f110
        self._mode = mode
        self._v_max = float(v_max)
        self._ep_return = 0.0
        self._ep_len = 0
        self._ep_idx = 0

    def step(self, action):
        # 정책 action은 정규화([-1,1]) — NormalizeActions(dreamer.py:55)가 raw로 변환해 들어감.
        # 여기선 변환 전이라 f1tenth raw scale로 환산해 실시간 출력
        # (steer [-0.4189,0.4189]rad, speed [-5,20]m/s — f1tenth_env.py:35-36/183-184).
        a = action["action"] if isinstance(action, dict) else action
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        steer = float(a[0]) * 0.4189
        # raw speed = NormalizeActions 역매핑: [-1,1] -> [V_MIN(-5), v_max]
        speed = (float(a[1]) + 1.0) / 2.0 * (self._v_max + 5.0) - 5.0
        print(
            f"[drive] steer={steer:+.3f} rad  speed={speed:6.2f} m/s   "
            f"(norm: {a[0]:+.2f}, {a[1]:+.2f})",
            flush=True,
        )

        promise = self._damy.step(action)

        def wrapped():
            out = promise()
            obs, reward, done, info = out
            self._ep_return += float(reward)
            self._ep_len += 1
            try:
                self._f110.render(mode=self._mode)
            except Exception as exc:
                # 창을 닫으면 f110 rendering 이 'Rendering window was closed.' 를 던진다.
                raise WindowClosed(str(exc))
            if done:
                self._ep_idx += 1
                print(
                    f"[watch] 에피소드 {self._ep_idx}: return={self._ep_return:.2f} "
                    f"length={self._ep_len} cause={info.get('cause')}",
                    flush=True,
                )
                self._ep_return = 0.0
                self._ep_len = 0
            return out

        return wrapped

    def reset(self):
        return self._damy.reset()

    def __getattr__(self, name):  # id / observation_space / action_space 등 위임
        return getattr(self._damy, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="runs/stage1_map_easy3")
    ap.add_argument("--ckpt", default=None, help="기본: <logdir>/latest.pt")
    ap.add_argument("--best", action="store_true",
                    help="logdir의 현재 run best(policy_best_lap*.pt)를 자동 선택. "
                         "더 빠른 lap이 나와 파일명이 바뀌어도 항상 최신 best를 시연(latest.pt는 best 아님).")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--mode", default="human", choices=["human", "human_fast"])
    ap.add_argument("--task", default=None, help="task override (예: f1tenth_Oschersleben)")
    ap.add_argument("--v_max", type=float, default=None,
                    help="action-space 속도상한(m/s). 캡 정책 시연 시 학습값과 일치시킬 것 "
                         "(예: cap-15 정책이면 --v_max 15). 미지정 시 config 기본(20).")
    args = ap.parse_args()

    logdir = (PROJECT_ROOT / args.logdir).resolve() if not os.path.isabs(args.logdir) else pathlib.Path(args.logdir)
    # ckpt 우선순위: 명시(--ckpt) > best 자동(--best) > latest.pt
    if args.ckpt:
        ckpt_src = pathlib.Path(args.ckpt)
    elif args.best:
        # run best는 더 빠른 lap마다 policy_best_lap{X}s_step{Y}k.pt로 교체되므로(파일명 가변)
        # 가장 최근 갱신(mtime 최신) 1개를 자동 선택 → 항상 현재 best 반영.
        cands = sorted(logdir.glob("policy_best_lap*.pt"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise FileNotFoundError(f"best 스냅샷 없음: {logdir}/policy_best_lap*.pt")
        ckpt_src = cands[-1]
        print(f"[watch] --best 자동 선택: {ckpt_src.name}")
    else:
        ckpt_src = logdir / "latest.pt"
    if not ckpt_src.exists():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_src}")

    # 학습이 latest.pt 를 쓰는 중일 수 있으므로 스냅샷 복사 후 로드
    snap = pathlib.Path("/tmp/watch_drive_snapshot.pt")
    shutil.copy2(ckpt_src, snap)
    print(f"[watch] 스냅샷 복사: {ckpt_src} -> {snap}")

    config = build_config(task_override=args.task)
    config.logdir = str(logdir)
    if args.v_max is not None:
        config.v_max = args.v_max  # 캡 정책 시연: 학습 action space와 일치(NormalizeActions 매핑 보정)

    logger = tools.Logger(pathlib.Path("/tmp/watch_drive_log"), 0)

    env = make_env(config, "eval", 0)
    f110 = find_f110(env)
    renv = RenderDamy(Damy(env), f110, args.mode, config.v_max)

    obs_space = renv.observation_space
    act_space = renv.action_space
    config.num_actions = act_space.shape[0]

    print(f"[watch] device={config.device} task={config.task} episodes={args.episodes} mode={args.mode}")
    agent = Dreamer(obs_space, act_space, config, logger, dataset=None).to(config.device)
    agent.requires_grad_(False)
    ckpt = torch.load(snap, map_location="cpu")
    # strict=False: policy_*.pt(partial=_wm+actor)는 critic/value 키가 없다. 시연(주행)은
    # actor.mode() + world model latent만 쓰고 critic은 imagination 학습에만 쓰이므로,
    # missing key를 무시하고 로드해도 추론 정상. full ckpt면 missing 0이라 동일.
    missing, unexpected = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    n_actor_critic_missing = sum(1 for k in missing if ".value" in k or "_slow_value" in k)
    if missing or unexpected:
        print(f"[watch] partial 로드: missing={len(missing)}키"
              f"(critic/value {n_actor_critic_missing} 등, 주행 무관) unexpected={len(unexpected)}")
    agent.eval()
    print("[watch] 모델 로드 완료. 실시간 창을 띄웁니다 (창이 안 뜨면 메시지를 확인하세요).")

    eval_policy = functools.partial(agent, training=False)
    cache = collections.OrderedDict()
    try:
        tools.simulate(
            eval_policy,
            [renv],
            cache,
            "/tmp/watch_drive_eps",
            logger,
            is_eval=True,
            episodes=args.episodes,
        )
        print("[watch] 완료: 요청한 에피소드 관람 종료.", flush=True)
    except WindowClosed as exc:
        print(f"[watch] 창이 닫혀 관람을 종료합니다. ({exc})", flush=True)
    except KeyboardInterrupt:
        print("[watch] 사용자 중단.", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
