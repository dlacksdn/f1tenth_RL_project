"""GF prefill collector (decision #23, planning/005 §1-C #23).

prefill=0 (#23): dreamer 자체 random prefill 대신, GapFollower 정책으로 초기 N env-step을
수집해 logdir/train_eps 에 적재한다. 이후 `dreamer.py main`을 같은 logdir + prefill=0 으로
실행하면 `count_steps(traindir)`가 이 데모를 인식 → GF trajectory로 학습을 시작한다.

obs/action 포맷 정합 (envs/f1tenth.py + envs/wrappers.py 체인):
  - obs["lidar"]: [0,1] normalized = raw/LIDAR_MAX → GapFollower는 raw meters를 쓰므로 ×LIDAR_MAX 복원.
  - action: make_env 체인의 NormalizeActions가 [-1,1]→raw 매핑 (wrappers.py:42
    original=(a+1)/2*(high-low)+low). 따라서 GF raw (steer,speed)를 [-1,1]로 역변환해 반환:
      norm_steer = steer / S_MAX           (대칭: low=-S_MAX, high=S_MAX)
      norm_speed = 2*(speed - V_MIN)/(V_MAX - V_MIN) - 1
  - GapFollower.process_lidar → (speed, steer); action_space 순서는 [steer, speed].
  - GF는 stateless(현재 scan만) → 단일 인스턴스 재사용.

사용:
  python scripts/gf_prefill.py --task f1tenth_map_easy3 --logdir <STAGE1_LOGDIR> --steps 10000
"""
import argparse
import pathlib
import sys

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (_ROOT / "vendor" / "dreamerv3-torch", _ROOT / "scripts",
          _ROOT / "pkg" / "src", _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dreamer as D  # noqa: E402
import tools  # noqa: E402
from parallel import Damy  # noqa: E402
from dryrun_bench import build_config  # noqa: E402
from pkg.drivers import GapFollower  # noqa: E402
from dreamer_f1tenth.envs.f1tenth_env import (  # noqa: E402
    LIDAR_MAX, S_MAX, V_MIN, V_MAX,
)


def make_gf_agent(num_envs):
    """simulate-호환 GF agent. obs(numpy, batched) → {'action': tensor(B,2), 'logprob': ...}."""
    gfs = [GapFollower() for _ in range(num_envs)]

    def agent(obs, done, state):
        lidar = np.asarray(obs["lidar"], dtype=np.float32)  # (B, 1080) normalized
        acts = []
        for i in range(lidar.shape[0]):
            raw = lidar[i] * LIDAR_MAX                       # raw meters 복원
            speed, steer = gfs[i].process_lidar(raw)
            ns = float(np.clip(steer / S_MAX, -1.0, 1.0))
            nv = float(np.clip(2.0 * (speed - V_MIN) / (V_MAX - V_MIN) - 1.0, -1.0, 1.0))
            acts.append([ns, nv])
        a = torch.tensor(np.asarray(acts, dtype=np.float32))  # (B, 2), 학습엔 미사용 logprob은 dummy
        return {"action": a, "logprob": torch.zeros(a.shape[0])}, state

    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="f1tenth_map_easy3")
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--steps", type=int, default=10000)  # #23: 첫 10K env-step
    args = parser.parse_args()

    config = build_config({"task": args.task})
    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.logdir = str(logdir)
    config.traindir = logdir / "train_eps"
    config.traindir.mkdir(parents=True, exist_ok=True)
    tools.set_seed_everywhere(config.seed)

    env = D.make_env(config, "train", 0)
    envs = [Damy(env)]
    logger = tools.Logger(logdir, 0)
    cache = {}
    agent = make_gf_agent(len(envs))

    print(f"[gf_prefill] task={args.task} steps={args.steps} → {config.traindir}")
    tools.simulate(agent, envs, cache, config.traindir, logger,
                   limit=config.dataset_size, steps=args.steps)
    n = D.count_steps(config.traindir)
    print(f"[gf_prefill] done. count_steps(traindir)={n} (target ~{args.steps})")
    for e in envs:
        e.close()


if __name__ == "__main__":
    main()
