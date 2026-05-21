"""main() 학습루프 충실 재현 + NaN 출처 추적 (smoke_findings #4).

diag(순수 _train)는 안정이나 main()은 NaN. 차이는 agent.__call__ 경로
(pretrain burst + _policy로 env 행동 + 정책수집 데이터 버퍼 적재). 본 스크립트는
tools.simulate(agent, ...) 루프(=main과 동일)로 재현하고, 매 청크 후
(1) wm/actor 파라미터 NaN, (2) 버퍼에 저장된 obs/action NaN 을 점검해 출처를 잡는다.
"""
import os
import pathlib
import sys
import tempfile

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "vendor" / "dreamerv3-torch"))
sys.path.insert(0, str(_ROOT / "scripts"))

import dreamer as D  # noqa: E402
import tools  # noqa: E402
from dryrun_bench import build_config, _isfloat  # noqa: E402


def overrides():
    ov = {"batch_size": 16}
    s = os.environ.get("DRYRUN_OVERRIDES", "").strip()
    if s:
        for kv in s.split(","):
            k, v = kv.split("=")
            ov[k.strip()] = int(v) if v.strip().lstrip("-").isdigit() else (
                float(v) if _isfloat(v.strip()) else v.strip())
    return ov


def params_nan(module, tag):
    bad = [n for n, p in module.named_parameters()
           if not torch.isfinite(p).all()]
    return bad


def buffer_nan(eps):
    """저장된 에피소드에서 obs(lidar/state)·action NaN 점검."""
    hits = []
    for eid, ep in eps.items():
        for key in ("lidar", "state", "action"):
            if key in ep:
                arr = np.asarray(ep[key], dtype=np.float32)
                if not np.isfinite(arr).all():
                    hits.append((eid[:8], key, arr.shape,
                                 int(np.isnan(arr).any()), int(np.isinf(arr).any())))
    return hits


def main():
    assert torch.cuda.is_available()
    config = build_config(overrides())
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="repro_"))
    config.logdir = str(tmp)
    config.traindir = tmp / "train_eps"
    config.evaldir = tmp / "eval_eps"
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    ar = config.action_repeat
    config.steps //= ar
    config.time_limit //= ar
    tools.set_seed_everywhere(config.seed)
    print(f"precision={config.precision} batch={config.batch_size} pretrain={config.pretrain}")

    from parallel import Damy
    env = D.make_env(config, "train", 0)
    train_envs = [Damy(env)]
    acts = train_envs[0].action_space
    config.num_actions = acts.shape[0]

    import torch.distributions as torchd
    ra = torchd.independent.Independent(torchd.uniform.Uniform(
        torch.tensor(acts.low).repeat(config.envs, 1),
        torch.tensor(acts.high).repeat(config.envs, 1)), 1)

    def random_agent(o, d, s):
        a = ra.sample()
        return {"action": a, "logprob": ra.log_prob(a)}, None

    logger = tools.Logger(tmp, 0)
    eps = {}
    print("prefill 600 random ...")
    tools.simulate(random_agent, train_envs, eps, config.traindir, logger,
                   limit=config.dataset_size, steps=600)

    dataset = D.make_dataset(eps, config)
    agent = D.Dreamer(train_envs[0].observation_space, train_envs[0].action_space,
                      config, logger, dataset).to(config.device)
    agent.requires_grad_(False)

    # --- forward hook: 첫 non-finite 출력 모듈 핀포인트 ---
    flagged = {}

    def mk_hook(name):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and not torch.isfinite(t).all():
                if "first" not in flagged:
                    flagged["first"] = name
                    fin = t.detach().float()
                    print(f"  [HOOK] first non-finite OUTPUT @ {name}: "
                          f"shape={tuple(t.shape)} nan={int(torch.isnan(fin).any())} "
                          f"inf={int(torch.isinf(fin).any())} absmax={fin.abs().max().item():.3g}")
        return hook

    for n, m in agent._wm.named_modules():
        if len(list(m.children())) == 0:  # leaf modules only
            m.register_forward_hook(mk_hook("wm." + n))

    # main()과 동일: agent로 collect(=내부에서 pretrain+train+policy). 청크 단위로 NaN 추적.
    CHUNK = 50
    state = None
    total = 0
    for it in range(40):
        # 청크 직전 파라미터/버퍼 상태
        try:
            state = tools.simulate(agent, train_envs, eps, config.traindir, logger,
                                   limit=config.dataset_size, steps=CHUNK, state=state)
        except Exception as e:
            print(f"\n[!] EXCEPTION during collect chunk it={it} (total~{total}): "
                  f"{type(e).__name__}: {str(e)[:100]}")
            print("  wm params NaN:", params_nan(agent._wm, "wm")[:6])
            print("  actor params NaN:", params_nan(agent._task_behavior.actor, "actor")[:6])
            print("  buffer NaN hits:", buffer_nan(eps)[:6])
            break
        total += CHUNK
        wm_bad = params_nan(agent._wm, "wm")
        ac_bad = params_nan(agent._task_behavior.actor, "actor")
        val_bad = params_nan(agent._task_behavior.value, "value")
        buf_bad = buffer_nan(eps)
        uc = agent._update_count
        # 최근 정책 action 분포
        print(f"it={it:>2} total={total:>4} updates={uc:>4} | "
              f"wm_nan={len(wm_bad)} actor_nan={len(ac_bad)} value_nan={len(val_bad)} "
              f"buf_nan={len(buf_bad)}")
        if wm_bad or ac_bad or val_bad or buf_bad:
            print("  wm:", wm_bad[:4], "actor:", ac_bad[:4], "value:", val_bad[:4])
            print("  buffer hits:", buf_bad[:6])
            break
    else:
        print(f"[OK] {total} steps collected via agent loop, no NaN.")


if __name__ == "__main__":
    main()
