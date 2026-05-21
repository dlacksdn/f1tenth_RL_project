"""world model NaN 진단 (notes/smoke_findings.md #4).

순수 WM-train 루프로 model_grad_norm·losses 추이를 step별 로깅해 발산 onset을
핀포인트하고, NaN 발생 시 어느 단계(embed/post-logit/decoder/param/grad)가 먼저
오염됐는지 보고한다. 정책 collection은 배제(랜덤 데이터 고정) → WM 학습 동역학만 격리.

usage: python scripts/diag_wm_nan.py            # batch16, precision16
       DRYRUN_OVERRIDES="precision=32" python scripts/diag_wm_nan.py
"""
import os
import pathlib
import sys
import tempfile

import numpy as np
import torch

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_VENDOR = _PROJECT_ROOT / "vendor" / "dreamerv3-torch"
sys.path.insert(0, str(_VENDOR))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import dreamer as D  # noqa: E402
import tools  # noqa: E402
from dryrun_bench import build_config, _isfloat  # noqa: E402

N_COLLECT = int(os.environ.get("DIAG_COLLECT", "2000"))
N_TRAIN = int(os.environ.get("DIAG_TRAIN", "300"))
LOG_EVERY = 5


def parse_overrides():
    ov = {}
    s = os.environ.get("DRYRUN_OVERRIDES", "").strip()
    if s:
        for kv in s.split(","):
            k, v = kv.split("=")
            k, v = k.strip(), v.strip()
            ov[k] = int(v) if v.lstrip("-").isdigit() else (float(v) if _isfloat(v) else v)
    ov.setdefault("batch_size", 16)
    return ov


def tensor_health(name, t):
    if t is None:
        return f"{name}=None"
    t = t.detach().float()
    return (f"{name}: min={t.min().item():.3g} max={t.max().item():.3g} "
            f"absmax={t.abs().max().item():.3g} nan={int(torch.isnan(t).any())} "
            f"inf={int(torch.isinf(t).any())}")


def param_grad_health(wm):
    bad_p, bad_g, gmax, pmax = [], [], 0.0, 0.0
    for n, p in wm.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            bad_p.append(n)
        pmax = max(pmax, p.detach().abs().max().item())
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                bad_g.append(n)
            gmax = max(gmax, p.grad.detach().abs().max().item())
    return bad_p, bad_g, gmax, pmax


def main():
    assert torch.cuda.is_available()
    overrides = parse_overrides()
    config = build_config(overrides)
    tmp = tempfile.mkdtemp(prefix="wm_nan_")
    logdir = pathlib.Path(tmp)
    config.logdir = str(logdir)
    config.traindir = logdir / "train_eps"
    config.evaldir = logdir / "eval_eps"
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    ar = config.action_repeat
    config.steps //= ar
    config.time_limit //= ar
    tools.set_seed_everywhere(config.seed)
    print(f"precision={config.precision} batch_size={config.batch_size} "
          f"batch_length={config.batch_length} model_lr={config.model_lr} "
          f"grad_clip={config.grad_clip}")

    from parallel import Damy
    env = D.make_env(config, "train", 0)
    train_envs = [Damy(env)]
    acts = train_envs[0].action_space
    config.num_actions = acts.shape[0]

    import torch.distributions as torchd
    random_actor = torchd.independent.Independent(
        torchd.uniform.Uniform(
            torch.tensor(acts.low).repeat(config.envs, 1),
            torch.tensor(acts.high).repeat(config.envs, 1)), 1)

    def random_agent(o, d, s):
        a = random_actor.sample()
        return {"action": a, "logprob": random_actor.log_prob(a)}, None

    logger = tools.Logger(logdir, 0)
    train_eps = {}
    tools.simulate(random_agent, train_envs, train_eps, config.traindir, logger,
                   limit=config.dataset_size, steps=N_COLLECT)
    print(f"collected episodes={len(train_eps)} "
          f"steps={sum(len(next(iter(e.values()))) for e in train_eps.values())}")

    dataset = D.make_dataset(train_eps, config)
    agent = D.Dreamer(train_envs[0].observation_space, train_envs[0].action_space,
                      config, logger, dataset).to(config.device)
    agent.requires_grad_(False)
    wm = agent._wm

    print(f"{'step':>4} {'model_loss':>10} {'lidar':>8} {'state':>7} {'kl':>7} "
          f"{'grad_norm':>9} {'pmax':>9} {'gmax':>10}")
    for step in range(1, N_TRAIN + 1):
        data = next(dataset)
        try:
            agent._train(data)
        except Exception as e:
            print(f"\n[!] EXCEPTION at train step {step}: {type(e).__name__}: {str(e)[:120]}")
            # 어느 텐서가 먼저 오염됐는지 직접 forward로 점검
            with torch.no_grad():
                d2 = wm.preprocess(data)
                embed = wm.encoder(d2)
                print("  " + tensor_health("embed", embed))
                post, prior = wm.dynamics.observe(
                    embed, d2["action"], d2["is_first"])
                print("  " + tensor_health("post_logit", post.get("logit")))
                print("  " + tensor_health("post_deter", post.get("deter")))
                feat = wm.dynamics.get_feat(post)
                print("  " + tensor_health("feat", feat))
                for hname, head in wm.heads.items():
                    try:
                        out = head(feat)
                        m = out.mode() if hasattr(out, "mode") else out
                        print("  " + tensor_health(f"head[{hname}].mode", m))
                    except Exception as he:
                        print(f"  head[{hname}] err: {str(he)[:80]}")
            bad_p, bad_g, gmax, pmax = param_grad_health(wm)
            print(f"  NaN/Inf params({len(bad_p)}): {bad_p[:6]}")
            print(f"  NaN/Inf grads({len(bad_g)}): {bad_g[:6]}")
            print(f"  param_absmax={pmax:.3g} grad_absmax={gmax:.3g}")
            break

        m = agent._metrics
        bad_p, bad_g, gmax, pmax = param_grad_health(wm)
        if step % LOG_EVERY == 0 or bad_p or bad_g:
            def last(k):
                v = m.get(k, [float('nan')])
                return v[-1] if v else float('nan')
            print(f"{step:>4} {last('model_loss'):>10.3f} {last('lidar_loss'):>8.3f} "
                  f"{last('state_loss'):>7.3f} {last('kl'):>7.3f} "
                  f"{last('model_grad_norm'):>9.2f} {pmax:>9.3g} {gmax:>10.3g}"
                  + ("  <BAD>" if (bad_p or bad_g) else ""))
        if bad_p:
            print(f"  [!] NaN/Inf in params after step {step}: {bad_p[:6]}")
            break
    else:
        print(f"\n[OK] {N_TRAIN} train steps 완주, NaN 없음.")


if __name__ == "__main__":
    main()
