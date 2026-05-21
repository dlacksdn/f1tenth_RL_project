"""A19 dry-run bench (Phase 2-4, Stage 1 진입 게이트).

planning/005 §11-A, §6-3 / planning/013 §7 / planning/014 §3 기준.

측정:
  A = env_step_avg_ms   (어댑터 1 step = action_repeat=2 sim frame 포함, map_easy3)
  B = train_step_avg_ms (agent._train 1회 = batch_size×batch_length forward+backward)
  C = max VRAM (MB)     = torch.cuda.max_memory_reserved()/1024**2
  D = 500K wall-clock 추정 (min), 시나리오 A(단일 500K)·B(2-stage 1M) 양쪽

★ train 빈도 정정 (013 §7 / 본 세션 코드 정독):
  _should_train = Every(batch_steps / train_ratio) = Every(8*64 / 512) = Every(1)
  → agent-step마다 1회 train. (§11-A 원식의 1/train_ratio 가정은 코드와 불일치)
  → N_train = N_agent (지배항).

★ steps 단위 (dreamer.py:218 config.steps //= action_repeat):
  configs steps=5e5 (env-step/sim 단위) → //2 → 250K agent-step(=adapter step=정책 결정).
  "500K 예산" = 250K agent-step = 250K env.step() 호출.

Pass: C ≤ 6400 MB (8GB×0.8) AND (해당 시나리오 D) ≤ 1440 min (24h).
"""
import argparse
import os
import pathlib
import sys
import tempfile
import time

import numpy as np
import ruamel.yaml as ruamel_yaml
import torch

# --- vendor dreamerv3-torch 임포트 경로 ---
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_VENDOR = _PROJECT_ROOT / "vendor" / "dreamerv3-torch"
sys.path.insert(0, str(_VENDOR))

import dreamer as D  # noqa: E402  (make_env, make_dataset, Dreamer, count_steps)
import tools  # noqa: E402

def _isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


N_COLLECT_WARMUP = 200
N_COLLECT_TIMED = 1000
N_TRAIN_WARMUP = 10
N_TRAIN_TIMED = 100
VRAM_CAP_MB = 6400.0      # 8GB × 0.80
WALLCLOCK_CAP_MIN = 1440.0  # 24h


def build_config(overrides=None):
    """dreamer.py __main__ 경로를 그대로 재현 (defaults + f1tenth, args_type coercion).

    overrides: {key: value} — §6-3 분기 재측정용(batch_size/batch_length/steps/train_ratio 등).
    """
    configs = ruamel_yaml.YAML(typ="safe", pure=True).load(
        (_VENDOR / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    defaults = {}
    for name in ["defaults", "f1tenth"]:
        recursive_update(defaults, configs[name])
    if overrides:
        defaults.update(overrides)

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser.parse_args([])
    return config


def main():
    assert torch.cuda.is_available(), "CUDA 필요 (집컴 GPU 전용)"
    # §6-3 분기 재측정용 override: DRYRUN_OVERRIDES="batch_size=16,batch_length=64"
    overrides = {}
    env_ov = os.environ.get("DRYRUN_OVERRIDES", "").strip()
    if env_ov:
        for kv in env_ov.split(","):
            k, v = kv.split("=")
            k = k.strip()
            v = v.strip()
            overrides[k] = int(v) if v.lstrip("-").isdigit() else float(v) if _isfloat(v) else v
    config = build_config(overrides)
    if overrides:
        print(f"[override] {overrides}")

    tmp = tempfile.mkdtemp(prefix="a19_dryrun_")
    logdir = pathlib.Path(tmp)
    config.logdir = str(logdir)
    config.traindir = logdir / "train_eps"
    config.evaldir = logdir / "eval_eps"
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)

    # main()의 카운터 보정 (//action_repeat) — TimeLimit 등 정합
    ar = config.action_repeat
    config.steps //= ar
    config.eval_every //= ar
    config.log_every //= ar
    config.time_limit //= ar

    tools.set_seed_everywhere(config.seed)

    print("=" * 64)
    print("A19 dry-run bench")
    print(f"  device={config.device} precision={config.precision} compile={config.compile}")
    print(f"  batch_size={config.batch_size} batch_length={config.batch_length} train_ratio={config.train_ratio}")
    batch_steps = config.batch_size * config.batch_length
    every = batch_steps / config.train_ratio
    trains_per_agentstep = 1.0 / every if every else 0.0
    print(f"  batch_steps={batch_steps} -> Every({every}) -> trains/agent-step={trains_per_agentstep}")
    print(f"  action_repeat={ar}  config.steps(//ar, agent-step)={config.steps}")
    print("=" * 64)

    # --- env (map_easy3 train env) ---
    from parallel import Damy  # noqa: E402
    env = D.make_env(config, "train", 0)
    train_envs = [Damy(env)]
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    print(f"Action space: {acts}  num_actions={config.num_actions}")

    # random agent (prefill과 동일)
    import torch.distributions as torchd
    random_actor = torchd.independent.Independent(
        torchd.uniform.Uniform(
            torch.tensor(acts.low).repeat(config.envs, 1),
            torch.tensor(acts.high).repeat(config.envs, 1),
        ),
        1,
    )

    def random_agent(o, d, s):
        action = random_actor.sample()
        logprob = random_actor.log_prob(action)
        return {"action": action, "logprob": logprob}, None

    logger = tools.Logger(logdir, 0)
    train_eps = {}

    # --- 수집: warmup 후 timed (A 측정) ---
    print(f"[A] collect warmup {N_COLLECT_WARMUP} steps ...")
    state = tools.simulate(
        random_agent, train_envs, train_eps, config.traindir, logger,
        limit=config.dataset_size, steps=N_COLLECT_WARMUP,
    )
    print(f"[A] collect timed {N_COLLECT_TIMED} steps ...")
    t0 = time.perf_counter()
    state = tools.simulate(
        random_agent, train_envs, train_eps, config.traindir, logger,
        limit=config.dataset_size, steps=N_COLLECT_TIMED, state=state,
    )
    t1 = time.perf_counter()
    A_ms = (t1 - t0) * 1000.0 / N_COLLECT_TIMED
    n_eps = len(train_eps)
    ep_lens = [len(next(iter(e.values()))) for e in train_eps.values()]
    print(f"[A] env_step_avg_ms = {A_ms:.3f}  (episodes={n_eps}, total_steps={sum(ep_lens)})")

    # --- agent build ---
    print("[B] build Dreamer agent ...")
    train_dataset = D.make_dataset(train_eps, config)
    agent = D.Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    # --- train warmup (CUDA init / cudnn autotune / AMP scaler) ---
    print(f"[B] train warmup {N_TRAIN_WARMUP} steps ...")
    for _ in range(N_TRAIN_WARMUP):
        agent._train(next(train_dataset))
    torch.cuda.synchronize()

    # --- train timed (B 측정) + VRAM (C) ---
    torch.cuda.reset_peak_memory_stats()
    print(f"[B] train timed {N_TRAIN_TIMED} steps ...")
    t0 = time.perf_counter()
    for _ in range(N_TRAIN_TIMED):
        agent._train(next(train_dataset))
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    B_ms = (t1 - t0) * 1000.0 / N_TRAIN_TIMED
    C_mb = torch.cuda.max_memory_reserved() / 1024 ** 2
    C_alloc_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"[B] train_step_avg_ms = {B_ms:.3f}")
    print(f"[C] max_memory_reserved = {C_mb:.1f} MB  (allocated peak {C_alloc_mb:.1f} MB)")

    # --- D 산출 (양 시나리오) ---
    # N_agent = agent-step 수. trains_per_agentstep=1 (f1tenth) → N_train=N_agent.
    def estimate_D(n_agent):
        n_train = n_agent * trains_per_agentstep
        env_min = n_agent * A_ms / 1000.0 / 60.0
        train_min = n_train * B_ms / 1000.0 / 60.0
        return env_min, train_min, env_min + train_min

    N_A = int(config.steps)          # 단일 500K → 250K agent-step
    N_B = 2 * N_A                    # 2-stage → 500K agent-step
    envA, trainA, D_A = estimate_D(N_A)
    envB, trainB, D_B = estimate_D(N_B)

    print("=" * 64)
    print("RESULTS")
    print(f"  A (env_step_avg_ms)   = {A_ms:.3f} ms")
    print(f"  B (train_step_avg_ms) = {B_ms:.3f} ms")
    print(f"  C (VRAM reserved)     = {C_mb:.1f} MB   (cap {VRAM_CAP_MB:.0f})")
    print(f"  trains/agent-step     = {trains_per_agentstep}")
    print(f"  --- 시나리오 A (단일 500K = {N_A} agent-step) ---")
    print(f"  D_A = {D_A:.1f} min  (env {envA:.1f} + train {trainA:.1f})  [{D_A/60:.2f} h]")
    print(f"  --- 시나리오 B (2-stage 1M = {N_B} agent-step) ---")
    print(f"  D_B = {D_B:.1f} min  (env {envB:.1f} + train {trainB:.1f})  [{D_B/60:.2f} h]")
    print("=" * 64)
    vram_pass = C_mb <= VRAM_CAP_MB
    dA_pass = D_A <= WALLCLOCK_CAP_MIN
    dB_pass = D_B <= WALLCLOCK_CAP_MIN
    print(f"  VRAM Pass (C<=6400)        : {vram_pass}")
    print(f"  D_A Pass (<=1440min/24h)   : {dA_pass}")
    print(f"  D_B Pass (<=1440min/24h)   : {dB_pass}")
    print(f"  OVERALL (scenario A gate)  : {vram_pass and dA_pass}")
    print("=" * 64)

    for e in train_envs:
        try:
            e.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
