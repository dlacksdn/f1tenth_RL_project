#!/usr/bin/env python3
"""A-4 평가 게이트 하니스 (019 §5 / 020 §4-1·§5(6) / 005 A11·A12·A13·A16).

독립 실행 스크립트 — env 물리/판정/reward 무수정. watch_drive.py의
make_env + Damy + Dreamer load 패턴을 재사용하되, tools.simulate 대신 단일 env
직접 rollout 루프로 per-episode 완주 여부(info['cause']=='lap_complete', 2-lap)와
per-lap lap_time_s(obs['log_lap_time_s']>0)를 수집한다.

판정(완주율 completion-only 우선, 005·009 결정 B):
  - A11 (map_easy3)        완주율 ≥ 0.80                       (★ median lap 게이트 없음)
  - A16 (map_easy3 재평가) 완주율 ≥ 0.70
  - A12 (Oschersleben)     완주율 ≥ 0.80
  - A13 (Oschersleben)     완주 lap median ≤ 120  ∧  best ≤ 110

per-lap lap_time 정의(020 §1-2): Δenv_step×0.02 = lap당 실경과 sim 시간. env가 lap
증가 step에만 obs['log_lap_time_s']>0 주입. 완주 ep당 lap 2개 → 모집단에서 median/best.

사용:
  cd /home/dlacksdn/f1tenth_RL_project && source .venv/bin/activate
  python scripts/eval_gate.py --ckpt runs/stage1_map_easy3/latest.pt \
      --task f1tenth_map_easy3 --episodes 20
옵션: --gate A11,A16 (기본: task별 기본 게이트) / --logdir (JSON 저장 위치 override)
"""
import argparse
import functools
import json
import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VENDOR = PROJECT_ROOT / "vendor" / "dreamerv3-torch"
sys.path.insert(0, str(VENDOR))

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# 게이트 사양 (순수 데이터·함수 — 실모델/시뮬레이터 불요, test_eval_gate에서 검증)
# ---------------------------------------------------------------------------
# completion_min: 완주율 하한(None=미적용). lap_median_max/lap_best_max: 완주 lap
# 통계 상한(None=미적용). A11/A16/A12 = completion-only(005·009), A13 = lap-time only.
GATE_SPECS = {
    "A11": dict(task="f1tenth_map_easy3",    completion_min=0.80, lap_median_max=None, lap_best_max=None),
    "A16": dict(task="f1tenth_map_easy3",    completion_min=0.70, lap_median_max=None, lap_best_max=None),
    "A12": dict(task="f1tenth_Oschersleben", completion_min=0.80, lap_median_max=None, lap_best_max=None),
    "A13": dict(task="f1tenth_Oschersleben", completion_min=None, lap_median_max=120.0, lap_best_max=110.0),
}
# task별 기본 게이트(미지정 시).
DEFAULT_GATES = {
    "f1tenth_map_easy3": ["A11"],
    "f1tenth_Oschersleben": ["A12", "A13"],
}

LAP_COMPLETE_CAUSE = "lap_complete"  # f1tenth_env.py:424 (2-lap, LAP_TARGET=2)


def is_completed(cause):
    """완주 판정: info['cause'] == 'lap_complete' (2-lap 완주, f1tenth_env.py:422-424)."""
    return cause == LAP_COMPLETE_CAUSE


def aggregate_episodes(episodes):
    """episode 결과 리스트를 완주율/lap 통계로 집계(순수 함수).

    episodes: [{"cause": str|None, "lap_times": [float, ...]}, ...].
      lap_times = 그 episode에서 관측된 per-lap lap_time_s(>0)들.
    반환: completion_rate, 완주 ep들의 per-lap 모집단 median/best, cause 분포 등.
    """
    n = len(episodes)
    completed = [ep for ep in episodes if is_completed(ep.get("cause"))]
    # per-lap lap_time 모집단: 완주 ep들의 lap_times만 (020 §1-2).
    laps = [float(t) for ep in completed for t in ep.get("lap_times", []) if t > 0.0]
    cause_counts = {}
    for ep in episodes:
        c = ep.get("cause")
        cause_counts[str(c)] = cause_counts.get(str(c), 0) + 1
    return {
        "n_episodes": n,
        "n_completed": len(completed),
        "completion_rate": (len(completed) / n) if n else 0.0,
        "n_laps": len(laps),
        "lap_median": float(np.median(laps)) if laps else None,
        "lap_best": float(min(laps)) if laps else None,
        "cause_counts": cause_counts,
    }


def evaluate_gate(gate_name, agg):
    """게이트 1건을 집계 결과에 적용 → PASS/FAIL + 체크 항목별 근거(순수 함수)."""
    spec = GATE_SPECS[gate_name]
    checks = []
    if spec["completion_min"] is not None:
        val = agg["completion_rate"]
        thr = spec["completion_min"]
        checks.append({"metric": "completion_rate", "value": val,
                       "op": ">=", "threshold": thr, "ok": val >= thr})
    if spec["lap_median_max"] is not None:
        val = agg["lap_median"]
        thr = spec["lap_median_max"]
        ok = (val is not None) and (val <= thr)
        checks.append({"metric": "lap_median", "value": val,
                       "op": "<=", "threshold": thr, "ok": ok})
    if spec["lap_best_max"] is not None:
        val = agg["lap_best"]
        thr = spec["lap_best_max"]
        ok = (val is not None) and (val <= thr)
        checks.append({"metric": "lap_best", "value": val,
                       "op": "<=", "threshold": thr, "ok": ok})
    passed = bool(checks) and all(c["ok"] for c in checks)
    return {"gate": gate_name, "task": spec["task"], "passed": passed, "checks": checks}


def resolve_gates(task, gate_arg):
    """--gate 인자(쉼표 구분) 또는 task별 기본 게이트를 리스트로 정규화."""
    if gate_arg:
        names = [g.strip() for g in gate_arg.split(",") if g.strip()]
    else:
        names = DEFAULT_GATES.get(task, [])
    for g in names:
        if g not in GATE_SPECS:
            raise ValueError(f"Unknown gate {g!r}. Choices: {list(GATE_SPECS)}")
        if GATE_SPECS[g]["task"] != task:
            raise ValueError(
                f"Gate {g} is for task {GATE_SPECS[g]['task']!r}, not {task!r}."
            )
    return names


# ---------------------------------------------------------------------------
# rollout (실모델/시뮬레이터 경로 — 위 순수 함수와 분리)
# ---------------------------------------------------------------------------
def build_config(task):
    """dreamer.py __main__ config 조립 복제(defaults + f1tenth) + eval 강제 오버라이드."""
    import ruamel.yaml as yaml
    import tools

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

    # eval 강제 오버라이드 (학습 GPU 무경쟁, 결정적 평가)
    config.device = "cpu"           # 학습 GPU와 경쟁 안 함
    config.precision = 32           # CPU에서 fp16 회피
    config.envs = 1
    config.parallel = False
    config.task = task
    config.eval_state_mean = True   # 결정적 eval (dreamer.py:95)
    return config


def run_episode(agent, env, max_steps=100000):
    """단일 env 1 episode rollout. simulate(is_eval)의 policy 호출 규약을 미러.

    반환: {"cause": str|None, "lap_times": [float...], "length": int, "return": float}.
    env는 TimeLimit 래퍼로 반드시 종료(timeout cause)되므로 max_steps는 안전망.
    """
    import torch

    obs = env.reset()
    agent_state = None
    is_first = True
    lap_times = []
    ep_return = 0.0
    length = 0
    cause = None
    while length < max_steps:
        # log_ 키는 encoder 입력에서 제외(tools.simulate와 동일), is_first 등은 포함.
        obs_batch = {k: np.stack([obs[k]]) for k in obs if "log_" not in k}
        done_arr = np.array([is_first])
        with torch.no_grad():
            action, agent_state = agent(obs_batch, done_arr, agent_state)
        is_first = False
        if isinstance(action, dict):
            a = {k: np.array(action[k][0].detach().cpu()) for k in action}
        else:
            a = np.array(action[0].detach().cpu())
        obs, reward, done, info = env.step(a)
        ep_return += float(reward)
        length += 1
        lt = float(obs.get("log_lap_time_s", 0.0))
        if lt > 0.0:
            lap_times.append(lt)
        if done:
            cause = info.get("cause")
            break
    return {"cause": cause, "lap_times": lap_times, "length": length, "return": ep_return}


def load_agent(config, ckpt_path):
    """Dreamer 생성 + ckpt agent_state_dict 로드. full/partial(inference-only) 모두 허용.

    snapshot_utils.save_inference_only가 만든 partial(_wm.*+actor.*)도 평가 가능하도록
    strict=False. value/_slow_value 등 평가 불필요 키 누락은 무해(추론은 wm+actor만 사용).
    """
    import torch
    import tools
    from dreamer import Dreamer, make_env

    env = make_env(config, "eval", 0)
    obs_space = env.observation_space
    act_space = env.action_space
    config.num_actions = act_space.shape[0]

    logger = tools.Logger(pathlib.Path("/tmp/eval_gate_log"), 0)
    agent = Dreamer(obs_space, act_space, config, logger, dataset=None).to(config.device)
    agent.requires_grad_(False)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["agent_state_dict"] if "agent_state_dict" in ckpt else ckpt
    result = agent.load_state_dict(state, strict=False)
    missing, unexpected = len(result.missing_keys), len(result.unexpected_keys)
    if unexpected:
        print(f"[eval_gate] WARN unexpected keys={unexpected} (예상 외)", flush=True)
    print(f"[eval_gate] state_dict load: missing={missing} unexpected={unexpected} "
          f"(partial snapshot이면 value/_slow_value 등 missing 정상)", flush=True)
    agent.eval()
    return agent, env


def print_table(agg, gate_results):
    """stdout 요약 표."""
    print("\n========== eval_gate 결과 ==========", flush=True)
    print(f"episodes      : {agg['n_episodes']}", flush=True)
    print(f"완주(lap_complete): {agg['n_completed']}", flush=True)
    print(f"완주율        : {agg['completion_rate']:.3f}", flush=True)
    lm = agg["lap_median"]
    lb = agg["lap_best"]
    print(f"완주 lap 수   : {agg['n_laps']}", flush=True)
    print(f"lap median(s) : {lm:.3f}" if lm is not None else "lap median(s) : -", flush=True)
    print(f"lap best(s)   : {lb:.3f}" if lb is not None else "lap best(s)   : -", flush=True)
    print(f"cause 분포    : {agg['cause_counts']}", flush=True)
    print("------------------------------------", flush=True)
    for gr in gate_results:
        verdict = "PASS" if gr["passed"] else "FAIL"
        print(f"[{gr['gate']}] {verdict}", flush=True)
        for c in gr["checks"]:
            v = c["value"]
            vs = f"{v:.3f}" if isinstance(v, float) else str(v)
            mark = "OK" if c["ok"] else "XX"
            print(f"   {mark} {c['metric']} {vs} {c['op']} {c['threshold']}", flush=True)
    print("====================================\n", flush=True)


def main():
    ap = argparse.ArgumentParser(description="A-4 평가 게이트 (A11/A12/A13/A16)")
    ap.add_argument("--ckpt", required=True, help="체크포인트 경로(full 또는 inference-only)")
    ap.add_argument("--task", required=True,
                    choices=["f1tenth_map_easy3", "f1tenth_Oschersleben"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--gate", default=None,
                    help="쉼표 구분 게이트(기본: task별). 예: A11,A16 / A12,A13")
    ap.add_argument("--logdir", default=None,
                    help="JSON 저장 디렉터리(기본: ckpt 부모)")
    args = ap.parse_args()

    gates = resolve_gates(args.task, args.gate)
    ckpt_path = pathlib.Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (PROJECT_ROOT / ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")

    config = build_config(args.task)
    print(f"[eval_gate] task={args.task} episodes={args.episodes} gates={gates} "
          f"ckpt={ckpt_path}", flush=True)
    agent, env = load_agent(config, ckpt_path)
    eval_policy = functools.partial(agent, training=False)

    episodes = []
    try:
        for i in range(args.episodes):
            res = run_episode(eval_policy, env)
            episodes.append(res)
            print(f"[eval_gate] ep {i + 1}/{args.episodes}: cause={res['cause']} "
                  f"laps={[round(t, 2) for t in res['lap_times']]} "
                  f"len={res['length']} return={res['return']:.1f}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    agg = aggregate_episodes(episodes)
    gate_results = [evaluate_gate(g, agg) for g in gates]
    print_table(agg, gate_results)

    # JSON 출력: runs/<...>/eval_gate_{task}_{step}.json
    step = _infer_step(ckpt_path)
    out_dir = pathlib.Path(args.logdir) if args.logdir else ckpt_path.parent
    out_path = out_dir / f"eval_gate_{args.task}_{step}.json"
    payload = {
        "task": args.task,
        "ckpt": str(ckpt_path),
        "step": step,
        "episodes": args.episodes,
        "aggregate": agg,
        "gates": gate_results,
        "all_passed": all(gr["passed"] for gr in gate_results),
        "per_episode": episodes,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[eval_gate] JSON 저장: {out_path}", flush=True)

    # 모든 게이트 PASS → exit 0, 하나라도 FAIL → exit 1 (CI 활용).
    sys.exit(0 if payload["all_passed"] else 1)


def _infer_step(ckpt_path):
    """ckpt 파일명/형제 메타에서 step 추정(실패 시 'unknown'). 파일명 step_{N}k 패턴 우선."""
    import re
    m = re.search(r"step_?(\d+)k", ckpt_path.name)
    if m:
        return f"{m.group(1)}k"
    return ckpt_path.stem  # 예: 'latest'


if __name__ == "__main__":
    main()
