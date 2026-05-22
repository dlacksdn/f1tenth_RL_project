#!/usr/bin/env python3
"""학습 return 곡선 플롯 (ppt용). metrics.jsonl의 train_return/eval_return을 시각화.

train_return = 에피소드 done마다 기록된 episode return(reward 합).
eval_return  = eval 라운드 평균 return.

사용:
  cd /home/dlacksdn/f1tenth_RL_project && source .venv/bin/activate
  python scripts/plot_returns.py --logdir runs/stage1_map_easy3
옵션: --out <png경로> --window <rolling mean 윈도우> --title <제목>
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")  # 헤드리스(창 없이 png 저장)
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(metrics_path):
    """metrics.jsonl에서 (step, train_return), (step, eval_return) 시퀀스 추출."""
    tr_step, tr_val, ev_step, ev_val = [], [], [], []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            s = d.get("step")
            if "train_return" in d:
                tr_step.append(s)
                tr_val.append(d["train_return"])
            if "eval_return" in d:
                ev_step.append(s)
                ev_val.append(d["eval_return"])
    return (np.array(tr_step), np.array(tr_val),
            np.array(ev_step), np.array(ev_val))


def rolling_mean(y, window):
    """단순 이동평균(끝단은 가능한 범위만)."""
    if len(y) < 2 or window <= 1:
        return y.copy()
    w = min(window, len(y))
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def main():
    ap = argparse.ArgumentParser(description="학습 return 곡선 플롯(ppt용)")
    ap.add_argument("--logdir", default="runs/stage1_map_easy3")
    ap.add_argument("--out", default=None,
                    help="기본: <logdir>/return_curve-N.png (덮어쓰기 X, 다음 빈 번호 자동)")
    ap.add_argument("--window", type=int, default=15, help="train_return 이동평균 윈도우")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    logdir = pathlib.Path(args.logdir)
    if not logdir.is_absolute():
        logdir = root / logdir
    metrics_path = logdir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.jsonl 없음: {metrics_path}")

    tr_s, tr_v, ev_s, ev_v = load_metrics(metrics_path)
    # --out 미지정 시 덮어쓰지 않고 다음 빈 번호로 저장(return_curve-1.png, -2, ...).
    if args.out:
        out = pathlib.Path(args.out)
    else:
        n = 1
        while (logdir / f"return_curve-{n}.png").exists():
            n += 1
        out = logdir / f"return_curve-{n}.png"
    title = args.title or f"Training Return — {logdir.name}"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # train_return: 원시 산점(연한 점) + 이동평균(진한 선)
    if len(tr_s):
        ax.scatter(tr_s, tr_v, s=10, alpha=0.25, color="#4C72B0",
                   label="train_return (per episode)")
        if len(tr_s) >= 2:
            rm = rolling_mean(tr_v, args.window)
            ax.plot(tr_s, rm, color="#1F3B73", lw=2.0,
                    label=f"train_return (rolling mean, w={args.window})")

    # eval_return: 마커 + 선
    if len(ev_s):
        ax.plot(ev_s, ev_v, color="#C44E52", marker="o", ms=5, lw=1.5,
                label="eval_return (deterministic)")

    ax.set_xlabel("environment step")
    ax.set_ylabel("episode return")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out)
    print(f"[plot] 저장: {out}")
    print(f"[plot] train_return: {len(tr_s)}개 (step {tr_s.min() if len(tr_s) else '-'}~"
          f"{tr_s.max() if len(tr_s) else '-'}), eval_return: {len(ev_s)}개")
    if len(tr_v):
        print(f"[plot] train_return 최근값 {tr_v[-1]:.1f}, 최대 {tr_v.max():.1f}")
    if len(ev_v):
        print(f"[plot] eval_return 최근값 {ev_v[-1]:.1f}, 최대 {ev_v.max():.1f}")


if __name__ == "__main__":
    main()
