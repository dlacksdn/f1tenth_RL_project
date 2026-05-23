# 030 — ★중요 함정: metrics step(env) vs ckpt step(policy) 단위 차이 + best ckpt 매핑 (2026-05-24)

> "eval-best(160k) 정책 ckpt가 없다=데이터 손실"이라 오진했다가 정정. 실제로는 action_repeat=2로
> 인한 **단위 차이**였고 데이터 손실 없음. 다음 세션이 또 헷갈리지 않도록 기록.

## ★ 단위 함정 (반드시 기억)
- **metrics.jsonl의 "step" = env step** (action_repeat 반영, 곱해진 값).
- **ckpt 파일명 step_{N}k / counters.train = policy step** (agent._step, action_repeat 안 곱함).
- **관계: env step = policy step × action_repeat (= ×2).**
  - 검증: metrics 마지막 171002 / latest.pt counters.train 84992 = **2.012 ≈ 2**.
- 따라서 "metrics 160k"와 "step_80k.pt"는 **같은 시점**이다(혼동 금지).
  - metrics eval step: 0,10000,…,160000,170000 (env, 10k 간격)
  - = policy step: 0,5000,…,80000,85000
  - step ckpt 파일: step_5k…step_85k (policy, 5k 간격) = env 10k…170k. **모든 eval 시점 ckpt 존재.**
- eval_every=10000(env) → 코드에서 `config.eval_every //= action_repeat`(dreamer.py:227)로 policy
  5000 단위 트리거. 그래서 step ckpt가 policy 5k 간격.

## ★ Oschersleben Stage2 best 정책 매핑 (확정)
| best 기준 | metrics step(env) | policy step | 파일(원본) | KEEP 백업 |
|---|---|---|---|---|
| **eval_return 최대 = 336.7** | 160000 | 80000 | step_80k.pt | KEEP_oscher_step80k_FULL.pt (md5동일) |
| lap_time 최단 = 16.6초/lap | 170000 | 85000 | step_85k.pt(=latest.pt) | KEEP_oscher_lapbest16.6s_step85k_FULL.pt |
| (17.4초 lap, 초기 백업) | 160000 | 80000 | (=step_80k) | KEEP_oscher_lap17.4s_step80k.pt 등 |

- **사용자가 "best"로 지정 = eval_return 최대(336) = policy 80k = step_80k.pt.** 이미 KEEP 백업됨
  (어제 17.4초 백업 때 넣은 FULL이 바로 이것). 시연:
  ```bash
  python scripts/watch_drive.py --ckpt runs/stage2_oschersleben/KEEP/KEEP_oscher_step80k_FULL.pt \
    --task f1tenth_Oschersleben --episodes 3
  ```
- lap_time 기준 best(16.6초)는 policy 85k=latest.pt=step_85k.pt → KEEP_oscher_lapbest16.6s_step85k_FULL.pt.

## 데이터 손실 없음 (오진 정정)
- 초기 의심: latest.pt counters train=84992(85k)인데 metrics 171k → "save가 85k에서 멈춤,
  86k~171k 휘발"이라 오판. **틀림.** 84992(policy)×2≈170k(env)=metrics와 일치. ckpt 정상 저장.
- 디스크 929G free, save 정상. step_5k~85k(policy) 전부 존재 = 전 학습구간 보존.

## resume 시 주의 (latest.pt 단위)
- resume은 latest.pt(policy 85k)에서 이어감. logger는 metrics 마지막(env 171k)에서 표시 시작
  (그래서 train.log가 [165994]부터 찍힘 = env step). 불일치 아님, 단위 차이일 뿐.
- 029의 resume 커맨드 유효. 재학습 시 metrics step은 env, 내부 policy step은 ÷2로 환산해 해석할 것.

## KEEP 최종 (runs/stage2_oschersleben/KEEP/, .gitignore *.pt → 로컬 보존)
- KEEP_oscher_step80k_FULL.pt        ← eval-best(336), 사용자 지정 best
- KEEP_oscher_lapbest16.6s_step85k_FULL.pt ← lap-best(16.6초)
- KEEP_oscher_lap17.4s_step80k.pt / _best_lap17.4s_step80k.pt (partial, 80k)
- KEEP_policy_lap16.6s_step85k.pt / _best_lap16.6s_step85k.pt (partial, 85k)

## 관련
- 029(정지/watch_drive --best/bin), 028(hang수정), VIDEO_HANDOFF.md(동영상 미완 작업).
