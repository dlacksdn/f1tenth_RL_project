# 016 — 병렬 env 가속 분석 + envs=8 채택 (Stage 1 학습 운영)

> 2026-05-22. 집컴(Parsec 원격, GPU) 세션. Phase 5 Stage 1 학습 중 wall-clock 가속 검토.
> 선행: A19(envs=1, 단일500K=20.9h, notes/dryrun_results.md), planning/007(fixed-HP), planning/015(시나리오 B).
> 본 문서는 학습 운영 결정(가속). 알고리즘 HP 불변(007 준수) — 데이터 수집 병렬성만 변경.

---

## 1. 병목 측정 (envs=1, Stage 1 학습 중)

- 학습 python 프로세스 **CPU 99% = 정확히 1코어** (12코어 중), loadavg 1.0.
- GPU util **~30%** (16~58% 변동), VRAM 4.9/8GB.
- fps **~7 env-step/s** → 500K env-step / 7 ≈ 19.8h (A19 20.9h와 정합).

→ 1차 추정: "env 수집(CPU 단일 스레드)이 병목, GPU 70% 유휴" → 병렬 env로 가속 기대.

## 2. envs=8 병렬 실측 (검증, /tmp/par8, parallel=True)

| 지표 | envs=1 | envs=8 |
|---|---|---|
| env-step/s (dataset 증가율 30초 측정) | ~7 | **11** (8배 아님, ~1.5배) |
| loadavg | 1.0 | **1.15** (여전히 ~1코어) |
| GPU util | ~30% | ~30% (불변) |

- 안정성: 8 subprocess crash 없이 prefill→학습 진입→model_loss 77→9.6 수렴 확인.
- **결론: 병렬 env가 코어·GPU를 못 채우고 fps도 거의 그대로(~1.5배).**

## 3. 진단 정정 — 병목은 env 수집이 아니라 train 루프

- 1차 추정(env CPU 병목)은 **틀림**. envs=8로 env 수집을 8배 늘려도 wall-clock 1.5배만 단축.
- 실제 병목: **world model train 루프의 직렬 반복**. 12M 소형 모델 × batch_size=16 학습이
  GPU를 30%만 점유하는 **작은 커널 다발 + Python launch 오버헤드**로 wall-clock을 지배.
  env를 병렬화해도 매 스텝 train이 직렬로 끼어 전체 시간을 결정 → 코어 추가 효과 미미.
- GPU 30%는 "유휴"가 아니라 "소형 모델이라 더 못 채움". → fixed-HP 하에선 20.9h가 거의 하한.

## 4. f110 멀티프로세스 picklable 수정 (envs/f1tenth.py)

- dreamerv3 `Parallel`은 env 인스턴스를 cloudpickle해 worker로 전송. f110 `F110Env`가
  `EzPickle` 미초기화로 pickle 실패(`AttributeError: _ezpickle_args`).
- 수정: `F1Tenth` 어댑터에 `__getstate__/__setstate__` 추가 — ctor 인자(task/action_repeat/seed)만
  전송, worker에서 env 재생성(EzPickle 원리). vendor f110 내부 미변경. → 병렬 정상 작동.

## 5. 결정 — envs=8 채택 (Stage 1, ~14h)

- envs=8 → ~1.5배 (20.9h → **~14h**, 7시간 절약). 코어 활용은 제한적이나 wall-clock 이득 실재.
- **fixed-HP 불변(007 준수)**: train_ratio=512, batch_size=16, batch_length=64, precision=16 모두 유지.
  envs는 알고리즘 HP가 아니라 **데이터 수집 병렬성** → 학습 알고리즘 동일.
- A19 estimate(envs=1, 20.9h) 대비 변경점: **envs=8**. 발표 시 명기.
- on-policy 수집 분포 소폭 변화 가능하나 replay buffer 기반이라 영향 미미.

## 6. 안전장치 (멀티프로세스 리스크 관리)

- **체크포인트**: dreamer.py main이 `eval_every`(1e4 step)마다 `latest.pt` 저장 = envs=8에서 ~15분.
  crash 시 같은 logdir 재시작 → 그 지점부터 resume(train_eps replay 보존). 최대 손실 ≤15분.
- **crash 자동 감지 모니터** 상시 가동: 프로세스 사망 시 즉시 알림 → resume.
- 14h 장시간 멀티프로세스 안정성은 미검증이나, 위 장치로 다운사이드 = "15분 손실 후 재개" 수준.

## 7. 폴백
- envs=8가 14h 중 불안정(반복 crash/hang) 시 → **envs=1 복구**(20.9h 안정, A19 검증값).
- Stage 2(Oschersleben)도 동일 envs=8 적용 검토(맵만 다르고 학습 비용 동일, 시나리오 무관).
