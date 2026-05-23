# 025 — Stage2 자동 트리거 (Stage1 500k 완료 → fine-tune 무인 시작)

> 사용자 결정(2026-05-23): zero-shot 스킵 확정 + Stage1 500k 완료 후 **자동으로** Stage2
> fine-tune 시작. 운영값은 024 기본(warm_lr_scale=0.5, joint_replay_ratio=0.3)으로 확정.
> A-2 구현(12c0a1c) + fail-fast 가드(1198b68) + 정적 검증 PASS 완료 상태에서 가동.

## 사전 교차검증 (사용자 요청, 2026-05-23)
- DreamerV3 lr 3개(model 1e-4 / actor·critic 3e-5) = 논문 Table W.1·NM512·danijar 3출처 일치 확정.
- fixed-HP 준수: f1tenth 블록 알고리즘 HP override = 0개(defaults 상속).
- 차량 물리 = upstream f1tenth_gym byte-identical(git 3e54a7a 도입, 90a43af 줄바꿈만). adapter `params=` 미전달.
- 의도적 조정분(007 허용): 모델 차원 12M 커스텀(dyn_deter1024/units256/dyn_discrete16, decision #3,
  005:95/010:43 문서화) + reward/guard/log_ 레이어(프로젝트 본질). 학습 동역학 HP·물리와 무관.

## 메커니즘 (2단계)
### 1) background 폴링 (토큰 0)
- harness-tracked background bash가 Stage1을 폴링. 도는 동안 Claude는 잠들어 토큰 0.
- 트리거 조건: **Stage1(map_easy3) 프로세스 종료 AND last_step≥500000**.
  - 프로세스 종료 = GPU 반환 신호(dreamer는 500k 후 ~510k까지 더 돌며 GPU 점유 → step만 보면 OOM).
  - crash(프로세스 없음 + step<500k)면 조건 미충족 → train_watchdog가 resume → 폴링 계속 대기.
- 조건 충족 시 background 종료 → harness가 Claude를 1회 재호출.

### 2) 재호출 시 Claude 절차 (★★ 계획 변경 2026-05-23 — Stage2 자동 시작 보류!)
**배경**: Stage1 후반(429k~469k) eval 5연속 완주 실패(completed=0, lap_time 7s = 같은
지점 반복 실패). 사용자 결정: "바로 Stage2 넘어가는 건 아닌 것 같다. 500k 찍으면 방향성 고민."
→ **stage2_watchdog 자동 기동 금지.** 깨어나면 측정+보고만 하고 사용자와 방향 논의.

깨어나면 이대로:
1. Stage1 프로세스 부재 + last_step≥500k 재확인. GPU 반환 확인(nvidia-smi).
2. 최종 그래프 생성: `python scripts/plot_returns.py --logdir runs/stage1_map_easy3
   --title "Stage1 final 0~500k"` → return_curve-N.png(0~500k 전체).
3. **eval_gate로 완주율 정밀 측정**(metrics의 듬성한 0/1보다 신뢰도↑):
   `python scripts/eval_gate.py --ckpt runs/stage1_map_easy3/latest.pt
    --task f1tenth_map_easy3 --gate A11 --episodes 20`
   → 완주율(A11≥0.80?) + median/best lap_time. ★ GPU 비었으니 실행 가능.
4. 가능하면 어느 지점에서 실패하는지(lap_time 7s 지점) 단서 수집.
5. 사용자에게 보고 + 방향성 옵션 제시(예: 추가 학습/snapshot 중 best 선택/reward·시작위치
   재검토/그래도 Stage2). **Stage2는 사용자 명시 승인 후에만.**
- stage2_watchdog.sh는 보존하되 자동 기동 안 함(방향 확정 시 수동 사용).

## scripts/stage2_watchdog.sh (커밋 1198b68 직후 추가)
- Stage1 train_watchdog.sh 패턴 + warm-load/joint 인자. start_train은 항상 풀 커맨드:
  - 첫 실행: runs/stage2_oschersleben/latest.pt 부재 → dreamer _do_warm(world model warm, lr×0.5).
  - crash resume: latest.pt 존재 → resume 우선(warm 무시, scaled-lr 복원). joint는 독립 유지(V4).
- 사전 fail-fast: Stage1 생존 시 중단(OOM 방지) / warm_ckpt 부재 중단 / joint_dir 빈 npz 중단.
- TARGET_STEP=500000(configs steps=5e5 상한). fine-tune은 더 일찍 충분할 수 있음 → snapshot/eval
  모니터링 후 조기 중단 가능(상한일 뿐).

## 실행 커맨드 (stage2_watchdog 내부, 024 검증 완료 경로)
```
python -u dreamer.py --configs f1tenth --task f1tenth_Oschersleben \
  --logdir <abs>/runs/stage2_oschersleben \
  --warm_load_ckpt <abs>/runs/stage1_map_easy3/latest.pt --warm_lr_scale 0.5 \
  --joint_replay_dir <abs>/runs/stage1_map_easy3/train_eps --joint_replay_ratio 0.3 \
  --envs 8 --parallel True --log_every 500
```

## 제약
- Stage1과 Stage2 동시 실행 금지(GPU 8.2GB, OOM). 프로세스 종료 확인이 선결.
- env 물리/reward/판정/fixed-HP 무변경(A-2는 main 배선 + 운영 스크립트만).
- 자동 시작은 검증된 경로 + fail-fast 가드로 보호. 깨어났을 때 GPU/로그 재확인 후 보고.

## 마일스톤 그래프 (사용자 요청 2026-05-23)
- 폴링(/tmp/stage1_poll.sh)이 step **300k·400k 도달 시 plot_returns.py 자동 실행**
  → return_curve-N.png 증분 저장(덮어쓰기 X, [[output-no-overwrite]]). 500k 완료 시 500k 그래프 후 트리거.
- plot_returns.py: --out 미지정 시 return_curve-N.png 다음 빈 번호 자동(2026-05-23 수정).
- 진동(eval 0/1)이 후반(400~500k)에 잦아드는지 추세 확인용. 깨어났을 때 300/400/500k 그래프 일괄 제시.

## 다음 단계
Stage2 시작 후: snapshot(Oschersleben bin 10초폭/110s) + eval로 주행시간 추이 모니터.
충분한 주행시간 도달 시 조기 종료 판단(사용자 보고). forgetting/적응속도 보고 lr·ratio 재조정 여지.

## ★ Stage2 실제 시작 (2026-05-23 13:07)
Stage1 516k에서 수동 종료 → Oschersleben zero-shot 진단(026: 감속 못함=속도 정책 과적합)
→ 사용자 "바로 Oschersleben 학습" 결정 → stage2_watchdog.sh 기동.
- 검증 로그(train.log): joint 0.3(355 stage1 eps 로드, dataset_size 200k 제한), lr×0.5
  (model 5e-5/actor·critic 1.5e-5), warm-load 104 _wm.* keys unexpected=0 _wm-missing=0
  (actor/critic/optim fresh). 첫 eval -2.7 collision(actor fresh 당연).
- dreamer pid 6445, logdir=runs/stage2_oschersleben(원본 stage1 불변), watchdog 생존(crash resume).
- 멀티맵 기대(joint 0.3로 map_easy3도?): 가능성 있으나 보장 X(30%<100% 학습량/정책 상충/
  Stage2 eval은 Oschersleben만). 확인하려면 Stage2 후 map_easy3 별도 eval.
