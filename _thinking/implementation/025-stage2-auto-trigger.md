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

### 2) 재호출 시 Claude 절차 (★ 깨어나면 이대로)
1. Stage1 프로세스 부재 + last_step≥500k 재확인(이중 안전).
2. GPU 여유 확인(nvidia-smi — Stage1 반환으로 충분히 비었는지).
3. `scripts/stage2_watchdog.sh`를 detached 기동:
   ```
   cd /home/dlacksdn/f1tenth_RL_project
   setsid nohup ./scripts/stage2_watchdog.sh \
     > runs/stage2_oschersleben/watchdog.log 2>&1 < /dev/null &
   ```
4. ~1분 후 Stage2 dreamer 생존(pid) + train.log의 `[warm-load] loaded N _wm.* keys` +
   `[joint-replay] ratio=0.3` 로그 확인.
5. 사용자에게 보고(시작 시각/pid/warm-load·joint 로그/초기 step).

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

## 다음 단계
Stage2 시작 후: snapshot(Oschersleben bin 10초폭/110s) + eval로 주행시간 추이 모니터.
충분한 주행시간 도달 시 조기 종료 판단(사용자 보고). forgetting/적응속도 보고 lr·ratio 재조정 여지.
