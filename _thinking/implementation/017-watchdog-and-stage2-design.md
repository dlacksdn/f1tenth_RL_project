# 017 — crash 자동재개 watchdog + Stage 2 fine-tune(#21) 해석 확정/설계

> 2026-05-22. 집컴(Parsec, GPU) 세션. Phase 5 Stage 1(map_easy3 500K, envs=8) 학습 진행 중.
> 본 분기 산출: ① 무인 안정성 watchdog 작성·가동, ② 005 #21 warm-load 해석 확정, ③ Stage 2 구현 설계.
> 선행: planning/015(시나리오 B), implementation/016(envs=8 가속), 005 #21, 002 F5-4.

---

## 1. crash 자동재개 watchdog (scripts/train_watchdog.sh) — 작성·가동 완료

- **목적**: envs=8 멀티프로세스 14h 무인 학습의 crash 대비. 016 §6 "crash 자동 감지 모니터 상시 가동" 구현.
- **동작**:
  - `CHECK_INTERVAL=120s` 주기로 `pgrep -f "dreamer.py.*--task <TASK>.*--envs 8"` 생존 체크.
  - 사망 감지 시 `CONFIRM=15s` 후 재확인(순간 미검출 오탐 방지).
  - 사망 확정 시 `metrics.jsonl`의 `last_step` 판별:
    - `last_step ≥ TARGET_STEP`(기본 500000=configs steps, env-step) → **정상 완료**로 간주, watchdog 종료(재시작 안 함).
    - 미달 → **crash 판단**, 같은 logdir로 resume(`latest.pt` + `train_eps` replay 보존, 최대 손실 ≤15분).
  - **★ 생존 중인 학습은 절대 죽이거나 재시작하지 않음** — 감시만.
- **인자**: `$1 TASK`(기본 f1tenth_map_easy3), `$2 LOGDIR`(기본 runs/stage1_map_easy3), `$3 TARGET_STEP`(기본 500000).
  → Stage 2(Oschersleben)에도 인자만 바꿔 재사용 가능.
- **검증**(학습 무방해): `is_alive` 패턴이 실제 학습 PID 정확 매칭, `last_step` 파싱 정상, `bash -n` 문법 OK.
- **가동**: `setsid nohup ./scripts/train_watchdog.sh ... > runs/stage1_map_easy3/watchdog.log 2>&1 < /dev/null &`.
  watchdog.log에 가동 메시지만(개입 0회) — 학습 정상 생존 방증.
- **폴백**(016 §7): 반복 crash/hang 시 envs=1 복구(20.9h 안정).

## 2. 005 #21 warm-load 해석 확정 (사용자 결정)

- 005 #21 원문: "`latest.pt` 복사 → world model **weights** warm, actor/critic/model_opt fresh. lr 절반(R3)."
- 002 F5-4: latest.pt가 optimizer state까지 복사 → Adam momentum stale → "fresh optimizer 옵션 필요".
- **확정 해석**: **World model weights만 warm load. actor/critic weights는 재초기화(fresh) + 모든 optimizer(model/actor/value) state fresh. lr 절반.**
  - 근거: 005가 "world model weights"만 콕 집어 warm 명시 → actor/critic은 weights부터 fresh.
    트랙별 reward landscape/주행정책 차이로 인한 negative transfer 회피. 보수적/안전.
- (014 §4서 보류했던 Phase 3 구현 항목 → 본 트랙서 구현.)

## 3. Stage 2 구현 설계 (다음 분기 — vendor dreamer.py 신중 수정)

dreamer.py:306-309(resume 로직) 기반:

- **로드 분기**:
  - `logdir/latest.pt` 존재 → 기존대로 전체 resume(Stage 2 crash 시 watchdog 호환 유지).
  - **부재 시에만**(Stage 2 첫 시작) `--warm_load_ckpt`에서 `agent_state_dict` 중 `_wm.*` 키만
    `strict=False` 로드 → actor/critic은 생성자 초기화 유지(fresh), optimizer는 미로드(fresh).
- **신규 flag**(configs.yaml f1tenth에 key 추가 → argparse 자동 노출): `warm_load_ckpt`, `warm_lr_scale`(0.5), `joint_replay_ratio`(0.3).
- **lr 절반**: 옵티마이저 생성부에서 model/actor/value lr × warm_lr_scale.
- **#9 joint replay**: Stage1 traindir를 별도 로드해 0.3 비율 혼합 샘플링(A16 미달 시 0.5, #31). 가장 복잡 — make_dataset/replay 구조 추가 파악 필요.
- **검증**: 단위테스트 + 회귀(pytest 28 유지, CPU 가능분). torch.compile 시 state_dict 키 prefix(`_orig_mod.`) 주의 — 양쪽 동일 config라 일치 예상이나 확인.

## 4. 학습 상태 스냅샷 (본 분기 시점)

- HEAD 8dcbbff(=a03411f + docs 1커밋), pytest 28/28.
- Stage 1: PID 단일 생존, step ~51K/500K, model_loss 2.60↓, value_mean 12.79↑ (정상 수렴).

## 5. 남은 작업

1. (준비) Stage 2 fine-tune 코드 구현(§3) — Stage 1 학습 중 미리. **미구현**(설계만 완료).
2. Stage 1 완료(~내일) 후: A11(map_easy3 completion-only 2-lap ≥80%, 009 결정 B).
3. 통과 시 Stage 2(Oschersleben) warm-load+fresh+joint replay 0.3, envs=8 → A12/A13 → Stage 3 A16.
