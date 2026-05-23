# 028 — Stage2 hang 버그 수정 + Oschersleben fine-tune 재시작 (2026-05-23)

> 027 권장안 b 구현 → 테스트(hang 안 함 확증) → Stage2 재시작·step 진행 확인. 완료.

## 작업 1 — 버그 수정 (027 권장안 b: train_eps 유효성 가드)
**파일**: vendor/dreamerv3-torch/stage2_utils.py (joint_episode_generator + make_joint_dataset)
- `joint_episode_generator`에 `new_episodes=None` 인자 추가. gen_new 분기로 결정돼도
  `new_episodes`(=train_eps)에 len≥2 에피소드가 0개면(`any(len(...) >= 2 ...)` False) 그 yield를
  `gen_old`로 우회. gen_old(Stage1 355eps)는 항상 len≥2라 절대 hang 안 함.
- `make_joint_dataset`은 `episodes`(=train_eps) ref를 `new_episodes=episodes`로 전달.
- ★ 하위호환: `new_episodes=None`이면 가드 비활성 → 순수 ratio. 기존 호출/테스트 회귀 0.
- ★ Stage1 경로(dreamer.py:147 make_dataset) 무변경 → Stage1 회귀 0.
- ref 공유 전제 확인: train_eps dict는 simulate cache로 in-place 갱신(tools.py:144,163) +
  make_joint_dataset에 같은 ref 전달(dreamer.py:314) → generator가 최신 train_eps를 봄.
  성숙(len≥2 1개+) 시 가드 해제되어 의도된 ratio=0.3 자동 복귀.

## 작업 2 — 테스트 (dreamer_f1tenth/tests/test_joint_replay.py)
신규 4개 + timeout 가드 헬퍼(`_run_with_timeout`, daemon thread):
- test_empty_train_eps_no_hang: 빈 train_eps → hang 없이 gen_old fallback 배치(obs=old값).
- test_len1_train_eps_no_hang: 모든 train_eps len==1 → hang 없이 연속 배치(전부 old fallback).
- test_guard_recovers_after_maturity: 미성숙 동안 전부 old → len≥2 in-place 추가 시 new 등장
  (ratio 복귀, new_frac>0.5). simulate cache ref 갱신 모사.
- test_guard_disabled_when_none: new_episodes=None → 순수 ratio(하위호환).
- pytest 결과: **75 passed** (기존 71 + 신규 4).
- ★ negative control: `tools.sample_episodes({len1})` 직접 호출 → `timeout 8` EXIT=124(=hang)로
  가드 부재 시 실제 무한 루프 확증. 가드 적용 테스트는 timeout 없이 통과.

## 작업 3 — Stage2 재시작
- 잔여물 정리: runs/stage2_oschersleben 첫 hang 산물(step0 metrics, train.log, events, eval_eps 20npz,
  빈 train_eps) 내용 확인 후 rm -rf → mkdir. latest.pt 부재 유지(=_do_warm 발동).
  ★ runs/stage1_map_easy3 원본 불변(latest.pt + 536 npz 보존 확인).
- 기동: setsid nohup scripts/stage2_watchdog.sh > watchdog.log. dreamer PID 16772.
  운영값(스크립트 고정): joint_replay_ratio=0.3, warm_lr_scale=0.5, warm=stage1/latest.pt.
- ★ 시작 검증(hang 재발 X 확정):
  - warm-load 104 _wm.* keys (unexpected=0) / joint-replay ratio=0.3 (355 stage1 eps) 로그 정상.
  - step 진행: etime 01:12~02:12 step=0,update_count=1(가드 작동=첫 train 통과 지점) →
    02:42 step=512(uc=129) → 03:12 step=832(npz=2) → 03:42 step=1008(npz=8). 단조 증가.
  - GPU util 11~42% 변동(idle 아님), CPU~87%, fps 6.7. train_eps npz 0→2→8 생성.
  - 027 hang 증상(CPU95%/GPU4%/step0 11분 고정)과 명확히 상이 → 수정 성공.

## 다음 단계
- watchdog가 target 500k까지 감시·crash resume. 마일스톤 모니터링(plot_returns 번호증분).
- Oschersleben zero-shot eval_return 음수(초기, 026 진단대로 fresh actor)→ fine-tune 진행 관찰.
- 관련: 027(근본원인+수정안), 026(warm 근거), 024(A-2 검증), 019 §3(사양).
