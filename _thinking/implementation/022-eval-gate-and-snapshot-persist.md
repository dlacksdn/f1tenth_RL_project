# 022 — A-4 평가 게이트 + snapshot persist/명칭 (구현 기록)

작성: 2026-05-22. 분기: commit 1218efc(021 §6 정정) 직후. 모델 Opus 4.7(1M).
선행: 020(검수 지시) > 019(계획 §5) > 021(직전 구현·snapshot 정책). SSOT는 `_thinking/planning/`.

이번 분기 = (A) A-4 평가 게이트 신규 + (B) snapshot 개선 2건. env 물리/판정/reward,
fixed-HP 무변경. 신규 config 키 0(dead config 0). pytest 40 → **61 passed**.

---

## A. `scripts/eval_gate.py` 신규 (독립 실행, env 무수정 경로)

019 §5 / 020 §5(6)·§4-1 / 005 A11·A12·A13·A16 사양 구현. 학습 프로세스·env에 일절
개입하지 않는 별도 평가 스크립트.

### A-1. 로드·rollout 경로
- `build_config(task)`: dreamer.py `__main__`의 config 조립(defaults + f1tenth)을 복제
  (watch_drive.py와 동일). eval 강제 오버라이드 `device=cpu / precision=32 / envs=1 /
  parallel=False / eval_state_mean=True`(dreamer.py:95 결정적 분기). 고정 pose는 env
  `default_pose`(f1tenth_env.py:67/74)로 자동 — 트랙별 고정.
- `load_agent`: `make_env(config,"eval",0)` → `Dreamer(...).requires_grad_(False)` → ckpt
  `agent_state_dict` 로드. **strict=False** — full ckpt 및 inference-only partial
  (`_wm.*`+actor.*, value/_slow_value missing) 둘 다 평가 가능. missing/unexpected 카운트 로깅.
- `run_episode`: tools.simulate 대신 단일 env 직접 루프. simulate(tools.py:166-188)의 policy
  호출 규약을 미러 — `obs_batch={k:stack([obs[k]]) for k in obs if "log_" not in k}`,
  `done_arr=[is_first]`, `action,state=agent(obs_batch,done_arr,state)`, dict action이면
  `{k:np.array(action[k][0]...)}`로 env.step. is_first=True는 첫 step만(agent 순환 state reset).
  env는 TimeLimit 래퍼로 반드시 종료(timeout cause) → 무한루프 없음(max_steps 안전망).

### A-2. 신호·집계
- 완주: `info['cause']=='lap_complete'`(f1tenth_env.py:424, 2-lap LAP_TARGET=2 @:89).
- per-lap lap_time: `obs['log_lap_time_s']>0` step 값(f1tenth_env.py:439, lap 증가 step에만
  Δenv_step×0.02 주입; 020 §1-2 per-lap 확정). 완주 ep당 lap 2개 → 모집단 median/best.
- 순수 함수(실모델/시뮬레이터 불요, test 대상):
  - `is_completed(cause)` → cause=='lap_complete'.
  - `aggregate_episodes(eps)` → completion_rate, 완주 ep들 lap 모집단 median/best, cause 분포.
  - `evaluate_gate(name, agg)` → PASS/FAIL + 체크 항목별 근거.
  - `resolve_gates(task, arg)` → task별 기본/명시 게이트 정규화(task 불일치·미지정 게이트 ValueError).

### A-3. 게이트 임계 (GATE_SPECS)
| gate | task | completion_min | lap_median_max | lap_best_max | 근거 |
|------|------|----------------|----------------|--------------|------|
| A11  | map_easy3 | 0.80 | — | — | **completion-only**(005·009 결정 B, 020 §4-1; GF×1.5 미사용) |
| A16  | map_easy3 | 0.70 | — | — | 재평가 완화 임계 |
| A12  | Oschersleben | 0.80 | — | — | completion-only |
| A13  | Oschersleben | — | 120.0 | 110.0 | lap-time only(median∧best, 005:187) |
- 기본 게이트: map_easy3→[A11], Oschersleben→[A12,A13]. `--gate A11,A16`로 명시 가능.
- 경계: completion `>=`(0.80 정확히 PASS), lap `<=`. 완주 0이면 lap_median/best=None → A13 FAIL.

### A-4. 출력
- stdout 요약 표(완주율/median/best/cause/게이트별 PASS·FAIL).
- JSON `runs/<...>/eval_gate_{task}_{step}.json`(per-episode 포함). step은 파일명
  `step_{N}k` 패턴 우선, 아니면 stem(예 'latest').
- exit code: 전 게이트 PASS→0, 하나라도 FAIL→1(CI 활용).

### A-5. 테스트 — `dreamer_f1tenth/tests/test_eval_gate.py` (17 passed)
- 합성 episode info 시퀀스(완주/충돌/timeout/diverged 혼합). 실모델·시뮬레이터 불요.
- 검증: 완주율 계산, lap 모집단(완주 ep만), median/best, A11 completion-only(lap 게이트 부재 확인),
  A11↔A16 임계 분리(0.75→A11 FAIL/A16 PASS), A13 median∧best·best 초과 FAIL·완주0 FAIL,
  경계 포함(0.80 PASS), resolve_gates 기본/명시/오류.

---

## B. snapshot 개선 2건 (vendor dreamer.py / snapshot_utils.py)

### B-1. persist (021 §6 약점 해소)
- 문제: `snapshot_bins`/`snapshot_best`가 메모리 dict라 watchdog resume 시 `{}`로 리셋 →
  디스크 기존 `policy_lap*`/`policy_best*` 파일을 모른 채 재저장 → step suffix 다른 중복 누적.
- 해결:
  - snapshot_utils에 순수 헬퍼 추출(테스트 가능):
    - `pack_snapshot_state(bins, best)` → `{"bins":...,"best":...}` (직렬화용).
    - `restore_snapshot_state(checkpoint)` → `(bins, best)`. 하위호환: "snapshot_state" 부재 시
      `({},{})`. 항상 **새 dict** 반환(원본 mutation 누수 차단).
  - dreamer.py save 블록 `items_to_save["snapshot_state"]=pack_snapshot_state(...)`(:390).
  - dreamer.py resume 블록(latest.pt 존재 시) `snapshot_bins,snapshot_best=restore_snapshot_state(checkpoint)`(:328).
    init `{}/{}`(:312-313)는 latest.pt 부재(신규 run) 시 그대로 유지.

### B-2. 명칭 명확화
- "global best" → **"run best(트랙별 = logdir/process 단위)"**. 전 트랙 통합 best가 아니라
  이 run(logdir)의 최단 lap policy 1개임을 주석/docstring에 명시(dreamer.py:309-313,
  snapshot_utils.py update_diversity_snapshots docstring·:147 주석). 파일명 `policy_best_*` 유지.
- 트랙 전환(Stage2=별도 logdir/process)은 snapshot이 이미 독립 — 정상, 무수정.

### B-3. 테스트 — test_snapshot.py +4 (총 61 passed)
- `test_pack_restore_roundtrip_inmemory`: bins/best 왕복 동일.
- `test_restore_returns_fresh_dicts`: 복원 dict mutation이 원본 미오염.
- `test_restore_backward_compat_missing_key`: 부재/빈/None → ({},{}).
- `test_pack_restore_through_torch_save`: torch.save/load 직렬화 경로(latest.pt 미러).

---

## C. 완료 게이트 / 검증
- `pytest dreamer_f1tenth/tests/` **61 passed** (기준선 40 + eval_gate 17 + snapshot 4).
- 변경 파일: vendor/dreamerv3-torch/{dreamer.py, snapshot_utils.py}, dreamer_f1tenth/tests/test_snapshot.py
  + 신규 scripts/eval_gate.py, dreamer_f1tenth/tests/test_eval_gate.py.
- `git diff --stat -- dreamer_f1tenth/envs/` 비어있음 → env 물리/판정/reward 무변경.
- fixed-HP(train_ratio/batch/batch_length/precision) 무변경. 신규 config 키 0(dead config 0).
- 학습 보호: 작업 중 detached Stage1(envs=8) 읽기만. vendor 변경은 실행 중 프로세스 미반영
  → 이번 분기 종료 후 **승인 하 1회 재시작**으로 일괄 반영(트랙별 bin 정정 포함).

## D. 다음
- 재시작(latest-resume): watchdog 정지 → 학습 그룹 SIGTERM → setsid 재기동(latest.pt 자동 resume)
  → watchdog 재기동.
- 이후: A-2(warm-load/joint replay, Stage2 진입 전), LeWM 연계(보류, 021 §2 모달리티 불일치).
