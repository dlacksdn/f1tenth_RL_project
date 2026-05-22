# 023 — eval_gate sanity 검증 + 평가 철학 결정 (고정 pose deterministic 유지)

작성: 2026-05-22. 선행: 022(A-4 eval_gate 구현). commit a448a6b 이후.
코드 변경 없음 — sanity 검증 결과 + 사용자 평가 설계 결정 기록.

---

## 1. eval_gate.py sanity 검증 (하니스 실동작 OK)

현재 `latest.pt`(step~159k 스냅샷)로 map_easy3 3 에피소드 실행(CPU, 학습 무방해).

- ckpt 로드: `missing=0 unexpected=0` (full ckpt 정상).
- lap_time 신호 경로 작동: `obs['log_lap_time_s']` → `laps=[8.76]`(1바퀴 8.76s 포착).
  사전 실측 map_easy3 lap 8~13s와 일치. 2-lap 완주 전 collision 종료라 미완주.
- 완주 판정: `cause=collision` → completion=False. 완주율 0.000.
- 집계: 완주 ep 0 → `lap_median/best=null`(None 처리). aggregate_episodes 정상.
- 게이트: A11 FAIL(`completion_rate 0.0 >= 0.8` 미달), completion-only 확인(lap 게이트 없음).
- JSON 출력 + exit(FAIL→1) 정상. 학습 생존 확인(CPU 추론, GPU 무경쟁).

→ **하니스 완성·검증.** 현재 미완주는 학습 미성숙(159k/500k ~32%) 탓이지 버그 아님.

## 2. 발견: 결정적 eval → 완주율 이진화

3 에피소드가 **완전 동일**(cause=collision, laps=[8.76], len=616, return=155.8 × 3).
이유: 사양대로 `eval_state_mean=True`(결정적 latent) + `actor.mode()`(결정적 행동) +
고정 pose(env default_pose) → 매 에피소드 결정적 동일 trajectory.

함의: 20 ep를 돌려도 전부 동일 → **완주율은 0.0 또는 1.0만 가능**(중간값 불가).
A11 `≥0.80`은 사실상 "이 고정 기준점에서 2-lap 완주 가능한가"(0/1) 판정으로 환원.
(사양 내재 특성. 019 §5-1 "eval_state_mean=True, 고정 pose" 명시, 020 §2-4가 동일
원리를 snapshot diversity 맥락에서 이미 인지.)

## 3. 사용자 결정 (2026-05-22): 고정 pose deterministic 유지

- pose/policy에 변동을 줘 완주율 분포를 만드는 것 = **generalization(다양한 시작점
  robustness) 목표**. 현재 프로젝트 목표는 그게 아니라 **고정 기준점에서 빠른 주행시간**.
- 따라서 **평가 설계 현행 유지**(고정 pose, eval_state_mean=True, actor.mode()). 코드 무변경.
- 완주율 이진화는 문제 아님 — "고정점에서 완주하는가(0/1)" + 완주 시 **lap_time(주행시간)**이 핵심.

### 평가 신호 운용
- 완주율(completion): 고정 기준점 완주 가능 여부 게이트(A11/A16/A12).
- **lap_time(median/best)**: eval_gate가 게이트 판정과 무관하게 **항상 산출·리포트**
  (JSON/stdout 표). 주행시간 추적은 이 출력으로 충분. A13(osch)만 lap_time을 게이트로 사용.
- A11/A16(map_easy3)은 005·009 확정대로 **completion-only**(임계 변경 없음). 주행시간은
  게이트가 아니라 리포트로 관찰 — 빠른 주행 목표는 lap_time median/best 모니터로 달성.

## 4. 다음 할 것 목록

1. **A-2 Stage2 fine-tune (🔴 미구현, Stage2 진입 전 필수)** — warm-load(_wm.* 공유텐서
   strict=False) + fresh optim + lr scale + joint replay(map_easy3+Oschersleben 혼합).
   019 §3 사양. 학습과 독립적으로 지금 착수 가능하나 작업량 큼.
2. **A-6 prefill 자동배선 + networks_1d 주석 (🟢 자투리)** — 재시작 불요.
3. **Stage1 학습 모니터** — 159k→500k. value_mean/완주율(eval_gate 주기 실측) 추이.
4. **학습 성숙 후 eval_gate 실측(20 ep)** — 고정 pose 결정적이라 완주율 0/1 + lap_time
   median/best 관찰. A11(≥0.80)·A16(≥0.70) 판정 + 주행시간 리포트.
