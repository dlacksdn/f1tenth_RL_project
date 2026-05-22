# 015 — Phase 4: Reward + Episode (arclength progress + R_lap + termination)

> 2026-05-22. 집컴 세션. HEAD 14fc2ec 기준 분기. 시나리오 A/B 공통 invariant(planning/014 §3).
> 선행 SSOT: [planning/005 §4-1·4-3·4-4](../planning/005-f1tenth_dreamerV3_version3.md),
> [planning/009 결정 A/B](../planning/009-lap-detection-and-A11.md),
> [implementation/008 centerline](./008-centerline-reextraction.md), [notes/smoke_findings #4](../notes/smoke_findings.md).
> 산술 검증 SSOT: [notes/reward_arithmetic_verification.md](../notes/reward_arithmetic_verification.md).

---

## 1. 구현 (dreamer_f1tenth/envs/f1tenth_env.py)

이전: `reward = 0.0` skeleton (Phase 1-2). actor/value degenerate(value~1e-7).
본 분기: progress reward + arclength lap 판정 + R_lap + termination 페널티 완성.

### 1-1. arclength windowed closest-point progress (009 결정 A)
- `_load_centerline`이 누적호장 `s` (N,) 추가 반환 (기존 xy, tangent에 더해).
- `_windowed_closest_idx(pos)`: 이전 `_closest_idx` ±window (인덱스 순환) 에서만 closest 탐색.
  → self-intersection 구간 global-argmin 점프 차단. reverse_guard도 이 idx 사용(기존 global argmin 교체).
- window는 **거리(m) 기반** `SEARCH_FWD_M=1.5 / SEARCH_BACK_M=0.5` → reset 시 `ds_mean=L_track/N`으로
  트랙별 인덱스 개수 자동 환산 (map_easy3 fwd≈67/back≈23, Osch fwd≈32/back≈11). **매직넘버·맵 overfit 없음.**
- reset: start pose의 전역 closest_idx를 기준(0)으로 `_total_arclen=0`, `_lap_count_arc=0` 초기화.

### 1-2. progress reward (005 §4-3)
- step: `raw_delta = s[new_idx] - s[prev_idx]`, start/finish wrap 보정(±L_track/2).
- `_total_arclen += raw_delta` (부호 포함 누적).
- `progress_r = ALPHA_PROGRESS(1.0) * clip(raw_delta, 0.0, PROGRESS_CAP=0.5)`. 후진=0.
- step-cap 근거: v_max 20×0.02s=0.4m + 0.1 여유. 정상 주행 미접촉, wrap-artifact만 차단.

### 1-3. arclength lap 판정 + R_lap (009 §2, 005 §4-3)
- `current_lap = total_arclen // L_track`의 **high-water-mark** → 새 lap에서만 R_lap 1회 지급.
  경계 왕복 reward farming 차단 = 005 velocity_dot_tangent>0 방향가드를 arclength로 대체.
- R_lap: TRACK_CONFIGS에 추가 — map_easy3=25.0 / Oschersleben=100.0 (통일 금지).
- `lap_count_arc >= LAP_TARGET(2)` → cause='lap_complete', is_last. **f110 lap_count 미사용**
  (009 §1 map_easy3 double-count 회피; info엔 디버그로만 기록).

### 1-4. termination 페널티 (005 §4-4 + smoke_findings #4)
- 우선순위 유지: diverged > collision > reverse > lap_complete > timeout (#24).
- collision/reverse/**diverged** = PENALTY_TERMINAL(-10). diverged 페널티는 smoke_findings #4 잔여 반영
  (f110 dynamics 발산 = 비정상 실패 → 충돌과 동급 페널티). lap_complete=is_last(성공), 나머지=is_terminal.

### 1-5. A17 reward component 분리 로깅
- info dict: reward_progress/collision/reverse/diverged/lap + arclen_s/total_arclen/lap_count_arc/closest_idx.

---

## 2. 검증

### 2-1. 회귀 + 단위테스트 — pytest 28/28 PASS
- 기존 23 유지(reverse_guard windowed idx 교체에도 무영향).
- 신규 dreamer_f1tenth/tests/test_reward.py 5건:
  component합=reward, progress∈[0,0.5], arclength 2-lap lap_complete, R_lap=lap당1회·값정확,
  collision 페널티 -10, windowed closest_idx 무점프(국소 이동만).

### 2-2. reward 산술 (notes/reward_arithmetic_verification.md)
- GF 실측: progress 보상 합 ≈ 누적 주행거리 (Osch 550.27 vs total_arclen 550.4, rel_err<5%).
- map_easy3 f110 lap=2 double-count를 arclength lap_arc=1로 정확히 회피.

### 2-3. dreamer.py main() smoke — reward 신호 정상 (★)
이전 reward=0 → value~1e-7 degenerate. reward 투입 후:
- value_mean **25.5** (std 7.7, 비영 학습), train_return **70.8** / length **884 step**(정책이 오래 주행 학습).
- model_loss 58→**2.0 수렴**, imag_reward min−10/max0.5(충돌·progress 반영).
- 파라미터 NaN **0건**(latest.pt 366 텐서 전부 finite).
- grad_norm 간헐 inf/nan = **precision=16 AMP GradScaler 정상 거동**(inf grad→update skip→scale backoff),
  파라미터 finite·loss 수렴·value 학습이 확증. 학습 붕괴 아님. → smoke_findings #4 reward 투입 후 재확인 완료.

---

## 3. 미해결 / 후속
- map_easy3 centerline start-end 3.819m 갭(green-ribbon 추출 한계) — wrap 보정+step-cap이 흡수, GF 실측 정상.
- reward_calibrate.py(005 §4-3 헬퍼): L_track이 코드 SSOT라 불필요 → 미구현.
- f110 dynamics 발산 근본원인(ST 적분, base_classes.py:488)은 여전 미규명 — diverged 가드+페널티로 차단.
- 시나리오 A/B 분기 작업(eval_heldout/도메인랜덤화/fresh_optim)은 교수님 확답 후(planning/014 §4).

## 4. 다음 단계
- Phase 5(실학습) 진입 전: 시나리오 확정(교수님 확답) 대기. Phase 4는 양 시나리오 공통이라 완료.
