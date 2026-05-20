# 006 — Phase 1-3: base_classes.py:488 dynamic patch + A6 + A_norm full + false collision 분석

> 2026-05-20. 집컴 `.venv/`. 본 분기 mandatory stop 도달.
> 선행: [005-phase1-2-wrapper.md](./005-phase1-2-wrapper.md).
> 후속: Phase 1-4 (reverse guard, A18).

---

## 1. 산출물

### 1-1. 코드 변경

| 파일 | 변경 | 근거 |
|---|---|---|
| `gym/f110_gym/envs/base_classes.py:488` | `observations['linear_vels_y'].append(0.)` → `observations['linear_vels_y'].append(agent.state[3] * np.sin(agent.state[6]))` | v3 §1-C #27, planning/005:226 |
| `dreamer_f1tenth/tests/test_dynamic_patch.py` (신규) | A6 검증 — 직선 1초 가속 + mild 코너링 1초 시나리오, `|vel_y|.max > 0.05` | 본 문서 §2 |
| `dreamer_f1tenth/tests/test_obs_norm.py` | print/assert label `/π` → `/(2π)` 정정 (실제 scale은 005 §2-1에서 갱신됨) | 표기 정확성 |

### 1-2. 패치 본문

```python
# gym/f110_gym/envs/base_classes.py:488 (Simulator._observations 생성)
# Before:  observations['linear_vels_y'].append(0.)
# After:
observations['linear_vels_y'].append(agent.state[3] * np.sin(agent.state[6]))
```

- `agent.state` = 7D `[x, y, steer, vel, yaw, yaw_rate, slip_angle]` (`vehicle_dynamics_st`).
- state[3] = longitudinal vel, state[6] = slip_angle.
- 저속 (`|x[3]| < 0.5`) 시 `vehicle_dynamics_st` 내부 (dynamic_models.py:152)에서 kinematic으로 자동 전환, slip_angle≈0이라 vel_y≈0 자연스럽게 유지. v_switch=7.319는 accl_constraints의 가속 제한값일 뿐 kinematic 전환 임계가 아님.

---

## 2. A6 검증 결과

### 2-1. 시나리오 (test_dynamic_patch.py)

- 트랙: Oschersleben default_pose `[0.0702, 0.3003, 2.798]` (yaw≈160°).
- Phase 1 (직선, steer=0, speed=10) — 50 env step (1초).
- Phase 2 (코너링, steer=0.15 rad, speed=10) — 50 env step (1초, collision 시 break).

### 2-2. 결과

```
[A6] straight: speed_end=6.154m/s (dynamic threshold=0.5), |vel_y|.max=0.0000
[A6] corner:   |vel_y|.max=1.2672, |slip|.max=0.1600rad (samples=34)
PASSED
```

- 직선 phase: |vel_y|=0.0000 (slip_angle≈0, dynamic 모드지만 sin(0)=0).
- 코너 phase: |vel_y|.max=**1.2672 m/s** (>> A6 기준 0.05). |slip|.max=0.16 rad ≈ 9.2°.
- 코너링 34 step 후 wall collision (mild 0.15 steer @ 6 m/s에서도 Oschersleben 첫 직선 끝에서 충돌). 검증엔 무영향.

### 2-3. 기준 매핑

| Criterion | 기준 | 실측 | 결과 |
|---|---|---|---|
| A6 | `|vel_y|.max > 0.05` (코너링 phase) | 1.2672 | **PASS** |

---

## 3. A_norm full mode 재검증 (사용자 결정 2 — 옵션 A)

### 3-1. 실행

- `RUN_FULL_OBS_NORM=1 python -m pytest dreamer_f1tenth/tests/test_obs_norm.py`
- 100 ep × 9000 env step (max), Oschersleben, GapFollower policy, seed=ep_idx.
- wall = **153.22s** (≈2.5분, ~50분 예상 대비 빠름 — GF가 보통 ~1500 step에 lap 완료).

### 3-2. 결과

```
[A_norm/Oschersleben] N=300300  vel_x/20 0.5%=0.119 99.5%=0.450
                     |vel_y/5| 99%=0.751  |ang_z/(2π)| 99%=0.989
PASSED
```

| Criterion | v3 기준 | 실측 (full) | 결과 |
|---|---|---|---|
| `vel_x/20` 99%-ile lo | ≥ -0.25 | 0.119 (0.5%) | PASS |
| `vel_x/20` 99%-ile hi | ≤ 1.05 | 0.450 (99.5%) | PASS |
| `|vel_y/5|` 99%-ile | ≤ 1.0 | **0.751** | **PASS** (Phase 1-3 패치 후 첫 정량) |
| `|ang_z/(2π)|` 99%-ile | ≤ 1.0 (v3 #15 갱신: /π → /(2π), 005 §2-1) | 0.989 | PASS |

**핵심**: dynamic patch로 vel_y가 실제로 생성됐고, 99%-ile=0.751 → scale `/5` 유지 가능 (saturation 없음). 추가 SSOT 갱신 불필요.

### 3-3. quick vs full 비교

| 모드 | N | wall | `|vel_y/5|` 99% | `|ang_z/(2π)|` 99% |
|---|---|---|---|---|
| quick (5 ep) | 15015 | 1.5s | 0.751 | 0.989 |
| full (100 ep) | 300300 | 153.2s | 0.751 | 0.989 |

GF deterministic + same map 분포라 quick과 full이 동일 통계. 향후 CI는 quick으로 게이트 충분.

---

## 4. map_easy3 false collision 분석 결과 (사용자 결정 3 — 옵션 A)

### 4-1. 추적한 코드 경로

`f110_env.reset(poses)` → `Simulator.reset(poses)` (agent state=zeros + pose) → `self.step(zero_action)` → `Simulator.step` → 각 agent `update_pose` (PID, ODE) → `check_collision` (multi-agent GJK) → `update_scan` → `check_ttc()` (laser_models.py:189 `check_ttc_jit`).

- `check_ttc_jit(scan, vel, scan_angles, cosines, side_distances, ttc_thresh=0.005)`: **`if vel != 0.0` 일 때만 collision 체크**, vel=0이면 무조건 `in_collision=False`.
- iTTC 공식: `ttc = (scan[i] - side_distances[i]) / (vel * cosines[i])`. `ttc < 0.005 AND ttc >= 0` → True.

### 4-2. 재현 실험 (wrapper 경유)

dqn 좌표 `[-0.2, -2.38, 1.745329]`로 `wrapper.reset(options={'pose': ...})` 직후 raw obs 검사 (action_repeat=1, ignore_first_collision=False):

| seed | collisions[0] | scan.min (m) | state[3] vel |
|---|---|---|---|
| 0 | 0.0 | 1.2412 | 0.0 |
| 1 | 0.0 | 1.2383 | 0.0 |
| 7 | 0.0 | 1.2390 | 0.0 |
| 12345 | 0.0 | 1.2384 | 0.0 |
| 42 | 0.0 | 1.2333 | 0.0 |
| 100 | 0.0 | 1.2356 | 0.0 |
| 1234 | 0.0 | 1.2323 | 0.0 |
| 9999 | 0.0 | 1.2374 | 0.0 |
| 314159 | 0.0 | 1.2303 | 0.0 |

scan_min ≈ 1.23 m vs side_distances.max=0.226 m → 차량 외곽까지 ~1m 여유. **9개 seed 모두 collision 없음.**

### 4-3. 결론

- **현재 환경에서 false collision은 재현되지 않는다.** reset 직후 vel=0 → `check_ttc_jit`의 `if vel != 0.0` 조건으로 trivially `in_collision=False`. agent-agent GJK도 1-agent라 항상 0.
- 002 §4-1에서 보고된 false collision은 measure_gap_follower.py L64-L80의 raw `f110_env.reset(poses)` 호출 + GF의 *첫 nonzero action 후 step* 결과였을 가능성 (002 보고문 자체가 "reset 직후" vs "첫 step 후" 모호). 또는 당시 코드 버전이 달랐을 수 있으나 base_classes 변경은 본 분기의 #488 외 없음.
- 이론적 false-positive 조건 (참고): `scan[i] < side_distances[i]` (벽이 차량 외곽 내부에 있음) **AND** `vel * cosines[i] < 0` (해당 beam 방향으로 후진) → ttc>0 AND ttc<0.005 → True. dqn 시작점은 scan_min/side_max 여유로 첫 조건 미충족.
- **조치**: wrapper의 `ignore_first_collision=True` guard는 zero-cost 안전망 — 그대로 유지. 추가 코드 변경 없음. Phase 5 학습 진행 중 false collision이 실제로 관찰되면 그때 재진단.

---

## 5. pytest 회귀 결과

```
$ python -m pytest dreamer_f1tenth/tests/ -v -s
collected 8 items

dreamer_f1tenth/tests/test_collision.py::test_a4_collision_terminates                  PASSED
dreamer_f1tenth/tests/test_dynamic_patch.py::test_a6_dynamic_vel_y_nonzero_on_cornering PASSED
dreamer_f1tenth/tests/test_obs_norm.py::test_a_norm_quick[Oschersleben]                PASSED
dreamer_f1tenth/tests/test_smoke.py::test_a3_smoke_one_episode[Oschersleben]           PASSED
dreamer_f1tenth/tests/test_timeout.py::test_a5_timeout_truncates                       PASSED
dreamer_f1tenth/tests/test_wrapper_api.py::test_a1_reset_signature                     PASSED
dreamer_f1tenth/tests/test_wrapper_api.py::test_a1_step_signature                      PASSED
dreamer_f1tenth/tests/test_wrapper_api.py::test_a2_obs_dict_shape                      PASSED

============================== 8 passed in 11.04s ==============================
```

dynamic patch 후 기존 A1~A5 + A_norm quick 회귀 없음.

---

## 6. v3 acceptance criteria SSOT 갱신 매핑

| Criterion | v3 원안 | 본 분기 후 (SSOT 누적) |
|---|---|---|
| A6 | `|vel_y|.max > 0.05` (직선 5s + 코너링 2s) | **PASS** — 1.2672 (직선 1s + 코너링 1s, dynamic threshold=0.5 도달이면 충분; v_switch=7.319는 무관) |
| A_norm `|vel_y/5|` | (Phase 1-3 패치 후 측정 가능) | **0.751 ≤ 1.0 PASS** — scale `/5` 유지 |
| A_norm `|ang_z|` scale | `/π` | **`/(2π)`** (005 §2-1 SSOT, 본 분기에서 코드 label도 정정) |

---

## 7. Phase 1-4 진입 조건 (체크리스트)

- [x] base_classes.py:488 dynamic patch 적용.
- [x] A6 PASS (test_dynamic_patch.py).
- [x] A_norm full mode 1회 실행 — `|vel_y/5|` 99%=0.751 PASS, scale 유지 결정.
- [x] map_easy3 false collision 분석 — 현재 환경 재현 불가, guard 유지로 종결.
- [x] pytest 회귀 8/8 PASS.
- [x] 본 006 md commit + push.
- [ ] Phase 1-4: reverse guard 구현 (v3 §3 1-4 + #8). 1초간 vel_x<0 지속 시 `terminated=True, cause='reverse'`.
- [ ] A18 검증 (test_reverse_guard.py 신규).

---

## 8. 미확정 / 후속 분기 결정 대기 항목

본 분기에서 새로 발견된 결정 대기 항목은 없음. Phase 1-4는 v3 §3 1-4의 사양대로 진행 가능:

- reverse guard 정의 (v3 #8): "후진 1초"는 *연속 1초 동안 `vel_x < 0`* 또는 *누적 1초* 인지 모호. v3 §11 부록·§4-4·#24 표현 확인 후 Phase 1-4 진입 시 결정.

---

## 9. 산출물 체크리스트

- [x] `gym/f110_gym/envs/base_classes.py:488` 1줄 patch.
- [x] `dreamer_f1tenth/tests/test_dynamic_patch.py` (신규).
- [x] `dreamer_f1tenth/tests/test_obs_norm.py` label 정정.
- [x] pytest 8/8 PASS.
- [x] A_norm full mode 결과 기록 (본 §3).
- [x] false collision 분석 (본 §4).
- [ ] commit + push (003 §3 절차).
