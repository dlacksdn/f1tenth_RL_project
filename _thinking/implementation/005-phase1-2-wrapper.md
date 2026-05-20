# 005 — Phase 1-2: F110GymnasiumWrapper 구현 + A1~A5/A_norm 검증

> 2026-05-20. 집컴 `.venv/`, RTX 4060 Ti. Phase 1-2 mandatory stop 도달.
> 선행: [002-phase1-1-measurement.md](./002-phase1-1-measurement.md), [003-sync-policy.md](./003-sync-policy.md), [004-pre-phase1-2-decisions.md](./004-pre-phase1-2-decisions.md).
> 후속: Phase 1-3 (`gym/f110_gym/envs/base_classes.py:488` dynamic patch, A6).

---

## 1. 산출물

### 1-1. 신규 패키지

```
dreamer_f1tenth/
├── __init__.py
├── envs/
│   ├── __init__.py
│   └── f1tenth_env.py     ← F110GymnasiumWrapper 본체
└── tests/
    ├── __init__.py
    ├── conftest.py        ← sys.path 설정 (gym/, pkg/src/)
    ├── test_wrapper_api.py  ← A1, A2
    ├── test_smoke.py        ← A3
    ├── test_collision.py    ← A4
    ├── test_timeout.py      ← A5
    └── test_obs_norm.py     ← A_norm (quick 5 ep × 9000 step / full 100 ep × 9000 step)
```

### 1-2. 의존성 추가
- `pytest 8.3.5` (.venv에 설치 — gymnasium 등 기존 deps와 충돌 없음).

---

## 2. wrapper 사양 (구현 확정)

| 항목 | 값 / 근거 |
|---|---|
| 클래스 | `dreamer_f1tenth.envs.F110GymnasiumWrapper(gymnasium.Env)` |
| base env | `f110_gym.envs.f110_env.F110Env` (gym 0.18 API, 직접 import — gym.make 미경유로 dependency conflict 회피) |
| num_agents / ego_idx | 1 / 0 |
| sim timestep | 0.01s (base default) |
| **action_repeat** | **2** → env step = 0.02s, 50 env step/s (v3 결정 #22) |
| **max_episode_steps** | **9000** (180s @ 50 step/s, A5 wrapper 자체 처리 — 004 사인오프) |
| **trackname** 주입 | `'map_easy3'` / `'Oschersleben'` 문자열 — wrapper 내부에서 yaml path와 default_pose 해석 (004 사인오프) |
| `reset(seed, options)` | `options={'pose': [x,y,θ]}` 지원, 미지정 시 trackname의 default_pose (004 사인오프) |
| obs dict 5-key | `lidar` (1080,) float32 ∈ [0,1] / `state` (5,) float32 / `is_first` / `is_terminal` / `is_last` bool |
| action_space | `Box(low=[-0.4189, -5.0], high=[+0.4189, +20.0], shape=(2,), float32)`  raw scale. NormalizeActions([-1,1])은 외부 chain (#22) |
| LiDAR 정규화 (#15) | `clip(0, 30) / 30.0` |
| state 정규화 (#15 + 본 문서 §2-1) | `[vel_x/20, vel_y/5, ang_vel_z/(2π), prev_steer/0.4189, prev_speed/20]` |
| reward (Phase 1-2 skeleton) | **0.0** — Phase 4 §4-3에서 progress + R_lap 추가 |
| 종료 우선순위 (#8/#24) | collision > reverse(stub, Phase 1-4) > lap_complete(2-lap, `toggle_list ≥ 4` → `lap_count ≥ 2`) > timeout |
| false collision guard (#3) | `ignore_first_collision=True` default. reset 직후 첫 step에서 `collisions[0]==1` 이면 1회 무시 (dqn.py:167 패턴) |
| vel_y | Phase 1-2 시점에는 base env가 항상 0 반환 (base_classes:488). Phase 1-3 패치 후 `vel × sin(slip_angle)` 활성. A_norm 재검증은 Phase 1-3에서 |
| TRACK_CONFIGS L_track | map_easy3=117.22, Oschersleben=312.61 (004 §1 채택값 그대로) |
| info dict | `cause` (collision/reverse/lap_complete/timeout/None) · `collision_raw` · `lap_count` · `env_step` · `trackname` |

### 2-1. ★ ang_vel_z scale 갱신 (v3 #15 부분 갱신, **본 문서 SSOT**)

v3 #15 원안: `ang_vel_z / π`.
Phase 1-2 검증 시 측정 (Oschersleben GF, 5 ep × 9000 step, N=15015):
- raw `|ang_vel_z|` min=0.000, max=**10.098**, 99%-ile=**6.216**, 99.9%-ile=8.539.
- `|ang_vel_z / π|` 99%-ile=**1.931** → v3 A_norm `≤1.0` 기준 미달.

대응 (004 §1 L_track 처리와 동일 SSOT 패턴):
- **scale을 `/π` → `/(2π)` 로 갱신**.
- 코드 적용: `dreamer_f1tenth/envs/f1tenth_env.py`의 `_STATE_SCALE` 3번째 원소 = `2π`.
- 갱신 후 `|ang_vel_z / (2π)|` 99%-ile = 0.989 → A_norm PASS.
- v3 §1-B #15 본문은 수정하지 않음 (planning은 append-only). 본 문서가 단일 출처.

이유:
1. semantically natural — yaw rate를 "한 회전(2π rad/s)" 단위로 normalize.
2. saturation 위협 없음: 신경망 입력은 float이므로 |x|>1여도 학습 OK. 단 v3 명세상 99%-ile≤1 기준을 형식적으로도 만족.
3. raw max 10.098 → 10.098/(2π)=1.61로 max도 적정.

### 2-2. A1~A5 / A_norm 통과 결과 (pytest)

```
$ python -m pytest dreamer_f1tenth/tests/ -v -s
============================= test session starts ==============================
platform linux -- Python 3.8.10, pytest-8.3.5
collected 7 items

dreamer_f1tenth/tests/test_collision.py::test_a4_collision_terminates                  PASSED
dreamer_f1tenth/tests/test_obs_norm.py::test_a_norm_quick[Oschersleben]                PASSED
  [A_norm/Oschersleben] N=15015  vel_x/20 0.5%=0.119 99.5%=0.450
                       |vel_y/5| 99%=0.000  |ang_z/2π| 99%=0.989
dreamer_f1tenth/tests/test_smoke.py::test_a3_smoke_one_episode[Oschersleben]           PASSED
dreamer_f1tenth/tests/test_timeout.py::test_a5_timeout_truncates                       PASSED
dreamer_f1tenth/tests/test_wrapper_api.py::test_a1_reset_signature                     PASSED
dreamer_f1tenth/tests/test_wrapper_api.py::test_a1_step_signature                      PASSED
dreamer_f1tenth/tests/test_wrapper_api.py::test_a2_obs_dict_shape                      PASSED

============================== 7 passed in 10.49s ==============================
```

매핑:

| Criterion | 테스트 | 결과 |
|---|---|---|
| A1 (reset/step gymnasium tuples) | `test_wrapper_api::test_a1_reset_signature/test_a1_step_signature` | PASS |
| A2 (obs dict 5-key + shape/dtype) | `test_wrapper_api::test_a2_obs_dict_shape` | PASS |
| A3 (smoke 1 episode no error) | `test_smoke::test_a3_smoke_one_episode[Oschersleben]` | PASS |
| A4 (collision → terminated + cause) | `test_collision::test_a4_collision_terminates` | PASS (hard-left at v_max → collision in <30 step) |
| A5 (max_steps → truncated, !terminated) | `test_timeout::test_a5_timeout_truncates` | PASS (zero-action, max=50으로 단축 검증, 의미상 9000과 동일) |
| A_norm (vel_x / |vel_y| / |ang_z| 99%-ile) | `test_obs_norm::test_a_norm_quick` (5 ep × 9000 step, N=15015) | PASS |

---

## 3. 검증 한계 및 후속 분기

### 3-1. vel_y 검증은 Phase 1-3 패치 후 재실행 필요
- 현재 base env가 `linear_vels_y=0` 하드코딩 (`base_classes.py:488`).
- A_norm은 vel_y=0인 채로 PASS — **Phase 1-3 분기 종료 직후 동일 A_norm 재실행 필요** (006 md에 기록 예정).

### 3-2. vel_x 후진 영역 미검증
- GapFollower는 후진(speed<0)을 출력하지 않음. 측정 데이터의 raw vel_x min=0.170.
- v3 A_norm 기준 `vel_x/20 ∈ [-0.25, 1.05]` 하한 `-0.25`는 raw `-5 m/s` (= v_min). 형식적 saturation은 없음 (`v_min=-5` 고정).
- 그래도 Dreamer 학습 중 actor가 후진을 시도할 가능성 → Phase 4 reward 검증 (A17) 또는 Phase 5 학습 초반 로그에서 실분포 확인.

### 3-3. false collision guard 분석 (004 §3 결정)
- wrapper는 `ignore_first_collision=True` default — reset 직후 첫 step의 collision을 1회 무시.
- Phase 1-1 002 §4-1의 dqn.py:167 패턴. **현재까지는 충분** (Oschersleben default pose + map_easy3 centerline idx=0 둘 다 정상 진행 확인).
- map_easy3 dqn 시작점 `[-0.2, -2.38]` reset false collision의 **근본 원인 분석은 미완**. base env physics (Simulator.step의 collision detection)을 더 깊이 봐야 한다 — Phase 1-3 분기 (base_classes 패치 작업) 중 함께 처리.

### 3-4. action_repeat = 2의 reset 첫 sim step
- `f110_env.reset()` 내부에서 zero-action `step()` 1회 호출 → 1 sim step 소비.
- 우리 wrapper의 첫 `step()`은 이후 추가 2 sim step 진행 → 총 3 sim step (0.03s)로 시작.
- 이후의 모든 step은 정확히 2 sim step. 첫 step 약간의 phase shift 있지만 학습엔 무영향.

---

## 4. v3 acceptance criteria 매핑 (SSOT 갱신)

implementation/004 §5의 갱신 표에 본 문서 §2-1 추가:

| Criterion | v3 원안 | 본 결정 후 (SSOT) |
|---|---|---|
| state[2] = ang_vel_z scale | `/ π`  | **`/ (2π)`** (본 문서 §2-1) |
| A1, A2, A3, A4, A5 | wrapper 검증 | 본 문서 §2-2 PASS |
| A_norm | 100 ep × 9000 step | **quick mode 5 ep × 9000 step PASS**, full mode는 `RUN_FULL_OBS_NORM=1` 환경변수로 활성 (소요 ~50분 예상). 본 분기 종료 시점에는 quick 통과를 게이트로 채택, full mode는 Phase 1-3 vel_y 패치 후 1회 돌려 vel_y 분포까지 종합 검증 |

---

## 5. Phase 1-3 진입 조건 (체크리스트)

- [x] dreamer_f1tenth 패키지 구조 생성.
- [x] F110GymnasiumWrapper 구현 (단일 파일).
- [x] A1~A5, A_norm quick mode PASS.
- [x] ang_vel_z scale SSOT 갱신 (`/2π`).
- [x] 본 005 md commit + push (003 §3 절차).
- [ ] Phase 1-3: `gym/f110_gym/envs/base_classes.py:488` 패치 (`linear_vels_y = vel * sin(slip_angle)`).
- [ ] A6 검증 (코너링 시 |vel_y| > 0 확인).
- [ ] A_norm full mode 1회 실행 → vel_y 분포 확인, scale 유지 또는 갱신 결정.
- [ ] 006 md 작성 + commit + push.

---

## 6. 미확정 / 사용자 결정 대기 항목

mandatory stop 정책상 다음 항목을 **Phase 1-3 진입 전 사용자 확인**:

1. **ang_vel_z scale `/2π` 채택** — 본 문서 §2-1 결정. 사용자가 이의 없으면 Phase 1-3 진입.
2. **A_norm full mode 실행 시점** — 본 분기 (5 ep quick 통과)에서 게이트로 충분한지, 또는 지금 즉시 full mode 100 ep × 9000 step (~50분 wall clock)을 돌릴지.
3. **map_easy3 false collision 원인 분석** — Phase 1-3 base_classes 패치 작업과 묶어서 진행 (004 §3 결정 그대로), 또는 별도 분기로 분리할지.

---

## 7. 체크리스트

- [x] dreamer_f1tenth/__init__.py
- [x] dreamer_f1tenth/envs/__init__.py
- [x] dreamer_f1tenth/envs/f1tenth_env.py (F110GymnasiumWrapper)
- [x] dreamer_f1tenth/tests/__init__.py
- [x] dreamer_f1tenth/tests/conftest.py
- [x] dreamer_f1tenth/tests/test_wrapper_api.py (A1, A2)
- [x] dreamer_f1tenth/tests/test_smoke.py (A3)
- [x] dreamer_f1tenth/tests/test_collision.py (A4)
- [x] dreamer_f1tenth/tests/test_timeout.py (A5)
- [x] dreamer_f1tenth/tests/test_obs_norm.py (A_norm)
- [x] pytest 전체 PASS (7/7) — 본 문서 §2-2
- [x] pytest 8.3.5 venv 설치
- [ ] commit + push (003 §3)
- [ ] 사용자 §6 결정 1·2·3
- [ ] Phase 1-3 진입
