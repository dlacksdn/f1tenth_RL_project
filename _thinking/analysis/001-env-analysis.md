# F1Tenth Gym 환경 분석

> 다른 에이전트가 처음 들어와도 바로 이해할 수 있도록 정리한 환경 분석 문서.
> 대상 경로: `/home/dlacksdn/f1tenth-riders-quickstart`

---

## 0. 프로젝트 구조 요약

```
f1tenth-riders-quickstart/
├── gym/f110_gym/envs/        # 핵심 시뮬레이터 (Pure Python)
│   ├── f110_env.py           # OpenAI Gym 인터페이스, 리워드 정의
│   ├── base_classes.py       # RaceCar / Simulator 클래스
│   ├── dynamic_models.py     # 차량 동역학 (Single Track)
│   ├── laser_models.py       # 2D LiDAR 시뮬레이션
│   └── collision_models.py   # GJK 충돌 검사
├── f1tenth_gym_ros/          # ROS Bridge (제출/평가용)
│   ├── scripts/gym_bridge_bare.py
│   ├── params.yaml           # ROS 토픽 / LiDAR 고정값
│   └── maps/                 # 트랙 맵 (yaml + png)
├── pkg/                      # 에이전트 코드
│   ├── src/pkg/drivers.py    # GapFollower, DisparityExtender
│   ├── src/pkg/main.py       # Pure Gym 실행 진입점
│   └── nodes/f1tenth_ros_agent.py
└── docker-compose.yml
```

**두 가지 실행 모드:**
- **Pure Gym**: `pkg/src/pkg/main.py` — 학습/실험용 Python 직접 실행
- **ROS Bridge**: Docker + ROS — 제출/평가용

---

## 1. 변경 가능한 파라미터 (코드 위치 포함)

### 1-1. 차량 물리 파라미터
**위치**: [f110_env.py:491-494](../../gym/f110_gym/envs/f110_env.py#L491-L494)

```python
params = {
    'mu': 1.0489,        # 노면 마찰 계수
    'C_Sf': 4.718,       # 앞바퀴 코너링 강성
    'C_Sr': 5.4562,      # 뒷바퀴 코너링 강성
    'lf': 0.15875, 'lr': 0.17145,    # 앞/뒤축 거리 (m)
    'h': 0.074,          # 무게중심 높이
    'm': 3.74,           # 질량 (kg)
    'I': 0.04712,        # Z축 관성 모멘트
    's_min': -0.4189, 's_max': 0.4189,    # 조향각 한계 (rad) ≈ ±24°
    'sv_min': -3.2, 'sv_max': 3.2,         # 조향 속도 한계
    'v_switch': 7.319,   # 휠스핀 전환 속도
    'a_max': 9.51,       # 최대 가속도
    'v_min': -5.0, 'v_max': 20.0,    # 속도 한계
    'width': 0.31, 'length': 0.58,    # 차체 크기
}
```

### 1-2. LiDAR 파라미터
**위치**: [base_classes.py:56](../../gym/f110_gym/envs/base_classes.py#L56), [laser_models.py:327](../../gym/f110_gym/envs/laser_models.py#L327)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `num_beams` | 1080 | 레이저 빔 수 |
| `fov` | 4.7 rad (≈269°) | 시야각 |
| `max_range` | 30.0 m | 최대 감지 거리 |
| `std_dev` | 0.01 | 노이즈 표준편차 |

ROS 쪽 고정값: [params.yaml:22-24](../../f1tenth_gym_ros/params.yaml#L22-L24)

### 1-3. 시뮬레이션 타임스텝
[f110_env.py:505](../../gym/f110_gym/envs/f110_env.py#L505) — `self.timestep = 0.01` (100 Hz)

### 1-4. 드라이버 알고리즘 파라미터
**위치**: [drivers.py](../../pkg/src/pkg/drivers.py)

- **GapFollower** (line 4-11): `BUBBLE_RADIUS`, `STRAIGHTS_SPEED=9.0`, `CORNERS_SPEED=6.0`, `STRAIGHTS_STEERING_ANGLE=π/18`
- **DisparityExtender** (line 127-132): `CAR_WIDTH=0.31`, `DIFFERENCE_THRESHOLD=2.0`, `SPEED=5.0`, `SAFETY_PERCENTAGE=300`

### 1-5. 리워드 함수
**위치**: [f110_env.py:660-686](../../gym/f110_gym/envs/f110_env.py#L660-L686)

```python
reward = 1000 * self.timestep        # 매 스텝 +10
if argmin(scans) in [300, 780]:
    reward -= 1                      # 전방 장애물
else:
    reward += 2                      # 측면 여유
if min(scans) < 0.5:
    reward -= 5                      # 근접 경고
if collision:
    reward = 0                       # 충돌 (음수 X)
```

### 1-6. 맵 선택
**위치**: [main.py:19](../../pkg/src/pkg/main.py#L19)

`Oschersleben`, `SOCHI`, `SOCHI_OBS`, `SILVERSTONE`, `SILVERSTONE_OBS`, `map_easy3`

---

## 2. 주요 구조적 이슈 (실제 학습/제출에 영향)

### 2-1. 충돌 리워드 = 0 (음수 페널티 아님)
[f110_env.py:685](../../gym/f110_gym/envs/f110_env.py#L685)
- 매 스텝 +10씩 받다가 충돌하면 그 step만 0이 되고 종료
- **랩 완료 보너스도 없음** → 정책이 "느리지만 안전한 주행"으로 수렴하기 쉬움

### 2-2. 관측되지 않는 상태 변수
[base_classes.py:488](../../gym/f110_gym/envs/base_classes.py#L488)
- 실제 state는 7차원 `[x, y, steer_angle, vel, yaw, yaw_rate, slip_angle]`
- 그러나 **`steer_angle`, `slip_angle` 미공개** — 코너링 판단 핵심인데 에이전트가 못 봄
- `linear_vels_y`는 동역학이 계산함에도 `0`으로 하드코딩

### 2-3. ROS Bridge vs Pure Gym 타이밍 불일치
[gym_bridge_bare.py:211-212](../../f1tenth_gym_ros/scripts/gym_bridge_bare.py#L211-L212)

| 항목 | 주파수 |
|------|--------|
| 물리 시뮬 | 100 Hz |
| ROS 드라이브 명령 | 50 Hz |
| ROS 관측 발행 | 250 Hz |

→ ROS 모드에서는 **제어 명령 1개당 물리 스텝 2회**. 학습은 Pure Gym, 제출은 ROS — 같은 정책이 다르게 동작.

### 2-4. 충돌 후 동작이 모드별로 다름
- **Pure Gym** ([f110_env.py:600](../../gym/f110_gym/envs/f110_env.py#L600)): 충돌 시 `done=True` 즉시 종료
- **ROS Bridge** ([gym_bridge_bare.py:278-281](../../f1tenth_gym_ros/scripts/gym_bridge_bare.py#L278-L281)): 충돌해도 시뮬 계속, 차만 정지

### 2-5. LiDAR 위치 오프셋 미적용
- params.yaml에 `scan_distance_to_base_link: 0.275` (LiDAR가 차량 앞쪽 27.5cm)
- Pure Gym ([base_classes.py:297](../../gym/f110_gym/envs/base_classes.py#L297))에서는 차량 중심에서 스캔 계산
- 실차 ↔ 시뮬 간 27.5cm 차이

### 2-6. 맵 이진화 임계값이 YAML과 불일치 (silent bug)
[laser_models.py:374-375](../../gym/f110_gym/envs/laser_models.py#L374-L375)
- 코드는 `pixel > 128` 하드코딩
- YAML의 `occupied_thresh: 0.45`, `free_thresh: 0.196`이 **무시됨**

### 2-7. 후진 랩 카운트 버그 (reward hacking)
[f110_env.py:589-598](../../gym/f110_gym/envs/f110_env.py#L589-L598)
- 출발선 근접 토글로 카운트 — 방향성 체크 없음
- 차량이 후진으로 출발선을 통과해도 lap_count 증가

---

## 3. 분석/개선 우선순위

| Tier | 항목 | 영향 |
|------|------|------|
| **S** | 리워드 함수 재설계 (충돌 페널티 음수화, 랩 완료 보너스) | 학습 수렴 방향 |
| **A** | `steer_angle` / `slip_angle` 관측 추가 | 학습 신호 |
| **A** | 맵 이진화 임계값을 YAML 값으로 교체 | Silent bug |
| **B** | ROS/Gym 타이밍 일치화 또는 학습 시 50Hz 가정 | sim2sim 갭 |
| **B** | 후진 랩 카운트 방지 | Reward hacking 차단 |
| **B** | LiDAR 27.5cm 오프셋 적용 | 실차 일치 |
| **C** | 충돌 후 종료 정책 ROS/Gym 통일 | 평가 일관성 |

---

## 4. 출발 포즈 / 맵 좌표 메모

- `Oschersleben`: `[0.0702245, 0.3002981, 2.79787]` (1대), resolution `0.04295 m/px`
- `SOCHI`: `[0.8007017, -0.2753365, 4.1421595]` (1대)
- `map_easy3`: resolution `0.02 m/px`
- 초기 포즈는 [main.py:50-66](../../pkg/src/pkg/main.py#L50-L66), [gym_bridge_bare.py:178-200](../../f1tenth_gym_ros/scripts/gym_bridge_bare.py#L178-L200)에 **하드코딩** (랜덤화 wrapper 없음)
