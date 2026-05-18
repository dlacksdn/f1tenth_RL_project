# F1TENTH 환경 세팅 현황

> WSL2 (Ubuntu 20.04, Python 3.8) 기준으로 세팅 완료

---

## 폴더 구조

```
~/
├── f1tenth_env/                          ← Python 가상환경 (라이브러리)
├── f1tenth-riders-quickstart/            ← 시뮬레이터 + 학습 코드
│   ├── gym/f110_gym/envs/
│   │   ├── f110_env.py                   ← 교수님 제공 파일로 교체됨
│   │   └── base_classes.py               ← check_collision 수정됨
│   └── pkg/src/pkg/
│       ├── dqn.py                        ← 교수님 제공 파일 + 추가 수정됨
│       └── maps/
│           ├── map_easy3.png             ← 교수님 제공
│           ├── map_easy3.yaml            ← 교수님 제공
│           └── (기타 기본 맵들)
├── f1tenth_ws/                           ← ROS2 하드웨어 드라이버 (이 수업과 무관)
└── reference_file/                       ← 참고 자료
    ├── 001-configuration.md
    ├── 002-setting.md                    ← 이 파일
    ├── f110_env.py                       ← 원본 보관
    ├── dqn.py                            ← 원본 보관
    ├── map_easy3.png
    └── map_easy3.yaml
```

---

## 설치된 패키지

가상환경 경로: `~/f1tenth_env`

| 패키지 | 버전 |
|--------|------|
| pip | 22.0.3 (교수님 지정) |
| setuptools | < 58.0.0 (교수님 지정) |
| f110-gym | 0.2 |
| gym | 0.18.0 |
| torch | 2.4.1 (CPU 버전) |
| torchvision | 0.19.1 |
| torchaudio | 2.4.1 |
| numpy | 1.24.4 |
| matplotlib | 3.7.5 |
| numba | 0.58.1 |

> **GPU 환경(집 PC)에서는** `pip install torch torchvision torchaudio` 후 CUDA 버전 별도 설치 필요.
> 코드 내부에서 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` 패턴으로 자동 분기.

---

## 수정된 파일

### 1. `base_classes.py` — check_collision 수정

```python
# 수정 전
self.collisions = np.maximum(self.collisions, new_collisions)

# 수정 후 (충돌 상태가 누적되지 않고 매 스텝마다 갱신)
# self.collisions = np.maximum(self.collisions, new_collisions)
self.collisions = new_collisions
```

### 2. `dqn.py` — RACETRACK 및 시작 포즈 수정

```python
# 수정 전
RACETRACK = 'SOCHI'
poses = np.array([[0.8007017, -0.2753365, 4.1421595]])  # SOCHI 시작점

# 수정 후
RACETRACK = 'map_easy3'
# SOCHI 시작점: poses = np.array([[0.8007017, -0.2753365, 4.1421595]])
poses = np.array([[-0.2000, -2.3800, 1.745329]])  # map_easy3 좌측 직선 구간
```

> 시작 포즈는 `map_easy3.yaml`의 `starting_angle: 1.745329`과 이미지 픽셀 분석으로 계산.
> 실행 후 차가 벽에 갇히면 좌표 조정 필요.

---

## 실행 방법

```bash
# 가상환경 활성화
source ~/f1tenth_env/bin/activate

# 환경 동작 확인 (선택)
cd ~/f1tenth-riders-quickstart/pkg/src
python -m pkg.main

# DQN 학습 실행
cd ~/f1tenth-riders-quickstart/pkg/src/pkg
python dqn.py
```

---

## 참고 사항

- `f1tenth_ws/`는 실제 F1TENTH 차량용 ROS2 하드웨어 드라이버로, 이 시뮬레이터 프로젝트와 **무관**
- 렌더링 없이 빠른 학습을 원하면 `dqn.py`의 `env.render()` 호출 제거
- 맵 순서: **map_easy3** → Oschersleben (교수님 권장)
- 교수님 코드 출처: https://oss.inha.ac.kr/project/detail/176
