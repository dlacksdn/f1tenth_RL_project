# F1TENTH Simulator Configuration

> 출처: AIE4003_RL_F1TENTH.pdf (RILS LAB @ INHA UNIV, Woo-Jin Ahn)

---

## 1. 환경 설치 순서

1. Visual Studio Code
2. Anaconda (가상환경: `f1tenth`, Python 3.8)
3. Git
4. PyTorch
5. F1tenth Gym

---

## 2. Anaconda 가상환경 생성

```bash
conda create -n f1tenth python=3.8
conda activate f1tenth
```

---

## 3. F1tenth Gym 설치

```bash
# setuptools 및 pip 버전 고정 (중요)
pip install "setuptools<58.0.0"
python -m pip install "pip==22.0.3"

# 리포지토리 클론 및 gym 설치
git clone https://gitlab.com/acrome-colab/riders-poc/f1tenth-riders-quickstart --config core.autocrlf=input
cd f1tenth-riders-quickstart
pip install --user -e gym
```

동작 확인:
```bash
cd pkg/src
python -m pkg.main
```

---

## 4. 코드 파일 배치

| 파일 | 위치 |
|------|------|
| `f110_env.py` (기존 파일 대체) | `f1tenth-riders-quickstart/gym/f110_gym/envs/f110_env.py` |
| `dqn.py` | `f1tenth-riders-quickstart/pkg/src/pkg/dqn.py` |
| `map_easy.png`, `map_easy.yaml` | `f1tenth-riders-quickstart/pkg/src/pkg/maps/` |

코드 출처: https://oss.inha.ac.kr/project/detail/176

---

## 5. 코드 수정: `base_classes.py`

### 파일 경로
```
f1tenth-riders-quickstart/gym/f110_gym/envs/base_classes.py
```

### `check_collision` 함수 수정

충돌 후 에피소드가 끝나지 않고 계속 충돌 상태로 유지되는 문제를 해결하기 위해
`self.collisions` 업데이트 방식을 변경한다.

**수정 전:**
```python
def check_collision(self):
    """
    Checks for collision between agents using GJK and agents' body vertices

    Args:
        None

    Returns:
        None
    """
    # get vertices of all agents
    all_vertices = np.empty((self.num_agents, 4, 2))
    for i in range(self.num_agents):
        all_vertices[i, :, :] = get_vertices(np.append(self.agents[i].state[0:2], self.agents[i].state[4]),
                                              self.params['length'], self.params['width'])

    # calculate new collisions
    new_collisions, self.collision_idx = collision_multiple(all_vertices)

    # if collision becomes 1, it stays so until end
    self.collisions = np.maximum(self.collisions, new_collisions)  # ← 이 줄
```

**수정 후:**
```python
def check_collision(self):
    """
    Checks for collision between agents using GJK and agents' body vertices

    Args:
        None

    Returns:
        None
    """
    # get vertices of all agents
    all_vertices = np.empty((self.num_agents, 4, 2))
    for i in range(self.num_agents):
        all_vertices[i, :, :] = get_vertices(np.append(self.agents[i].state[0:2], self.agents[i].state[4]),
                                              self.params['length'], self.params['width'])

    # calculate new collisions
    new_collisions, self.collision_idx = collision_multiple(all_vertices)

    # if collision becomes 1, it stays so until end
    # self.collisions = np.maximum(self.collisions, new_collisions)  # ← 주석 처리
    self.collisions = new_collisions                                  # ← 이 줄로 교체
```

> 핵심: `np.maximum(...)` 줄을 주석 처리하고, `self.collisions = new_collisions` 로 교체.
> 이렇게 하면 충돌 상태가 누적되지 않고 매 스텝마다 새로 갱신된다.

---

## 6. DQN 학습 루프 (슬라이드 33 코드)

```python
for n_epi in range(10000):
    epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
    obs, r, done, info = env.reset(poses=poses)
    s = preprocess_lidar(obs['scans'][0])
    done = False

    env.render()
    laptime = 0.0

    while not done:
        actions = []
        a = q.sample_action(torch.from_numpy(s).float(), epsilon, memory.size())
        action = q.action_to_stearing(a)
        actions.append(action)
        actions = np.array(actions)

        obs, r, done, info = env.step(actions)
        s_prime = preprocess_lidar(obs['scans'][0])

        done_mask = 0.0 if done else 1.0
        memory.put((s, a, r / 100, s_prime, done_mask))
        s = s_prime

        laptime += r
        env.render(mode='human_fast')

        if done:
            laptimes.append(laptime)
            plot_durations(laptimes)
            lap = round(obs['lap_times'][0], 3)
            if int(obs['lap_counts'][0]) == 2 and fastlap > lap:
                torch.save(q.state_dict(), work_dir + '_' +
                           round(obs['lap_times'][0], 3)) + '_' + s
                fastlap = lap
            break

    if memory.size() > train_start:
        train(q, q_target, memory, optimizer)
```

---

## 7. 관측값(obs) 구성

`env.step(actions)` 반환값 `obs`에서 사용 가능한 키:

| 키 | 타입 | 예시 값 | 설명 |
|----|------|---------|------|
| `ego_idx` | int | 0 | 에이전트 인덱스 |
| `scans` | list\[array\] | array\[1080\] | LiDAR 스캔값 |
| `poses_x` | list | \[0.8007017\] | x 위치 |
| `poses_y` | list | \[-0.2753365\] | y 위치 |
| `poses_theta` | list | \[4.1421595\] | 방향각 (rad) |
| `linear_vels_x` | list | \[0.042795\] | x 방향 선속도 |
| `linear_vels_y` | list | \[0.0\] | y 방향 선속도 |
| `ang_vels_z` | list | \[0.0\] | z축 각속도 |
| `collisions` | ndarray | \[0.\] | 충돌 여부 |
| `lap_times` | ndarray | \[0.02\] | 현재 랩 타임 |
| `lap_counts` | ndarray | \[0.\] | 완주 횟수 |

### LiDAR 스캔 특성
- 총 **1080개** 빔, **270°** 범위
- 인덱스 **0** = 차량 기준 오른쪽, **1079** = 왼쪽

---

## 8. Custom 환경 구조 (`f110_env.py`)

```python
class F110Env(gym.Env, utils.EzPickle):
    """OpenAI gym environment for F1TENTH..."""
    metadata = {'render.modes': ['human', 'human_fast']}

    def __init__(self, **kwargs): ...
    def __del__(self): ...
    def _check_done(self): ...
    def _update_state(self, obs_dict): ...
    def step(self, action): ...   # ← reward 설정 가능
    def reset(self, poses): ...
    def update_map(self, map_path, map_ext): ...
    def update_params(self, params, index=-1): ...
    def render(self, mode='human'): ...
```

- **주석에서 차량 재원 확인 가능**
- **`step()` 함수 내부에서 reward 설계**

---

## 9. 맵별 시작 위치

> **PDF에는 구체적인 초기 poses 값이 명시되어 있지 않습니다.**
> 슬라이드에서는 `env.reset(poses=poses)`로 호출하며, poses 변수를 별도로 설정한다고만 언급됩니다.

일반적인 poses 설정 형식 (F1tenth Gym 표준):

```python
# poses = [[x, y, theta], ...]
poses = np.array([[0.0, 0.0, 0.0]])  # 예시
obs, r, done, info = env.reset(poses=poses)
```

맵별 실제 시작 좌표는 각 맵의 `.yaml` 파일 또는 `maps/` 폴더 내 설정 파일에서 확인하거나,
직접 시뮬레이터를 실행해 적절한 출발 지점을 찾아야 합니다.

---

## 10. Map Easy 활용 팁

- 처음부터 Oschersleben(큰 트랙)으로 학습하면 알고리즘 완성도 확인이 어려움
- **Map Easy** (간단한 트랙) → **Oschersleben** 순서로 학습 권장
- `env.render()` 호출을 제거하면 렌더링 없이 빠른 학습 가능
- `map_easy.png`, `map_easy.yaml`을 `pkg/src/pkg/maps/`에 복사하면 사용 가능

---

## 11. 기타 필요 패키지

```bash
pip install matplotlib
# 실행 시 오류 발생하는 패키지는 오류 메시지 확인 후 추가 설치
```
