# venv 재구축 & 동작 검증 (노트북, WSL2)

> 2026-05-20 세션 기록. 002 문서 작성 시점과 폴더 구조/venv 상태가 어긋나 있어 재정렬.

---

## 1. 진단: 002 문서 ↔ 실제 불일치

| 항목 | 002 문서 | 실제 (재정렬 전) |
|------|---------|----------------|
| 프로젝트 경로 | `~/f1tenth-riders-quickstart/` | `~/f1tenth_RL_project/` (통합됨) |
| venv 경로 | `~/f1tenth_env/` | `~/f1tenth_RL_project/env/` |
| venv 상태 | 정상 | `env/bin/pip` shebang이 옛 경로 가리킴 → pip 실행 불가 |
| torch | 2.4.1 CPU | 2.4.1+cu121 (단 `cuda.is_available()=False`) |
| f110-gym | 설치됨 | `pip install -e gym` 미수행 (sys.path 우회로만 import) |

원인: 폴더 이름 변경(`f1tenth-riders-quickstart` → `f1tenth_RL_project`) 후 venv 재생성을 안 했음.

---

## 2. 조치: venv 재생성 (A안)

```bash
cd /home/dlacksdn/f1tenth_RL_project
mv env env.broken                          # 백업
python3.8 -m venv env
source env/bin/activate

pip install "pip==22.0.3"
pip install "setuptools<58.0.0"
pip install -e gym                         # f110-gym editable (numpy/numba/gym/scipy/pyglet 등 자동)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib==3.7.5              # torch index에 없으므로 PyPI에서 별도 설치

rm -rf env.broken                          # 검증 후 삭제
```

주의: `torch`와 `matplotlib`을 한 번에 설치하면 `--index-url`이 pytorch 전용이라 matplotlib을 못 찾고 트랜잭션 전체 롤백됨 → **분리 설치 필요**.

---

## 3. 최종 패키지 상태

```
python      3.8.10
torch       2.4.1+cpu   (cuda=False)
torchvision 0.19.1+cpu
torchaudio  2.4.1+cpu
gym         0.18.0
numpy       1.24.4
matplotlib  3.7.5
numba       0.58.1
llvmlite    0.41.1
scipy       1.10.1
pyglet      1.5.0
Pillow      7.2.0
f110-gym    0.2 (editable, /home/dlacksdn/f1tenth_RL_project/gym)
```

---

## 4. 동작 검증

### 헤드리스 smoke test → 정상
```python
env = gym.make('f110_gym:f110-v0',
               map='.../maps/Oschersleben', map_ext='.png', num_agents=1)
obs, r, done, info = env.reset(poses=np.array([[0.0702245, 0.3002981, 2.79787]]))
# obs keys: ego_idx, scans, poses_x/y/theta, linear_vels_x/y, ang_vels_z,
#           collisions, lap_times, lap_counts
# scan shape: (1080,)
# 5 steps OK, done=False
```

### 렌더링 (`python -m pkg.main`) → 부분 성공
- 1차: `pyglet.window.NoSuchConfigException` — WSLg가 MSAA(`sample_buffers=1, samples=4`) 미지원
- 진단: `screen.get_matching_configs(Config(sample_buffers=1, samples=4, ...))` → 0개 매칭
- 시도: [rendering.py:70-73](../../gym/f110_gym/envs/rendering.py)에서 MSAA 옵션 제거 → 창은 떴으나 **까만 빈 창** (맵 안 보임)
- 결정: **수정 원복** (`git checkout --`)
  - 이유 1: `rendering.py`는 git 추적 대상 → push되면 집컴(GPU 환경)에도 MSAA 빠진 버전 동기화됨
  - 이유 2: 노트북에선 까만 창 띄워봤자 의미 없음
  - 이유 3: 집컴은 GPU/네이티브 GL에서 정상 렌더링 확인됨

---

## 5. 운영 분업

| 머신 | 용도 | 가능 작업 |
|------|------|----------|
| 노트북 (WSL2, CPU) | 로직 개발 | env reset/step, observation 가공, reward 함수 설계, 코드 작성 |
| 집컴 (GPU) | 시각 확인 + 훈련 | `env.render()`, DQN 등 학습 루프 |

집컴으로 옮길 때:
1. git pull
2. `python3.8 -m venv env && pip install ...` (CUDA torch는 `--index-url https://download.pytorch.org/whl/cu121` 등)
3. `dqn.py`에 GPU 분기 코드 필요 시 추가 (`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`)
   - 현재 `dqn.py`엔 `cuda`/`device` 관련 코드 없음 → 추가 작업 필요

---

## 6. 이번 세션에서 코드 변경 없음

- `env/` 폴더만 새로 생성 (`.gitignore`로 git 영향 0)
- `base_classes.py`, `dqn.py`: 기존 수정 그대로 유지 (이번 세션에서 안 건드림)
- `rendering.py`: 일시 수정 후 원복 → git clean

---

## 7. 002 문서와의 관계

002는 "이런 식으로 세팅했다"의 스냅샷, 004는 "그 스냅샷이 어긋난 걸 재정렬했다"의 기록.
append-only 원칙대로 002는 손대지 않음. 향후 머신 옮기거나 venv 재구축할 일 있으면 이 문서 참고.
