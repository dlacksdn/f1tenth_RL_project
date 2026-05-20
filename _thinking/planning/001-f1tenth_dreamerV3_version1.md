# 001 — F1TENTH Dreamer-V3 베이스라인 구현 계획 (Version 1)

> **작성일**: 2026-05-20
> **모드**: Plan (interview workflow, direct draft — consensus/critic 미적용)
> **선행 문서**: [analysis/001](../analysis/001-env-analysis.md), [analysis/002](../analysis/002-Select_Implementation.md), [analysis/003~006](../analysis/003-dreamer_code_analysis_part1.md), [env_setting/001~004](../env_setting/001-configuration.md)
> **목적**: F1TENTH 시뮬레이션 환경에서 Dreamer-V3 (NM512/dreamerv3-torch 12M) 베이스라인 학습 계획. LeWorldModel 기반 Offline RL 추가과제는 별도 계획.

---

## 0. Requirements Summary

### 0-1. 프로젝트 목표

1. **Map Easy 완주** — 평가 비중 30%
2. **Oschersleben 완주 + lap time 최적화** — 평가 비중 10%
3. **알고리즘 발표** — 평가 비중 60%. state/action/reward/episode 종료조건/curriculum 설계 스토리.
4. **100초대 성능 policy 저장** — LeWorldModel(Offline RL 추가과제)용 데이터 수집 정책 weights 별도 보관.

### 0-2. 환경·하드웨어 제약

| 항목 | 값 |
|---|---|
| 학습 머신 | 집컴 RTX 4060Ti, 12-core CPU, 2TB SSD |
| 개발 머신 | 노트북 (WSL2, CPU only, 로직 개발만) |
| 환경 | F1TENTH Gym, Python 3.8, torch 2.4.1 |
| 시뮬레이터 경로 | `/home/dlacksdn/f1tenth_RL_project/` (gym + pkg 통합) |
| Dreamer 경로 | `/home/dlacksdn/dreamerv3-torch/` (NM512 클론) |
| 제출 포맷 | Pure Gym only (ROS 통합 OUT-of-scope) |

### 0-3. 차량 spec — 변경 정책

| 카테고리 | 파라미터 | 변경 정책 |
|---|---|---|
| 차체 (변경 금지) | `m, I, lf, lr, h, mu, C_Sf, C_Sr, width, length` | 절대 금지 |
| 액추에이터 한계 (변경 금지) | `sv_min/sv_max=±3.2, a_max=9.51` | 동역학 자체가 바뀌어 평가환경과 불일치 위험 → 금지 |
| 제어 한계 (변경 가능, 베이스라인은 default) | `s_min/s_max=±0.4189, v_min/v_max=[-5, 20]` | 학습 plateau 시 ablation으로만 검토. 베이스라인은 default. 최종 평가는 default 값으로 검증 |

---

## 1. 핵심 설계 결정 (Interview 확정)

| # | 항목 | 결정 |
|---|---|---|
| 1 | 구현체 | `NM512/dreamerv3-torch` (PyTorch 포팅 표준) |
| 2 | 스코프 | 베이스라인 Dreamer + 100초대 snapshot 저장. LeWorldModel/Offline RL은 별도 계획 |
| 3 | 모델 크기 | **12M** (논문 Table 3: d=256, deter=8d=1024, base CNN ch=d/16=16, codes=d/16=16) |
| 4 | Action space | Continuous 2-dim `(steer, speed)`, `actor.dist='normal'`, `absmax=1.0`, `imag_gradient='dynamics'` |
| 5 | LiDAR encoder | **1D Conv 신설** — `ConvEncoder1D`. 1080 빔 → stride-2 5 stage → flatten |
| 6 | 보조 observation | `vel_x, vel_y, ang_vel_z, prev_action` (4 + 2 = 6 scalar) |
| 7 | Reward | **Progress + 종료 신호** — per-step progress(+α) + collision(-10) + lap complete(+50) |
| 8 | Episode | 2-lap 완주(`is_last=True`) / 충돌(`is_terminal=True`) / 180초 timeout(`is_last=True`) / 후진 감지 (terminal -10) |
| 9 | Curriculum | **순차 fine-tune**: Map Easy 500K → Oschersleben fine-tune 500K (총 1M) |
| 10 | Snapshot 정책 | **Hybrid D**: lap_time threshold 자동 저장 + `eval_every`마다 interval snapshot 전수 (총 ~16GB) |
| 11 | Step budget | 초기 1M, 유동 조정 (resume 가능) |
| 12 | Exploration | `expl_behavior='greedy'` (Dreamer 기본). F1TENTH는 dense reward라 Plan2Explore 불필요 |
| 13 | 제출 | Pure Gym only. 발표에 "ONNX export로 ROS2 노드 재사용 가능" 멘션 정도 |

---

## 2. Acceptance Criteria (Testable)

### 2-1. 환경·통합 (Phase 1 완료 기준)

- [ ] **A1** `f110_gym:f110-v0` 환경이 `gymnasium`-style API로 reset/step 가능 (`F110GymnasiumWrapper` 작성, `reset() → (obs, info)`, `step() → (obs, reward, terminated, truncated, info)`)
- [ ] **A2** obs는 dict로 `{'lidar': shape (1080,), 'state': shape (6,), 'is_first': bool, 'is_terminal': bool, 'is_last': bool}` 형태
- [ ] **A3** `python -c "import gymnasium; env = make('f110...'); obs, _ = env.reset(); obs, r, term, trunc, _ = env.step(env.action_space.sample())"`이 에러 없이 실행
- [ ] **A4** 충돌 시 `terminated=True`, `info['cause']='collision'` 검증 (테스트 스크립트 `tests/test_termination.py`)
- [ ] **A5** 180초(18000 step) 도달 시 `truncated=True`, `terminated=False` 검증
- [ ] **A6** `obs['state']`의 `vel_y`가 base_classes에서 하드코딩 0이 아닌 실제 동역학 값으로 노출됨 — `tests/test_obs_vels.py`에서 코너링 중 `|vel_y| > 0.01` 단언

### 2-2. Encoder·모델 (Phase 2 완료 기준)

- [ ] **A7** `ConvEncoder1D` 클래스가 `networks.py`에 추가되어 `(B, T, 1080)` 입력 → `(B, T, outdim)` 출력. 5 stage stride-2 conv1d, 채널 진행 1→16→32→64→128→256, 1080→540→270→135→67→33
- [ ] **A8** `MultiEncoder`가 정규식 매칭으로 `lidar`를 1D Conv 경로로, `state`를 MLP 경로로 라우팅 (configs `lidar_keys: 'lidar'`, `mlp_keys: 'state'`, `cnn_keys: '$^'`)
- [ ] **A9** `configs.yaml`에 `f1tenth: ...` block 신설, 12M hyperparameter 매핑 (`dyn_hidden=256, dyn_deter=1024, dyn_stoch=32, dyn_discrete=16, units=256, mlp_units=256, cnn_depth=16, video_pred_log=False, decoder mlp_keys='state'`)
- [ ] **A10** 모델 파라미터 수 검증 — `sum(p.numel() for p in agent.parameters() if p.requires_grad) ∈ [10M, 14M]`

### 2-3. 학습·평가 (Phase 3 완료 기준)

- [ ] **A11** Map Easy에서 500K env step 학습 후 eval lap_time 중앙값 ≤ 60초 (Map Easy 완주 능력)
- [ ] **A12** Oschersleben fine-tune 500K 후 eval에서 2-lap 완주율 ≥ 80%
- [ ] **A13** Oschersleben fine-tune 후 eval lap_time best ≤ 110초 (100초대 목표 도달)
- [ ] **A14** lap_time threshold(≤110초) 자동 snapshot이 `logdir/snapshots/policy_lap{X}s_step{Y}.pt`에 저장됨 (LeWorldModel용)
- [ ] **A15** `eval_every`마다 interval snapshot `logdir/snapshots/step_{N}k.pt` 저장됨 (분석·재현용)
- [ ] **A16** Map Easy에서 fine-tune 후 재평가 시 2-lap 완주율 ≥ 70% (catastrophic forgetting 한도 — 미달 시 snapshot D로 복구)

### 2-4. Reward·safety (재현성)

- [ ] **A17** Reward 함수가 `terminated_by_collision`, `lap_completed`, `progress_per_step` 세 component를 분리 로깅 (logger metrics)
- [ ] **A18** 후진 1초 누적 시 terminal 종료 검증 — `tests/test_reverse_guard.py` (차량을 강제 후진 액션으로 1.1초 가하면 `terminated=True, info['cause']='reverse'`)

---

## 3. Implementation Steps

### Phase 1 — Env wrapper (예상 2~3일)

#### 1-1. 디렉토리 구조 신설

```
f1tenth_RL_project/
├── env/                          # 기존 venv
├── gym/                          # 기존 f110-gym editable
├── pkg/                          # 기존
└── dreamer_f1tenth/              # 신규 — Dreamer 학습 코드
    ├── envs/
    │   └── f1tenth_env.py        # gymnasium wrapper
    ├── networks_1d.py            # ConvEncoder1D
    ├── configs_f1tenth.yaml      # 12M + f1tenth suite override
    ├── train.py                  # dreamer.py wrapper (snapshot D 정책 포함)
    └── tests/
        ├── test_termination.py
        ├── test_obs_vels.py
        └── test_reverse_guard.py
```

`/home/dlacksdn/dreamerv3-torch/`는 git clone 그대로 보존 (수정은 PR 형식으로 명확히). 단 `networks.py`에 `ConvEncoder1D` 추가는 in-place 수정 불가피 — `dreamerv3-torch/networks_f1tenth.py` 별도 파일에 정의 후 `MultiEncoder` 패치만 in-place로.

#### 1-2. `F110GymnasiumWrapper` 작성 (`dreamer_f1tenth/envs/f1tenth_env.py`)

기능:
- `f110_gym:f110-v0` 인스턴스 보유
- `reset(seed=None, options=None)` → `(obs_dict, info)`
- `step(action)` → `(obs_dict, reward, terminated, truncated, info)`
- action: `np.array([steer, speed])` ∈ `[-1, 1]^2`, env에 전달 시 `[s_min, s_max], [v_min, v_max]`로 affine 변환
- obs_dict 구성:
  - `'lidar'`: 원본 `obs['scans'][0]` shape (1080,) float32
  - `'state'`: concat `[vel_x, vel_y, ang_vel_z, steer_prev, speed_prev, ?]` shape (6,) — prev_action 2-dim 포함하면 5 dims만. 6번째는 placeholder로 `progress_ratio` (0~1) 추가
  - `'is_first'`, `'is_terminal'`, `'is_last'` bool
- reward: §3-4 함수
- terminated/truncated: §3-3 episode 종료 조건

검증: `tests/test_termination.py`, `test_obs_vels.py`, `test_reverse_guard.py` 통과.

#### 1-3. `base_classes.py` 추가 패치

analysis/001 §2-2가 지적한 `linear_vels_y=0` 하드코딩 검증:
- `f110_env.py:_update_state` 또는 `base_classes.py:update_pose` 에서 `vel_y`가 실제 동역학 출력값인지 확인
- 하드코딩이면 `state[3] * sin(state[6])` (slip_angle 기반) 또는 dynamic_models 출력에서 직접 노출하도록 패치
- 사용자 기존 `check_collision` 패치는 이미 반영됨 (env_setting/001 §5)

#### 1-4. 후진 가드 (`reverse_guard.py`)

매 step:
- 트랙 centerline tangent vector를 미리 계산 (맵 yaml 로드 시 1회)
- 차량 위치에서 가장 가까운 centerline 점의 tangent와 차량 velocity vector dot product
- `dot < 0` → 후진 카운터 증가, 0 이상이면 리셋
- 1초(100 step) 누적 시 `terminated=True, reward += -10, info['cause']='reverse'`

centerline 데이터는 트랙 yaml 옆에 `map_easy3_centerline.csv` 형식으로 사전 생성 (별도 스크립트 `scripts/extract_centerline.py`).

### Phase 2 — Encoder + Config (예상 2일)

#### 2-1. `ConvEncoder1D` 신설 (`dreamer_f1tenth/networks_1d.py`)

```
class ConvEncoder1D(nn.Module):
    def __init__(self, input_length=1080, depth=16, act='SiLU', norm=True, kernel_size=4):
        # 5 stage stride-2 Conv1d
        # ch: 1 → 16 → 32 → 64 → 128 → 256
        # len: 1080 → 540 → 270 → 135 → 67 → 33
        # outdim = 256 * 33 = 8448
```

각 stage: `Conv1d(stride=2, k=4, bias=False)` + `LayerNorm(채널)` + `SiLU`. 마지막 단에서 flatten.

`Conv1dSamePad` 별도 작성 (analysis/006 §3-3의 `Conv2dSamePad` 1D 버전).

#### 2-2. `MultiEncoder` 패치

`networks.py:293-357` 의 라우팅 로직에 `lidar` regex 매칭 분기 추가:
- 기존: cnn_keys(2D image), mlp_keys(vector)
- 추가: `lidar_keys` (1D LiDAR) → `ConvEncoder1D` 경로
- 출력 concat: `cnn_out + lidar_out + mlp_out`

`MultiDecoder`도 대칭으로 `lidar` decoder 추가 — 단, **LiDAR 재구성은 decoder loss에 비중 큼**. F1TENTH는 LiDAR 자체가 핵심 신호라 reconstruct 손실로 representation 학습. `SymlogDist` 사용 (analysis/004 §1-3).

#### 2-3. `configs_f1tenth.yaml` 작성

`configs.yaml`의 `defaults`를 베이스로 다음 override:

```yaml
f1tenth:
  # 12M model size (논문 Table 3)
  dyn_hidden: 256
  dyn_deter: 1024
  dyn_stoch: 32
  dyn_discrete: 16
  units: 256
  # encoder
  encoder: {lidar_keys: 'lidar', mlp_keys: 'state', cnn_keys: '$^', mlp_units: 256, mlp_layers: 5}
  decoder: {lidar_keys: 'lidar', mlp_keys: 'state', cnn_keys: '$^', mlp_units: 256, mlp_layers: 5}
  # actor/critic
  actor: {layers: 2, dist: 'normal', units: 256}
  critic: {layers: 2, units: 256}
  reward_head: {layers: 2, units: 256}
  cont_head: {layers: 2, units: 256}
  # task
  task: 'f1tenth_map_easy3'  # 또는 'f1tenth_oschersleben'
  time_limit: 18000
  action_repeat: 2
  steps: 5e5  # 500K each phase
  # logging
  video_pred_log: False  # image obs 없음
  # behavior
  imag_horizon: 15
  discount: 0.997
  imag_gradient: 'dynamics'
  expl_behavior: 'greedy'
  expl_until: 0
  # WM
  kl_free: 1.0
  dyn_scale: 0.5
  rep_scale: 0.1
  # snapshot
  snapshot_lap_threshold: 110.0  # seconds
  snapshot_interval_save: True   # eval_every마다 별도 파일
```

#### 2-4. `make_env` 수정 (`dreamer.py:146-203` 또는 patch 파일)

`suite == 'f1tenth'` 분기 추가:
- `from dreamer_f1tenth.envs.f1tenth_env import F110GymnasiumWrapper`
- `subtask`에서 맵명 추출 (`map_easy3`, `oschersleben`)
- wrapper 체인: `F110GymnasiumWrapper(map=...) → NormalizeActions → TimeLimit(18000) → SelectAction → UUID`

### Phase 3 — Snapshot 정책 (예상 1일)

#### 3-1. `train.py` 작성 (NM512 `dreamer.py` main loop 확장)

기존 main loop의 `latest.pt` 저장 부분을 다음으로 교체:

```
# 매 eval 후
if config.snapshot_interval_save:
    torch.save(items_to_save, snapshots_dir / f'step_{step//1000}k.pt')
if eval_lap_time and eval_lap_time < best_lap_time:
    best_lap_time = eval_lap_time
    if eval_lap_time <= config.snapshot_lap_threshold:
        torch.save(items_to_save, snapshots_dir / f'policy_lap{eval_lap_time:.0f}s_step{step//1000}k.pt')
# latest.pt는 그대로 (resume용)
torch.save(items_to_save, logdir / 'latest.pt')
```

`eval_lap_time`은 `simulate` eval 모드 metric에서 추출 — `tools.py:simulate` 의 `eval_lengths`/`eval_scores` 외에 `eval_lap_times` 추가 수집 필요.

### Phase 4 — Reward·Episode wrapper (예상 1일)

#### 4-1. Reward 함수 (`F110GymnasiumWrapper.step`)

```
reward = 0.0
# Progress (centerline arclength delta)
progress = centerline_arclength(curr_pos) - centerline_arclength(prev_pos)
reward += 1.0 * progress  # scale α=1.0, plateau 시 조정
# Lap complete
if lap_count_increased:
    reward += 50.0
# Collision
if collision:
    reward += -10.0
    terminated = True
# Reverse (1초 누적)
if reverse_counter >= 100:
    reward += -10.0
    terminated = True
```

centerline_arclength는 사전 생성된 csv lookup table 기반 nearest-neighbor + 보간.

### Phase 5 — 학습 실행 (예상 wall-clock 24시간)

#### 5-1. Stage 1 — Map Easy 500K

```bash
python dreamer_f1tenth/train.py \
    --configs f1tenth \
    --task f1tenth_map_easy3 \
    --logdir ~/logdir/f1tenth_v1_map_easy \
    --steps 5e5
```

체크포인트:
- `~/logdir/f1tenth_v1_map_easy/latest.pt`
- `~/logdir/f1tenth_v1_map_easy/snapshots/step_*.pt`

#### 5-2. Stage 2 — Oschersleben fine-tune 500K

```bash
cp ~/logdir/f1tenth_v1_map_easy/latest.pt ~/logdir/f1tenth_v1_oschersleben/latest.pt
python dreamer_f1tenth/train.py \
    --configs f1tenth \
    --task f1tenth_oschersleben \
    --logdir ~/logdir/f1tenth_v1_oschersleben \
    --steps 5e5
```

resume 메커니즘이 `latest.pt`를 로드해 fine-tune 시작.

#### 5-3. Stage 3 — Map Easy 재평가 (forgetting 검증)

stage 2 종료 후 weights로 Map Easy 평가만:
```bash
python dreamer_f1tenth/eval.py \
    --logdir ~/logdir/f1tenth_v1_oschersleben \
    --task f1tenth_map_easy3 \
    --eval_episode_num 20
```

A16 미달 시 snapshot D에서 Map Easy 성능이 좋은 시점 weights로 fallback.

### Phase 6 — 발표 자료 (예상 1일)

- state/action/reward/episode 설명 슬라이드
- curriculum 학습 곡선 (TensorBoard log → matplotlib export)
- 2-lap 주행 영상 (Map Easy + Oschersleben) 캡처
- 알고리즘 기여: "12M 모델 + 1D Conv LiDAR encoder + Progress reward + 순차 fine-tune"

---

## 4. Risks and Mitigations

| # | Risk | 영향 | Mitigation |
|---|---|---|---|
| R1 | `vel_y` 하드코딩 0이면 slip 정보 학습 신호 X | observation 정보량 감소 | A6 테스트로 사전 검증. 하드코딩이면 base_classes 패치 (Phase 1-3) |
| R2 | `ConvEncoder1D` 출력 차원 8448이 RSSM 입력으로 과도 → 학습 불안정 | WorldModel KL loss 발산 | 학습 첫 50K step 모니터링. 발산 시 마지막 단에 `AdaptiveAvgPool1d(8)` 추가해 256·8=2048로 압축 |
| R3 | Map Easy → Oschersleben fine-tune 시 catastrophic forgetting | A16 미달 → Map Easy 완주율 ↓ → 30% 평가 손해 | (a) snapshot D에서 Map Easy 우수 시점 weights 보관 → fallback. (b) fine-tune lr 절반(`3e-5 → 1.5e-5`)으로 점진 적응. (c) replay buffer에 Map Easy 데이터 일부 유지 (mixed replay) — 단 구현 비용 큼 |
| R4 | 12M 모델 + 1D Conv 8448 outdim으로 실제 파라미터 수가 12M 크게 초과 | VRAM 부족, 학습 속도 저하 | A10 테스트로 사전 검증. 초과 시 (a) `mlp_layers=3`으로 축소, (b) ConvEncoder1D 채널 8→16→32→64→128로 축소 |
| R5 | Progress reward α=1.0 스케일이 lap_complete +50과 균형 안 맞음 | actor가 progress만 누적, lap 완주 신호 무시 | RewardEMA 5/95 quantile이 자동 정규화. 발산 시 α=0.1로 축소 또는 lap_complete=+100으로 강화 |
| R6 | 후진 가드의 centerline tangent lookup이 매 step CPU 호출 → step time 증가 | 학습 wall-clock 증가 | centerline kd-tree로 nearest-neighbor O(log n) + 결과 캐싱. 1080-LiDAR forward보다 비용 작음 |
| R7 | resume 시 `_should_train._last` 등 카운터 reset되어 첫 train step burst | 학습 분포 일시 불안정 | analysis/006 §4-4 명시 위험. 카운터 state도 checkpoint에 포함하도록 `train.py`에 추가 |
| R8 | RTX 4060Ti VRAM 8GB일 가능성 (16GB 아닐 수도) | batch_size=16, length=64로 OOM | A10 직후 dry-run 학습 1K step으로 VRAM 측정. OOM 시 batch_size=8 또는 length=32로 축소 |
| R9 | `sample_episodes` seed=0 하드코딩으로 다른 seed 학습해도 episode sampling 순서 동일 | 재현성·다양성 분석 영향 | analysis/006 §2-2 명시 위험. 본 계획에선 그대로 둠. ablation 시 `tools.py:323`에 `np.random.RandomState(config.seed)` 패치 |
| R10 | F1TENTH가 image 환경이 아니라 `video_pred_log=True`면 `models.py:182` KeyError | 학습 시작 실패 | configs `video_pred_log: False` 명시 (A9 포함). 검증 dry-run으로 확인 |
| R11 | 12M 모델이 200M default 가정 hyperparameter와 안 맞아 학습 정체 | 수렴 못 함 | 논문 Table 3 매핑 외 다른 hyperparameter는 default 유지. `kl_free=1.0`, `dyn_scale=0.5/rep_scale=0.1` 등 dimension-agnostic이라 안전 |
| R12 | 학습된 정책이 sim2sim (다른 timestep, 다른 noise)에서 성능 깨짐 | 평가 환경 다르면 부정확 | 본 계획은 학습=평가 모두 Pure Gym 동일 환경. ROS 통합 안 함. 발표용 멘션만 |
| R13 | 1M step 학습이 24시간보다 오래 걸림 (env step 0.1초보다 느림) | 일정 초과 | 첫 10K step wall-clock 측정 후 추정. 초과 시 (a) `train_ratio=512 → 1024`로 학습 빈도 절반 (b) batch_size 축소 |
| R14 | F1TENTH gym의 `f110-v0` 환경이 gym 0.18 API라 gymnasium 변환 시 reset/step 시그니처 미스매치 | wrapper 작성 오류 | `F110GymnasiumWrapper`가 두 API 둘 다 처리하도록 conditional 작성. 검증은 A3 |

---

## 5. Verification Steps

### 5-1. Per-Phase 검증

| Phase | 검증 명령 | 통과 조건 |
|---|---|---|
| 1 | `pytest dreamer_f1tenth/tests/` | A1~A6, A18 통과 |
| 2 | `python -c "from dreamer_f1tenth import train; train.dry_run(steps=100)"` | A7~A10 통과, 100 step 학습 에러 없음 |
| 3 | 학습 100 step 후 `ls ~/logdir/.../snapshots/` 확인 | step_0k.pt 존재 |
| 4 | dry-run에서 reward component 로그 확인 | progress/lap_complete/collision 각각 분리 기록 |
| 5-1 | Map Easy 500K 학습 종료 후 `python eval.py --task f1tenth_map_easy3` | A11 통과 |
| 5-2 | Oschersleben fine-tune 500K 종료 | A12, A13 통과 |
| 5-3 | Map Easy 재평가 | A16 통과 (또는 fallback) |
| 6 | 발표 자료 검토 | curriculum 곡선 + 영상 캡처 + 알고리즘 contribution 명시 |

### 5-2. 전체 통합 verification

학습 종료 후 다음 모두 만족:
- [ ] **V1** `~/logdir/f1tenth_v1_oschersleben/snapshots/policy_lap*.pt` 최소 1개 존재 (LeWorldModel용)
- [ ] **V2** `~/logdir/f1tenth_v1_oschersleben/metrics.jsonl` 의 마지막 eval에서 `eval_lap_time_best <= 110.0`
- [ ] **V3** `~/logdir/f1tenth_v1_oschersleben/metrics.jsonl` 의 마지막 eval에서 `lap_completion_rate >= 0.8`
- [ ] **V4** Map Easy 재평가 `eval_lap_completion_rate >= 0.7` (forgetting 검증)
- [ ] **V5** TensorBoard에서 학습 곡선이 plateau 또는 단조 증가 (발산 없음)
- [ ] **V6** 발표 슬라이드에 state/action/reward/episode 종료조건 + 2-lap 영상 포함

---

## 6. Tunable Hyperparameters (학습 후 조정 영역)

학습 결과 plateau 또는 미달 시 다음을 ablation:

| 우선순위 | 파라미터 | 기본 | 조정 시도 범위 | 검증 비용 |
|---|---|---|---|---|
| 1 | Reward `progress α` | 1.0 | 0.1, 5.0 | 50K step retrain |
| 2 | Reward `lap_complete` | +50 | +100, +200 | 50K step retrain |
| 3 | `imag_horizon` | 15 | 10, 20 | 100K step retrain |
| 4 | `train_ratio` | 512 | 256, 1024 | 100K step retrain |
| 5 | `v_max` (차량) | 20 | 15 (안전), 25 (시도) | 100K step retrain. 단 평가 호환성 유의 |
| 6 | `discount` | 0.997 | 0.99, 0.995 | 100K step retrain |
| 7 | 1D Conv `depth` | 16 | 8 (경량), 32 (성능) | 200K step retrain |

`s_min/s_max` 조정은 비추천 (interview §5 분석 결과).

---

## 7. 출력 산출물

학습 종료 후 다음을 보유:

| 산출물 | 경로 | 용도 |
|---|---|---|
| Map Easy 학습 weights | `~/logdir/f1tenth_v1_map_easy/latest.pt` | Map Easy 평가, forgetting fallback |
| Oschersleben fine-tune weights | `~/logdir/f1tenth_v1_oschersleben/latest.pt` | 메인 평가 정책 |
| 100초대 policy snapshots | `~/logdir/.../snapshots/policy_lap*.pt` | **LeWorldModel offline RL 데이터 수집용** |
| Interval snapshots 100개 | `~/logdir/.../snapshots/step_*k.pt` | 학습 곡선 ablation, 재현 |
| 학습 곡선 jsonl | `~/logdir/.../metrics.jsonl` | 발표 그래프 |
| 2-lap 주행 영상 | `~/logdir/.../videos/*.mp4` (또는 별도 캡처) | 발표 |
| Replay buffer | `~/logdir/.../train_episodes/*.npz` | LeWorldModel offline dataset 직접 사용 가능 (단 contract 확인 필요) |

**LeWorldModel과의 데이터 contract 확인 필요** (별도 계획에서):
- LeWorldModel이 요구하는 transition tuple 형태 (s, a, r, s', is_terminal, ...)
- Dreamer .npz는 `{reward, action, lidar, state, is_first, is_terminal, is_last, discount}` 키 — 매핑 가능한지 확인

---

## 8. 일정 (Wall-Clock 추정)

| Phase | 소요 | 머신 |
|---|---|---|
| Phase 1 (env wrapper) | 2~3일 | 노트북 |
| Phase 2 (encoder + config) | 2일 | 노트북 |
| Phase 3 (snapshot 정책) | 1일 | 노트북 |
| Phase 4 (reward·episode) | 1일 | 노트북 |
| Phase 5-1 (Map Easy 500K) | 12시간 | 집컴 |
| Phase 5-2 (Oschersleben 500K) | 12시간 | 집컴 |
| Phase 5-3 (Map Easy 재평가) | 1시간 | 집컴 |
| Phase 6 (발표 자료) | 1일 | 노트북 |
| **총** | **~10일** | — |

발표 일정에 따라 Phase 5 budget 조정 (resume 가능하므로 중간 종료 OK).

---

## 9. 본 계획에서 의도적 제외

| 항목 | 사유 | 후속 계획 |
|---|---|---|
| LeWorldModel Offline RL | 별도 계획 (본 계획 종료 후) | 002 또는 후속 문서 |
| Inverse RL | 추가과제 미선택 | — |
| ROS Bridge 통합 | 대회 평가 sim only 가정 | 대회 진출 확정 시 별도 |
| 다중 환경 병렬 (`envs > 1`) | 단일 GPU 충분 | 학습 wall-clock 부족 시 도입 |
| Plan2Explore | dense reward라 불필요 | sparse reward 환경 ablation 시 |
| 차체 파라미터 ablation | 사용자 정책 (변경 금지) | — |
| LiDAR 노이즈 모델링 (sim2real) | sim only 평가 | 실차 이식 시 |

---

## 10. 다음 단계

1. **본 계획 사용자 리뷰** — 미흡한 부분, 추가 결정 사항, 일정 조정
2. **승인 후 Phase 1 시작** — `dreamer_f1tenth/envs/f1tenth_env.py` 작성
3. **Phase 1 종료 시점에 LeWorldModel data contract 사전 확인** (Phase 5 시작 전)
4. **Phase 5-1 완료 시점에 결과 보고 후 Phase 5-2 진행 결정**

---

## 11. Open Questions (잔여 결정 사항)

본 계획에서 명시하지 않은 항목 — Phase별 진입 시점에 결정:

1. **centerline 데이터** — Map Easy와 Oschersleben의 centerline csv를 직접 추출할지, F1TENTH 커뮤니티 raceline 사용할지 (Phase 4 시작 시점)
2. **eval lap_time 측정 방식** — env가 노출하는 `lap_times[0]` 키 사용 vs wrapper에서 직접 측정 (Phase 1 시작 시점)
3. **발표 시각화 도구** — TensorBoard export vs matplotlib 별도 그래프 (Phase 6 시작 시점)
4. **batch_size 8 vs 16** — VRAM 검증 후 결정 (Phase 2 dry-run 시점)
5. **fine-tune lr 조정** — R3 forgetting 발생 여부에 따라 (Phase 5-2 진입 시점)

---

> **상태**: pending approval. Phase 1 착수는 사용자 승인 후.
