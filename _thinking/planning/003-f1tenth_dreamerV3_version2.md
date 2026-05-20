# 003 — F1TENTH Dreamer-V3 베이스라인 구현 계획 (Version 2)

> **작성일**: 2026-05-20
> **모드**: Plan v2 (Critic 평가 002 대응 — REJECT → REVISE 응답)
> **선행 문서**:
> - [planning/001](./001-f1tenth_dreamerV3_version1.md) (v1 원본)
> - [planning/002](./002-critic_version1.md) (Critic, REJECT)
> - [analysis/001~006](../analysis/) (코드/환경 분석)
> - [env_setting/001~004](../env_setting/) (venv/하드웨어 상태)
> - `/home/dlacksdn/dreamerv3-torch/` (소스 직접 검증)
> **변경 정책**: v1의 §0(요구사항)·§3(Phase 분할 골격)·§7(산출물) 구조는 유지하고, Critic 002 Top 5 + 모든 CRITICAL/MAJOR를 반영해 결정·검증·리스크를 보강.

---

## 0. v1 → v2 핵심 변경 요약

| 영역 | v1 상태 | v2 변경 |
|---|---|---|
| `preprocess` image KeyError (F1-1, CRITICAL) | `video_pred_log=False`로 우회 시도 (실패) | `WorldModel` 서브클래스 + `preprocess` override 명시 (§3 결정 #14, Phase 2-0) |
| 일정/VRAM 추정 (F2-1/F2-2, CRITICAL) | 12h/12h 무근거, GPU SKU 미확정 | A19 신설 (dry-run wall-clock + VRAM 측정, sub-decision tree). GPU SKU **RTX 4060Ti 16GB 가정, 16GB 미확보 시 8GB fallback 분기**를 §0-2/§5에 명시 |
| ConvEncoder1D outdim 9728→256 bottleneck (F1-3) | outdim 8448 그대로 RSSM 입력 | 마지막에 `Linear(8448, 512)` projection. ConvDecoder1D 별도 acceptance A7b 추가 |
| Observation normalization (F5-7) | LiDAR 0~30m raw, state 스케일 다양, 정책 미명시 | LiDAR `/30.0` clip → `symlog`, state 각 dim별 정규화 표 명시 (§3 결정 #16) |
| Reward 스케일 (F3-3/F3-4/F5-5) | α=1.0 임의 | 트랙 길이 기반 산술 검증표 (§4-3). lap_complete 방향 가드 + progress step-cap |
| centerline 추출 (F5-11) | Phase 4 Open Question | **Phase 1로 격상** (§3-A 결정 #18) |
| §11 Open Questions 11건 | 결정 미룸 | §1 결정표로 격상 (#13~#26). 잔여는 v2 §10 Open Questions 4건만 |
| Acceptance | A1~A18 (A11~A16 기준 임의) | A1~A19. A11~A14는 GapFollower baseline 기반 + 측정 도구·시점 명시 |
| Risks | R1~R14 | R1~R20 (replay disk, LiDAR domain, reward hacking, joint replay 등 보강) |
| DreamerV3 정당화 | 없음 | §0-4 추가 (SKEPTIC 002 대응) |
| Critic 대응표 | — | §9 신설, 002의 F1-x/F2-x/F3-x/F4-x/F5-x 항목별 매핑 |

---

## 0-1. 프로젝트 목표 (v1과 동일)

1. Map Easy 완주 — 평가 30%
2. Oschersleben 완주 + lap time 최적화 — 평가 10%
3. 알고리즘 발표 — 평가 60%
4. 100초대 policy snapshot 저장 (LeWorldModel용)

## 0-2. 환경·하드웨어 제약 (GPU SKU 확정)

| 항목 | 값 |
|---|---|
| 학습 머신 GPU | **RTX 4060Ti 16GB 가정** (default). 환경설정 시 `nvidia-smi`로 확인 → 8GB일 경우 §5의 Fallback profile 적용 |
| 학습 머신 CPU/Disk/RAM | 12-core CPU, 2TB SSD, RAM 미확정 → 환경설정에서 `free -h`로 확인 (RAM 16GB 미만이면 dataset_size 축소) |
| 개발 머신 | 노트북, WSL2, CPU only (env wrapper, dry-run < 1K step) |
| Python / torch | 3.8, torch 2.4.1 (집컴은 cu121 빌드로 재설치 필요 — env_setting/004 §5) |
| Sim 경로 | `/home/dlacksdn/f1tenth_RL_project/` (gym + pkg 통합) |
| Dreamer 경로 | `/home/dlacksdn/dreamerv3-torch/` (NM512 클론). **vendor-in 정책**: 본 계획에서 fork 적용 |
| 제출 포맷 | Pure Gym only (ROS 통합 제외) |

## 0-3. 차량 spec — 변경 정책 (v1과 동일)

차체·액추에이터 변경 금지. 제어 한계(`s_min/s_max=±0.4189, v_min/v_max=[-5, 20]`)는 default 사용, 최종 평가도 default.

## 0-4. 왜 DreamerV3인가 (Critic 002 SKEPTIC 대응)

Critic이 정당하게 지적: F1TENTH는 dense reward + closed-track으로 SAC/PPO만으로도 풀린다. 그럼에도 본 계획이 DreamerV3를 채택하는 사유:

1. **평가 비중 60%가 알고리즘 발표**다. 학습 곡선·imagination rollout·world model latent 분석 등 발표 임팩트 자산이 SAC/PPO보다 풍부하다.
2. **추가 과제 LeWorldModel(Offline RL)**과 직접 연결된다. DreamerV3의 world model을 그대로 offline data generator로 쓸 수 있어 본 계획과 추가 과제 사이 코드/데이터 contract 비용이 작다.
3. SAC/PPO 베이스라인 비교는 발표 ablation 슬라이드 1장으로 충당 (GapFollower lap_time 측정으로 대체 가능).
4. 본 계획은 **단일 정책 학습**이 목표가 아니라 **12M 모델 검증 + 100초대 policy 다양성 확보**가 목표이므로, 동일 wall-clock에서 SAC가 더 빠를 수는 있어도 발표 자산 측면에서는 손해.

리스크는 (a) wall-clock 초과, (b) 12M 모델 hyperparameter 비표준성 — 둘 다 §1-A19, R-series에서 명시적 측정·대응.

---

## 1. 핵심 설계 결정 (Decision Table — 002 Top 5 #5 Open Question 격상 포함)

### 1-A. v1에서 결정된 항목 (#1~#13)

| # | 항목 | 결정 |
|---|---|---|
| 1 | 구현체 | `NM512/dreamerv3-torch` **fork (vendor-in)** — 본 프로젝트 `dreamerv3-torch/`를 직접 수정. submodule 아님. PR 제출 안 함 |
| 2 | 스코프 | 베이스라인 + 100초대 snapshot. LeWorldModel/Offline RL/IRL은 별도 계획 |
| 3 | 모델 크기 | **12M** — `dyn_hidden=256, dyn_deter=1024, dyn_stoch=32, dyn_discrete=16, units=256, mlp_units=256`. `cnn_depth`는 미사용이지만 default(=32) 유지하고 cnn 경로가 비활성임을 cnn_keys로 차단. Table B.1 비표준성은 R11에서 모니터링 |
| 4 | Action space | 2-dim `(steer, speed)`, `actor.dist='normal'`, `absmax=1.0`. wrapper에서 `[-1, 1]^2 → [s_min, s_max] × [v_min, v_max]` affine. `imag_gradient='dynamics'` |
| 5 | LiDAR encoder | **ConvEncoder1D** (1D Conv 5-stage stride-2) **+ Linear(flatten_dim → 512) projection** |
| 6 | 보조 obs | `state` (5-dim): `[vel_x, vel_y, ang_vel_z, prev_steer, prev_speed]`. v1의 6번째 `progress_ratio` placeholder는 **제거** (Ambiguity Risk 대응) |
| 7 | Reward | Progress + 종료신호 — §4-3 산술 검증 후 확정 스케일. lap_complete 방향 가드, progress step-cap 포함 |
| 8 | Episode 종료 | 충돌(terminal) / 2-lap 완주(last) / 180s timeout(last/truncated) / 후진 1초(terminal). 우선순위: **collision > reverse > lap_complete > timeout** (§4-4) |
| 9 | Curriculum | 순차 fine-tune Map Easy 500K → Oschersleben 500K. **joint replay 옵션**: A16 미달 위험 시 Map Easy replay 30% 유지 (R3 mitigation 강화) |
| 10 | Snapshot | Hybrid: (a) lap_time ≤ 110s 자동 저장 (best 갱신 무관, 모두 저장), (b) `eval_every=1e4`마다 interval, (c) `latest.pt` 항상 |
| 11 | Step budget | 초기 1M. Phase 2 dry-run으로 wall-clock 추정 후 조정 |
| 12 | Exploration | `expl_behavior='greedy'`. F1Tenth dense reward라 Plan2Explore 불필요 |
| 13 | 제출 | Pure Gym only. 발표에 ONNX export 가능성 멘션 |

### 1-B. v1 Open Questions → v2 결정 격상 (#14~#26)

| # | 항목 | 결정 | 근거/참조 |
|---|---|---|---|
| 14 | **preprocess image KeyError** | `models.py`를 fork 수정해 `obs["image"] = obs["image"] / 255.0` 라인을 `if "image" in obs:` 가드. 대안 (서브클래스)은 `tools.recursive_update` 흐름과 충돌해 비채택 | 002 F1-1 CRITICAL |
| 15 | LiDAR/state normalization | LiDAR: `clip(0, 30) / 30.0 → symlog` 적용 후 ConvEncoder1D 입력. State: `vel_x/20, vel_y/5, ang_vel_z/π, prev_steer/0.4189, prev_speed/20`로 [-1,1] 정규화 후 `symlog_inputs=True` MLP 통과 | 002 F5-7 |
| 16 | Encoder 출력 dim | ConvEncoder1D flatten ≈ 256×34=8704 (F1-4 ceil 보정 적용) → `Linear(8704, 512)` → state MLP 256 → concat 768 | 002 F1-3 |
| 17 | seed 정책 | 학습은 **single seed=0** (wall-clock 예산 절약). `sample_episodes`의 seed=0 하드코딩은 그대로 둠 (R9). 발표에서 "1 seed only" 한계 명시 | 002 F5-1 |
| 18 | centerline 추출 시점 | **Phase 1로 격상**. 별도 스크립트 `scripts/extract_centerline.py`로 Map Easy + Oschersleben 양쪽 csv 생성. 방법: 맵 이미지 → skeletonize (`skimage.morphology.skeletonize`) → arclength 누적 | 002 F5-11 |
| 19 | eval 프로토콜 | `eval_episode_num=20`, eval 시 pose는 학습과 동일 시작점 고정(randomization 없음), 차량 spec default 강제, noise `std_dev=0.01` 그대로 사용. Eval 시 actor `eval_state_mean=True` (이미 default `False`인데 v2에서 **True로 변경** — exploration noise off) | 002 F5-2 |
| 20 | logging stack | **TensorBoard + jsonl 동시** (NM512 default). wandb 미사용. 발표 차트는 jsonl → matplotlib | 002 F5-3 |
| 21 | checkpoint optimizer carry-over | fine-tune(`Stage 2`) 시 `latest.pt` 복사 → world model weights는 warm, actor·critic·model_opt는 **fresh** (`train.py`에 fresh-optim flag 추가). lr 절반 R3 mitigation과 정합 | 002 F5-4 |
| 22 | action wrapper 체인 | `F110GymnasiumWrapper → NormalizeActions([-1,1]) → TimeLimit(18000 sim step = 9000 env step @ action_repeat=2) → SelectAction("action") → UUID → Damy(또는 Parallel=False)` | 002 F1-11 |
| 23 | prefill 정책 | random policy prefill **`prefill=0`으로 비활성**. 대신 첫 10K env step을 **GapFollower 정책으로 수집**(별도 collector). 그 후 정상 Dreamer 학습 시작 | 002 F5-9 |
| 24 | termination 우선순위 | 한 step에 다중 발생 시: `collision → reverse → lap_complete → timeout`. reward 합산은 동일 step 내 모두 더함, 단 lap_complete reward는 방향 dot > 0 일 때만 | 002 F5-8 |
| 25 | Map 명명 | 평가 트랙 = `map_easy` (env_setting/001 §4 기준 .png/.yaml 파일명). v1의 `map_easy3`는 analysis/001에 등장하나 본 계획에서는 **`map_easy`로 통일**. `task='f1tenth_map_easy'`로 명명 | 002 F5-10 |
| 26 | torch.compile | `compile=False`로 명시 (configs override). ConvEncoder1D + 1D padding 동적성으로 TorchInductor fallback 위험 | 002 F5-15 |

### 1-C. Top 5 Critic 우선순위 직접 매핑

| Critic Top 5 | v2 반영 위치 |
|---|---|
| #1 preprocess KeyError fix | 결정 #14, Phase 2-0 acceptance A20 |
| #2 wall-clock + VRAM dry-run | Acceptance A19, §5 Phase 2 dry-run gate |
| #3 ConvEncoder outdim + obs normalization + ConvDecoder1D | 결정 #15, #16, Acceptance A7/A7b |
| #4 Reward 재설계 (lap 방향 가드, progress cap, 트랙 길이 산술) | §4-3 표, 결정 #7, R17 |
| #5 §11 Open Questions 격상 | 결정 #17~#26 |

---

## 2. Acceptance Criteria (Testable, 측정도구·기준 출처·시점 명시)

> 표기: 〔도구〕는 검증 도구, 〔Phase〕는 측정 시점, 〔출처〕는 기준값 산출 근거.

### 2-1. 환경·통합 (Phase 1 완료 기준)

- [ ] **A1** `f110_gym:f110-v0` → `F110GymnasiumWrapper`로 `reset(seed) → (obs, info)`, `step(action) → (obs, r, terminated, truncated, info)` 5-tuple. 〔도구〕`pytest tests/test_wrapper_api.py`. 〔Phase 1-2 종료〕
- [ ] **A2** obs dict: `{'lidar': (1080,) float32, 'state': (5,) float32, 'is_first': bool, 'is_terminal': bool, 'is_last': bool}`. 〔도구〕`pytest`. 〔Phase 1-2 종료〕
- [ ] **A3** smoke test: 1 episode 끝까지 무에러 실행. 〔도구〕`python -m dreamer_f1tenth.smoke`. 〔Phase 1-2 종료〕
- [ ] **A4** 충돌 시 `terminated=True ∧ info['cause']=='collision'`. 〔도구〕`pytest tests/test_termination.py::test_collision`. 〔Phase 1-2〕
- [ ] **A5** 18000 simulator step (9000 env step @ action_repeat=2) 도달 시 `truncated=True ∧ terminated=False`. 〔도구〕`pytest::test_timeout`. 〔Phase 1-2〕
- [ ] **A6** base_classes.py 패치 후, 직선 가속 5초 + 코너링 2초 시나리오에서 `|vel_y|.max() > 0.05`. 〔도구〕`pytest tests/test_obs_vels.py`. 〔출처〕dynamic single-track 모드에서 slip_angle≠0 검증. 〔Phase 1-3 종료, 패치 후〕
- [ ] **A18** 강제 후진 액션 1.1초 가하면 `terminated=True ∧ info['cause']=='reverse'`. 〔도구〕`pytest tests/test_reverse_guard.py`. 〔Phase 1-4〕
- [ ] **A_centerline** Map Easy + Oschersleben centerline csv 생성됨, 각 점이 `(arclength, x, y, tangent_x, tangent_y)` 5-col. 〔도구〕`python scripts/extract_centerline.py --verify`. 〔Phase 1-1〕

### 2-2. 모델·Encoder·Critic dry-run (Phase 2 완료 기준)

- [ ] **A7** `ConvEncoder1D(1080) → (B,T,512)` (5-stage Conv1d + Linear projection). shape 검증. 〔도구〕`pytest tests/test_encoder.py`. 〔Phase 2-1〕
  - 5-stage Conv1d 출력 길이: 1080→540→270→135→**68→34** (ceil 보정, F1-4 반영). Flatten 256×34=8704 → Linear(8704, 512)
- [ ] **A7b** `ConvDecoder1D(512+latent) → (B,T,1080)` reconstruction. SymlogDist loss. 〔도구〕`pytest tests/test_decoder.py`. 〔Phase 2-1〕
- [ ] **A8** `MultiEncoder` 패치에 `lidar_keys` 분기 추가, `len(shape)==1 ∧ lidar regex` → 1D Conv 경로. configs `lidar_keys='lidar', mlp_keys='state', cnn_keys='$^'`. 〔도구〕dry-run print "Encoder LIDAR/MLP shapes". 〔Phase 2-2〕
- [ ] **A9** `configs_f1tenth.yaml` 작성 — 12M hyperparam + `compile: False, video_pred_log: False, prefill: 0, eval_state_mean: True`. 〔도구〕`python -c "from train import load_config; print(load_config())"`. 〔Phase 2-3〕
- [ ] **A10** `sum(p.numel() for p in agent.parameters() if p.requires_grad) ∈ [10M, 14M]` **AND** RSSM/encoder/decoder/heads 비율 보고 (RSSM ~30%, encoder+decoder ~50%, heads ~20% 기대). 〔도구〕`python scripts/param_audit.py`. 〔Phase 2-3〕
- [ ] **A20** `WorldModel.preprocess` image KeyError 해결 검증: vector-only obs로 100 train step + 10 eval step 무에러. 〔도구〕`pytest tests/test_preprocess_patch.py`. 〔Phase 2-0, 결정 #14 검증〕
- [ ] **A19** **dry-run wall-clock + VRAM gate** (Critic Top 5 #2 대응):
  - 1K env step + 100 train step 측정
  - 측정값 A: env_step_avg_ms, B: train_step_avg_ms, C: peak VRAM MB, D: 추정 500K wall-clock = `(1000*A + 500K/train_ratio*B) / 60` 분
  - **Pass 조건**:
    - C ≤ `(GPU_total_VRAM × 0.80)` MB (16GB → 12800MB, 8GB → 6400MB)
    - D ≤ 24h (= 1440 min)
  - **Fail 분기** (사전 결정):
    - VRAM 초과 → batch_size 16→8 → 재측정. 여전히 초과 시 batch_length 64→32
    - wall-clock 초과 → train_ratio 512→1024 → 재측정. 여전히 초과 시 steps 500K→300K + 발표에서 한계 명시
    - GPU 8GB SKU 확정 시 시작부터 precision=16 (AMP) + batch_size=8 적용
  - 〔도구〕`python scripts/dryrun_bench.py`. 〔Phase 2-4, **Stage 1 학습 진입 게이트**〕

### 2-3. 학습·평가 (Phase 3 완료 기준 — 기준값 출처 명시)

- [ ] **A11** Map Easy 500K 학습 후 eval (20 ep) **median lap_time ≤ `GapFollower_baseline × 1.5`초**. GapFollower baseline은 Phase 1-1 dry-run에서 실측 (예상 30~40초 → 기준 45~60초). 〔도구〕`python -m dreamer_f1tenth.eval --task f1tenth_map_easy`. 〔Phase 5-1 종료〕
- [ ] **A12** Oschersleben fine-tune 500K 후 eval 20 ep에서 **2-lap 완주율(`lap_count[0] >= 2`) ≥ 80%**. 〔도구〕동일. 〔Phase 5-2 종료〕
- [ ] **A13** Oschersleben fine-tune 후 eval **median lap_time ≤ 120초 AND best lap_time ≤ 110초** (이중 기준, F4-5 대응). 〔도구〕동일. 〔Phase 5-2〕
- [ ] **A14** lap_time ≤ 110s 정책은 **best 갱신 무관, 모두 저장** (LeWorldModel용 다양성, F4-6 대응). 파일명 `policy_lap{X:.1f}s_step{Y}k.pt`. 〔도구〕`ls logdir/.../snapshots/`. 〔Phase 5-2 중 trigger 시점〕
- [ ] **A15** `eval_every=1e4`마다 interval snapshot `step_{N}k.pt` 생성, 1M step 총 100개. 디스크 ~20GB(논문 비교 시 §7 16GB→20GB로 수정). 〔도구〕`du -sh snapshots/`. 〔Phase 5 진행 중〕
- [ ] **A16** Map Easy 재평가에서 2-lap 완주율 ≥ 70% (forgetting 한도). 미달 시 joint replay (결정 #9) 또는 snapshot D fallback. 〔도구〕동일 eval 스크립트. 〔Phase 5-3〕

### 2-4. Reward·safety (재현성)

- [ ] **A17** Reward component 분리 로깅: `reward/progress`, `reward/collision`, `reward/lap_complete`, `reward/reverse`. 〔도구〕TensorBoard scalar tags 확인. 〔Phase 4 작성 직후 dry-run〕

---

## 3. Implementation Steps

### Phase 1 — Env wrapper + centerline (예상 3~4일, 노트북 CPU)

#### 1-0. 사전 의존성 (Critic F2-3 대응)
노트북 venv에 `gymnasium`, `pyyaml`, `scikit-image`, `scipy`, `tensorboard` 추가 설치. dreamerv3-torch `requirements.txt`와 합집합. CPU only torch 2.4.1 유지.

#### 1-1. centerline 추출 (Phase 1 격상, 002 F5-11)
`scripts/extract_centerline.py`:
- 입력: `maps/{map_name}.png`, `maps/{map_name}.yaml` (resolution, origin)
- skimage skeletonize → 폐곡선 정렬 (start point는 yaml origin 부근 nearest) → arclength 누적 + tangent 계산
- 출력: `maps/{map_name}_centerline.csv` 5-col `(s, x, y, tx, ty)`
- 검증: Oschersleben centerline 총 길이가 트랙 실제 길이(약 3700m 미스케일링, 본 sim 좌표계에서는 §4-3에서 측정) 근방인지 확인

#### 1-2. `F110GymnasiumWrapper` (`dreamer_f1tenth/envs/f1tenth_env.py`)
- gym 0.18 4-tuple → gymnasium 5-tuple 변환
- `reset(seed, options)` → `poses` 결정: 트랙별 default start (env_setting/001 §9)
- obs dict 구성 (결정 #6)
- reward 계산 (§4)
- action wrapper 체인 (결정 #22)

#### 1-3. base_classes.py 패치 (linear_vels_y, F1-8)
- 현 코드 `observations['linear_vels_y'].append(0.)` (L488) → `vel_y = state[3] * sin(state[6])` (slip_angle 기반)
- dynamic_models.py가 single-track dynamic 모드인지 확인 (kinematic이면 slip_angle=0으로 vel_y=0 — A6 실패 가능)

#### 1-4. reverse_guard (centerline tangent dot velocity)
- 결정 #24 우선순위 적용
- 카운터 reset: dot ≥ 0이면 0으로 reset. F4-8 우려: 완전 정지 시 dot ≈ 0 → reset 발생, 이는 의도된 동작 (정지는 후진 아님)

### Phase 2 — Encoder/Decoder + Config + 패치 + dry-run gate (예상 3일)

#### 2-0. dreamerv3-torch fork 적용 (결정 #1, 002 F1-6)
**변경 파일 list**:
- `dreamerv3-torch/models.py:182` → `if "image" in obs: obs["image"] = obs["image"] / 255.0`
- `dreamerv3-torch/networks.py` MultiEncoder/MultiDecoder에 `lidar_keys` 분기 (L297-357 인근 + decoder 대칭) — 변경 라인 약 6곳
- `dreamerv3-torch/dreamer.py:make_env` (L146-203)에 `suite=='f1tenth'` 분기
- `dreamerv3-torch/configs.yaml`에 `f1tenth:` block append
- 패치 사본을 `_thinking/patches/` 아래에 diff로도 보관 (재현용)

#### 2-1. `ConvEncoder1D` / `ConvDecoder1D` (`dreamer_f1tenth/networks_1d.py`)
- 5-stage stride-2 Conv1d, ch 1→16→32→64→128→256
- 출력 길이 ceil 보정: 1080→540→270→135→68→34. flatten = 8704
- `Linear(8704, 512)` projection (결정 #16)
- Decoder는 거울상 ConvTranspose1d + 최종 SymlogDist

#### 2-2. MultiEncoder/MultiDecoder 패치
- `lidar_keys` 정규식 매칭, `len(shape)==1 ∧ lidar` → 1D path
- forward concat 순서: `[lidar_out, mlp_out]`
- outdim = 512 + 256 = 768

#### 2-3. `configs_f1tenth.yaml` (결정 #3, #14, #19, #20, #23, #26 반영)
```yaml
f1tenth:
  task: 'f1tenth_map_easy'
  time_limit: 9000           # env step 기준 (action_repeat=2, sim 18000 step)
  action_repeat: 2
  steps: 5e5
  prefill: 0                 # GapFollower 별도 collector (결정 #23)
  envs: 1
  parallel: False
  precision: 32              # GPU 8GB일 경우 16 (AMP)
  compile: False             # 결정 #26
  video_pred_log: False
  eval_episode_num: 20
  eval_state_mean: True      # 결정 #19
  # 12M model
  dyn_hidden: 256
  dyn_deter: 1024
  dyn_stoch: 32
  dyn_discrete: 16
  units: 256
  encoder: {lidar_keys: 'lidar', mlp_keys: 'state', cnn_keys: '$^',
            mlp_units: 256, mlp_layers: 5, symlog_inputs: True}
  decoder: {lidar_keys: 'lidar', mlp_keys: 'state', cnn_keys: '$^',
            mlp_units: 256, mlp_layers: 5, vector_dist: symlog_mse}
  actor: {layers: 2, dist: 'normal', units: 256}
  critic: {layers: 2, units: 256}
  reward_head: {layers: 2, units: 256}
  cont_head: {layers: 2, units: 256}
  # snapshot
  snapshot_lap_threshold: 110.0
  snapshot_save_all_below_threshold: True   # 결정 #14 (A14, F4-6)
  snapshot_interval_save: True
  # dataset
  dataset_size: 200000        # F5-14 — 1M → 200K로 축소 (LiDAR float16 저장 시 ~3GB RAM)
  replay_lidar_fp16: True
```

#### 2-4. dry-run benchmark (A19 gate)
- `scripts/dryrun_bench.py` 실행 → 측정값으로 A19 분기 결정
- **이 게이트를 통과해야 Phase 5 진입**

### Phase 3 — Snapshot 정책 + train.py wrapper (1일)

- `train.py`에서 NM512 `dreamer.py:main` 흐름을 monkey-patch (결정 #14의 fork와 별개)
- snapshot 3종 (lap-threshold 모두, interval, latest)
- counter state checkpoint 저장 (R7): `_should_train._last` 등 5개 counter도 ckpt에 포함

### Phase 4 — Reward + Episode (1일, Phase 1 직후로 앞당김 — Critic Top 5 #4)

#### 4-1. Reward 함수 (§4-3 스케일 적용)
```python
reward = 0.0
# progress = clip(centerline_arclength(curr) - centerline_arclength(prev), 0, 0.5)
progress = clip(arclen_delta, 0.0, 0.5)        # F3-4 step-cap
reward += alpha_progress * progress
if collision:
    reward += -10.0; terminated = True; cause = 'collision'
elif reverse_counter >= 100:
    reward += -10.0; terminated = True; cause = 'reverse'
elif lap_count_increased and velocity_dot_tangent > 0:
    reward += R_lap_complete                    # 방향 가드 (F3-3)
    if lap_count >= 2: is_last = True
```

#### 4-2. Joint replay 옵션 (결정 #9, R3)
fine-tune 시 `--joint_replay_ratio 0.3` 플래그로 Map Easy episode를 buffer 30% 비율 유지.

### Phase 5 — 학습 실행 (집컴 GPU, 24h 예산)

Stage 1 (Map Easy 500K), Stage 2 (Oschersleben 500K, latest.pt warm load + fresh optim), Stage 3 (Map Easy 재평가, A16). 명령은 v1 §5와 동일하되 `--task f1tenth_map_easy`, `--joint_replay_ratio 0.3` 추가.

#### 5-Fallback (A19 분기에서 결정된 profile)
| Trigger | Profile |
|---|---|
| GPU 8GB | precision=16, batch_size=8, batch_length=64 유지 |
| wall-clock D > 24h | train_ratio 512→1024, steps 500K→300K |
| 둘 다 fail | precision=16 + train_ratio=1024 + steps=300K |

### Phase 6 — 발표 자료 (1일)

발표 contribution 요지 (Critic STAKEHOLDER 대응): "12M DreamerV3 + 1D LiDAR encoder + progress reward + 순차 fine-tune + LeWorldModel 데이터 generator로의 재활용". GapFollower vs DreamerV3 비교 슬라이드 1장 포함.

---

## 4. Reward·Episode 세부

### 4-1. Reward 컴포넌트 (결정 #7)
- progress (clip + α 보정)
- collision = -10 (terminal)
- reverse = -10 (terminal, 1초 누적)
- lap_complete = R_lap (방향 가드)

### 4-2. Episode 종료 (결정 #8, #24)
우선순위: collision > reverse > lap_complete (lap_count≥2) > timeout.

### 4-3. **Reward 스케일 산술 검증** (Critic Top 5 #4, F5-5)

> 트랙 길이는 Phase 1-1 centerline 추출 후 정확 측정. 아래는 v2 작성 시점 예상값 (analysis/001 §4, resolution 기반 추정).

| 트랙 | 트랙 길이 추정 (centerline arclength) | 평균 step 당 progress (예상 lap_time × 100Hz × action_repeat=2 → env step) | α=1.0 시 lap당 progress reward 총합 | lap_complete R 권장값 (progress 총합의 30~50%) | 적용 결정 |
|---|---|---|---|---|---|
| Map Easy (resolution 0.02) | **약 50~80m** (Phase 1-1에서 정확 측정) | 50~80m / (~50s × 50step/s) = 0.02~0.032 m/step | 50~80 | **R_lap = 30** | α=1.0, R_lap=30 |
| Oschersleben (resolution 0.04295) | **약 300~400m** (Phase 1-1 측정 필요) | 300~400m / (~110s × 50step/s) = 0.05~0.07 m/step | 300~400 | **R_lap = 100~150** | α=1.0, R_lap=100 |

→ **결정**: `alpha_progress = 1.0`, `R_lap_complete = 100` 통일. Phase 1-1 centerline 측정 후 트랙 길이가 위 추정에서 ±50% 이상 벗어나면 본 표 재산정하고 v2-patch 문서로 기록.

step-cap `clip(progress, 0, 0.5)`은 0.5m/step = lap당 최대 reward 한도 (50step/s × 0.5m = 25m/s × 2step physics) — 정상 주행 progress (≤0.07m/step)는 cap에 안 닿고, shortcut/오류 progress만 차단됨.

### 4-4. Termination 우선순위 적용 코드 골격
```python
terminated = False; cause = None; r_extra = 0.0
if collision:
    terminated, cause, r_extra = True, 'collision', -10.0
elif reverse_counter >= 100:
    terminated, cause, r_extra = True, 'reverse', -10.0
elif lap_count_increased_and_forward:
    r_extra += R_lap
    if lap_count >= 2: is_last = True; cause = 'lap_complete'
if not terminated and env_step >= 9000:
    truncated = True; cause = cause or 'timeout'
reward = progress * alpha + r_extra
```

---

## 5. Risks and Mitigations (R1~R20)

| # | Risk | 영향 | Mitigation |
|---|---|---|---|
| R1 | vel_y 하드코딩 0 (kinematic 모드 가능성) | obs 정보량↓ | A6 + dynamic 모드 확인 (Critic F1-8) |
| R2 | ConvEncoder1D 출력 → RSSM bottleneck | KL 발산 | Linear(8704, 512) projection (결정 #16), 50K step KL 모니터 |
| R3 | catastrophic forgetting | A16 미달 | joint replay 30% (결정 #9), fresh optim + lr 절반, snapshot fallback |
| R4 | 12M 실제 param 초과 | VRAM | A10 비율 체크 + A19 dry-run |
| R5 | Reward 스케일 불균형 | reward hacking | §4-3 산술 검증 + RewardEMA + lap 방향 가드 |
| R6 | centerline lookup CPU cost | step time↑ | scipy kd-tree O(log n) + cache |
| R7 | resume 시 counter reset | burst | counter state ckpt 저장 (Phase 3) |
| R8 | GPU 8GB SKU | OOM | precision=16 + batch_size=8 fallback (A19) |
| R9 | sample_episodes seed=0 | 재현성 | 발표에 "1 seed" 한계 명시 (결정 #17) |
| R10 | preprocess image KeyError | 즉시 실패 | 결정 #14 + A20 (Critic F1-1) |
| R11 | 12M hyperparam 비표준 | 수렴 정체 | A10 비율 모니터 + 첫 100K step KL/loss 추적 |
| R12 | sim2sim drift | sim only 가정으로 무시 | 발표 멘션 |
| R13 | 1M step > 24h | 일정 초과 | A19 사전 분기 |
| R14 | gym 0.18 → gymnasium 변환 비용 | wrapper 작업↑ | poses randomization 미사용 (결정 #19 고정 pose) — wrapper 단순화 |
| **R15** | **Replay 디스크 누적** (002 F3-1) | 디스크 부족 | dataset_size=200K + LiDAR fp16 저장 (결정 #15, F5-14) |
| **R16** | **LiDAR domain randomization 부재** (002 F3-2) | sim2sim 갭 | sim only 평가로 본 계획 제외, Phase 5 이후 ablation |
| **R17** | **Lap reward hacking** (002 F3-3) | 후진 통과 시도 | lap_complete 방향 가드 (§4-1) + reverse penalty 우선순위 |
| **R18** | **Progress shortcut** (002 F3-4) | 비현실적 progress | step-cap `clip(0, 0.5m)` (§4-1) |
| **R19** | **Optimizer carry-over stale momentum** (002 F5-4) | fine-tune 불안정 | fresh optim (결정 #21) |
| **R20** | **torch.compile + 1D Conv 호환성** (002 F5-15) | compile 실패 | `compile: False` (결정 #26) |

---

## 6. Verification Steps

### 6-1. Per-Phase 검증

| Phase | 검증 명령 | 통과 조건 |
|---|---|---|
| 1-1 | `python scripts/extract_centerline.py --verify` | A_centerline, 트랙 길이 표 (§4-3) 갱신 |
| 1-2~1-4 | `pytest dreamer_f1tenth/tests/` | A1~A6, A18 |
| 2-0 | `pytest tests/test_preprocess_patch.py` | A20 |
| 2-1~2-3 | `python scripts/param_audit.py` + dry-run | A7, A7b, A8, A9, A10 |
| 2-4 | `python scripts/dryrun_bench.py` | **A19 게이트** |
| 3 | 학습 200 step → ckpt 저장 확인 | snapshot 3종 생성 |
| 4 | reward log 확인 | A17 컴포넌트 분리 |
| 5-1 | Map Easy 500K | A11 |
| 5-2 | Oschersleben 500K | A12, A13, A14 |
| 5-3 | Map Easy 재평가 | A16 (또는 fallback) |
| 6 | 발표 자료 | 산출물 §7 모두 포함 |

### 6-2. 전체 통합 verification (V1~V6은 v1과 동일, V7 추가)

- V7 **A19 dry-run 측정 결과가 `_thinking/notes/dryrun_results.md`로 기록**되어 재현 가능

---

## 7. 산출물 (v1 §7과 동일하되 디스크 수치 정정)

| 산출물 | 경로 | 비고 |
|---|---|---|
| Stage 1 weights | `~/logdir/f1tenth_v2_map_easy/latest.pt` | |
| Stage 2 weights | `~/logdir/f1tenth_v2_oschersleben/latest.pt` | 메인 |
| 100초대 snapshots | `~/logdir/.../snapshots/policy_lap*.pt` | LeWorldModel용 다양성 |
| Interval snapshots | `~/logdir/.../snapshots/step_*k.pt` | 100개, ~20GB (v1 16GB 정정) |
| metrics.jsonl + TB | `~/logdir/.../` | |
| centerline csv | `maps/*_centerline.csv` | |
| dry-run 결과 | `_thinking/notes/dryrun_results.md` | A19 |
| patch diff | `_thinking/patches/` | 재현성 |

---

## 8. 일정 (v1 §8 보강)

| Phase | 소요 | 머신 |
|---|---|---|
| 1 (env + centerline) | 3~4일 | 노트북 |
| 2 (encoder + config + fork + dry-run) | 3일 | 노트북 + 집컴(2-4만) |
| 3 (snapshot/train.py) | 1일 | 노트북 |
| 4 (reward) | 1일 | 노트북 |
| 5-1 | 12h ± A19 분기 | 집컴 |
| 5-2 | 12h ± A19 분기 | 집컴 |
| 5-3 | 1.5h (eval 20 ep × ~3min, F2-4 보정) | 집컴 |
| 6 (발표) | 1일 | 노트북 |
| **총** | **~11~13일** | |

---

## 9. Critic 002 대응표 (항목별 매핑)

표기: **해결** = v2에서 명시 fix, **Phase 결정** = Phase 진입 시점에 결정/측정, **미반영** = 의도적 제외 + 사유.

### 9-1. CRITICAL

| Critic # | 항목 | v2 대응 |
|---|---|---|
| F1-1 | preprocess image KeyError | **해결** — 결정 #14 (models.py fork), Acceptance A20, R10 |
| F2-1 | Phase 5 wall-clock 무근거 | **해결** — A19 dry-run gate, §5 Fallback profile |
| F2-2 | GPU SKU 미확정 + VRAM 추정 부재 | **해결** — §0-2에서 16GB 가정 + 8GB fallback 분기 명시, A19에서 측정 |

### 9-2. MAJOR — F1 시리즈 (기술 정합성)

| Critic # | v2 대응 |
|---|---|
| F1-2 (12M Table B.1 비표준) | **Phase 결정** — A10 비율 보고 + R11 모니터링. NM512 README 12M preset 직접 검증은 Phase 2-3 |
| F1-3 (outdim 8448 bottleneck) | **해결** — 결정 #16, Linear(8704, 512) projection |
| F1-4 (stride-2 산술 오류) | **해결** — A7에 ceil 보정 (1080→540→270→135→68→34) |
| F1-5 (ConvDecoder1D 별도 작업) | **해결** — A7b 별도 acceptance, Phase 2-1 일정 포함 |
| F1-6 (MultiEncoder patch 작업량) | **해결** — Phase 2-0 변경 파일 list 명시, fork 정책 vendor-in |
| F1-7 (gym 0.18 → gymnasium 변환) | **해결** — 결정 #14, #19(고정 pose), R14, Phase 1-2 |
| F1-8 (vel_y 하드코딩 + slip_angle) | **해결** — Phase 1-3 + A6 단언 (dynamic 모드 확인) |
| F1-9 (action_repeat 100Hz 의미) | **해결** — 결정 #22, configs `time_limit=9000` 주석 |

### 9-3. MINOR — F1 시리즈

| F1-10, F1-11 | **해결** — 결정 #1 vendor-in, 결정 #22 wrapper 체인에 Damy 명시 |

### 9-4. F2 시리즈 (일정 현실성)

| Critic # | v2 대응 |
|---|---|
| F2-3 (노트북 Phase 1~4 가능성) | **해결** — Phase 1-0 의존성 사전 설치 명시 |
| F2-4 (재평가 1시간 너무 짧음) | **해결** — §8에서 1.5h로 보정, eval_episode_num=20 명시 |

### 9-5. F3 시리즈 (Risk 충분성)

| Critic # | v2 대응 |
|---|---|
| F3-1 (replay 디스크) | **해결** — R15, dataset_size=200K + LiDAR fp16 |
| F3-2 (LiDAR domain randomization) | **미반영** — 사유: sim only 평가 가정. R16 ablation 후속 |
| F3-3 (lap reward hacking 방향) | **해결** — §4-1 방향 가드, R17 |
| F3-4 (progress shortcut) | **해결** — step-cap, R18 |
| F3-5 (sample_episodes seed=0) | **미반영(부분)** — 결정 #17, 발표에 한계 명시. R9 |
| F3-6 (forgetting mitigation 약함) | **해결** — joint replay 30% (결정 #9), R3 강화 |
| F3-7 (counter ckpt) | **해결** — Phase 3, R7 구체 fix |
| F3-8 (R8 fallback 늦음) | **해결** — A19 dry-run 게이트로 Phase 5 진입 전 결정 |

### 9-6. F4 시리즈 (Acceptance 검증 가능성)

| Critic # | v2 대응 |
|---|---|
| F4-1 (A6 시점) | **해결** — A6에 "Phase 1-3 패치 후" 명시 |
| F4-2 (A10 ±20% 약함) | **해결** — A10에 비율 보고 추가 |
| F4-3 (A11 60초 근거 없음) | **해결** — A11 기준 = GapFollower baseline × 1.5 (Phase 1-1 측정) |
| F4-4 (2-lap 정의 모호) | **해결** — 결정 #8: 2-lap 도달이 정식 완주. lap_complete reward는 매 lap 발생, episode 종료는 lap_count≥2 |
| F4-5 (A13 best 1회 통과 가능) | **해결** — A13 이중 기준 (median ≤120 AND best ≤110) |
| F4-6 (A14 다양성 부족) | **해결** — `snapshot_save_all_below_threshold=True` (결정 #10, A14) |
| F4-7 (snapshot 디스크 16GB→20GB) | **해결** — §7 정정 |
| F4-8 (후진 카운터 reset 의미) | **해결** — Phase 1-4 의도된 동작 명시 |
| F4-9 (info['cause'] 일관성) | **해결** — §4-4에 cause 4종 ('collision', 'reverse', 'lap_complete', 'timeout') 명시 |

### 9-7. F5 시리즈 (누락 항목)

| Critic # | v2 대응 |
|---|---|
| F5-1 (seed 정책) | **해결** — 결정 #17 |
| F5-2 (eval 프로토콜) | **해결** — 결정 #19 |
| F5-3 (logging stack) | **해결** — 결정 #20 |
| F5-4 (ckpt optimizer carry-over) | **해결** — 결정 #21 |
| F5-5 (reward shaping 산술) | **해결** — §4-3 표 |
| F5-6 (action normalize) | **해결** — 결정 #22 wrapper 체인 |
| F5-7 (obs normalization) | **해결** — 결정 #15 |
| F5-8 (termination 우선순위) | **해결** — 결정 #24, §4-4 |
| F5-9 (prefill 정책) | **해결** — 결정 #23 (prefill=0 + GapFollower collector) |
| F5-10 (map_easy 명명) | **해결** — 결정 #25 (`map_easy` 통일) |
| F5-11 (centerline 시점) | **해결** — Phase 1-1로 격상 (결정 #18) |
| F5-12 (lap_times 시작값) | **Phase 결정** — Phase 1-2 wrapper에서 실측. 미흡 시 wrapper 자체 측정 |
| F5-13 (rollback 시나리오) | **해결** — A16 미달 시 fallback = `policy_lap*.pt` 중 Map Easy 호환 weights, joint replay 재시작 |
| F5-14 (dataset_size RAM) | **해결** — 결정 (configs `dataset_size=200000`) |
| F5-15 (torch.compile 호환성) | **해결** — 결정 #26 (`compile: False`) |

### 9-8. Ambiguity Risks & Multi-Perspective

| 항목 | v2 대응 |
|---|---|
| obs 6번째 placeholder | **해결** — state 5-dim으로 축소 (결정 #6) |
| Hybrid "D" 정의 | **해결** — 결정 #10 명확화 |
| latest.pt 옵티마이저 stale | **해결** — 결정 #21 (fresh optim) |
| EXECUTOR (작업량 명시) | **해결** — Phase 2-0 파일 list |
| STAKEHOLDER (알고리즘 contribution 약함) | **부분 반영** — §0-4 정당화 + §6 비교 슬라이드. 진정한 contribution은 LeWorldModel 후속 계획에서 |
| SKEPTIC (DreamerV3 vs SAC) | **해결** — §0-4 |

---

## 10. v2 Open Questions (잔여 4건)

본 v2에서도 결정 못 하고 Phase 진입 시점 측정·결정 필요한 항목:

1. **트랙 실제 centerline 길이** — Phase 1-1 측정 후 §4-3 표 갱신. 예상 ±50% 초과 시 reward 스케일 재산정 (v2-patch 문서로).
2. **GPU SKU 확정 (4060Ti 8GB vs 16GB)** — 환경설정 직후 `nvidia-smi`로 확정. A19 dry-run 시작 전.
3. **GapFollower baseline lap_time** — Phase 1-1 dry-run에서 Map Easy + Oschersleben 각 5 ep 실측. A11/A13 기준 갱신.
4. **dynamic_models.py 모드(kinematic vs dynamic single-track) 확인** — A6 검증 가능성 결정. kinematic 모드면 vel_y 패치 자체가 무의미 → R1 mitigation 변경 필요.

---

## 11. 다음 단계

1. 본 v2 사용자 리뷰 — 결정표 #14~#26, A19 게이트, §4-3 스케일, §9 대응표 검토
2. (필요 시) Architect/Critic v2 재평가
3. 승인 시 Phase 1-0 (의존성 설치) → Phase 1-1 (centerline + GapFollower baseline 측정) 착수
4. Phase 1-1 결과를 §4-3 표·A11·A13 기준에 반영 (v2-patch 문서)
5. Phase 2-4 A19 게이트 통과 후 집컴 학습 Stage 1 개시

---

> **상태**: pending approval. v1 대비 결정 13건 추가, Acceptance 1건 추가 (A19) + A7b/A20 신설, Risk 6건 추가 (R15~R20), Critic 002 항목별 대응표 §9 신설.
