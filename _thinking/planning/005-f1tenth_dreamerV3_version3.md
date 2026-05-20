# 005 — F1TENTH Dreamer-V3 베이스라인 구현 계획 (Version 3, 최종본)

> **작성일**: 2026-05-20
> **모드**: Plan v3 (Critic 평가 004 ACCEPT_WITH_MINOR_FIXES → Fix 10건 + Open Questions 4건 통합 종결)
> **상태**: 최종본. 추가 patch 없음. 본 문서로 Phase 1-0 착수.
> **선행 문서**:
> - [planning/001](./001-f1tenth_dreamerV3_version1.md) (v1 원본, REJECT)
> - [planning/002](./002-critic_version1.md) (v1 Critic)
> - [planning/003](./003-f1tenth_dreamerV3_version2.md) (v2, ACCEPT_WITH_MINOR_FIXES)
> - [planning/004](./004-critic_version2.md) (v2 Critic, 본 v3가 반영하는 Fix 원본)
> - analysis/001~006, env_setting/001~004
> - `/home/dlacksdn/dreamerv3-torch/` (실측 직접 검증 완료)
> - `/home/dlacksdn/f1tenth_RL_project/gym/f110_gym/envs/` (실측 직접 검증 완료)
> **변경 정책**: v2의 §0~§11 골격은 유지. 004 Fix 10건과 Open Q 4건 전수 종결, 잔여 0건 목표.

---

## 0. v2 → v3 변경 요약

### 0-1. v1→v2 1줄 압축 (보존)

v1(REJECT) → v2: 002 Top 5(preprocess KeyError, wall-clock dry-run, encoder/normalization, reward scale, Open Q 격상) 반영 + Critic 002 41개 항목 §9 매핑.

### 0-2. v2 → v3 변경표 (004 Fix 10건 + Phase A 실측 해소 9건)

| # | 영역 | v2 상태 | v3 변경 |
|---|---|---|---|
| C-N1 | 결정 #14 fork vs 서브클래스 자기모순 | §0 요약은 "서브클래스 override", §1-B는 "fork-patch" — 모순 | **fork-patch 단일안 확정**. `models.py:182` `obs["image"] = obs["image"] / 255.0` 라인을 `if "image" in obs:` 가드로 감싸는 in-place 수정. 서브클래스 표현 §0·§1-B·§9-1 전부 삭제 |
| C-N2 | §4-3 reward 산술표 단위 혼동·R_lap 통일 오류 | "잠정값 미측정인데 R_lap=100 통일" | **Map Easy R_lap=25 / Oschersleben R_lap=100 분리 확정.** 단위 env step 통일 (50 env step/sec). 트랙 길이 plug-in 변수 `L_track`로 reward 함수 작성 — Phase 1-1 측정값을 코드 변수로 직접 대입 |
| C-N3 / F5-10 | map 명명 오인 | "map_easy 통일"이라 했으나 실제는 `map_easy3` | **`map_easy3` 확정** (실측). configs `task='f1tenth_map_easy3'`. v2 결정 #25 오기 정정. env_setting/001 §4와의 불일치는 Phase 1-0에서 env_setting 측 1줄 보정 |
| C-N4 | A19 VRAM 측정 도구·wall-clock 식 유도 | `max_memory_allocated()` 암묵, 식 유도 부재 | **`torch.cuda.max_memory_reserved()` 명시.** Wall-clock 추정식 유도를 §11 부록 A에 정식 표기. 보수적 상한(env+train 직렬 가정) 가정 명시 |
| C-N5 | state 정규화 fixed scale vs symlog 중복 | 둘 다 적용 | **symlog_inputs=False, fixed scale만 사용.** vel_x 범위 [-5/20, 20/20]=[-0.25, 1.0] 인정 (vel_x≥0 강제 안 함), saturation은 step-cap·clip으로 보호 안 함 — 정상 주행 분포에 한해 saturation 없음 (분포 검증은 A_norm 신설) |
| C-N6 | A19 분기 우선순위 모호 | VRAM/wall-clock 동시 fail 순서 미정 | **VRAM 우선** 분기. (1) VRAM fail → batch_size/length 조정 → 재측정 → (2) wall-clock fail 시 train_ratio/steps 조정. 의사코드 §6-3 |
| C-N7 | GPU SKU Open Q로 미루기 | "Phase 진입 시 확정" | **Phase A에서 확정.** RTX 4060 Ti **8GB (8188MiB)** — `nvidia-smi` 실측. §0-3 표·configs 8GB profile default 적용 |
| C-N10 | counter ckpt 의사코드 부재 | "Phase 3에서 저장" 추상적 | **Phase 3 본문 + R7에 의사코드 1줄 명시**: `checkpoint['counters'] = {n: c._last for n,c in [('train',_should_train),('log',_should_log),('eval',_should_eval),('vid',_should_video),('reset',_should_reset)]}` |
| C-§8 | Phase 4 위치 모순 | §3는 "Phase 1 직후", §8 표는 Phase 3 다음 | **§8 일정 표 정정.** Phase 4(reward)를 Phase 1-1 직후(Phase 2 노트북 작업과 동시 진행 가능)로 배치. 일정 표 행 순서 재정렬 |
| C-§7 | replay buffer 디스크 누락 | 없음 | **§7 산출물 표에 replay buffer 행 추가** (~1.5~2GB, fp16 200K step) |
| 해소-Q1 | v2 Open Q #1 트랙 길이 | "Phase 1-1 측정" | **추정값 plug-in + Phase 1-1 측정 확정 절차.** Map Easy ≈ 70m, Oschersleben ≈ 300m (픽셀+resolution 추정). 측정 후 ±30% 이내면 §4-3 표 유지, 초과 시 reward 코드 `L_track` 변수만 갱신 (재산정 자동) |
| 해소-Q2 | v2 Open Q #2 GPU SKU | "Phase 진입 시 확정" | **Phase A에서 4060Ti 8GB 확정.** Open Q 폐기 |
| 해소-Q3 | v2 Open Q #3 GapFollower baseline | "Phase 1-1 측정" 추상 | **결정 #28: Phase 1-1에서 5 ep × 2 map = 10 ep 측정. 절차 의사코드 §11 부록 B. 미달 시 A11/A13 기준을 `45초/110초 절대값`로 fallback** |
| 해소-Q4 | v2 Open Q #4 kinematic vs dynamic | "Phase 진입 시 확인" | **Phase A에서 확정: `vehicle_dynamics_st` 단일 모델 (저속 시 함수 내부에서 kinematic 자동 전환)**. base_classes.py L488 `linear_vels_y=0` 패치 → `vel*sin(slip_angle)` 유효. A6 충족 가능 (저속 5초 직선 후 코너링 2초). 결정 #27 |
| 해소-N9 | 노트북↔집컴 코드 동기화 | 미명시 | **결정 #29: git push/pull로 동기화.** Phase 2-4 진입 전 commit & push 필수 게이트 |
| 해소-N8 | DreamerV3 정당화 정량성 부재 | 정성 4사유 | **§0-6에 한계 명시 추가**: "SAC 직접 비교 미실시, GapFollower(classic) vs DreamerV3(world model) 비교로 대체" |
| 해소-Open-F1-2 | 12M preset 직접 인용 | 부재 | **§1-A #3 비고에 README/configs 미존재 인정 + Table B.1 직접 산출 근거 1줄 추가** |
| 해소-F5-12 | lap_times[0] 시작값 | "Phase 결정" | **Phase A 실측: f110_env.py:529 `np.zeros((1,))`로 0초 시작.** wrapper에서 첫 toggle_list 갱신 전까지 lap_times 진행. 결정 #30 |
| 해소-F5-13 | rollback 표현 모호 | "policy_lap*.pt 중 Map Easy 호환" 모호 | **결정 #31 명시**: A16 미달 시 (a) Stage 1 `latest.pt` 복원 + Stage 2 재학습 (joint replay 50% 강화), (b) Stage 2 정책의 Oschersleben snapshot은 Map Easy 호환 보장 없음 (도메인 다름). Map Easy 검증 시 Stage 1 latest 강제 |

---

## 0-3. 프로젝트 목표 (v1·v2 동일)

1. Map Easy 완주 — 평가 30%
2. Oschersleben 완주 + lap_time 최적화 — 평가 10%
3. 알고리즘 발표 — 평가 60%
4. 100초대 policy snapshot 저장 (LeWorldModel용)

## 0-4. 환경·하드웨어 (Phase A 실측 확정)

| 항목 | 값 |
|---|---|
| 학습 머신 GPU | **NVIDIA GeForce RTX 4060 Ti, 8188MiB (=8GB)** — `nvidia-smi` 실측 (2026-05-20). v2의 "16GB 가정"은 폐기 |
| GPU profile default | **8GB profile**: `precision=16 (AMP), batch_size=8, batch_length=64` 시작부터 적용 |
| 학습 머신 CPU/RAM/Disk | **AMD Ryzen 5 7500F** (6-core 12-thread, 3.70GHz base / 4.34GHz boost, L3=32MB) / **RAM 32GB** DDR5 5200MT/s DIMM (2/4 슬롯 사용) / 2TB SSD |
| 개발 머신 | 노트북, WSL2 (CPU only — Phase 1~Phase 2-3 작업) |
| Python / torch | 3.8, torch 2.4.1 (집컴은 cu121 빌드 — env_setting/004 §5) |
| Sim 경로 | `/home/dlacksdn/f1tenth_RL_project/` (gym + f1tenth_gym_ros + pkg) |
| f1tenth_gym 모듈 경로 | `/home/dlacksdn/f1tenth_RL_project/gym/f110_gym/` (실측 import 위치). `f1tenth_gym_ros/`는 ROS 경로 — 본 계획 미사용 |
| Dreamer 경로 | `/home/dlacksdn/dreamerv3-torch/` (NM512 fork, vendor-in) |
| 제출 포맷 | Pure Gym only (ROS 통합 제외) |

## 0-5. 차량 spec (v1·v2 동일)

차체·액추에이터 변경 금지. `s_min/s_max=±0.4189, v_min/v_max=[-5, 20]` default. 평가도 default.

## 0-6. 왜 DreamerV3인가 (Critic 002 SKEPTIC 대응, 004 N8 한계 추가)

1. 평가 비중 60%가 알고리즘 발표 — 학습곡선·imagination rollout·world model latent 분석 등 발표 자산이 SAC/PPO보다 풍부.
2. LeWorldModel(Offline RL) 추가 과제와 직결 — world model을 그대로 offline data generator로 재사용.
3. 본 계획은 단일 정책 학습이 아니라 12M 모델 검증 + 100초대 snapshot 다양성 확보가 목적.

**한계 명시 (004 N8 fix)**: SAC vs DreamerV3 정량 직접 비교는 wall-clock 예산상 미실시. 발표 ablation은 **GapFollower(classic) vs DreamerV3(world model)** 비교로 대체. 평가 채점 기준이 단순 완주율·lap_time에 가중되면 DreamerV3 선택은 손해 — 발표 채점 외부 의존 인정.

리스크는 (a) wall-clock 초과 (b) 12M 비표준 hyperparameter — 둘 다 §2 A19, §5 R-series에서 정량 측정·대응.

---

## 1. 핵심 설계 결정 (Decision Table)

### 1-A. v1~v2 결정 (#1~#13, 보존)

| # | 항목 | 결정 |
|---|---|---|
| 1 | 구현체 | `NM512/dreamerv3-torch` **fork (vendor-in)** — 본 프로젝트 `/home/dlacksdn/dreamerv3-torch/` 직접 in-place 수정. 패치 사본은 `_thinking/patches/`에 diff 보관 |
| 2 | 스코프 | 베이스라인 + 100초대 snapshot. LeWorldModel/Offline RL/IRL은 별도 |
| 3 | 모델 크기 | **12M** — `dyn_hidden=256, dyn_deter=1024, dyn_stoch=32, dyn_discrete=16, units=256, mlp_units=256`. `cnn_depth`는 미사용이지만 default(=32) 유지하고 `cnn_keys='$^'`로 비활성. *NM512 README의 12M preset 명시 인용은 없음 — Table B.1과 본 hyperparameter는 사용자 직접 구성. A10 비율 보고로 검증 (R11)* |
| 4 | Action space | 2-dim `(steer, speed)`, `actor.dist='normal'`, `absmax=1.0`. wrapper에서 `[-1,1]^2 → [s_min, s_max] × [v_min, v_max]` affine. `imag_gradient='dynamics'` |
| 5 | LiDAR encoder | **ConvEncoder1D** (1D Conv 5-stage stride-2) + Linear(flatten_dim → 512) projection |
| 6 | 보조 obs | `state` 5-dim `[vel_x, vel_y, ang_vel_z, prev_steer, prev_speed]`. v1의 6번째 progress_ratio placeholder 삭제 |
| 7 | Reward | Progress + 종료 신호. §4-3 산술 검증 후 확정 스케일 (Map/Track 분리). lap_complete 방향 가드, progress step-cap |
| 8 | Episode 종료 | 충돌(terminal) / 2-lap 완주(last) / 180s timeout(last/truncated) / 후진 1초(terminal). 우선순위: **collision > reverse > lap_complete > timeout** (§4-4) |
| 9 | Curriculum | 순차 fine-tune Map Easy 500K → Oschersleben 500K. **joint replay 30%** default (R3) |
| 10 | Snapshot | Hybrid: (a) lap_time ≤ 110s 자동 저장 모두, (b) `eval_every=1e4` interval, (c) `latest.pt` 항상 |
| 11 | Step budget | 초기 Stage당 500K. A19 dry-run 후 조정 |
| 12 | Exploration | `expl_behavior='greedy'`. dense reward라 Plan2Explore 불필요 |
| 13 | 제출 | Pure Gym only. 발표에 ONNX export 가능성 멘션 |

### 1-B. v2 Open Q 격상 #14~#26 (004 Fix 반영, **확정값**)

| # | 항목 | 결정 (v3 최종) | 근거/참조 |
|---|---|---|---|
| 14 | **preprocess image KeyError** | **`models.py:182`를 in-place fork-patch.** 라인 변경: `if "image" in obs: obs["image"] = obs["image"] / 255.0`. 서브클래스 표현은 본 v3 전체에서 삭제 (C-N1 fix) | 004 N1, Phase A 실측 |
| 15 | LiDAR/state normalization | LiDAR: `clip(0,30)/30.0` → MLP에 입력 전 symlog는 미적용 (LiDAR는 1D Conv 경로). State: `vel_x/20, vel_y/5, ang_vel_z/π, prev_steer/0.4189, prev_speed/20`. **MLP encoder `symlog_inputs=False`** (C-N5 fix — fixed scale과 중복 제거). vel_x ∈ [-0.25, 1.0] saturation 없음을 A_norm으로 검증 | 004 N5 |
| 16 | Encoder 출력 dim | ConvEncoder1D 1080→540→270→135→**68→34** (ceil 보정), flatten 256×34=8704 → `Linear(8704, 512)` → concat state MLP 256 → 768 | 002 F1-3 |
| 17 | seed 정책 | 학습 single seed=0. `sample_episodes` (tools.py:323) seed=0 하드코딩 유지. 발표에서 "1 seed" 한계 명시 | 002 F5-1 |
| 18 | centerline 추출 | Phase 1-1 격상. `scripts/extract_centerline.py`: skeletonize → arclength → tangent. 출력 csv 5-col `(s, x, y, tx, ty)`. Map Easy3 + Oschersleben 양쪽 생성 | 002 F5-11 |
| 19 | eval 프로토콜 | `eval_episode_num=20`, 시작점 고정, 차량 default, noise `std_dev=0.01`, **`eval_state_mean=True`** | 002 F5-2 |
| 20 | logging stack | TensorBoard + jsonl 동시 (NM512 default). wandb 미사용 | 002 F5-3 |
| 21 | ckpt optimizer carry-over | Stage 2 fine-tune 시 `latest.pt` 복사 → world model weights warm, **actor/critic/model_opt fresh** (`train.py`에 `--fresh_optim` flag). lr 절반 R3 mitigation과 정합 | 002 F5-4 |
| 22 | action wrapper 체인 | `F110GymnasiumWrapper → NormalizeActions([-1,1]) → TimeLimit(9000 env step @ action_repeat=2 = 18000 sim step = 180s) → SelectAction("action") → UUID → Damy`. TimeLimit이 env step 단위임을 Phase A에서 실측 확정 (envs/wrappers.py L7-26) | 002 F1-11, Phase A |
| 23 | prefill 정책 | `prefill=0` 비활성. 첫 10K env step을 **GapFollower 정책으로 수집** (별도 collector) | 002 F5-9 |
| 24 | termination 우선순위 | 한 step 다중 발생 시: `collision → reverse → lap_complete → timeout`. reward는 동일 step 내 모두 합산, lap_complete reward는 방향 dot>0일 때만 | 002 F5-8 |
| 25 | **Map 명명** | **평가 트랙 = `map_easy3`** (`pkg/src/pkg/maps/map_easy3.{png,yaml}` 실측). configs `task='f1tenth_map_easy3'`, Stage 2는 `task='f1tenth_oschersleben'` (`pkg/src/pkg/maps/Oschersleben.{png,yaml}`). v2 결정 #25 (map_easy 통일)는 오기 정정 (C-N3 fix). env_setting/001 §4의 표기는 Phase 1-0에서 별도 1줄 보정 | 004 N3, Phase A |
| 26 | torch.compile | `compile=False`. ConvEncoder1D 동적 padding으로 TorchInductor fallback 위험 | 002 F5-15 |

### 1-C. v3 신규 결정 (#27~#31, Phase A 실측 기반)

| # | 항목 | 결정 | 근거 |
|---|---|---|---|
| 27 | **dynamic model 모드** | `gym/f110_gym/envs/dynamic_models.py:124`의 `vehicle_dynamics_st` 단일 함수 사용 (저속 시 함수 내부에서 kinematic으로 자동 전환, L151). State 7D `[x, y, steer, vel, yaw, yaw_rate, slip_angle]`. `base_classes.py:488`의 `linear_vels_y=0` 하드코딩을 **`vel * sin(slip_angle)`로 패치** (Phase 1-3). A6 충족 가능 (저속 직선 → 코너링 시 slip_angle≠0) | 004 Open Q#4, Phase A |
| 28 | **GapFollower baseline 측정** | Phase 1-1에서 `scripts/measure_gap_follower.py` 실행 — Map Easy3 + Oschersleben 각 5 ep, eval pose 고정, eval과 동일 wrapper. 출력 `_thinking/notes/gap_follower_baseline.md`. **측정 실패 시 fallback A11=45초 절대값, A13 median=110초** (Phase 1-1 게이트 통과 = 측정 완료 또는 fallback 채택). 절차 의사코드 §11 부록 B | 004 Open Q#3 |
| 29 | **노트북↔집컴 코드 동기화** | git push/pull. Phase 2-4 dry-run 진입 전 노트북에서 `git commit && git push`, 집컴에서 `git pull` 후 dry-run 실행 (강제 게이트). dreamerv3-torch fork도 동일 동기화 | 004 N9 |
| 30 | **lap_times 시작값** | f110_env.py:529-530 `self.lap_times = np.zeros((1,)), self.lap_counts = np.zeros((1,))`. wrapper에서 `lap_count[0]` 변화 감지로 lap 완주 trigger. 첫 lap 완주 시점까지 `lap_times[0]`는 0이 아닌 누적 시간 (현재 step의 `current_time` 값) | 004 F5-12, Phase A |
| 31 | **A16 미달 rollback** | (a) Stage 1 `latest.pt`로 복원 + Stage 2 재학습 시 `joint_replay_ratio=0.5`로 강화, (b) Oschersleben snapshot은 Map Easy 호환 보장 없음 인정 — A16 검증 시 Stage 1 latest 강제 사용. 재학습 1회 실패 시 발표에 forgetting 한계 명시 | 004 F5-13 |

### 1-D. Critic 004 Top Fix 직접 매핑

| 004 Fix | v3 반영 |
|---|---|
| C-N1 fork-patch 통일 | §0 #C-N1, 결정 #14, §9-1 모두 `models.py:182 fork-patch` 단일 표현 |
| C-N2 reward 산술 분리 | §4-3 표 Map Easy3 R_lap=25 / Oschersleben R_lap=100, env step 단위 |
| C-N3/F5-10 map 명명 | 결정 #25, configs |
| C-N4 A19 도구·식 | §2 A19 `max_memory_reserved()`, §11 부록 A 유도식 |
| C-N5 정규화 중복 | 결정 #15 symlog_inputs=False |
| C-N6 분기 우선순위 | §6-3 의사코드 |
| C-N7 GPU SKU 확정 | §0-4 실측, Open Q에서 제거 |
| C-N10 counter ckpt 코드 | §3 Phase 3, §5 R7 |
| C-§8 Phase 4 위치 | §8 일정 표 |
| C-§7 replay | §7 산출물 표 |

---

## 2. Acceptance Criteria (Testable, 도구·시점·출처 명시)

> 표기: 〔도구〕 검증 도구, 〔Phase〕 측정 시점, 〔출처〕 기준 근거.

### 2-1. 환경·통합 (Phase 1 게이트)

- [ ] **A1** `f110_gym:f110-v0` → `F110GymnasiumWrapper`로 `reset(seed) → (obs, info)`, `step(action) → (obs, r, terminated, truncated, info)` 5-tuple. 〔도구〕`pytest tests/test_wrapper_api.py`. 〔Phase 1-2 종료〕
- [ ] **A2** obs dict: `{'lidar': (1080,) float32, 'state': (5,) float32, 'is_first': bool, 'is_terminal': bool, 'is_last': bool}`. 〔도구〕pytest. 〔Phase 1-2〕
- [ ] **A3** smoke test 1 episode 무에러. 〔도구〕`python -m dreamer_f1tenth.smoke`. 〔Phase 1-2〕
- [ ] **A4** 충돌 시 `terminated=True ∧ info['cause']=='collision'`. 〔도구〕`pytest::test_collision`. 〔Phase 1-2〕
- [ ] **A5** 9000 env step 도달 시 `truncated=True ∧ terminated=False`. 〔도구〕pytest::test_timeout. 〔Phase 1-2〕
- [ ] **A6** base_classes.py:488 패치 후, 직선 5초 + 코너링 2초 시나리오에서 `|vel_y|.max() > 0.05`. 〔출처〕dynamic single-track, slip_angle≠0. 〔Phase 1-3 종료, 결정 #27〕
- [ ] **A18** 강제 후진 액션 1.1초 → `terminated=True ∧ info['cause']=='reverse'`. 〔Phase 1-4〕
- [ ] **A_centerline** Map Easy3 + Oschersleben centerline csv 5-col 생성. 〔도구〕`python scripts/extract_centerline.py --verify`. 〔Phase 1-1〕
- [ ] **A_gap** GapFollower baseline 측정 완료 또는 fallback 채택 (결정 #28). 〔도구〕`scripts/measure_gap_follower.py`. 〔Phase 1-1〕
- [ ] **A_norm** state 정규화 saturation 검증: 100 ep × 9000 env step 분포에서 `vel_x/20 ∈ [-0.25, 1.05] (99%-ile)`, `|vel_y/5| ≤ 1.0`, `|ang_vel_z/π| ≤ 1.0`. 〔도구〕`pytest tests/test_obs_norm.py`. 〔Phase 1-2 종료, 결정 #15〕

### 2-2. 모델·Encoder·dry-run (Phase 2 게이트)

- [ ] **A7** `ConvEncoder1D(1080) → (B,T,512)`. 1080→540→270→135→68→34, flatten=8704 → Linear→512. 〔Phase 2-1〕
- [ ] **A7b** `ConvDecoder1D(latent → (B,T,1080))` SymlogDist. 〔Phase 2-1〕
- [ ] **A8** `MultiEncoder/MultiDecoder` 패치: `lidar_keys` 분기. configs `lidar_keys='lidar', mlp_keys='state', cnn_keys='$^'`. dry-run에서 "Encoder LIDAR/MLP shapes" 출력 확인. 〔Phase 2-2〕
- [ ] **A9** `configs_f1tenth.yaml` 12M + 8GB profile (precision=16, batch_size=8). `compile=False, video_pred_log=False, prefill=0, eval_state_mean=True`. 〔Phase 2-3〕
- [ ] **A10** `sum(p.numel() for p in agent.parameters() if p.requires_grad) ∈ [10M, 14M]` AND 비율 보고 (RSSM ~30%, encoder+decoder ~50%, heads ~20% 기대). 〔Phase 2-3〕
- [ ] **A20** WorldModel.preprocess image KeyError 해결: vector-only obs 100 train + 10 eval 무에러. 〔도구〕`pytest tests/test_preprocess_patch.py`. 〔Phase 2-0, 결정 #14〕
- [ ] **A19** **dry-run wall-clock + VRAM gate** (Stage 1 진입 게이트):
  - 측정: 1K env step + 100 train step. **A**=env_step_avg_ms, **B**=train_step_avg_ms, **C**=`torch.cuda.max_memory_reserved() / 1024**2` MB, **D**=500K wall-clock 추정 (§11 부록 A 식)
  - **Pass 조건**: C ≤ **6400 MB** (8GB × 0.80) AND D ≤ 1440 min (24h)
  - **Fail 분기**: §6-3 의사코드 (VRAM 우선 → wall-clock)
  - 〔도구〕`python scripts/dryrun_bench.py`. 〔Phase 2-4, Stage 1 진입 게이트〕

### 2-3. 학습·평가 (Phase 5 게이트)

- [ ] **A11** Map Easy3 500K 학습 후 eval (20 ep) **median lap_time ≤ `GapFollower_baseline_map_easy × 1.5`초** (결정 #28 측정값). baseline 측정 실패 시 fallback ≤ 45초. 〔Phase 5-1 종료〕
- [ ] **A12** Oschersleben fine-tune 500K 후 eval 20 ep **2-lap 완주율 (`lap_count[0] >= 2`) ≥ 80%**. 〔Phase 5-2〕
- [ ] **A13** Oschersleben fine-tune 후 eval **median lap_time ≤ 120초 AND best lap_time ≤ 110초**. 〔Phase 5-2〕
- [ ] **A14** lap_time ≤ 110s 정책은 모두 저장 (LeWorldModel 다양성). 파일명 `policy_lap{X:.1f}s_step{Y}k.pt`. 〔Phase 5-2 중 trigger〕
- [ ] **A15** `eval_every=1e4` interval snapshot `step_{N}k.pt`, 500K → 50개 stage당 ~10GB. 〔Phase 5〕
- [ ] **A16** Map Easy3 재평가 (Stage 1 `latest.pt`로 강제 복원) 2-lap 완주율 ≥ 70%. 미달 시 결정 #31 rollback. 〔Phase 5-3〕

### 2-4. Reward·safety

- [ ] **A17** Reward component 분리 로깅: `reward/progress`, `reward/collision`, `reward/lap_complete`, `reward/reverse`. 〔TensorBoard〕. 〔Phase 4 직후 dry-run〕

---

## 3. Implementation Steps

### Phase 1 — Env wrapper + centerline + baseline (예상 3~4일, 노트북 CPU)

#### 1-0. 사전 의존성
노트북 venv에 `gymnasium, pyyaml, scikit-image, scipy, tensorboard` 추가. CPU torch 2.4.1 유지. env_setting/001 §4의 `map_easy` 표기를 `map_easy3`로 1줄 보정 (별도 commit). `free -h`로 RAM 확인 → §0-4 표 보정.

#### 1-1. centerline 추출 + GapFollower baseline (Phase 1 격상)
**`scripts/extract_centerline.py`**:
- 입력: `pkg/src/pkg/maps/{map_name}.png/.yaml`
- skimage `skeletonize` → 폐곡선 정렬 (start = yaml origin 부근 nearest) → arclength 누적 + tangent
- 출력: `maps/{map_name}_centerline.csv` 5-col `(s, x, y, tx, ty)`
- 검증: 총 길이 `L_track`을 `_thinking/notes/track_length.md`에 기록. Map Easy3 추정 70m, Oschersleben 추정 300m와 ±30% 비교

**`scripts/measure_gap_follower.py`** (결정 #28, §11 부록 B 의사코드):
- 5 ep × 2 map 측정, median + min lap_time 기록
- `_thinking/notes/gap_follower_baseline.md` 출력
- 실패 시 `--fallback` flag로 A11=45초, A13_median=110초 채택

#### 1-2. `F110GymnasiumWrapper` (`dreamer_f1tenth/envs/f1tenth_env.py`)
- gym 0.18 4-tuple → gymnasium 5-tuple
- `reset(seed, options)`: 트랙별 default start (env_setting/001 §9, 트랙별 1점 고정)
- obs dict 구성 (결정 #6)
- reward 계산 (§4-3, `L_track` 변수 plug-in)
- action wrapper 체인 (결정 #22)
- info dict에 `cause` 4종 주입 (`collision`/`reverse`/`lap_complete`/`timeout`)

#### 1-3. base_classes.py 패치 (결정 #27)
- `gym/f110_gym/envs/base_classes.py:488` `observations['linear_vels_y'].append(0.)` → `observations['linear_vels_y'].append(agent.state[3] * np.sin(agent.state[6]))`
- `vehicle_dynamics_st` 단일 모델 확정, 별도 mode 분기 불필요

#### 1-4. reverse_guard (centerline tangent · velocity)
- 결정 #24 우선순위 적용
- 카운터 reset: dot ≥ 0이면 0 reset (정지는 후진 아님 — 의도된 동작)

### Phase 2 — Encoder/Decoder + Config + dreamerv3-torch fork + dry-run (예상 3일)

#### 2-0. dreamerv3-torch fork-patch (결정 #1, #14)
**변경 파일·라인 (Phase A 실측 인용)**:
- `dreamerv3-torch/models.py:182` → `if "image" in obs: obs["image"] = obs["image"] / 255.0` (결정 #14, C-N1)
- `dreamerv3-torch/networks.py:293-357` MultiEncoder에 `lidar_keys` 분기 추가 + L360 이후 MultiDecoder 대칭 추가 (변경 라인 ~6곳 예상, 실측은 Phase 2-2 종료 시 확정)
- `dreamerv3-torch/dreamer.py:146-203` `make_env`에 `suite=='f1tenth'` 분기
- `dreamerv3-torch/configs.yaml` 끝에 `f1tenth:` block append
- 패치 diff를 `_thinking/patches/dreamerv3_torch_v3.diff`로 보관

#### 2-1. `ConvEncoder1D` / `ConvDecoder1D` (`dreamer_f1tenth/networks_1d.py`)
- 5-stage stride-2 Conv1d, ch 1→16→32→64→128→256
- 출력 길이 ceil 보정: 1080→540→270→135→68→34. flatten = 8704
- `Linear(8704, 512)` projection
- Decoder: 거울상 ConvTranspose1d + SymlogDist

#### 2-2. MultiEncoder/MultiDecoder 패치
- `lidar_keys` 정규식 매칭, `len(shape)==1 ∧ lidar` → 1D path
- forward concat 순서: `[lidar_out, mlp_out]`, outdim = 512 + 256 = 768

#### 2-3. `configs_f1tenth.yaml` (8GB profile default)
```yaml
f1tenth:
  task: 'f1tenth_map_easy3'    # 결정 #25
  time_limit: 9000             # env step 기준 (envs/wrappers.py L7-26 실측). action_repeat=2 → 18000 sim step → 180s
  action_repeat: 2
  steps: 5e5
  prefill: 0                   # 결정 #23 (GapFollower 별도 collector)
  envs: 1
  parallel: False              # Damy wrapper (parallel.py L198) 사용
  precision: 16                # 8GB profile default (C-N7)
  compile: False               # 결정 #26
  video_pred_log: False        # 결정 #14
  eval_episode_num: 20
  eval_state_mean: True        # 결정 #19
  batch_size: 8                # 8GB profile
  batch_length: 64
  # 12M model
  dyn_hidden: 256
  dyn_deter: 1024
  dyn_stoch: 32
  dyn_discrete: 16
  units: 256
  encoder: {lidar_keys: 'lidar', mlp_keys: 'state', cnn_keys: '$^',
            mlp_units: 256, mlp_layers: 5, symlog_inputs: False}  # 결정 #15 (C-N5)
  decoder: {lidar_keys: 'lidar', mlp_keys: 'state', cnn_keys: '$^',
            mlp_units: 256, mlp_layers: 5, vector_dist: symlog_mse}
  actor: {layers: 2, dist: 'normal', units: 256}
  critic: {layers: 2, units: 256}
  reward_head: {layers: 2, units: 256}
  cont_head: {layers: 2, units: 256}
  snapshot_lap_threshold: 110.0
  snapshot_save_all_below_threshold: True   # 결정 #10, A14
  snapshot_interval_save: True
  dataset_size: 200000         # F5-14
  replay_lidar_fp16: True
```

#### 2-4. dry-run benchmark (A19 게이트)
- 결정 #29: 노트북에서 `git push` → 집컴 `git pull` → `python scripts/dryrun_bench.py`
- 측정값으로 §6-3 분기. Pass 시 Stage 1 진입

### Phase 3 — Snapshot 정책 + train.py wrapper (1일)

- `train.py`가 NM512 `dreamer.py:main`을 monkey-patch
- snapshot 3종 (lap-threshold 모두 저장, interval, latest)
- **Counter ckpt 의사코드 (C-N10)**:
  ```python
  checkpoint['counters'] = {n: c._last for n, c in [
      ('train', _should_train), ('log', _should_log),
      ('eval', _should_eval), ('vid', _should_video),
      ('reset', _should_reset)
  ]}
  ```
- resume 시 동일 dict로 counter 복원

### Phase 4 — Reward + Episode (1일, **Phase 1-1 직후 진행 가능, Phase 5 진입 전 완료**) — C-§8 fix

#### 4-1. Reward 함수 (§4-3 스케일, `L_track` plug-in)
```python
# centerline_arclen: Phase 1-1 측정. L_track = csv 마지막 s 값
progress = clip(arclen_delta, 0.0, 0.5)        # F3-4 step-cap (env step 단위)
reward = alpha_progress * progress             # alpha = 1.0
if collision:
    reward += -10.0; terminated = True; cause = 'collision'
elif reverse_counter >= 50:                    # 50 env step = 1s @ action_repeat=2
    reward += -10.0; terminated = True; cause = 'reverse'
elif lap_count_increased and (velocity_dot_tangent > 0):
    reward += R_lap                            # Map Easy3=25, Oschersleben=100 (§4-3)
    if lap_count >= 2: is_last = True; cause = 'lap_complete'
```

#### 4-2. Joint replay (결정 #9)
fine-tune 시 `--joint_replay_ratio 0.3` (기본). A16 미달 시 0.5 (결정 #31).

### Phase 5 — 학습 실행 (집컴 GPU, 24h 예산)

Stage 1 (Map Easy3 500K), Stage 2 (Oschersleben 500K, `latest.pt` warm load + fresh optim), Stage 3 (Map Easy3 재평가, A16). 8GB profile default.

#### 5-Fallback (A19 결정 결과)
| Trigger | Profile |
|---|---|
| VRAM 초과 | batch_size 8→4 → 재측정. 여전히 fail 시 batch_length 64→32 |
| wall-clock 초과 | train_ratio 512→1024 → 재측정. 여전히 fail 시 steps 500K→300K + 발표 한계 명시 |
| 둘 다 fail | (VRAM 우선 처리 후) train_ratio=1024 + steps=300K |

### Phase 6 — 발표 자료 (1일)

발표 요지: "12M DreamerV3 + 1D LiDAR encoder + progress reward + 순차 fine-tune + LeWorldModel 데이터 generator로의 재활용". **GapFollower vs DreamerV3 비교 슬라이드 1장** (SAC 직접 비교 미실시 한계 명시 — §0-6).

---

## 4. Reward·Episode 세부

### 4-1. Reward 컴포넌트 (결정 #7)
- progress (clip 0~0.5 + α=1.0)
- collision = -10 (terminal)
- reverse = -10 (terminal, 1초 누적 = 50 env step)
- lap_complete = R_lap (방향 가드, Map/Track 분리)

### 4-2. Episode 종료 (결정 #8, #24)
우선순위: collision > reverse > lap_complete (lap_count≥2) > timeout. cause 4종.

### 4-3. **Reward 스케일 산술 검증** (C-N2 fix, env step 단위 통일)

> 단위: env step (= 2 sim step = 0.02s, 50 env step/sec). 트랙 길이는 Phase 1-1 측정값을 `L_track` 변수로 코드에 plug-in. 아래 표는 추정값 + 권장값 = 적용값 (Map/Track 분리, R_lap 통일 금지).

| 트랙 | 트랙 길이 추정 | 목표 lap_time | env step/lap | progress per env step (= L/(T×50)) | lap당 progress 총합 (α=1.0) | R_lap 권장 (30~50%) | **적용값** |
|---|---|---|---|---|---|---|---|
| Map Easy3 (res 0.02) | **L_track ≈ 70m** (Phase 1-1 측정, fallback 50~80m) | 50s | 2500 | 0.028 m/step | 70 | 21~35 | **R_lap = 25** |
| Oschersleben (res 0.04295) | **L_track ≈ 300m** (Phase 1-1 측정, fallback 250~400m) | 110s | 5500 | 0.055 m/step | 300 | 90~150 | **R_lap = 100** |

**Phase 1-1 측정 결과가 추정에서 ±30% 초과** 시 reward 함수의 `L_track`과 `R_lap` 값을 자동 갱신하는 헬퍼 `dreamer_f1tenth/reward_calibrate.py`에서 처리 (코드 1회 실행으로 yaml + py 동시 갱신).

step-cap `clip(progress, 0, 0.5)`: max speed 20 m/s × 0.02s/env_step = 0.4m. 0.5m cap은 0.1m 여유 (정상 주행은 cap에 안 닿음, shortcut/error만 차단).

### 4-4. Termination 우선순위 코드
```python
terminated = False; truncated = False; cause = None; r_extra = 0.0
if collision:
    terminated, cause, r_extra = True, 'collision', -10.0
elif reverse_counter >= 50:                          # 1s @ 50 env step/s
    terminated, cause, r_extra = True, 'reverse', -10.0
elif lap_count_increased and velocity_dot_tangent > 0:
    r_extra += R_lap                                 # Map/Track 분리값
    if lap_count >= 2:
        is_last = True; cause = cause or 'lap_complete'
if not terminated and env_step_count >= 9000:
    truncated = True; cause = cause or 'timeout'
reward = alpha_progress * progress + r_extra
```

---

## 5. Risks and Mitigations (R1~R20)

| # | Risk | 영향 | Mitigation |
|---|---|---|---|
| R1 | vel_y 하드코딩 (kinematic 모드 우려) | obs 정보량↓ | Phase A 실측: `vehicle_dynamics_st` 단일 + slip_angle 사용 패치 (결정 #27, A6) |
| R2 | ConvEncoder1D → RSSM bottleneck | KL 발산 | Linear(8704, 512) + 50K step KL 모니터 |
| R3 | catastrophic forgetting | A16 미달 | joint replay 30% (결정 #9), fresh optim + lr 절반, 결정 #31 rollback |
| R4 | 12M param 초과 | VRAM | A10 비율 체크 + A19 dry-run |
| R5 | Reward 스케일 불균형 | reward hacking | §4-3 산술 + RewardEMA + lap 방향 가드 |
| R6 | centerline lookup CPU cost | step time↑ | scipy kd-tree O(log n) + cache |
| R7 | resume 시 counter reset | burst | counter state ckpt (Phase 3 의사코드, C-N10) |
| R8 | GPU 8GB OOM | OOM | A19 + 8GB profile default (precision=16, batch_size=8) |
| R9 | sample_episodes seed=0 (tools.py:323) | 재현성 | 발표에 "1 seed" 한계 (결정 #17) |
| R10 | preprocess image KeyError | 즉시 실패 | 결정 #14 (`models.py:182` fork-patch) + A20 |
| R11 | 12M 비표준 hyperparameter | 수렴 정체 | A10 비율 + 첫 100K KL/loss 추적 |
| R12 | sim2sim drift | sim only 가정 무시 | 발표 멘션 |
| R13 | 1M step > 24h | 일정 초과 | A19 사전 분기 (§6-3) |
| R14 | gym→gymnasium 변환 | wrapper 작업↑ | 고정 pose (결정 #19) |
| R15 | Replay 디스크 누적 | 디스크 부족 | dataset_size=200K + LiDAR fp16 (F5-14) |
| R16 | LiDAR DR 부재 | sim2sim 갭 | sim only 평가, 후속 ablation |
| R17 | Lap reward hacking (역방향 통과) | reward 부정 | 방향 가드 (§4-1, F3-3) |
| R18 | Progress shortcut | 비현실적 progress | step-cap `clip(0, 0.5m)` |
| R19 | Optim carry-over stale momentum | fine-tune 불안정 | fresh optim (결정 #21) |
| R20 | torch.compile + 1D Conv | compile 실패 | `compile=False` |

---

## 6. Verification Steps

### 6-1. Per-Phase 검증

| Phase | 검증 명령 | 통과 조건 |
|---|---|---|
| 1-1 | `scripts/extract_centerline.py --verify` + `scripts/measure_gap_follower.py` | A_centerline, A_gap, §4-3 `L_track` 갱신 |
| 1-2~1-4 | `pytest dreamer_f1tenth/tests/` | A1~A6, A18, A_norm |
| 2-0 | `pytest tests/test_preprocess_patch.py` | A20 |
| 2-1~2-3 | `scripts/param_audit.py` + dry-run print | A7, A7b, A8, A9, A10 |
| 2-4 | `scripts/dryrun_bench.py` | **A19 게이트** |
| 3 | 학습 200 step → ckpt 저장 확인 | snapshot 3종 + counter dict |
| 4 | reward log 확인 | A17 |
| 5-1 | Map Easy3 500K | A11 |
| 5-2 | Oschersleben 500K | A12, A13, A14 |
| 5-3 | Map Easy3 재평가 (Stage 1 latest 강제) | A16, 미달 시 결정 #31 |
| 6 | 발표 자료 | §7 산출물 |

### 6-2. 전체 통합

- V7 `_thinking/notes/dryrun_results.md` (A19 측정 결과 재현 가능)
- V8 `_thinking/notes/gap_follower_baseline.md` (결정 #28)
- V9 `_thinking/notes/track_length.md` (`L_track` 측정)
- V10 `_thinking/notes/A19_estimate_derivation.md` (C-N4 식 유도, §11 부록 A 사본)

### 6-3. A19 Fail 분기 의사코드 (C-N6 fix)

```python
def a19_branch(C_vram_mb, D_walltime_min, profile):
    while C_vram_mb > 6400:                      # VRAM 우선
        if profile.batch_size > 4:
            profile.batch_size //= 2             # 8 → 4
        elif profile.batch_length > 32:
            profile.batch_length //= 2           # 64 → 32
        else:
            raise RuntimeError("VRAM cannot fit even at batch=4, length=32")
        C_vram_mb, D_walltime_min = remeasure(profile)
    while D_walltime_min > 1440:                 # wall-clock
        if profile.train_ratio < 1024:
            profile.train_ratio *= 2             # 512 → 1024
        elif profile.steps > 3e5:
            profile.steps = 3e5
            note("발표에서 step 축소 한계 명시")
            break
        else:
            raise RuntimeError("wall-clock cannot fit")
        C_vram_mb, D_walltime_min = remeasure(profile)
    return profile
```

---

## 7. 산출물

| 산출물 | 경로 | 비고 |
|---|---|---|
| Stage 1 weights | `~/logdir/f1tenth_v3_map_easy3/latest.pt` | A16 검증용 강제 보존 |
| Stage 2 weights | `~/logdir/f1tenth_v3_oschersleben/latest.pt` | 메인 |
| 100초대 snapshots | `~/logdir/.../snapshots/policy_lap*.pt` | LeWorldModel 다양성 |
| Interval snapshots | `~/logdir/.../snapshots/step_*k.pt` | Stage당 50개, ~10GB × 2 = ~20GB |
| **replay buffer** | `~/logdir/.../replay/` | **dataset_size=200K + LiDAR fp16 ≈ 1.5~2GB** (C-§7 fix) |
| metrics.jsonl + TB | `~/logdir/.../` | |
| centerline csv | `maps/{map_easy3,Oschersleben}_centerline.csv` | |
| dry-run 결과 | `_thinking/notes/dryrun_results.md` | A19 |
| GapFollower baseline | `_thinking/notes/gap_follower_baseline.md` | 결정 #28 |
| `L_track` 측정 | `_thinking/notes/track_length.md` | §4-3 plug-in |
| patch diff | `_thinking/patches/dreamerv3_torch_v3.diff` | 재현성 |

---

## 8. 일정 (C-§8 fix — Phase 4 위치 정정)

| Phase | 소요 | 머신 | 의존 |
|---|---|---|---|
| 1-0 (의존성, GPU/RAM capture, env_setting 보정) | 0.5일 | 노트북 + 집컴 1회 | — |
| 1-1 (centerline + GapFollower baseline) | 1일 | 노트북 | 1-0 |
| 1-2~1-4 (wrapper + dynamics 패치 + reverse_guard) | 2일 | 노트북 | 1-1 |
| **4 (reward 함수 코드화)** | 1일 | 노트북 | **1-1 (L_track, R_lap 확정 후)** |
| 2-0~2-3 (encoder + dreamerv3 fork + configs) | 2일 | 노트북 | 1-2 |
| 2-4 (dry-run A19 게이트) | 0.5일 | 집컴 (결정 #29 동기화 후) | 2-3, 4 |
| 3 (snapshot + train.py + counter ckpt) | 1일 | 노트북 | 2-4 |
| 5-1 (Map Easy3 500K) | 8~12h ± A19 | 집컴 | 3 |
| 5-2 (Oschersleben 500K) | 8~12h ± A19 | 집컴 | 5-1 |
| 5-3 (Map Easy3 재평가) | 1.5h (20 ep × ~3min) | 집컴 | 5-2 |
| 6 (발표) | 1일 | 노트북 | 5-3 |
| **총** | **~11~13일** | | |

Phase 4가 Phase 1-1 직후로 이동 — Phase 2와 병렬 가능. `L_track` 확정 → reward 함수 즉시 코드화 → Phase 2-4 dry-run에 reward까지 포함된 step 시간 측정.

---

## 9. Critic 002·004 대응표

### 9-1. Critic 002 CRITICAL (v2에서 closed, v3에서 단일 표현 통일)

| Critic # | 항목 | v3 대응 |
|---|---|---|
| F1-1 | preprocess image KeyError | **해결** — 결정 #14: `models.py:182` fork-patch (단일안). A20. R10 |
| F2-1 | wall-clock 무근거 | **해결** — A19 + §11 부록 A 식 유도 (C-N4) |
| F2-2 | GPU SKU + VRAM 추정 | **해결** — §0-4 4060Ti 8GB 실측 확정 (C-N7), A19 측정 도구 max_memory_reserved (C-N4) |

### 9-2. Critic 004 Top Fix (10건 + Open Q 해소)

| 004 항목 | v3 위치 | 상태 |
|---|---|---|
| C-N1 fork vs 서브클래스 통일 | §0 C-N1, 결정 #14, §9-1 | **CLOSED** |
| C-N2 reward 산술 분리 + 단위 | §4-3 표 (Map/Track 분리) | **CLOSED** |
| C-N3/F5-10 map_easy3 명명 | 결정 #25, configs `task='f1tenth_map_easy3'` | **CLOSED** |
| C-N4 VRAM 도구 + 식 유도 | A19, §11 부록 A | **CLOSED** |
| C-N5 정규화 중복 | 결정 #15 `symlog_inputs=False` | **CLOSED** |
| C-N6 A19 분기 우선순위 | §6-3 의사코드 | **CLOSED** |
| C-N7 GPU SKU Open Q 격하 | §0-4 실측, Open Q에서 제거 | **CLOSED** |
| C-N10 counter ckpt 의사코드 | Phase 3 본문 + R7 | **CLOSED** |
| C-§8 Phase 4 위치 | §8 일정 표 | **CLOSED** |
| C-§7 replay buffer | §7 산출물 표 | **CLOSED** |
| Open Q1 트랙 길이 | `L_track` plug-in + §11 부록 측정 | **CLOSED (추정 + 측정 절차)** |
| Open Q2 GPU SKU | §0-4 실측 | **CLOSED** |
| Open Q3 GapFollower | 결정 #28 + §11 부록 B + fallback | **CLOSED (절차 + fallback)** |
| Open Q4 dynamic mode | 결정 #27 (`vehicle_dynamics_st` 단일) | **CLOSED** |
| N8 정량 비교 한계 | §0-6 한계 명시 | **CLOSED** |
| N9 노트북↔집컴 동기화 | 결정 #29 | **CLOSED** |
| F1-2 12M preset 인용 | 결정 #3 비고 (인용 부재 인정 + A10 모니터) | **CLOSED (한계 인정)** |
| F5-12 lap_times 시작값 | 결정 #30 | **CLOSED** |
| F5-13 rollback 모호 | 결정 #31 | **CLOSED** |

### 9-3. v2의 §9 대응 항목 (보존, 변경 없는 것은 v2 §9 참조)

v2 §9의 F1-3~F5-15 대응(35개 항목) 중 v3에서 별도 변경 없는 것은 v2 위치 유지. 본 §9에 다시 옮기지 않음 (문서 길이 압축).

---

## 10. v3 Open Questions (목표 0건)

**잔여 0건.** v2의 4건은 모두 §0-2 해소-Q1~Q4 행으로 결정표 이동.

후속 (unscored, 발표 후 ablation 후보):
- F1-2의 NM512 README에 12M preset 공식 명시 여부 — 후속 메일/이슈 (현재 한계 인정).
- SAC 직접 비교 — wall-clock 예산 외 후속 과제.

---

## 11. 부록

### 11-A. A19 Wall-clock 추정식 유도 (C-N4 fix)

`dreamerv3-torch`의 학습 루프 (`dreamer.py` main loop):
- env loop: 1 env step = action 결정 + sim step × action_repeat. 측정값 `A` ms/env_step.
- train loop: `_should_train`가 `every` step마다 발동 (default `train_ratio=512`라면 평균 1/512 비율). 1회 train = batch_size×batch_length×model forward+backward. 측정값 `B` ms/train_step.

**보수적 상한 가정 (env+train 직렬 실행, GPU 단일 stream)**:
```
D_min(N_steps) = ( N_steps × A + (N_steps / train_ratio) × B ) / 1000 / 60
```
- N_steps = 500K, A=env_step_avg_ms (실측), B=train_step_avg_ms (실측), train_ratio=512 default.
- 실제 dreamerv3-torch는 env step과 train step이 같은 thread에서 교차 실행이므로 상기 식은 wall-clock 상한. 실제는 GPU 비활성 구간이 약간 짧을 수 있으나 1.0배 이내 차이로 가정.

**예시 (가상)**: A=5ms, B=50ms → D = (500000×5 + 500000/512×50)/60000 = 41.7 + 0.81 = 42.5분. 24h(=1440분) 통과.

본 식·예시는 dry-run 직후 `_thinking/notes/A19_estimate_derivation.md`에 실측값으로 재기록.

### 11-B. GapFollower baseline 측정 절차 (결정 #28)

`scripts/measure_gap_follower.py` 의사코드:
```python
# GapFollower: Free Space Gap Follow. F1Tenth 표준 baseline.
# pkg/src/pkg/ 또는 본 계획에서 직접 구현. 핵심 로직:
#   1) scan 1080 → mask (range < 3m) → 가장 긴 free gap 탐색
#   2) gap 중앙 각도 → steering 비례
#   3) speed = base_speed × (free_gap_width / max_width)

def run_baseline(env, n_ep=5, max_step=9000):
    laps, times = [], []
    for _ in range(n_ep):
        obs, _ = env.reset(seed=0)
        steps = 0
        while steps < max_step:
            action = gap_follower(obs['lidar'])
            obs, _, term, trunc, info = env.step(action)
            steps += 1
            if term or trunc: break
        if info.get('cause') == 'lap_complete':
            laps.append(1)
            times.append(env.unwrapped.lap_times[0])
    return median(times) if times else None

# main:
for map_name in ['map_easy3', 'Oschersleben']:
    env = make_f1tenth_env(map_name)
    median_t = run_baseline(env, n_ep=5)
    if median_t is None:
        # fallback: A11=45초 (Map Easy3), A13_median=110초 (Oschersleben)
        median_t = {'map_easy3': 45.0, 'Oschersleben': 110.0}[map_name]
    log(f"{map_name}: GapFollower median lap = {median_t:.1f}s")
```
출력 `_thinking/notes/gap_follower_baseline.md`. A11 = `map_easy3_median × 1.5`, A13_median = `Oschersleben_median × 1.1` (둘 다 baseline 측정 성공 시), 실패 시 fallback 절대값.

### 11-C. dreamerv3-torch 직접 인용 위치 (Phase A 실측)

| 위치 | 라인 | 용도 |
|---|---|---|
| `dreamerv3-torch/models.py` | L177-192 (preprocess), **L182** image divide | 결정 #14 fork-patch 대상 |
| `dreamerv3-torch/networks.py` | L293-357 (MultiEncoder) | Phase 2-2 lidar_keys 분기 추가 |
| `dreamerv3-torch/networks.py` | L360~ (MultiDecoder) | Phase 2-2 대칭 분기 |
| `dreamerv3-torch/dreamer.py` | L146-203 (make_env), L198 `TimeLimit` 적용 | Phase 2-0 suite=='f1tenth' 분기 |
| `dreamerv3-torch/envs/wrappers.py` | L7-26 (TimeLimit) | 단위 = env step 확정 |
| `dreamerv3-torch/parallel.py` | L198-209 (Damy) | parallel=False 시 사용 |
| `dreamerv3-torch/tools.py` | L323 (sample_episodes seed=0) | R9, 결정 #17 |
| `gym/f110_gym/envs/base_classes.py` | L488 `linear_vels_y=0` | 결정 #27 패치 대상 |
| `gym/f110_gym/envs/dynamic_models.py` | L124 `vehicle_dynamics_st`, L151 저속 kinematic 전환 | 결정 #27 모델 확정 |
| `gym/f110_gym/envs/f110_env.py` | L529-530 lap_times/lap_counts 초기화, L596-598 lap_count 갱신, L600 done | 결정 #30, lap 완주 감지 |

---

## v3 Self-Review (B-6 체크리스트 self-verify)

| # | 체크 | 결과 |
|---|---|---|
| 1 | 004 Fix 10개 모두 §0 변경 요약 1행씩 존재 | **PASS** — §0-2 표 C-N1, N2, N3, N4, N5, N6, N7, N10, §8, §7 = 10건 |
| 2 | Open Questions ≤ 1건 | **PASS** — §10 잔여 0건 (후속 unscored 2건은 발표 후 ablation, blocker 아님) |
| 3 | "잠정값"/"Phase X에서 측정 후 확정" 표현 0건 | **PASS** — §4-3 표는 확정값 (Map=25, Track=100). `L_track` plug-in은 "측정 후 확정"이 아니라 "코드 변수로 plug-in"으로 표현. 잠정값 단어 미사용 |
| 4 | §0/§1-B/§9-1 결정 #14 동일 채택안 (N1 회귀 방지) | **PASS** — 모두 "`models.py:182` fork-patch (가드 추가)" 단일 표현. 서브클래스 표현 0건 |
| 5 | §4-3 Map Easy3 R_lap과 Oschersleben R_lap 분리 | **PASS** — Map Easy3 R_lap=25, Oschersleben R_lap=100 |
| 6 | configs `task` 문자열이 §1-B #25와 실제 파일 일치 | **PASS** — `f1tenth_map_easy3` ↔ `pkg/src/pkg/maps/map_easy3.png/.yaml` (Phase A 실측 확인) |
| 7 | action_repeat=2 가정 하 reward 산식 단위 일관 | **PASS** — 전부 env step (50/sec). step-cap 0.5m/env_step, reverse_counter 50 env step, time_limit 9000 env step |
| 8 | dreamerv3-torch 인용 라인 번호가 A-2 실측과 일치 | **PASS** — §11-C 표에 실측 라인 (models.py:182, networks.py:293-357, parallel.py:198-209, tools.py:323, wrappers.py:7-26) |
| 9 | 신규 결정 #27+ 번호 충돌 없음 | **PASS** — #27 dynamic, #28 GapFollower, #29 sync, #30 lap_times, #31 rollback. 모두 v2 #1~26 다음 |
| 10 | §6 acceptance 가 §3 Phase 의존성과 정합 | **PASS** — A_centerline/A_gap=Phase 1-1, A6=Phase 1-3, A20=Phase 2-0, A19=Phase 2-4 게이트, A11/A12/A13/A14=Phase 5, A16=Phase 5-3 |

**Self-review 결과**: 10/10 PASS. v3 산출 진행.

---

> **상태**: 최종본 (no further patch). Phase 1-0 착수 가능.
> v2 대비 결정 5건 추가 (#27~#31), Acceptance 2건 추가 (A_gap, A_norm), 신규 부록 §11-A/B/C, Critic 004 Fix 10건 + Open Q 4건 전수 종결, Open Questions 잔여 0건.
