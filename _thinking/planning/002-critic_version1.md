# 비판적 평가: 001-f1tenth_dreamerV3_version1.md

**평가자**: oh-my-claudecode:critic (ADVERSARIAL 모드)
**평가 대상**: /home/dlacksdn/f1tenth_RL_project/_thinking/planning/001-f1tenth_dreamerV3_version1.md
**Verdict**: **REJECT (REVISE 필요)**
**요약**: CRITICAL 2건, MAJOR 20+건, MINOR 다수

**Pre-commitment 예측**: (1) 12M hyperparam 매핑이 논문 Table 3 수치와 어긋날 가능성, (2) `models.preprocess`의 `image` 키 하드코딩이 vector-only env에서 KeyError 유발, (3) `video_pred_log=False`만으로 부족할 가능성, (4) GPU VRAM 추정 부재, (5) catastrophic forgetting mitigation 약함, (6) RTX 4060Ti의 실제 사양(8GB/16GB) 미확정, (7) Pure Gym은 gym 0.18 (gymnasium 아님) — wrapper 작업량 과소평가, (8) `f110-v0`의 reward/done 시그니처가 4-tuple(gym 0.18)이라 gymnasium 5-tuple 변환 필요. → 7개 중 6개 적중.

---

## 1. 기술 정합성

### F1-1. **`preprocess()`의 `image` 키 하드코딩이 무조건 KeyError** — Severity: **CRITICAL**

`models.py:182`에서 `obs["image"] = obs["image"] / 255.0`이 **무조건 실행**됩니다. `video_pred_log=False`로 끄는 것은 **video 시각화만** 끄는 것이지 `preprocess`는 우회하지 않습니다. 계획서 §2-3(A9)와 R10은 `video_pred_log=False`만 명시했는데 이는 R10(KeyError) 해결과 무관합니다. 모든 train step·eval step의 `Dreamer._policy → wm.preprocess`에서 매번 실패합니다.

- **증거**: `models.py:182` `obs["image"] = obs["image"] / 255.0` (assert 없이 직접 인덱싱). analysis/005 §1-4가 이미 "image 키 무조건 접근" 명시.
- **Fix**: `preprocess` 자체를 패치하거나 `WorldModel`을 서브클래싱. 계획서는 "in-place 수정 불가피" 선언했지만 `dreamerv3-torch/networks_f1tenth.py` 분리 정책과 충돌. 명확한 fork/patch 정책 필요.
- Confidence: HIGH.

### F1-2. **12M hyperparameter 매핑이 논문 Table 3와 다를 가능성** — Severity: **MAJOR**

계획서 §1 #3은 "논문 Table 3: d=256, deter=8d=1024, base CNN ch=d/16=16, codes=d/16=16"이라 주장합니다. 그러나 DreamerV3 논문 Table B.1 (모델 크기 스케일링)에서:
- 12M ≈ Small size. `dyn_hidden`/`units`는 보통 `d`와 같지만, **`dyn_stoch=32, dyn_discrete=16`**으로 codes=16을 stoch 자리에 매핑한 것은 비표준입니다. 원본 논문은 stoch과 discrete를 분리해 (stoch=32, classes=16 또는 32, classes=8 등) 표기합니다.
- `cnn_depth=16`은 **1D Conv encoder를 신설**하므로 의미 없는 파라미터입니다 (`cnn_keys: '$^'`로 CNN 경로 비활성). configs.yaml에 cnn_depth 적었는데 사용되지 않습니다 — **혼란 유발**.
- `mlp_units=256`: 기본 1024 → 256으로 축소했는데, 이는 encoder/decoder vector path에만 영향. analysis/004 §3-1이 명시한 outdim에 직접 영향.
- **`dyn_hidden=256` + `dyn_deter=1024`** 조합: `dyn_deter`가 GRU hidden인데 `dyn_hidden`은 GRU 직전 MLP hidden입니다. 보통 `deter ≈ 2~4×hidden` 비율인데 4배는 큰 편 — 학습 불안정 우려.

- **Fix**: 논문 Table B.1 원전 인용 또는 NM512 README의 size preset (있다면) 확인. 12M 사이즈 preset을 명시한 PR/이슈 인용 추가. 그리고 cnn_depth는 configs에서 제거하거나 주석 처리.
- Confidence: MEDIUM (NM512 정식 12M preset 직접 검증 안 됨).

### F1-3. **1D Conv outdim=8448은 feat_size와 합쳐져 RSSM 입력에 위험** — Severity: **MAJOR**

계획서 A7: `256 × 33 = 8448`. RSSM `_obs_out_layers`의 입력은 `deter ⊕ embed = 1024 + (8448 + 256) ≈ 9728` → MLP가 hidden=256으로 압축. **representation bottleneck이 극단적**(9728→256 = 38배 축소). encoder 정보가 obs_out_layers에서 대량 소실. R2가 인지하나 mitigation이 "AdaptiveAvgPool1d(8)"로 임의 — 사전 검증 없음.

- **Fix**: ConvEncoder1D 마지막에 `Linear(8448, 256~512)` 명시적 projection 추가 권장. 또는 stride/depth로 자연스럽게 outdim ~512~1024로 떨어지게 stage 재설계.
- Confidence: HIGH (전형적 over-bottleneck 패턴).

### F1-4. **1D Conv stage 수치 산술 오류** — Severity: **MINOR**

A7: "1080→540→270→135→67→33". stride-2 + kernel=4 + `Conv1dSamePad`라면 SAME padding에서 `out = ceil(in/2)`. 1080→540→270→135→**68**→34. `Conv2dSamePad` 공식 ([networks.py:771-798]) 이식 시 그렇습니다. 135 → 68 (홀수 입력 → ceil) → 34. 계획서의 "135→67→33"은 floor 가정으로 1 픽셀씩 어긋남.

- **Fix**: 실제 ConvEncoder1D 구현 후 `print(x.shape)`로 검증, A7 문구를 수정.
- Confidence: HIGH.

### F1-5. **`MultiDecoder`에 lidar 재구성 추가는 자명하지 않음** — Severity: **MAJOR**

§2-2가 "LiDAR decoder 추가, SymlogDist 사용"이라 명시했지만 1D ConvTranspose 디코더 작성 자체가 별도 ~100줄 작업입니다. analysis/004 §3-4의 `ConvDecoder`는 2D ConvTranspose 기반. 1D 거울상 디코더 신설을 Phase 2 "2일" 일정에 묻어두는 것은 과소평가.

- **Fix**: A7 옆에 A7b "ConvDecoder1D 구현 및 출력 shape `(B,T,1080)` 검증"을 추가하고 별도 1일 할당. 또는 lidar를 vector path로 (`mlp_keys: 'lidar'`) 우회하는 옵션 명시.
- Confidence: HIGH.

### F1-6. **MultiEncoder 패치 — 기존 `cnn_keys`/`mlp_keys` 외 `lidar_keys` 추가는 단순 patch 아님** — Severity: **MAJOR**

[networks.py:309-322]의 라우팅은 `len(shape)==3 ∧ cnn match` (이미지) / `len(shape) in (1,2) ∧ mlp match` (벡터)로 결정. LiDAR `(1080,)`은 len=1이라 mlp_keys로 매칭됩니다. `lidar_keys`라는 **세 번째 경로**를 추가하려면:
- `__init__` 시그니처에 lidar_keys 추가
- `len(shape)==1 ∧ lidar match` 분기 신설
- forward concat 순서 결정
- `outdim` 계산에 `_lidar.outdim` 합산
- decoder도 동일 패치

계획서 §2-2 "라우팅 로직에 lidar regex 매칭 분기 추가"는 한 줄로 적었지만 코드 작업 4~6 곳 in-place 수정이 필요합니다. "in-place 수정 불가피"라 했지만 Phase 1-1 디렉토리 정책 "git clone 그대로 보존"과 직접 충돌.

- **Fix**: 명시적으로 `dreamerv3-torch/networks.py`에 fork 적용 또는 patch file 생성. 변경 라인 list 작성 (예: L297, L316, L321, L326, L346...). 그리고 fork 정책에 따라 dreamerv3-torch를 submodule로 두지 말고 직접 vendor in.
- Confidence: HIGH.

### F1-7. **gym 0.18 → gymnasium 변환 wrapper 작업량 과소평가** — Severity: **MAJOR**

env_setting/004 §3: `gym==0.18.0` 사용 중. f110-v0의 `step()` 반환은 `(obs, r, done, info)` 4-tuple. gymnasium은 5-tuple `(obs, r, terminated, truncated, info)`. 계획서 R14가 "wrapper conditional 작성"으로 처리했으나:
- `f110-v0.reset(poses=poses)` — `poses` 인자가 키워드 강제, gymnasium의 `reset(seed, options)`와 시그니처 불일치
- env_setting/001 §9 — "poses 변수를 별도 설정", 매 reset 시 manual seeding 필요
- gymnasium 어댑터에 `seed`를 `poses` randomization으로 매핑해야 함 — randomization wrapper 없음(analysis/001 §4)

- **Fix**: Phase 1에 "poses randomization 설계" (트랙 따라 시작점 sampling) 명시. A1을 "gym 0.18 인터페이스로 reset/step 가능"으로 변경하거나 별도 reset adapter 작성.
- Confidence: HIGH.

### F1-8. **`linear_vels_y=0` 패치 — base_classes.py 488 하드코딩 확인됨** — Severity: **MAJOR**

`base_classes.py:488` `observations['linear_vels_y'].append(0.)` 직접 확인. R1/A6 이미 인지. 단, `vel_y` 계산은 단순치 않음 — state는 `[x, y, steer_angle, vel, yaw, yaw_rate, slip_angle]`이고 `vel_y = vel * sin(slip_angle)`. 계획서 §1-3 "state[3] * sin(state[6])"이 맞지만, **dynamic_models의 single-track 모델이 slip_angle을 항상 0 아닌 값으로 출력하는지 검증 필요**. kinematic 모드면 slip_angle=0이라 vel_y=0.

- **Fix**: dynamic_models.py 모드 (kinematic vs dynamic) 사전 확인. A6에 "dynamic single-track 모드 활성화 검증" 추가.
- Confidence: HIGH.

### F1-9. **action_repeat=2가 100Hz 시뮬레이션에 미치는 영향 미명시** — Severity: **MAJOR**

configs `action_repeat=2`. F1Tenth는 100Hz physics. action_repeat=2면 **50Hz 제어 → 100Hz physics**. 그러나 analysis/001 §2-3: ROS 모드는 50Hz 제어, Pure Gym은 100Hz 제어. 즉 계획서의 action_repeat=2는 ROS 평가환경(50Hz)에 맞추는 효과지만 **명시 안 됨**.

`time_limit=18000` = 18000 simulator steps. action_repeat=2면 env step 기준 9000. dreamer.py:213-216에서 `time_limit / action_repeat = 9000`. 18000 step × 0.01s = 180초 ← 의도와 일치하나 action_repeat 의미 헷갈림.

- **Fix**: `time_limit`을 simulator step 기준인지 env step 기준인지 명시. dreamer.py:213-216 흐름을 §2-3에 짧게 주석.
- Confidence: HIGH.

### F1-10. **`make_env` 분기 추가 — `dreamer.py:146-203` in-place 수정 불가피** — Severity: **MINOR**

계획서 §2-4 "또는 patch 파일". `make_env`는 `suite`별로 if-elif 체인 — fork 외 깔끔한 monkey-patch가 어려움. 명시적으로 "dreamer.py 직접 수정" 선언 필요.

- **Fix**: `dreamerv3-torch/dreamer.py` fork 또는 wrapper script (`dreamer_f1tenth/train.py`)에서 `dreamer.make_env`를 monkey-patch한 뒤 `dreamer.main()` 호출.

### F1-11. **`Damy(env)` 시그니처 누락** — Severity: **MINOR**

dreamer.py:237-245의 env wrapping은 `Parallel(env, 'process')` (configs `parallel=True`) 또는 `Damy(env)`. 계획서 §2-4의 wrapper 체인에서 Damy/Parallel 미언급.

---

## 2. 일정 현실성

### F2-1. **Phase 5-1/5-2 각각 12시간 추정은 근거 없음** — Severity: **CRITICAL**

§8: "Map Easy 500K, 12시간" "Oschersleben 500K, 12시간" — 산출 근거 0. DreamerV3 12M, train_ratio=512, batch_size=16, batch_length=64의 RTX 4060Ti 실측치는:
- 공개 벤치마크 부재. NM512/dreamerv3-torch README는 환경별 학습 시간 명시 안 함.
- 단일 GPU에서 DreamerV3 1M step (DMC) — 보통 24~48시간 (논문 ablation 부록). 12M 모델 + 1080 LiDAR encoder는 vector환경보다 무거움.
- F1Tenth 시뮬레이션 step도 GIL 묶인 Python — 환경 step 비용이 크기 때문에 GPU forward 대비 env step이 병목 (analysis/002 §2 인지).
- **단순 추정**: env step ≈ 5~10ms (LiDAR 1080 ray + collision GJK), train step ≈ 50~150ms (12M model, batch_size=16). train_ratio=512면 env step 2당 train 1 → wall clock은 env step 시간이 지배. 500K env step × 7ms ≈ 1시간 단순 + train step burst ≈ 추가 6~12시간.

추정치는 그럴듯하지만 **첫 10K step 측정으로 추정**이라는 R13의 fallback이 너무 늦음 (Phase 5 진입 후 발견).

- **Fix**: Phase 2 dry-run에 "1K env step + 100 train step wall-clock 측정 → 500K 추정" 추가. A10 옆에 "추정 wall-clock < 24h"를 acceptance에 명시.
- Confidence: MEDIUM (실측 없음, 가정 기반).

### F2-2. **RTX 4060Ti VRAM 추정 없음, 8GB/16GB 미확정** — Severity: **CRITICAL**

R8이 이를 인지하지만 환경설정 문서(env_setting/001~004) 전체에 GPU 사양 확정 정보 없음. 계획 제목엔 "RTX 4060Ti 16GB" 가정하나 4060Ti는 **8GB·16GB 두 SKU 모두 존재**. 16GB가 아닐 가능성 정량화 없음.

12M model + batch_size=16 × batch_length=64 × (LiDAR 1080 + state 6 + action 2) + RSSM intermediate 활성:
- 활성 ≈ B·T·embed = 16·64·8704 = 8.9M float32 = 35MB (단일 layer 활성, RSSM 64 step 펼치면 ×2~5)
- Adam optim state ≈ 2× 모델 ≈ 24M param × 8 bytes = 192MB
- **AMP=False (precision=32 기본)** → AMP 검토 없이 16GB 가정. AMP enable 시 절반.

- **Fix**: 환경설정 문서에 GPU spec 확정 (nvidia-smi 출력 또는 SKU 확인). A10에 "VRAM peak < 13GB" 측정 단계 추가.
- Confidence: HIGH.

### F2-3. **Phase 1-4 (env wrapper + encoder + config + reward) = 6~7일이 노트북 CPU only로 가능한가** — Severity: **MAJOR**

env_setting/004: 노트북은 CPU only torch 2.4.1, gymnasium 미설치. dreamerv3-torch는 `device='cuda:0'` 기본, CPU에서 dry-run 시 별도 `--device cpu` flag 필요. dry-run(`steps=100`) 자체가 CPU에서 train step 50~200ms이면 100 step ≈ 1분이라 가능하지만, **gymnasium 미설치 환경**에서 wrapper 테스트는 의존성 추가 작업 필요.

- **Fix**: Phase 0 (선행) "노트북 venv에 gymnasium + dreamerv3-torch 의존성 설치" 명시. requirements.txt 확인.
- Confidence: HIGH.

### F2-4. **fine-tune 후 Map Easy 재평가만 1시간 — 일정 너무 짧음** — Severity: **MINOR**

20 episode 평가에 episode당 평균 60초 + reset overhead ≈ 30분이라 1시간은 타이트. eval_episode_num=20 인지, 10인지 명시 없음.

---

## 3. Risk 충분성

### F3-1. **누락 위험: replay buffer 디스크 무한 증가** — Severity: **MAJOR**

analysis/006 §2-6 명시: "디스크는 영구 누적". 1M step × LiDAR 1080 × float32 = 4.3GB raw, np.savez_compressed로 50~70% 압축해도 **2~3GB**. interval snapshot 100개(§7) + replay 합치면 **20GB 이상**. 2TB SSD라 절대 한계는 아니지만 R 목록에 없음.

- **Fix**: R15 신설. replay LiDAR dtype을 float16으로 저장 (analysis/002 §5 권장). 별도 정리 스크립트 또는 dataset_size 도달 시 디스크 GC 명시.

### F3-2. **누락 위험: LiDAR ray dropout / noise robustness** — Severity: **MAJOR**

analysis/001 §1-2: `std_dev=0.01` 노이즈는 매우 작음. 실차 LiDAR는 5~10cm noise + dropout(max_range=30이면 무한대 처리). 학습 정책이 깨끗한 LiDAR에 과적합되어 평가 환경(다른 noise, 다른 dropout)에서 깨질 위험. §9가 "sim only 평가"로 명시적 제외했으나 ROS 평가가 다른 LiDAR sim을 쓰면(analysis/001 §2-3 LiDAR offset 미적용) — **sim2sim 갭이 더 큼**.

- **Fix**: R16 신설. 또는 LiDAR domain randomization(noise×2, dropout 5%) 옵션을 Phase 5-1 후 ablation에 추가.

### F3-3. **누락 위험: reward hacking — `lap_completed +50`이 후진 통과 시도 유발** — Severity: **MAJOR**

analysis/001 §2-7: "후진 랩 카운트 버그 — 출발선 근접 토글로 카운트, 방향성 체크 없음". 계획서 §3-1: `lap_count_increased`로 +50. 후진 가드 1초 누적 −10 vs lap_complete +50 → **후진해서 출발선 통과하면 -10+50=+40 순이익**. 후진 가드는 1초 누적까지 grace period가 있어 짧은 후진 → 통과 → 멀리 가서 다시 진행 가능.

- **Fix**: lap_count_increased 판정 시 방향 체크 (centerline tangent dot velocity > 0). 또는 R17 신설 + Phase 4 reward 함수에 방향 가드.
- Confidence: HIGH.

### F3-4. **누락 위험: progress reward의 trajectory loop / shortcut 악용** — Severity: **MAJOR**

centerline arclength delta로 progress 정의 시, centerline 정의가 자기교차 트랙 (Oschersleben처럼 복잡한 트랙)에서 nearest neighbor가 잘못된 segment에 붙으면 비현실적으로 큰 progress 발생. 또는 트랙 좁은 곳에서 벽 타기로 arclength 부풀리기.

- **Fix**: progress reward에 max delta cap 추가 (예: `clip(progress, 0, 0.5m)`). Phase 4 reward에 명시.
- Confidence: HIGH.

### F3-5. **누락 위험: sample_episodes seed=0 하드코딩의 평가 신뢰성 영향** — Severity: **MINOR**

R9가 인지했으나 "그대로 둠"으로 결정. 그러나 발표용 plot의 학습 곡선 분산을 보이려면 multi-seed가 필요한데 sample_episodes가 seed=0 고정이라 진정한 multi-seed가 안 됨. 발표 시 "1 seed only"라는 한계 명시 필요.

### F3-6. **누락 위험: Curriculum forgetting mitigation이 너무 약함** — Severity: **MAJOR**

R3가 인지하나 mitigation (a) snapshot fallback (b) lr 절반 (c) mixed replay — 모두 사후/완화 조치. **사전 예방**(EWC, A-GEM, Reservoir replay) 없음. A16 = 70% 완주율은 임의 기준. F1Tenth에서 curriculum forgetting 실제 보고 사례 없으나 일반 RL에서 fine-tune은 forgetting이 흔함 (특히 reward landscape가 다른 트랙).

- **Fix**: replay buffer를 phase 전환 시 **삭제 안 하고 합치는** 방식 (Map Easy 데이터 50% + Oschersleben 50%) 사전 결정. 또는 curriculum 자체를 "공동 학습 (joint training with random map switching)"으로 대체.

### F3-7. **누락 위험: `_should_train._last` 카운터 비저장 (R7 인지하나 fix 모호)** — Severity: **MINOR**

R7 "카운터 state도 checkpoint에 포함하도록 train.py에 추가" — 구체 fix 부재. `Dreamer.__dict__`에 `_should_train`, `_should_log`, `_should_expl`, `_should_reset`, `_should_pretrain` 5개 모두 attribute. nn.Module이 아니라서 state_dict에 자동 안 들어감.

- **Fix**: train.py wrapper에 명시. `checkpoint['counters'] = {n: c._last for n, c in [...]}` 패턴.

### F3-8. **R8 fallback이 너무 늦음** — Severity: **MAJOR**

R8: "A10 직후 dry-run 학습 1K step으로 VRAM 측정". 그러나 batch_size=8 fallback은 train_ratio가 `batch_size·batch_length / train_ratio` 식으로 학습 빈도에 영향(`Every((8·64)/512) = Every(1)`) — 학습이 env step마다 발생 → wall-clock 2배 증가. 일정 영향 미고려.

---

## 4. Acceptance Criteria 검증 가능성

### F4-1. **A6 (`|vel_y| > 0.01`)이 base_classes 패치 *후*에만 의미가 있는데, 패치 전후 구분 없음** — Severity: **MAJOR**

A6은 코너링 중 단언이지만 base_classes.py:488이 하드코딩 0인 한 무조건 실패. Phase 1-3이 패치 단계인데 A6은 Phase 1 종료 기준에 포함. 즉 A6 통과 = Phase 1-3 완료. 시점 명시 필요.

- **Fix**: A6을 "Phase 1-3 패치 후"라 부연.

### F4-2. **A10 (10M~14M)의 ±20% 허용 범위는 12M 매핑 검증으로 약함** — Severity: **MINOR**

12M ± 2M는 hyperparameter 매핑 오차 절대 허용 범위가 너무 큼. 14M까지 허용하면 mlp_units, ConvEncoder1D 채널 등 어떤 조합이든 통과. 진정한 검증은 layer별 파라미터 비율.

- **Fix**: A10 보강 — `world_model.parameters() / total_params`로 RSSM·encoder·decoder·heads 비율을 합산해 논문 분포와 비교.

### F4-3. **A11 (eval lap_time 중앙값 ≤ 60초 in Map Easy) 근거 없음** — Severity: **MAJOR**

Map Easy 트랙 길이 정보 부재. analysis/001 §4: `map_easy3: resolution 0.02 m/px`만 적힘. 트랙 길이 ≠ 픽셀. 60초 기준은 어디서? GapFollower(`STRAIGHTS_SPEED=9.0`)로 Map Easy 완주 시간이 35초인지 60초인지 90초인지 알 수 없음. 기준 자체가 임의.

- **Fix**: Phase 1-1 dry-run에서 GapFollower baseline lap_time 측정 → A11 기준 = baseline × 1.5.
- Confidence: HIGH.

### F4-4. **A12 (2-lap 완주율 ≥ 80%) — "2-lap"인지 "lap" 단위인지 모호** — Severity: **MAJOR**

§1 #8: "Episode = 2-lap 완주" → 즉 episode 종료조건이 2-lap이라면 episode 완료율과 2-lap 완주율은 동의어. 그러나 §0-1 #2 "Oschersleben 완주 + lap time 최적화" — 1-lap만으로 완주 정의되는지 2-lap이 완주인지 명확하지 않음. A12와 §0-1 #2 사이의 정의 일관성 부재.

- **Fix**: "1-lap 완주 시 episode 종료 + 보너스, 2-lap 완주 시 추가 보너스" 또는 "2-lap이 표준 완주"로 단일 정의.

### F4-5. **A13 (best ≤ 110초)는 "best"라 1 episode만 통과해도 됨** — Severity: **MAJOR**

20 episode 중 1 episode만 110초 이내면 통과 — 운빨로 통과 가능. 평균/중앙값 기준이 더 적합.

- **Fix**: "median lap_time ≤ 120초 AND best ≤ 110초"로 이중 기준.

### F4-6. **A14 (자동 snapshot) — threshold 110초가 eval에서만 트리거되는데 eval 빈도가 `eval_every=1e4` 기본** — Severity: **MAJOR**

500K env step에 eval 50회 (1e4 주기). lap_time이 110초 이내로 떨어진 첫 eval에서만 snapshot, 그 이후 더 좋아져도 best 기준으로 갱신. **여러 100초대 정책 다양성**(LeWorldModel offline 데이터 다양성 목표)을 위해 더 자주 저장해야 할 수도. §0-1 #4 명시: "100초대 성능 policy 저장" — 단수일 가능성이 큰 정책.

- **Fix**: "best가 아니라 모든 110초 이내 정책 별도 저장" 또는 `eval_every` 축소(1e4 → 5e3).

### F4-7. **A15 (interval snapshot 100개, 16GB)** — VRAM/디스크 압박 — Severity: **MINOR**

500K step × eval_every=1e4 = 50 snapshot. 1M step 총 100 snapshot. 12M 모델 + Adam state ≈ 200MB → 100개 = 20GB. §7은 "총 ~16GB"라 적었으나 산출 안 맞음.

### F4-8. **A18 (후진 1초 누적) — 후진 측정 방식과 acceptance test의 분리** — Severity: **MINOR**

후진 카운터 reset 조건이 "dot ≥ 0이면 리셋"으로 0 근처(완전 정지)에서 카운터가 reset된 후 다시 후진하면 누적 시간이 무한히 누적될 수 있음. 그리고 강제 후진 1.1초 테스트에 trajectory time과 sim time 구분 명시 없음.

### F4-9. **A1, A2, A3, A4, A5 — 모두 wrapper 단위 테스트** — 측정 가능성 좋음. 단 A4의 `info['cause']` 키 일관성은 wrapper 구현 의존. — Severity: **MINOR**

`info['cause']`에 'collision', 'reverse', 'lap_complete', 'timeout' 모두 들어가야 한다는 정의가 없음.

---

## 5. 누락 항목

### F5-1. **seed 정책 부재** — Severity: **MAJOR**
- Multi-seed 학습 여부, env reset seed 정책, sample_episodes seed=0 우회 여부 모두 미명시. R9가 부분 언급.

### F5-2. **eval 프로토콜 명세 부재** — Severity: **MAJOR**
- `eval_episode_num`, 초기 pose randomization, 차량 spec default 강제, eval 시 noise 제거 여부, ROS 모드 평가 여부 모두 미명시.
- §0-3 "최종 평가는 default 값으로" 한 줄만 — 구체 절차 없음.

### F5-3. **logging stack 미명시** — Severity: **MAJOR**
- TensorBoard? wandb? jsonl만? §7에 "metrics.jsonl"만. 발표 차트 도구는 §11-3이 "Open Questions"로 미룸. configs.yaml `tools.Logger`는 TB+JSONL 둘 다.

### F5-4. **checkpoint 정책 일부만 명시** — Severity: **MAJOR**
- §3 "Snapshot 정책 Hybrid D" 있으나 `latest.pt` 외 carry-over 정책, fine-tune 시 `latest.pt` 강제 복사(§5-2)는 옵티마이저 state까지 복사하므로 lr 절반(R3) 적용 시 옵티마이저 momentum이 stale. 새 옵티마이저 시작 옵션 부재.

### F5-5. **reward shaping 세부 미확정** — Severity: **MAJOR**
- progress α=1.0 시작이지만 lap_complete +50, collision -10, reverse -10 — 스케일이 진짜 균형인지 검증 없음. F1Tenth lap 한 바퀴 progress 총합 ≈ 트랙 길이 100~300m → α=1.0이면 lap당 +100~300 reward 누적 vs lap_complete +50 → **lap_complete가 너무 약함**.
- **Fix**: 단순 산술 검증: "Oschersleben 트랙 길이 측정 → α 보정". A17이 component 분리 로깅을 요구하므로 사후 보정 가능하나 사전 검증 필요.

### F5-6. **action space 정규화 정의 모호** — Severity: **MAJOR**
- §1-2 "action: [-1, 1]^2 affine 변환". 그러나 `actor.dist='normal'` + `absmax=1.0`은 ContDist의 mode를 absmax로 클립 ([004 §1-6])하지만 sample은 std로 인해 범위 밖으로 나갈 수 있음. NormalizeActions wrapper는 [networks_f1tenth] 외부 적용 필요.

### F5-7. **observation normalization 정책 미명시** — Severity: **MAJOR**
- LiDAR raw range 0~30m. encoder 입력에 `symlog_inputs=True` (configs.yaml:46)가 적용되지만 ConvEncoder1D는 별도 — symlog 통과 여부 명시 없음. state vector도 vel_x [-5, 20], ang_vel [-?, ?] 등 스케일 다양. 정규화 미명시면 학습 불안정.
- **Fix**: ConvEncoder1D 입력 직전에 `symlog`적용 또는 `lidar / max_range`로 [0, 1] 정규화. configs에 명시.

### F5-8. **termination 조건 우선순위 부재** — Severity: **MAJOR**
- 충돌 + 시간초과 + lap완주 + 후진 — 동일 step에 다중 발생 시 우선순위? 예: lap완주와 timeout이 같은 step에 발생하면 reward는 +50 + 0인지 +50인지?

### F5-9. **prefill 정책 미명시** — Severity: **MAJOR**
- dreamer.py:251-281 prefill=2500 step 랜덤 정책 수집. F1Tenth 랜덤 정책은 거의 즉시 충돌 → prefill data가 충돌만 가득. world model 학습에 편향.
- **Fix**: prefill을 GapFollower(analysis/001 §1-4)로 1만 step 수집, 또는 prefill=0으로 끄고 expl_until 사용.

### F5-10. **Map Easy의 정의 — `map_easy3` vs `map_easy`** — Severity: **MAJOR**
- env_setting/001 §4: `map_easy.png`, `map_easy.yaml`. analysis/001 §4: `map_easy3` resolution 0.02. 계획서: `f1tenth_map_easy3`. 명명 일관성 부재 — `map_easy`는 1, 2, 3 버전이 다른가? 평가 트랙이 정확히 어느 것인가?

### F5-11. **centerline 추출 — Open Question (§11-1)에 던졌으나 Phase 4 시점 너무 늦음** — Severity: **MAJOR**
- 후진 가드(§1-4)는 Phase 1-4에서 centerline 사용. 즉 centerline은 Phase 1 시점에 필요. §11-1이 Phase 4로 미룬 것은 일정 오류.

### F5-12. **lap_times[0] vs 직접 측정 — Open Question (§11-2)** — Severity: **MINOR**
- env가 reset 시 `self.lap_times`를 0으로 reset (f110_env.py:529). 그러나 wrapper에서 episode 첫 step에서 어떻게 보일지 명시 없음.

### F5-13. **rollback 시나리오 부재** — Severity: **MAJOR**
- Phase 5-2 중간에 학습 발산하면? Phase 5-3 결과가 A16 미달이면 fallback은 어느 시점 weights인가? "snapshot D" 모호.

### F5-14. **dataset_size=1M 메모리 footprint** — Severity: **MAJOR**
- analysis/006 §2-6: dataset_size=1e6은 **메모리** 한도. LiDAR 1080 × float32 × 1M = 4.3GB RAM + obs/action/state 등 = 6~8GB RAM. 노트북(env_setting/004)은 RAM 명시 없음, 집컴 12-core CPU + RAM도 미명시. OOM 위험.
- **Fix**: dataset_size를 100K~200K로 축소하거나 LiDAR float16 저장.

### F5-15. **`compile=True` (configs 기본)와 1D Conv encoder 호환성 미검증** — Severity: **MAJOR**
- torch.compile + Conv1dSamePad의 동적 F.pad — TorchInductor가 동적 shape에서 fallback할 수 있음. 컴파일 실패 시 R 외 디버깅 비용. dreamer.py:46-50에서 `_wm`/`_task_behavior`을 compile.
- **Fix**: configs에 `compile: False` 명시 또는 ConvEncoder1D를 정적 padding으로 작성.

---

## What's Missing — 추가 누락

- LeWorldModel data contract pre-check를 Phase 1 종료에 두었으나(§10-3) "LeWorldModel이 요구하는 transition"이 무엇인지 자체가 불명. 별도 명세 없이 contract pre-check 불가능.
- 발표 데드라인 미명시. "발표 일정에 따라 Phase 5 budget 조정"(§8) — 학기 종료일 등 외부 데드라인이 anchor.
- 차량 spec default 강제 — 학습 시와 평가 시 동일한 차량 spec을 enforce하는 단위 테스트 부재.
- ConvEncoder1D 학습 첫 50K step 모니터링(R2) — 어떤 metric으로 발산을 감지? KL loss > 100? gradient norm? 정량 기준 없음.

---

## Ambiguity Risks

- §1-2 "obs_dict 구성"의 6번째 dim이 `progress_ratio` placeholder — **placeholder는 학습 input에 들어가서는 안 됨**. RSSM에 noise 입력. 어떻게 채울지 명시 필요.
- §1 #5 "stride-2 5 stage" → §2-1 "5 stage stride-2 conv1d" → A7 "5 stage". 그러나 출력 길이 33은 stride-2 5번 적용 결과의 정확치 (1080/2^5 = 33.75)인지 계산 검증 필요.
- §5-2 "cp ~/logdir/.../latest.pt"가 옵티마이저 state까지 복사 — fine-tune 시작 시 Adam momentum이 stale. lr 절반(R3)과 충돌. "fresh optimizer로 시작" 옵션 명시 필요.
- §3 #10 "Hybrid D"의 D가 무엇인지(`Hybrid` 정의가 본 문서에 없음, "D"는 인터뷰 결정 라벨로 추정).

---

## Multi-Perspective Notes

### EXECUTOR 관점
- Phase 1-2의 "F110GymnasiumWrapper 작성" — 가이드 코드 없이 2~3일은 빡빡. 특히 reverse_guard, centerline 추출, lap_count 방향 체크, `is_first/is_terminal/is_last` 키 추가, dict obs 구성 등 wrapper 책임 다수.
- F1-1 (image KeyError)로 Phase 2 dry-run 즉시 실패. 디버깅 1일 추가.
- networks.py 패치 line list 명시 없으면 어디를 고칠지 헷갈림.

### STAKEHOLDER 관점
- 평가 비중 60% = 알고리즘 발표. 발표 자료가 §8에서 1일만 할당, §6 "12M 모델 + 1D Conv LiDAR encoder + Progress reward + 순차 fine-tune" — **algorithm contribution이 약함**. 기존 컴포넌트 조합일 뿐 새로운 기여 없음. 발표 채점에 약한 카드.
- "LeWorldModel 데이터 수집용 정책 weights" 별도 보관 — LeWorldModel 명세 부재로 무엇을 수집해야 할지 불명.

### SKEPTIC 관점
- 강한 반대 논점: **DreamerV3는 F1Tenth 같은 빠른 시뮬레이션에 과한 알고리즘**. 1080-LiDAR + dense reward + closed-track 환경은 SAC, PPO로도 충분히 풀린다. 12M 모델 + world model 학습은 24시간 GPU 시간을 차지하지만, GapFollower 같은 classic algorithm이 60~80초대 lap을 즉시 달성한다 (analysis/001 §1-4). DreamerV3 선택은 **알고리즘 발표 60%**의 발표 임팩트를 위한 것이지 성능 최적 선택이 아님 — 이 점이 발표에서 들통날 수 있음.
- 또한 본 계획은 ralplan/critic 미적용 직접 초안(§0 헤더). 인터뷰 한 번에 13개 결정을 다 잡았는데 트레이드오프 분석 깊이 부족.

---

## Verdict Justification

**VERDICT: REJECT (REVISE 필요)**

ADVERSARIAL 모드 작동. CRITICAL 2개(F1-1, F2-1+F2-2), MAJOR 20+개, MINOR 다수. 단일 CRITICAL인 `preprocess` image 키 KeyError만으로도 Phase 2 dry-run 즉시 실패. 일정/VRAM 추정의 근거 부재가 학습 단계 진입 후 발견되면 1주일 이상 손실. 누락 항목(seed/eval/logging/centerline 시점/observation normalization/reward 스케일 산술)은 단순 누락이 아니라 architectural 결정이 필요한 사항이며 사후 결정 시 wrapper/configs/train.py 전부 재작업.

Realist Check:
- F1-1 (image KeyError): 실제 worst case = Phase 2 dry-run 첫 train step 즉시 실패, 디버깅 1일 이내 해결. **CRITICAL 유지** — 검증 없이 진행하면 일정 즉각 손실 + 모르고 넘어가서 wrapper에 dummy `image` 키 넣는 잘못된 우회 가능성.
- F2-1/F2-2 (일정/VRAM): 실제 worst case = Phase 5 진입 후 측정해서 batch_size 축소 → wall-clock 2배. **CRITICAL 유지** — Phase 2 dry-run 단계에서 측정 필수.
- F3-3 (reward hacking 후진+lap): 실제 worst case = 학습 진행 안 되거나 이상한 정책 수렴. **MAJOR 유지**.
- F5-9 (prefill): random policy로 prefill하면 즉시 충돌 데이터만 — world model이 충돌 직전 분포만 학습. **MAJOR 유지**.

ADVERSARIAL 모드 활성 사유: F1-1 CRITICAL 1개 + 즉시 MAJOR 5개 이상 → 시스템적 검증 결손 패턴.

**REVISE 요건**: 아래 Top 5를 해결한 v2 plan 제출. 그 외 MAJOR 항목은 v2에서 Open Question 또는 Phase 진입 시 결정으로 분류 가능.

---

## Open Questions (unscored)

- DreamerV3 vs PPO/SAC의 wall-clock 우월성 — analysis/002에서 다루지 않음. 발표 임팩트가 우선이면 DreamerV3 유지, 시간 우선이면 SAC + custom encoder가 합리적.
- NM512 README/issues에 12M preset 공식 명시 여부 — 직접 확인 안 함.
- f110-v0의 `lap_times` 시작값이 episode reset 시 항상 0인지 검증 필요.
- centerline tangent 기반 후진 가드의 GJK 충돌 처리와 race condition.

---

## 총평

본 계획은 **인터뷰 결정 13건이 잡힌 직접 초안**이며 critic·consensus 미적용을 자인합니다(§0 헤더). 그 결과로 (a) dreamerv3-torch 코드 자체의 핵심 가드(image 키 하드코딩) 미반영, (b) GPU/VRAM/일정 추정 무근거, (c) 핵심 누락 결정사항(seed/eval/observation normalization/centerline 시점/prefill 정책)이 §11 Open Questions로 밀려 Phase 진입 후 결정 — risk 누적. 강점은 (1) acceptance criteria 18개로 비교적 측정 가능한 구조, (2) Risk 14개의 구체성, (3) Phase 분할이 적당. 그러나 강점이 약점을 상쇄하지 않습니다.

**현재 상태로 Phase 1 착수 시 Phase 2 dry-run에서 image KeyError로 즉시 막힙니다. v2 plan 필수.**

---

## Top 5 우선 수정사항

1. **F1-1 — `WorldModel.preprocess`의 image 키 KeyError 해결책 명시.** `preprocess`를 fork하거나 `WorldModel` 서브클래스 또는 monkey-patch 명시. `video_pred_log=False`만으로 부족함을 plan에 명문화. (Phase 2 dry-run blocker)

2. **F2-1 + F2-2 — Phase 2 dry-run에 wall-clock + VRAM 측정 acceptance 추가.** GPU SKU(4060Ti 8GB vs 16GB) 환경설정에서 확정. A19 신설: "1K env step + 100 train step wall-clock 측정 → 500K 추정치 < 24h. VRAM peak 측정 < 가용 메모리의 80%." 미달 시 batch_size/dataset_size/precision 조정 사전 결정.

3. **F1-3 + F5-7 — ConvEncoder1D outdim 축소 + observation normalization 정책 명시.** `outdim ≈ 512`로 projection 추가. LiDAR `/max_range` 또는 `symlog` 정규화 정책을 configs에 명시. ConvDecoder1D 구현 별도 acceptance(A7b).

4. **F3-3 + F3-4 + F5-5 — Reward 함수 재설계.** lap_count 방향 체크(centerline tangent dot velocity > 0인 통과만 카운트), progress reward에 step-wise cap, lap_complete reward 스케일을 트랙 길이 기반 산술 검증. Phase 4를 Phase 1 직후로 앞당기고 reward 단위 테스트 추가.

5. **F5-2 + F5-1 + F5-3 + F5-4 + F5-6 + F5-9 + F5-10 — §11 Open Questions 대폭 축소.** 다음을 plan v2에서 결정 사항(§1 결정 표)으로 격상:
   - seed 정책 (single seed=0 vs multi-seed; sample_episodes seed=0 우회 여부)
   - eval 프로토콜 (`eval_episode_num`, pose randomization, noise 강제 default)
   - logging stack (TB + jsonl 또는 wandb 결정)
   - checkpoint 정책의 optimizer state carry-over 정책(특히 fine-tune 시작 시 fresh vs warm)
   - action wrapper 체인 정확한 순서 명시 + NormalizeActions 위치
   - prefill 정책 (random → GapFollower 또는 prefill=0)
   - `map_easy` vs `map_easy3` 명명 통일 및 평가 대상 트랙 파일명 확정

---

**관련 파일** (절대 경로):
- /home/dlacksdn/f1tenth_RL_project/_thinking/planning/001-f1tenth_dreamerV3_version1.md
- /home/dlacksdn/dreamerv3-torch/models.py (L177-192 preprocess image 키)
- /home/dlacksdn/dreamerv3-torch/networks.py (L293-357 MultiEncoder 라우팅; L771-798 Conv2dSamePad)
- /home/dlacksdn/dreamerv3-torch/configs.yaml (L33-60 RSSM/encoder/decoder 기본값)
- /home/dlacksdn/f1tenth_RL_project/gym/f110_gym/envs/base_classes.py (L440-490 obs 구성; L488 vel_y=0 하드코딩)
- /home/dlacksdn/f1tenth_RL_project/gym/f110_gym/envs/f110_env.py (L529-690 lap_times/check_done/reward)
- /home/dlacksdn/f1tenth_RL_project/_thinking/analysis/001-env-analysis.md (§2 구조적 이슈)
- /home/dlacksdn/f1tenth_RL_project/_thinking/analysis/005-dreamer_code_analysis_part3.md (§1-4 image 키 경고)
- /home/dlacksdn/f1tenth_RL_project/_thinking/analysis/006-dreamer_code_analysis_part4.md (§2-2 sample_episodes seed=0; §2-6 dataset_size 메모리/디스크)
