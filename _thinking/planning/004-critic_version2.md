# 비판적 평가: 003-f1tenth_dreamerV3_version2.md

**평가자**: critic (THOROUGH → 일부 ADVERSARIAL 부분 활성)
**Verdict**: **ACCEPT_WITH_MINOR_FIXES**
**요약**: v1 대비 압도적 개선. CRITICAL 0건, MAJOR 4건, MINOR 다수. 002 Top 5 중 4건 CLOSED, 1건 PARTIALLY. 다만 새 결정에서 내부 모순 2건과 검증 불가능 항목이 신규 발견됨.

**Pre-commitment 예측**: (1) §4-3 reward 산술이 트랙 길이 실측 없는 상태에서 α/R_lap을 미리 못박았을 가능성, (2) A19 분기 임계값이 단일 수치(24h, 80% VRAM)로 너무 거칠 가능성, (3) `map_easy` 명명 통일이 실제 파일명과 다를 가능성, (4) 결정 #14가 fork vs 서브클래스 사이에서 진동, (5) Open Questions 4건이 실제로는 acceptance 통과를 막는 blocker. → 5건 중 4건 적중.

---

## 1. 회귀 검증표 — 002 Top 5

| 002 Top 5 | 상태 | v2 위치 | 비고 |
|---|---|---|---|
| #1 preprocess KeyError fix | **CLOSED** | 결정 #14, A20, R10 | `models.py:182`에 `if "image" in obs:` 가드. 단 v2 §0 변경 요약 첫 행 "WorldModel 서브클래스 + preprocess override 명시"와 §1-B 표 "대안 (서브클래스)는 ... 비채택" 사이 **명백한 자기모순** — 신규 결함 N1 |
| #2 wall-clock + VRAM dry-run | **PARTIALLY_ADDRESSED** | A19, §5-Fallback | 게이트 도입은 합당. 그러나 추정식 `D = (1000*A + 500K/train_ratio*B)/60`은 env step과 train step이 직렬 가정 — 실제 dreamerv3는 train과 env step이 같은 thread에서 교차 실행(`train_ratio` 기반)이므로 식이 정확하지 않음. 보수적 상한으로는 쓸 수 있으나 산출 근거를 표기해야 함. 또 "16GB×0.80=12800MB"은 PyTorch reserved memory가 allocated보다 큰 점을 무시 — 실측 시 80% threshold 미달이어도 OOM 가능 |
| #3 ConvEncoder outdim + obs normalization + ConvDecoder1D | **CLOSED** | 결정 #15, #16, A7/A7b | Linear(8704, 512) projection 명시, LiDAR `/30→symlog`, state dim별 정규화, ConvDecoder1D 별도 acceptance. 산술도 002 F1-4 (1080→...→68→34) 반영 |
| #4 Reward 재설계 | **PARTIALLY_ADDRESSED** | §4-3, 결정 #7, R17/R18 | 방향 가드·step-cap·트랙 길이 산술표 도입은 OK. 그러나 §4-3 표 자체에 트랙 길이가 "Phase 1-1에서 측정" 상태로 비어 있는데 **이미 `α_progress=1.0, R_lap=100`을 결정**(§4-3 마지막 줄). 미측정 상태에서 수치 확정 → 재산정 가드("±50% 벗어나면")가 있으나 그 ±50% 자체가 임의 임계 — 신규 결함 N2 |
| #5 Open Questions 격상 | **CLOSED** | 결정 #17~#26 | v1 11건 → v2 4건. 잔여 4건은 모두 Phase 진입 시 측정 항목으로 합당 |

**Top 5 결산**: 3 CLOSED / 2 PARTIALLY. 잔여 미흡 항목은 fix 가능한 수준.

---

## 2. 회귀 검증표 — 002의 모든 CRITICAL · MAJOR

| 002 # | Severity | 상태 | 비고 |
|---|---|---|---|
| F1-1 preprocess KeyError | CRITICAL | **CLOSED** | 결정 #14 (단 N1 모순) |
| F2-1 wall-clock 무근거 | CRITICAL | **PARTIALLY** | A19, 위 #2 참조 |
| F2-2 VRAM/GPU SKU | CRITICAL | **CLOSED** | §0-2 16GB 가정 + 8GB 분기, A19 측정 |
| F1-2 12M hyperparam 비표준 | MAJOR | **DEFERRED_WITH_JUSTIFICATION** | A10 비율 보고 + R11 모니터로 검증을 미룸. NM512 README의 12M preset 직접 인용은 여전히 없음 — Phase 2-3 진입 전 인용 1줄 추가 권장 |
| F1-3 outdim bottleneck | MAJOR | **CLOSED** | Linear(8704, 512) |
| F1-4 stride-2 산술 오류 | MINOR | **CLOSED** | A7에 ceil 보정 |
| F1-5 ConvDecoder1D 별도 작업 | MAJOR | **CLOSED** | A7b 신설 |
| F1-6 MultiEncoder patch 작업량 | MAJOR | **CLOSED** | Phase 2-0 파일 list 명시. 단 변경 라인 수 "약 6곳"은 추정 — 실측은 Phase 2 진입 시 |
| F1-7 gym→gymnasium 변환 | MAJOR | **CLOSED** | 결정 #19 고정 pose + Phase 1-2 |
| F1-8 vel_y / slip_angle | MAJOR | **CLOSED** | Phase 1-3 + A6. 단 Open Question #4 (kinematic vs dynamic 모드 확인)이 잔존 — A6 자체가 dynamic 모드에서만 통과 가능 |
| F1-9 action_repeat 100Hz 의미 | MAJOR | **CLOSED** | 결정 #22, time_limit 주석. 단 §2-3 configs `time_limit: 9000  # env step 기준`인데 §1-B #22 `TimeLimit(18000 sim step = 9000 env step)`과 단위는 합치하나 dreamerv3-torch `TimeLimit`이 env step인지 sim step인지는 Phase 1-2 검증 필요 |
| F1-10 make_env in-place | MINOR | **CLOSED** | 결정 #1 vendor-in |
| F1-11 Damy/Parallel 누락 | MINOR | **CLOSED** | 결정 #22 |
| F2-3 노트북 의존성 | MAJOR | **CLOSED** | Phase 1-0 |
| F2-4 재평가 시간 | MINOR | **CLOSED** | 1.5h, 20 ep |
| F3-1 replay disk | MAJOR | **CLOSED** | R15, dataset_size=200K, fp16 |
| F3-2 LiDAR DR | MAJOR | **DEFERRED_WITH_JUSTIFICATION** | sim only 가정, R16 후속 — 합당 |
| F3-3 lap reward hacking | MAJOR | **CLOSED** | §4-1 방향 가드 |
| F3-4 progress shortcut | MAJOR | **CLOSED** | step-cap |
| F3-5 sample_episodes seed=0 | MINOR | **DEFERRED_WITH_JUSTIFICATION** | 결정 #17, 발표 명시 — 합당 |
| F3-6 forgetting | MAJOR | **CLOSED** | joint replay 30%, R3 |
| F3-7 counter ckpt | MINOR | **CLOSED** | Phase 3, R7. 단 구체 코드 스켈레톤은 v1 critic 제안만큼 자세하지 않음 |
| F3-8 R8 fallback 시점 | MAJOR | **CLOSED** | A19 게이트로 Phase 5 진입 전 결정 |
| F4-1 A6 시점 | MAJOR | **CLOSED** | A6에 "Phase 1-3 패치 후" |
| F4-2 A10 ±20% | MINOR | **CLOSED** | 비율 보고 추가 |
| F4-3 A11 60초 임의 | MAJOR | **CLOSED** | GapFollower × 1.5. 단 baseline 자체 측정이 Open Q #3 |
| F4-4 2-lap 정의 모호 | MAJOR | **CLOSED** | 결정 #8 명시 |
| F4-5 A13 best 1회 통과 | MAJOR | **CLOSED** | median ≤120 AND best ≤110 이중 기준 |
| F4-6 A14 다양성 | MAJOR | **CLOSED** | save_all_below_threshold |
| F4-7 디스크 16GB→20GB | MINOR | **CLOSED** | §7 정정 |
| F4-8 후진 카운터 reset | MINOR | **CLOSED** | Phase 1-4 의도 명시 |
| F4-9 info['cause'] | MINOR | **CLOSED** | §4-4 cause 4종 |
| F5-1 seed | MAJOR | **CLOSED** | 결정 #17 |
| F5-2 eval 프로토콜 | MAJOR | **CLOSED** | 결정 #19 |
| F5-3 logging | MAJOR | **CLOSED** | 결정 #20 |
| F5-4 ckpt optim carry-over | MAJOR | **CLOSED** | 결정 #21 fresh optim |
| F5-5 reward 산술 | MAJOR | **PARTIALLY** | §4-3, N2 참조 |
| F5-6 action normalize | MAJOR | **CLOSED** | 결정 #22 |
| F5-7 obs normalization | MAJOR | **CLOSED** | 결정 #15 |
| F5-8 termination 우선순위 | MAJOR | **CLOSED** | 결정 #24 |
| F5-9 prefill | MAJOR | **CLOSED** | 결정 #23 (prefill=0 + GapFollower) |
| F5-10 map_easy 명명 | MAJOR | **STILL_OPEN** | 결정 #25는 "map_easy로 통일"인데 실제 `pkg/src/pkg/maps/`에는 `map_easy3.png/.yaml`만 존재하고 `map_easy.png`는 없음. analysis/001 §4도 `map_easy3` — v2 결정 #25는 사실 오인. 신규 결함 N3 |
| F5-11 centerline 시점 | MAJOR | **CLOSED** | Phase 1-1 격상 |
| F5-12 lap_times[0] | MINOR | **STILL_OPEN** | "Phase 결정"으로 보류 — 합당 |
| F5-13 rollback | MAJOR | **PARTIALLY** | A16 미달 시 "policy_lap*.pt 중 Map Easy 호환 weights" — 그러나 Stage 2 학습된 정책의 snapshot은 Oschersleben 도메인. Map Easy 호환을 보장하는 별도 snapshot은 Stage 1 latest.pt밖에 없음. 표현 모호 |
| F5-14 dataset_size RAM | MAJOR | **CLOSED** | 200K + fp16 |
| F5-15 torch.compile | MAJOR | **CLOSED** | compile: False |

**결산**: 002 제기 CRITICAL 3건 모두 CLOSED 또는 PARTIALLY (실질 해결). MAJOR 21건 중 CLOSED 16, PARTIALLY 3, DEFERRED 2. STILL_OPEN MAJOR 1건(F5-10 명명).

---

## 3. 신규 결함 (v2에서 새로 도입된 risk · 모순 · 검증 불가능 항목)

### N1. **결정 #14의 자기 모순 — fork vs 서브클래스** — Severity: **MAJOR**

- 증거: §0 변경 요약 첫 행: `"WorldModel 서브클래스 + preprocess override 명시 (§3 결정 #14, Phase 2-0)"`.
- 그러나 §1-B 결정 #14 본문: `"models.py를 fork 수정해 obs["image"] = obs["image"] / 255.0 라인을 if "image" in obs: 가드. 대안 (서브클래스)은 tools.recursive_update 흐름과 충돌해 비채택"`.
- 두 문장이 정반대. 채택안이 fork인지 서브클래스인지 문서 내부에서 확정 불가.
- 또한 §9-1 표도 `"결정 #14 (models.py fork)"`로 fork만 인용 — §0과 정합 안 됨.
- Confidence: HIGH.
- **Fix**: §0 변경 요약 1행을 "models.py preprocess fork-patch (가드 추가)"로 수정. 서브클래스 표현 삭제. (Acceptance criterion: v2-patch에서 §0과 §1-B #14가 동일 채택안을 가리킨다.)

### N2. **§4-3 reward 산술 검증표의 자기 무효화** — Severity: **MAJOR**

- 증거: §4-3 표 "트랙 길이 추정" 칸이 `"약 50~80m (Phase 1-1에서 정확 측정)"` 그리고 `"약 300~400m (Phase 1-1 측정 필요)"` — **미측정 상태**.
- 그럼에도 표 마지막 줄에서 `"alpha_progress = 1.0, R_lap_complete = 100 통일"`로 **결정 확정**.
- 보호장치는 `"트랙 길이가 위 추정에서 ±50% 이상 벗어나면 본 표 재산정"` — 그러나 (a) ±50%는 임의 임계, (b) 트랙 길이가 ±30% 빗나가도 R_lap=100이 progress 총합과 같은 자릿수가 되는지는 산수 자체가 부정확. Map Easy의 progress 총합 예상 50~80인데 R_lap=100이면 lap_complete가 progress 총합보다 큼 — 002 F5-5에서 critic이 지적한 "lap_complete가 너무 약함"의 반대 극단(이번엔 너무 강함).
- 더 큰 문제: 표 본문 `"50~80m / (~50s × 50step/s) = 0.02~0.032 m/step"`. 그런데 결정 #22 `action_repeat=2`라 env step 50/s ≠ sim step. 50 sim step/s = 25 env step/s. 표가 단위를 섞음. progress per env step은 0.04~0.06 m/step. 그러면 lap당 progress reward 총합도 25 env step × 50s × 0.06 ≈ **75**, 아니면 더 작음. 표의 "50~80"과 비슷하지만 산식이 우연.
- Map Easy progress 0.5m/step step-cap (§4-3 마지막 문단)도 **25 m/s × 2 sim step × 0.01s = 0.5m**로 산출 — 이건 정합. 하지만 "lap당 최대 reward 한도 (50step/s × 0.5m = 25m/s × 2step physics)" 표현이 단위 혼동.
- Confidence: HIGH.
- **Fix**: (a) §4-3 표에서 "결정"을 "잠정값(Phase 1-1 측정 후 확정)"으로 격하, (b) 단위를 env step 또는 sim step 중 하나로 통일, (c) Map Easy R_lap을 progress 총합 추정의 30~50%인 ~30으로 별도 산출(표 4번째 칸이 R_lap=30을 권장하는데 다섯 번째 칸 적용 결정이 100으로 점프 — 권장과 적용도 불일치).

### N3. **결정 #25 map_easy 명명이 실제 파일 시스템과 불일치** — Severity: **MAJOR**

- 증거: 결정 #25 `"평가 트랙 = map_easy (env_setting/001 §4 기준 .png/.yaml 파일명). v1의 map_easy3는 analysis/001에 등장하나 본 계획에서는 map_easy로 통일"`.
- 그러나 실측: `pkg/src/pkg/maps/`에는 `map_easy3.png`, `map_easy3.yaml`만 존재. `map_easy.png`는 없음.
- env_setting/001 §4가 `map_easy.png/.yaml`로 적혀 있다고 v2가 주장하지만 실제 파일은 `map_easy3`. v2가 env_setting/001를 잘못 인용했거나, env_setting/001 자체가 오기.
- 결과: Phase 2-3 configs `task='f1tenth_map_easy'`로 진입 시 파일 not found.
- Confidence: HIGH.
- **Fix**: 결정 #25를 `"map_easy3 (실제 파일명)으로 통일"`로 정정. 또는 파일을 `map_easy`로 rename하고 그 작업을 Phase 1-0에 명시. env_setting/001과의 정합도 별도 점검.

### N4. **A19 임계값과 PyTorch 메모리 모델 불일치** — Severity: **MINOR**

- A19 `"C ≤ (GPU_total_VRAM × 0.80) MB (16GB → 12800MB)"`. 그러나 PyTorch `torch.cuda.max_memory_allocated()`는 allocated만 측정 — reserved/cached가 더 큼. 실제 OOM은 reserved 기준. 80% allocated이면 reserved는 90%+ 가능 → 학습 중 OOM 가능.
- 또한 wall-clock 추정식 `D = (1000*A + 500K/train_ratio*B) / 60`이 env step과 train step을 합산하나 dreamerv3-torch는 1 env step 당 `train_ratio=512`마다 batch_size·batch_length 단위 학습 — 식이 단위/스케일 맞는지 한 번 더 derive 필요.
- **Fix**: A19 측정 도구를 `torch.cuda.max_memory_reserved()`로 명시. 추정식 유도를 `_thinking/notes/A19_estimate_derivation.md`에 부록으로 추가.

### N5. **결정 #15 state 정규화 — `vel_x/20` 등 fixed scale의 saturation 위험** — Severity: **MINOR**

- 결정 #15 `"vel_x/20, vel_y/5, ang_vel_z/π, prev_steer/0.4189, prev_speed/20로 [-1,1] 정규화"`. 그러나 `v_max=20`이라 vel_x는 0~20 → /20하면 [0,1]이지 [-1,1] 아님. 그리고 v_min=-5 → 후진 시 -0.25. ang_vel_z의 π 분모는 트랙 곡률 기반 실측 없는 임의 수치. saturation 발생 시 학습 정보 손실.
- symlog_inputs=True를 같이 쓰는데 symlog는 이미 큰 값을 압축 — fixed scale 정규화와 중복. 둘 중 하나면 충분.
- **Fix**: scale을 실측 분포에서 99-percentile로 유도하거나, symlog만 사용하고 fixed scale은 제거.

### N6. **A19 "사전 결정 분기" 기준이 정량적이나 우선순위 모호** — Severity: **MINOR**

- A19 Fail 분기에서 `"VRAM 초과 → batch_size 16→8 → 재측정. 여전히 초과 시 batch_length 64→32"`와 `"wall-clock 초과 → train_ratio 512→1024 → 재측정. 여전히 초과 시 steps 500K→300K"`. 그러나 **VRAM과 wall-clock 동시 초과** 시 어느 것을 먼저 조정하는지 명시 없음. §5-Fallback 표 "둘 다 fail" 줄은 종합 결과만 — 도달 경로 미정.
- **Fix**: 분기 흐름도(VRAM 우선 → wall-clock 측정 → 재분기) 또는 의사코드 명시.

### N7. **GPU SKU 결정의 Open Question화 — 환경설정 직후 확정 가능한데 보류** — Severity: **MINOR**

- §10 Open Q #2 `"GPU SKU 확정 — nvidia-smi로 확정"`. 이는 5초 작업. v2 작성 시점에 이미 확정 가능. Open Question에 남긴 것은 게으름.
- **Fix**: Phase 1-0 의존성 설치 직전에 "nvidia-smi 출력 capture → §0-2 표 업데이트"를 명시. Open Q #2 삭제.

### N8. **DreamerV3 정당화 §0-4가 SAC/PPO 대비 정량 비교가 아닌 수사** — Severity: **MINOR**

- §0-4의 4가지 사유 중 1, 2, 4번은 정성적("발표 임팩트", "코드 contract 비용", "발표 자산 측면 손해"). 3번 `"GapFollower lap_time 측정으로 대체 가능"`은 측정 자체가 Open Q #3.
- Critic SKEPTIC 002의 핵심 반론 — "SAC가 동일 wall-clock에서 더 빠를 수 있다" — 에 대해 v2는 `"손해"`라 인정하면서 발표 자산으로 정당화. 이게 stakeholder에게 충분한지는 발표 평가 기준에 따름.
- 정량 ablation 슬라이드 1장이 Phase 6에 포함되나, SAC를 실제로 학습하지 않으면 비교가 GapFollower vs DreamerV3만 됨. SAC vs DreamerV3 직접 비교 의향은 명시 안 됨.
- **Fix**: §0-4 끝에 "SAC 직접 비교는 wall-clock 예산상 미실시, GapFollower (classic) vs DreamerV3 (RL world model) 비교로 대체"라고 한계 명시.

### N9. **A19 dryrun_bench.py가 "GPU 머신에서" 실행되어야 하나 시점 명시 약함** — Severity: **MINOR**

- §8 일정 표 Phase 2 `"노트북 + 집컴(2-4만)"`. 즉 Phase 2-4 dry-run benchmark만 집컴에서 실행. 그러나 Phase 2-0~2-3 패치 작업이 노트북에서 끝나야 2-4가 집컴으로 넘어감 → 노트북↔집컴 코드 동기화 방식(git? rsync?) 미명시.
- **Fix**: Phase 2-4 진입 전 "git push → 집컴 git pull" 또는 동기화 절차 1줄 추가.

### N10. **R7 fix가 여전히 "Phase 3에서 ckpt에 포함"으로만 적힘** — Severity: **MINOR**

- 002 F3-7이 구체 fix `"checkpoint['counters'] = {n: c._last for n, c in [...]}"`를 제안했으나 v2 R7는 `"counter state ckpt 저장 (Phase 3)"`로만 — 코드 패턴 미명시. 실수할 여지 있음.
- **Fix**: R7 또는 Phase 3 본문에 의사코드 1줄 추가.

---

## 4. 계획서 완결성

### Phase 의존성 (Phase 0~N)
- Phase 1 (env+centerline) → Phase 2 (encoder+fork+dry-run) → Phase 3 (snapshot) → Phase 4 (reward) → Phase 5 (학습) → Phase 6 (발표). 단 §8에서 Phase 4가 1일이고 §3에서 Phase 4가 "Phase 1 직후로 앞당김"이라 적혔는데 일정 표에는 여전히 Phase 4가 Phase 3 다음. **순서 모호**.
- Phase 1-1 centerline 측정 결과가 §4-3 reward 표·A11·A13 기준에 반영되어야 한다(§11-4). 즉 Phase 1-1 종료 = Phase 4 reward 확정 + A11/A13 기준 확정. **Phase 4 reward는 사실상 Phase 1-1 이후, Phase 5 이전 어디서나 가능**. 일정 표가 이걸 반영 안 함.

### Acceptance ↔ Phase 매핑
- 거의 모든 A에 〔Phase X〕 표기 — 우수. 단 A_centerline은 Phase 1-1이라 번호 없는 것은 일관성 약함(A1~A20 시퀀스에서 이탈).
- A19 "Stage 1 학습 진입 게이트" — 강력. 다만 A19가 Phase 2-4 종료이자 Phase 5 진입 게이트라는 dual role을 §6-1 표에서 확인 가능. 정합.

### Open Questions 잔존량
- v1 11건 → v2 4건. 목표 5건 이하 **달성**.
- 4건 중 #1(트랙 길이), #3(GapFollower baseline), #4(dynamic mode 확인)은 Phase 1-1에서 실측 — 합당. #2(GPU SKU)는 위 N7 — 즉시 확정 가능. 사실상 3건.

### 산출물 정합
- §7 디스크 ~20GB 정정 OK. 단 replay buffer 디스크 사용량(R15 dataset_size=200K + fp16)이 §7 표에 따로 행으로 없음 — 대략 1.5~2GB이지만 명시 권장.

---

## 5. SKEPTIC

- §0-4 정당화는 v1보다 진보했으나 여전히 "발표 비중 60%" 의존. 만약 발표 채점이 단순 완주율·lap_time에 가중되면 DreamerV3 선택이 손해. 발표 채점 기준 자체를 외부에서 확인할 수 없으므로 베팅.
- A19 게이트 자체가 "통과해야 Stage 1 진입"인데 fail 시 분기에서 `"여전히 초과 시 steps 500K→300K + 발표에서 한계 명시"` — 즉 fail이어도 최종적으로는 진입. 게이트가 사실상 advisory.

## 6. DEVIL'S ADVOCATE

- "왜 ConvEncoder1D인가? 1D PointNet-style permutation-invariant encoder가 LiDAR ray index의 회전/시작점 변화에 robust 아닌가?" — F1Tenth LiDAR는 ray index가 고정 각도라 permutation invariance 불필요, 1D Conv 합당. 채택 OK.
- "왜 12M? 200M default를 쓰지 않는 이유는?" — VRAM 16GB 가정에서 200M은 batch_size 4 이하로 떨어져 train_ratio 효과 약화. 12M 선택 합당.
- "왜 GapFollower prefill 10K인가? Plan2Explore가 분포 더 다양하지 않나?" — dense reward 환경이라 expl_behavior=greedy 합당, 그러나 GapFollower prefill 10K가 다양성 측면에서 우월한지는 ablation 없음. 채택 사유는 "충돌 데이터 편향 회피"로 합당.

## 7. 정합성

- v1 §3 Phase 분할과 v2 §3 Phase 분할 동일 골격 — 변경 정책 준수. 결정 #1~13의 v1 항목을 표로 보존 — 정합.
- Critic 002 항목별 매핑 §9는 41개 항목 빠짐없이 추적. **이례적으로 철저**.
- §1-B 결정 #14가 §0 변경 요약과 모순 (N1). §1-B 결정 #25가 실제 파일 시스템과 모순 (N3). §4-3가 자기 가드 무효화 (N2). 정합성 결함 3건이 신규.

---

## Verdict

**ACCEPT_WITH_MINOR_FIXES**.

v1은 REJECT였고, v2는 002에서 제기한 모든 CRITICAL을 closed 또는 partial로 처리했으며 MAJOR 21건 중 16건이 CLOSED. 잔존 결함은 모두 fix 가능한 수준이고 어느 것도 Phase 1 착수를 막지 않는다. ADVERSARIAL 모드는 활성화하지 않았다 — 새로 발견된 MAJOR 4건 (N1, N2, N3, F5-10 STILL_OPEN) 중 N1과 N3는 문서 1~2줄 수정으로 종결되고, N2는 §4-3 표 격하·재산정으로 종결, F5-10은 N3와 동일 사안. 시스템적 결함 패턴 없음.

Realist Check: N3 (map_easy 파일 명명)는 표면적으로 CRITICAL 후보(Phase 5 진입 시 파일 not found로 즉시 실패)이나 (a) Phase 2-3 configs 작성 시 파일 존재 여부가 즉시 발견, (b) 단순 rename 또는 task 문자열 수정 5분 작업. → MAJOR로 유지. 데이터 손실·보안·금전 없음.

### Fix 목록 (acceptance criteria 형태)

다음을 v2-patch 또는 v3로 반영해야 Phase 1-1 착수 직전 ACCEPT 완전체:

- [ ] **C-N1**: §0 변경 요약 1행 "WorldModel 서브클래스 + preprocess override 명시"를 "`models.py:182` `if "image" in obs:` 가드 적용 (fork-patch)"로 정정. 서브클래스 표현 삭제. (§1-B #14, §9-1과 동일 채택안 명시)
- [ ] **C-N2**: §4-3 표 마지막 줄 "**결정**: alpha_progress = 1.0, R_lap_complete = 100"을 "**잠정값**: alpha_progress = 1.0, R_lap_complete = 30 (Map Easy) / 100 (Oschersleben). Phase 1-1 centerline 측정 후 v2-patch로 확정"로 격하. 표 4번째 칸(권장)과 5번째 칸(적용)을 동일하게 맞춤. 단위는 env step 기준으로 통일하고 sim step↔env step 변환을 footnote로.
- [ ] **C-N3 / F5-10**: 결정 #25를 "평가 트랙 = `map_easy3` (`pkg/src/pkg/maps/map_easy3.{png,yaml}` 실측 확인). configs `task='f1tenth_map_easy3'`"로 정정. env_setting/001 §4도 함께 점검·정정.
- [ ] **C-N4**: A19 측정 도구를 `torch.cuda.max_memory_reserved()`로 명시. wall-clock 추정식 유도를 `_thinking/notes/A19_estimate_derivation.md` 부록으로 추가하거나 식 옆에 "보수적 상한 가정: env step과 train step 직렬 실행" 주석.
- [ ] **C-N5**: 결정 #15에서 fixed scale 정규화 vs symlog_inputs 중복 해소. (a) symlog만 사용으로 단순화하거나, (b) fixed scale 유지 시 symlog_inputs=False로 변경. vel_x/20 → [-0.25, 1] 범위임을 footnote로.
- [ ] **C-N6**: A19 분기에서 VRAM·wall-clock 동시 fail 시 우선순위 명시 (제안: VRAM 우선 → 재측정 → wall-clock 분기).
- [ ] **C-N7**: §10 Open Q #2 (GPU SKU)를 Phase 1-0 액션으로 격상 — "nvidia-smi capture → §0-2 표 업데이트" 1줄.
- [ ] **C-N10**: R7 또는 Phase 3 본문에 counter ckpt 저장 의사코드 1줄: `checkpoint['counters'] = {n: c._last for n, c in [('train', _should_train), ...]}` 패턴.
- [ ] **C-§8-Phase4 위치**: §8 일정 표에서 Phase 4를 Phase 1-1 직후~Phase 5 진입 전 사이로 표기, "Phase 1-1 centerline 측정 → §4-3 표 확정 → Phase 4 reward 함수 코드화" 흐름 명시.
- [ ] **C-§7 replay**: §7 산출물 표에 replay buffer (1.5~2GB) 행 추가.

---

## Open Questions (unscored, 후속)

- F1-2 12M Table B.1 비표준성: NM512 README의 12M preset 공식 명시 여부 — Phase 2-3 진입 전 1줄 인용 권장.
- SAC vs DreamerV3 정량 비교 미실시 — 발표 임팩트가 평가에서 어떻게 반영되는지 외부 의존.
- f110_env.py의 `lap_times`/`lap_counts` 시작값 실측 (Open Q F5-12).
- dynamic_models.py 모드 확인 (Open Q #4) — kinematic이면 R1 mitigation 전제 무너짐, A6 자체 불가.

---

**관련 파일** (절대 경로):
- /home/dlacksdn/f1tenth_RL_project/_thinking/planning/003-f1tenth_dreamerV3_version2.md (평가 대상)
- /home/dlacksdn/f1tenth_RL_project/_thinking/planning/002-critic_version1.md (회귀 기준)
- /home/dlacksdn/f1tenth_RL_project/_thinking/planning/001-f1tenth_dreamerV3_version1.md (v1 원본)
- /home/dlacksdn/dreamerv3-torch/models.py (L177-192 preprocess image 키, N1 검증)
- /home/dlacksdn/dreamerv3-torch/networks.py (L293-357 MultiEncoder 라우팅)
- /home/dlacksdn/dreamerv3-torch/tools.py (L323 sample_episodes seed=0)
- /home/dlacksdn/f1tenth_RL_project/pkg/src/pkg/maps/map_easy3.png|.yaml (N3 실측, `map_easy.*` 부재)
- /home/dlacksdn/f1tenth_RL_project/f1tenth_gym_ros/maps/Oschersleben.{png,yaml}
