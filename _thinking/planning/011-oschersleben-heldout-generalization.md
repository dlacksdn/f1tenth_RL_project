# 011 — Oschersleben held-out 일반화 프로토콜 (결정 #32, curriculum 폐기)

> 2026-05-22. 집컴 세션, Phase 2-4(A19) 진입 직전 사용자 핵심 지시에서 파생.
> 사용자 지시(2026-05-22, 원문): "map_easy 만으로 훈련을 하고, oschersleben map에서 측정을 진행한다. 즉, Oschersleben map은 훈련에 사용되어서는 안 된다." + 다운스트림 3개 결정(예산/OOD기준/A16)은 Claude 최선 판단 위임.
> 선행: [005 v3 §0-3 목표, §1 #9/#21/#25/#31, §3 Phase 5, §6 A11~A16](./005-f1tenth_dreamerV3_version3.md), [implementation/004 결정 #2 (A13)](../implementation/004-pre-phase1-2-decisions.md), [008 snapshot](./008-snapshot_policy_refinement.md), [009 lap+A11](./009-lap-detection-and-A11.md).
> 본 문서는 **train/eval 트랙 분리 + Phase 5 구조의 신 SSOT**. v3 §1의 curriculum 결정(#9/#21/#25 Stage 2/#31)을 **supersede**. 충돌 시 본 문서 우선(SSOT 룰: 높은 번호 우선).

---

## 1. 결정 #32 — Oschersleben는 held-out 평가 전용 (훈련 금지)

**결정**: 학습은 **`map_easy3` 단일 트랙으로만** 수행한다. **`Oschersleben`은 학습에 단 한 번도 사용하지 않으며**, map_easy3로 학습한 정책을 **zero-shot(가중치 갱신 없이)** 으로 평가하는 held-out 일반화 테스트 트랙이다.

**근거**:
- 동일 트랙으로 학습·평가하면 기록이 잘 나오는 것은 당연 — 일반화 성능을 측정할 수 없다. Oschersleben을 fine-tune하면 A12/A13는 "본 트랙에 과적합된 정책"을 측정하게 되어 평가 의미가 소실된다.
- 원 프로젝트 프레이밍(env_setting/003 §3 표)과도 정합: **Map Easy = 연습/알고리즘 완성도(평가 30%)**, **Oschersleben = 본 경기 트랙(평가 10%)**. "연습 트랙으로 학습 → 본 경기 트랙에서 일반화 측정"이 더 정직하고 발표(평가 60%) 서사로도 강하다.
- 알고리즘 발표(평가 60%)의 핵심 메시지: DreamerV3 world-model이 단일 트랙 학습만으로 미관측 트랙에 일반화하는가.

**불변 조건(invariant)**: 어떤 세션·스크립트·config도 Oschersleben을 replay buffer / train env / fine-tune에 투입하지 않는다. Oschersleben 등장은 **평가 경로(eval env, A12~A14)에 한정**.

---

## 2. supersede되는 기존 결정 (v3 §1 / impl/004)

| 기존 | 내용 | 본 결정 후 상태 |
|---|---|---|
| #9 Curriculum | "순차 fine-tune Map Easy 500K → Oschersleben 500K, joint replay 30%" | **폐기.** 단일 스테이지(map_easy3)만. joint replay 개념 소멸. |
| #21 ckpt optimizer carry-over | "Stage 2 fine-tune 시 latest.pt warm + fresh optim (`--fresh_optim`)" | **무효.** Stage 2 부재 → warm-load/fresh-optim 분기 불요. `--fresh_optim` flag 미구현(Phase 3에서 제외). |
| #25 Map 명명 (Stage 2 부분) | "Stage 2는 `task='f1tenth_oschersleben'`" | **Stage 2 부분만 폐기.** 평가 트랙 = map_easy3 학습은 유지. Oschersleben config task는 **eval-only** 진입점으로만 사용. |
| #31 A16 rollback | "A16 미달 시 Stage 1 latest 복원 + Stage 2 재학습(joint_replay 0.5)" | **무효** (§5 참조, A16 폐기). |

★ #2 (A13 = baseline×1.5 = 45.5s)는 **fine-tune 가정**으로 산정된 값 → §4에서 재검토(zero-shot 임계값 OPEN).

---

## 3. 다운스트림 결정 (사용자 위임 → Claude 아키텍트 판단)

### 3-1. 학습 예산 → **map_easy3 단일 스테이지 500K 유지** (A19 baseline)
- **결정**: 문서 baseline = `steps=5e5`(현 configs 값) 단일 스테이지. 본 세션 A19는 **단일 500K wall-clock**으로 추정한다.
- **근거**: (a) 일반화 연구에서 단일 트랙 과학습은 map_easy3 기하에 **과적합** → Oschersleben zero-shot 전이를 오히려 해칠 수 있다. 더 길게 = 더 좋음이 아니다. (b) 500K는 DreamerV3 표준 예산권이며 24h 안에 여유 있게 들어간다(2-stage 1M → 단일 500K로 wall-clock 절반).
- **contingency(보류)**: map_easy3 in-dist 평가(A11)가 **underfit**(미달)이고 A19가 충분한 headroom을 보이면, ≤1M로 확장은 선택지. 단 기본값은 500K. 확장은 별도 사용자 승인 + 새 노트.

### 3-2. Oschersleben(OOD) 평가 → **지표 재정의 + 임계값 defer**
- **결정**: A12/A13/A14를 **zero-shot 일반화 지표**로 재정의(§4). 수치 pass 임계값은 **OPEN** — map_easy3 학습 정책의 **첫 zero-shot 평가 실측 후** 사용자와 확정. 임의 임계값 발명 금지.
- **근거**: 미관측 312m 트랙에서 fine-tune 가정 임계값(2-lap≥80%, lap_time≤45.5s)은 거의 확실히 infeasible → 그대로 두면 무의미한 게이트. zero-shot은 **완주 자체가 성과**이므로 완주율·진행률 중심 지표가 적절. 정직하고 데이터 기반.

### 3-3. A16 (forgetting 재평가) → **폐기**
- **결정**: A16 + 결정 #31 삭제. Stage 2 학습이 없으므로 catastrophic forgetting 개념 자체가 소멸. map_easy3 in-distribution 검증은 **A11이 단일 출처**로 수행(별도 재평가 불요).
- Phase 5-3(map_easy3 재평가 스테이지) 삭제.

---

## 4. 재정의된 acceptance criteria (신 SSOT)

> v3 §6-2-3, impl/004 §5 표를 본 절이 supersede. Phase 5 평가 코드는 본 절을 참조.

**A11 (in-distribution, map_easy3)** — 유지. map_easy3 500K 학습 후 eval 20 ep, **median lap_time ≤ `GF_baseline_map_easy × 1.5`** (측정 실패 시 fallback ≤ 45s). 〔009 §A11〕 = 학습 트랙 성능의 단일 출처.

**A12 (OOD 완주, Oschersleben zero-shot)** — 재정의.
- 1차(primary) 지표: **lap-completion rate** (`lap_count[0] ≥ 1` 비율, 20 ep) + **progress fraction** (첫 crash까지 진행거리 / L_track=312.61m).
- pass 임계값: **OPEN** (첫 zero-shot 측정 후 확정). 기존 "2-lap ≥ 80%"는 fine-tune 가정 → 폐기.

**A13 (OOD lap_time, Oschersleben zero-shot)** — 재정의.
- ≥1 lap 완주한 ep에 한해 median/best lap_time **보고(report)**. 기존 "≤45.5s hard gate"(#2)는 zero-shot에서 무의미 → **hard pass 게이트 해제**, 리포트 지표로 강등. 임계값 OPEN.

**A14 (정책 snapshot 저장)** — 유지하되 기준 트랙 변경. 기존 "Oschersleben lap_time ≤ 110s 정책 저장"은 fine-tune 산정. → **map_easy3 학습 중 best in-dist 정책 + Oschersleben zero-shot에서 우수한 정책** 모두 저장(LeWorldModel 다양성). 파일명 규약 `policy_*.pt`은 008 유지. 구체 임계값 OPEN.

**A16** — **삭제** (§3-3).

★ centerline(A_centerline), GF baseline(A_gap)은 양 트랙 모두 유지 — Oschersleben eval env 구동에 L_track/centerline 필요. **eval 경로이므로 #32 불변조건과 무충돌.**

---

## 5. 코드/명세 영향 범위 (후속 Phase에서 반영)

- **vendor/dreamerv3-torch/configs.yaml f1tenth 블록**: `task: 'f1tenth_map_easy3'` 유지. line 192 주석 "Stage 2 -> 'f1tenth_Oschersleben'"는 **오해 유발** → 본 결정 반영해 정정(§6에서 본 세션 처리). Oschersleben은 **별도 eval config/CLI override**로만 진입.
- **Phase 3 (train.py wrapper, snapshot)**: 단일 스테이지 가정. `--fresh_optim`/warm-load(#21) 분기 **불구현**. counter ckpt(C-N10)는 유지.
- **Phase 5 구조 재정의**:
  - 5-1: map_easy3 500K 학습 → A11.
  - 5-2: **(학습 아님)** map_easy3 학습 정책을 **Oschersleben zero-shot 평가** → A12/A13/A14 측정. 가중치 갱신·replay 투입 절대 금지.
  - 5-3: **삭제** (A16 폐기).
- **Reward (Phase 4, §4-3)**: R_lap 분리(Map Easy3=25 / Oschersleben=100)는 **학습 reward**. Oschersleben은 학습 안 하므로 Oschersleben R_lap=100은 **학습에 미사용**. eval은 lap_time/완주 기반이라 reward 불요 → Oschersleben reward 항목은 사실상 dead(평가 시 env가 계산해도 무시). L_track=312.61m은 eval env progress/centerline에 여전히 필요(유지).
- **planning/008 snapshot**: Stage 2 weights(`f1tenth_v3_oschersleben/latest.pt`) 산출물 삭제. 단일 logdir `f1tenth_v3_map_easy3/`만. (008은 append-only이므로 본 문서가 정정 출처.)

---

## 6. 본 세션 즉시 조치 + OPEN 항목

**즉시(본 세션)**:
- [x] 본 문서(planning/011) 작성 — 결정 #32 SSOT.
- [ ] configs.yaml line 192 stale 주석 정정(Stage 2→Oschersleben 제거, held-out 명시) — 미래 세션의 Stage 2 재도입 방지용 최소 수정.
- [ ] Phase 2-4 A19는 §3-1대로 **단일 500K** wall-clock 추정으로 진행.

**OPEN (사용자 확정 필요, 측정 후)**:
- A12 완주율 / progress fraction pass 임계값 (zero-shot 첫 측정 후).
- A13 lap_time 리포트 → 게이트화 여부 (zero-shot 첫 측정 후).
- A14 snapshot 저장 임계값.
- (contingency) map_easy3 500K→1M 확장 여부 (A11 underfit + A19 headroom 동시 충족 시).
