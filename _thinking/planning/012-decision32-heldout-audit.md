# 012 — 결정 #32 / 011 held-out 계획 독립 감사 (계획서 재작성 지시서)

> 2026-05-22. 독립 검증자(별도 Opus 세션) 작성. **읽기 전용 감사 결과물.**
> 대상: [planning/011 Oschersleben held-out](./011-oschersleben-heldout-generalization.md) + configs.yaml line 192 정정.
> 교차검증 SSOT: [005 v3](./005-f1tenth_dreamerV3_version3.md), [impl/004](../implementation/004-pre-phase1-2-decisions.md), [008 snapshot](./008-snapshot_policy_refinement.md), [009 lap+A11](./009-lap-detection-and-A11.md), [impl/009 vendor-in](../implementation/009-phase2-0-vendor-fork-patch.md), [impl/012 config quirk](../implementation/012-phase2-3-config-param-audit.md), [env_setting/003 spec](../env_setting/003-project-spec.md).
> HEAD `73ad47c` 확인. 작업트리: 011 신규 + configs.yaml line 192 주석 정정(미커밋).
> **이 문서는 011 작성 에이전트가 읽고 011을 재작성(supersede)하기 위한 지시서다.** 동의 도장이 아니라 결함·누락·대안 목록이다.

---

## 0. 한 줄 평결

**APPROVE-WITH-CHANGES.** 핵심 방향(Oschersleben held-out 분리)은 과학적으로 타당하고 사용자 지시에 충실하다. 단, 아래 **치명 결함 2건(CF-1, CF-2) + High 위험 3건(R1, R2, R3)** 을 011 재작성 시 반드시 반영해야 Phase 2-4(A19) 진입 가능.

---

## 1. 치명적 결함 (재작성 필수)

### CF-1 — 011 §4 A11이 009의 supersede를 잘못 되돌렸다 (SSOT 위반)
- **현상**: [011:57](./011-oschersleben-heldout-generalization.md#L57)은 A11을 *"median lap_time ≤ GF_baseline×1.5 (fallback 45s)"* 로 적고 `〔009 §A11〕`을 근거로 인용한다.
- **모순**: [009:34](./009-lap-detection-and-A11.md#L34)·[009:47](./009-lap-detection-and-A11.md#L47)의 결정 B는 A11을 **"2-lap 완주율(completion-only), 정밀 lap_time 측정 안 함"** 으로 명시 supersede했다. 011은 009를 *지지 근거로* 인용하면서 실제로는 **009가 폐기한 005 원안([005:185](./005-f1tenth_dreamerV3_version3.md#L185))을 부활**시켰다.
- **왜 치명적**: A11은 #32 체제에서 §3-3에 의해 **유일한 in-dist 게이트**다. 그 정의가 "completion-only(009)"인지 "lap_time(011 본문)"인지 문서만으로 판별 불가 → 게이트 자체가 불능 상태.
- **재작성 지시**: A11을 둘 중 하나로 **명시 확정**하라.
  - (권장) completion-only 유지 → 011은 009 인용을 정정하고 lap_time 문구 삭제.
  - lap_time으로 되돌리려면 "009 결정 B를 supersede한다"를 **명시 선언**하고 그 근거를 적어라(현재는 무선언 부활 = SSOT 룰 위반).

### CF-2 — zero-shot eval 전용 실행 경로가 코드에 부재, #32 불변조건이 미보장
- **코드 근거**: [dreamer.py:147](../../vendor/dreamerv3-torch/dreamer.py#L147) `suite, task = config.task.split("_", 1)` 하나로 트랙 결정. [dreamer.py:243-244](../../vendor/dreamerv3-torch/dreamer.py#L243)는 `train_envs`·`eval_envs`를 **동일 `config.task`로** 생성. [f1tenth.py:51-54](../../vendor/dreamerv3-torch/envs/f1tenth.py#L51) `task==trackname` 그대로 전달.
- **결론**: dreamerv3-torch에는 **"train≠eval 트랙" 메커니즘이 없다.** [011:76](./011-oschersleben-heldout-generalization.md#L76)·[011:80](./011-oschersleben-heldout-generalization.md#L80)의 "별도 eval config/CLI override"는 **설계·구현물이 0**이다.
- **구체 위험**: zero-shot 측정을 위해 순진하게 `--task f1tenth_Oschersleben`을 주면 [dreamer.py:243](../../vendor/dreamerv3-torch/dreamer.py#L243)에 의해 **train_env까지 Oschersleben이 되어 #32 불변조건([011:19](./011-oschersleben-heldout-generalization.md#L19))을 정면 위반**한다. 불변조건은 선언만 있고 강제 장치가 없다.
- **재작성 지시**: 011에 다음을 추가하라.
  1. zero-shot eval-only 진입점의 **구체 메커니즘** — checkpoint 로드 + Oschersleben env + **train_env 미생성/가중치 갱신 0** 보장 방식(별도 eval 스크립트인지, dreamer.py eval-only 분기인지, `--steps` 0 + 무학습 경로인지). Phase 3/5의 막연한 위임이 아니라 어느 파일/진입점인지 명시.
  2. **acceptance A_heldout(신규)**: "train_env·replay buffer에 Oschersleben trackname이 단 한 번도 인스턴스화되지 않음"을 검증하는 테스트. 이게 있어야 [011:19](./011-oschersleben-heldout-generalization.md#L19) 불변조건이 실효.

---

## 2. High 위험 (재작성 강력 권고)

### R1 — 채점 10% + "Oschersleben 완주" 완성도 항목 손실 위험을 011이 미고지
- **명세 근거**: [003:24-28](../env_setting/003-project-spec.md#L24) — 팀 lap_time **10% = Oschersleben** F1/10 Race Simulation 순위 차등 배점. 알고리즘 완성도 세부에 **"Oschersleben 완주 여부"** 명시. [003:50](../env_setting/003-project-spec.md#L50) 발표 템플릿은 **"실제 주행 영상 포함 2바퀴 lap time 기재"** 요구.
- **문제**: map_easy3 L_track=117.22m → Oschersleben 312.61m([impl/004:62](../implementation/004-pre-phase1-2-decisions.md#L62)), **2.7배 길이 + 상이 기하**. 단일 트랙 학습 정책의 zero-shot 완주는 본질적으로 어렵다. 011([011:16-17](./011-oschersleben-heldout-generalization.md#L16))은 이를 *"더 정직·강한 발표 서사"* 로만 프레이밍하고, **10% 점수와 완성도 항목을 사실상 포기할 수 있다는 trade-off를 명시하지 않았다.**
- **재작성 지시**: 011에 "zero-shot 채택 시 위험" 절을 추가 — Oschersleben 미완주 시 10% 손실 + 완성도 항목 미달 가능성을 명시하고, 그럼에도 사용자 지시(훈련 금지)가 우선함을 적어라. 발표(60%) 서사로 보전하는 논리도 함께. **이 trade-off 미고지가 본 감사가 본 가장 큰 누락이다.**

### R2 — eval-only 가드/acceptance 부재 (CF-2와 연동)
- CF-2의 A_heldout이 acceptance 표(§4)에 추가되어야 한다. 현재 §4는 A11/A12/A13/A14만 다루고 "Oschersleben이 학습 경로에 안 들어감"을 검증하는 항목이 없다.

### R3 — PyYAML quirk와 A19/eval override의 상호작용 미해소
- **근거**: [impl/012:27](../implementation/012-phase2-3-config-param-audit.md#L27) — PyYAML 6이 `5e5`·`1e-4` 등 소수점 없는 지수를 **문자열로 파싱**, dreamerv3 args_type가 coerce 안 함 → 실제 dreamer.py argparse 경로에서 잠재 TypeError. param_audit의 `_coerce`만 처리하고 학습 경로는 미해결.
- **문제**: [011:39](./011-oschersleben-heldout-generalization.md#L39)이 "현 configs 값 steps=5e5"를 그대로 A19 baseline으로 plug-in하면서 이 quirk를 언급 안 함. A19 dry-run·eval override(`--task`, `--steps`) 모두 이 경로([configs.yaml:196](../../vendor/dreamerv3-torch/configs.yaml#L196))를 탄다.
- **재작성 지시**: A19 진입 전 `steps: 5e5` 문자열이 실제 dreamer.py 경로에서 coerce되는지 확인을 선결 조건으로 명시(미해결 시 §11-A wall-clock 식의 N_steps 자체가 무의미).

---

## 3. Med / Low 위험

| # | 우선 | 항목 | 근거 |
|---|---|---|---|
| R4 | Med | **008 snapshot 머신러리 부분 정합 미완.** [008:32](./008-snapshot_policy_refinement.md#L32)·[008:67](./008-snapshot_policy_refinement.md#L67)의 트랙별 임계(map_easy3=45/Osch=110)·interval snapshot은 **학습 중 eval 인터리브** 전제. Oschersleben이 eval-only면 그 interval 경로가 무의미. [011:83](./011-oschersleben-heldout-generalization.md#L83)은 logdir 1개만 정정했고, A14 Oschersleben snapshot이 *학습 루프 밖 zero-shot 시점*에 어떤 메커니즘으로 저장되는지 미설계([011:66](./011-oschersleben-heldout-generalization.md#L66) "OPEN"). | 008:32-67, 011:66,83 |
| R5 | Low | 008 §4 용량 ~21GB(2-stage)가 단일 stage(~10.5GB)로 미갱신. | 008:52-58 |
| R6 | Low | [003:128](../env_setting/003-project-spec.md#L128) "학습 순서 권장 Map Easy → Oschersleben"는 명세 *권장*. #32가 의도적으로 벗어나므로 발표에서 "권장과 다른 선택의 이유"를 명시 방어 필요. | 003:128 |

---

## 4. 4개 결정별 동의/반대 + 대안

### 결정 #32 (held-out) — **동의 (조건부)**
사용자 지시(훈련 금지)는 hard constraint. 동일 트랙 학습·평가는 일반화 측정 불가([011:15](./011-oschersleben-heldout-generalization.md#L15))라는 논거 옳음. **단 CF-2 강제 장치 + R1 trade-off 고지 전제.**

### (2) 예산 500K 유지 — **약한 반대 / 논거 교체 권고**
- **문제**: *"단일 트랙 과적합 → zero-shot 악화"*([011:40](./011-oschersleben-heldout-generalization.md#L40))는 DreamerV3 맥락에서 **레버를 잘못 짚었다.** world-model은 본 적 없는 트랙 기하를 step 수만으로 일반화하지 않는다. 500K vs 1M은 **부차 레버**이고, 진짜 한계는 **단일 트랙 학습 전제 자체**(R1과 직결).
- **더 나은 대안 (사용자 제약 "map_easy만 훈련" 내에서 완전 합법)**: 학습 트랙 **도메인 랜덤화** — (a) start pose 랜덤화([impl/004:39](../implementation/004-pre-phase1-2-decisions.md#L39) START_POSES 인프라 존재), (b) 주행 방향 반전, (c) friction/dynamics·LiDAR noise 섭동, (d) map_easy 변종 존재 시 다중 연습트랙. 이것이 zero-shot 전이를 실제로 끌어올리는 1차 레버. 011은 전혀 검토 안 함.
- **결론**: 500K **기본값은 수용**(24h·DreamerV3 표준권). 단 *근거를 "과적합 방지" → "단일 트랙에선 step 한계효용 낮음 + 도메인 랜덤화 우선"으로 교체*하고, 도메인 랜덤화를 별도 결정으로 검토하라.

### (3) OOD 지표 재정의 + 임계값 defer — **동의 (가드 조건)**
- fine-tune 가정 임계(2-lap≥80%, ≤45.5s [impl/004:22](../implementation/004-pre-phase1-2-decisions.md#L22))를 hard-gate 해제한 판단([011:64](./011-oschersleben-heldout-generalization.md#L64)) 타당.
- **"측정 후 합리화" 차단**: 지표 *정의*는 지금 동결. [011:60](./011-oschersleben-heldout-generalization.md#L60)이 완주율·progress fraction(=dist/312.61) 정의한 것은 좋다. progress fraction 재현성은 Phase 4 arclength windowed-closest-point([009:22](./009-lap-detection-and-A11.md#L22))가 Oschersleben에서 정확해야 성립(009가 Osch lap=550m 정상 확인 → 근거 있음). **"임계값은 미정이되 실측 raw 수치는 무조건 보고"를 사전 약정으로 명문화**하라.
- A13 리포트 강등([011:64](./011-oschersleben-heldout-generalization.md#L64))은 zero-shot 합리적이나, **10% 평가(Osch lap_time 순위) + 발표 "2바퀴 lap_time 기재"([003:50](../env_setting/003-project-spec.md#L50))** 요구로 인해 게이트가 아니어도 **반드시 측정·기재** 명시(R1).

### (4) A16 폐기 — **동의**
Stage 2 부재 → forgetting 개념 소멸([011:48](./011-oschersleben-heldout-generalization.md#L48)) 논리적으로 옳음. [005:190](./005-f1tenth_dreamerV3_version3.md#L190) A16·[005:133](./005-f1tenth_dreamerV3_version3.md#L133) #31 모두 Stage 2 의존이라 무효 타당. in-dist를 A11 단일 출처로 축소 무방 — **단 CF-1 선결**해야 A11이 실효 게이트로 남는다. (#21 `--fresh_optim`·joint replay 키가 [configs.yaml:186-220](../../vendor/dreamerv3-torch/configs.yaml#L186)에 부재 = 정합 확인.)

---

## 5. Phase 2-4 진입 전 체크리스트 (011 재작성 시 반영)

- [ ] **CF-1**: A11을 completion-only(009) **또는** lap_time(005)으로 확정 + 011 §4의 009 오인용 정정.
- [ ] **CF-2 / R2**: zero-shot eval-only 진입점 **구체 설계** + train_env가 Oschersleben을 절대 인스턴스화 안 함을 검증하는 acceptance(A_heldout) 추가.
- [ ] **R1**: zero-shot 채택 시 10% lap_time + Oschersleben 완주 완성도 손실 가능성 명시 고지(사용자 수용 확인). 발표 서사 보전 논리 병기.
- [ ] **R3**: A19 전에 `steps: 5e5` 문자열 quirk(impl/012)가 실제 dreamer.py 경로에서 coerce되는지 확인.
- [ ] **(2) 권고**: 도메인 랜덤화(start pose/방향/dynamics/noise)를 zero-shot 일반화 1차 레버로 별도 결정화 검토.
- [ ] **R4**: A14 Oschersleben snapshot의 *학습 루프 밖* 저장 메커니즘 설계 또는 명시적 OPEN 처리.
- [ ] **G 정합 확인**: 단일 500K는 [005:565](./005-f1tenth_dreamerV3_version3.md#L565) wall-clock 식(N=500K)과 무모순(2-stage→단일로 N만 축소, 식 동일). 단 [impl/009:96](../implementation/009-phase2-0-vendor-fork-patch.md#L96)의 `steps=5e5 //action_repeat` agent-step vs env-step 회계 모호성은 **#32가 만든 게 아니라 기존 미해소 사항을 011이 상속** — A19 측정 시 A(ms/env_step) 단위와 N_steps 단위 일치를 명시 확정.

---

## 6. 불확실 표기 (감사 범위 한계)

- R3·G의 `steps` 회계 단위(agent-step vs env-step)는 impl/009·012 기록만으로 dreamer.py main loop의 `//action_repeat` 적용 지점을 단정 못 함. **A19 진입 시 dreamer.py main loop 카운터 분기를 직접 정독해 확정 필요.** (본 감사는 읽기 전용 범위에서 make_env/eval_envs 라인까지만 정독, main loop 카운터는 미정독.)
- 도메인 랜덤화 대안의 구현 비용·기존 wrapper(F110GymnasiumWrapper) 지원 여부는 미확인. 별도 검토 권고.
