# 013 — Oschersleben held-out 프로토콜 REV2 (011 supersede, 감사 012 반영)

> 2026-05-22. 집컴 세션. [011 held-out 결정 #32](./011-oschersleben-heldout-generalization.md)을 독립 감사 [012](./012-decision32-heldout-audit.md)의 CF-1/CF-2 + R1~R6 + G를 반영해 **재작성(supersede)**.
> 011의 결정 #32 코어(Oschersleben 훈련 금지·zero-shot held-out)는 유지. 011 §4 A11 정의·eval 경로·예산 근거·trade-off 누락을 정정/보강.
> 교차 SSOT: [005 v3](./005-f1tenth_dreamerV3_version3.md), [impl/004](../implementation/004-pre-phase1-2-decisions.md), [008](./008-snapshot_policy_refinement.md), [009 lap+A11](./009-lap-detection-and-A11.md), [impl/012](../implementation/012-phase2-3-config-param-audit.md), [env_setting/003](../env_setting/003-project-spec.md).
> 본 문서가 train/eval 분리·Phase 5 구조·acceptance의 **신 SSOT**. 011과 충돌 시 013 우선.

---

## 1. 결정 #32 코어 (011에서 유지)

학습은 **`map_easy3` 단일 트랙으로만**. **`Oschersleben`은 학습에 단 한 번도 사용 금지** — map_easy3 학습 정책을 **zero-shot(가중치 갱신 0)** 으로 평가하는 held-out 트랙. 불변조건: 어떤 세션·스크립트도 Oschersleben을 replay/train env/fine-tune에 투입하지 않는다. supersede 대상(v3 #9 curriculum / #21 fresh-optim / #25 Stage2 task / #31 A16 rollback)은 011 §2 그대로 유효.

---

## 2. CF-1 정정 — A11은 completion-only (009 결정 B 준수)

- **011의 오류**: 011 §4가 A11을 *"median lap_time ≤ GF×1.5 (fallback 45s)"* 로 적고 `〔009 §A11〕`을 근거 인용 → 실제로는 [009:30-37](./009-lap-detection-and-A11.md#L30) **결정 B(사용자 승인 2026-05-21)가 폐기한 005 원안([005:185](./005-f1tenth_dreamerV3_version3.md#L185))을 무선언 부활**시킨 SSOT 위반.
- **확정(본 문서)**: **A11 = map_easy3 500K 학습 후 eval 20 ep, 2-lap 완주율(completion-only) ≥ 80%**. 정밀 lap_time 게이트 없음. lap 판정은 [009 결정 A](./009-lap-detection-and-A11.md#L18) **centerline arclength wrap** 신호(Phase 4 windowed closest-point) 사용. GF baseline(~12.34s/lap, impl/008)은 참고 보존, 게이트 미사용.
- **근거**: 009 결정 B는 "Stage 1(map_easy3)은 주행 학습 확인 warmup, 정밀 lap_time은 별도"라 명시. #32에서 정밀 lap_time은 Oschersleben zero-shot eval(A13 report)에만 둔다 → 정합.

---

## 3. CF-2 / R2 정정 — zero-shot eval-only 실행 경로 구체 설계 + 가드

- **코드 사실**: [dreamer.py:147](../../vendor/dreamerv3-torch/dreamer.py#L147) `suite,task=config.task.split("_",1)` 단일 결정. [dreamer.py:243-244](../../vendor/dreamerv3-torch/dreamer.py#L243) `train_envs`·`eval_envs`를 **동일 `config.task`로** 생성. → dreamerv3-torch에 train≠eval 트랙 메커니즘 **부재**. 순진하게 `--task f1tenth_Oschersleben`을 주면 **train_env까지 Oschersleben** → #32 불변조건 정면 위반(012 CF-2 정확).
- **확정 설계 — 별도 eval-only 스크립트 `scripts/eval_heldout.py` (Phase 5-2에서 구현)**:
  1. map_easy3 학습 산출 checkpoint(`~/logdir/f1tenth_v3_map_easy3/latest.pt`) 로드.
  2. **Oschersleben eval env 1개만 생성**(F1Tenth 어댑터 직접 인스턴스화). **train_envs·replay·dataset·model_opt·_train 호출 일절 없음**. 가중치 갱신 0.
  3. eval policy(`eval_state_mean=True`)로 N ep 롤아웃 → 완주율/progress fraction/lap_time 수집.
  - dreamer.py 본체를 재사용하지 않으므로 train/eval 트랙 결합([dreamer.py:243](../../vendor/dreamerv3-torch/dreamer.py#L243))을 원천 회피. (대안: dreamer.py에 `--eval_only`+`--eval_task` 분기 추가도 가능하나, 결합 위험이 커 별도 스크립트를 1순위로 한다.)
- **A_heldout (신규 acceptance, §9 표에 추가)**: "학습 경로(train_envs·replay buffer·traindir)에 `Oschersleben` trackname이 단 한 번도 인스턴스화되지 않음"을 검증. 구현: (a) train 진입점에서 `assert resolved_train_task != Oschersleben` 가드, (b) pytest로 configs f1tenth `task=='f1tenth_map_easy3'` + 학습 logdir 산출물에 Oschersleben 미등장 확인. 이 가드가 있어야 §1 불변조건이 선언이 아닌 **실효**.

---

## 4. R1 정정 — zero-shot 채택의 채점 trade-off 명시 (★사용자 수용 확인 필요)

- **명세 근거**: [env_setting/003:24-28](../env_setting/003-project-spec.md#L24) — 팀 점수 **10% = Oschersleben lap_time 순위 차등**, 알고리즘 완성도 세부에 **"Oschersleben 완주 여부"**. [003:50](../env_setting/003-project-spec.md#L50) 발표 템플릿 **"실제 주행 영상 2바퀴 lap_time 기재"** 요구. [003:128](../env_setting/003-project-spec.md#L128) "학습 순서 권장 Map Easy → Oschersleben".
- **trade-off**: map_easy3(117.22m) → Oschersleben(312.61m, [impl/004:62](../implementation/004-pre-phase1-2-decisions.md#L62)), **2.7배 길이 + 상이 기하**. 단일 트랙 학습 정책의 zero-shot 완주는 본질적으로 어렵다 → **Oschersleben 미완주 시 10% lap_time 점수 + 완성도 항목·발표 2바퀴 기재 요구를 충족 못 할 수 있다.**
- **본 문서 입장**: 사용자 지시("Oschersleben 훈련 금지")가 hard constraint로 우선. 단 위 trade-off는 **사용자가 명시적으로 수용했는지 확인 필요(OPEN-U1)**. 보전 논리: 발표(60%)에서 "일반화(zero-shot transfer)를 정량 측정한 정직한 실험"으로 프레이밍 + "권장 순서와 다른 선택의 이유"(003:128)를 명시 방어(R6).

---

## 5. 예산(결정 #2 011) 정정 + 도메인 랜덤화 제안 (★사용자 결정 필요)

- **011 근거 철회**: 011의 *"단일 트랙 과학습 → zero-shot 악화"* 는 레버를 잘못 짚음(012 R2 정확). DreamerV3 world-model은 **본 적 없는 트랙 기하를 step 수만으로 일반화하지 않는다.** 500K vs 1M은 부차 레버.
- **예산 확정**: map_easy3 단일 **500K 기본 유지**(24h·DreamerV3 표준권). A19도 단일 500K. **단 근거 교체**: "과적합 방지"가 아니라 *"단일 트랙에선 step 한계효용이 낮고, 일반화 1차 레버는 도메인 다양성"*.
- **★ 도메인 랜덤화 제안 (결정 #33 후보, OPEN-U2)**: zero-shot 전이의 실제 1차 레버. 사용자 제약("map_easy만 훈련") 내에서 완전 합법:
  - (a) **start pose 랜덤화** — 현재 wrapper는 [f1tenth_env.py:198](../../dreamer_f1tenth/envs/f1tenth_env.py#L198) **고정 `_default_pose`만** 사용(랜덤화 미배선 = 단일 시작점 과적합 위험 실재). `reset(options={"pose":...})` 인프라는 이미 존재 → centerline 위 랜덤 s에서 시작하도록 배선.
  - (b) 주행 방향/구간 다양화, (c) friction·dynamics 섭동, (d) LiDAR noise 주입.
  - **비용**: 신규 구현(Phase 4 episode 설계와 연동). wrapper 지원 여부 일부 미확인(start pose는 인프라 有, 나머지는 조사 필요).
  - **본 문서 입장**: zero-shot 목표를 진지하게 추구하려면 (a)는 강력 권고. 채택 여부·범위는 사용자 결정.

---

## 6. R3 해소 — PyYAML quirk는 실제 학습 경로에 미발생 (실측)

- **실측(본 세션, .venv)**: dreamer.py가 쓰는 **ruamel.yaml(typ=safe)** 로 configs.yaml 로드 시 `steps:5e5`→`500000.0 float`, defaults `model_lr:1e-4`→`0.0001 float`로 **정상 coerce**. impl/012가 우려한 string 파싱은 **param_audit.py의 PyYAML 경로에만** 존재했고 실제 학습 config-load엔 없음.
- **잔여 주의**: CLI override(`--steps`, `--lr`)는 argparse args_type 경로 → bare exponent 대신 **numeric/소수점 표기**로 전달. A19 baseline은 yaml 값 사용이므로 무영향.

---

## 7. G 해소 — steps 단위 확정 (main loop 직접 정독)

- **확정**: configs `steps:5e5`는 **env-step 단위**. [dreamer.py:218](../../vendor/dreamerv3-torch/dreamer.py#L218) `config.steps //= action_repeat(2)` → 250000 agent-step. [dreamer.py:307](../../vendor/dreamerv3-torch/dreamer.py#L307) 메인 루프는 `agent._step`(agent-step) 기준 종료. [dreamer.py:41/83](../../vendor/dreamerv3-torch/dreamer.py#L41) `_step=logger.step//2`(agent), `logger.step=2*_step`(env).
- **결론**: **"500K 예산" = 500K env-step = 250K agent-step(=정책 결정 횟수)**. agent-step당 어댑터가 action_repeat=2 env-step 소비.
- **A19 단위 정합 (§11-A 식)**: D=(N·A + (N/train_ratio)·B)/1000/60. A19에서 N과 A·B 단위를 일치시켜야 함:
  - A=env_step_avg_ms(어댑터 내부 2 sim step 포함 여부 측정 시 명확화), B=train_step_avg_ms(agent-step당 1 train의 분율은 train_ratio).
  - [dreamer.py:35](../../vendor/dreamerv3-torch/dreamer.py#L35) `_should_train=Every(batch_size*batch_length / train_ratio)`, 단위는 agent-step. → **A19 dryrun_bench는 agent-step 기준으로 N 카운트하고, env wall은 agent-step×(2 env-step)로 환산**해 D 산출. 측정 시 raw 로그로 단위 명기.

---

## 8. R4 / R5 / R6 정정

- **R4 (A14 snapshot, Med)**: Oschersleben snapshot은 **학습 루프 밖 zero-shot 시점**(scripts/eval_heldout.py 실행)에 저장. 008의 학습 중 interval-eval 인터리브 경로는 **map_easy3에만** 적용. A14 재정의(§9). 구체 임계값은 OPEN.
- **R5 (용량, Low)**: 008 §4 ~21GB(2-stage) → **단일 stage ~10.5GB**로 정정(map_easy3 logdir 1개). 008은 append-only이므로 본 문서가 정정 출처.
- **R6 (발표 방어, Low)**: 003:128 "권장 순서 Map Easy→Oschersleben"를 #32가 의도적으로 벗어남 → 발표에서 "일반화 정량 측정을 위한 의도적 held-out 설계"로 명시 방어.

---

## 9. 재정의된 acceptance criteria (신 SSOT, 005 §6-2-3·011 §4 supersede)

| Criterion | 정의 (REV2) | 상태 |
|---|---|---|
| **A11** in-dist | map_easy3 500K 후 eval 20 ep, **2-lap 완주율 ≥ 80% (completion-only, arclength lap)**. lap_time 게이트 없음 | 확정 (CF-1) |
| **A12** OOD 완주 | Oschersleben **zero-shot** eval 20 ep: **lap-completion rate(`lap_count≥1`) + progress fraction(첫 crash까지 dist / 312.61m)**. 임계값 OPEN(첫 측정 후) | 재정의 |
| **A13** OOD lap_time | ≥1 lap ep에 한해 median/best lap_time **측정·기재(report)**. hard-gate 해제. 단 10%·발표 요구로 **반드시 측정** | 재정의 |
| **A14** snapshot | map_easy3 학습 중 best in-dist 정책 + Oschersleben zero-shot 우수 정책 저장(루프 밖). 임계 OPEN | 재정의 |
| **A_heldout** (신규) | 학습 경로에 Oschersleben trackname **0회 인스턴스화** 검증(가드+pytest) | 신규 (CF-2) |
| **A16** | **삭제** (Stage 2 부재 → forgetting 소멸) | 폐기 |

★ A_centerline·A_gap(양 트랙)은 eval 경로이므로 #32와 무충돌, 유지.

---

## 10. OPEN 항목 + Phase 2-4 진입 체크리스트

**★ 사용자 결정 필요 (진행 전 확인)**:
- **OPEN-U1**: zero-shot 채택의 채점 trade-off(§4 — Oschersleben 미완주 시 10% + 완성도·발표 2바퀴 요구 손실 가능) 수용 확인.
- **OPEN-U2**: 도메인 랜덤화(§5 결정 #33 후보, 최소 start pose 랜덤화) 채택 여부·범위.

**측정 후 확정(defer)**: A12 완주율/progress 임계, A13 lap_time 게이트화 여부, A14 snapshot 임계.

**Phase 2-4(A19) 진입 전 (012 §5 체크리스트 반영)**:
- [x] CF-1 A11 completion-only 확정.
- [x] CF-2 eval-only 메커니즘 설계 + A_heldout 정의 (구현은 Phase 5-2).
- [x] R3 PyYAML quirk 실제 경로 무영향 실측 확인.
- [x] G steps 단위(env-step→//2→agent-step) main loop 정독 확정.
- [ ] OPEN-U1/U2 사용자 응답.
- [ ] A19: dryrun_bench를 agent-step 기준 N으로 작성, D 단위 명기, 단일 500K 추정.
