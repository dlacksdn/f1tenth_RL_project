# 014 — 결정 #32 PROVISIONAL 처리 + 2-시나리오 분기 (Oschersleben 훈련 허용 미확정)

> 2026-05-22. 집컴 세션. 사용자 통지: **Oschersleben을 학습에 써도 되는지(허용 유무)가 미확정.** 사용자가 점심 후 교수님께 확답을 받아 전달 예정.
> 본 문서는 [011](./011-oschersleben-heldout-generalization.md)/[013 REV2](./013-heldout-protocol-rev2.md)의 결정 #32를 **PROVISIONAL(잠정)** 로 강등하고, 확답 전까지 어느 쪽이든 싸게 피벗하도록 2-시나리오 분기와 "결정-무관/결정-종속" 작업 경계를 고정한다.
> 충돌 시 우선순위: 본 문서(014) > 013 > 011. 단 #32 코어 내용은 013 유지, 본 문서는 **상태(status)와 진행 가능 범위**만 규정.

---

## 1. 상태 강등 — 결정 #32 = PROVISIONAL

- 013/011의 결정 #32(Oschersleben 훈련 금지·zero-shot held-out)는 **사용자 잠정 지시**였으나, **외부(교수님) 허용 유무가 미확정**임이 판명. → **PROVISIONAL**.
- 확답 도착 전까지: #32에만 종속된 산출물(코드/acceptance 확정)은 **착수 금지**, 설계만 양쪽 병기.
- 확답 도착 시: 본 문서를 잇는 새 문서 1개(planning/015)로 **확정 전환**(어느 시나리오인지 + supersede 관계 명시).

---

## 2. 2-시나리오 정의

| | **시나리오 A — 훈련 금지(현 #32/013)** | **시나리오 B — 훈련 허용(원 v3 curriculum 부활)** |
|---|---|---|
| 학습 트랙 | map_easy3 단일 | map_easy3 → Oschersleben 순차 fine-tune (#9) |
| Oschersleben | zero-shot held-out 평가 전용 | Stage 2 학습 + 평가 |
| 총 step 예산 | 단일 500K (= 250K agent-step) | 2-stage ~1M (Stage당 500K) |
| 부활/유효 결정 | #32, 013 §1~§9 | v3 #9(curriculum)/#21(fresh-optim warm load)/#25(Stage2 task)/#31(A16 rollback) |
| A11 | 2-lap 완주율 ≥80% (013 CF-1) | 동일 (009 결정 B) |
| A12/A13 | zero-shot, 임계 OPEN (013 §9) | fine-tune 후, 005/004 #2 원 임계(2-lap≥80%, lap_time≤45.5s) |
| A16 | 폐기 (013 §3-3) | 부활 (Stage1 latest 강제 재평가 ≥70%) |
| eval_heldout.py / A_heldout | 필요 (013 §3) | 불필요(dreamer.py 기본 eval 경로로 충분) |
| 도메인 랜덤화(#33 후보) | 강하게 권고(일반화 1차 레버) | 선택(curriculum이 이미 도메인 다양성 일부 제공) |
| 채점 trade-off(R1) | 10% lap_time·완주 항목 손실 위험(OPEN-U1) | 위험 낮음(본 트랙 학습) |

---

## 3. 결정-무관 작업 (확답 전 진행 가능) — ★ 이번 세션 A19 포함

A19가 측정하는 **per-step 비용은 두 시나리오에서 동일**하다(같은 12M 네트워크·batch_size=8·length=64·차원; 트랙은 CPU 물리만 바뀌고 GPU 학습 비용 불변).

- **A = env_step_avg_ms**: map_easy3 기준 측정(어댑터 action_repeat=2 소비). 트랙 무관 수준의 물리비용.
- **B = train_step_avg_ms**: 네트워크·배치 고정 → **시나리오 무관**.
- **C = max VRAM(MB)**: 시나리오 무관. Pass: ≤ 6400MB(8GB×0.8).
- **D = wall-clock 추정**: 시나리오별로 **둘 다 산출**한다.
  - 시나리오 A: N = 500K env-step (= 250K agent-step). D_A.
  - 시나리오 B: N = 2×500K. D_B ≈ 2×D_A (+ Stage 전환 오버헤드 무시 가능).
  - 식: 005 §11-A `D=(N·A + (N/train_ratio)·B)/1000/60`. **단위는 013 §7 확정**(steps=env-step → //action_repeat=2 → agent-step; 루프는 agent._step 기준). dryrun_bench는 **agent-step 기준 N**으로 카운트, env wall은 agent-step×2 환산. raw 로그로 단위 명기.
- **Pass 조건**: C ≤ 6400MB AND (해당 시나리오 D ≤ 1440min). 두 D를 모두 보고하고, 시나리오 확정 시 해당 D로 게이트 판정.
- **Fail 시 분기**(005 §6-3): VRAM 우선(batch 8→4→length 64→32), 그다음 wall-clock(train_ratio 512→1024→steps 5e5→3e5). 이 분기도 시나리오 무관(per-step 조정).

→ **A19 dry-run은 지금 진행해도 확답이 어느 쪽이든 그대로 유효.**

기타 결정-무관: pytest 회귀(21/21), R3 ruamel coerce 실측(013 §6, 완료), G steps 단위 확정(013 §7, 완료).

---

## 4. 결정-종속 작업 (확답 전 착수 금지, 설계만 병기)

| 항목 | A에서 | B에서 | 확답 전 처리 |
|---|---|---|---|
| `scripts/eval_heldout.py` + A_heldout 가드 | 필요(013 §3) | 불필요 | 설계만 013에 보존, 구현 보류 |
| 도메인 랜덤화(#33 후보, start pose 등) | 권고 | 선택 | OPEN-U2 보류, 설계만 |
| Phase 3 `--fresh_optim`/warm-load(#21) | 불구현 | 구현 필요 | 양쪽 설계 메모, 착수 보류 |
| acceptance 임계 확정(A12/A13/A16) | 013 §9 | 005/004 #2 | 시나리오 확정 후 |
| snapshot logdir/용량(008) | 1 stage ~10.5GB | 2 stage ~21GB | 시나리오 확정 후 |

---

## 5. 확답 도착 시 피벗 절차

1. planning/015 작성: 확정 시나리오 명시 + 본 문서(014) PROVISIONAL 해제.
2. **A 확정 시**: 013을 정식 SSOT로 승격. eval_heldout/A_heldout 구현(Phase 5-2), 도메인 랜덤화(OPEN-U2) 결정.
3. **B 확정 시**: v3 #9/#21/#25/#31 부활을 명시. 013의 A11(completion-only)만 유지(009 결정 B는 시나리오 무관), 나머지 #32 재정의는 무효화. Phase 3에 `--fresh_optim` 포함.
4. A19 산출 D_A/D_B 중 해당 값으로 A19 Pass/Fail 최종 판정.

---

## 6. OPEN (사용자/외부 확답 대기)

- **OPEN-EXT (블로커 아님, Phase 5 게이트)**: Oschersleben 훈련 허용 유무 — 교수님 확답(점심 후).
- **OPEN-U1 (보류, "이따가")**: zero-shot 채택의 채점 trade-off 수용(시나리오 A에서만 유효).
- **OPEN-U2 (보류)**: 도메인 랜덤화 채택 여부·범위(시나리오 A에서 권고).

→ 위 셋 모두 **A19 진입을 막지 않는다**(§3). 이번 세션은 A19를 진행한다.
