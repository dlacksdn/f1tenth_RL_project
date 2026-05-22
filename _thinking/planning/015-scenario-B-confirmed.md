# 015 — 시나리오 B 확정 (Oschersleben 훈련 허용) + 014 PROVISIONAL 해제

> 2026-05-22. 집컴(Parsec 원격, GPU 가용) 세션. **교수님 확답 도착: Oschersleben을 학습에 써도 됨.**
> 본 문서는 [planning/014](./014-heldout-provisional-dual-scenario.md) §5-3 피벗 절차에 따라
> 결정 #32의 PROVISIONAL을 해제하고 **시나리오 B(훈련 허용 / curriculum 부활)** 로 확정한다.
> 충돌 시 우선순위: 본 문서(015) > 014 > 013 > 011. 시나리오 무관(invariant) 결정은 그대로 유효.

---

## 1. 확정 — 시나리오 B (014 §2 표 B열)

- **Oschersleben 훈련 허용** (교수님 확답). 014의 결정 #32 PROVISIONAL **해제**.
- 014가 강등했던 v3 원안 curriculum **부활**. zero-shot held-out 노선(시나리오 A) 폐기.

### 1-1. 부활하는 v3 결정 (014 §2 B열 / §5-3)
| # | 내용 |
|---|---|
| #9 | map_easy3 → Oschersleben 순차 fine-tune (curriculum), joint replay 0.3 |
| #21 | Stage 2 warm load (`latest.pt`) + **fresh optimizer** (lr 절반 옵션) |
| #25 | Stage 2 task = Oschersleben |
| #31 | A16 미달(Stage1 재평가 <70%) 시 rollback / joint_replay_ratio 0.5 |

### 1-2. 유지 (시나리오 무관 — 014에서도 양쪽 공통이라 명시)
- **009 결정 A** (arclength windowed-progress lap 판정): Phase 4서 구현 완료(implementation/015).
- **009 결정 B** (A11 = map_easy3 completion-only 2-lap 완주율 ≥80%): 유지.
- Phase 1~4 전부 유지(reward/env 인터페이스는 시나리오 무관 invariant).

### 1-3. 무효화 (시나리오 A 전용 — 착수 안 함)
- `scripts/eval_heldout.py` + A_heldout 가드 (013 §3): **불필요**. Oschersleben이 이제 학습+평가 트랙
  → dreamer.py 기본 eval 경로로 충분(014 §2 B열).
- 도메인 랜덤화(#33 후보, OPEN-U2): curriculum이 도메인 다양성 일부 제공 → **선택**(미채택, 보류).
- OPEN-U1(zero-shot 채점 trade-off): 시나리오 A 전용 → **소멸**(본 트랙 학습이므로 위험 낮음).
- 013의 #32 재정의(훈련 금지·zero-shot·OOD 지표·A12/A13 임계 OPEN) 중 시나리오 A 종속분 무효.

---

## 2. Phase 5 구조 (v3 §3 Phase 5 / 005 부활)

2-stage 총 ~1M env-step (Stage당 500K env-step = 250K agent-step, A19 기준 Stage당 ~20.9h).

| Stage | 트랙 | 방식 | 산출 |
|---|---|---|---|
| 1 | map_easy3 | scratch 500K. ckpt `latest.pt` | Stage1 모델 |
| 2 | Oschersleben | Stage1 `latest.pt` **warm load + fresh optim**(#21), joint replay 0.3(#9) 500K | Stage2 모델 |
| 3 | map_easy3 | Stage2 모델 재평가 (A16, catastrophic forgetting 체크) | A16 판정 |

- **★ Stage 1(map_easy3 500K)은 시나리오 A/B 공통 invariant** → 확답 전에도 시작 가능했고, 지금 시작해도 무손실.
- Stage 2 warm-load/fresh-optim(#21)은 **Phase 3 구현 필요**(014 §4서 보류했던 것 → 본 분기/후속서 구현).

## 3. Acceptance (시나리오 B = 005/004 원 임계 부활; 정확값은 Phase 5-2 평가 설계서 재확인)
- A11 (map_easy3): 2-lap 완주율 ≥80% (completion-only, 009 결정 B).
- A12 (Oschersleben): 2-lap 완주율 ≥80%. lap 판정은 arclength(009 결정 A).
- A13 (Oschersleben): lap_time 기준 — 005 §4-3·004 #2 원안 부활(median/best). Phase 5-2서 정확값 확정.
- A16 (map_easy3 재평가): 2-lap 완주율 ≥70%. 미달 시 #31 rollback.

## 4. 예산/저장 (008)
- 총 ~1M env-step (2 stage). snapshot logdir ~21GB (2 stage, 008 §). 집컴 디스크 확인.
- precision=16, batch_size=16, train_ratio=512 (007 fixed-HP). A19 PASS 프로파일 그대로.

## 5. 다음 작업
1. (본 세션) Stage 1 map_easy3 500K 본 학습 시작 (GPU 백그라운드).
2. Stage 1 학습 중 Stage 2 fine-tune 코드(#21 fresh_optim/warm-load, #9 joint replay) 준비.
3. Stage 1 완료 후: A11 평가(completion-only) → Stage 2(Osch) → A12/A13 → Stage 3 재평가(A16).

## 6. OPEN (잔여)
- OPEN-U2(도메인 랜덤화): 시나리오 B에선 선택 → 미채택 보류. 일반화 부족 시 재고.
- Stage 2 warm-load 세부(fresh optim lr 절반 적용 여부 #21): Phase 3 구현 시 확정.
