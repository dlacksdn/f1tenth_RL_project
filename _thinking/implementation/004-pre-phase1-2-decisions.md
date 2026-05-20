# 004 — Phase 1-2 진입 전 사용자 결정 기록

> 2026-05-20. implementation/002 §6 미확정 3항목에 대한 사용자 결정 확정.
> 선행: [002-phase1-1-measurement.md](./002-phase1-1-measurement.md), [003-sync-policy.md](./003-sync-policy.md).
> 후속: Phase 1-2 wrapper 작성 (`dreamer_f1tenth/envs/f1tenth_env.py`).

---

## 1. 결정 #1 — map_easy3 L_track = 117.22m 채택

- **결정**: reward 코드의 `L_track` 변수에 측정값 **117.22m** 그대로 대입.
- **근거**:
  - v3 §0-2 해소-Q1 룰: 추정 70m 대비 측정값 +67% 편차(>30%) → 코드 변수만 갱신, reward 자동 재산정.
  - Oschersleben이 동일 알고리즘으로 +4% 편차로 검증됨 → centerline 추출 신뢰도 확보.
- **영향 범위**:
  - Phase 4 reward 코드 (`dreamer_f1tenth/rewards/` 또는 wrapper 내부 progress reward).
  - v3 §4-3 reward 산술표는 L_track=70m 기준의 _예시값_. 실제 코드는 L_track=117.22m 기반으로 작성하고, 산술 결과를 재계산하여 별도 노트로 검증 예정.
  - Oschersleben은 v3 표 그대로 유지 (편차 ±30% 이내).

## 2. 결정 #2 — A13 Oschersleben 기준을 baseline×1.5 = 45.5s로 강화

- **결정**: A13 Oschersleben acceptance 기준을 **median ≤ 45.5s, best ≤ 45.5s** 로 강화 (GF baseline 30.36s × 1.5).
- **근거**:
  - GF 측정값 30.36s가 v3 §2 절대값(median≤120s, best≤110s)을 압도 → 절대값을 그대로 두면 Dreamer가 GF보다 훨씬 느려도 통과되어 평가 의미 상실.
  - v3 §11 부록 B 의사코드의 "baseline×1.5" 룰을 그대로 채택.
- **영향 범위**:
  - Phase 5 평가 스크립트 (Stage 2 종료 후) — Oschersleben 통과 기준 변경.
  - v3 §2 acceptance criteria 표는 v3 문서이므로 수정 금지. 본 004 문서가 단일 출처(SSOT).
  - map_easy3 A11은 fallback 45s 그대로 유지 (GF 측정 실패로 baseline×1.5 산정 불가).

## 3. 결정 #3 — map_easy3 dqn 시작점 false collision은 Phase 1-2에서 함께 처리

- **결정**: `[-0.2, -2.38]` reset 직후 `obs['collisions'][0]=1` 발생 원인 분석은 **Phase 1-2 wrapper 작성 중 동시 진행**.
- **근거**:
  - wrapper의 `reset()` 내부에서 zero-action step을 처리할 때 같은 이슈가 재발할 가능성 → 표준화 작업과 분석을 한 분기에서 묶는 것이 효율적.
  - dqn.py:167의 `done = False` override 패턴을 wrapper에서 정식 처리할지, 또는 시작점을 strict하게 검증할지 결정 필요.
- **처리 방침**:
  - wrapper의 reset 후 첫 step에서 `collisions[0] == 1` 이면 false collision suspect로 분류, 해당 step의 collision flag를 무시하는 옵션을 wrapper 인자로 노출.
  - 분석 결과에 따라 START_POSES를 갱신할 수 있음. 현재 GF 측정에 사용한 centerline idx=0 (`[8.620, 11.860, 2.356]`)은 wrapper 기본값으로 채택.

---

## 4. Phase 1-2 진입 조건 (체크리스트)

- [x] 결정 #1, #2, #3 사용자 확정 (본 문서).
- [x] L_track 값 117.22m 확정 (map_easy3), 312.61m 확정 (Oschersleben).
- [x] START_POSES 초안 확정 (implementation/002 §4-2).
- [ ] 본 문서 commit + push (003 §3 절차).
- [ ] `dreamer_f1tenth/envs/f1tenth_env.py` 작성 (v3 §3 Phase 1-2).
- [ ] A1~A5, A_norm pytest 통과.

---

## 5. v3 acceptance criteria 갱신 매핑 (SSOT)

본 문서는 v3 §2 acceptance criteria의 다음 항목을 갱신·구체화한다 (v3 문서는 append-only로 수정 금지, 본 문서가 우선):

| Criterion | v3 원안 | 본 결정 후 |
|---|---|---|
| L_track (map_easy3) | 70 (추정) | **117.22** (측정 확정) |
| L_track (Oschersleben) | 300 (추정) | **312.61** (측정 확정) |
| A13 Oschersleben median | ≤ 120s | **≤ 45.5s** (baseline×1.5) |
| A13 Oschersleben best | ≤ 110s | **≤ 45.5s** (baseline×1.5) |
| A11 map_easy3 | 45s fallback | **45s 유지** (GF 측정 실패) |

이후 Phase 5 평가 코드는 본 표를 참조.
