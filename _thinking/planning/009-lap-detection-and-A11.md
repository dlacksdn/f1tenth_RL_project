# 009 — lap 검출 방식 + A11 평가 기준 정밀화

> 2026-05-21. 노트북 세션. implementation/008 §6(map_easy3 lap_count double-count) 발견에서 파생.
> 선행: [005 v3 A11(line185)/A12/A13/A16, §4-3 R_lap](./005-f1tenth_dreamerV3_version3.md), [implementation/008 §6](../implementation/008-centerline-reextraction.md).
> 본 문서는 lap 검출 메커니즘과 A11 acceptance의 **신 SSOT**. Phase 4(episode/reward)·Phase 5(평가)가 준수.

---

## 1. 배경 — f110 lap_count double-count (map_easy3)

implementation/008 §6: corrected map_easy3에서 GF가 **1바퀴(94.1m < L_track 100.57m)도 못 돌고 `lap_count=2`** (start 근처 2.6m 구간에서 0→1→2). f110_env lap 카운터(start 근처 toggle, 결정 #30)가 본 트랙·start 조합에서 1바퀴당 ≥2회 toggle. Oschersleben은 정상(2 lap=550m).

**왜 무시 못 하나 (시간 측정 무관)**: Phase 4 §4-3 reward가 `lap_count 증가 → R_lap` 을 주므로, double-count는 (a) **R_lap reward hacking**(start 인근 왕복으로 보상 중복 수령) → Stage 1 학습 붕괴, (b) `lap_complete(lap_count≥2)` 종료가 1바퀴 미만에 발동. 즉 "완주만 하면 되는" map_easy3라도 **완주/보상 트리거 신호가 신뢰 가능해야** 한다.

---

## 2. 결정 A — lap 검출을 centerline arclength 기반으로 (사용자 승인 2026-05-21)

f110 `lap_count` 대신 **centerline arclength progress(s)가 L_track을 wrap**하는 것으로 lap 완주 판정.
- 양 맵 통일, start pose 위치 무관하게 robust, f110 toggle quirk 회피.
- Phase 4 progress reward가 어차피 arclength를 쓰므로 정합적.
- 구현 시 self-intersection 구간의 global-argmin closest_idx 점프를 막기 위해 **windowed closest-point progress 추적** 필요 (이전 idx ±window). → 본질적으로 **Phase 4 episode 설계 범위 → Phase 4에서 구현**.
- R_lap 트리거·lap_complete 종료 모두 이 arclength lap 신호 사용 (f110 lap_count 미사용).
- **구현 시점: Phase 4.** 본 분기(centerline 보정)는 결정 기록까지.

(reverse_guard의 vel_x<0 gate도 arclength/closest-point 신뢰성과 함께 Phase 4에서 재점검 — 현재는 안전장치로 유지, implementation/007 §4.)

---

## 3. 결정 B — A11을 completion-only로 완화 (사용자 승인 2026-05-21)

| | v3 원안 (A11, line185) | **신 (본 문서)** |
|---|---|---|
| map_easy3 Stage 1 평가 | median lap_time ≤ GF_baseline×1.5 (또는 fallback 45s) | **2-lap 완주율 기반 (completion-only)**. 정밀 lap_time 측정 안 함. |

- 근거: Stage 1(map_easy3)은 "주행 학습 확인" warmup. 정밀 lap_time은 Stage 2(Oschersleben A13: median≤120s, best≤110s)에만 둔다.
- 완주율 임계: 잠정 **≥ 80%** (Oschersleben A12와 정합). Phase 5 평가 설계 때 확정.
- map_easy3 GF baseline(implementation/008: ~12.34s/lap 실측)은 참고로 보존하되 A11 게이트에는 미사용.
- A16(Stage 1 재평가 2-lap 완주율 ≥70%)은 그대로 — 단 §2의 arclength lap 신호로 판정.

---

## 4. 영향받는 acceptance (Phase 5)

| Criterion | 변경 |
|---|---|
| A11 | lap_time → **2-lap 완주율(completion-only)** (결정 B) |
| A12 (Osch 완주율 ≥80%) | lap 판정을 arclength 신호로 (결정 A). 기준 유지 |
| A13 (Osch lap_time) | 유지. lap_time 측정도 arclength wrap 시점 기반 |
| A16 (easy3 재평가 완주율 ≥70%) | lap 판정을 arclength 신호로 (결정 A). 기준 유지 |

---

## 5. Phase 4 구현 체크포인트

- [ ] wrapper에 windowed closest-point progress 추적(s, 누적 lap) 추가.
- [ ] lap_complete 종료·R_lap 트리거를 arclength lap 신호로 (f110 lap_count 미사용).
- [ ] map_easy3·Oschersleben 모두 1 lap = s가 0→L_track 1회 wrap인지 검증.
- [ ] A11 평가 스크립트를 completion-only로.
