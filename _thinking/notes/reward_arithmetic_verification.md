# Reward 산술 검증 — Phase 4 (L_track 실측 기반 재계산)

> 2026-05-22. 집컴 세션. Phase 4 reward + arclength lap 구현(f1tenth_env.py) 직후 검증.
> 근거: planning/005 §4-3 산술표 + 009 결정 A(arclength windowed progress). 코드 SSOT L_track.
> 005 §4-3 표는 **추정 L_track(70/300m)** 기반이었음 → 실측(100.57/275.18m)으로 재계산.

## 1. 단위 (005 §4-3, 013 §7 확정)
- env step = action_repeat(2) × sim_timestep(0.01s) = **0.02s**, **50 env step/s**.
- progress = env-step당 centerline arclength 증분(m), α=1.0, step-cap clip(0, 0.5).
- step-cap 근거: v_max=20 m/s × 0.02s = **0.4m/step** 물리 상한 → 0.5 cap은 0.1m 여유.
  정상 주행은 cap 미접촉, shortcut/wrap-artifact만 차단.

## 2. 트랙별 재계산 (실측 L_track)

| 트랙 | L_track(실측) | 목표 lap_time | env step/lap | progress/step (=L/(T·50)) | lap당 progress 합(α=1) | R_lap(적용) | R_lap/lap합 |
|---|---|---|---|---|---|---|---|
| map_easy3 | **100.57m** | 50s | 2500 | 0.0805 m/step | **100.57** | **25** | 24.9% |
| Oschersleben | **275.18m** | 110s | 5500 | 0.0500 m/step | **275.18** | **100** | 36.3% |

- progress/step(0.08, 0.05) ≪ step-cap 0.5 → **정상 주행은 cap에 안 닿음**(확인).
- R_lap 비율 24.9% / 36.3% → 005 §4-3 권장 "30~50%" 근방. map_easy3는 약간 낮으나
  005 적용값(25/100)을 **그대로 유지**(009 결정 B: map_easy3는 completion-only warmup,
  R_lap 정밀 균형보다 progress 신호가 주). R_lap 통일 금지 원칙 준수.

## 3. GF 실측 대조 (구현 검증, seed=0, max_step 충분)

| 트랙 | 종료 | lap_arc | f110 lap | total_arclen | progress합 = rew_sum − R_lap합 | L_track·lap 대비 |
|---|---|---|---|---|---|---|
| map_easy3 | collision@1457 | 1 | 2 | 175.3m | 185.17 − 25(+10 충돌) ≈ 170.2 | 1.74·100.57=175 ✓ |
| Oschersleben | lap_complete@3005 | 2 | 2 | 550.4m | 750.27 − 200 = 550.27 | 2·275.18=550.36 ✓ |

→ **progress 보상 합 ≈ 실제 누적 주행거리**(step-cap·wrap 오차 <0.05 상대오차, test_progress_sum_approx_arclength).
→ R_lap: Osch 2회×100=200, map_easy3 1회×25(완주 1바퀴 후 충돌). 경계 왕복 farming 없음(high-water-mark).

## 4. f110 lap_count double-count 회피 검증 (009 §1 핵심)
- map_easy3: f110 lap_count=2를 **무시**(info에 디버그로만 기록), arclength lap_arc=1 사용.
  → 009 §1의 start 근처 double-count(1바퀴 미만에 lap=2)를 arclength wrap이 정확히 회피.
- lap_complete·R_lap·progress 모두 arclength 신호. f110 lap_count는 판정 미사용.

## 5. termination·페널티
- 우선순위: diverged > collision > reverse > lap_complete(arclength 2-lap) > timeout (#24).
- collision/reverse/diverged = **-10**(005 §4-1 + smoke_findings #4: 발산도 비정상 실패 → 페널티).
- lap_complete: is_last(성공 종료, is_terminal=False). 실패 종료(diverged/collision/reverse)는 is_terminal.

## 6. 방향 가드 (005 §4-1 velocity_dot_tangent>0 대체)
- 005 원안은 f110 lap_count 증가 ∧ dot>0. arclength 구현은 **total_arclen high-water-mark**가
  방향 가드 내장: 역방향 누적은 lap 감소 → 재전진해도 새 high-water 아니면 R_lap 미지급.
  경계 ±오실레이션 reward farming 원천 차단(R17 mitigation).

## 7. 미해결/후속
- map_easy3 centerline start-end 3.819m 갭(green-ribbon 추출 한계, 닫힌 루프인데 열림).
  → wrap 보정(±L_track/2)이 흡수, step-cap이 잔여 점프 차단. lap 판정 정상 동작 확인(GF 실측).
  Oschersleben은 0.043m로 사실상 닫힘.
- reward_calibrate.py(005 §4-3 헬퍼)는 L_track이 이미 코드 SSOT라 **불필요** → 미구현.
