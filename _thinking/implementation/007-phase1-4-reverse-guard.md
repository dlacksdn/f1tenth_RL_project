# 007 — Phase 1-4: reverse_guard (A18) + Oschersleben centerline 오정합 BLOCKER 발견

> 2026-05-21. **노트북**(`env/`, torch 2.4.1+cpu) 세션. Phase 1-4 종료, mandatory stop.
> 선행: [006-phase1-3-dynamic-patch.md](./006-phase1-3-dynamic-patch.md).
> 관련 결정: [planning/005 v3 §3 1-4, §4-3 line 318, A18(line 164), #8/#24](../planning/005-f1tenth_dreamerV3_version3.md).

---

## 0. 세션 환경 메모

- 이 머신은 **노트북**(`env/`, torch+cpu). 집컴은 `.venv/`.
- env에 gymnasium·pytest 미설치 상태였음 → `pip install gymnasium==0.29.1 pytest` 설치(설치 완료, gymnasium 0.29.1 / pytest 8.3.5). f110-gym/gym/numpy/scipy/torch는 기존 존재.
- SSH 인증 **동작 확인**(`ssh -T git@github.com` → "Hi dlacksdn! successfully authenticated"). 부팅 노트의 "노트북 미세팅"은 갱신됨 — push 가능.
- 부수 작업으로 planning/007(fixed-HP fidelity), planning/008(snapshot 정책 정밀화) 작성·commit(6f67236 등). 본 분기와 별개.

---

## 1. 산출물 (코드)

| 파일 | 변경 |
|---|---|
| `dreamer_f1tenth/envs/f1tenth_env.py` | (a) `_MAPS_DIR` + TRACK_CONFIGS에 `centerline_csv` 경로, (b) `_load_centerline()` 헬퍼(csv→xy(N,2)/tangent(N,2)), (c) `__init__`에서 centerline 로드, (d) reset/`__init__`에 `_reverse_counter=0`, (e) `step()` reverse_guard dot 로직 + 종료 우선순위 collision>reverse>lap>timeout, (f) info에 `reverse_counter`, (g) `REVERSE_COUNTER_LIMIT=50` 상수 |
| `dreamer_f1tenth/tests/test_reverse_guard.py` (신규) | A18: 강제 후진(speed=-3) → env_step≤55 terminated ∧ cause='reverse' ∧ counter==50. + forward(speed=+3) 1s → counter=0 유지 |

### 1-1. reverse_guard 사양 (v3 option A + vel_x<0 gate, §4 결정)

매 env step 후:
```python
pos = (poses_x, poses_y); yaw = poses_theta
vel_x = linear_vels_x  # body-frame longitudinal (= state[3])
vel_y = linear_vels_y  # body-frame lateral (#27 patch)
vel_world = R(yaw) @ [vel_x, vel_y]
closest_idx = argmin(||centerline_xy - pos||²)   # ★ global argmin (개선 필요 — §3)
dot = vel_world · tangent[closest_idx]
if vel_x < 0.0 and dot < 0.0:   # ★ vel_x<0 gate (§4)
    reverse_counter += 1
else:
    reverse_counter = 0
# 우선순위: collision > (reverse_counter>=50) > lap(>=2) > timeout(>=9000)
```

---

## 2. A18 검증 결과

```
[A18] reverse terminated at env_step=50, reverse_counter=50, cause='reverse'   (map_easy3)
[A18-reset] forward 1s: max reverse_counter=0 (expect 0)
pytest 전체: 10 passed (8 기존 + A18 2개)
```

| Criterion | 기준 | 실측 | 결과 |
|---|---|---|---|
| A18 | 강제 후진 1.1s → terminated ∧ cause='reverse' | env_step=50 종료, cause='reverse' | **PASS** |
| A18-reset | 전진 시 counter reset | max counter=0 | **PASS** |
| 회귀 A1~A6,A5,A_norm | 무영향 | A_norm N=15015 (006과 동일, vel_y/5=0.751, ang_z/(2π)=0.989) | **PASS** |

- A18은 **map_easy3**에서 검증. (Oschersleben centerline 결함 — §3 — 때문에 map_easy3 채택. map_easy3는 002에서 start pose=centerline idx0로 정합돼 있음.)

---

## 3. ★ BLOCKER — Oschersleben centerline 오정합 (Phase 4 진입 전 필수 해소)

### 3-1. 증상
- reverse_guard 1차 구현(vel_x gate 전, dot<0만) 직후 A_norm quick N이 15015→1390 급감.
- 추적: **GapFollower(베이스라인)가 정상 전진 중 step 278에서 `cause='reverse'` 오종료** (seed 0/1/2 동일).

### 3-2. 근본 원인 — centerline이 실제 주행 루프와 불일치
계측 증거 (Oschersleben, GF 1-lap, guard 비활성 scratch):

| 항목 | 값 |
|---|---|
| 차량–centerline 거리 d_cl | 평균 7.8m, **최대 19m** (트랙 half-width ~6m 초과) |
| closest_idx 불연속 점프 | t=215에서 288→**807** (anti-parallel segment에 Euclidean snap) → tangent 반대 → dot<0 누적 |
| world extent (yaml origin=[-55.08,-33.58], res=0.04295, 2000²) | x[-55.1,30.8] y[-33.6,52.3] |
| **centerline y범위** | **−20.6~39.9 (60.5m)** |
| **차량 주행 y범위** | **−6.3~26.2 (32.5m)** |

→ centerline이 차가 한 번도 안 가는 y<−6.3 영역까지 ~14m 뻗음. centerline y범위가 주행 y범위의 **≈2배**. 단순 offset/scale이 아니라 **형상 자체가 다른(더 큰) 루프**. windowed projection으로도 해소 안 됨(d_cl 오히려 15m로 악화) → centerline 데이터 자체 문제 확정.

### 3-3. 파급
1. **Phase 4 progress reward**: 동일 projection으로 arclen_delta 계산 → idx 288→807 점프 시 reward 폭발. **학습 불가.**
2. **L_track=312.61** (이 centerline arclength): 루프 부풀려졌으면 부정확 → §4-3 reward scale(line 363, 0.055 m/step / R_lap) 연쇄 오류 가능.
3. reverse_guard: §4 gate로 우회했으나 dot 신호 자체는 Oschersleben에서 신뢰 불가.

### 3-4. Phase 4 진입 전 필수 작업 (TODO)
- [ ] `scripts/extract_centerline.py` 디버그: Oschersleben에서 잘못된 connected component/branch 선택 또는 정합 오류 원인 규명. (map_easy3는 정상 — start=idx0 정합 확인됨.)
- [ ] 재추출 centerline을 free-space 픽셀 + 차량 GF 궤적에 오버레이 검증 (d_cl이 half-width 내인지).
- [ ] L_track 재측정 → 004 §1(312.61) 갱신 여부 결정.
- [ ] 재추출 후 reverse_guard dot 신호 Oschersleben 재검증 + Phase 4 progress reward projection 신뢰성 확인.

---

## 4. 사용자 결정 기록 (2026-05-21)

1. **reverse_guard dot 방식**: v3 option A(centerline tangent·velocity dot) 채택.
2. **centerline 오정합 BLOCKER 대응**: **reverse_guard에 `vel_x<0` gate 추가** + centerline 재추출은 Phase 4 전 별도 분기.
   - 근거: counter 증가 조건을 `vel_x<0 AND dot<0`로 강화. 참 후진(body-frame longitudinal state[3]<0)일 때만 → GF 등 전진(vel_x>0) 정책의 false reverse 원천 차단. v3 option A의 tangent 조건은 유지(게이팅만 추가).
   - 효과 검증: GF Oschersleben **2-lap 완주**(cause='lap_complete', step 3003, maxrev=0). A18 여전히 PASS. A_norm N 복원.

---

## 5. v3 acceptance SSOT 누적

| Criterion | 본 분기 후 |
|---|---|
| A18 | **PASS** (map_easy3, env_step=50 종료, cause='reverse') |
| reverse_guard 사양 | v3 option A + **vel_x<0 gate** (본 문서 §4, robust 안전장치) |
| Oschersleben centerline | **BLOCKER — Phase 4 전 재추출 필수** (§3) |

---

## 6. 다음 단계

- 즉시 분기 종료: 본 007 commit + push (HEAD 6f67236에서 분기).
- 다음 mandatory stop: Phase 2-0 직전 (precision=16↔NM512 `_use_amp` 매핑 + document-specialist로 `models.py:182` 사전 분석 권장).
- **Phase 4 진입 전**: §3-4 centerline 재추출 분기 선행 필수.

---

## 7. 체크리스트

- [x] f1tenth_env.py reverse_guard 구현 (centerline 로드 + dot + vel_x gate + 우선순위).
- [x] test_reverse_guard.py (A18 + forward reset).
- [x] pytest 10/10 PASS, 회귀 없음.
- [x] vel_x<0 gate로 GF false reverse 해소 검증 (2-lap 완주).
- [x] Oschersleben centerline BLOCKER 기록 (§3, Phase 4 전 해소).
- [ ] commit + push.
