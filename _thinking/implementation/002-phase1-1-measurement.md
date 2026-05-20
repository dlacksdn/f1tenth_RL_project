# 002 — Phase 1-1: centerline 추출 + GapFollower baseline 측정

> 2026-05-20. 집컴(.venv/, RTX 4060 Ti). Phase 1-1 mandatory stop 도달.
> 선행: [001-phase1-0-deps.md](./001-phase1-0-deps.md)

---

## 1. 산출물

### 1-1. 신규 스크립트
- `scripts/extract_centerline.py` — skimage skeletonize 기반 centerline 추출 (v3 결정 #18 충족)
- `scripts/measure_gap_follower.py` — GapFollower baseline 측정 (v3 결정 #28 / §11 부록 B 의사코드 구현)

### 1-2. 신규 데이터 파일
- `maps/map_easy3_centerline.csv` (5492 pts, 5-col `s,x,y,tx,ty`)
- `maps/Oschersleben_centerline.csv` (6548 pts, 5-col `s,x,y,tx,ty`)
- `_thinking/notes/track_length.md`
- `_thinking/notes/gap_follower_baseline.md` (현재 Oschersleben 측정값 + map_easy3 DNF 기록)

---

## 2. L_track 측정 결과 (A_centerline)

| Map | L_track 측정 (m) | v3 추정 (m) | 비율 | v3 §0-2 해소-Q1 룰 |
|---|---|---|---|---|
| **map_easy3** | **117.22** | 70 | **1.67** | >30% 초과 → reward 코드 `L_track=117.22` 직접 대입 |
| **Oschersleben** | **312.61** | 300 | 1.04 | ±30% 이내 → 기존 표 유지 가능 |

### 2-1. centerline 알고리즘
- 입력: `pkg/src/pkg/maps/{name}.{png,yaml}`
- ROS convention thresholding: `pixel > (1 - free_thresh)*255` → free mask
- `skimage.morphology.skeletonize` → `keep_largest_cc` → `prune_branches(iterations=10)` → direction-continuation walk
- pixel→world: `x = ox + col*res, y = oy + (H-1-row)*res` (ROS 좌표계: origin = 좌하단)

### 2-2. map_easy3 L_track 신뢰도
- 추정 70m는 v3 §0-2의 rough estimate. 실측 117m가 정확값일 가능성 높음 (근거: Oschersleben이 4% 편차로 검증됨, 동일 알고리즘).
- 이미지 분석: black pixel 40,100개 (1.6%), bbox가 거의 이미지 전체. 얇은 벽이 복잡한 회로를 형성.
- map_easy3 자유공간 비율 97.6% → "거의 모든 픽셀이 free, 벽은 thin lines" 구조. centerline 추출 시 keep_largest_cc + prune_branches로 메인 loop 보존.

---

## 3. GapFollower baseline 측정 결과 (A_gap)

| Map | Median (s) | Min (s) | n / DNF | Fallback | 비고 |
|---|---|---|---|---|---|
| **Oschersleben** | **30.36** | 30.36 | 5 / 0 | ❌ | sim + GF deterministic → 5회 동일값. A_gap PASS |
| **map_easy3** | — | — | 0 / 5 | ✅ A11=45s | GapFollower로 측정 불가 → v3 결정 #28 fallback 채택 |

### 3-1. map_easy3 DNF 근본 원인
- 1차 시도 (dqn.py 동일 포즈 `[-0.2, -2.38, 1.745329]`): reset 직후 `obs['collisions'][0]=1` → step 1에서 즉시 done. wall=1s.
- 2차 시도 (centerline idx=0 = `[8.620, 11.860, 2.356]`): collision 없음. 차가 step 100~900에 걸쳐 y=11.86 → 52.19로 escape (map y 경계 13.6 초과).
- 진단: image edge는 wall로 처리되지 않음. map_easy3는 thin walls가 복잡한 회로를 만드는 구조라 LiDAR가 큰 gap을 image 외부 방향에서 발견 → GapFollower가 그 방향으로 steer. 알고리즘과 맵 구조의 mismatch.
- DQN 학습용 시작점 (-0.2, -2.38)이 reset 직후 false collision을 일으키는 이유는 별도 분석 필요했으나, fallback 결정에 영향 없어 보류.

### 3-2. Fallback 채택 근거
- v3 §0-2 해소-Q3: "측정 실패 시 A11=45초 절대값, A13 median=110초 절대값로 fallback"
- v3 §11 부록 B의 절차에 명시: "측정 완료 또는 fallback 채택 = Phase 1-1 게이트 통과"
- 본 케이스: 5/5 DNF + 원인 진단 완료 → fallback이 정당한 결정

---

## 4. 코드 픽스 / 결정 사항

### 4-1. `measure_gap_follower.py`의 `done = False` override
- dqn.py:167 동일 패턴 적용. reset 내부 zero-action step에서 false collision이 발생할 수 있음.
- 적용 위치: `run_episode()` 내, reset 직후.

### 4-2. START_POSES 결정
- `map_easy3`: `[8.620, 11.860, 2.356]` (centerline idx=0). dqn.py 포즈는 reset collision 발생.
- `Oschersleben`: `[0.0702245, 0.3002981, 2.79787]` (main.py 검증된 포즈).

### 4-3. extract_centerline.py 알고리즘 디테일
- `prune_branches(iterations=10)`: degree-1 픽셀 반복 제거. 분기/dead-end 제거하고 main loop만 남김.
- `order_skeleton_loop`: direction-continuation greedy walk. 최소 degree 픽셀에서 시작, prev_dir과 alignment 높은 후보 선호.

---

## 5. v3 acceptance criteria 매핑

- [x] **A_centerline**: map_easy3 + Oschersleben centerline CSV 5-col 생성. `python scripts/extract_centerline.py --verify` 실행 완료.
- [x] **A_gap**: GapFollower baseline 측정 완료 (Oschersleben 30.36s) 또는 fallback 채택 (map_easy3 A11=45s). v3 결정 #28 게이트 통과.

---

## 6. 미확정/사용자 결정 필요 항목

1. **A13 Oschersleben 기준 재검토**: GF=30.36s는 v3 A13 절대값(median≤120s, best≤110s)을 매우 크게 통과. baseline×1.5=45.5s로 강화할지? 또는 v3 절대값 유지?
2. **map_easy3 L_track=117.22m의 reward 식 반영**: v3 §4-3 reward 산술표는 L_track=70m 기준으로 검증되었음. L_track=117.22m로 갱신 시 progress reward scale 자동 재산정 (v3 해소-Q1 룰).
3. **(낮은 우선순위) map_easy3 dqn 시작점 false collision 분석**: 현재 fallback로 우회. 차후 Phase 1-2 wrapper 작성 시 같은 이슈가 재발할 수 있음.

---

## 7. 다음 단계 (Phase 1-2 진입 조건)

v3 §3 Phase 1-2: `dreamer_f1tenth/envs/f1tenth_env.py` — F110GymnasiumWrapper 작성.
- 4-tuple → 5-tuple 변환
- obs dict 5-key (lidar, state, is_first, is_terminal, is_last)
- reset(seed, options), terminated/truncated 분리
- 트랙별 default start (env_setting/001 §9, 본 문서 §4-2 START_POSES 그대로 채택 가능)

Phase 4 reward 코드 작성 시 본 문서 §2의 L_track 값을 직접 대입.

---

## 8. 체크리스트

- [x] scripts/extract_centerline.py 작성 + 양맵 실행
- [x] scripts/measure_gap_follower.py 작성 + 양맵 실행
- [x] maps/{name}_centerline.csv 생성 (2개)
- [x] _thinking/notes/track_length.md
- [x] _thinking/notes/gap_follower_baseline.md
- [x] L_track 확정 (map_easy3=117.22, Osch=312.61)
- [x] GapFollower baseline 확정 또는 fallback 채택
- [ ] 사용자 결정: §6 항목 1, 2
- [ ] Phase 1-2 진입
