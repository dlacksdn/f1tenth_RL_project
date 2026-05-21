# 008 — centerline 재추출 (Phase 1-1 BLOCKER 수정) + 신 SSOT

> 2026-05-21. 노트북(`env/`). Phase 1-4(007)에서 발견한 Oschersleben centerline 오정합 BLOCKER를 근본 수정.
> 선행: [007-phase1-4-reverse-guard.md §3](./007-phase1-4-reverse-guard.md), [002-phase1-1-measurement.md](./002-phase1-1-measurement.md).
> 본 문서는 centerline·L_track·map_easy3 default_pose·GF baseline의 **신 SSOT** (004/002 해당 값 supersede).

---

## 1. 근본 원인

맵 PNG는 벽을 **얇은 선**으로만 그려 free 픽셀이 Oschersleben 99.0% / map_easy3 97.6%다. 벽이 닫힌 루프라 free 공간은 **바깥 / 트랙리본 / 인필드** 의 분리된 CC로 나뉜다.
기존 `extract_centerline.py`는 `skeletonize(free_mask)` 후 `keep_largest_cc`로 **가장 큰 free CC(=바깥 영역, Osch 83%)** 의 skeleton을 골랐다 → 트랙이 아니라 맵 바깥 경계를 그리는 엉뚱한 큰 루프. (시각 오버레이로 확정.)

추가로, skeleton walk 방향이 임의라 **centerline +s(arclength 증가) 방향이 주행 방향과 반대**일 수 있음(Oschersleben이 그러함) → Phase 4 progress(arclen_delta)가 전진 시 음수가 되는 문제.

---

## 2. 수정 (`scripts/extract_centerline.py`)

1. **ribbon CC 선택**: `START_POSES`(트랙 위 (x,y,yaw))의 (x,y) 픽셀이 속한 free CC를 4-connectivity(`label(connectivity=1)`, 얇은 벽이 대각 누출 차단)로 찾아 **그 CC만 skeletonize**. (`keep_largest_cc`는 START_POSES 없는 맵의 legacy fallback로만 잔존.)
2. **orientation 정렬**: ordered centerline에서 start (x,y) 최근접점의 local 방향과 start yaw heading의 dot<0이면 **루프 순서 reverse** → +s = 전진 방향. (Oschersleben에서 발동.)

`START_POSES = {'map_easy3': (1.02,-14.66,-2.819842), 'Oschersleben': (0.0702245,0.3002981,2.79787)}` — wrapper default_pose와 일치.

---

## 3. 검증 (시각 오버레이 + 정량)

| 맵 | ribbon CC | L_track | dot<0 (GF, forward) | 비고 |
|---|---|---|---|---|
| Oschersleben | 7.0% | **275.18m** | **0.0%** (orientation reverse 후) | 리본 정중앙 정확 (overlay 확인) |
| map_easy3 | 29.6% (green serpentine) | **100.57m** | 2.8% (max streak 13) | green ribbon 정중앙 (overlay 확인) |

- 오버레이 이미지로 양 맵 centerline이 두 벽 사이 정중앙을 따라감을 직접 확인.
- dot<0 비율: 전진 주행 시 거의 0 → **Phase 4 progress reward 정합 확보**. reverse_guard도 dot 신호만으로 정상 동작(007 vel_x<0 gate는 이제 redundant지만 안전장치로 유지).

---

## 4. 신 SSOT (004/002 supersede)

| 항목 | 기존 | **신값 (본 문서)** | 출처 |
|---|---|---|---|
| Oschersleben L_track | 312.61 (004) | **275.18m** | corrected ribbon centerline |
| map_easy3 L_track | 117.22 (004) | **100.57m** | green-ribbon centerline |
| map_easy3 default_pose | (8.620,11.860,2.356) 트랙밖 (002 §4-2) | **(1.02, −14.66, −2.819842)** on-track, clearance 2.37m | green ribbon max-clearance |
| Oschersleben default_pose | (0.0702245,0.3002981,2.79787) | **유지** (on-track 검증됨) | — |
| map_easy3 GF baseline | DNF→fallback A11=45s (002) | **~12.34s/lap 실측** (5 ep 동일) | 본 문서 §5 |
| Oschersleben GF baseline | 30.36s (002) | **30.52s/lap** (재확인, ≈일치) | — |

→ `dreamer_f1tenth/envs/f1tenth_env.py` TRACK_CONFIGS의 default_pose·L_track을 위 신값으로 갱신함.
→ Phase 4 reward 식의 L_track plug-in은 본 표 사용. **A11 fallback 45s → 실측 12.34s 기반 재검토 가능** (median ≤ baseline×1.5 ⇒ ~18.5s).

---

## 5. GF baseline 재측정 (corrected setup)

- map_easy3 (on-track start): 5 ep 모두 주행 성공(DNF 해소). 1-lap **12.34s** (deterministic 동일).
- Oschersleben: 1-lap 30.52s, 2-lap 60.06s (002의 30.36s와 일치).

---

## 6. ★ 신규 발견 — map_easy3 lap_count double-count (Phase 4 전 결정 필요)

corrected map_easy3에서 GF 종료 시 `lap_count=2`인데 **누적 주행거리 94.1m < L_track 100.57m** (1바퀴 미만). lap_count 0→1 (617 step, 91.5m), 1→2 (637 step, 94.1m) — 2.6m 만에 두 번 증가.
- 원인 추정: f110_env lap 카운터(start 근처 toggle, #30)가 본 start pose 근처에서 1바퀴에 2회 이상 toggle. (serpentine이 start 인근을 재통과하거나 출발 직후 toggle 중복.)
- 영향: **A12(2-lap 완주율)·A13·A16·lap_complete 종료(lap_count≥2)가 map_easy3에서 신뢰 불가.** Oschersleben은 정상(2 lap=60s/550m).
- 해결 후보(미결정): (a) lap 카운팅을 f110 lap_count 대신 **centerline arclength s wrap(>= L_track)** 기반으로 (Phase 4 episode 설계), (b) double-count 안 되는 start pose 탐색, (c) Phase 4까지 보류.
- **centerline BLOCKER와는 별개 이슈.** 본 분기에서는 발견·기록까지.

---

## 7. 산출물

- `scripts/extract_centerline.py` (ribbon CC + orientation 수정).
- `maps/map_easy3_centerline.csv` (4475 pts, L=100.57), `maps/Oschersleben_centerline.csv` (5763 pts, L=275.18) 재생성.
- `dreamer_f1tenth/envs/f1tenth_env.py` TRACK_CONFIGS default_pose·L_track 갱신.
- `_thinking/notes/track_length.md` 재생성.
- pytest 10/10 PASS (A18 포함 회귀 없음, default_pose·centerline 변경 반영).

---

## 8. 체크리스트

- [x] free CC 구조 진단 + 시각 오버레이로 버그 확정.
- [x] extract_centerline.py ribbon CC 선택 + orientation 정렬.
- [x] 양맵 재추출 + 오버레이 검증 + dot 방향 검증.
- [x] wrapper TRACK_CONFIGS 갱신 (default_pose, L_track).
- [x] GF baseline 재측정 (map_easy3 DNF 해소).
- [x] pytest 10/10.
- [ ] commit + push.
- [ ] map_easy3 lap double-count 결정 (§6, Phase 4 전).
