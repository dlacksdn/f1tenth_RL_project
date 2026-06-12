# 001 — Centerline과 Reward 설계 (발표 준비용 Q&A 정리)

> 목적: 발표(10분, 영어) 핵심인 **reward 함수 설계**를 엄밀하게 설명하기 위한 정리.
> 코드 근거: `dreamer_f1tenth/envs/f1tenth_env.py`
> 관련 문서: implementation/002(measurement), 008(centerline 재추출), 015(reward)

---

## 0. 한 줄 결론

> **Centerline = 트랙을 1D 좌표 `s`로 매개변수화한 곡선(자/ruler). Reward는 차를 이 곡선에 투영해 얻은 `s`의 "전진량(종방향)"만 측정한다. 횡방향 이탈은 무관(자유), 트랙 이탈은 충돌 종료로만 제어. 트랙 길이는 곡선의 부산물 스칼라로 랩 판정·보너스 밸런싱에만 쓰인다.**

---

## 1. Centerline이란 무엇인가 / 왜 측정했나

### 1-1. centerline = 점들의 순서열(곡선), track length = 그 곡선의 끝점 스칼라
- **Centerline**: 트랙 리본 중앙을 따라가는 순서열 `(sᵢ, xᵢ, yᵢ, txᵢ, tyᵢ)`
  - `s` = 시작점부터 누적 호길이(cumulative arc-length)
  - `(x,y)` = 좌표
  - `(tx,ty)` = 단위 접선벡터(unit tangent, 트랙 진행 방향)
  - → **트랙을 1차원 좌표 `s`로 매개변수화한 곡선 전체**
- **Track length `L_track`** = `centerline.s[-1]` (스칼라 1개)

→ **포함관계**: centerline을 추출하면 track length는 공짜로 따라 나오는 부산물(끝점 `s`). 역은 불가 — 숫자 하나로 곡선을 복원 못 함.
- 산출물 실체: CSV 곡선(`maps/map_easy3_centerline.csv` 등). 문서 002가 헤드라인으로 `L_track` 숫자만 보고한 건 그게 사람이 검산 가능한 스칼라라서일 뿐.
- 추출 방법: `scripts/extract_centerline.py` — free-space mask → `skimage.morphology.skeletonize` → `keep_largest_cc` → `prune_branches` → direction-continuation walk.

### 1-2. 측정 동기: "보상 때문"은 절반만 맞다
centerline은 보상 하나가 아니라 **환경의 기하 백본**으로 3곳에 동시 사용:

| CSV 열 | 역할 |
|---|---|
| `s` (호길이) | **progress 보상** + **lap 판정**(종료조건) |
| `(x,y)` (좌표) | 차 위치 투영 → closest point 찾기 (위 둘의 전제) |
| `(tx,ty)` (접선) | **역주행(reverse) 감지** (종료조건) |

즉 정확히는: **reward(progress + R_lap) + 종료조건(2바퀴 lap, 역주행)** 세 메커니즘이 공유하는 단일 기준선.

---

## 2. 트랙 길이를 재야만 reward를 설계할 수 있나? → 핵심(progress)은 길이 불필요

reward 식 분해:
```
reward = progress  +  R_lap·(랩 이벤트)  −  10·(종료: 충돌/역주행/발산)
```

| 항 | L_track 필요? | 이유 |
|---|---|---|
| **① progress** `clip(s_now − s_prev, 0, 0.5)` | **사실상 불필요** | **국소 호길이 변화량(local Δs)**. 트랙이 100m든 275m든 무관. cap=0.5도 물리량(v_max·dt=20×0.02=0.4m+여유)에서 나옴, 길이 무관 |
| **② R_lap 보너스** | **필요** | 랩 판정 `total_arclen // L_track`이 R_lap 트리거 + 2바퀴 종료를 켬 |
| **③ 종료 페널티 −10** | 불필요 | 충돌/역주행/발산 플래그 기반 |

**핵심**: progress(dense signal)를 만드는 데 필요한 건 "트랙 길이(스칼라)"가 아니라 **"centerline 곡선(매개변수화)"**. 차의 (x,y)를 곡선에 투영해 `s`를 읽어야 전진량이 나오기 때문. 길이 숫자 하나로는 불가능.

L_track이 진짜 쓰이는 곳:
- **랩 카운팅**(②) — 누적 호길이가 총길이 배수를 넘으면 1바퀴.
- **R_lap 크기 보정** — R_lap=25(easy)/100(osch)은 "한 바퀴 progress 총합 ≈ L_track"에 대해 보너스가 ~25~36% 비중이 되도록 *밸런싱*한 값. 보상 작동 원리가 길이에 의존하는 게 아니라 *튜닝*에 참조.

### 2-1. 엄밀성 단서: progress에 L_track이 등장하는 유일한 지점 = seam 보정
출발선 통과 시 `s`가 `L_track`→`0`으로 점프 → `Δs`가 거대한 음수. 이를 `±L_track/2` wrap-around로 보정(f1tenth_env.py L346-349). **닫힌 루프의 modular 상수일 뿐, 보상을 스케일하지 않음.** seam 한 점 제외하면 progress는 완전히 길이-독립적.

---

## 3. "centerline을 지키며 주행 → reward↑, 벗어나면 reward↓" 인가? → 아니오 (정반대 이해 필요)

centerline은 **따라갈 경로(path)가 아니라 전진량을 재는 자(ruler)**.

코드 흐름 (f1tenth_env.py):
```
1. 차 (x,y) ──projection──> centerline 위 closest point        [L341 _windowed_closest_idx]
2. 그 점의 호길이 s 를 읽음                                       [L343]
3. reward = clip(s_now − s_prev, 0, 0.5)  ← s 증가량(전진)만      [L344]
```
- `_windowed_closest_idx`에서 **차–centerline 거리(`d2`)는 closest point 찾는 데만 쓰고 argmin 후 즉시 버림**(L268). reward에 **단 한 번도 안 들어감**.

### 결론: reward = 횡방향(lateral) 이탈이 아니라 종방향(longitudinal) 전진만 측정
- centerline 정중앙을 달리든 / 안쪽 벽에 붙든 / 코너를 깎든 → projection Δs 같으면 **reward 동일**.
- 차는 **횡방향 자유** → 정책이 알아서 **최적 레이싱 라인** 탐색 (centerline은 코너를 안 깎으므로 가장 느린 라인).
- 만약 cross-track-error 패널티를 넣었다면 차를 centerline에 묶어 느리게 만들었을 것. 이 설계는 **"전진 측정"과 "어디로 달릴지"를 분리** → 빠른 기록(map_easy3 6.1s/lap)의 한 요인.

### "트랙 이탈 금지"는 누가 강제? → 벽 충돌 종료
centerline 거리 패널티가 아니라 **wall collision → −10 + 종료**(L320). **경계선은 centerline이 아니라 wall.**

> **슬라이드용 한 줄**: *Reward = 차 위치를 centerline에 투영한 1D 좌표 `s`의 전진량. 횡방향 이탈 무관, 트랙 이탈은 충돌 종료로만 제어.*

---

## 4. 단위 접선벡터 `(tx,ty)`는 왜 저장하나 → 역주행 감지 전용

접선 = 각 centerline 점에서 트랙의 "앞" 방향. 쓰이는 곳은 reverse guard **한 군데뿐**(progress엔 미사용):
```
dot = vel_world · tangent[closest_idx]              [L366]
if vel_x < 0  AND  dot < 0:  reverse_counter++       [L371]
   → 50스텝(=1초) 연속 시 종료, cause='reverse', −10
```
- `dot < 0` = 속도벡터가 트랙 앞 방향과 반대 = 역주행.
- **지역(local) 접선이 필요한 이유**: 트랙이 휘므로 "앞"이 매 지점 다름. 전역 heading 하나로는 판정 불가.
- `vel_x < 0` 게이트: centerline 오정합 대비 안전장치(전진 정책의 false reverse 방지, L367-369).

### 왜 reverse 종료가 필요한가
progress는 `clip(...,0,...)`라 후진 시 reward 0(음수 아님)이지만, 그것만으론 **잘못된 방향 에피소드를 끝내지 못함**. 차가 뒤로 기거나 제자리 회전하며 에피소드 낭비 가능 → reverse guard가 −10으로 깔끔히 종료해 명확한 학습 신호 제공.

---

## 5. 주의: 문서 002의 L_track 값은 나중에 정정됨

- 문서 002 측정: map_easy3 **117.22m**, Oschersleben **312.61m**.
- 이 값은 **틀림**: `keep_largest_cc`가 트랙 리본이 아니라 **맵 바깥 경계(자유공간 97.6%)** 를 센터라인으로 잡음. (GapFollower가 y=52로 탈주한 게 징후, 002 §3-1)
- 문서 008에서 START_POSES 기반으로 올바른 리본 CC만 골라 재추출 → **map_easy3=100.57m, Oschersleben=275.18m** (코드 SSOT, f1tenth_env.py L68/L76).

---

## 6. 발표 reward 슬라이드 추천 흐름

1. **문제**: 완주만 보상하면 sparse → 학습 안 됨. dense reward 필요.
2. **아이디어**: 트랙을 1D 좌표 `s`로 매개변수화한 **centerline** 추출 → 매 스텝 "centerline 따라 전진한 호길이 Δs"를 보상. (베이스라인 DQN의 이산 waypoint idx → **연속 호길이**로 업그레이드)
3. **하나의 곡선이 3가지 해결**: progress 보상 + 정확한 랩 판정(f110 내장 lap_count는 더블카운트 버그) + 접선으로 역주행 감지.
4. **횡방향 자유**: reward는 종방향 전진만 측정 → 정책이 최적 레이싱 라인 자유 탐색. 트랙 이탈은 충돌 종료로 제어.
5. **트랙 길이**: 곡선의 부산물 스칼라 → 랩 판정·보너스 밸런싱에만. 보상 핵심 신호는 길이가 아니라 곡선.
