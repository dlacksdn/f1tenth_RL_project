# 005 — Reward 함수 알기 쉽게 + f110 기본 reward를 왜 버렸나 (발표 핵심 60%)

> 목적: 발표 알고리즘 핵심인 **progress reward 설계**를 비유로 쉽게 + 코드로 엄밀하게 정리.
> 추가로 "f110_gym 기본 reward가 있는데 우리 건 어떻게 연동되나?"(=무시·대체)를 코드 근거로 확정.
> 코드 근거: `dreamer_f1tenth/envs/f1tenth_env.py`, `gym/f110_gym/envs/f110_env.py`.
> 관련: understand/001(centerline=ruler·종방향), 002(reward·종료는 wrapper 소유), 003(reverse),
>       implementation/015(reward arclength), notes/reward_arithmetic_verification.md.
> ★ 001 정정: 002 표 C가 "base env reward(0)"이라 했으나 실제 base reward는 0이 아니라
>   시간+LiDAR 휴리스틱(아래 §6). wrapper가 그 값을 받아 버리고 자체 계산하는 건 동일.

---

## 0. 한 문장 핵심

> **트랙을 1차원 줄자로 펴놓고, 매 순간 "방금 몇 미터 앞으로 갔나"를 보상으로 준다.**
> 나머지(windowed/wrap/cap/high-water-mark)는 전부 "이 측정이 망가지지 않게 하는 안전장치"다.

---

## 1. 왜 dense reward가 필요한가 (문제 정의)

- 가장 단순한 보상 "완주=+100, 아니면 0"은 **sparse**: 초기엔 완주를 절대 못 함 → 보상이 계속 0
  → 뭘 잘했는지 신호 없음 → **학습 불가**.
- 해결: **매 스텝 조금씩 주는 dense 보상**. "방금 앞으로 나아갔으면 칭찬".
- (베이스라인 DQN은 이산 waypoint 인덱스를 썼고, 우리는 **연속 호길이**로 업그레이드 — 001 §6.)

---

## 2. "앞으로 갔다"를 어떻게 재나 — 트랙을 줄자로 (arclength)

- 트랙 한가운데 선(**centerline**)을 미리 그려두고, 출발점부터 **거리를 누적**해 번호를 매김:
  0m, 1m, 2m, … 이게 **arclength(호길이), 코드의 `s`**. 트랙을 쭉 펴서 만든 **1차원 자**.
- 차의 (x,y)를 이 자에 **수직 투영(projection)** → "지금 몇 m 지점이냐(`s`)"를 읽음.
- 코드: `_load_centerline`이 `s,(x,y),tangent` 반환. 투영 = `_windowed_closest_idx`로 closest point.

```
출발 ●━━━━━━━━━━━━━━━━━━━● 한바퀴
     0m  10m  20m …  L_track(map_easy3 100.57 / Osch 275.18)
```

---

## 3. 보상의 본질 = 이번 스텝에 늘어난 거리

```
progress = s_now − s_prev        # 앞으로 간 거리(m)   [f1tenth_env.py:342-344]
reward   = clip(progress, 0, 0.5) # 후진은 0으로 깎음     [:390]
```
- 1m 전진 → +1, 정지 → 0, 후진 → 0(벌 없음·상 없음).
- **이게 전부.** 아래 §4는 이 측정이 깨지지 않게 하는 안전장치.

---

## 4. 안전장치 3개 (015가 복잡해 보이는 이유)

| # | 문제 | 해결 | 코드 |
|---|---|---|---|
| ① **줄자 점프** | 8자형 교차구간에서 투영이 반대편으로 튐 → 가짜 순간이동 | "직전 위치 근처(앞 1.5m/뒤 0.5m)에서만 찾기" (windowed closest-point) | `:261-269` |
| ② **결승선 통과** | 한 바퀴 돌면 s가 100m→0m로 뚝 → 가짜 −100m 후진 | 변화량이 ±L_track/2 넘게 크면 한 바퀴 길이 가감(wrap 보정) | `:346-349` |
| ③ **이상치 상한** | ①②에도 남는 튐 | 한 스텝 보상 최대 0.5m로 cap (v_max 20×0.02=0.4m+여유) | `:390` |

- ①의 window는 **거리(m) 기반**(`SEARCH_FWD_M=1.5/BACK_M=0.5`)이라 reset 시 트랙별 점 간격으로
  인덱스 자동 환산 → **매직넘버·맵 overfit 없음** (015 §1-1).

---

## 5. 한 바퀴 보너스(R_lap) + 실패 페널티

### 5-1. R_lap (랩 완주 보너스)
- 누적거리가 트랙 한 바퀴를 넘을 때마다 보너스: **map_easy3 +25 / Oschersleben +100** (`:420-421`).
- **꼼수(reward farming) 차단**: 결승선 앞뒤 왕복으로 반복 수령 못 하게,
  "지금까지 도달한 최대 바퀴 수(**high-water-mark**)를 넘을 때만" 지급 (`:352-354`).
- **2바퀴(LAP_TARGET=2) 완주 → 에피소드 성공 종료**(`cause='lap_complete'`, `:422-424`).
- f110 내장 lap_count는 map_easy3에서 double-count 버그 → **미사용**, 우리 arclength로 자체 판정.

### 5-2. 종료 페널티 −10
- 충돌 / 역주행 / 발산(diverged) → **−10 + 즉시 종료** (`:407-417`). 우선순위 diverged>collision>reverse.

### 전체 보상 한 줄
```
reward = 전진거리(0~0.5) + 랩보너스(25 or 100) − 10(충돌·역주행·발산)
```
- 횡방향 이탈은 무관(레이싱 라인 자유), 트랙 이탈은 충돌 종료로만 제어 (001 §3).

---

## 6. ★ f110_gym 기본 reward는 어떻게 연동되나? → 무시·완전 대체

### 6-1. 기본 reward는 실제로 존재한다 (0이 아님)
`gym/f110_gym/envs/f110_env.py:660-686`에 손으로 짠 휴리스틱:
```python
reward = 1000 * self.timestep                  # ① 생존 보상: 안 죽으면 매 스텝 +10
if 300 <= argmin(scans) <= 780: reward -= 1     # ② 가장 가까운 벽이 정면이면 −1
elif ...:                       reward += 2     #    측면이면 +2
if min(scans) < 0.5:            reward -= 5     # ③ 벽 너무 가까우면 −5
if collision:                   reward = 0      # ④ 충돌 시 0
```
- 과제 베이스라인 쪽이 끼워넣은 reward shaping. 위쪽 `goals=[...]` 체크포인트는 주석 처리(죽은 코드).

### 6-2. 우리는 통째로 버린다
```python
raw, _r, base_done, _i = self._env.step(action_2d)   # _r(base reward) → 언더스코어로 버림 [:314]
...
reward = progress_r + collision_r + reverse_r + diverged_r + lap_r   # 자체 계산 [:430]
```
- base env에서 **실제 쓰는 건 obs(raw)와 done뿐**, **reward는 100% 우리 것으로 대체**.
- = understand/002 "reward·종료는 wrapper 소유"의 구체적 근거.

### 6-3. 왜 버렸나 (발표 차별점)
| 기본 reward(휴리스틱) | 우리 reward(arclength progress) |
|---|---|
| **시간 생존**(+10/step) → 천천히 안 죽기 = 고득점 (레이싱과 정반대) | **전진 거리** → 빨리 멀리 = 고득점 |
| LiDAR 빔 인덱스 등 손으로 짠 규칙(조잡·트랙 의존) | 트랙 기하(centerline) 기반 원칙적 측정 |
| "어디까지 갔나" 개념 없음 | 누적거리로 랩 판정·완주까지 일관 |

→ ①"생존 보상"이 빠른 주행 동기를 안 주는 게 핵심 결함. "시간이 아니라 전진 거리를 보상"이
   빠른 랩타임(map_easy3 6.1s)의 근거.

---

## 7. ★ 효과 증거 (reward 투입 전후, 015 §2-3)

| 지표 | reward=0 (이전) | progress reward 투입 후 |
|---|---|---|
| value(가치 추정) | ~1e-7 (죽은 학습) | **25.5** (살아있는 학습) |
| 에피소드 길이 | 짧음 | **884 step** (오래 주행) |
| model_loss | — | 58 → **2.0 수렴** |

- reward 산술 검증(notes): GF 실측 progress 합 ≈ 누적 주행거리 (Osch 550.27 vs 550.4, rel_err<5%).
- → "dense reward 설계가 학습을 켰다"를 숫자로 입증.

---

## 8. 발표 슬라이드 추천 흐름 (4컷)

1. **문제**: 완주만 보상하면 신호 0 → 학습 불가 (sparse).
2. **아이디어**: 트랙을 줄자(centerline arclength)로 펴서 매 스텝 "전진 거리" 보상 (dense).
3. **공식**: `전진(0~0.5) + 랩보너스(25/100) − 충돌벌점(10)`. 횡방향 자유(레이싱 라인 탐색).
   기본 f110 reward(시간+LiDAR 휴리스틱)는 빠른 주행 못 유도 → **버리고 재정의**.
4. **증거**: value 1e-7 → 25.5, 884 step 주행 (reward가 학습을 켰다).

> 안전장치 3개(§4)는 **질문 대비용**. 슬라이드엔 넣지 말 것(청중이 길을 잃음).
