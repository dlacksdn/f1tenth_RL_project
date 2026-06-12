# 003 — reverse_guard 메커니즘 + Oschersleben centerline 오정합 (발표 준비용 Q&A)

> 목적: 역주행 종료조건(reverse_guard)의 동작과, 그 과정에서 드러난 centerline 오정합
> 디버깅 사례를 정리. 발표에서 "환경 설계의 엄밀성 / 디버깅 능력" 근거로 활용.
> 코드 근거: `dreamer_f1tenth/envs/f1tenth_env.py` (L363-374)
> 원문: implementation/007-phase1-4-reverse-guard.md, 008(재추출). 관련: understand/001.

---

## 1. reverse_guard 메커니즘 — "역주행이 1초 지속되면 종료"

목적: 차가 트랙을 거꾸로 달리는 에피소드를 끊는 종료조건. 매 env step 후:

```
① body-frame 속도:  vel_x(종방향), vel_y(횡방향, #27 패치)
② world-frame 변환:  vel_world = R(yaw) · [vel_x, vel_y]
③ 차 위치의 가장 가까운 centerline 점:  closest_idx
④ dot = vel_world · tangent[closest_idx]      ← 속도를 "트랙 앞 방향"에 투영
⑤ if  vel_x < 0  AND  dot < 0:  reverse_counter++   else  reverse_counter = 0
⑥ reverse_counter ≥ 50 (= 50스텝 × 0.02s = 1초)  →  종료, cause='reverse', reward −10
```

- `dot > 0` 전진 / `dot < 0` 역방향. tangent는 centerline의 단위 접선(트랙 진행방향).
- 50스텝 연속이어야 종료 → 순간적 미끄러짐/후진은 무시(노이즈 강건).

---

## 2. "게이트(gate)"란 정확히 무엇인가

**게이트 = 어떤 동작이 일어나려면 반드시 통과해야 하는 추가 전제조건(AND 관문).**
여기서는 reverse 카운터 증가에 붙인 **`vel_x < 0` 조건**.

```python
# 1차 구현 (게이트 없음)
if dot < 0:
    reverse_counter += 1

# 게이트 추가 후 (현재 코드)
if vel_x < 0.0 and dot < 0.0:   # ← 'vel_x < 0' 가 게이트
    reverse_counter += 1
```

- `dot<0`인데 `vel_x≥0`(전진) → 게이트가 막음 → 카운터 안 올라감.
- `dot<0`이고 `vel_x<0`(후진) → 통과 → 카운터 올라감.

### 왜 오류 면역이 되나
- `dot`는 centerline 접선 의존 → centerline 틀리면 `dot`도 틀림(거짓 dot<0).
- `vel_x`는 **차량 상태에서 직접 나오는 값** → centerline과 무관하게 신뢰 가능.
- 전진 중 GapFollower는 항상 `vel_x>0` → 게이트에서 무조건 막힘 → **틀린 centerline 거짓 reverse 원천 차단**.

### 뉘앙스
게이트는 기존 `dot<0`(v3 option A tangent 로직)을 바꾸지 않고 **앞에 AND 전제 하나 추가**.
"tangent 조건 유지, 게이팅만 추가" → 정상 역주행은 잡고 거짓만 거름.

---

## 3. Oschersleben centerline 오정합을 어떻게 알아냈나 (탐정 과정)

### 발단: 정상 베이스라인이 거짓 reverse로 죽음
- `dot<0`만 쓰던 1차 reverse_guard 직후 A_norm 샘플 15015 → **1390 급감**(에피소드 조기종료).
- 추적: **GapFollower(정방향으로 잘 달리는 검증된 베이스라인)가 step 278에서 거짓 `cause='reverse'`**, seed 0/1/2 재현.
- 전진만 하는 차가 reverse로 죽음 = 로직이나 centerline이 틀렸다는 신호 → 기하 계측.

### 결정적 증거 (guard 끄고 GF 1바퀴 측정)
| 계측 | 값 | 의미 |
|---|---|---|
| 차–centerline 거리 `d_cl` | 평균 7.8m, **최대 19m** | half-width ~6m인데 19m 떨어짐 → centerline이 실제 주행로 아님 |
| `closest_idx` 점프 | t=215에 288 → **807** | 반대방향(anti-parallel) 구간으로 snap → tangent 반대 → dot<0 누적 → step278 거짓 reverse |
| centerline y범위 | −20.6~39.9 (**60.5m**) | |
| 실제 주행 y범위 | −6.3~26.2 (**32.5m**) | centerline이 y<−6.3까지 ~14m 더 뻗음 (≈2배) |

### 결론
- offset/scale 어긋남이 아니라 **형상 자체가 다른 더 큰 폐곡선**.
- windowed projection으로도 해소 안 됨(d_cl 15m로 악화) → **centerline 데이터 자체 결함 확정**.
- 근본원인(008): `extract_centerline.py`의 `keep_largest_cc`가 트랙 리본이 아니라 **맵 바깥 경계 루프** 선택(자유공간 97.6% 구조).

> 탐지 논리: **정상 베이스라인의 false positive → 기하 계측 → "너무 멀고(19m>6m), 너무 크고(2배), 반대방향 snap" 정량 증거.**

---

## 4. 오정합이면 어떤 문제가 생기나 (3가지 파급)

| # | 문제 | 심각도 | 이유 |
|---|---|---|---|
| 1 | **progress reward 붕괴** | **치명적** | reward도 같은 투영으로 `s` 증분 계산. idx 점프 시 Δs 폭발 → reward 폭주 → 학습 불가 |
| 2 | **L_track=312.61 부정확** | 높음 | 틀린(부풀려진) centerline arclength → reward scale 연쇄 오류 |
| 3 | **reverse dot 신호 신뢰불가** | 중간 | vel_x 게이트로 증상만 우회, tangent 자체는 틀림 |

### 게이트는 임시 우회였지 진짜 수정이 아니다
- `vel_x<0` 게이트는 **증상 3만 가림. 1·2(progress reward, L_track)는 방치**.
- 그래서 007은 **BLOCKER**로 표기, "Phase 4(reward) 진입 전 centerline 재추출 필수".
- 게이트 덕에 Phase 1-4는 map_easy3(정합됨)에서 검증하고 닫음. 실제 해결은 **008**: START_POSES로 올바른 리본 CC 재추출 → L_track 275.18m, d_cl half-width 내, 투영 신뢰 회복.

---

## 5. 발표 활용 포인트

centerline 하나가 **progress·lap·reverse 세 군데에 물려** 있으니 그게 틀리면 셋 다 무너진다 —
"왜 centerline 품질이 reward의 전제조건인가 / 어떻게 디버깅했나"를 보여주는 구체적 사례.
(정상 베이스라인 false positive를 신호로 삼아 기하 계측으로 근본원인까지 추적.)
