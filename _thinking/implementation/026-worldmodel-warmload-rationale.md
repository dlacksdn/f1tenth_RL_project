# 026 — world model warm-load 설계 근거 + Oschersleben zero-shot 진단

> 2026-05-23 대화 정리. "왜 world model만 warm-load하고 actor는 버리나? world model도
> map_easy3에 과적합 아닌가?"라는 질문에 대한 답 + Oschersleben zero-shot 주행 진단.

## 0. env step ↔ 에피소드 관계
- 가로축 env step = **모든 에피소드 길이의 누적합**. 한 에피소드 = reset~done(충돌/완주/timeout).
- Stage1 최종: env-step 516,292 / **536 에피소드** (train_return 기록 = npz 536개), 평균 ~963 step/ep.
- 에피소드 길이 가변: 즉시 박으면 train_return≈-7.6(짧음), 완주면 ~246(긺).

## 1. world model이 실제로 배우는 것 (RSSM dynamics)
"현재 상태(lidar 벽거리+속도) + 행동(조향/속도) → 다음 상태 + 보상" 예측. 두 지식이 **혼합**된다:
- **(a) 트랙-무관 물리**: 마찰·관성·차량동역학("이 속도+이 조향→이만큼 회전", "고속+급조향=언더스티어"),
  lidar가 벽을 비추는 방식. → 어느 트랙에서도 유효, 재사용 가치 큼.
- **(b) 트랙-고유 기하**: map_easy3 코너 순서·벽 모양. lidar 패턴 시퀀스로 국소 기하를 외움. → 트랙 특화.

→ **질문자 지적 맞다: world model도 map_easy3에 부분 과적합.** Oschersleben 코너 미학습이 그 증거.
- "map을 통째로 외우나?" = 전역 지도(top-down)가 아니라 **lidar 기반 국소 동역학**("이런 벽 패턴에선
  이렇게 움직인다")을 학습. map_easy3 패턴에 편향된 것.

## 2. 왜 world model만 warm-load, actor/critic은 버리나 (#21)
재사용성 차이:

| | 트랙 특화 정도 | warm-load |
|---|---|---|
| world model | 물리(a, 재사용 큼) + 기하(b, 특화) 혼합 | ✅ 가져감 — 물리 재사용 + 기하는 fine-tune 갱신(freeze 아님) |
| actor(정책) | "map_easy3=무조건 풀스로틀" 강하게 특화 | ❌ 버림 — 가져가면 Oschersleben서도 풀스로틀 고집(나쁜 출발점) |

비유(한 서킷만 연습한 드라이버 → 새 서킷):
- 차 제어 감각·물리 직관(world model a) = 그대로 사용
- 그 서킷 코너 순서(world model b) = 다시 익힘 → warm 후 **freeze 아니라 계속 fine-tune**
- "여기선 몇 km로" 주행 플랜(actor) = 서킷 전용이라 통째 새로 작성

★ 정직: warm-load가 "트랙무관 지식만 깨끗이"는 아니다. map_easy3 기하도 묻어온다. 정당화는
(1) 물리 재사용 가치 큼 (2) freeze 아니라 fine-tune으로 기하 갱신 (3) actor보다 재사용성 높음.

## 3. Oschersleben zero-shot 진단 (latest.pt 516k, watch_drive 1ep)
- 결과: 131 step만에 collision(progress 28.5, return 18.5). 한 코너도 못 돎.
- ★ steer/speed 로그 분석(핵심): **speed 평균 19.8 / 거의 항상 20.0(최대)**, steer는 풀조향(±1.0)까지 격렬.
  → **"안 꺾는 게 아니라 감속을 안 한다."** 조향은 최대로 하나 20m/s 풀스로틀로 코너 돌입 →
  언더스티어로 벽. 사용자가 본 "안 꺾음"은 고속이라 못 도는 시각적 인상.
- 원인: **속도 정책이 map_easy3(짧고 빠른 트랙, 항상 최대속도가 최적=13.16초 우수 기록의 비결)에 과적합.**
  그 "무조건 밟아"를 Oschersleben(긴 트랙, 코너 감속 필요)에 그대로 가져옴. world model도 Oschersleben
  코너에서 "이 속도면 박는다"를 모름(reward head 예측 부정확).
- → **Stage2 fine-tune이 정확히 고칠 문제**: Oschersleben 데이터로 "코너선 감속" 학습. 일반화가 아니라
  적응이 목표인 상황과 부합. fine-tune 효과 예측 명확(속도 정책 적응 + world model 기하 갱신).

## 4. 결론
- world model warm-load = 물리(재사용) + 기하(fine-tune 갱신)의 혼합 출발점. actor보다 재사용성 높아 warm-start.
- Oschersleben 충돌이 "world model의 기하 부분은 fine-tune으로 갱신해야 함"을 실증.
- map_easy3 13.16초 우수 = 풀스로틀이 최적인 빠른 트랙에서의 강점(Stage1 목표 달성).
- 관련: [[lewm-data-contract]], 019 §3(A-2 사양), 024(실행커맨드), 025(자동트리거).
