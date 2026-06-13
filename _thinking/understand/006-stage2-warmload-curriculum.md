# 006 — 2-Stage 커리큘럼 + Selective Warm-Load (발표 핵심 · 모델기반 RL의 강점)

> 목적: Stage1(map_easy3)→Stage2(Oschersleben) 커리큘럼에서 **왜 world model만 warm-load하고
> actor/critic은 버리는가**를 비유로 쉽게 + 코드로 엄밀하게. Q&A("world model도 과적합 아닌가?")
> 방탄 포함.
> 코드 근거: vendor/dreamerv3-torch/stage2_utils.py, dreamer.py, dreamer_f1tenth/envs/f1tenth_env.py.
> 1차 자료: implementation/017(설계 확정), 024(구현+실행커맨드), 026(근거·zero-shot 실증).
> 관련: understand/005(reward), 012(파라미터 ~81%가 world model).

---

## 0. 한 문장 핵심

> **쉬운 트랙에서 배운 "차의 물리 감각(world model)"은 어려운 트랙으로 물려주고,
> "그 트랙 전용 주행 정책(actor)"은 버리고 새로 배운다.**
> → 환경 자체를 모델링하는 **모델기반 RL이라서 가능한 전이학습**.

---

## 1. 2-Stage 커리큘럼 개요

| | Stage 1 | Stage 2 |
|---|---|---|
| 트랙 | map_easy3 (쉬움, 100.57m) | Oschersleben (어려움, 275.18m) |
| 시작점 | 백지(scratch) | **Stage1 결과를 부분 warm-load** |
| 성격 | 빠른 트랙(풀스로틀이 최적) | 긴 트랙(코너 감속 필요) |

쉬운 트랙에서 먼저 익히고, 그 지식을 어려운 트랙으로 **이어받아** 시작하는 커리큘럼.

---

## 2. warm-load = "전부"가 아니라 "선택적으로" 가져온다 (핵심)

Stage2 시작 시 Stage1 체크포인트에서 **world model weights만** 골라 로드:

| 구성요소 | Stage2 시작 | 왜 |
|---|---|---|
| **World Model**(RSSM/encoder/decoder/reward·cont head) | ✅ **warm-load** | "차의 물리·LiDAR 동역학" → 트랙 바뀌어도 차는 같으니 재사용 |
| **Actor / Critic** | ❌ **fresh(재초기화)** | "이 트랙 전용 주행법" → 새 트랙엔 오히려 방해(negative transfer) |
| **Optimizer**(Adam momentum) | ❌ 전부 fresh | 이전 트랙의 stale momentum이 새 학습 왜곡 방지 |
| **learning rate** | ×0.5 | warm-start된 world model을 급격히 망가뜨리지 않게 |

- 구현: `extract_warm_state(agent_state_dict)` = `{k:v if k.startswith("_wm.")}` (stage2_utils.py).
  compile=False라 `_orig_mod.` prefix 없음 → `_wm.*` 직접 매칭(024 검수).
- `load_state_dict(strict=False)` → actor/critic은 생성자 초기화 유지, optimizer 미로드.

---

## 3. world model이 실제 배우는 것 — 정직한 근거 (★ Q&A 방탄)

world model = **"현재 상태(LiDAR 벽거리+속도) + 행동(조향/속도) → 다음 상태 + 보상"** 예측기.
여기엔 두 지식이 **혼합**돼 있다 (026 §1):

- **(a) 트랙-무관 물리**: 마찰·관성·차량동역학("이 속도+이 조향→이만큼 회전", "고속+급조향=언더스티어"),
  LiDAR가 벽을 비추는 방식. → **어느 트랙에서도 유효, 재사용 가치 큼.**
- **(b) 트랙-고유 기하**: map_easy3 코너 순서·벽 모양을 LiDAR 패턴 시퀀스로 외움. → **트랙 특화.**

> ★ "world model도 map_easy3에 과적합 아닌가?" → **맞다.** world model도 (b) 때문에 부분 과적합이다.
> 전역 지도(top-down)를 외우는 게 아니라 **LiDAR 기반 국소 동역학**("이런 벽 패턴에선 이렇게 움직인다")을
> map_easy3 편향으로 학습한 것.
>
> **그럼에도 warm-load하는 정당화 3가지**:
> (1) 물리(a) 재사용 가치가 큼  (2) **freeze가 아니라 fine-tune** → 기하(b)는 Oschersleben 데이터로 갱신
> (3) actor보다 재사용성이 높음.

---

## 4. 왜 actor/critic은 버리나 + 드라이버 비유

- **actor(정책)** 는 "map_easy3 = 무조건 풀스로틀"처럼 **트랙에 강하게 특화**. 가져가면 Oschersleben에서도
  풀스로틀을 고집하는 **나쁜 출발점**.

**비유 — 한 서킷만 연습한 드라이버가 새 서킷에 가면**:
| 가진 것 | 대응 | warm-load |
|---|---|---|
| 차 제어 감각·물리 직관 | world model (a) | 그대로 사용 |
| 그 서킷 코너 순서 | world model (b) | 다시 익힘(warm 후 **fine-tune**, freeze 아님) |
| "여기선 몇 km로" 주행 플랜 | actor | 서킷 전용 → 통째 새로 작성 |

---

## 5. ★ zero-shot 진단 = warm-load 필요성의 실증 (026 §3)

Stage1 정책을 그대로 Oschersleben에 투입(zero-shot) → **한 코너도 못 돌고 조기 충돌.**

- steer/speed 로그 분석 핵심: **steer는 풀조향까지 격렬한데, speed가 거의 항상 최대(20m/s)** →
  **"안 꺾는 게 아니라 감속을 안 한다."** 20m/s 풀스로틀로 코너 돌입 → 언더스티어로 벽.
- 원인: **속도 정책이 map_easy3(풀스로틀이 최적인 빠른 트랙)에 과적합** → 코너 감속이 필요한
  Oschersleben에 "무조건 밟아"를 그대로 가져옴. world model도 그 코너에서 "이 속도면 박는다"를 모름.
- → **Stage2 fine-tune이 정확히 고칠 문제**: Oschersleben 데이터로 "코너선 감속" 학습 + world model 기하 갱신.
  목표가 일반화가 아니라 **적응(fine-tune)** 인 상황과 부합.

→ "actor를 버리는 이유"가 추측이 아니라 **데이터로 입증**된 사례.

---

## 6. catastrophic forgetting 방지 — joint replay

새 트랙만 학습하면 world model이 옛 트랙을 까먹을 위험 → Stage2 학습 중 **Stage1 데이터를 0.3 비율로 섞어**
샘플링("새 트랙 70% + 옛 트랙 30%").
- `joint_episode_generator(gen_old, gen_new, ratio)`: `rng.rand()<0.3`이면 Stage1 episode (stage2_utils.py).
- joint_replay_dir = Stage1 traindir(train_eps), Stage2 traindir와 분리 = 옛/새 풀 분리.

---

## 7. 구현 확정값 (024 — 017은 설계라 값 미정이었음)

- 로드 분기: **resume 우선** — `logdir/latest.pt` 있으면 전체 resume(crash 시 watchdog 호환),
  **없을 때만**(Stage2 첫 시작) `_do_warm=True`로 `_wm.*` warm-load.
- 확정 운영 파라미터: `warm_lr_scale=0.5`, `joint_replay_ratio=0.3`, `envs=8`.
- **zero-shot 평가 게이트 폐기 → 바로 fine-tune**: 목표 = "Oschersleben 주행시간"(일반화 아님).
- **GPU 제약 순차 실행**: Stage1(~5.2GB)+Stage2(~3.3GB) 동시 = OOM(8GB) → Stage1 완료 후 Stage2.
- fixed-HP 불변: train_ratio=512/batch16/batch_length64/precision16, env 물리·reward 무변경.

---

## 8. 발표 메시지 / 슬라이드 추천 (1~2컷)

- **제목**: 2-Stage Curriculum + Selective Warm-Load
- **그림**: Stage1(easy) ──[World Model weights만]──▶ Stage2(hard);  actor/critic ✕(fresh)
- **메시지**: world model = 트랙-무관 차량 물리(파라미터의 ~81%, 012) → 재사용 / actor = 트랙-종속 → 새로 학습.
  **모델프리(DQN)는 대부분이 정책이라 재사용할 게 없지만, 모델기반(Dreamer)은 환경 모델을 물려준다.**
- **실증 1컷**(선택): Stage1 정책 zero-shot → Oschersleben에서 "감속 미흡"으로 충돌 → fine-tune 필요성 입증.
- **곁가지(질문 대비)**: world model도 부분 과적합이나 (1)물리 재사용 (2)freeze 아닌 fine-tune (3)actor보다
  재사용성으로 정당화. lr×0.5 + joint replay 0.3(망각 방지).

---

## 9. watchdog (발표 가치 낮음 — 운영 안정성)

14h 무인 학습 중 프로세스 사망 시 마지막 체크포인트에서 **자동 재개**하는 감시 스크립트(017 §1).
"장시간 학습을 무인 안정 운영" 정도로만 언급, 알고리즘 슬라이드엔 미포함 권장.
