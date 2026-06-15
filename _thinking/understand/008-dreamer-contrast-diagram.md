# 008 — 순정 DreamerV3 → f1tenth 대조 다이어그램: 설계·색규약·코드근거·검수 전말

> 목적: 발표(영어 10분, 모델기반 RL로 F1TENTH 자율주행)의 핵심 섹션인 **"순정 DreamerV3 ↔ f1tenth
> 맞춤형(vendor-in)이 어떻게 다른가"** 대조 다이어그램의 설계·색규약·코드근거·검수 과정을
> 빠짐없이 정리. 다른 세션이 이어받는 인수인계 + 발표 준비 레퍼런스로 자족적으로 작성.
> 코드 근거: vendor/dreamerv3-torch/{networks,models,configs.yaml,stage2_utils,dreamer},
>   dreamer_f1tenth/{networks_1d.py, envs/f1tenth_env.py}, 순정 비교본 /home/dlacksdn/dreamerv3-torch.
> 논문: _thinking/raws/Dreamer-V3.pdf (Hafner et al. 2023).
> 관련: understand/001(reward)·002(wrapper)·003(reverse)·004(gym API)·005(reward 쉬움)·006(2-stage)·007(알고리즘/그림).
> 작성일: 2026-06-15.

---

## §0. 한 줄 요약 + 이 문서의 위치

> **"우리는 DreamerV3 알고리즘을 바꾼 게 아니라, 입력(라이다 관측)·출력(action·progress reward)·환경
> 어댑터·2-stage 운영만 f1tenth에 맞게 끼웠다."** — 알고리즘 코어(`models.py`의 RSSM·actor-critic 학습)는
> 사실상 무변경이고, 바뀐 건 관측을 처리하는 encoder/decoder(1D-CNN)와 환경 연결뿐이다.

대조 다이어그램은 이 메시지를 **색**으로 증명한다: 변경(빨강·주황)은 소수 지점에 몰려 있고, 무변경(회색)이
압도적(특히 Behavior 패널 전체)이다.

문서 범위: ① 대조 그림 형식 결정 → ② 색 라벨링 체계(그림 독해의 열쇠) → ③ 패널별 변경점 →
④ 코드로 확정한 사실 → ⑤ 발표용 수식 → ⑥ 자주 헷갈리는 개념 → ⑦ 작도 실무 교훈 → ⑧ 검수 이력.

---

## §1. 대조 그림 형식 결정

### 1-1. 후보 3안과 채택

| 안 | 형식 | 장점 | 단점 | 채택 |
|---|---|---|---|---|
| **A** | 순정 골격 위에 **바뀐 박스만 색칠** | "어디가 바뀌었나"가 한눈에. 무변경(회색)이 압도적으로 남아 "코어 무변경" 메시지 자동 성립 | — | **✅ 채택** |
| B | 좌=순정 / 우=f1tenth 나란히 대조 | 1:1 비교 명확 | 그림 2개라 작업량 2배, 슬라이드에서 글씨 작아짐 | |
| C | 변경점 6개를 카드식 | 깔끔, 텍스트 발표 적합 | "전체 파이프라인 속 어디"라는 공간감 약함 | |

→ A안 채택 이유: **순정 DreamerV3의 골격(3패널)을 그대로 유지**하면서 변경 지점만 색으로 덮으면,
"우리가 건드린 곳"과 "그대로 둔 곳"이 같은 그림 안에서 대비된다. 회색(무변경)이 그림의 대부분을
차지하는 것 자체가 발표 핵심 주장("알고리즘은 검증된 DreamerV3 그대로")의 시각적 증거다.

### 1-2. 최종 다이어그램 구성

순정 DreamerV3의 3패널 구조를 따른다(논문 Figure 1과 동형):

- **패널 A — World Model (φ)**: world model(내부 시뮬레이터) 학습. RSSM 6컴포넌트.
- **패널 B — Actor-Critic Learning in Imagination**: 학습된 world model을 고정하고 상상(latent)에서 정책 학습.
- **패널 C — Environment Interaction**: 실제 환경과 굴러 데이터 수집 → replay buffer.
- **녹색 배지**: 2-Stage Curriculum + Selective Warm-Load (순정에 없는 신규).

---

## §2. 색 라벨링 4색 체계 ★(대조 그림 독해의 열쇠)

이 그림을 읽는 모든 것은 **네 색의 의미를 아는 데서 시작**한다. 색은 단순 강조가 아니라
**"순정 대비 무엇이/얼마나 바뀌었나"를 분류하는 코드**다.

### 2-1. 네 색은 두 개의 축을 표현한다

- **빨강·주황·회색**은 한 축이다 — "순정 DreamerV3 대비 **변경 정도**" (구조교체 / 의미교체 / 무변경).
- **녹색**은 다른 축이다 — "순정엔 **아예 없던 신규** 블록".

| 색 | 범례 정의 | 진짜 의미 | 비유 |
|---|---|---|---|
| 🔴 **빨강** | architecture-level 교체 | **신경망 레이어 구조 자체**(또는 환경 구조)를 통째로 갈아끼움 | "**눈을 바꿨다**"(카메라→라이다) |
| 🟠 **주황** | I/O 의미 교체 (NN 무변경) | 신경망은 **그대로**, 그게 먹고 뱉는 **데이터의 의미·범위만** 재정의 | "**핸들 범위·채점 기준**을 바꿨다" |
| ⚫ **회색** | 순정 무변경 | DreamerV3 코드 그대로 | "**뇌(학습 알고리즘)는 안 건드림**" |
| 🟢 **녹색** | 신규 f1tenth 전용 | 순정 학습 루프에 없던 개념 추가 | "**전학 커리큘럼**을 새로 끼움" |

### 2-2. 빨강 ↔ 주황의 구분 (가장 중요)

발표 정밀성의 핵심은 이 둘을 구분하는 것이다.

- 🔴 **빨강 = 신경망을 진짜 교체**한 곳. 코드로 `networks.py`/`networks_1d.py`가 바뀐 지점.
  딱 **encoder, decoder, environment** 3군데뿐이고, 전부 "관측이 이미지→라이다로 바뀐" 데서 파생.
- 🟠 **주황 = 신경망은 손도 안 댔고**, 그 박스가 다루는 **action·reward의 의미/범위만** f1tenth 차량용으로 바꾼 것.
  코드로는 env/wrapper 레벨 수정이며 NN 구조는 무변경.

> 한 문장: *"빨강은 신경망을 갈았고, 주황은 신경망이 다루는 데이터의 뜻을 갈았다."*

### 2-3. 박스별 색 매핑표 (전체)

| 패널 | 박스/요소 | 색 | 근거 |
|---|---|---|---|
| A | Sequence model (GRU) | ⚫ | RSSM 코어 무변경 |
| A | **Encoder** | 🔴 | image 2D-CNN → lidar 1D-CNN + state MLP |
| A | Dynamics predictor | ⚫ | 무변경 |
| A | model state s_t={h_t,z_t} | ⚫ | 무변경 |
| A | Reward predictor | 🟠 | NN 구조 무변경, 회귀 타깃이 progress reward로 바뀜 |
| A | Continue predictor | ⚫ | 무변경 |
| A | **Decoder** | 🔴 | image decoder 제거 → lidar 1D-deconv + state |
| A | world model loss(L_pred/L_dyn/L_rep) | ⚫ | 무변경 |
| B | **패널 전체**(actor·critic·predictors·losses) | ⚫ | 상상 actor-critic 학습 전부 무변경 |
| C | **Environment** | 🔴* | F110Env + 3겹 wrapper (※2-4 색 논쟁) |
| C | Encoder/RSSM | 🔴 | A의 encoder와 동일 |
| C | action a_t | 🟠 | 2D 연속 [steer,speed], 정규화·repeat |
| C | reward(progress) | 🟠 | centerline Δs로 재정의, stock reward 폐기 |
| 배지 | 2-Stage Warm-Load | 🟢 | 순정에 없는 신규 운영 |

### 2-4. 색 정합성 이슈 2건 (발표 전 정리 필요)

**(이슈 1) reward predictor: 회색 → 🟠주황으로 바꾸는 게 더 정확.**
- reward predictor의 **NN 구조는 무변경**(회색의 근거)이지만, 회귀 **타깃**(학습하는 보상)이
  stock f110 reward → progress reward로 바뀐다 = **데이터 의미 교체 = 주황의 정의에 정확히 부합**.
- 게다가 C패널의 progress reward(주황)와 같은 색이 되어 "보상 관련은 다 주황"으로 시각적으로 연결된다.
- 결론: **reward predictor 박스를 주황으로 칠하고**, "now regresses progress reward" 주석은 짧게 유지.
  (회색+빨간주석 혼용은 애매했음. critic 검수도 주황 권고.)

**(이슈 2) Environment: 빨강의 정의 모순 → 범례를 넓히거나 색을 바꿔라.**
- 문제: 빨강을 "**신경망 아키텍처 교체**"로 좁게 정의하면, **Environment는 신경망이 아니다**
  (시뮬레이터 + 파이썬 wrapper). 범례와 자기모순.
- 해결 택1:
  - (a) **범례를 'architecture-level 교체'로 확장** → encoder/decoder(NN 교체)와 environment(환경 교체)가
    "통째로 새로 만든 곳"으로 자연스럽게 묶임. (현재 그림 유지, 범례 한 줄만 수정) ← 권장
  - (b) **Environment를 🟢초록(신규)으로** → F110GymnasiumWrapper/F1Tenth adapter는 신규 작성이므로
    "신규" 의미엔 맞음. 단 NormalizeActions는 stock 재사용이라 일부 애매.
- 어느 쪽이든 **"빨강=NN만"이라는 좁은 정의와 Environment 빨강의 충돌**을 없애야 색 질문에 방어된다.

---

## §3. f1tenth 변경점 — 패널별 상세 (대조 그림 본문)

각 패널이 **무엇을 하는 곳인지** + **무엇이 왜 그 색인지**를 코드 근거와 함께.

### 3-1. 패널 A — World Model (φ): "차의 세계를 머릿속 시뮬레이터로 학습"

**역할:** world model은 *현재 상태 + 행동 → 다음 상태 + 보상*을 예측하는 **내부 시뮬레이터**다.
6개 컴포넌트(전부 파라미터 φ). 점선 **RSSM = sequence model + encoder + dynamics predictor**(Dreamer 심장).

- **① Sequence model (GRU)** ⚫: `h_t = f_φ(h_{t-1}, z_{t-1}, a_{t-1})`. 과거를 요약한 **결정적** 잠재 h_t.
  h_t가 유일한 deterministic state. 무변경.
- **② Encoder** 🔴: `z_t ~ q_φ(z_t | h_t, x_t)`, 관측 x_t를 보고 **posterior z** 추출.
  순정은 카메라 이미지를 2D-CNN으로, **우리는 라이다 1080빔을 1D-CNN(6-stage, ch128, flatten 2176) +
  차량 state 5D를 MLP**로 갈아끼움. → *관측 형식이 이미지→1D 레이저라 인코더 회로 통째 교체.*
- **③ Dynamics predictor** ⚫: `ẑ_t ~ p_φ(ẑ_t | h_t)`, 관측 **없이** h_t만으로 prior ẑ 예측. 무변경.
  encoder의 "관측 본 z"와 dynamics의 "관측 없이 예측한 ẑ"를 **KL(representation↔dynamics, free bits)**로
  맞추는 게 RSSM 학습의 본질.
- **model state `s_t = {h_t, z_t}`** ⚫: 이후 모든 head의 입력. 무변경.
- **④ Reward predictor** 🟠(§2-4): `r̂_t ~ p_φ(r̂_t | h_t, z_t)`. NN 구조 무변경, 타깃만 progress reward.
- **⑤ Continue predictor** ⚫: `ĉ_t ~ p_φ(ĉ_t | h_t, z_t)`, 에피소드 지속(1−종료) 예측. 무변경.
- **⑥ Decoder** 🔴: `x̂_t ~ p_φ(x̂_t | h_t, z_t)`, model state로 관측 재구성. 순정 image decoder **제거**,
  lidar+state 복원(1D-deconv)으로 교체.

**A패널 요점:** 신경망을 진짜 바꾼 건 **②encoder · ⑥decoder뿐**(센서/관측 직결). RSSM 학습 메커니즘은 순정 그대로.

### 3-2. 패널 B — Actor-Critic in Imagination: "머릿속 시뮬레이터 안에서만 주행 연습"

**역할:** A에서 배운 world model을 **고정(fixed)**하고, **실제 환경 없이** 잠재공간에서만
**H=15스텝** 상상 주행을 굴려 actor(정책)와 critic(가치)을 학습.

흐름: `Actor π_θ`가 a_τ 제안 → `Sequence+Dynamics`가 다음 상태 상상 → `Reward/Continue predictor`가
보상·지속 예측 → `Critic v_ψ`가 가치 추정 → `λ-return target`으로 actor·critic 업데이트.
(우상단 `analytic gradient`, 우하단 `Bellman/λ-return`.)

> 📌 **이 패널은 전부 ⚫회색 = 완전 무변경.** 발표의 가장 강한 한 방: *"강화학습의 심장인 상상 기반
> actor-critic 학습을 한 줄도 안 바꿨다."*
> **왜 무변경이 가능한가:** world model이 관측을 일단 잠재 z로 압축한 뒤엔, 그 z가 라이다든 이미지든
> 학습 로직은 동일하다. 그래서 A의 encoder만 라이다용으로 바꾸면(빨강) B는 그대로 작동한다.

### 3-3. 패널 C — Environment Interaction: "실제 환경과 굴러 데이터 수집"

**역할:** 학습된 정책으로 진짜 시뮬레이터(F110)를 한 스텝씩 굴려 `(x_t, a_t, r_t)`를 모아
**replay buffer**에 저장 → A패널 학습 재료로 순환.

흐름(엄밀): `Environment → 관측 x_t → Encoder/RSSM → s_t → [Actor] → a_t → Environment`.
※ 그림에서 Actor 노드가 생략돼 a_t가 Encoder/RSSM에서 직접 나오는 것처럼 보이지만, **action을 만드는 건
언제나 Actor**다. RSSM은 관측을 잠재로 인식할 뿐 행동을 만들지 않는다. (발표 시 보강 권장.)

- 🔴 **Environment**: 순정 단일 환경 → **F110Env + 3겹 wrapper**(NormalizeActions / F1Tenth adapter /
  F110GymnasiumWrapper). 세 겹이 단위 통역(정규화↔물리), API 통역(5-tuple↔4-tuple), MDP 정의(reward·종료·정규화) 담당.
- 🔴 **Encoder/RSSM**: A의 ②encoder와 동일(라이다용).
- 🟠 **a_t**: 연속 2D `[steer ±0.4189 rad, speed −5~20 m/s]`, `[-1,1]↔물리`, `action_repeat×2`.
- 🟠 **progress reward**: `centerline Δs(clip 0~0.5)`, stock f110 reward 폐기.
- `(x_t, a_t, r_t)` → replay buffer.

**중요 설계:** Dreamer가 보는 건 **라이다 + 자기 속도뿐**. **차의 절대 위치(pose)는 Dreamer에 안 주고
reward(centerline 진행)로만 흡수** → 위치 비의존 정책.

### 3-4. 녹색 배지 — 2-Stage Curriculum + Selective Warm-Load

순정 단일 학습 루프엔 없던 **신규 운영 전략**.
- **Stage1(map_easy3, 쉬움)** 학습 → **Stage2(Oschersleben, 어려움)**로 전이.
- **world model weights만 warm-load** — 차의 물리·라이다 동역학은 트랙 바뀌어도 유효하니 재사용.
- **actor/critic = fresh** — "이 트랙 전용 주행법"은 새 트랙에 방해(negative transfer)라 새로 학습.
- **optimizer = fresh**(stale momentum 차단), **lr×0.5**(warm WM 급변 방지), **joint replay 0.3**(옛 트랙 망각 방지).

> **모델기반 RL이라서 가능한 전이학습.** 모델프리(DQN)는 대부분이 정책이라 물려줄 게 없지만,
> Dreamer는 환경 모델(world model)을 통째로 물려준다. ← 발표 차별점.

### 3-5. 변경점 6개 요약 (대조 그림이 말하는 것)

| # | 변경점 | 색 | 근거 |
|---|---|---|---|
| 1 | 관측/인코더: image(2D-CNN) → lidar 1080빔(1D-CNN)+state(MLP), image head 제거 | 🔴 | understand 002·004 |
| 2 | reward: stock f110 reward → progress reward(centerline Δs) 자체 계산 | 🟠 | understand 001·005 |
| 3 | 환경 연결: NormalizeActions + F1Tenth adapter + F110GymnasiumWrapper 3겹 | 🔴/🟠 | understand 002·004 |
| 4 | action: 연속 2D [steer ±0.4189, speed −5~20], [-1,1]↔물리, repeat×2 | 🟠 | understand 004 |
| 5 | 2-stage 커리큘럼 + selective warm-load(WM만) | 🟢 | understand 006 |
| 6 | **무변경 강조**: 알고리즘 코어(models.py)·RSSM·actor-critic·hyperparameter 고정 | ⚫ | §4 |

---

## §4. 코드로 확정한 사실 (file:line 근거)

발표/Q&A에서 추측 없이 답하기 위한 코드 사실들. 전부 직접 읽어 확인.

### 4-1. actor = analytic gradient, critic = λ-return regression
- f1tenth가 쓰는 config: `imag_gradient: 'dynamics'` (`vendor/dreamerv3-torch/configs.yaml:86`, defaults).
  f1tenth 섹션(`:193~`)이 **override하지 않으므로** defaults 상속.
- **Actor** (`models.py:418`): `if imag_gradient=='dynamics': actor_target = adv` → advantage를 잠재 통해
  직접 backprop = **analytic(dynamics) gradient**. (REINFORCE 아님. 연속 행동이라 torch 구현 default가 이것.)
- **Critic** (`models.py:330`): `value_loss = -value.log_prob(target.detach())` → λ-return target이
  `.detach()`로 고정된 **회귀(regression)**. analytic 아님.
- ⇒ 그림의 "analytic gradient" 라벨은 **actor에만** 해당. 논문이 일반론으로 말하는 REINFORCE와 다름(우리 코드 기준 analytic이 정답).

### 4-2. λ-return = 논문 식 (5)
논문 page 5에서 critic loss와 λ-return이 **한 번호 (5)**에 묶임:
```
L(ψ) ≐ −Σ_{t=1}^T ln p_ψ(R^λ_t | s_t)
R^λ_t ≐ r_t + γ c_t ( (1−λ) v_t + λ R^λ_{t+1} ) ,   R^λ_T ≐ v_T          ... (5)
```
- 왼쪽 `R^λ_t` = **λ-return 타깃**, 오른쪽 `L(ψ)` = 그 타깃에 대한 critic 회귀.
- 직관: "한 스텝 실제 보상 + 그 다음은 critic 추정치 v로 부트스트랩"을 λ로 보간. λ=0이면 1-step, λ=1이면 MC.
- 파라미터: λ=0.95, T(horizon)=16(논문)/H=15(우리 config), γ=0.997.

### 4-3. replay buffer 저장 = (x, a, r) + episode flag
`tools.py`: `transition["action"]=a`(:196), `["reward"]=r`(:197), `["discount"]=1−done`(:198, =continue flag),
obs(x: lidar/state + is_first/terminal/last)는 add_to_cache(:256)로 dict 저장. → 공식 구현체도 **(x,a,r)+continue flag** 저장.

### 4-4. ★reward predictor는 환경 상호작용에 쓰이지 않는다 (개념 정정)
- 실제 환경 상호작용에서 r_t는 **환경(F110 wrapper)이 progress reward로 직접 계산**(`f1tenth_env.py:430`).
- reward predictor는 world model의 head로, **두 곳에서만** 동작:
  ① **Dynamics learning**: 환경이 준 실제 r_t를 **타깃으로 회귀 학습**(`models.py:142` `-pred.log_prob(data["reward"])`).
  ② **Imagination**: 실제 환경이 없으니 상상 보상 r̂_τ를 **생성**.
- ⇒ "reward predictor가 환경에서 r_t를 추출"은 틀림. C패널엔 reward predictor가 없다(환경이 스스로 계산).

### 4-5. encoder 구조 = lidar(1D-CNN) + state(MLP) → concat
`networks.py` MultiEncoder.forward: `outputs.append(self._lidar(obs["lidar"]))` (1D-CNN),
`outputs.append(self._mlp(state))`, `outputs = torch.cat(outputs, -1)`. → encoder는 **하나의 모듈, 두 갈래**.
state는 별개 외부 모듈이 아니라 **encoder의 두 번째 갈래**. z_t엔 lidar+state 정보가 **둘 다** 녹아듦.
- lidar 스펙: `networks_1d.py:106` `depths=(16,32,64,128,128,128)` → 6-stage, 최종 ch128,
  길이 1080→...→17, flatten = 128×17 = **2176**. (그림 숫자와 일치. §8 stale 주석 경고 참고.)

### 4-6. 기타 확정값
- **H=15**: `configs.yaml:85` `imag_horizon: 15`(f1tenth override 없음). `models.py:303~`에서 사용.
- **action**: `f1tenth_env.py:35-36` `S_MIN/MAX=∓0.4189`, `V_MIN/MAX=−5/20`. NormalizeActions [-1,1]↔물리(`dreamer.py`). action_repeat=2 sub-step(`:313~`, `configs.yaml:201`).
- **progress reward**: `f1tenth_env.py:390` `clip(raw_delta, 0, PROGRESS_CAP)`, `PROGRESS_CAP=0.5`(:87).
  raw_delta=centerline arclength Δs(:344). stock f110 reward는 `_r`로 버려짐(:314). 전체 reward(:430)=progress+R_lap+페널티.
- **2-stage**: WM-only warm `stage2_utils.py:21-27`(`_wm.*` 추출) + `dreamer.py` strict=False.
  실제 운영값(understand 006 §7): `warm_lr_scale=0.5`, `joint_replay_ratio=0.3`, `envs=8`.
  (※ configs.yaml 기본값은 1.0/0.0 — 0.5/0.3은 Stage2 실행 override. 발표 전 logdir config dump로 0.3 재확인 권장.)
- **모델 축소**: f1tenth `dyn_discrete=16`, `dyn_stoch=32`(`configs.yaml:230`) vs stock 32×32. 8GB GPU 제약. (발표 가치.)

### 4-7. 무변경의 코드 증거 (발표 핵심)
- `models.py`(알고리즘 코어)는 stock 대비 **사실상 무변경**(device 인자·image 가드 정도). RSSM body·GRUCell·
  dynamics·continue·ImagBehavior(actor/critic/imagine/lambda_return/actor_loss) **변경 0**.
- world model loss: `models.py:149` `model_loss = sum(scaled) + kl_loss`, free bits `configs.yaml:59` `kl_free:1.0`,
  dyn_scale=0.5/rep_scale=0.1(`:57-58`). → L_pred+L_dyn+L_rep 구조, free bits=max(free,KL) 그대로.

---

## §5. 발표용 수식 모음 (평문 + 복붙 LaTeX)

### 5-1. progress reward (전체)
평문:
```
r_t = clip(Δs_t, 0, 0.5) + R_lap·𝟙_lap − 10·𝟙_fail ,   Δs_t = s_t − s_{t-1}
```
- `Δs_t = s_t − s_{t-1}`: centerline 호길이 진행량
- `clip(x,0,0.5) = min(max(x,0), 0.5)`: 후진은 0, 상한 0.5(=v_max·dt=20×0.02=0.4+여유, 측정 이상치 차단)
- `𝟙_lap`: 새 랩(high-water-mark) 1/0, `𝟙_fail`: 충돌·역주행·이탈 1/0
- `R_lap = 25`(map_easy3) / `100`(Oschersleben)

LaTeX:
```latex
r_t = \underbrace{\mathrm{clip}(\Delta s_t, 0, 0.5)}_{\text{progress}}
    + \underbrace{R_{\text{lap}}\,\mathbb{1}_{\text{lap}}}_{\text{lap bonus}}
    - \underbrace{10\,\mathbb{1}_{\text{fail}}}_{\text{penalty}},\qquad \Delta s_t = s_t - s_{t-1}
```

### 5-2. 관측 x_t (lidar + state)
평문:
```
x_t = ( o^lidar_t , o^state_t )
o^lidar_t = clip(scan, 0, 30m) / 30                       ∈ [0,1]^1080
o^state_t = [ v_x/20, v_y/5, ω_z/2π, prev_steer/0.4189, prev_speed/20 ]  ∈ ℝ^5
            (각 성분 [-10,10] clamp)
```
- lidar: 1080빔 거리(0~30m) → ÷30 정규화. 1D 시퀀스 → 1D-CNN.
- state: 속도 3개(종/횡/yaw rate) + 직전 행동 2개. 격자 없음 → MLP. **위치(pose)는 미포함**.
- 근거: `f1tenth_env.py:50`(_STATE_SCALE), `:222-227`(lidar), `:237-242`(state).

LaTeX:
```latex
x_t = (o^{\text{lidar}}_t,\, o^{\text{state}}_t),\quad
o^{\text{lidar}}_t = \tfrac{\mathrm{clip}(\text{scan},0,30)}{30}\in[0,1]^{1080},\;
o^{\text{state}}_t = \big[\tfrac{v_x}{20},\tfrac{v_y}{5},\tfrac{\omega_z}{2\pi},
\tfrac{a^{\text{steer}}_{t-1}}{0.4189},\tfrac{a^{\text{speed}}_{t-1}}{20}\big]\in\mathbb{R}^5
```

### 5-3. λ-return / critic loss (논문 식 5)
```latex
\mathcal{L}(\psi)\doteq-\sum_{t=1}^{T}\ln p_\psi(R^\lambda_t\mid s_t),\qquad
R^\lambda_t\doteq r_t+\gamma c_t\big((1-\lambda)v_t+\lambda R^\lambda_{t+1}\big),\quad R^\lambda_T\doteq v_T
```

### 5-4. world model loss (참고)
```
L(φ) = E_q[ Σ_t ( β_pred·L_pred + β_dyn·L_dyn + β_rep·L_rep ) ]
L_pred = −ln p_φ(x_t|h_t,z_t) − ln p_φ(r_t|h_t,z_t) − ln p_φ(c_t|h_t,z_t)
L_dyn  = max(1, KL[ sg(q_φ(z_t|h_t,x_t)) ‖ p_φ(z_t|h_t) ])
L_rep  = max(1, KL[ q_φ(z_t|h_t,x_t) ‖ sg(p_φ(z_t|h_t)) ])
```
(β_dyn=0.5, β_rep=0.1, free bits=1.0; sg=stop-gradient. 무변경.)

---

## §6. 자주 헷갈리는 개념 (Q&A 방탄)

- **x_t(관측) vs z_t(잠재):** x_t는 encoder **입력**(라이다+상태, 날것), z_t는 encoder **출력**(압축된 잠재).
  정반대다. encoder 입력 화살표는 항상 x_t. (손그림에서 x↔z 뒤바뀜 주의.)
- **r̂_t(보상) vs v̂_t(가치):** r̂는 reward predictor 출력(이 스텝 보상), v̂(또는 v_ψ)는 critic 출력(미래 가치 합).
  reward predictor 출력은 r̂. replay buffer에 저장되는 건 (x,a,**r**)이지 v가 아님.
- **reward predictor의 진짜 역할:** "보상을 만드는 모듈"이 아니라 **"상상 속에서 보상을 예측"**하는 모듈.
  실제 보상은 환경(progress)이 만든다. 환경 상호작용 패널엔 reward predictor 없음(§4-4).
- **clip 안 하면?** 신경망이 즉사(NaN)하진 않는다(symlog/twohot/return-norm이 완충). 진짜 위험은
  **측정 버그(8자 교차·결승선)로 생긴 가짜 거대 보상을 정책이 믿고 학습해 망가지는 것**. cap이 그 오염을 원천 차단.
  정상 Δs(≤0.4)는 0.5에 안 걸리므로 정상 학습 영향 0.
- **s_t = {h_t, z_t} 무변경:** 구조(그릇)는 순정 그대로. 바뀐 건 z를 만드는 encoder(빨강)뿐 →
  z의 그릇은 같고 담기는 내용만 라이다 기반. 그래서 RSSM 이후 전부 무변경으로 작동.
- **encoder는 하나, 두 갈래:** lidar→1D-CNN, state→MLP로 따로 처리 후 **concat**. state는 별개 모듈이 아니라 encoder의 두 번째 갈래.
- **analytic gradient가 우리 기준 정답:** 논문 일반론은 REINFORCE를 말하나, 우리 config는 `imag_gradient='dynamics'`라
  actor가 analytic(dynamics backprop). 발표/Q&A는 analytic으로 답.

---

## §7. 다이어그램 작도 실무 교훈 (Gemini / 나노바나나)

발표 그림을 Gemini 이미지 모델로 만들며 얻은 교훈. 재현/재작업 시 시간 절약용.

- **image-edit은 실패한다:** 순정 그림을 첨부하고 "이 박스만 색칠/편집"(byte-for-byte 보존)을 시키면,
  모델이 전체를 **재생성**하면서 수식·한글 텍스트를 환각으로 깨뜨린다(예: `analytic gradient`→`최타함 마`,
  `Actor`→`영현식정 코도`). 텍스트 빽빽한 손그림 다이어그램의 구조적 약점.
- **full-generation이 안정적:** 순정 그림을 **"참고용"으로만** 첨부하고, 영어 라벨로 **처음부터 새로** 그리게
  하면(from scratch) 모델이 자기 텍스트를 통제해 깔끔하다. 라벨은 영어+수식으로 통일(한글 깨짐 방지).
- **2-pass 분할:** 색칠 한 번, 주석/배지 추가 한 번으로 나누면 각 단계가 단순해져 안정적.
- **텍스트 렌더 한계:** `Gymnasium` 같은 긴 고유명사 철자는 반복적으로 흘린다(`Gymnasin`). 프롬프트로 못 고침
  → 최종 글자 보정은 PPT/그림툴에서 수동(흰 박스+텍스트박스, 1분)이 가장 확실.
- **참고 이미지 계승 주의:** 직전(결함 있는) 결과를 참고로 다시 첨부하면 그 결함(`Verify:` 환각)을 베껴온다.
  → 참고를 순정 원본으로 되돌리거나 텍스트만으로 생성.
- **최종 생성 프롬프트 원문**(full-gen, 영어): 패널 A/B/C + 색범례 + 배지 스펙을 한 번에 기술한 버전.
  (대화 로그 보관. 재생성 시 그 프롬프트에서 수정점만 바꿔 2~3컷 뽑아 베스트 선택.)

---

## §8. 검수 이력 + 발표 전 미결 항목

### 8-1. 독립 critic 에이전트 검수 결과 (편향 배제용 별도 에이전트)
12개 핵심 항목(encoder concat, actor=analytic/critic=regression, H=15, action 범위, progress reward,
2-stage warm-load, RSSM 무변경, world model loss 구조, s_t, 패널 B 무변경 등) **전부 코드 일치(OK)**.
다이어그램의 알고리즘 구조·색칠 논리가 코드와 거의 완벽히 일치 → **발표 사용 가능** 판정.

### 8-2. 발표 전 처리할 것

| 우선 | 항목 | 조치 |
|---|---|---|
| 🟡 | Environment 색 정합성(§2-4) | 범례 'architecture-level'로 확장 or 초록 |
| 🟡 | reward predictor 색(§2-4) | 회색 → 주황 |
| 🟢 | `networks_1d.py` stale 주석 | docstring(9행)·인라인 주석(134-136행)이 옛 스펙 `ch256/flat8704`로 거짓.
실제는 `ch128/flat2176`(코드 동작은 정상). **다이어그램은 OK**, 코드 주석만 수정해 슬라이드-코드 충돌 제거 |
| 🟢 | 배지 `joint replay 0.3` | understand 006 §7에 실행값 0.3 기록됨(정확). 100% 확정하려면 Stage2 logdir config dump로 재확인 |
| 🟢 | 패널 C Actor 노드 | 생략돼 a_t가 RSSM에서 나오는 것처럼 보임 → Actor 추가 or 발표 시 말로 보강 |

### 8-3. 손그림(사용자 직접 작도) 검수 결론
사용자 손그림은 깔끔본을 옮긴 것으로, **유일한 실제 수정사항은 `o^lidar = clip(scan,0,30)` 에 `/30` 정규화 추가**.
(검수 과정에서 손글씨 x_t/r̂_t를 z_t/v̂_t로 오독해 잘못 지적한 바 있으나, 실제로는 정상이었음 — 재지적 말 것.)

---

## §9. 발표 멘트(그림 띄우고 할 말, 추천)

> *"빨강(encoder/decoder/env)·주황(action/reward 의미)만 바뀌었고, 상상 기반 actor-critic 학습 코어는
> 전부 회색 = `models.py`는 사실상 무변경이다. 우리가 한 건 알고리즘 교체가 아니라 — 라이다 관측,
> progress 보상, 환경 어댑터, 2-stage 커리큘럼을 끼운 것이다."*

색 라벨링(§2)부터 짚으면 청중이 색만 보고 "어디가 바뀌었나"를 따라온다.
