# 007 — DreamerV3 알고리즘 자체 + 순정 3패널 그림 텍스트 해설 (발표 핵심)

> 목적: 발표 알고리즘 파트(world model / RSSM / imagination / actor-critic)를 **논문 기준으로 엄밀하게 + 알기 쉽게** 정리.
> 직전까지 understand/001~006은 전부 "우리가 만든 환경" 위주였고, 본 문서가 처음으로 "그 환경 위에서 DreamerV3가 어떻게 학습하나"를 다룬다.
> 동반 그림(이 문서가 텍스트로 1:1 해설하는 대상):
>   `/home/dlacksdn/f1tenth_RL_project/_thinking/diagram/dreamer v3 순정그림.png` (순정 DreamerV3, 3패널)
> 코드 근거: analysis/003~006 (NM512 dreamerv3-torch 분석, 분석대상 경로 `/home/dlacksdn/dreamerv3-torch`).
> 관련: understand/004(gym API·obs dict), 005(reward), 006(2-stage warm-load).
> ★ 이 문서는 **순정 DreamerV3 알고리즘**을 다룬다. f1tenth 맞춤 수정(lidar encoder·progress reward·wrapper)은 다음 문서(대조 그림)에서.

---

## 0. 한 줄 요약

> DreamerV3는 **① 경험으로 머릿속 시뮬레이터(world model)를 짓고 → ② 그 안에서 상상으로 actor·critic을 훈련하고 → ③ 학습된 actor로 실제 환경과 상호작용해 데이터를 모은다.** ①②③이 Replay Buffer를 통해 무한 순환한다. 실제 환경 스텝은 적게(③), 학습은 상상으로 많이(②) → 데이터 효율이 높은 **모델기반 RL**.

---

## 1. 큰 그림 — 3패널 + Replay Buffer 순환

```
        ┌─────────────────────────────────────────────────┐
        │ 1. Dynamics Learning (World Model, params φ)      │  ← 경험으로 시뮬레이터 학습
        └─────────────────────────────────────────────────┘
                  │ 학습된 world model의 post(s_t)가 출발점
                  ▼
        ┌─────────────────────────────────────────────────┐
        │ 2. Behavior Learning (Actor π_θ / Critic v_ψ)     │  ← 상상 속에서 정책 훈련
        │    Latent Imagination, H=15, world model FIXED    │
        └─────────────────────────────────────────────────┘
                  │ 학습된 actor 재사용
                  ▼
        ┌─────────────────────────────────────────────────┐
        │ 3. Environment Interaction                        │  ← 실제 환경에서 데이터 수집
        └─────────────────────────────────────────────────┘
                  │ (x_t, a_t, r_t) + episode flag
                  ▼  Replay Buffer ──(샘플)──▶ 다시 패널 1
```

---

## 2. 표기 사전 (그림·수식에 등장하는 모든 기호)

| 기호 | 뜻 | 비고 |
|---|---|---|
| `x_t` | 관측 (observation) | 논문 표기. (= 일반 RL의 `o_t`) |
| `a_t` | 행동 (action) | 우리 환경은 연속 2D `[steer, speed]` |
| `r_t` | 보상 (reward) | |
| `c_t` | continue(지속) 플래그 | `c_t = 1 − is_terminal` (죽으면 0) |
| `h_t` | **결정적** recurrent state | sequence model(GRU) 출력. 6개 중 유일한 deterministic |
| `z_t` | **posterior** stochastic state | encoder가 관측 `x_t`를 **보고** 만든 것 |
| `ẑ_t` | **prior** stochastic state | dynamics predictor가 관측 **없이** 예측한 것 |
| `s_t = {h_t, z_t}` | **model state** | actor·critic·predictor의 입력 (코드의 "feat", concat) |
| `φ / θ / ψ` | world model / actor / critic 파라미터 | 학습 lane·옵티마이저 3개 분리 |
| `γ` | 할인율 | 0.997 |
| `λ` | λ-return 블렌딩 | 0.95 |
| `H` | imagination horizon | **15** (논문 hyperparameter = 코드 `imag_horizon`) |
| `S` | reward EMA 정규화 스케일 | return의 5~95 퍼센타일 폭, `max(1, S)` |
| `β_pred/β_dyn/β_rep` | world model loss 가중 | 1 / 0.5 / 0.1 |
| `sg(·)` | stop-gradient | |

- 시간 인덱스 관례: **패널 1·3 = `t`**(실제 시퀀스), **패널 2 = `τ`**(상상 시퀀스, `τ=t…t+H`).

---

## 3. 패널 1 — Dynamics Learning (World Model, 전부 φ)

### 3-1. World Model = 6개 컴포넌트 (논문 식 3)

| # | 컴포넌트 | 논문 수식 | 개성 (한 줄) | 코드(analysis) |
|---|---|---|---|---|
| 1 | **Sequence model** | `h_t = f_φ(h_{t−1}, z_{t−1}, a_{t−1})` | 과거를 `h`에 압축하는 시간의 등뼈(GRU). 유일한 결정적 state | RSSM `img_step`의 GRUCell (004 §2-3) |
| 2 | **Encoder**(representation) | `z_t ∼ q_φ(z_t \| h_t, x_t)` | 관측 `x_t`를 **보고** posterior `z` 생성 = 현실 보정 | MultiEncoder + RSSM `obs_step` posterior (004 §2-4, §3-1) |
| 3 | **Dynamics predictor** | `ẑ_t ∼ p_φ(ẑ_t \| h_t)` | 관측 **없이** `h`만으로 prior `ẑ` 예측 = 상상의 엔진 | RSSM `img_step` prior 분포 (004 §2-3) |
| 4 | **Reward predictor** | `r̂_t ∼ p_φ(r̂_t \| h_t, z_t)` | model state → 보상 (255-bin DiscDist) | heads['reward'] (005 §1-1) |
| 5 | **Continue predictor** | `ĉ_t ∼ p_φ(ĉ_t \| h_t, z_t)` | model state → 지속확률 (Bernoulli) | heads['cont'] (005 §1-1) |
| 6 | **Decoder** | `x̂_t ∼ p_φ(x̂_t \| h_t, z_t)` | 관측 재구성 = representation 학습을 강제하는 신호 | MultiDecoder (004 §3-3) |

- **RSSM = 1 + 2 + 3** (sequence model + encoder + dynamics predictor). **encoder는 RSSM의 일부**다(전처리 모듈이 아님).
- **`s_t = {h_t, z_t}`** = `h_t`(sequence model) ⊕ `z_t`(encoder)를 묶은 것. 4·5·6 predictor의 입력이자 actor/critic 입력.
  - ★ `s_t`로 들어가는 건 `h_t`와 **posterior `z_t`** 뿐. **prior `ẑ_t`는 `s_t`에 안 들어간다**(KL 전용).

### 3-2. posterior `z` vs prior `ẑ` — KL로 묶임 (DreamerV3 핵심 대비)

- **encoder(2)** 는 `x_t`를 봐서 **posterior `z_t`** 를 만든다 → 정확하지만 관측이 있어야만 가능.
- **dynamics predictor(3)** 는 `h_t`만으로 **prior `ẑ_t`** 를 예측 → 관측 없이 굴릴 수 있어 **imagination(패널 2)의 엔진**.
- 둘을 **KL로 가까이** 끌어당겨, 상상(prior)이 현실(posterior)을 따라가게 만든다 = **KL balancing**:
  - `L_dyn(φ) = max(1, KL[ sg(q_φ(z_t|h_t,x_t)) ‖ p_φ(z_t|h_t) ])` — prior를 post로 끌어당김 (β_dyn=0.5, 더 세게)
  - `L_rep(φ) = max(1, KL[ q_φ(z_t|h_t,x_t) ‖ sg(p_φ(z_t|h_t)) ])` — post를 prior로 끌어당김 (β_rep=0.1, 약하게)
  - `max(1, ·)` = **free bits**(1 nat 미만이면 안 줄임). 그림의 "KL (free bits)".
  - 비대칭(0.5 > 0.1): post가 "정답"이라 prior를 더 세게 끌어옴, encoder가 prior의 부정확함에 끌려가 망가지는 것 방지. (004 §2-7)

### 3-3. ★ World Model의 loss는 "하나"로 귀결

그림 아래쪽에서 흩어진 loss들이 한 점 `L(φ)`로 합류한다. 헷갈리지 말 것:

```
L(φ) = E_{q_φ}[ Σ_t ( β_pred·L_pred + β_dyn·L_dyn + β_rep·L_rep ) ]      ← 단일 스칼라
         └ reconstruction+reward+continue ┘   └─ KL 두 방향(free bits) ─┘
```
- `L_pred(φ) = −ln p_φ(x_t|z_t,h_t) − ln p_φ(r_t|z_t,h_t) − ln p_φ(c_t|z_t,h_t)` — **세 predictor의 NLL 합**.
- **세 항의 정답지는 전부 Replay Buffer에서**:

| `L_pred` 항 | 예측(분포) | 정답(replay) |
|---|---|---|
| decoder | `x̂_t` | `x_t` (관측) ← = **reconstruction loss** |
| reward | `r̂_t` | `r_t` (보상) |
| continue | `ĉ_t` | `c_t = 1 − is_terminal` (종료플래그) |

- **코드로도 단일 loss**: `model_loss = Σ(head NLL) + kl_loss` → 단일 `_model_opt` 한 번 backprop (005 §1-2).
- 단, **전체 DreamerV3로는 loss 3개**: world model `L(φ)` / actor `L(θ)` / critic `L(ψ)` — 옵티마이저·gradient lane 분리. "패널 1 안에서는 loss 하나, 그림 전체로는 셋."

### 3-4. `x_t`의 역할 (자주 헷갈림)

`x_t`는 `s_t`로 **직접 가지 않는다.** 역할은 딱 둘:
1. **Encoder(2)의 입력** → `z_t` 생성
2. **Decoder(6) 출력 `x̂_t`의 정답지** → reconstruction loss 비교 대상

→ 그림에서 `x_t → s_t` 화살표가 있으면 오류. `x_t`는 encoder로 들어가고, decoder 출력과 비교될 뿐.

### 3-5. Replay Buffer 내용물

`(x_t, a_t, r_t)` + **episode flag (continue/reset)**. 그림에 `(x_t,a_t,r_t)`만 적으면 불완전 — `is_terminal`(→`c_t`)·`is_first`(RSSM 리셋용)도 저장돼야 `L_pred`의 continue 항·`obs_step` 리셋이 가능하다. (코드: `obs['cont']=1−is_terminal`, 005 §1-4)
- (우리 환경 특화, understand/004): `terminated≠is_terminal`. 2바퀴 완주는 `terminated`지만 `is_terminal=False`(완주는 죽음 아님 → `c=1`). 충돌/역주행/발산만 `is_terminal=True`(`c=0`).

---

## 4. 패널 2 — Behavior Learning (Actor-Critic in Imagination)

### 4-1. I. Imagination Rollout (H=15, world model FIXED)

상상으로 미래 15스텝을 굴린다. **실제 환경·관측 없음.**
```
s_τ ──(stop-grad 입력)──▶ Actor π_θ(a_τ|s_τ) ──a_τ──▶
     Sequence model(1): h_{τ+1}=f_φ(h_τ, ẑ_τ, a_τ)
     Dynamics predictor(3): ẑ_{τ+1}∼p_φ(ẑ_{τ+1}|h_{τ+1})
     ──▶ s_{τ+1}={h_{τ+1}, ẑ_{τ+1}} ──▶ 반복 (×15)
```
- ★ **상상의 stochastic state는 prior `ẑ_τ`** (관측 없으니 encoder 못 씀, dynamics predictor로). → 패널 3의 posterior와 정반대.
  - 미세 예외: **출발점 `s_τ`(τ=0)만 posterior `z_t`**(replay에서 가져온 시작 latent, `_imagine`의 `start=post`), 이후 14스텝이 prior. 발표엔 "`ẑ_τ=prior`" 단순화로 OK.
- ★ **Actor는 `a_τ`만 출력.** 다음 state `h_{τ+1}/ẑ_{τ+1}`은 **world model**이 만든다(actor 아님). Actor에서 `h`가 나가는 화살표는 오류.
- **world model은 FIXED**(φ 동결): 지금은 정책만 훈련. world model 학습은 패널 1. 단 미분가능해서 gradient는 통과(아래 analytic gradient).

### 4-2. II. λ-Return — 각 시점의 "가치 목표"

각 `s_τ`에서 reward·continue·critic을 뽑아 종합 점수를 만든다:
```
R^λ_τ = r̂_τ + γ·ĉ_τ·[ (1−λ)·v_ψ(s_{τ+1}) + λ·R^λ_{τ+1} ],   R^λ_{t+H} = v_ψ(s_{t+H})  (bootstrap)
```
- `r̂_τ`=reward predictor, `ĉ_τ`=continue predictor(→discount `γ·ĉ`), `v_ψ`=critic.
- "당장 받은 보상(`r̂`) + 앞으로 받을 가치(`v`)"를 `λ`로 블렌딩. `λ=1`→몬테카를로, `λ=0`→1-step TD.
- ★ **`v_t`(=`v_ψ(s_t)`)는 critic이다. true value 아님.** true value는 모르는 값이고, λ-return이 그 추정 target. critic이 자기 예측으로 target을 만들되 실제 `r̂`가 매 스텝 섞여 점점 정확해짐(TD bootstrapping).
  - bootstrap에 쓰는 `v`는 **현재 critic**(slow EMA critic 아님 — slow는 critic loss의 별도 항). (005 §3-5)

### 4-3. III. 두 개의 loss

**(a) Actor loss — return 최대화**
- Advantage `A_τ = (R^λ_τ − v_ψ(s_τ)) / max(1, S)`. ("예상보다 잘했나", `S`=reward EMA 정규화)
- `L(θ) = −Σ_τ ( A_τ + η·H[π_θ(a_τ|s_τ)] )` (우리 구현 기준)
- ★ **gradient 종류 (코드 vs 논문 차이, §6)**:
  - **우리 구현(NM512, 연속 행동)** = `imag_gradient='dynamics'` = **analytic gradient**(advantage를 학습된 dynamics를 통해 직접 backprop). `log π` 항 **없음**.
  - **논문 일반형 / 순정 그림 표기** = REINFORCE형: `L(θ)=−Σ sg((R^λ−v_ψ)/max(1,S))·log π_θ(a|s) + ηH`. (discrete 기본)
  - 순정 그림은 **논문 REINFORCE형으로 그려져 있다** → 발표 시 "논문 기준으로 그렸다"고 설명. 우리 실제 코드는 analytic.
- `η·H[·]` = entropy 보너스(탐색 유지).

**(b) Critic loss — λ-return 회귀**
- `L(ψ) = −Σ_τ ln p_ψ( sg(R^λ_τ) | s_τ )` — λ-return target(stop-grad)에 회귀. symlog two-hot(255-bin DiscDist).
- **slow EMA critic**으로 안정화(target network, polyak τ=0.02, self-distill 항). (005 §3-8)
- 순정 그림의 critic 표기 `v_ψ(R_τ|s_τ)` = "critic이 state `s_τ`에서 **return `R_τ`의 분포**를 예측" → `L(ψ)`의 분포 NLL과 정합. 표준 scalar `v_ψ(s_τ)`의 분포형 표기.

### 4-4. 본질 3줄 (발표 단순화용)

> ① 상상으로 15스텝 굴린다 → ② 각 시점의 λ-return 목표를 만든다 → ③ Actor는 그 가치를 높이게(analytic gradient), Critic은 그 가치를 맞히게 학습.

나머지(EMA scale `S`·slow critic·symlog two-hot·stop-gradient·entropy)는 전부 **학습 안정화 장치** → 발표 본 슬라이드에선 "나머지는 안정화 기법" 한 줄로 덮어도 됨. 상세는 백업 슬라이드.

---

## 5. 패널 3 — Environment Interaction (Data Collection Loop)

학습된 actor로 실제 환경과 상호작용해 Replay Buffer를 채운다.
```
POMDP Environment ──x_t──▶ RSSM(obs_step) ──s_t={h_t, z_t}──▶ Actor ──a_t──▶ Environment
                    (recurrent: h_{t−1}, z_{t−1}, a_{t−1})              │
                                                                       ▼
                                          (x_t, a_t, r_t) + episode flag ──▶ Replay Buffer ──▶ 패널 1
```
- ★ **여기서 stochastic state는 posterior `z_t`** (RSSM이 실제 `x_t`를 봐서 obs_step으로 추론). → **패널 2(prior `ẑ`)와 정반대.** 이 대비가 핵심.
- **world model은 inference only**(여기선 학습 안 함, 현재 latent state 추적용).
- **Actor만 사용**. critic·predictor는 행동 결정에 불필요(학습 전용).
- training: `actor.sample()`(탐색) / eval: `actor.mode()`(결정적). v3 greedy는 sample 자체가 탐색(별도 노이즈 없음). (003 §1-2)

---

## 6. ★ 핵심 대비 — posterior vs prior (세 맥락)

| 맥락 | stochastic state | 누가 만드나 | 왜 |
|---|---|---|---|
| **패널 1 학습** | posterior `z_t` | encoder (관측 봄) | head들을 post로 학습, prior는 KL 상대편 |
| **패널 2 상상** | **prior `ẑ_τ`** | dynamics predictor (관측 없음) | imagination은 환경 없이 굴려야 함 |
| **패널 3 환경** | posterior `z_t` | encoder/obs_step (관측 봄) | 실제 관측으로 현재 latent 추론 |

> "상상은 눈 감고 굴리니까 prior, 학습·실제는 눈 뜨고 보정하니까 posterior." **발표에서 이 한 줄을 짚으면 점수 포인트.**

---

## 7. 코드(NM512) vs 논문 차이 (정직하게)

| 항목 | 논문(원본 JAX DreamerV3) / 순정 그림 | 우리 코드(NM512 PyTorch, 연속 행동) |
|---|---|---|
| Actor gradient | REINFORCE형(log π·advantage) 통일 표기 | **analytic gradient**(`imag_gradient='dynamics'`, log π 없음) |
| 발표 처리 | 순정 그림은 논문 REINFORCE형으로 그림 → "논문 기준" 설명 | 실제 구현은 analytic |
| 첨자 | 그림 수식은 `t`, rollout은 `τ` 혼용 | "논문 기준"으로 통일 설명 |
| H=15 | 논문 hyperparameter (도구 부재로 raws PDF 직접 확인은 못 함, 발표 전 부록 "Imagination horizon" 한 번 확인 권장) | 코드 `imag_horizon=15` (003 §2, 확실) |

- 그 외 v3 고유(v1 그림엔 없음): 이산 잠재(32×32 categorical)·symlog/twohot·continue head·reward EMA·slow critic·KL balancing/free bits. → "원래 그림 vs v3 차이" 질문 대비.

---

## 8. 자주 헷갈리는 Q&A (이 세션에서 실제로 나온 것)

1. **`v_t`가 critic이냐 true value냐?** → critic `v_ψ(s_t)`. true value는 모르는 값, λ-return이 그 추정 target. (§4-2)
2. **loss가 하나냐 여럿이냐?** → world model(패널1)은 `L(φ)` 하나. 전체는 `L(φ)/L(θ)/L(ψ)` 셋. (§3-3)
3. **Replay에 `x,a,r`만? `c_t`는 뭐랑 대조?** → episode flag(`1−is_terminal`)도 저장. 그게 continue predictor 정답. (§3-5)
4. **`x_t → s_t` 화살표?** → 오류. `x_t`는 encoder 입력 + decoder 재구성 정답지일 뿐. (§3-4)
5. **Actor가 `h_{τ+1}`을 출력?** → 아니다. Actor는 `a_τ`만. 다음 state는 world model(sequence+dynamics). (§4-1)
6. **상상도 posterior?** → 아니다, prior `ẑ_τ`(출발점만 posterior). 환경(패널3)이 posterior. (§6)
7. **H=15 논문 명시?** → 논문 hyperparameter이자 코드 설정값. (§7)
8. **이렇게 복잡?** → 본질은 3줄(상상 굴림 → λ-return → actor↑/critic맞춤), 나머지는 안정화 장치. (§4-4)

---

## 9. 다음 작업 (대조 그림)

순정(본 문서) ↔ f1tenth 맞춤형 대조 그림. 수정점은 (a)lidar encoder (b)progress reward (c)wrapper 3겹 (d)2-stage warm-load이고, **알고리즘 코어(RSSM/world model/actor-critic)는 거의 무변경**(models.py 9줄). 근거·실측 diff는 새 세션 핸드오프 프롬프트 참조. understand/006(2-stage), 001·005(reward), 004(wrapper)와 연결.
