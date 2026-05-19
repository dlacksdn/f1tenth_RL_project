# 005 - Dreamer-v3 코드 분석 (Part 3: ImagBehavior 세부 + 분포 head)

> **목적**: `NM512/dreamerv3-torch` 코드 자체 이해. F1TENTH 통합 설계가 아니라 **알고리즘 코드 자체** 분석.
> **선행 문서**:
> - [003-dreamer_code_analysis_part1.md](003-dreamer_code_analysis_part1.md) — 진입점·WorldModel·ImagBehavior 개요
> - [004-dreamer_code_analysis_part2.md](004-dreamer_code_analysis_part2.md) — RSSM·Encoder/Decoder
> **분석 대상**: `/home/dlacksdn/dreamerv3-torch/models.py`, `tools.py`
> **작성일**: 2026-05-20

---

## 0. 다음 세션 핸드오프 (READ FIRST)

### 현재까지 진행 상태
- ✅ 1단계: 진입점 + Config (`dreamer.py`, `configs.yaml`) — 003 문서
- ✅ 2단계: WorldModel + ImagBehavior 개요 (`models.py`) — 003 문서
- ✅ 3단계: RSSM 동역학 (`networks.py`의 `RSSM`) — 004 §1
- ✅ 4단계: Encoder/Decoder (`networks.py`) — 004 §2
- ✅ 5단계: **ImagBehavior 세부** — λ-return, RewardEMA, slow target self-distill, advantage 정규화 — **이 문서 §1**
- ✅ 6단계: **Heads & 분포** — `SymlogDist`/`DiscDist`/`OneHotDist`/twohot — **이 문서 §2**
- ⏭️ **7단계 (NEXT)**: Exploration — `exploration.py`의 `Plan2Explore`, `Random`. WorldModel의 disagreement ensemble이 어떻게 imagination reward로 들어가는지.

### 전체 로드맵 (003에서 정의, 변경 없음)
1. ✅ 진입점·실행 흐름 (`dreamer.py`)
2. ✅ WorldModel (`models.py`)
3. ✅ RSSM 동역학 (`networks.py`의 `RSSM`)
4. ✅ Encoder/Decoder (`networks.py`)
5. ✅ ImagBehavior 세부
6. ✅ Heads & 분포
7. ⏭️ **Exploration** (`exploration.py`) — `Plan2Explore`, `Random`
8. 데이터 파이프라인 (`tools.py`의 `load_episodes`/`sample_episodes`/`from_generator`/`simulate`)
9. 유틸: Optimizer 래퍼(✅ §1-9에서 다룸), EMA(✅ §1-1, §1-2), slow target(✅ §1-2), gradient clipping(✅ §1-9)
- **제외**: `envs/`, `parallel.py`, `Dockerfile`, `requirements.txt`

### 파일 위치
```
/home/dlacksdn/dreamerv3-torch/
├── dreamer.py         365 lines  (✅ 003)
├── models.py          441 lines  (✅ 003 + 005 §1)
├── networks.py        810 lines  (✅ 003 + 004)
├── tools.py          1000 lines  (✅ 005 §1-§2 부분, 8단계에서 데이터 파이프라인)
├── exploration.py     135 lines  (⏭️ 7단계)
├── parallel.py        209 lines  (제외)
├── configs.yaml       184 lines  (✅ 003)
└── envs/                         (제외)
```

### 사용자 작업 컨벤션 (반드시 지킬 것)
- `_thinking/analysis/`는 **append-only**. 기존 파일 절대 수정 금지.
- 새 파일은 명시적 요청("N 문서로 저장")이 있을 때만 저장.
- 파일 이름은 `NNN-dreamer_code_analysis_partK.md` 형식 (003·004와 통일).
- **한글로 응답**.
- 코드 인용은 markdown link `[파일:라인](../../../dreamerv3-torch/파일#L라인)` 형식.
- 분석은 **dreamer-v3 코드 자체 이해**가 목적. F1TENTH 통합 얘기 섞지 말 것.
- 사용자 메시지 "진행해" = 다음 단계 진행 신호.

### 다음 에이전트 진입 명령 (그대로 사용)
> "`_thinking/analysis/005-dreamer_code_analysis_part3.md`의 핸드오프 섹션을 보고 7단계(Exploration)로 진행해. `/home/dlacksdn/dreamerv3-torch/exploration.py`의 `Plan2Explore`(ensemble disagreement 보너스, one_step head 학습 흐름)와 `Random`(랜덤 베이스라인)을 코드 레벨로 정리하고, `dreamer.py`에서 `_expl_behavior`가 언제 호출되는지(`expl_behavior` config 분기, `expl_amount`/`expl_until`/`expl_decay` 스케줄)도 확인해. 결과 보고 후 사용자에게 저장 요청 받기를 기다려. 저장 파일명은 `006-dreamer_code_analysis_part4.md`."

---

## 1. ImagBehavior 세부 (`models.py` + `tools.py`)

### 1-0. 함수 호출 그래프
```
ImagBehavior._train(start, objective)            # models.py:290
├── _update_slow_target()                         # polyak EMA
├── _imagine(start, actor, H=15)                  # imagination rollout
│   └── tools.static_scan(step, range(H), ...)
├── reward = objective(feat, state, action)       # 외부 주입 람다
├── _compute_target(feat, state, reward)
│   ├── cont_head(feat).mean                      # γ = discount · P(continue)
│   ├── value(feat).mode()
│   └── tools.lambda_return(...)
│       └── tools.static_scan_for_lambda_return   # 역방향 스캔
├── _compute_actor_loss(feat, action, target, weights, base)
│   └── self.reward_ema(target, ema_vals)         # 5/95 quantile EMA
├── value loss (slow_value self-distill 포함)
└── _actor_opt(...) + _value_opt(...)             # AMP, clip, decoupled WD
```

### 1-1. `RewardEMA` — 5/95 quantile EMA ([models.py:11-26](../../../dreamerv3-torch/models.py#L11-L26))

```python
def __init__(self, device, alpha=1e-2):
    self.range = torch.tensor([0.05, 0.95], device=device)

def __call__(self, x, ema_vals):
    flat_x = torch.flatten(x.detach())                       # ① target을 1D로
    x_quantile = torch.quantile(flat_x, q=self.range)        # ② 현재 batch의 5%, 95% quantile
    ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals   # ③ in-place EMA
    scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)   # ④ 5~95 range = scale (하한 1.0)
    offset = ema_vals[0]
    return offset.detach(), scale.detach()
```

핵심:
- `alpha=1e-2` → 매 actor 업데이트마다 1%만 새 값 반영, 99% 과거 보존. 매우 느린 EMA.
- `ema_vals`는 `nn.Module`의 buffer로 등록 ([models.py:285-287](../../../dreamerv3-torch/models.py#L285-L287)) → `torch.save`에 같이 저장.
- 호출 위치는 **단 한 곳**: `_compute_actor_loss`에서 `target` 정규화 ([models.py:404-408](../../../dreamerv3-torch/models.py#L404-L408)).
- value loss나 critic 학습에는 들어가지 않음 — actor advantage 스케일 정규화 전용.
- `scale ≥ 1.0` 클리핑 → 보상이 극단적으로 좁은 분포일 때 scale=0으로 폭발하는 것 방지.

### 1-2. `_update_slow_target` — polyak EMA ([models.py:435-441](../../../dreamerv3-torch/models.py#L435-L441))

```python
def _update_slow_target(self):
    if self._config.critic["slow_target"]:                   # True (default)
        if self._updates % self._config.critic["slow_target_update"] == 0:   # =1 매 step
            mix = self._config.critic["slow_target_fraction"]                # =0.02
            for s, d in zip(self.value.parameters(),
                            self._slow_value.parameters()):
                d.data = mix * s.data + (1 - mix) * d.data   # in-place polyak
        self._updates += 1
```

- `slow_target_update=1` (configs.yaml 기본): 매 ImagBehavior._train 호출마다 EMA 한 번.
- `slow_target_fraction=0.02`: τ=0.02. half-life ≈ 34 step.
- `_slow_value = copy.deepcopy(self.value)` ([models.py:258](../../../dreamerv3-torch/models.py#L258))로 생성. requires_grad는 그대로지만 옵티마이저에 들어가지 않아 gradient는 받지 않고 EMA만으로 갱신.
- 호출 시점: `_train` 진입 직후 ([models.py:295](../../../dreamerv3-torch/models.py#L295)) — actor/value 업데이트 **이전**.

### 1-3. `objective`는 외부 주입 람다

`_train(self, start, objective)` 시그니처. reward를 만드는 `objective`는 호출처에서 정의.

`dreamer.py`의 task behavior 호출 ([dreamer.py:108-112](../../../dreamerv3-torch/dreamer.py#L108-L112) 부근):
```python
reward = lambda f, s, a: self._wm.heads["reward"](
    self._wm.dynamics.get_feat(s)).mode()
self._task_behavior._train(start, reward)
```

- `objective: (feat, state, action) → reward[T, B*N, 1]` 임의의 람다.
- 기본은 `reward_head(feat).mode()` — `DiscDist.mode()` = `symexp(Σ p·b)`. **원래 스케일의 reward**가 lambda_return에 들어감 (symlog 공간 X).
- Plan2Explore에서는 disagreement 보너스로 교체 (7단계 예정).

### 1-4. `_imagine` — imagination rollout ([models.py:351-369](../../../dreamerv3-torch/models.py#L351-L369))

```python
flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))      # (B,T,...) → (B*T,...)
start = {k: flatten(v) for k, v in start.items()}            # post를 펼침

def step(prev, _):
    state, _, _ = prev
    feat = dynamics.get_feat(state)
    inp = feat.detach()                                       # ★ actor에는 detach된 feat
    action = policy(inp).sample()                             # straight-through 적용
    succ = dynamics.img_step(state, action)                   # prior 한 스텝
    return succ, feat, action

succ, feats, actions = tools.static_scan(
    step, [torch.arange(horizon)], (start, None, None))       # horizon=15
states = {k: torch.cat([start[k][None], v[:-1]], 0)           # 첫 step에 start 끼우기
          for k, v in succ.items()}
```

세부 포인트:
- **`inp = feat.detach()`** ([L359](../../../dreamerv3-torch/models.py#L359)) — actor가 RSSM으로 gradient를 흘리지 못하게 차단. 단, `succ = img_step(state, action)`의 `state`는 detach되지 않아 dynamics gradient는 다음 step의 feat까지 살아 있음. 그래서 `imag_gradient='dynamics'` 모드에서 actor가 RSSM·decoder를 통해 학습 가능.
- horizon = `imag_horizon = 15` (configs.yaml).
- `start`는 WorldModel `_train`이 만든 post(`[B=16, T=64, ...]`)를 `flatten`으로 합쳐 batch dim이 `16*64=1024`. imagination은 **1024개 시작점에서 동시에 15-step rollout**.
- 출력 shape: `feats=[T, B*T_data, F]`, `actions=[T, B*T_data, A]`. `feats[0]`은 start state의 feat, `feats[T-1]`은 마지막 prior의 feat.
- `states`는 모든 step의 prior state (첫 자리에 start post를 끼움) — `_compute_target`의 cont head 입력.

### 1-5. `_compute_target` — λ-return + weights ([models.py:371-389](../../../dreamerv3-torch/models.py#L371-L389))

```python
inp = dynamics.get_feat(imag_state)
discount = self._config.discount * heads["cont"](inp).mean   # γ · P(continue)  (γ=0.997)
value = self.value(imag_feat).mode()                          # [T, B, 1]
target = tools.lambda_return(
    reward[1:],          # [T-1, B, 1]   (t=1..T-1)
    value[:-1],          # [T-1, B, 1]   (t=0..T-2)
    discount[1:],        # [T-1, B, 1]
    bootstrap=value[-1], # [B, 1]        terminal value
    lambda_=0.95,
    axis=0,
)
weights = torch.cumprod(
    torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
).detach()
return target, weights, value[:-1]                            # base = V(s_t)
```

인덱싱·shape 가이드 (H=imag_horizon=15, N=B*T_data=1024):
```
imag_feat:    [H=15,   N, F]    feat[0]=start, feat[H-1]=마지막 prior
imag_action:  [H,      N, A]
imag_state:   [H,      N, ...]
reward:       [H,      N, 1]
value:        [H,      N, 1]
discount:     [H,      N, 1]

lambda_return 입력 (axis=0 기준):
  reward[1:]    → 길이 H-1=14   r_1..r_14
  value[:-1]    → 길이 H-1      V_0..V_13
  discount[1:]  → 길이 H-1      γ_1..γ_14
  bootstrap=value[-1]            V_14
target: tuple of 14, 각 [N, 1]            → stack(dim=1) → [N, 14, 1]
weights: cumprod, 길이 H=15               → weights[:-1] = [14, N, 1]
actor_loss = weights[:-1] * actor_target[:-1]
```

핵심 디테일:
- **discount는 학습되는 cont head의 출력**. `cont = Bernoulli(p)`, `cont.mean = p`. 종료 가능성이 높은 state에서 γ가 자동으로 작아짐. hard done 처리 없이 부드러운 종료.
- **reward를 한 칸 뒤로 미는 인덱싱**: `target_t ← reward_{t+1} + γ_{t+1} (λ V_{t+1} + (1-λ) target_{t+1})`. λ-return의 표준 정의.
- **`weights`** = `[1, γ_0, γ_0·γ_1, …]` cumprod → t step의 actor loss가 종료 확률에 따라 가중. `.detach()`로 gradient 차단 (continue head를 actor에서 학습시키지 않음).
- `value`는 `self.value(imag_feat).mode()` — slow target이 아니라 **현재 value의 mode**. 부트스트랩은 현재 value, 정답 target은 별도 계산되어 value loss가 그 target에 맞춰 학습됨.

### 1-6. `tools.lambda_return` ([tools.py:691-717](../../../dreamerv3-torch/tools.py#L691-L717))

```python
next_values = torch.cat([value[1:], bootstrap[None]], 0)      # V_{t+1}, 마지막은 bootstrap
inputs = reward + pcont * next_values * (1 - lambda_)         # r_t + γ(1-λ) V_{t+1}
returns = static_scan_for_lambda_return(
    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    (inputs, pcont), bootstrap)
```

재귀식:
```
G_T   = bootstrap
G_t   = (r_{t+1} + γ_{t+1}(1-λ) V_{t+1}) + γ_{t+1} λ G_{t+1}
      = r_{t+1} + γ_{t+1} [(1-λ) V_{t+1} + λ G_{t+1}]
```

`λ=1` → discounted Monte Carlo, `λ=0` → 1-step TD. Dreamer-v3 기본 `discount_lambda=0.95`.

**`static_scan_for_lambda_return`** ([tools.py:671-688](../../../dreamerv3-torch/tools.py#L671-L688)):
- `reversed(indices)` 시간 역방향 스캔 (재귀가 미래→현재 방향이므로).
- 매 step `last = fn(last, inputs[i], pcont[i])` 후 `torch.cat(..., dim=-1)`로 누적 → reshape & flip → `torch.unbind(dim=0)` → tuple of T-1 텐서.
- ⚠️ 반환 타입이 **tuple** — 그래서 `_compute_actor_loss`/value loss에서 `torch.stack(target, dim=1)`이 필요.

### 1-7. `_compute_actor_loss` — 3가지 gradient 모드 ([models.py:391-433](../../../dreamerv3-torch/models.py#L391-L433))

```python
inp = imag_feat.detach()                                     # actor 입력 detach
policy = self.actor(inp)
target = torch.stack(target, dim=1)                           # tuple → [N, T-1, 1]

if reward_EMA:
    offset, scale = self.reward_ema(target, self.ema_vals)
    normed_target = (target - offset) / scale
    normed_base   = (base   - offset) / scale
    adv = normed_target - normed_base                         # advantage (정규화됨)
else:
    adv = target - base

if imag_gradient == "dynamics":         # default for continuous
    actor_target = adv                                        # path-gradient: target→feat→action
elif imag_gradient == "reinforce":      # default for discrete
    actor_target = log π(a|s)[:-1] * (target - V(s)).detach()
elif imag_gradient == "both":
    mix = imag_gradient_mix
    actor_target = mix*target + (1-mix)*log_prob*adv.detach()

actor_loss = -weights[:-1] * actor_target
```

3 모드 의미:
- **`dynamics`**: actor 출력 → RSSM(img_step에서 state 유지) → feat → value/reward를 통한 path gradient. continuous action에서 효과적 (액션이 미분가능). `target`이 그대로 loss에 들어가 backprop이 전체 imagination을 통과.
- **`reinforce`**: 표준 policy gradient. discrete action(onehot)에서 사용. baseline은 `V(s)` 현재값. `(target - V).detach()`로 critic을 통해 backprop 안 됨.
- **`both`**: 두 항을 `mix` 비율로 결합.

엔트로피 보너스는 별도로 ([models.py:317](../../../dreamerv3-torch/models.py#L317)):
```python
actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]   # entropy=3e-4
```

`weights[:-1]`로 곱해 종료 후 step의 loss는 0에 수렴. `[:-1]`은 마지막 step은 bootstrap이라 target이 없기 때문.

### 1-8. Value loss — 두 항 동등 합산 ([models.py:322-332](../../../dreamerv3-torch/models.py#L322-L332))

```python
value = self.value(value_input[:-1].detach())                # imag_feat detach
target = torch.stack(target, dim=1)
value_loss = -value.log_prob(target.detach())                # ① λ-return target에 대한 NLL
slow_target = self._slow_value(value_input[:-1].detach())
if critic["slow_target"]:
    value_loss -= value.log_prob(slow_target.mode().detach()) # ② slow target의 mode에 대한 NLL
value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])
```

- 두 항이 **단순 합산** (가중치 없음). slow_target 항은 self-distill로 작용 → critic이 자기 EMA로 끌려가 학습 안정화.
- `value`는 `symlog_disc` 분포 (255-bin DiscDist) — `log_prob(target)`은 target을 symlog 공간에서 255 bin twohot 인코딩한 cross-entropy (§2-2 참조).
- `target.detach()` 명시적: λ-return은 dynamics gradient를 받지만 value loss용으로는 끊는다.
- `value_input[:-1].detach()` — value head 학습이 actor·world model로 흘러가지 않도록 차단.

### 1-9. Optimizer 흐름 ([tools.py:720-772](../../../dreamerv3-torch/tools.py#L720-L772))

```python
def __call__(self, loss, params, retain_graph=True):
    self._opt.zero_grad()
    self._scaler.scale(loss).backward(retain_graph=retain_graph)   # AMP scale
    self._scaler.unscale_(self._opt)                                # unscale 전에 clip
    norm = torch.nn.utils.clip_grad_norm_(params, self._clip)       # ① global norm clip
    if self._wd:
        self._apply_weight_decay(params)                            # ② decoupled WD
    self._scaler.step(self._opt)
    self._scaler.update()
    self._opt.zero_grad()
    return {"_loss": ..., "_grad_norm": norm}
```

- **AMP + grad clip 정석 패턴**: `scale → backward → unscale → clip → step → update`.
- **Decoupled weight decay** ([L767-772](../../../dreamerv3-torch/tools.py#L767-L772)): `var.data *= (1 - wd)`. AdamW와 동일한 방식이지만 직접 구현. `wd_pattern='.*'` 아니면 NotImplemented (모든 파라미터 동일하게).
- `retain_graph=True` — 같은 graph로 actor → value를 연달아 backward 하기 위해.
- 호출 순서 ([models.py:346-348](../../../dreamerv3-torch/models.py#L346-L348)):
  ```python
  metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
  metrics.update(self._value_opt(value_loss, self.value.parameters()))
  ```
  actor 먼저, value 나중. 둘 다 `imag_feat`를 detach하므로 두 graph가 독립적이지만 imagination rollout(`img_step` 내부 state)을 공유하기 때문에 `retain_graph=True` 필수.

### 1-10. 설계 포인트

1. **`feat.detach()`는 actor 입력에만**: imagination rollout 자체의 `img_step(state, action)`은 state를 detach하지 않아 dynamics gradient가 살아 있다. 이 비대칭성이 `imag_gradient='dynamics'` 모드의 핵심.
2. **discount = γ · cont_head.mean**: hard done이 아닌 soft termination. weights도 cont를 통해 자동 0 수렴.
3. **RewardEMA는 actor만**: critic은 raw target에 학습 → critic 출력이 EMA에 의존하지 않아 부트스트랩 자기참조 회피.
4. **Value loss self-distill (slow_target.mode 항)**: target net을 부트스트랩으로만 쓰는 게 아니라 별도 항으로 직접 distill → λ-return target이 잡음일 때도 slow가 anchor 역할.
5. **`slow_target_update=1` + `fraction=0.02`**: 매 step polyak. 큰 주기 + hard copy 대신 부드러운 EMA로 안정성.
6. **3가지 imag_gradient 모드**: continuous는 `dynamics`(path), discrete는 `reinforce`(REINFORCE), `both`는 두 모드 결합. v3는 환경별로 actor.dist에 따라 자동 선택.
7. **`retain_graph=True`**: actor와 value가 같은 imagination graph를 공유 — 두 번 backward 가능해야 함.
8. **lambda_return이 tuple 반환**: `static_scan_for_lambda_return`이 unbind로 끝나서 tuple. 호출처에서 `torch.stack(..., dim=1)`로 시간축을 두 번째로 두는 형태(`[N, T, ...]`)로 정규화.
9. **`objective` 람다 주입 설계**: reward 함수를 외부에서 갈아끼울 수 있어 task behavior(=실제 reward)와 explore behavior(=disagreement 보너스)를 같은 `ImagBehavior` 클래스로 학습 가능.

---

## 2. Heads & 분포 (`tools.py`)

### 2-1. `symlog` / `symexp` ([tools.py:23-28](../../../dreamerv3-torch/tools.py#L23-L28))

```python
symlog(x) = sign(x) · log(1 + |x|)
symexp(x) = sign(x) · (exp(|x|) - 1)
```

- 부호 보존 로그 변환. `symlog(0)=0`, 단조증가·미분가능(원점 제외 매끄러움).
- `symexp(symlog(x)) = x` 정확히 역함수.
- 미분: `d symlog/dx = 1/(1+|x|)`. 큰 |x|에서 gradient가 줄어 outlier에 둔감.
- Dreamer-v3는 이 함수를 **세 곳**에서 사용:
  1. encoder MLP 입력 — raw 관측을 압축 ([networks.py:659-660](../../../dreamerv3-torch/networks.py#L659-L660))
  2. `SymlogDist` — vector decoder 출력 ([tools.py:532-561](../../../dreamerv3-torch/tools.py#L532-L561))
  3. `DiscDist` — reward/value head ([tools.py:452-506](../../../dreamerv3-torch/tools.py#L452-L506))

### 2-2. `DiscDist` — twohot 인코딩 categorical ([tools.py:452-506](../../../dreamerv3-torch/tools.py#L452-L506))

reward head, value head 둘 다 이거. 스칼라 회귀를 **255-bin 분류**로 바꾼다.

#### bucket 구성
```python
self.buckets = torch.linspace(-20.0, 20.0, steps=255)  # symlog 공간 255개
self.width   = 40.0 / 255 ≈ 0.157
```
- 도메인: **symlog 공간**의 [-20, 20]. symexp 복원하면 원래 스케일 [-exp(20)+1, exp(20)-1] ≈ ±4.85e8. 사실상 모든 보상/가치를 커버.

#### `mode()` / `mean()`
```python
expected_symlog = Σ_i probs_i · buckets_i      # 가중합 (255 bin)
return symexp(expected_symlog)                  # 원래 스케일로 복원
```
- `mode`와 `mean`이 동일 — 진짜 mode가 아닌 expectation. categorical의 expectation을 점추정으로 쓴다.

#### `log_prob(x)` — twohot 인코딩 (핵심)
```python
x = symlog(x)                                          # 1. target을 symlog 공간으로
# x shape: [..., 1] 가정 (reward/value는 마지막 차원 1)
below = max index where buckets[i] <= x[..., None]    # [..., 1] (broadcast 위해 차원 추가)
above = below + 1 (혹은 동일)
below = clip(below, 0, 254);  above = clip(above, 0, 254)
dist_to_below = |buckets[below] - x|
dist_to_above = |buckets[above] - x|
weight_below = dist_to_above / total                   # 가까운 쪽이 큰 가중치
weight_above = dist_to_below / total
target = onehot(below)*w_below + onehot(above)*w_above # 두 bin에만 질량
target = target.squeeze(-2)                            # ★ [..., 1, 255] → [..., 255]
log_pred = logits - logsumexp(logits)                  # log_softmax
return (target * log_pred).sum(-1)                     # cross-entropy
```

**왜 twohot인가**:
- 표준 one-hot은 가까운 두 bin 사이의 부드러운 회귀 신호를 못 준다. twohot은 정확히 두 인접 bin에 거리 가중치로 질량을 분배 → 회귀와 분류의 장점 결합.
- C51(distributional RL)의 projection과 같은 아이디어.

**구현 디테일**:
- `x[..., None]`로 마지막에 차원 하나 추가 후 `buckets`와 broadcast 비교 → `below`/`above`가 `x`와 같은 leading shape. value/reward는 `[..., 1]`이라 `target.squeeze(-2)`로 그 차원을 제거해야 `log_pred [..., 255]`와 곱이 맞는다.
- target이 [-20, 20] 밖이면 `clip(below/above, 0, 254)` → out-of-range gradient는 끝 bin에 모임. 정상 학습 범위는 symexp(20)≈4.85e8까지라 거의 안 부딪힘.

### 2-3. `SymlogDist` — vector reconstruction ([tools.py:532-561](../../../dreamerv3-torch/tools.py#L532-L561))

MultiDecoder의 vector head에서 사용. **연속 분포가 아니라 MSE의 symlog 버전**.

```python
log_prob(value):
    distance = (self._mode - symlog(value))**2      # target만 symlog
    distance = where(distance < 1e-8, 0, distance)  # 미세 노이즈는 0
    return -distance.sum(축 ≥ 2)
```

- `self._mode`는 raw logits (Linear 출력) — 이미 symlog 공간의 값으로 학습됨.
- `mode()` / `mean()` = `symexp(self._mode)` → 추론 시 원래 스케일 복원.
- agg='sum': 마지막 차원들을 모두 합. shape `[B, T, D]`면 D를 합쳐 `[B, T]` loss.
- 분포가 아니라 **deterministic predictor + MSE loss**의 클래스 포장. `log_prob`이 가우시안 log-prob의 -MSE 항만 남기고 정규화 상수와 표준편차를 제거한 형태.
- `tol=1e-8`: float16 AMP에서 underflow로 NaN 발생 방지. 일반 학습에서는 거의 안 걸림.

### 2-4. `MSEDist` — image reconstruction ([tools.py:509-529](../../../dreamerv3-torch/tools.py#L509-L529))

`SymlogDist`에서 symlog만 뺀 것. image_dist 기본값.

```python
log_prob(value) = -((mode - value)**2).sum(축 ≥ 2)
```
- 이미지는 `[B, T, H, W, C]` — 축 2,3,4를 합쳐 `[B, T]` loss.
- 이미지는 이미 `[-0.5, 0.5]`로 정규화된 후라 symlog 불필요.

### 2-5. `OneHotDist` — straight-through 카테고리 ([tools.py:425-449](../../../dreamerv3-torch/tools.py#L425-L449))

`torch.distributions.OneHotCategorical` 상속. RSSM stoch (discrete), actor (discrete action)에 사용.

#### unimix
```python
if unimix_ratio > 0:
    probs = softmax(logits)
    probs = probs*(1-α) + α/K        # 균등 1%(=0.01) 혼합
    logits = log(probs)              # 다시 로그로 (super().__init__이 logits 받음)
```
- α=0.01: 모든 카테고리에 최소 0.01/K 확률 보장 → 0 확률로 인한 log(0)·dead exploration 방지.
- 0이 되지 않으므로 KL divergence도 안정.

#### Straight-through estimator
```python
def sample(self, ...):
    sample = super().sample(...).detach()    # one-hot (gradient 끊김)
    probs = super().probs
    sample += probs - probs.detach()         # ★ forward: one-hot, backward: probs
    return sample

def mode(self):
    _mode = F.one_hot(argmax(logits))
    return _mode.detach() + logits - logits.detach()    # 같은 트릭
```

- `a + b - b.detach()`: forward 값은 `a + 0 = a` (one-hot), backward gradient는 `b`의 것(probs). PyTorch의 STE 표준 패턴.
- 이 트릭 덕분에 discrete RSSM stoch에 backprop이 흐른다 (path gradient).

#### RSSM에서의 wrapping
`OneHotDist(logit, unimix=0.01)` 자체는 `[..., 32, 32]` 카테고리 1개. RSSM은 [networks.py:246-247](../../../dreamerv3-torch/networks.py#L246-L247)에서 `torchd.independent.Independent(dist, 1)`로 마지막 32-그룹을 독립으로 reinterpret → **KL이 32개 카테고리의 KL 합**으로 계산됨. `log_prob`도 32개 합. 이 wrapping이 없으면 32×32를 단일 큰 카테고리로 보게 되어 의도와 어긋남.

### 2-6. `ContDist` — continuous actor 래퍼 ([tools.py:564-590](../../../dreamerv3-torch/tools.py#L564-L590))

`Normal` 또는 `Independent(Normal, k)`를 감싸 `absmax` 제약 추가.

```python
def mode(self):
    out = self._dist.mean
    if absmax is not None:
        out *= (absmax / clip(|out|, min=absmax)).detach()    # |out| > absmax이면 스케일 다운
    return out
```

- `absmax=1.0` (ImagBehavior actor) → 액션이 [-1, 1] 박스 밖으로 나가는 걸 부드럽게 제한. `tanh`보다 약한 클리핑.
- detach: 클리핑 자체에 gradient가 흐르지 않게.
- `sample`은 `rsample` 사용 → reparameterization gradient 가능.

### 2-7. `Bernoulli` — cont head ([tools.py:593-617](../../../dreamerv3-torch/tools.py#L593-L617))

continue 확률(에피소드 종료 안 함)을 예측. `torch.distributions.Bernoulli` 래퍼.

```python
def mode(self):
    _mode = torch.round(self._dist.mean)            # 0 or 1
    return _mode.detach() + mean - mean.detach()    # STE

def log_prob(self, x):
    _logits = self._dist.base_dist.logits
    log_p0 = -softplus(logits)
    log_p1 = -softplus(-logits)
    return (log_p0*(1-x) + log_p1*x).sum(-1)        # numerically stable BCE
```

- `softplus` 기반 BCE — 직접 `log(p)·x + log(1-p)·(1-x)` 하면 logits가 클 때 overflow. 이 형태는 항상 안정.
- `.mean`은 Bernoulli이라 = `sigmoid(logits)` = 확률. 그래서 `_compute_target`의 `cont_head(inp).mean`이 discount 계수가 된다 (§1-5 참조).

### 2-8. `UnnormalizedHuber` ([tools.py:620-631](../../../dreamerv3-torch/tools.py#L620-L631))

```python
log_prob(event) = -(√((event - μ)² + threshold²) - threshold)
```
- pseudo-Huber loss. threshold=1에서 작은 오차 ≈ 0.5·err², 큰 오차 ≈ |err| - 0.5.
- Normal log-prob처럼 쓸 수 있는 robust loss. 기본 config에서는 거의 안 쓰임 (옵션).

### 2-9. `SafeTruncatedNormal`, `TanhBijector`, `SampleDist`

- **`SafeTruncatedNormal`** ([L634-649](../../../dreamerv3-torch/tools.py#L634-L649)): `trunc_normal` actor 선택 시. sample 후 `[low+ε, high-ε]`로 STE clip. `_mult`로 액션 스케일 조정.
- **`TanhBijector`** ([L652-668](../../../dreamerv3-torch/tools.py#L652-L668)): SAC식 `tanh_normal` actor. inverse에서 ±1 근처 NaN 방지 clamp.
- **`SampleDist`** ([L398-422](../../../dreamerv3-torch/tools.py#L398-L422)): tanh_normal 같이 analytic mean/mode가 없는 분포용. N=100 샘플로 mean·mode·entropy 추정. 기본 config는 안 씀.

### 2-10. 분포-head 매핑 한눈에

| Head | 분포 class | dist 이름 | log_prob target | mode 출력 |
|---|---|---|---|---|
| **image decoder** | `MSEDist` | `mse` | raw pixel | raw + 0.5 |
| **vector decoder** | `SymlogDist` | `symlog_mse` | symlog(target) | symexp(logits) |
| **reward head** | `DiscDist` | `symlog_disc` | twohot(symlog(r), 255 bin) | symexp(Σ p·b) |
| **value head** | `DiscDist` | `symlog_disc` | twohot(symlog(V), 255 bin) | symexp(Σ p·b) |
| **cont head** | `Bernoulli` | `binary` | BCE(continue) | round + STE |
| **RSSM stoch (discrete)** | `Independent(OneHotDist, 1)` | — | KL between post/prior | one-hot + STE |
| **RSSM stoch (continuous)** | `ContDist(Independent(Normal,1))` | — | KL Normal | mean (clipped) |
| **actor (continuous)** | `ContDist(Independent(Normal,1))`, `tanh(mean)` | `normal` | log Normal | tanh(mean), absmax=1 |
| **actor (discrete)** | `OneHotDist` | `onehot` | log onehot | argmax + STE |

### 2-11. 학습 신호 흐름 with 분포

```
data["image"]    ←→  MSEDist        ← decoder image head
data[vec_key]    ←→  SymlogDist     ← decoder vector head (symlog target)
data["reward"]   ←→  DiscDist       ← reward head (twohot symlog 255 bin)
data["cont"]     ←→  Bernoulli      ← cont head (1-is_terminal)
post.stoch       ←→  OneHotDist KL  ← prior.stoch       (representation/dynamics loss)

imagination:
  prior.stoch    ← OneHotDist.sample (STE)
  actor          ← ContDist|OneHotDist.sample (STE for onehot, rsample for cont)
  reward         ← DiscDist.mode() = symexp 복원 (원래 스케일 reward)
  value          ← DiscDist.mode()
  cont           ← Bernoulli.mean = sigmoid (soft discount)
  target(λ-ret)  ← DiscDist.log_prob (value 학습 시, twohot)
```

### 2-12. 설계 포인트

1. **회귀 → 분류 변환 (twohot DiscDist)**: reward·value 스케일이 환경마다 천차만별이라 가우시안 회귀는 분산 튜닝이 필요. 255-bin twohot은 분산 hyperparameter를 제거하면서 분포적 표현을 유지.
2. **symlog 공간에서 분류**: bucket이 symlog의 [-20, 20] → 원래 스케일 ±4.85e8까지 단일 hyperparameter로 커버. **Dreamer-v3가 모든 환경에서 같은 config로 도는 핵심.**
3. **STE 일관성**: OneHotDist (RSSM stoch, discrete action), Bernoulli (cont head) 모두 같은 `a.detach() + b - b.detach()` 패턴. forward는 hard, backward는 soft.
4. **unimix 0.01**: discrete categorical의 모든 곳(RSSM stoch, discrete actor)에 적용. 0 probability 회피.
5. **SymlogDist는 진짜 분포가 아님**: log_prob이 -MSE에 symlog target만 더한 형태. 정규화 상수 없는 "loss 함수의 분포 인터페이스 래핑". MSE/Huber가 그렇듯 maximum likelihood ≠ scale-aware.
6. **`ContDist.absmax`**: tanh-saturation 없이 부드럽게 액션 박스 제약. mean의 gradient는 살리되 크기만 detach.
7. **out-of-range는 clip**: DiscDist twohot이 [-20, 20] 밖이면 끝 bin에 몰림 → 발산 시에도 학습이 무너지지 않음. 안전장치.
8. **`Independent(OneHotDist, 1)` wrapping**: 32×32 latent에서 32개 카테고리를 독립적으로 보게 만들어 KL이 합 형태로 계산되게 함. RSSM의 `get_dist`에서만 적용 ([networks.py:246-247](../../../dreamerv3-torch/networks.py#L246-L247)).

---

## 3. 다음 세션이 바로 시작할 작업 (7단계: Exploration)

### 분석 항목 (`exploration.py` 135줄)
1. **`Random`** — uniform action 베이스라인. 어떤 분포(actor와 같은 분포?)를 쓰는지, `_train`이 no-op인지.
2. **`Plan2Explore`** —
   - **one_step ensemble**: 몇 개 head(`disag_models`)를 어떤 입력으로 학습시키는지. target은 next embed인가 next stoch인가.
   - **disagreement reward**: ensemble의 예측 분산(variance / std)을 어떤 reduction으로 스칼라화하는지.
   - **ImagBehavior reuse**: `Plan2Explore`가 내부에 별도 `ImagBehavior`를 두고 그 `objective`에 disagreement를 람다로 주입하는 구조 확인.
3. **`dreamer.py`에서 `_expl_behavior` 호출** — `config.expl_behavior`(`'greedy'`/`'random'`/`'plan2explore'`) 분기, 학습 시점, eval 시 사용 여부.
4. **`expl_amount`/`expl_until`/`expl_decay`** — 가능한 ε-greedy 비슷한 스케줄 있는지 ([dreamer.py](../../../dreamerv3-torch/dreamer.py) `_policy` 함수에서 확인).

### 작업 흐름 (권장)
1. `exploration.py` 전체 정독.
2. `dreamer.py`에서 `_expl_behavior`, `expl_*` 키워드로 호출 흐름 추적.
3. `models.py`의 `WorldModel.heads`에 `disag` head가 추가되는지(있다면 어디서), 없다면 Plan2Explore 내부에서 ensemble을 별도로 가지는지 확인.
4. 본 문서 동일 형식으로 정리(섹션 헤더, 표, 코드 블록, [파일:라인](링크)).
5. 사용자에게 결과 보고 → "006 문서로 저장" 요청 받으면 `006-dreamer_code_analysis_part4.md` 생성.

---

## 4. 컨벤션 재확인

- `_thinking/`은 append-only. 기존 문서 절대 수정 금지.
- 명시적 "N 문서로 저장" 요청 있을 때만 새 파일 작성.
- 파일 이름은 `NNN-dreamer_code_analysis_partK.md` 형식.
- 한글 응답.
- "진행해" = 다음 단계 진행.
- 분석 대상은 **dreamer-v3 코드 자체**. F1TENTH 통합 얘기 섞지 말 것.
