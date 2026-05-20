# 005 - Dreamer-v3 코드 분석 (Part 3: WorldModel + RewardEMA + ImagBehavior 통합)

> **Revision**: 2026-05-20 — 4차원 진정성 감사 결과 직접 반영. 003 §2(WorldModel·ImagBehavior 개요) 본문을 본 문서로 흡수, RewardEMA를 단일 위치(§2)로 통합, 분포 카탈로그를 [004 §1](004-dreamer_code_analysis_part2.md#1-분포-카탈로그-toolspy)로 이전, λ-return 출력 shape 오류 정정, 누락 항목(video_pred, reward/cont outscale 차이, Optimizer 본체) 보강. 변경 내역은 [008](008-audit-changelog.md) 참조.
> **목적**: `NM512/dreamerv3-torch` 코드 자체 이해. F1TENTH 통합 설계가 아니라 **알고리즘 코드 자체** 분석.
> **선행 문서**: [003](003-dreamer_code_analysis_part1.md) 진입점·Config, [004](004-dreamer_code_analysis_part2.md) 분포·RSSM·Encoder/Decoder.
> **분석 대상**: `/home/dlacksdn/dreamerv3-torch/models.py`, 부분 `tools.py`.
> **인용 규약**: 줄 링크 `(파일#L<a>-L<b>)`만. 코드 블록 발췌 없음.

---

## 0. 핸드오프

### 현재까지 진행 상태
- ✅ 1단계 진입점·Config — [003](003-dreamer_code_analysis_part1.md)
- ✅ 2단계 분포 카탈로그 — [004 §1](004-dreamer_code_analysis_part2.md#1-분포-카탈로그-toolspy)
- ✅ 3단계 RSSM — [004 §2](004-dreamer_code_analysis_part2.md#2-rssm-동역학-networkspy의-rssm)
- ✅ 4단계 Encoder/Decoder — [004 §3](004-dreamer_code_analysis_part2.md#3-encoderdecoder-networkspy)
- ✅ 5단계 **WorldModel** — 본 문서 §1
- ✅ **RewardEMA** — 본 문서 §2
- ✅ 6단계 **ImagBehavior** — 본 문서 §3
- ✅ **Optimizer 래퍼** — 본 문서 §4
- ✅ **손실 흐름 한눈에** — 본 문서 §5
- ⏭️ **다음**: [006](006-dreamer_code_analysis_part4.md) — Exploration + 데이터 + 포팅 + 평가

### 컨벤션
- 한국어 응답.
- 코드 인용은 줄 링크만. 코드 블록 발췌 금지.
- 분석 대상은 **dreamer-v3 코드 자체**. F1TENTH 통합 코멘트 금지.
- 본 감사 작업 한정으로 003~006 직접 수정 허용.

---

## 1. WorldModel (`models.py`)

위치: [models.py:29-215](../../../dreamerv3-torch/models.py#L29-L215). encoder, RSSM dynamics, 3개 head를 묶은 nn.Module. 단일 옵티마이저 `_model_opt`로 전체 학습.

### 1-1. `__init__` — 구성요소 조립 ([models.py:30-106](../../../dreamerv3-torch/models.py#L30-L106))

| 컴포넌트 | 위치 | 역할 | 출력 shape |
|---|---|---|---|
| `encoder = MultiEncoder(shapes, **config.encoder)` | [L36](../../../dreamerv3-torch/models.py#L36) | obs(dict) → embed | `[B, T, embed_size]` |
| `embed_size = self.encoder.outdim` | [L37](../../../dreamerv3-torch/models.py#L37) | RSSM 입력 차원 결정 | — |
| `dynamics = RSSM(stoch, deter, hidden, ...)` | [L38-54](../../../dreamerv3-torch/models.py#L38-L54) | embed + action → latent state | `{stoch, deter, logit/mean/std, ...}` |
| `heads['decoder'] = MultiDecoder(feat_size, shapes, **config.decoder)` | [L60-62](../../../dreamerv3-torch/models.py#L60-L62) | feat → obs 재구성 분포(dict) | 키별 분포 |
| `heads['reward'] = MLP(feat_size, (255,), 2 layers, dist='symlog_disc', outscale=0.0)` | [L63-74](../../../dreamerv3-torch/models.py#L63-L74) | feat → reward 분포 | 255-bin DiscDist |
| `heads['cont'] = MLP(feat_size, (), 2 layers, dist='binary', outscale=1.0)` | [L75-86](../../../dreamerv3-torch/models.py#L75-L86) | feat → continue 확률 | Bernoulli |

**feat_size 계산** ([L56-59](../../../dreamerv3-torch/models.py#L56-L59)):
- discrete RSSM: `dyn_stoch · dyn_discrete + dyn_deter = 32·32 + 512 = 1536`.
- continuous RSSM: `dyn_stoch + dyn_deter = 32 + 512 = 544`.

**reward_head shape** ([L65](../../../dreamerv3-torch/models.py#L65)): `(255,) if dist=='symlog_disc' else ()`. 기본 symlog_disc이므로 255-bin DiscDist (분포 본체는 [004 §1-2](004-dreamer_code_analysis_part2.md#1-2-discdist--twohot-인코딩-categorical-toolspy452-506)).

**reward vs cont outscale 차이** (configs.yaml:54, 56):
- `reward_head.outscale = 0.0` — value/critic과 동일. Linear 마지막 weight가 거의 0 → 초기 예측 reward ≈ 0(symlog 공간), symexp 후 ≈ 0. 학습 초기 부트스트랩 안정.
- `cont_head.outscale = 1.0` — 일반 초기화. Bernoulli logit 0 근처 → continue 확률 ≈ 0.5. 종료 미정 상태에서 시작.

**옵티마이저** ([L89-98](../../../dreamerv3-torch/models.py#L89-L98)): `_model_opt = tools.Optimizer('model', self.parameters(), lr=1e-4, eps=1e-8, clip=1000, wd=0, opt='adam', use_amp=...)`. **encoder/dynamics/decoder/reward/cont 모두 단일 옵티마이저로 통합 학습**. WD=0이므로 weight decay 미적용.

**loss scale 딕셔너리** ([L103-106](../../../dreamerv3-torch/models.py#L103-L106)): `_scales = {reward: reward_head.loss_scale, cont: cont_head.loss_scale}`. configs 기본 둘 다 1.0. 다른 head(decoder, KL)는 scale 1.0 고정으로 `_scales.get(key, 1.0)` ([L143-146](../../../dreamerv3-torch/models.py#L143-L146)).

**`assert name in self.heads` for grad_heads** ([L87-88](../../../dreamerv3-torch/models.py#L87-L88)): config.grad_heads의 모든 이름이 실제 head 딕셔너리에 있는지 검증 — 오타 방지.

### 1-2. `_train(data)` — 단일 forward에서 모든 loss 계산 ([models.py:108-174](../../../dreamerv3-torch/models.py#L108-L174))

WorldModel 학습 알고리즘 핵심.

흐름:
1. `data = self.preprocess(data)` — §1-4.
2. `RequiresGrad(self)` 컨텍스트 + AMP autocast 안에서:
   - `embed = self.encoder(data)` — shape `[B, T, embed_size]`.
   - `post, prior = self.dynamics.observe(embed, data['action'], data['is_first'])` — RSSM observe ([004 §2-5](004-dreamer_code_analysis_part2.md#2-5-observe--imagine_with_action-networkspy127-152)).
   - `kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(post, prior, kl_free, dyn_scale, rep_scale)` — KL balancing ([004 §2-7](004-dreamer_code_analysis_part2.md#2-7-kl_loss--kl-balancing-본체-networkspy272-290)).
   - `assert kl_loss.shape == embed.shape[:2]` — `[B, T]` 형식 검증.
   - **head 순회** ([L128-137](../../../dreamerv3-torch/models.py#L128-L137)):
     - 매 head마다 `feat = self.dynamics.get_feat(post)` 새로 계산.
     - `feat = feat if grad_head else feat.detach()` — `name in self._config.grad_heads`로 결정. 기본 `['decoder', 'reward', 'cont']` 모두 포함 → 모두 representation에 grad 흘림.
     - `pred = head(feat)`. decoder는 `dict` 반환이라 `preds.update(pred)`로 키별 unpack. reward/cont는 단일 분포라 `preds[name] = pred`.
   - **loss 누적** ([L138-147](../../../dreamerv3-torch/models.py#L138-L147)):
     - `loss = -pred.log_prob(data[name])` per head — NLL.
     - `assert loss.shape == embed.shape[:2]` per loss.
     - `scaled = {k: v * self._scales.get(k, 1.0) for k, v in losses.items()}`.
     - `model_loss = sum(scaled.values()) + kl_loss`.
3. `metrics = self._model_opt(torch.mean(model_loss), self.parameters())` — 단일 옵티마이저 step.
4. metric 정리 + `prior_ent`/`post_ent` 계산 (자체 autocast 컨텍스트 안).
5. **`context` 딕셔너리** ([L167-172](../../../dreamerv3-torch/models.py#L167-L172)): `{embed, feat, kl, postent}` — Plan2Explore가 ensemble 학습 target으로 사용.
6. **`post = {k: v.detach() for k, v in post.items()}`** ([L173](../../../dreamerv3-torch/models.py#L173)) — ImagBehavior에 출발점으로 넘기되 gradient 차단.
7. 반환 `(post, context, metrics)`.

### 1-3. KL balancing 디테일

KL 본체 동작은 [004 §2-7](004-dreamer_code_analysis_part2.md#2-7-kl_loss--kl-balancing-본체-networkspy272-290) 참조. WorldModel `_train` 시 적용 위치:

- `kl_free = config.kl_free = 1.0`, `dyn_scale = 0.5`, `rep_scale = 0.1` (configs.yaml:57-59).
- `kl_loss`는 `[B, T]` shape이고 head loss들과 동일 차원으로 sum → `mean()` → 스칼라.
- **kl_loss 자체가 `_scales` 딕셔너리에 들어가 있지 않음** — 1.0 가중. 모든 head loss와 같은 비율로 합산.
- metric으로 `kl_free`, `dyn_scale`, `rep_scale`, `dyn_loss`, `rep_loss`, `kl`(=kl_value) 모두 로깅 ([L154-159](../../../dreamerv3-torch/models.py#L154-L159)).

### 1-4. `preprocess(obs)` ([models.py:177-192](../../../dreamerv3-torch/models.py#L177-L192))

`_train`과 `_policy` 모두에서 호출.

흐름:
1. dict 전체를 `torch.tensor(device, dtype=float32)`로 변환.
2. `obs['image'] = obs['image'] / 255.0` — **image 키 무조건 접근**. obs space에 image가 없으면 KeyError.
3. `discount` 키가 obs에 있으면 `obs['discount'] *= config.discount` 후 마지막 차원에 unsqueeze(-1) — `[B, T] → [B, T, 1]`.
4. `assert 'is_first' in obs` — RSSM `obs_step`의 first-step 리셋 필수.
5. `assert 'is_terminal' in obs` — cont head 학습 라벨 필수.
6. `obs['cont'] = (1.0 - obs['is_terminal']).unsqueeze(-1)` — Bernoulli target. `[B, T] → [B, T, 1]`.

**image 키 접근 강제** ([L182](../../../dreamerv3-torch/models.py#L182))는 코드상 무조건 실행 — vector-only obs space는 KeyError. 의도된 비전 환경 가정.

### 1-5. `video_pred(data)` — 학습 시각화 ([models.py:194-215](../../../dreamerv3-torch/models.py#L194-L215))

`config.video_pred_log = True` (configs.yaml:20)일 때 `Dreamer.__call__`이 train log 주기 + eval 후 호출 ([dreamer.py:74-76, 316-318](../../../dreamerv3-torch/dreamer.py#L74-L76)).

동작:
1. 배치 첫 6개 sequence 사용, 처음 5 step은 관측 기반 post:
   - `states, _ = dynamics.observe(embed[:6, :5], action[:6, :5], is_first[:6, :5])`.
   - `recon = heads['decoder'](get_feat(states))['image'].mode()[:6]` — 5 step 재구성.
   - `reward_post = heads['reward'](get_feat(states)).mode()[:6]`.
2. 6번째 step부터 imagination:
   - `init = {k: v[:, -1] for k, v in states.items()}` — 5번째 step의 post를 출발점.
   - `prior = dynamics.imagine_with_action(action[:6, 5:], init)` — 나머지 action으로 prior rollout ([004 §2-5](004-dreamer_code_analysis_part2.md#2-5-observe--imagine_with_action-networkspy127-152)).
   - `openl = heads['decoder'](get_feat(prior))['image'].mode()` — open-loop 예측.
   - `reward_prior = heads['reward'](get_feat(prior)).mode()`.
3. **세로 패널 구성** ([L210-215](../../../dreamerv3-torch/models.py#L210-L215)):
   - `model = cat([recon[:, :5], openl], 1)` — 모델 출력 (앞 5 = recon, 뒤 = openl).
   - `truth = data['image'][:6]` — 실제 시퀀스.
   - `error = (model - truth + 1.0) / 2.0` — 차이 시각화 (0.5 중심).
   - `return cat([truth, model, error], 2)` — 세 행을 세로(H 축)로 쌓아 한 비디오.
4. logger가 `video()`로 TensorBoard에 기록.

**image 키 하드코딩**: `data['image']`, `dists['image']` 직접 접근 — obs space에 image 키가 없으면 KeyError. `video_pred_log=False`로 끄는 게 vector-only 환경의 우회.

### 1-6. 설계 포인트 (WorldModel)

1. **3 head 단일 옵티마이저** — encoder/dynamics/decoder/reward/cont의 모든 파라미터를 `_model_opt` 하나로 통합 학습. actor/value는 별도 옵티마이저(§3-1).
2. **단일 forward에서 모든 head loss 계산** — 효율적. 단 `image` 키 하드코딩 ([L182, L201, L207, L211](../../../dreamerv3-torch/models.py#L182)).
3. **`grad_heads` 메커니즘** — 어느 head로부터 representation까지 gradient를 흘릴지 config로 제어. 기본은 셋 다 ON.
4. **kl_loss는 scale 딕셔너리 외부** — 기본 1.0 가중. reward·cont만 명시적 scale.
5. **`post` detach 후 반환** — ImagBehavior가 출발점으로만 쓰고 RSSM에 grad 흘리지 않게.
6. **`context` 딕셔너리 반환** — Plan2Explore의 ensemble 학습 데이터로 사용 (006 §1).
7. **AMP 컨텍스트** — `use_amp = (precision == 16)`. 기본 precision=32이므로 AMP OFF. precision=16 시 head loss 계산이 fp16에서 underflow 가능 — 그래서 metric autocast는 별도 컨텍스트로 둠.
8. **video_pred는 학습 진단** — train/eval 시각화 전용. reward의 mode가 곁가지로 계산되지만 사용 안 됨 (코드에 있되 logger에 안 넘김).
9. **reward outscale=0.0 vs cont outscale=1.0** — reward는 V≈0 초기화와 같은 정신(symlog 공간), cont는 0.5 확률 시작(불확실).

---

## 2. RewardEMA — 5/95 quantile EMA ([models.py:11-26](../../../dreamerv3-torch/models.py#L11-L26))

target return 정규화용 EMA. 5% / 95% quantile을 매우 느린 EMA로 유지.

### 2-1. 동작

`__init__(device, alpha=1e-2)`:
- `self.range = torch.tensor([0.05, 0.95], device=device)` — quantile 좌표.
- `self.alpha = 1e-2` — EMA momentum. 매 호출마다 1%만 새 값 반영.

`__call__(x, ema_vals)`:
1. `flat_x = x.detach().flatten()` — target을 1D로.
2. `x_quantile = torch.quantile(flat_x, q=self.range)` — 현재 batch의 5%, 95% quantile (shape `[2]`).
3. `ema_vals[:] = alpha * x_quantile + (1 - alpha) * ema_vals` — **in-place** EMA 갱신.
4. `scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)` — 5~95 range = scale. **하한 1.0** — 좁은 분포에서 scale=0 폭발 방지.
5. `offset = ema_vals[0]` — 정규화 zero point.
6. 반환 `(offset.detach(), scale.detach())`.

### 2-2. 등록·사용 위치

- `ema_vals`는 `ImagBehavior.__init__`에서 buffer로 등록 ([models.py:285-287](../../../dreamerv3-torch/models.py#L285-L287)) — `torch.save`/`load_state_dict`에 자동 포함.
- 초기값 `torch.zeros((2,))`.
- 호출처는 **단 한 곳**: `_compute_actor_loss` ([models.py:404-408](../../../dreamerv3-torch/models.py#L404-L408)) — target advantage 정규화.
- value loss나 critic 학습에는 들어가지 않음 → critic은 raw target에 학습되어 EMA 자기참조 회피.

### 2-3. 설계 의미

- `alpha=1e-2`: 99% 과거 보존. half-life ≈ 70 step. 매우 보수적 — batch별 분포 진동에 흔들리지 않음.
- 5/95 quantile (1/99이 아니라): outlier에 약간 robust하면서도 분포 전체를 커버.
- `scale ≥ 1.0` 클리핑: scale이 1 미만이면 정규화가 advantage를 부풀려 학습 불안 — 1을 하한으로.
- 환경마다 다른 reward 스케일을 actor 학습에 자동 적응시키는 메커니즘. v3가 환경별 hyperparameter 튜닝 없이 도는 핵심.

---

## 3. ImagBehavior (`models.py`)

위치: [models.py:218-441](../../../dreamerv3-torch/models.py#L218-L441). actor + value + slow target + reward EMA. imagination rollout 기반 actor-critic.

### 3-0. 함수 호출 그래프

`ImagBehavior._train(start, objective)` ([L290](../../../dreamerv3-torch/models.py#L290))
- `_update_slow_target()` — polyak EMA.
- `_imagine(start, actor, H=15)` — imagination rollout.
  - `tools.static_scan(step, range(H), (start, None, None))`.
- `objective(feat, state, action)` — 외부 주입 람다 (보통 reward head).
- `actor(imag_feat).entropy()` — actor 엔트로피.
- `dynamics.get_dist(imag_state).entropy()` — state 엔트로피 (사용 안 함, metric 외).
- `_compute_target(imag_feat, imag_state, reward)`.
  - `cont_head(inp).mean` → discount.
  - `value(imag_feat).mode()`.
  - `tools.lambda_return(...)`.
    - `static_scan_for_lambda_return(...)`.
- `_compute_actor_loss(imag_feat, imag_action, target, weights, base)`.
  - `self.reward_ema(target, ema_vals)`.
- value loss 계산.
- `_actor_opt(actor_loss, ...)`, `_value_opt(value_loss, ...)`.

### 3-1. `__init__` — actor, value, slow_value, 2 옵티마이저 ([models.py:219-288](../../../dreamerv3-torch/models.py#L219-L288))

| 컴포넌트 | 분포 / 옵션 | 위치 |
|---|---|---|
| `actor = MLP(feat → num_actions, layers=2, dist=actor.dist, std='learned', absmax=1.0, ...)` | continuous: `dist='normal'` (`ContDist(Normal(tanh(mean), sigmoid_std))`). discrete: `dist='onehot'`. | [L228-244](../../../dreamerv3-torch/models.py#L228-L244) |
| `value = MLP(feat → 255 if symlog_disc, layers=2, outscale=0.0)` | `dist='symlog_disc'` → DiscDist 255-bin. outscale=0.0이라 초기 V≈0. | [L245-256](../../../dreamerv3-torch/models.py#L245-L256) |
| `_slow_value = copy.deepcopy(self.value)` | target net. polyak EMA로만 갱신. 옵티마이저 미등록 → gradient 안 받음. | [L257-259](../../../dreamerv3-torch/models.py#L257-L259) |
| `_updates = 0` | slow target 갱신 카운터. | [L259](../../../dreamerv3-torch/models.py#L259) |
| `_actor_opt`, `_value_opt` (둘 다 `tools.Optimizer`) | 분리 — WorldModel `_model_opt`까지 총 3개. | [L261-282](../../../dreamerv3-torch/models.py#L261-L282) |
| `ema_vals` buffer (`torch.zeros((2,))`) + `reward_ema = RewardEMA(device)` | reward_EMA=True일 때만. checkpoint 포함. | [L283-288](../../../dreamerv3-torch/models.py#L283-L288) |

actor hyperparameter (configs.yaml:49-50):
- `lr=3e-5`, `eps=1e-5`, `grad_clip=100`.
- `entropy=3e-4` (loss 가중치).
- `dist='normal'` (continuous 환경 기본). 다른 분포는 [004 §1-9](004-dreamer_code_analysis_part2.md#1-9-head--분포-매핑-표) 참조.
- `std='learned'`, `min_std=0.1`, `max_std=1.0`.
- `unimix_ratio=0.01` (onehot일 때).

critic hyperparameter (configs.yaml:51-52):
- `lr=3e-5`, `eps=1e-5`, `grad_clip=100`.
- `dist='symlog_disc'` (DiscDist 255-bin).
- `slow_target=True`, `slow_target_update=1`, `slow_target_fraction=0.02`.
- `outscale=0.0`.

`kw = dict(wd=weight_decay=0, opt='adam', use_amp=...)` ([L260](../../../dreamerv3-torch/models.py#L260)) — actor/value 옵티마이저 공통 인자.

### 3-2. `_train(start, objective)` — 전체 흐름 ([models.py:290-349](../../../dreamerv3-torch/models.py#L290-L349))

흐름:
1. `_update_slow_target()` — §3-3.
2. **actor lane** (`RequiresGrad(self.actor)` + AMP autocast):
   - `imag_feat, imag_state, imag_action = _imagine(start, self.actor, imag_horizon=15)` — §3-4.
   - `reward = objective(imag_feat, imag_state, imag_action)` — §3-3 외부 람다.
   - `actor_ent = self.actor(imag_feat).entropy()` — entropy 계산.
   - `state_ent = world_model.dynamics.get_dist(imag_state).entropy()` — 계산하나 사용 안 함 (metric에도 안 들어감).
   - `target, weights, base = _compute_target(imag_feat, imag_state, reward)` — §3-5.
   - `actor_loss, mets = _compute_actor_loss(imag_feat, imag_action, target, weights, base)` — §3-7.
   - `actor_loss -= entropy * actor_ent[:-1, ..., None]` ([L317](../../../dreamerv3-torch/models.py#L317)) — entropy bonus 차감. `[:-1]`로 마지막 bootstrap step 제외.
   - `actor_loss = torch.mean(actor_loss)` — 스칼라로.
3. **value lane** (`RequiresGrad(self.value)` + AMP):
   - `value = self.value(value_input[:-1].detach())` — `value_input = imag_feat`. detach로 actor lane gradient 차단. `[:-1]`로 length T−1=14.
   - `target = torch.stack(target, dim=1)` — tuple → tensor. **shape `[T-1, N, 1]` (time-major)** — 005 이전 판에 적힌 `[N, T-1, 1]`은 오류였음. 자세히 §3-6.
   - `value_loss = -value.log_prob(target.detach())` — λ-return target NLL.
   - `if slow_target: slow_target = self._slow_value(value_input[:-1].detach()); value_loss -= value.log_prob(slow_target.mode().detach())` — slow self-distill 항 추가.
   - `value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])` — weights로 가중 후 평균.
4. metric 누적 (value mode, target stats, imag_reward, imag_action 등).
5. `RequiresGrad(self)` 컨텍스트에서:
   - `_actor_opt(actor_loss, self.actor.parameters())`.
   - `_value_opt(value_loss, self.value.parameters())`.
6. 반환 `(imag_feat, imag_state, imag_action, weights, metrics)`.

`RequiresGrad`는 `__enter__`에서 `requires_grad_(True)`, `__exit__`에서 `False` ([tools.py:31-39](../../../dreamerv3-torch/tools.py#L31-L39)) — actor·value lane이 독립적으로 grad on/off되어 forward 그래프가 다른 lane에 누출되지 않음.

### 3-3. `_update_slow_target` — polyak EMA ([models.py:435-441](../../../dreamerv3-torch/models.py#L435-L441))

동작:
- `if slow_target:` 가드.
- `if self._updates % slow_target_update == 0:` — `slow_target_update=1` (기본) → 매 `_train` 호출마다 갱신.
- `mix = slow_target_fraction = 0.02` — τ.
- `for s, d in zip(self.value.parameters(), self._slow_value.parameters()): d.data = mix*s.data + (1-mix)*d.data` — in-place polyak.
- `self._updates += 1`.

특징:
- `_slow_value`는 `copy.deepcopy(self.value)`이라 같은 구조. requires_grad는 그대로 True지만 옵티마이저 미등록 → gradient는 받지 않고 EMA로만 갱신.
- 호출 시점: `_train` 진입 직후, actor/value 업데이트 **이전**.
- `mix=0.02`, half-life ≈ 34 step. 매우 부드러움.
- 큰 주기 + hard copy 대신 매 step polyak — Dreamer-v3 안정성 기법.

### 3-3. `objective`는 외부 주입 람다

`_train(self, start, objective)` 시그니처. reward를 만드는 람다는 호출처에서 정의.

**task behavior 호출** ([dreamer.py:122-124](../../../dreamerv3-torch/dreamer.py#L122-L124)):
- `reward = lambda f, s, a: self._wm.heads["reward"](self._wm.dynamics.get_feat(s)).mode()`
- 입력 `s`(state)에서 `get_feat`로 feat 계산 후 reward head 통과.
- `.mode()` — `DiscDist.mode() = symexp(Σ p·b)` ([004 §1-2](004-dreamer_code_analysis_part2.md#1-2-discdist--twohot-인코딩-categorical-toolspy452-506)). **원래 스케일의 reward**가 lambda_return에 들어감 (symlog 공간 X).

**Plan2Explore 생성용 reward (다른 람다)** ([dreamer.py:51](../../../dreamerv3-torch/dreamer.py#L51)):
- `reward = lambda f, s, a: self._wm.heads["reward"](f).mean()`
- 입력 `f`(feat) 그대로 head 통과 — `get_feat` 단계 생략.
- `.mean()` — `DiscDist.mean()`은 `mode()`와 본문 동일 ([004 §1-2](004-dreamer_code_analysis_part2.md#1-2-discdist--twohot-인코딩-categorical-toolspy452-506))이라 결과 값은 같음.
- 이 람다는 `Plan2Explore(config, wm, reward)`로 전달되어 `_intrinsic_reward`의 extrinsic 보너스 항(`expl_extr_scale > 0`인 경우)에서 사용 (006 §1).

**Plan2Explore의 task behavior 학습 람다** ([exploration.py:104](../../../dreamerv3-torch/exploration.py#L104)): `self._behavior._train(start, self._intrinsic_reward)` — disagreement 보너스를 objective로 주입. ImagBehavior 클래스 재사용으로 explore policy를 같은 인프라로 학습.

### 3-4. `_imagine` — imagination rollout ([models.py:351-369](../../../dreamerv3-torch/models.py#L351-L369))

흐름:
1. `flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))` — `(B, T, ...) → (B·T, ...)`.
2. `start = {k: flatten(v) for k, v in start.items()}` — post를 펼침. WorldModel `_train`이 만든 post는 `[B=16, T=64, ...]`이라 batch dim이 `16·64 = 1024`.
3. **step 함수**:
   - `state, _, _ = prev` (이전 step의 (succ_state, feat, action)).
   - `feat = dynamics.get_feat(state)` — 현재 state의 feat.
   - `inp = feat.detach()` — **actor 입력은 detach** (actor가 RSSM에 직접 backprop 못 함).
   - `action = policy(inp).sample()` — actor 출력. STE 적용된 sample.
   - `succ = dynamics.img_step(state, action)` — **state는 detach 안 함**. dynamics gradient는 다음 step의 feat까지 살아 있음.
   - 반환 `(succ, feat, action)`.
4. `succ, feats, actions = tools.static_scan(step, [arange(horizon)], (start, None, None))` — horizon=15.
5. `states = {k: cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}` — 첫 자리에 start post를 끼우고 prior의 마지막 step은 제외 → `states[0]=start`, `states[t]=succ[t-1]`.
6. 반환 `(feats, states, actions)`. shape: `feats=[H=15, N=1024, F]`, `actions=[H, N, A]`, `states[k]=[H, N, ...]`.

**비대칭 detach의 의미**:
- `inp = feat.detach()` — actor가 dynamics를 통한 backprop 못 함. actor parameter 학습은 path-derivative(target → feat → ... → action)로만.
- `succ = img_step(state, action)`의 `state`는 detach 안 됨 — dynamics 자체는 grad 받음.
- 결과: `imag_gradient='dynamics'` 모드에서 actor 출력 → RSSM(미분가능) → 다음 feat → value/reward → target → loss → backprop이 전체 imagination 통과 가능.

### 3-5. `_compute_target` — λ-return + weights ([models.py:371-389](../../../dreamerv3-torch/models.py#L371-L389))

흐름:
1. `inp = world_model.dynamics.get_feat(imag_state)` — 모든 state의 feat (start 포함).
2. `discount = config.discount * world_model.heads['cont'](inp).mean` — `γ · P(continue)`. γ=0.997 (configs.yaml:76). cont head의 Bernoulli `.mean = sigmoid(logits)` = 확률.
3. `value = self.value(imag_feat).mode()` — shape `[H, N, 1]`. **slow target이 아니라 현재 value의 mode** — 부트스트랩은 현재 value, 학습 target은 별도 계산.
4. `target = tools.lambda_return(reward[1:], value[:-1], discount[1:], bootstrap=value[-1], lambda_=0.95, axis=0)` — §3-6.
5. `weights = cumprod(cat([ones_like(d[:1]), d[:-1]], 0), 0).detach()` — **선두 1을 끼우고** cumprod. `weights[0]=1`, `weights[t] = ∏_{i<t} discount_i`. shape `[H, N, 1]`.
6. 반환 `(target, weights, base=value[:-1])`.

**선두 1의 의미**: actor loss는 `weights[:-1] * actor_target`. weights[0]=1이라 첫 step은 discount 없이 full 가중치. 만약 선두 1이 없었다면 cumprod(discount)로 weights[0]=discount[0]이 되어 첫 step부터 discount가 곱해져 학습 신호가 약해짐.

**reward·value·discount 인덱스 슬라이싱** (lambda_return 입력 정렬, axis=0):
- `reward[1:]` — t=1..H-1. r_{t+1}로 사용.
- `value[:-1]` — t=0..H-2. V_t로 사용.
- `discount[1:]` — t=1..H-1. γ_{t+1}로 사용.
- `bootstrap = value[-1]` — V_H로 사용.
- 길이 모두 H-1 = 14.

shape 가이드 (H=15, N=B·T=1024):
- `imag_feat`: `[15, 1024, 1536]` (discrete RSSM)
- `imag_state[k]`: `[15, 1024, ...]`
- `reward`, `value`, `discount`: `[15, 1024, 1]`
- `weights`: `[15, 1024, 1]`
- `base = value[:-1]`: `[14, 1024, 1]`

### 3-6. `tools.lambda_return` + `static_scan_for_lambda_return` ([tools.py:671-717](../../../dreamerv3-torch/tools.py#L671-L717))

**lambda_return** ([tools.py:691-717](../../../dreamerv3-torch/tools.py#L691-L717)):
- 입력 dim 검증, axis가 0이 아니면 permute (호출처에선 axis=0이라 permute 안 일어남).
- `next_values = cat([value[1:], bootstrap[None]], 0)` — V_{t+1} 시퀀스. 마지막은 bootstrap.
- `inputs = reward + pcont * next_values * (1 - lambda_)` — λ-return의 "현재 step 부분" `r_{t+1} + γ_{t+1}(1-λ) V_{t+1}`.
- `returns = static_scan_for_lambda_return(lambda agg, cur0, cur1: cur0 + cur1*lambda_*agg, (inputs, pcont), bootstrap)` — 재귀 G_t = inputs[t] + pcont[t]·λ·G_{t+1}.
- axis 복원 후 반환.

재귀식 정리:
- `G_T = bootstrap = V_H`
- `G_t = inputs[t] + pcont[t]·λ·G_{t+1}`
- `inputs[t] = r_{t+1} + γ_{t+1}(1-λ) V_{t+1}`
- `pcont[t] = γ_{t+1}`
- 풀어쓰면: **G_t = r_{t+1} + γ_{t+1}[(1-λ) V_{t+1} + λ G_{t+1}]**.
- `λ=1`: discounted Monte Carlo. `λ=0`: 1-step TD. 기본 0.95.

**static_scan_for_lambda_return** ([tools.py:671-688](../../../dreamerv3-torch/tools.py#L671-L688)):
- `indices = reversed(range(T-1))` — 시간 역방향 스캔 (재귀가 미래→현재).
- 매 step `last = fn(last, inputs[index], pcont[index])` — `last` shape `[N, 1]` 유지.
- 누적: `outputs = torch.cat([outputs, last], dim=-1)` — 마지막 차원에 붙임. T-1 iteration 후 shape `[N, T-1]`.
- `outputs = reshape(outputs, [N, T-1, 1])` → `flip(dim=[1])` (시간 순서 복원) → `unbind(dim=0)`.

**중요한 shape 정정**: `unbind(dim=0)`는 dim 0(=N) 기준 분해 → **tuple of N 텐서, 각 shape `[T-1, 1]`**. (이전 005 판에 적힌 "tuple of T-1, 각 [N, 1]"은 오류였음.)

**후속 stack** ([models.py:325, 403](../../../dreamerv3-torch/models.py#L325)):
- `torch.stack(target, dim=1)` — N개의 `[T-1, 1]` 텐서를 dim=1에 쌓음 → 결과 shape **`[T-1, N, 1]`** (time-major).
- 이후 actor_loss/value_loss 계산이 time-major 기준 정합 — base=value[:-1] [T-1, N, 1], weights[:-1] [T-1, N, 1] 모두 time-major이라 broadcast 일관.

### 3-7. `_compute_actor_loss` — 3 모드 ([models.py:391-433](../../../dreamerv3-torch/models.py#L391-L433))

흐름:
1. `inp = imag_feat.detach()` — actor 입력 detach.
2. `policy = self.actor(inp)` — 분포 재구성.
3. `target = torch.stack(target, dim=1)` — `[T-1, N, 1]` (§3-6).
4. **RewardEMA 적용** (configs.yaml:30 기본 `reward_EMA=True`) ([L404-408](../../../dreamerv3-torch/models.py#L404-L408)):
   - `offset, scale = self.reward_ema(target, self.ema_vals)` — §2.
   - `normed_target = (target - offset) / scale`.
   - `normed_base = (base - offset) / scale`. (base = value[:-1] [T-1, N, 1].)
   - `adv = normed_target - normed_base` — advantage on normalized scale.
   - reward_EMA OFF: `adv = target - base` (raw).
5. **3 gradient 모드** ([L415-431](../../../dreamerv3-torch/models.py#L415-L431)):
   - `'dynamics'` (continuous 기본): `actor_target = adv` — path-derivative. RSSM이 미분가능하므로 target에서 actor parameter까지 grad 흐름.
   - `'reinforce'` (discrete 기본 — atari/crafter/minecraft/memorymaze config가 override): `actor_target = log π(a|s)[:-1][:, :, None] * (target - value(imag_feat[:-1]).mode()).detach()` — 표준 policy gradient. baseline은 현재 value 호출 (slow가 아님).
   - `'both'`: `mix = imag_gradient_mix=0.0` (기본). `actor_target = mix*target + (1-mix)*reinforce_term`. 기본 mix=0이라 사실상 reinforce.
6. `actor_loss = -weights[:-1] * actor_target` ([L432](../../../dreamerv3-torch/models.py#L432)) — `weights[:-1]` shape `[T-1, N, 1]`, actor_target shape `[T-1, N, 1]`. **actor_target에는 별도 [:-1] 슬라이싱 없음** (이전 005에 "actor_target[:-1]"이라 적은 부분은 오류).
7. 반환 `(actor_loss, metrics)`.

`_train`에서 후처리 ([L317](../../../dreamerv3-torch/models.py#L317)):
- `actor_loss -= entropy * actor_ent[:-1, ..., None]` — entropy bonus 차감. actor_ent shape `[H, N]`이라 `[:-1]`로 길이 맞춤 후 `[..., None]`로 마지막 차원 추가.
- 그 후 `mean()`로 스칼라화.

### 3-8. Value loss — λ-return + slow self-distill ([models.py:322-332](../../../dreamerv3-torch/models.py#L322-L332))

흐름:
1. `value = self.value(value_input[:-1].detach())` — DiscDist. `value_input = imag_feat`, `[:-1]` → length T-1. `.detach()` — actor lane gradient 차단.
2. `target = torch.stack(target, dim=1)` — `[T-1, N, 1]`. (actor lane에서 이미 stack했지만 value lane도 다시 stack — 둘은 다른 컨텍스트.)
3. `value_loss = -value.log_prob(target.detach())` — DiscDist NLL. shape `[T-1, N]` (DiscDist.log_prob이 마지막 차원 sum).
4. **slow self-distill** (slow_target=True 기본):
   - `slow_target = self._slow_value(value_input[:-1].detach())` — DiscDist.
   - `value_loss -= value.log_prob(slow_target.mode().detach())` — slow의 mode를 추가 target으로.
5. `value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])` — `weights[:-1]` shape `[T-1, N, 1]`. `value_loss[:, :, None]`로 마지막 차원 추가 → broadcast.

**두 항의 가중치는 동등**. slow 항을 별도로 가중하는 hyperparameter 없음. λ-return target과 slow target.mode가 단순 합산 cross-entropy.

`target.detach()` 명시: λ-return은 actor lane에서 dynamics gradient를 받지만 value loss에서는 차단 — value 학습 신호가 dynamics로 흘러가면 representation이 value에 끌려가 학습 불안.

`value_input[:-1].detach()`: value head 학습이 actor·world model로 누출 안 되게.

### 3-9. 설계 포인트 (ImagBehavior)

1. **`feat.detach()`는 actor 입력에만**: imagination rollout 자체의 `img_step(state, action)`은 state를 detach하지 않아 dynamics gradient가 살아 있다. 이 비대칭성이 `imag_gradient='dynamics'` 모드의 핵심.
2. **discount = γ · cont_head.mean**: hard done이 아닌 soft termination. weights도 cont를 통해 자동 0 수렴. (단, cont head는 train data로 학습되어 imagination 분포에서 정확성은 모델 일반화에 의존.)
3. **RewardEMA는 actor만**: critic은 raw target에 학습 → critic 출력이 EMA에 의존하지 않아 부트스트랩 자기참조 회피.
4. **Value loss self-distill (slow_target.mode 항)**: target net을 부트스트랩으로만 쓰는 게 아니라 별도 항으로 직접 distill → λ-return target이 잡음일 때도 slow가 anchor 역할.
5. **`slow_target_update=1` + `fraction=0.02`**: 매 step polyak. 큰 주기 + hard copy 대신 부드러운 EMA로 안정성.
6. **3가지 imag_gradient 모드**: continuous는 `dynamics`(path), discrete는 `reinforce`(REINFORCE), `both`는 두 모드 결합. v3는 환경별로 actor.dist에 따라 config로 선택 (atari/crafter 등 onehot 환경은 `imag_gradient='reinforce'` override).
7. **`retain_graph=True`** (Optimizer 본체는 §4): actor와 value가 같은 imagination graph를 공유 — 두 번 backward 가능해야 함.
8. **lambda_return이 tuple 반환**: `unbind(dim=0)`로 N개 tensor (each `[T-1, 1]`) → 호출처에서 `stack(dim=1)` → `[T-1, N, 1]` time-major.
9. **`objective` 람다 주입 설계**: reward 함수를 외부에서 갈아끼울 수 있어 task behavior(=실제 reward)와 explore behavior(=disagreement 보너스, 006 §1)를 같은 `ImagBehavior` 클래스로 학습 가능.
10. **`reward_EMA` toggle**: configs.yaml:30 기본 True. False 시 `adv = target - base` raw 정규화.
11. **state_ent 계산되나 미사용** ([L305](../../../dreamerv3-torch/models.py#L305)): 코드에 있으나 actor_loss/value_loss/metric 어디에도 안 들어감. JAX 원본의 잔재 추정.

---

## 4. Optimizer 래퍼 (`tools.py`)

위치: [tools.py:720-772](../../../dreamerv3-torch/tools.py#L720-L772). WorldModel·actor·value 세 옵티마이저가 모두 이 클래스 인스턴스.

### 4-1. `__init__` ([L720-747](../../../dreamerv3-torch/tools.py#L720-L747))

인자:
- `name`, `parameters`, `lr`, `eps=1e-4`, `clip=None`, `wd=None`, `wd_pattern=r".*"`, `opt='adam'`, `use_amp=False`.

내부:
- `self._opt`: opt 키에 따라 `Adam`/`Adamax`/`SGD`/`SGD(momentum=0.9)` 생성. `'nadam'`은 NotImplemented.
- `self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)` — AMP scaler.
- `assert 0 <= wd < 1`, `assert not clip or 1 <= clip` — 입력 검증.

### 4-2. `__call__(loss, params, retain_graph=True)` ([L749-765](../../../dreamerv3-torch/tools.py#L749-L765))

흐름 (AMP + grad clip 정석 패턴):
1. `assert len(loss.shape) == 0` — 스칼라 강제.
2. `self._opt.zero_grad()`.
3. `self._scaler.scale(loss).backward(retain_graph=retain_graph)` — AMP scale 후 backward.
4. `self._scaler.unscale_(self._opt)` — unscale 후 clip 가능 상태로.
5. `norm = clip_grad_norm_(params, self._clip)` — global norm clip.
6. `if self._wd: self._apply_weight_decay(params)` — decoupled weight decay.
7. `self._scaler.step(self._opt)` — scaler 통해 step (`inf`/`nan` 자동 스킵).
8. `self._scaler.update()` — scale factor 적응.
9. `self._opt.zero_grad()` (중복이지만 안전).
10. 반환 `{name_loss, name_grad_norm}` metric.

### 4-3. Decoupled weight decay ([L767-772](../../../dreamerv3-torch/tools.py#L767-L772))

- `wd_pattern != r".*"`이면 NotImplemented — 모든 파라미터에 동일 wd만 지원.
- `for var in varibs: var.data = (1 - self._wd) * var.data` — AdamW와 동등한 weight decay를 직접 구현.
- configs.yaml:60 `weight_decay=0`이라 기본 비활성.

### 4-4. `retain_graph=True`의 의미

`ImagBehavior._train`에서 actor_loss → backward 후 value_loss → backward를 연달아 호출. 두 loss가 같은 imagination graph(`_imagine` 출력)를 공유 — 첫 backward가 graph를 해제하면 두 번째가 실패. `retain_graph=True`로 첫 backward 후 graph 보존.

WorldModel의 `_model_opt`은 단독 backward이므로 `retain_graph`가 의미 없으나 default가 True라 그대로 사용.

### 4-5. 설계 포인트 (Optimizer)

1. **AMP 정석 패턴**: scale → backward → unscale → clip → step → update. PyTorch 공식 권장 순서.
2. **gradient inf/nan 자동 스킵**: `scaler.step`이 unscale된 grad에 inf/nan 있으면 step skip + scale 감소. RL의 비정상 loss에 강함.
3. **decoupled WD 직접 구현**: AdamW를 안 쓰고 직접 `var *= (1-wd)`. JAX 원본과 동등 동작 보존을 위한 선택으로 추정.
4. **`retain_graph=True`** 디폴트: ImagBehavior가 actor·value 두 lane backward 필요 — 안전한 디폴트.
5. **`assert` 입력 검증**: wd ∈ [0, 1), clip ≥ 1 — hyperparameter 실수 차단.

---

## 5. 손실 흐름 한눈에

```
                                  ─── WorldModel._train ───
obs(dict) ──┬─ excluded 제거 ─┬─ cnn keys → ConvEncoder ──┐
            │                  └─ mlp keys → MLP(symlog) ──┴─→ embed [B, T, E]
            └─ action ─────────────────────────────────────────────────┐
                                                                       │
embed + action + is_first ─→ RSSM.observe ─→ post, prior ──────────────┤
                                                  │                    │
                                                  ├─→ decoder(get_feat(post))  → image_dist (MSE) / vector_dist (SymlogDist)
                                                  ├─→ reward(get_feat(post))   → DiscDist (twohot symlog 255-bin)
                                                  ├─→ cont(get_feat(post))     → Bernoulli (binary)
                                                  └─→ KL(post||prior) ─→ dyn_loss·0.5 + rep_loss·0.1, kl_free=1.0
                                                  ─── sum + kl ──→ _model_opt (Adam + AMP + clip=1000)

post(detach) ─→ ImagBehavior._imagine(actor, H=15) ─→ (imag_feat, imag_state, imag_action) [H, N=B·T, ...]
                                                       │
                                                       ├─→ objective(feat, state, action) ──→ reward [H, N, 1]   (heads['reward'].mode())
                                                       ├─→ heads['cont'](feat).mean ─────────→ continue prob → discount = γ · cont
                                                       ├─→ value(imag_feat).mode() ──────────→ V [H, N, 1]
                                                       └─→ tools.lambda_return(r[1:], V[:-1], γ[1:], bootstrap=V[-1], λ=0.95)
                                                              └─→ tuple of N, each [T-1, 1] ─→ stack(dim=1) → target [T-1, N, 1]

  RewardEMA(target) ─→ (offset, scale)
  normed_target, normed_base
  adv = normed_target - normed_base
  actor_target = adv (dynamics) / log π · adv.detach() (reinforce) / mix (both)
  actor_loss = -weights[:-1] · actor_target - entropy · actor_ent[:-1]
            ─────────────────────────────────────────────→ _actor_opt (Adam, lr=3e-5, clip=100)

  value(imag_feat[:-1].detach()) ──→ value_loss = -log_prob(target.detach())
                                                  - log_prob(slow_value(imag_feat[:-1]).mode())   (slow self-distill)
  value_loss = mean(weights[:-1] · value_loss)
            ─────────────────────────────────────────────→ _value_opt (Adam, lr=3e-5, clip=100)
                                                                                  ↑
                                                              _slow_value (polyak τ=0.02, 매 step)
```

---

## 6. 설계 포인트 (통합)

1. **3개 옵티마이저** (model / actor / value) — 그래디언트 흐름 격리. 각 lane의 backward가 서로 영향 없도록 detach 위치를 신중히 배치.
2. **단일 forward에서 모든 head 손실 계산** — 효율적. `image` 키 하드코딩이 비전 환경 가정.
3. **KL balancing** (dyn 0.5 / rep 0.1) — Dreamer-v3가 v2 대비 가장 크게 바뀐 곳 중 하나.
4. **continue head로 discount 변조** — 종료 가까운 state 자동 절하.
5. **symlog_disc로 reward/value 표현** — 분포가 넓은 reward 안정 학습 ([004 §1-2](004-dreamer_code_analysis_part2.md#1-2-discdist--twohot-인코딩-categorical-toolspy452-506)).
6. **reward EMA + 5/95 quantile** — 환경별 reward scaling 자동화. actor lane만.
7. **slow value self-distill** — value loss에 두 항 더하기 트릭.
8. **path-derivative(dynamics) vs REINFORCE** 분기 — continuous는 path-grad가 분산 낮음, discrete는 REINFORCE.
9. **AMP + grad clip + decoupled WD** — Optimizer 래퍼가 정석 패턴 강제.
10. **`retain_graph=True`** — actor·value 두 lane backward 가능.

---

## 7. 다음 단계 안내

7~9단계: **Exploration + 데이터 + 포팅 + 평가** — [006](006-dreamer_code_analysis_part4.md).

확인 항목:
- `Random`/`Plan2Explore` — ensemble disagreement 보너스, ImagBehavior 재사용 구조, `_intrinsic_reward` 자기강화 위험.
- 데이터 파이프라인 — `load_episodes`, `sample_episodes` (시드 결정성, is_first 강제), `from_generator`, `simulate`, `add_to_cache`, `save_episodes`(BytesIO), `erase_over_episodes`/`dataset_size`, `Every`/`Once`/`Until` 카운터.
- JAX→PyTorch 포팅 — `static_scan`, `Conv2dSamePad`, `ImgChLayerNorm`, `weight_init`/`uniform_weight_init`, `GRUCell` 3-gate 합성.
- 평가·체크포인트 — eval rollout, `video_pred` (본 문서 §1-5에서 다룸, 평가용 호출 흐름은 006), `latest.pt` 단일 파일 정책, resume 시드 결정성, `recursively_collect/load_optim_state_dict`.

---

## 8. 컨벤션

- 한국어 응답.
- 코드 인용은 줄 링크만. 코드 블록 발췌 금지.
- 분석 대상은 **dreamer-v3 코드 자체**. F1TENTH 통합 코멘트 금지.
- 본 감사 작업 한정으로 직접 수정 허용 (1회 예외).
