# 003 - Dreamer-v3 코드 분석 (Part 1: 진입점 + WorldModel/ImagBehavior)

> **목적**: F1TENTH RL 프로젝트에서 사용할 `NM512/dreamerv3-torch`의 코드 자체를 이해한다. (이 단계는 F1TENTH 통합 설계가 아니라 **dreamer-v3 알고리즘 코드 분석**이 목적이다.)
> **선행 문서**: [001-env-analysis.md](001-env-analysis.md) (F1Tenth Gym 분석), [002-Select_Implementation.md](002-Select_Implementation.md) (구현체 선택 — NM512/dreamerv3-torch)
> **분석 대상 경로**: `/home/dlacksdn/dreamerv3-torch` (이미 clone 완료)
> **작성일**: 2026-05-20

---

## 0. 다음 세션 핸드오프 (Read first)

### 현재까지 진행 상태
- ✅ **1단계**: 진입점 + Config 구조 (`dreamer.py`, `configs.yaml`)
- ✅ **2단계**: WorldModel + ImagBehavior (`models.py`)
- ⏭️ **3단계 (NEXT)**: RSSM 동역학 (`networks.py`의 `RSSM`) — `obs_step`/`img_step`/`kl_loss`/`get_feat`/`observe`

### 전체 분석 로드맵 (코드 자체 분석, 환경 wrapper는 제외)
1. 진입점·실행 흐름 (`dreamer.py`) ✅
2. WorldModel (`models.py`) ✅
3. **RSSM 동역학 (`networks.py`의 `RSSM`)** ← 다음
4. Encoder/Decoder (`networks.py`의 `MultiEncoder`/`MultiDecoder`)
5. ImagBehavior 세부 (이미 2단계에서 일부 다룸, λ-return/EMA 등)
6. Heads & 분포 (`networks.py`의 `MLP`, `tools.py`의 `SymlogDist`/`DiscDist`/`OneHotDist`/twohot)
7. Exploration (`exploration.py`) — `Plan2Explore`, `Random`
8. 데이터 파이프라인 (`tools.py`의 `load_episodes`/`sample_episodes`/`from_generator`/`simulate`)
9. 유틸: optimizer 래퍼, EMA, slow target, gradient clipping
- **분석 대상 제외**: `envs/` (환경 어댑터), `parallel.py`(병렬 실행), `Dockerfile`, `requirements.txt`

### 파일 위치·라인 수 (분석 시 참조)
```
/home/dlacksdn/dreamerv3-torch/
├── dreamer.py         365 lines  (✅ 분석)
├── models.py          441 lines  (✅ 분석)
├── networks.py        810 lines  (⏭️ Part 2)
├── tools.py          1000 lines
├── exploration.py     135 lines
├── parallel.py        209 lines  (제외)
├── configs.yaml       184 lines  (✅ 분석)
└── envs/                         (제외)
```

### 사용자 작업 컨벤션 (CLAUDE.md 요약)
- `_thinking/analysis/`는 append-only. 기존 파일 수정 금지.
- 명시적 요청("003 문서로 저장")이 있을 때만 저장.
- 한글로 응답.
- 코드 인용은 markdown link `[파일:라인](상대경로#L라인)` 형식. 이 문서에서는 `../../../dreamerv3-torch/`로 상대 참조.

---

## 1. 진입점·실행 흐름 (`dreamer.py`)

### 1-1. main 흐름 ([dreamer.py:206-339](../../../dreamerv3-torch/dreamer.py#L206-L339))

1. logdir/traindir/evaldir 생성 → `load_episodes`로 기존 에피소드 로드
2. `make_env` × `config.envs`개 생성 → `Parallel(process)` 또는 `Damy`(인라인) 래핑
3. **Prefill**: 랜덤 정책으로 `config.prefill=2500` 스텝 수집 ([L252-281](../../../dreamerv3-torch/dreamer.py#L252-L281))
4. `Dreamer` 에이전트 생성 → `latest.pt` 있으면 resume
5. 메인 루프: **eval → train(`eval_every` 스텝) → 체크포인트 저장** 반복 ([L302-334](../../../dreamerv3-torch/dreamer.py#L302-L334))

### 1-2. Dreamer 클래스 ([L28-133](../../../dreamerv3-torch/dreamer.py#L28-L133))

핵심 3 컴포넌트 ([L44-56](../../../dreamerv3-torch/dreamer.py#L44-L56)):
- `self._wm = WorldModel(obs_space, act_space, step, config)`
- `self._task_behavior = ImagBehavior(config, wm)`
- `self._expl_behavior` = greedy / random / plan2explore (config.expl_behavior)

**훈련 트리거**: `_should_train = Every(batch_size*batch_length / train_ratio)` ([L35](../../../dreamerv3-torch/dreamer.py#L35))
- 기본 `(16*64)/512 = 2` → env step 2회마다 train step 1회

**`__call__(obs, reset, state, training)`**: env step에서 호출 → 필요시 train_step 누적 횟수만큼 호출 → policy 추론 ([L58-84](../../../dreamerv3-torch/dreamer.py#L58-L84))

**`_train(data)`** ([L117-133](../../../dreamerv3-torch/dreamer.py#L117-L133)): WorldModel → ImagBehavior → (옵션) ExplorationBehavior 순.

### 1-3. make_env ([L146-203](../../../dreamerv3-torch/dreamer.py#L146-L203))

task 이름을 `_`로 split → suite 분기 (`dmc/atari/dmlab/memorymaze/crafter/minecraft`).

공통 wrapper 체인:
- `TimeLimit(time_limit)` — 강제 종료
- `SelectAction(key="action")` — dict action에서 키 추출
- `UUID()` — 에피소드 식별자

### 1-4. Config 상속 ([dreamer.py:350-360](../../../dreamerv3-torch/dreamer.py#L350-L360))

- `defaults` 블록을 베이스로 깔고 `--configs A B C` 순서대로 **재귀 머지**
- nested dict는 키 단위로 덮어쓰기 → encoder/decoder/actor 등 부분 override 가능

### 1-5. configs.yaml 주요 디폴트

| 항목 | 값 | 의미 |
|---|---|---|
| `steps` | 1e6 | 총 환경 step (action_repeat 적용 전) |
| `action_repeat` | 2 | env step 1회 = 시뮬 2회 |
| `prefill` | 2500 | 랜덤 정책 수집 |
| `time_limit` | 1000 | 에피소드 최대 길이 |
| `batch_size × batch_length` | 16 × 64 | 학습 시퀀스 |
| `train_ratio` | 512 | env step당 train step 비율 |
| `dyn_stoch/dyn_discrete/dyn_deter` | 32/32/512 | RSSM 차원 (discrete 모드: stoch=32 카테고리, 각 32 클래스) |
| `feat_size` | stoch*discrete + deter = 1536 | actor/value/heads 입력 차원 |
| `encoder.symlog_inputs` | True | 입력에 symlog 적용 |
| `actor.dist` | `'normal'` | continuous |
| `critic.dist` / `reward_head.dist` | `'symlog_disc'` | twohot 255 bins |
| `discount` | 0.997 | γ |
| `discount_lambda` | 0.95 | λ-return |
| `imag_horizon` | 15 | imagination rollout 길이 |
| `imag_gradient` | `'dynamics'` | path-derivative (continuous), discrete는 `'reinforce'` |
| `grad_heads` | `['decoder','reward','cont']` | 어느 head에서 representation으로 grad 흘릴지 |
| `kl_free` / `dyn_scale` / `rep_scale` | 1.0 / 0.5 / 0.1 | KL balancing |

### 1-6. 추가 관찰
- **체크포인트**: `latest.pt` 단일 파일만 저장/덮어쓰기 — 중간 스냅샷 원하면 코드 수정 필요
- **`compile=True`** 기본: torch.compile 켜져 있음. 디버깅 시 끄는 게 편함
- **`make_dataset`** ([L140-143](../../../dreamerv3-torch/dreamer.py#L140-L143)): `sample_episodes` 제너레이터 → `from_generator` 배치 묶음

---

## 2. WorldModel & ImagBehavior (`models.py`)

### 2-1. WorldModel ([models.py:29-215](../../../dreamerv3-torch/models.py#L29-L215))

**구성요소 조립** ([L30-106](../../../dreamerv3-torch/models.py#L30-L106)):

| 컴포넌트 | 역할 | 출력 shape |
|---|---|---|
| `encoder = MultiEncoder(shapes, **config.encoder)` | obs(dict) → embed | `[B,T,embed_size]` |
| `dynamics = RSSM(...)` | embed+action → latent state | `{stoch, deter, logit, ...}` |
| `heads["decoder"] = MultiDecoder` | feat → obs 재구성 분포 | 관측 키별 분포 |
| `heads["reward"] = MLP(..., dist=symlog_disc)` | feat → reward 분포 | 255-bin twohot |
| `heads["cont"] = MLP(..., dist="binary")` | feat → continue 확률 | Bernoulli |

- `feat_size`: discrete RSSM면 `stoch*discrete + deter`, 아니면 `stoch + deter` ([L56-59](../../../dreamerv3-torch/models.py#L56-L59))
  - 기본값: `32*32 + 512 = 1536`
- `_scales`: reward/cont loss scale은 config에서, 그 외(decoder/kl)는 1.0 고정 ([L102-106](../../../dreamerv3-torch/models.py#L102-L106))
- **옵티마이저 1개로 WorldModel 전체 파라미터 통합 학습** (`_model_opt`, [L89-98](../../../dreamerv3-torch/models.py#L89-L98)) — encoder/dynamics/decoder/reward/cont 모두 함께

**`_train(data)`** ([L108-174](../../../dreamerv3-torch/models.py#L108-L174)) — WorldModel 학습 알고리즘 핵심:

```
1. preprocess(obs)                                    # image/255, cont=1-is_terminal
2. embed = encoder(data)                              # [B,T,E]
3. post, prior = dynamics.observe(embed, action, is_first)
   # post: 관측 반영 latent, prior: 관측 없이 예측한 latent
4. kl_loss, kl_value, dyn_loss, rep_loss = dynamics.kl_loss(post, prior, kl_free, dyn_scale, rep_scale)
   # KL balancing: dyn_scale=0.5 (prior가 post를 따라감), rep_scale=0.1 (post가 prior를 향함)
5. for each head (decoder/reward/cont):
       feat = dynamics.get_feat(post)
       if head not in grad_heads: feat = feat.detach()
       pred = head(feat)
       loss[name] = -pred.log_prob(data[name])
6. model_loss = Σ(scaled losses) + kl_loss
7. _model_opt(mean(model_loss))                       # 단일 옵티마이저 step
```

**Dreamer-v3 핵심 트릭**:
- **KL balancing** ([L121-126](../../../dreamerv3-torch/models.py#L121-L126)): `kl_loss` 두 항으로 분리
  - `dyn_scale=0.5`: prior가 post를 따라가는 방향 (prior 학습, post detach)
  - `rep_scale=0.1`: post가 prior에 가까워지는 방향 (representation 학습, prior detach)
  - `kl_free=1.0`: free bits (KL이 1.0 미만이면 무시)
- `grad_heads` ([L130-132](../../../dreamerv3-torch/models.py#L130-L132)): 어느 head에서 representation으로 grad 흘릴지 제어
- `_use_amp`: `precision==16`일 때만 autocast (기본 32이므로 OFF)
- 반환 `post`는 detach → ImagBehavior가 출발점으로만 사용

**`preprocess(obs)`** ([L177-192](../../../dreamerv3-torch/models.py#L177-L192)):
- ⚠️ **`obs["image"] = obs["image"] / 255.0` 항상 실행** ([L182](../../../dreamerv3-torch/models.py#L182)) — `image` 키 없는 환경(예: F1TENTH LiDAR만)이면 **KeyError 발생**. 통합 시 수정 필요.
- `cont = 1.0 - is_terminal` — continue head의 라벨
- `is_first`, `is_terminal` 모두 필수 (assert)

**`video_pred(data)`** ([L194-215](../../../dreamerv3-torch/models.py#L194-L215)): 평가용 시각화. **`image` 키 하드코딩** → vector-only obs에선 비활성(`video_pred_log: false`) 필수.

### 2-2. RewardEMA ([L11-26](../../../dreamerv3-torch/models.py#L11-L26))

target return 정규화용 EMA. **5%/95% quantile**을 EMA로 유지:
```python
scale = max(q95 - q05, 1.0)
offset = q05
normed_target = (target - offset) / scale
```
Dreamer-v3가 환경마다 다른 reward 스케일을 자동 적응시키는 메커니즘.

### 2-3. ImagBehavior ([L218-441](../../../dreamerv3-torch/models.py#L218-L441))

**구성** ([L219-288](../../../dreamerv3-torch/models.py#L219-L288)):
- `actor = MLP(feat → num_actions, dist=config.actor.dist)` — continuous면 `'normal'`, 이산이면 `'onehot'`
- `value = MLP(feat → 255 bins, dist='symlog_disc')` — twohot 분포
- `_slow_value = copy.deepcopy(self.value)` — target network, polyak ema ([L257-259](../../../dreamerv3-torch/models.py#L257-L259))
- 옵티마이저 **분리**: `_actor_opt`, `_value_opt`. WorldModel 옵티마이저까지 총 3개
- `ema_vals` buffer로 reward EMA 상태 저장 (체크포인트 포함, [L285-288](../../../dreamerv3-torch/models.py#L285-L288))

**`_train(start, objective)`** ([L290-349](../../../dreamerv3-torch/models.py#L290-L349)):

```
1. _update_slow_target()                              # slow_value ← polyak ema (mix=0.02)
2. imag_feat, imag_state, imag_action = _imagine(start, actor, horizon=15)
   # WorldModel의 dynamics만 사용, 환경 호출 없음
3. reward = objective(...)                            # heads["reward"](feat).mode()
4. target, weights, base = _compute_target(...)       # λ-return + discount cumprod
5. actor_loss = _compute_actor_loss(...) - entropy_bonus * actor_ent
6. value_loss = -value.log_prob(target.detach())
              - value.log_prob(slow_value.mode())     # slow target self-distill
7. actor_opt(actor_loss), value_opt(value_loss)
```

**`_imagine`** ([L351-369](../../../dreamerv3-torch/models.py#L351-L369)): 환경 step 없이 RSSM `img_step`만으로 15-step rollout. `start`(WorldModel.train의 post)에서 출발. `action = policy(feat.detach()).sample()` — actor에 그래디언트 흐름 차단 후 dynamics를 통한 path-grad 사용 가능.

**`_compute_target`** ([L371-389](../../../dreamerv3-torch/models.py#L371-L389)):
- `discount = config.discount * cont_head(feat).mean` — **continue 확률을 discount에 곱함** → 죽을 것 같은 state는 자동으로 가치 절하
- `target = tools.lambda_return(reward[1:], value[:-1], discount[1:], bootstrap=value[-1], λ=0.95)`
- `weights = cumprod(discount)` — imagination trajectory의 step별 가중치

**`_compute_actor_loss`** ([L391-433](../../../dreamerv3-torch/models.py#L391-L433)) — 3가지 모드:
- `dynamics` (기본, continuous): `adv = normed_target - normed_base` 를 그대로 actor loss로. RSSM이 미분가능하므로 path-derivative 사용
- `reinforce` (discrete): `log_prob(action) * (target - V).detach()`
- `both`: mix 보간
- **reward_EMA로 정규화한 advantage** 사용 → 학습 안정성

**Value loss 트릭** ([L324-332](../../../dreamerv3-torch/models.py#L324-L332)):
- `-log_prob(target)` (λ-return regression)
- `-log_prob(slow_target.mode())` (slow value 자기증류 추가)
- 두 항을 더해 사용 — Dreamer-v3 논문의 안정화 기법

### 2-4. 손실 흐름 한눈에

```
        ┌─ encoder ─ embed ─┐
obs ────┤                   ├─ RSSM.observe ─ post ─┐
        └─ action ──────────┘                       │
                                                    ├─ decoder ──→ recon_loss
                                                    ├─ reward ──→ reward_loss (symlog_disc)
                                                    ├─ cont   ──→ cont_loss (binary)
                                                    └─ KL(post||prior) ──→ dyn_loss + rep_loss
                                                    ─── (sum) → model_opt

post(detach) ─→ _imagine(actor, 15-step) ─→ imag_feat, action
  ├─ reward_head(feat) ──→ imag_reward
  ├─ cont_head(feat) ──→ discount
  ├─ value(feat) ──→ V
  └─ λ-return → target → reward_EMA normalize
     ├─ actor_loss = -weights * (normed_target - normed_base) - α·entropy ──→ actor_opt
     └─ value_loss = -log_prob(target) - log_prob(slow_target.mode()) ──→ value_opt
                                                                          ↑
                                                                  slow_value (polyak ema)
```

### 2-5. 분석에서 주목할 디자인 결정

1. **3개 옵티마이저** (model / actor / value) — 그래디언트 흐름 격리
2. **단일 forward에서 모든 head 손실 계산** — 효율적, but `image` 키 하드코딩 ([L182](../../../dreamerv3-torch/models.py#L182))
3. **KL balancing** (dyn 0.5 / rep 0.1) — Dreamer-v3가 v2 대비 가장 크게 바뀐 곳 중 하나
4. **continue head로 discount 변조** — 종료 가까운 state 자동 절하
5. **symlog_disc로 reward/value 표현** — 분포가 넓은 reward 안정 학습
6. **reward EMA + 5/95 quantile** — 환경별 reward scaling 자동화
7. **slow value self-distill** — value loss에 두 항 더하기 트릭
8. **path-derivative(dynamics) vs REINFORCE** 분기 — continuous는 path-grad가 분산 낮음

---

## 3. 다음 세션이 바로 시작할 작업

**3단계 분석: RSSM 동역학** — [networks.py](../../../dreamerv3-torch/networks.py)의 `RSSM` 클래스.

확인할 메서드/개념:
- `__init__`: 파라미터 (stoch=32, deter=512, discrete=32, hidden=512) 의미와 내부 모듈
- `observe(embed, action, is_first)`: 시퀀스를 따라 obs_step 반복 → post/prior 동시 산출
- `obs_step(prev_state, prev_action, embed, is_first)`: post 계산 (관측 반영)
- `img_step(prev_state, prev_action)`: prior 계산 (관측 없이 예측, imagination에서 사용)
- `imagine_with_action(action, state)`: 주어진 action 시퀀스로 prior rollout
- `get_feat(state)`: stoch와 deter를 concat → feat
- `get_dist(state)`: stoch에 대한 분포 (discrete면 OneHotCategorical, continuous면 Normal)
- `kl_loss(post, prior, free, dyn_scale, rep_scale)`: KL balancing 구현 본체
- `_suff_stats_layer`: stoch 분포 파라미터 산출 (`logit` 또는 `mean/std`)
- `unimix_ratio` (=0.01): 균등 분포와의 혼합 (categorical exploration 안정화)
- `initial='learned'`: 초기 state가 학습 가능한 파라미터

**진입 명령 (다음 에이전트)**:
> "RSSM 분석을 시작해. `_thinking/analysis/003-dreamer_code_analysis_part1.md`의 섹션 3에 명시된 항목들을 [networks.py](../../../dreamerv3-torch/networks.py)에서 읽고 정리해."

분석 후 산출물은 `004-dreamer_code_analysis_part2.md`로 누적.

---

## 4. 사용자 컨벤션 재확인

- `_thinking/`은 append-only. 기존 문서 절대 수정 금지.
- 명시적 요청 있을 때만 저장. 분석 단계 1개 끝날 때마다 사용자가 "N 문서로 저장"이라고 요청하는 패턴.
- 한글 응답.
- 사용자 메시지 "진행해"는 다음 단계 진행 신호.
- 사용자는 분석이 "F1TENTH 통합 설계"가 아니라 **dreamer-v3 코드 그 자체 이해**임을 강조했음. 통합 얘기를 섞지 말 것.
