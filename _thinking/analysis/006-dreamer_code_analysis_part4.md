# 006 - Dreamer-v3 코드 분석 (Part 4: Exploration · 데이터 · 포팅 · 평가)

> **Revision**: 2026-05-20 — 4차원 진정성 감사 결과 직접 반영. 기존 "보강" 명명 폐기 — §1·§2가 7·8단계의 정식 본문임을 명시. 신설 §3(JAX→PyTorch 포팅 디테일), §4(평가·체크포인트). 코드 블록 발췌 제거, 누락 항목(`weight_init`/`Once`/`Until`/`eval cache popitem`/`recursively_collect_optim_state_dict`/`enable_deterministic_run`) 보강. 변경 내역은 [008](008-audit-changelog.md) 참조.
> **목적**: `NM512/dreamerv3-torch` 코드 자체 이해 완결.
> **선행 문서**: [003](003-dreamer_code_analysis_part1.md), [004](004-dreamer_code_analysis_part2.md), [005](005-dreamer_code_analysis_part3.md).
> **분석 대상**: `/home/dlacksdn/dreamerv3-torch/exploration.py`, `tools.py`(데이터/포팅/유틸 부분), `dreamer.py`(평가 루프), `models.py`(video_pred 호출 관점).
> **인용 규약**: 줄 링크 `(파일#L<a>-L<b>)`만. 코드 블록 발췌 없음.

---

## 0. 핸드오프

### 현재까지 진행 상태
- ✅ 1단계 진입점·Config — [003](003-dreamer_code_analysis_part1.md)
- ✅ 2~4단계 분포·RSSM·Encoder/Decoder — [004](004-dreamer_code_analysis_part2.md)
- ✅ 5~6단계 WorldModel·RewardEMA·ImagBehavior + Optimizer — [005](005-dreamer_code_analysis_part3.md)
- ✅ 7단계 **Exploration** — 본 문서 §1
- ✅ 8단계 **데이터 파이프라인** — 본 문서 §2
- ✅ 9단계 **JAX→PyTorch 포팅 디테일** — 본 문서 §3
- ✅ **평가·체크포인트** — 본 문서 §4

### 컨벤션
- 한국어 응답.
- 코드 인용은 줄 링크만. 코드 블록 발췌 금지.
- 분석 대상은 **dreamer-v3 코드 자체**. F1TENTH 통합 코멘트 금지.
- 본 감사 작업 한정으로 003~006 직접 수정 허용. 이후 다시 append-only 복귀.

---

## 1. Exploration (`exploration.py`)

위치: [exploration.py:1-135](../../../dreamerv3-torch/exploration.py#L1-L135). `Random`(균등 베이스라인) + `Plan2Explore`(disagreement 보너스). `Dreamer.__init__`에서 `expl_behavior` config 키로 분기 ([dreamer.py:52-56](../../../dreamerv3-torch/dreamer.py#L52-L56)).

### 1-1. Random ([exploration.py:10-37](../../../dreamerv3-torch/exploration.py#L10-L37))

균등 분포 베이스라인.

**`actor(feat)`** ([L16-34](../../../dreamerv3-torch/exploration.py#L16-L34)):
- `feat` 인자를 받지만 본문에서 사용 안 함 — dummy 파라미터 (인터페이스 통일).
- `actor['dist'] == 'onehot'`이면 `OneHotDist(zeros(num_actions).repeat(envs, 1))` — logits=0 → softmax 균등. `unimix_ratio` 미지정이라 기본 0 적용(균등이라 어차피 변화 없음).
- 그 외 (continuous): `Independent(Uniform(low, high).repeat(envs, 1), 1)` — action_space의 low/high 박스에서 균등 sample.
- 매 호출마다 분포 객체 새로 생성 (메모리 할당 비용 있으나 throughput 영향 미미).

**`train(start, context, data)`** ([L36-37](../../../dreamerv3-torch/exploration.py#L36-L37)): `return None, {}` — no-op. 학습 없음.

### 1-2. Plan2Explore.`__init__` ([exploration.py:41-81](../../../dreamerv3-torch/exploration.py#L41-L81))

ImagBehavior 인스턴스 + ensemble + 옵티마이저 조립.

| 속성 | 정의 |
|---|---|
| `_behavior = models.ImagBehavior(config, world_model)` | `_task_behavior`와 별개의 ImagBehavior 인스턴스. actor·value·slow_value·RewardEMA 모두 별도 파라미터. world_model은 공유. |
| `self.actor = self._behavior.actor` | `_policy`에서 `expl_behavior.actor(feat)`로 dispatch하기 위한 alias. |
| `feat_size` | discrete: `stoch·discrete + deter`, continuous: `stoch + deter`. RSSM과 동일. |
| `stoch` (target 차원) | discrete: `dyn_stoch·dyn_discrete`(=1024), continuous: `dyn_stoch`(=32). |
| `size` | `{embed: world_model.embed_size, stoch: stoch, deter: dyn_deter, feat: dyn_stoch+dyn_deter}[disag_target]`. **continuous 가정 식** — discrete RSSM에서 `feat` 옵션 사용 시 차원이 안 맞음. 기본 `disag_target='stoch'`이라 보통 문제 없음. |
| `_networks = ModuleList([MLP(**kw) for _ in range(disag_models=10)])` | ensemble. 각 MLP는 `dist='normal'` 기본 — §1-5. |
| `_expl_opt = tools.Optimizer('explorer', ...)` | ensemble 전용 단일 옵티마이저. lr=model_lr=1e-4. |

MLP kw ([L60-69](../../../dreamerv3-torch/exploration.py#L60-L69)):
- `inp_dim = feat_size + (num_actions if disag_action_cond else 0)`.
- `shape = size` — 출력 차원.
- `layers = disag_layers = 4`, `units = disag_units = 400`, `act = config.act`.
- **`dist`, `std`, `min_std`, `max_std`, `absmax` 등은 명시 안 함** → MLP 기본값 사용.

### 1-3. Plan2Explore.`train` ([exploration.py:83-105](../../../dreamerv3-torch/exploration.py#L83-L105))

흐름:
1. `RequiresGrad(self._networks)` 컨텍스트:
   - `stoch = start['stoch']`. discrete면 `[B, T, 32, 32] → [B, T, 1024]` flatten.
   - `target = {embed: context['embed'], stoch: stoch, deter: start['deter'], feat: context['feat']}[disag_target]` — `disag_target='stoch'` 기본 → flattened stoch.
   - `inputs = context['feat']`. `disag_action_cond=True`면 action을 concat.
   - `metrics.update(self._train_ensemble(inputs, target))` — §1-3 이어서.
2. `RequiresGrad` 컨텍스트 **밖**에서: `metrics.update(self._behavior._train(start, self._intrinsic_reward)[-1])` — explore actor·value 학습.
3. 반환 `(None, metrics)`. `expl_` 접두어가 `Dreamer._train`에서 부착 ([dreamer.py:128](../../../dreamerv3-torch/dreamer.py#L128)).

**컨텍스트 격리 핵심**: ensemble 학습은 `RequiresGrad(self._networks)` 안 → ensemble만 grad on. `_behavior._train`은 컨텍스트 밖에서 자체적으로 actor/value lane을 켠다. 두 학습이 깔끔히 분리, world model로 grad 흐름 없음.

### 1-4. `_intrinsic_reward` ([exploration.py:107-120](../../../dreamerv3-torch/exploration.py#L107-L120))

imagination 중 호출되는 intrinsic reward objective.

흐름:
1. `inputs = feat`. `disag_action_cond=True`면 action concat.
2. **`preds = cat([head(inputs, torch.float32).mode()[None] for head in self._networks], 0)`** — shape `[disag_models=10, ...predicted_size]`.
   - 두 번째 인자 `torch.float32`는 MLP.forward의 `dtype` 파라미터로 전달되나 **본문에서 사용 안 함 (dead arg)** — [networks.py:657](../../../dreamerv3-torch/networks.py#L657). JAX 포팅 잔재.
   - `.mode()`는 `ContDist.mode() = tanh(linear_out)` — §1-5.
3. `disag = mean(std(preds, 0), -1)[..., None]` — ensemble 표준편차(0축=heads)를 stoch 차원에서 평균. shape `[..., 1]`.
4. `if disag_log: disag = log(disag)` — log 압축.
5. `reward = expl_intr_scale * disag`. `expl_extr_scale > 0`이면 `reward += expl_extr_scale * self._reward(feat, state, action)` (외부 reward 람다 — `Plan2Explore.__init__`이 받은 [dreamer.py:51](../../../dreamerv3-torch/dreamer.py#L51) 람다, `.mean()`).
6. 반환 `reward`.

### 1-5. ensemble head 분포 — `ContDist(Independent(Normal, 1))` 기본

`MLP.__init__`이 `dist`, `std`, `min_std`, `max_std`를 받지 않으면 시그니처 기본 ([networks.py:597-600](../../../dreamerv3-torch/networks.py#L597-L600))으로 fall back:

- `dist='normal'`, `std=1.0` (tensor), `min_std=0.1`, `max_std=1.0`, `absmax=None`.

`MLP.dist('normal', mean, std, shape)` 본문 ([networks.py:693-700](../../../dreamerv3-torch/networks.py#L693-L700)):
- `std = (max_std - min_std) * sigmoid(std + 2.0) + min_std = 0.9 · sigmoid(self._std + 2.0) + 0.1`.
- `self._std`는 `torch.tensor((1.0,))` (default) — **학습 안 됨** (Parameter가 아니라 attribute).
- 결과 `std ≈ 0.9 · sigmoid(3.0) + 0.1 ≈ 0.9 · 0.9526 + 0.1 ≈ 0.957` — **batch 전체에서 동일한 스칼라 상수**.
- 분포: `ContDist(Independent(Normal(tanh(linear_out), 0.957), 1), absmax=None)`.

**`mode()`** ([tools.py:577-581](../../../dreamerv3-torch/tools.py#L577-L581)): `out = self._dist.mean = tanh(linear_out)`. `absmax=None`이라 클리핑 없음.

**`log_prob(target)`** ([tools.py:589-590](../../../dreamerv3-torch/tools.py#L589-L590)): underlying `Independent(Normal).log_prob(target)` = isotropic Gaussian log-prob with std≈0.957. 사실상 `-MSE/(2·0.957²) + const`.

**ensemble target의 적합성**:
- `disag_target='stoch'`(기본): target은 one-hot 샘플 (값 ∈ {0, 1}). tanh-bounded mean의 출력 범위 [-1, 1]은 0~1을 커버 — 정상 회귀 가능. 0 쪽에서는 포화 위험 작음.
- `disag_target='embed'` 또는 `'feat'`: target 값이 [-1, 1] 밖일 수 있음 — tanh-bounded mean이 적합 못 함. 사실상 `'stoch'` 타깃과 강결합된 설계.

### 1-6. ensemble 다양성 메커니즘

`_train_ensemble` ([exploration.py:122-135](../../../dreamerv3-torch/exploration.py#L122-L135)):
- `targets = targets.detach(); inputs = inputs.detach()` — ensemble grad가 world model로 흐름 차단.
- `disag_offset > 0`이면 `targets = targets[:, offset:]`, `inputs = inputs[:, :-offset]` — `offset=1` 기본 (configs.yaml:91), 1-step 미래 예측.
- `preds = [head(inputs) for head in self._networks]` — 10개 head 동일 inputs로 forward.
- `likes = cat([mean(p.log_prob(targets))[None] for p in preds], 0)` — 각 head의 평균 log-likelihood.
- `loss = -mean(likes)` — 모든 head 합산 NLL.
- `_expl_opt(loss, self._networks.parameters())` — 단일 옵티마이저, 단일 backward.

**다양성의 출처**:
1. **초기화 차이**. `ModuleList([MLP(**kw) for _ in range(10)])` 생성 시 각 MLP가 `tools.weight_init` 호출하며 PyTorch 글로벌 RNG로 다른 난수 가중치. 글로벌 시드 고정이라 매 실행 같은 10개 head로 시작.
2. **minibatch noise 없음** — 10개 head 모두 정확히 같은 `(inputs, targets)` 받음. bootstrap 샘플링/head별 dropout 없음.
3. **stochastic forward 없음** — 추론(`mode`)은 결정적, 학습 시에도 `Normal` 샘플링 없이 `log_prob`만 계산.

**구조적 위험**: 학습이 진행될수록 10개 모델이 같은 minimum으로 수렴 → disagreement → 0. 코드에 명시적 다양성 보존 장치 없음. `disag_log=True`로 작은 disagreement도 보너스로 유지하는 게 hyperparameter 차원의 완화책.

### 1-7. `_behavior`와 `_task_behavior` 관계

```
dreamer.py:45                            self._task_behavior = ImagBehavior(config, self._wm)
dreamer.py:52-56                         self._expl_behavior = {greedy: lambda: self._task_behavior,
                                                                random: lambda: Random(...),
                                                                plan2explore: lambda: Plan2Explore(...)}[expl_behavior]()
exploration.py:46                        Plan2Explore._behavior = ImagBehavior(config, world_model)
exploration.py:47                        Plan2Explore.actor = self._behavior.actor   # alias for _policy dispatch
```

- `_task_behavior`와 `Plan2Explore._behavior`는 **완전히 별개의 ImagBehavior 인스턴스**. actor/value/slow_value/RewardEMA 모두 별도 파라미터, 별도 옵티마이저.
- 둘 다 같은 `self._wm`을 참조 → world model은 공유. imagination rollout도 같은 RSSM·heads를 통과.
- `self.actor = self._behavior.actor`는 nn.Module attribute alias로 `_policy`의 `self._expl_behavior.actor(feat)` 호출을 가능하게 함.

### 1-8. `expl_until=0` 함의

`Until` 카운터 ([tools.py:869-876](../../../dreamerv3-torch/tools.py#L869-L876)):
- `__call__(step)`: `if not self._until: return True`. **`_until=0` (falsy)이면 영원히 True**.

`Dreamer._policy` 분기 ([dreamer.py:97-105](../../../dreamerv3-torch/dreamer.py#L97-L105)):
- `training=False` (eval): `_task_behavior.actor.mode()`.
- `_should_expl(self._step)` True: `_expl_behavior.actor.sample()`.
- 그 외: `_task_behavior.actor.sample()`.

`expl_until = int(expl_until / action_repeat)` ([dreamer.py:38](../../../dreamerv3-torch/dreamer.py#L38)). configs.yaml:85 기본 `expl_until=0` → `_should_expl`이 영원히 True.

조합별 결과:
| `expl_behavior` | 학습 시 환경 정책 | eval 시 환경 정책 | task actor가 본 환경 데이터 |
|---|---|---|---|
| `greedy` | task actor.sample() | task actor.mode() | task actor 자기 데이터 |
| `random` | random actor.sample() | task actor.mode() | **0 episode** (random이 수집) |
| `plan2explore` | explore actor.sample() | task actor.mode() | **0 episode** (explore가 수집) |

함의:
- `random`/`plan2explore` 모드에서 task actor는 한 번도 자기 정책의 trajectory를 환경에서 본 적이 없음. 학습 신호는 모두 imagination + replay (다른 정책이 수집한 데이터로 학습된 world model).
- world model이 충분히 정확하면 문제없지만, model error가 있으면 task actor의 imagination 분포와 deploy 분포가 어긋남.
- `eval`은 항상 `task_behavior.actor.mode()`이라 eval 점수만으로 task actor 품질을 측정.
- `expl_until > 0`으로 설정하면 그 step까지만 explore actor, 이후 task actor로 환경 수집 → 점진 전환.

### 1-9. `_intrinsic_reward`의 데이터 분포 불일치

- **ensemble 학습 시 inputs/targets**: `context['feat']`, `start['stoch']` — WorldModel `_train`이 실제 replay 데이터로 만든 posterior. 환경 수집 trajectory 분포.
- **`_intrinsic_reward` 호출 시 inputs**: imagination 중 actor가 본 detached feat. **imagination prior 분포**.

두 분포는 동일하지 않음 (replay → encoder → post vs start post → img_step 반복 → prior). ensemble은 전자에서만 학습되고 후자에서 평가됨. 따라서:
- imagination이 데이터 분포에서 멀어지면 ensemble의 예측이 더 흩어져 disagreement가 인위적으로 커짐. **OOD bonus 자기강화 루프** 가능.
- ensemble이 imagination prior 분포에 잘 일반화하면 disagreement 감소 → 보너스 사라짐.
- 코드는 분포 차이를 별도 제어 안 함. `disag_log=True`가 유일한 안정화 장치.

### 1-10. `disag_action_cond=False` 함의

`exploration.py:97-102, 108-110`에서 분기:
- `inputs = context['feat']`이고 `disag_action_cond=True`면 `cat([inputs, action], -1)`.
- `False` 기본 (configs.yaml:94) → ensemble은 `(s_t → s_{t+1})` 회귀이지 `(s_t, a_t → s_{t+1})`가 아님.

결과:
- 같은 state에서 다른 action으로 다른 next state로 가는 분기를 ensemble이 학습 못함 → 그 분기의 분산은 **action 분산이 아닌 환경 stochasticity**로 흡수.
- epistemic uncertainty (모델 모름)와 aleatoric uncertainty (환경 잡음) 분리 약화.
- `True`로 켜면 `inp_dim`에 `num_actions` 더해지고 입력도 concat → action 조건부 동역학 ensemble. epistemic 분리에 유리.

기본 `False` 이유는 코드에 주석 없음. 추정: 작은 disagreement 신호가 action에 분산되어 expl 효율 저하 우려.

### 1-11. 설계 포인트 (Exploration)

1. **`Random.actor`의 dummy feat**: 인터페이스 통일을 위한 dead arg. throughput 영향 없음.
2. **`Plan2Explore`가 ImagBehavior 재사용**: explore policy를 task와 같은 인프라(actor/value/slow_value/RewardEMA)로 학습. `objective` 람다 주입 패턴 ([005 §3-3](005-dreamer_code_analysis_part3.md#3-3-objective는-외부-주입-람다))의 활용 사례.
3. **ensemble head 분포는 fixed-std Normal**: `dist='normal'` 기본에 `std=1.0` 고정 attribute → 효과적으로 isotropic Gaussian (std≈0.957) → MSE에 가까운 손실. tanh-bounded mean으로 stoch 타깃(0/1) 회귀가 자연스러움.
4. **dead `dtype` arg in MLP.forward**: JAX 포팅 잔재. `_intrinsic_reward`의 `head(inputs, torch.float32)` 두 번째 인자는 무시됨.
5. **ensemble 다양성은 초기화에서만**: bootstrap·dropout·stochastic forward 없음. 학습 진행 시 disag → 0 위험. `disag_log=True`로 부분 완화.
6. **`expl_until=0` 영구 explore**: `random`/`plan2explore` 모드에서 task actor는 env 데이터 0. world model 일반화에 의존.
7. **ensemble 학습 분포 vs imagination 분포 불일치**: OOD에서 보너스 자기강화 가능. 별도 제어 장치 없음.
8. **`disag_action_cond=False` 기본**: epistemic/aleatoric 미분리. 결정적 env 가정 강결합.
9. **`RequiresGrad` 분리**: ensemble grad만 on, world model로 안 흐름. `_behavior._train`은 컨텍스트 밖에서 자체 lane 관리.
10. **`disag_offset=1` 1-step 미래 예측**: target과 input을 시간축으로 어긋나게 — `feat_t`로 `target_{t+1}`을 예측. dynamics prediction 본질.

---

## 2. 데이터 파이프라인 (`tools.py`)

### 2-1. `load_episodes` ([tools.py:364-395](../../../dreamerv3-torch/tools.py#L364-L395))

디스크의 `.npz` 파일들을 OrderedDict로 로드.

흐름:
- `reverse=True` (기본): `reversed(sorted(directory.glob('*.npz')))` — 파일명 정렬의 역순 = 최신 episode 우선 (UUID 키가 시간순이라면).
- 매 파일 `np.load` → dict로 풀어 episodes에 추가.
- `total += len(episode['reward']) - 1` — transition 수(reward 길이 - 1) 누적.
- `if limit and total >= limit: break` — 한도 초과 시 중단.
- `reverse=False`: `sorted(directory.glob(...))` 정방향.

호출:
- `dreamer.py:231`: `train_eps = load_episodes(directory, limit=dataset_size)` — 최신 1M step.
- `dreamer.py:236`: `eval_eps = load_episodes(directory, limit=1)` — 1 step 이상 첫 파일만(=최소 1 episode).

OrderedDict 키: 파일명에서 확장자 제외 ([L378](../../../dreamerv3-torch/tools.py#L378)) — 예: `<env.id>-<length>`.

### 2-2. `sample_episodes` ([tools.py:323-361](../../../dreamerv3-torch/tools.py#L323-L361))

replay buffer 샘플 generator. `length` 길이의 sub-sequence를 무한 yield.

흐름:
1. `np_random = np.random.RandomState(seed=0)` — **시드 하드코딩 0**.
2. infinite loop:
   - `size = 0`, `ret = None`.
   - episode 선택 확률 `p = lengths / sum(lengths)` — 매 iteration 재계산 (episodes dict가 학습 도중 추가될 수 있음).
   - `while size < length`:
     - `episode = np_random.choice(episodes.values(), p=p)`.
     - `total = len(episode['reward'])`. `if total < 2: continue` (최소 1 transition 필요).
     - **첫 sub-sequence** (`ret is None`):
       - `index = randint(0, total - 1)` — episode 내 무작위 시작.
       - `ret = {k: v[index:min(index+length, total)].copy() for k, v in episode.items() if 'log_' not in k}` — `log_*` 키 제외.
       - `if 'is_first' in ret: ret['is_first'][0] = True` — **sub-sequence 시작도 강제 first**.
     - **잇기** (`ret is not None`, 즉 size < length이고 첫 sub가 짧았을 때):
       - `index = 0` (다음 episode 처음부터).
       - `possible = length - size`.
       - `ret = {k: np.append(ret[k], v[index:min(index+possible, total)], axis=0) for k, v in episode.items() if 'log_' not in k}`.
       - `if 'is_first' in ret: ret['is_first'][size] = True` — 잇기 경계도 first 마킹.
   - `yield ret`.

**핵심 디테일**:
- **`seed=0` 하드코딩** — `make_dataset(episodes, config)`이 train/eval 둘 다 같은 시드로 호출 → `train_dataset`/`eval_dataset` 두 generator가 같은 RNG 시퀀스로 시작. 다만 `episodes` 딕셔너리 내용·호출 빈도가 달라 실제 샘플 시퀀스 동기화는 안 됨.
- **`set_seed_everywhere(config.seed)`와 무관** — 자체 RandomState이라 글로벌 seed 영향 X. config.seed 바꿔도 episode 샘플 순서 동일.
- **첫 sub-sequence 시작 is_first 강제**: episode 중간에서 잘라도 RSSM이 latent 리셋. batch의 t=0은 항상 reset 가정.
- **잇기 경계의 `is_last`/`is_terminal`/`discount` 미처리**: 두 episode 잇기에서 reset 신호만 가고 종료 신호는 안 감. λ-return은 batch 내에서 계산되지 않으므로 영향 없으나, **WorldModel KL/recon loss는 잇기 경계 직전이 "이전 episode의 마지막"이라는 정보를 잃음**.
- **`if 'is_first' in ret` 가드**: env wrapper가 `is_first` 키를 안 넣어주면 잇기 처리가 누락 — fallback 없음.

### 2-3. `from_generator` ([tools.py:309-320](../../../dreamerv3-torch/tools.py#L309-L320))

generator → batch 묶음.

- `while True: batch = [next(generator) for _ in range(batch_size)]`.
- `data[key] = np.stack([batch[i][key] for i in range(batch_size)], 0)` — 키별 stack.
- yield `data` (dict, 값은 `[B, T, ...]` shape).

**단일 스레드 소비**:
- prefetch 없음, 멀티 워커 없음.
- `np.stack`도 단일 스레드 CPU.
- PyTorch `DataLoader` 미사용 이유:
  - `episodes` dict가 in-memory + 학습 도중 mutable (새 episode 추가). DataLoader의 dataset은 mutable 패턴에 약함.
  - 디스크 I/O가 학습 루프에 없어 prefetch 이득 작음.
  - 이미지 환경 기준 GPU forward >> CPU 샘플링이라 병목 안 됨.

### 2-4. `simulate` ([tools.py:128-249](../../../dreamerv3-torch/tools.py#L128-L249))

환경 step + 데이터 수집 + (옵션) 학습 루프.

**시그니처**: `simulate(agent, envs, cache, directory, logger, is_eval=False, limit=None, steps=0, episodes=0, state=None)`.

**state packing** ([L141-149](../../../dreamerv3-torch/tools.py#L141-L149)):
- `state` 미지정: `step=0`, `episode=0`, `done = np.ones(len(envs), bool)`, `length = zeros(...)`, `obs = [None]*len(envs)`, `agent_state = None`, `reward = [0]*len(envs)`.
- `done = ones` 초기화로 첫 iteration에서 모든 env reset 강제 — 깨끗한 시작.
- 지정 시 위 튜플 unpack.

**메인 루프** (`while (steps and step < steps) or (episodes and episode < episodes)`):
1. **reset 처리** ([L152-165](../../../dreamerv3-torch/tools.py#L152-L165)): `done.any()`이면 done인 env들 `envs[i].reset()` 호출. 결과 transition을 `add_to_cache`로 cache에 추가 (`reward=0.0`, `discount=1.0`).
2. **agent 호출**: `obs = {k: stack([o[k] for o in obs])}` (`log_` 제외) → `action, agent_state = agent(obs, done, agent_state)`. agent는 `Dreamer.__call__` ([dreamer.py:58](../../../dreamerv3-torch/dreamer.py#L58)) — 안에서 train step 트리거.
3. **env step**: `results = [e.step(a) for e, a in zip(envs, action)]`. obs/reward/done unpack.
4. **counter 갱신**: `episode += int(done.sum())`, `length += 1`, `step += len(envs)`, `length *= 1 - done`.
5. **cache 추가**: 매 env의 transition을 `add_to_cache(cache, env.id, transition)`. transition은 `o + {action, reward, discount}`.
6. **episode 종료 처리** ([L201-243](../../../dreamerv3-torch/tools.py#L201-L243)):
   - `save_episodes(directory, {env.id: cache[env.id]})` — 디스크에 .npz로 저장.
   - score/length 계산, log 키(`log_*`) 추출 후 cache에서 pop.
   - **학습 모드**: `step_in_dataset = erase_over_episodes(cache, limit)` → metric 기록.
   - **eval 모드**: `eval_scores`/`eval_lengths` 누적, episodes 충족 시 `eval_return`/`eval_length` 기록.
7. **eval 모드 cache 정리** ([L244-248](../../../dreamerv3-torch/tools.py#L244-L248)): `if is_eval: while len(cache) > 1: cache.popitem(last=False)` — FIFO로 마지막 episode 하나만 남김. eval cache는 video_pred에서 다시 쓰임.

반환: `(step - steps, episode - episodes, done, length, obs, agent_state, reward)` — 다음 호출 state.

**`reward`가 dead state**: state packing의 마지막 원소 `reward`는 unpack 후 [L180](../../../dreamerv3-torch/tools.py#L180)에서 새 reward로 덮어쓰여짐. 호환성 잔재로 보임.

### 2-5. `add_to_cache` / `save_episodes` ([tools.py:252-306](../../../dreamerv3-torch/tools.py#L252-L306))

**`add_to_cache(cache, id, transition)`**:
- `id not in cache`: `cache[id] = {k: [convert(v)]}`. 첫 transition 그대로.
- 이후 호출: 키별로 append. **새 키가 두 번째 transition에서 등장하면 첫 슬롯을 `convert(0 * val)`로 0 패딩** 후 새 값 append — `action` 키가 보통 이 경로 (reset 시점에는 action 없음).

함정:
- **첫 transition에 새 키가 나타나면 0-패딩 안 됨**. env wrapper가 첫 step에서만 던지는 특수 키가 있으면 첫 슬롯이 0이 아닌 실제 값.
- `convert(0 * v)`는 v dtype/shape 유지하면서 0. dict/nested 미지원.

**`save_episodes(directory, episodes)`**:
- 매 episode마다 파일명 `<id>-<length>.npz`.
- **BytesIO 경유**: `with io.BytesIO() as f1: np.savez_compressed(f1, **episode); f1.seek(0); with filename.open('wb') as f2: f2.write(f1.read())`.

BytesIO 경유 이유:
- `np.savez_compressed`는 내부적으로 ZIP 컨테이너를 만들며 여러 write 호출 발생. 직접 디스크 write 시 부분 쓰기 상태(학습 중간 crash)로 손상된 .npz 가능.
- BytesIO로 완전한 ZIP 바이트를 메모리에 만들고 단일 `f2.write`로 flush — **부분 쓰기 손상 회피**.
- 단점: episode 데이터를 메모리에 두 번 보유 (원본 + 압축 버퍼). 이미지 환경에서 spike 가능.
- 완전한 atomic write는 아님 (rename 트릭 안 씀). 파일명에 length 포함이라 덮어쓰기 충돌은 없음.

### 2-6. `erase_over_episodes` / `dataset_size` ([tools.py:267-277](../../../dreamerv3-torch/tools.py#L267-L277))

메모리 cache의 FIFO 한도 적용.

흐름:
- `for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):` — 키 알파벳 역순 순회.
- `if not dataset_size or step_in_dataset + (len(ep['reward']) - 1) <= dataset_size:` 한도 내면 추가. 초과면 `del cache[key]`.
- 반환 `step_in_dataset`.

**핵심**:
- **디스크는 영구 누적**. `save_episodes`가 무조건 새 파일 write, 삭제 로직 없음.
- **메모리(`train_eps` dict)에만 FIFO**.
- **`reversed(sorted(cache.items()))` 키 알파벳 역순** — cache 키는 env.id 기반 (UUID wrapper로 생성, [dreamer.py:200](../../../dreamerv3-torch/dreamer.py#L200)). UUID는 랜덤/유일이라 사전순 ≠ 시간순. **엄밀한 시간 FIFO 아님**.
- 한도 초과 시 그 에피소드는 메모리에서 제거 (디스크 파일은 남음).
- **resume 시**: `load_episodes(directory, limit=dataset_size)` ([dreamer.py:231](../../../dreamerv3-torch/dreamer.py#L231))로 디스크에서 최신 파일부터 dataset_size 한도까지 다시 로드. `reverse=True` 기본이라 최신 우선.

결과: 메모리 replay 윈도우 ≈ 최신 `dataset_size` step. 디스크는 전체 학습 trajectory 영구 보관. 학습 길어지면 디스크 사용 무한 증가 — 별도 정리 스크립트 필요.

`dataset_size=1e6` (configs.yaml:72) → 메모리 100만 step. 이미지면 수십 GB RAM. vector면 훨씬 작음.

### 2-7. 카운터 패밀리 — `Every`, `Once`, `Until` ([tools.py:842-876](../../../dreamerv3-torch/tools.py#L842-L876))

**`Every(every)`**:
- `__call__(step)`: `if not self._every: return 0`. 첫 호출은 `self._last = step; return 1`. 이후 `count = int((step - self._last) / every); self._last += every * count; return count`.
- **drift-free**: `_last`를 호출 시점 `step`이 아니라 `every * count` 단위로 갱신. 호출이 불규칙해도 평균 빈도 정확히 `1/every`.
- 비교: `self._last = step` 갱신이었다면 호출이 정확한 배수에서 안 일어날 때마다 잔여 step 잘려 장기적으로 호출 횟수 감소.
- `count`가 정수라 한 호출에서 여러 학습 step burst 반환 — `Dreamer.__call__`의 `for _ in range(steps)` 루프가 이를 받아 처리.
- envs 여러 개일 때 `_step += len(reset)` ([dreamer.py:82](../../../dreamerv3-torch/dreamer.py#L82))로 한 번에 `len(envs)` step 누적 → `_should_train(step)`이 `len(envs) / 2` 학습 step burst (train_ratio=512, batch_steps=1024 기준).

**`Once`** ([tools.py:858-866](../../../dreamerv3-torch/tools.py#L858-L866)):
- `__call__()`: 첫 호출 True 반환 후 `_once = False`. 이후 영원히 False.
- `Dreamer._should_pretrain`이 인스턴스. 첫 학습 호출에서 `pretrain=100` step burst trigger ([dreamer.py:62-64](../../../dreamerv3-torch/dreamer.py#L62-L64)).
- resume 시 `agent._should_pretrain._once = False`로 강제 ([dreamer.py:299](../../../dreamerv3-torch/dreamer.py#L299)) — 재시작 후 pretrain 안 함.

**`Until(until)`** ([tools.py:869-876](../../../dreamerv3-torch/tools.py#L869-L876)):
- `__call__(step)`: `if not self._until: return True`. 그 외 `return step < self._until`.
- `Dreamer._should_expl`이 인스턴스. `expl_until=0` 기본이라 영원히 True (§1-8).

### 2-8. 설계 포인트 (데이터)

1. **`sample_episodes` 시드 하드코딩 (`seed=0`)**: 글로벌 seed와 무관. train/eval 같은 RNG로 시작. config.seed 바꿔도 episode 샘플 순서 동일.
2. **잇기 경계 신호 처리 비대칭**: `is_first`만 강제 마킹, `is_last`/`is_terminal`/`discount`는 미처리 → recon/KL 측 정보 손실 가능.
3. **`Every` drift-free 카운팅**: `_last += every * count`로 평균 빈도 정확. 불규칙 호출에 강함.
4. **`save_episodes` BytesIO 경유**: 부분 쓰기 손상 회피용 atomic-like write. 메모리 2배 사용 트레이드오프.
5. **`from_generator` 단일 스레드**: prefetch/멀티워커 없음. dict mutability + GPU bottleneck 가정.
6. **`dataset_size` 메모리 한도**: 디스크 영구 누적 + 메모리 FIFO. erase는 env.id 사전순(시간순 아님).
7. **`add_to_cache` zero-padding**: 첫 transition에 새 키 나오면 0-패딩 안 됨. dict/nested 미지원.
8. **`simulate` state packing**: agent_state=(latent, action)이 호출 경계 넘어 RSSM 연속성 유지. reward는 dead.
9. **eval cache popitem**: `is_eval`이면 마지막 episode 하나만 남기고 FIFO 제거 — video_pred용 최소 메모리.
10. **`Once`/`Until`은 `Every`의 보조 패밀리**: 세 클래스가 함께 학습/탐험/burst 트리거를 구성.

---

## 3. JAX→PyTorch 포팅 디테일

이 §은 NM512가 JAX 원본 (danijar/dreamerv3)을 PyTorch로 옮기면서 남긴 구현 디테일을 정리. 알고리즘 자체와는 별개지만, 코드를 읽을 때 자주 부딪히는 패턴.

### 3-1. `static_scan` ([tools.py:795-839](../../../dreamerv3-torch/tools.py#L795-L839))

JAX `lax.scan`의 단순 구현. `RSSM.observe`/`imagine_with_action`/`ImagBehavior._imagine` 모두 이걸 통과.

흐름:
- `last = start` (초기 carry).
- `for index in range(inputs[0].shape[0])`:
  - `last = fn(last, *inp(index))` — fn은 (carry, *inputs[t]) → new_carry.
  - **첫 iteration** (`flag=True`): outputs 구조 초기화.
    - `last`가 dict면 `outputs = {key: value.clone().unsqueeze(0) for key, value in last.items()}` (dict of [1, ...]).
    - `last`가 tuple/list면 각 원소 재귀: 원소가 dict면 dict of [1, ...], 그 외면 [_last.clone().unsqueeze(0)] 리스트.
  - **이후 iteration**: 각 자리에 `cat([outputs[k], last[k].unsqueeze(0)], dim=0)` 누적.
- `if last is dict: outputs = [outputs]` (마지막 정규화).

특징:
- **네스티드 dict/tuple 처리**: `fn`이 dict나 tuple of dicts를 반환해도 자동으로 누적.
- **`clone()`**: 첫 iteration에서 `value.clone().unsqueeze(0)` — view가 아니라 복사. 안전 우선.
- **컴파일 최적화 부족**: 매 iteration `torch.cat` 호출 → 메모리 재할당 누적. JAX `lax.scan`의 `[N, *shape]` 사전 할당과 대조. 정확성 OK, 성능은 미흡.
- `ImagBehavior._imagine`은 H=15회만 도므로 영향 작음. `RSSM.observe`는 T=64회 — 영향 더 크지만 GPU forward 시간이 압도적이라 병목 안 됨.

### 3-2. `static_scan_for_lambda_return` ([tools.py:671-688](../../../dreamerv3-torch/tools.py#L671-L688))

λ-return 전용 별도 구현. `static_scan`과 달리 **역방향 + tensor concat 누적**.

흐름:
- `last = start = bootstrap` (shape `[N, 1]`).
- `indices = reversed(range(T-1))` — 미래 → 현재.
- `for index in indices: last = fn(last, inputs[0][index], inputs[1][index])` — last shape `[N, 1]` 유지.
- 첫 iteration: `outputs = last`. 이후: `outputs = cat([outputs, last], dim=-1)` — 마지막 차원에 붙임.
- T-1 iteration 후 `outputs` shape `[N, T-1]`.
- `reshape([N, T-1, 1])` → `flip(dim=[1])` (시간 순서 복원) → `unbind(dim=0)`.
- 반환: **tuple of N 텐서, 각 shape `[T-1, 1]`**.

`static_scan`을 일반화해서 쓸 수도 있지만 별도 함수로 분리 — 주석 ([tools.py:708-711](../../../dreamerv3-torch/tools.py#L708-L711)) "reimplement to optimize performance". 단방향 누적이라 dict 처리 코드를 생략한 단순 버전.

### 3-3. `Conv2dSamePad` ([networks.py:771-798](../../../dreamerv3-torch/networks.py#L771-L798))

TF의 `padding='SAME'`을 PyTorch에서 흉내. PyTorch 내장 padding은 입력 크기 의존이 안 됨 — 매 forward에서 동적 계산 필요.

흐름:
- `__init__`은 `nn.Conv2d` 그대로 (padding=0 기본).
- `forward`:
  - 입력 크기 `ih, iw` 추출.
  - `pad_h = max((ceil(ih/s) - 1)·s + (k-1)·d + 1 - ih, 0)` — TF SAME 공식.
  - `pad_w` 동일.
  - `pad_h > 0 or pad_w > 0`이면 `F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])` — 비대칭 padding.
  - `F.conv2d(x, weight, bias, stride, padding=0, dilation, groups)` — `self.padding=0`이라 추가 padding 없이 conv.

비대칭 padding 처리가 핵심. 짝수 stride일 때 한쪽이 한 픽셀 더 padding 받음 — TF와 일치.

### 3-4. `ImgChLayerNorm` ([networks.py:801-810](../../../dreamerv3-torch/networks.py#L801-L810))

채널 축 LayerNorm. PyTorch `nn.LayerNorm`은 마지막 차원에 적용되므로 채널 차원으로 옮기는 permute 필요.

흐름:
- `forward(x)`: `x.permute(0, 2, 3, 1)` — `(B, C, H, W) → (B, H, W, C)`.
- `self.norm(x)` — 마지막 축(C)에 LayerNorm.
- `x.permute(0, 3, 1, 2)` — `(B, H, W, C) → (B, C, H, W)` 복원.

BatchNorm 대신 사용:
- BatchNorm은 배치 통계 의존 → RL의 비정상 분포에서 train/eval mode 불일치 문제.
- LayerNorm은 sample-wise → mode 무관, 안정.
- 채널 축 LayerNorm은 instance norm과 비슷한 효과(공간 통계 무시, 채널만 정규화).

### 3-5. `weight_init` / `uniform_weight_init` ([tools.py:879-935](../../../dreamerv3-torch/tools.py#L879-L935))

dreamer-v3 초기화 정책. 모든 backbone Linear/Conv는 `weight_init`, 마지막 출력 layer는 `uniform_weight_init(outscale)`.

**`weight_init(m)`** ([L879-906](../../../dreamerv3-torch/tools.py#L879-L906)):
- `nn.Linear`:
  - `denoms = (in_features + out_features) / 2` (= Xavier의 fan_avg).
  - `scale = 1.0 / denoms`.
  - `std = sqrt(scale) / 0.87962566103423978` — 상수는 표준 truncated normal `[-2, 2]`의 std (=0.879...). std로 나누면 결과 분포의 std가 `sqrt(scale)`이 되도록 보정.
  - `nn.init.trunc_normal_(weight, mean=0, std=std, a=-2·std, b=2·std)`.
  - `bias`가 있으면 0으로 초기화.
- `nn.Conv2d`/`nn.ConvTranspose2d`:
  - `space = kernel_size[0] · kernel_size[1]`.
  - `in_num = space · in_channels`, `out_num = space · out_channels`.
  - 나머지는 Linear와 동일.
- `nn.LayerNorm`: `weight = 1.0`, `bias = 0.0`.

**핵심**: Xavier-like trunc_normal (fan_avg 기반). 0.879... 상수로 `[-2, 2]` 잘림에 의한 분산 축소 보정 → 결과 분포 std = `sqrt(scale)`.

**`uniform_weight_init(given_scale)`** ([L909-935](../../../dreamerv3-torch/tools.py#L909-L935)):
- `f(m)` 클로저 반환.
- `nn.Linear`:
  - `denoms = (in + out) / 2`.
  - `scale = given_scale / denoms`.
  - `limit = sqrt(3 * scale)`.
  - `nn.init.uniform_(weight, a=-limit, b=limit)` — `[-limit, limit]` 균등.
  - bias 0.
- Conv 동일 (space 곱.

**`outscale=0.0` 의미**:
- `scale = 0 / denoms = 0`.
- `limit = sqrt(0) = 0`.
- `uniform_(weight, a=0, b=0)` → 가중치가 **모두 0**.
- bias 0.
- 결과: layer 출력이 정확히 0 (입력 무관).
- value/critic, reward_head 모두 outscale=0.0 ([configs.yaml:52, 54](../../../dreamerv3-torch/configs.yaml#L52)) → 초기 V=0, reward=0 (symlog 공간). 학습 초기 부트스트랩 안정.

**`outscale=1.0`** (decoder, actor, cont_head):
- `limit = sqrt(3 / denoms)`.
- Glorot uniform과 동일 형태.
- decoder 출력이 0이 아닌 raw 픽셀/벡터에서 시작.

### 3-6. `GRUCell` ([networks.py:742-768](../../../dreamerv3-torch/networks.py#L742-L768))

표준 GRU의 3 gate를 한 Linear로 묶어 계산 + LayerNorm.

흐름:
- `__init__`: `Linear(inp_size + size, 3·size, bias=False)` + 옵션 `LayerNorm(3·size, eps=1e-3)`.
- `forward(inputs, state)`:
  - `state = state[0]` — Keras wraps state in list 컨벤션 (RSSM의 `_cell(x, [deter])` 호출 패턴).
  - `parts = layers(cat([inputs, state], -1))` shape `[B, 3·size]`.
  - `reset, cand, update = split(parts, [size]·3, -1)`.
  - `reset = sigmoid(reset)`.
  - `cand = self._act(reset * cand)` — `act=torch.tanh` 기본.
  - `update = sigmoid(update + self._update_bias)` — `update_bias=-1` 기본.
  - `output = update * cand + (1 - update) * state`.
  - 반환 `(output, [output])` — Keras 컨벤션으로 list 래핑.

**`update_bias=-1`의 의미**: sigmoid(0 + (-1)) ≈ 0.27 → 초기에 update gate 작음 → `output ≈ 0.73·state + 0.27·cand` 즉 state 보존 편향. 학습 초기 hidden state가 입력에 휩쓸리지 않음.

표준 PyTorch `nn.GRUCell`을 안 쓰는 이유:
- 3 gate를 별도 Linear 호출 (속도 손해).
- LayerNorm 통합 안 됨.
- update_bias 옵션 없음.
- JAX 원본과 동등 동작 보존을 위한 직접 구현.

### 3-7. AMP scaler 흐름 (Optimizer 본체는 005 §4)

PyTorch AMP의 정석 패턴이 `tools.Optimizer.__call__`에 그대로 반영. JAX 포팅이지만 AMP는 PyTorch 고유 — JAX는 자체 mixed precision 메커니즘이 다르다.

흐름 (005 §4-2 참조):
1. `_scaler.scale(loss).backward()` — loss를 scale_factor만큼 곱한 뒤 backward. grad가 fp16 underflow되지 않게.
2. `_scaler.unscale_(_opt)` — `clip_grad_norm_` 전에 unscale 필요 (clip은 실제 grad norm 기준).
3. `clip_grad_norm_(params, clip)` — global norm clip.
4. `_scaler.step(_opt)` — 내부적으로 grad에 inf/nan 체크. 있으면 step skip + scale_factor 감소.
5. `_scaler.update()` — scale_factor 적응.

`use_amp = (precision == 16)` — configs.yaml:18 기본 32이라 AMP OFF. precision=16 시 enable. AMP OFF여도 GradScaler는 enabled=False로 no-op 통과 → 코드 분기 없음.

### 3-8. Dead `dtype` arg in `MLP.forward` ([networks.py:657](../../../dreamerv3-torch/networks.py#L657))

`def forward(self, features, dtype=None):` — 두 번째 인자 `dtype`은 받지만 본문에서 사용 안 함.

JAX 원본의 dtype 변환 시그니처 잔재. 안전성·성능 영향 0.

호출처:
- 대부분의 호출은 `mlp(features)` 형식 — dtype 미전달.
- `Plan2Explore._intrinsic_reward`의 `head(inputs, torch.float32)` ([exploration.py:112](../../../dreamerv3-torch/exploration.py#L112)) — 두 번째 인자 전달하지만 무시됨.

### 3-9. 설계 포인트 (포팅)

1. **`static_scan` 단순성 우선**: 매 step `torch.cat` 누적. JAX `lax.scan`의 사전 할당 최적화 포기, 정확성 우선.
2. **`Conv2dSamePad` 동적 padding**: 입력 크기 의존이라 매 forward 계산. TF와 동등 동작 보존.
3. **`ImgChLayerNorm` permute 트릭**: PyTorch LayerNorm이 마지막 축 적용이라 채널을 마지막으로 옮겼다 복원.
4. **`weight_init` 0.879 상수**: 표준 truncated normal `[-2, 2]`의 std 보정. 결과 분포 std = `sqrt(scale)` 보장.
5. **`uniform_weight_init(0.0)` → 가중치 0**: outscale=0.0이 value/reward의 초기 0 출력을 만드는 메커니즘.
6. **`GRUCell` 직접 구현**: PyTorch 내장 대신 LayerNorm + update_bias 옵션 추가. 학습 초기 state 보존 편향.
7. **AMP scaler 정석 패턴**: scale → backward → unscale → clip → step → update. precision=32 기본에서는 no-op.
8. **dead `dtype` arg**: JAX 포팅 잔재. 인지하고 있으면 됨.

---

## 4. 평가·체크포인트

### 4-1. eval rollout ([dreamer.py:302-318](../../../dreamerv3-torch/dreamer.py#L302-L318))

메인 루프 안에서 train 직전마다 호출.

흐름:
- `eval_episode_num > 0` 가드.
- `eval_policy = functools.partial(agent, training=False)` — `Dreamer.__call__`에 training=False 고정.
- `simulate(eval_policy, eval_envs, eval_eps, evaldir, logger, is_eval=True, episodes=eval_episode_num)`.
  - `is_eval=True`이라 학습 트리거 안 됨 (`agent(training=False)` 호출이므로 `_train` 분기 건너뜀).
  - `episodes=eval_episode_num=10` 기본 (configs.yaml:13) — 10 에피소드 수집.
  - cache는 메모리 효율을 위해 마지막 episode만 남김 (§2-4).
- `video_pred_log=True`면 `video_pred = agent._wm.video_pred(next(eval_dataset))` 후 `logger.video('eval_openl', ...)` — eval 데이터셋에서 한 batch 뽑아 시각화.

**eval 모드 action**: `_policy`에서 `training=False`이라 `_task_behavior.actor(feat).mode()` — deterministic.

**eval 데이터 분리**:
- `eval_envs` ([dreamer.py:239](../../../dreamerv3-torch/dreamer.py#L239)) — train과 별도 env 인스턴스.
- `eval_eps` ([dreamer.py:236](../../../dreamerv3-torch/dreamer.py#L236)) — `load_episodes(limit=1)`로 1 episode만. evaldir에 저장.
- `eval_dataset` ([dreamer.py:286](../../../dreamerv3-torch/dreamer.py#L286)) — eval cache에서 샘플. video_pred 입력으로만 사용 (학습 안 함).

### 4-2. `video_pred` 평가용 호출 흐름

`video_pred`는 005 §1-5에서 동작 본체를 다룸. 호출 위치 두 군데:
1. **train log 주기**: `Dreamer.__call__` ([dreamer.py:74-76](../../../dreamerv3-torch/dreamer.py#L74-L76)) — `_should_log(step)` True이고 `video_pred_log=True`면 train_dataset에서 batch 뽑아 `train_openl` 로깅.
2. **eval 직후**: 메인 루프 ([dreamer.py:316-318](../../../dreamerv3-torch/dreamer.py#L316-L318)) — eval_dataset에서 뽑아 `eval_openl` 로깅.

둘 다 첫 6 batch, 첫 5 step은 관측 기반 recon, 6번째부터 imagination open-loop. 결과는 `[truth | model | error]` 세로 패널 비디오.

**`image` 키 하드코딩** — vector-only env에선 `video_pred_log: False`로 비활성 필요 (configs `dmc_proprio: video_pred_log: false`가 그 예).

### 4-3. `latest.pt` 단일 파일 정책 ([dreamer.py:330-334](../../../dreamerv3-torch/dreamer.py#L330-L334))

메인 루프 매 iteration 끝에 저장:
- `items_to_save = {agent_state_dict, optims_state_dict}` (옵티마이저는 `recursively_collect_optim_state_dict`로 수집 — §4-5).
- `torch.save(items_to_save, logdir / "latest.pt")` — **단일 파일 덮어쓰기**.

**함의**:
- 중간 스냅샷 없음. 학습 후반에 발산하면 직전 좋은 상태로 돌아갈 수 없음.
- 디스크 공간 절약.
- 스냅샷 원하면 코드 수정 필요 (e.g., `f"step-{step}.pt"`).

### 4-4. resume 시 결정성

`logdir / "latest.pt"` 존재 시 ([dreamer.py:295-299](../../../dreamerv3-torch/dreamer.py#L295-L299)):
- `checkpoint = torch.load(...)`.
- `agent.load_state_dict(checkpoint['agent_state_dict'])` — 모든 nn.Module 파라미터 복원.
- `recursively_load_optim_state_dict(agent, checkpoint['optims_state_dict'])` — 모든 옵티마이저 state 복원 (§4-5).
- `agent._should_pretrain._once = False` — pretrain 안 함.

**시드 결정성**:
- `main` 시작 시 `set_seed_everywhere(config.seed)` ([dreamer.py:207](../../../dreamerv3-torch/dreamer.py#L207)) — `torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed`, `random.seed` 모두 동일 seed.
- `enable_deterministic_run` (옵션, configs.yaml:9 기본 False): `os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'`, `torch.backends.cudnn.benchmark = False`, `torch.use_deterministic_algorithms(True)` — cuBLAS workspace 고정 + cudnn benchmark off + deterministic algo 강제.
- **`sample_episodes`는 자체 `RandomState(0)`** (§2-2) — 글로벌 시드와 무관. resume 시 generator 처음부터 재시작하므로 같은 episode를 같은 순서로 다시 봄.
- env reset 시드는 `seed + id` ([dreamer.py:152 외](../../../dreamerv3-torch/dreamer.py#L152)) — env별 다른 시드.

**결정성 한계**:
- resume 후 `_step`이 보존된다는 보장? `_step`은 `Dreamer.__init__`에서 `logger.step // action_repeat`로 재계산 — logger.step이 디스크 (`metrics.jsonl`)에서 복원되지 않으면 0부터 시작. 실제로는 `count_steps(traindir)`로 `.npz` 파일 길이 합산해 `step` 추정 ([dreamer.py:222](../../../dreamerv3-torch/dreamer.py#L222)).
- `_should_train._last` 등 카운터 state는 `agent.state_dict`에 안 들어감 (nn.Module attribute가 아니라 일반 attribute) → resume 후 첫 호출에서 reset.

### 4-5. `recursively_collect_optim_state_dict` / `recursively_load_optim_state_dict` ([tools.py:964-1000](../../../dreamerv3-torch/tools.py#L964-L1000))

`Dreamer` 인스턴스의 모든 nested attribute를 traverse하면서 `torch.optim.Optimizer` 인스턴스 발견 시 state_dict 수집.

**`recursively_collect_optim_state_dict`** ([L964-991](../../../dreamerv3-torch/tools.py#L964-L991)):
- 재귀 함수. `visited` 셋으로 cyclic reference 회피.
- `obj.__dict__` + (nn.Module이면 `named_modules` 추가)를 순회.
- attr가 `torch.optim.Optimizer`면 `optimizers_state_dicts[path] = attr.state_dict()`.
- 그 외 `hasattr('__dict__')`면 재귀.

**`recursively_load_optim_state_dict`** ([L994-1000](../../../dreamerv3-torch/tools.py#L994-L1000)):
- collect가 저장한 path를 따라 `getattr`로 옵티마이저 찾고 `load_state_dict`.

**왜 필요한가**:
- `nn.Module.state_dict`는 nn.Module 파라미터만 저장. 옵티마이저 state (Adam의 first/second moment)는 별도 객체 → 자동 저장 안 됨.
- `Dreamer._wm._model_opt._opt`, `Dreamer._task_behavior._actor_opt._opt`, `Dreamer._task_behavior._value_opt._opt`, (옵션) `Dreamer._expl_behavior._behavior._actor_opt._opt`, etc. — nested 위치.
- 재귀 traverse로 모든 옵티마이저 자동 발견·저장·복원.

### 4-6. `set_seed_everywhere` / `enable_deterministic_run` ([tools.py:950-961](../../../dreamerv3-torch/tools.py#L950-L961))

**`set_seed_everywhere(seed)`**:
- `torch.manual_seed(seed)`.
- `if cuda.is_available(): torch.cuda.manual_seed_all(seed)`.
- `np.random.seed(seed)`.
- `random.seed(seed)`.

**`enable_deterministic_run()`**:
- `os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'` — cuBLAS workspace 고정.
- `torch.backends.cudnn.benchmark = False` — cudnn auto-tuner OFF.
- `torch.use_deterministic_algorithms(True)` — deterministic algo 강제. nondeterministic op은 에러.

기본 `deterministic_run=False`이라 비활성. 디버깅·재현성 검증 시 활성화. 학습 속도 손해 있음.

### 4-7. 설계 포인트 (평가·체크포인트)

1. **eval은 메인 루프에 인라인** — `eval_every` 주기로 항상 실행. eval_episode_num=10 기본.
2. **`is_eval=True` cache 정리** — 메모리 효율 위해 마지막 episode만 남김 (video_pred용).
3. **`video_pred` 두 위치 호출** — train log 주기, eval 직후. 시각화 전용 (logger video).
4. **`latest.pt` 단일 파일** — 덮어쓰기. 스냅샷 없음. 디스크 절약 ↔ 안전성 트레이드오프.
5. **옵티마이저 state 재귀 수집/복원** — nested 위치 자동 처리. nn.Module state_dict가 안 잡는 부분 보완.
6. **시드는 글로벌만** — `sample_episodes`/env reset이 자체 시드라 글로벌 시드 영향 부분적. resume 시 generator 처음부터 재시작이라 시드 결정성 깨질 수 있음.
7. **`deterministic_run` 옵션** — 재현성 검증 시 활성, 평소는 OFF (속도 우선).

---

## 5. 분석 종료 선언

본 시리즈(003·004·005·006)로 NM512/dreamerv3-torch의 알고리즘 코드 자체(`dreamer.py`, `models.py`, `networks.py`, `tools.py`, `exploration.py`, `configs.yaml`) 분석을 종료한다.

### 5-1. 단계별 커버리지

| 단계 | 위치 |
|---|---|
| 1. 진입점·Config | [003](003-dreamer_code_analysis_part1.md) |
| 2. 분포 카탈로그 | [004 §1](004-dreamer_code_analysis_part2.md#1-분포-카탈로그-toolspy) |
| 3. RSSM | [004 §2](004-dreamer_code_analysis_part2.md#2-rssm-동역학-networkspy의-rssm) |
| 4. Encoder/Decoder | [004 §3](004-dreamer_code_analysis_part2.md#3-encoderdecoder-networkspy) |
| 5. WorldModel | [005 §1](005-dreamer_code_analysis_part3.md#1-worldmodel-modelspy) |
| 5a. RewardEMA | [005 §2](005-dreamer_code_analysis_part3.md#2-rewardema--595-quantile-ema-modelspy11-26) |
| 6. ImagBehavior | [005 §3](005-dreamer_code_analysis_part3.md#3-imagbehavior-modelspy) |
| 6a. Optimizer | [005 §4](005-dreamer_code_analysis_part3.md#4-optimizer-래퍼-toolspy) |
| 7. Exploration | 본 문서 §1 |
| 8. 데이터 파이프라인 | 본 문서 §2 |
| 9. 포팅 디테일 | 본 문서 §3 |
| 9a. 평가·체크포인트 | 본 문서 §4 |

### 5-2. 의도적 제외 (재확인)

- `envs/` — 환경 어댑터, 환경별 디테일.
- `parallel.py` — 멀티프로세스 환경 실행, 알고리즘 무관.
- `Dockerfile`, `requirements.txt`, `xvfb_run.sh` — 환경 구성.
- `tools.Logger` — TensorBoard/JSONL 출력, 알고리즘 무관.
- `offline_traindir`/`offline_evaldir` — 오프라인 학습 경로 (옵션).
- `debug` config — 디버깅 단축 설정.
- `tools.TimeRecording` — 프로파일링 utility.
- `tools.tensorstats` — metric 계산 utility.

### 5-3. 본 감사의 한계 (재확인 from 008)

- v2 논문 대조 제외 (사용자 결정).
- JAX 정식 구현 대조 제외 (포팅 충실성이 목적 아님).
- atomic claim 수준 100% 검증은 spot check 비율 존재. 핵심 알고리즘 경로(WorldModel `_train`, ImagBehavior `_train`, RSSM dynamics, 분포 클래스)는 line-by-line 검증, 보조 유틸은 시그니처 + 핵심 동작.
- unknown unknowns 잔존. configs.yaml 키 매핑으로 일부 보완.

---

## 6. 컨벤션 재확인

- `_thinking/analysis/`는 append-only — 본 감사 작업 한정 1회 예외.
- 한국어 응답.
- 코드 인용은 줄 링크만. 코드 블록 발췌 금지.
- 분석 대상은 **dreamer-v3 코드 자체**. F1TENTH 통합 코멘트 금지.
- 본 감사 종료 후 다시 append-only 복귀.
