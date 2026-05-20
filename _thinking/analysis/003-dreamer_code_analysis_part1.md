# 003 - Dreamer-v3 코드 분석 (Part 1: 진입점 + Config)

> **Revision**: 2026-05-20 — 4차원 진정성 감사 결과 직접 반영. 기존 §2(WorldModel·ImagBehavior 개요)와 RewardEMA를 [005](005-dreamer_code_analysis_part3.md)로 이전, 코드 블록 발췌 제거, F1TENTH 통합 코멘트 제거(규약 위반 정리), 누락 항목(`eval_state_mean`/`onehot_gumble`/`Once` pretrain) 보강. 변경 내역은 [008](008-audit-changelog.md) 참조.
> **목적**: F1TENTH RL 프로젝트에서 사용할 `NM512/dreamerv3-torch`의 **코드 자체** 이해.
> **선행 문서**: [001-env-analysis.md](001-env-analysis.md) F1Tenth Gym 분석, [002-Select_Implementation.md](002-Select_Implementation.md) 구현체 선택.
> **분석 대상 경로**: `/home/dlacksdn/dreamerv3-torch`.
> **인용 규약**: 줄 링크 `(파일#L<a>-L<b>)`만. 코드 블록 발췌 없음.

---

## 0. 핸드오프

### 현재까지 진행 상태
- ✅ 1단계 **진입점 + Config** — 본 문서
- ✅ 2단계 분포 카탈로그 — [004 §1](004-dreamer_code_analysis_part2.md#1-분포-카탈로그-toolspy)
- ✅ 3단계 RSSM — [004 §2](004-dreamer_code_analysis_part2.md#2-rssm-동역학-networkspy의-rssm)
- ✅ 4단계 Encoder/Decoder — [004 §3](004-dreamer_code_analysis_part2.md#3-encoderdecoder-networkspy)
- ✅ 5단계 WorldModel — [005 §1](005-dreamer_code_analysis_part3.md#1-worldmodel-modelspy)
- ✅ RewardEMA — [005 §2](005-dreamer_code_analysis_part3.md#2-rewardema--595-quantile-ema-modelspy11-26)
- ✅ 6단계 ImagBehavior — [005 §3](005-dreamer_code_analysis_part3.md#3-imagbehavior-modelspy)
- ✅ Optimizer 래퍼 — [005 §4](005-dreamer_code_analysis_part3.md#4-optimizer-래퍼-toolspy)
- ⏭️ 7~9단계 Exploration + 데이터 + 포팅 + 평가 — [006](006-dreamer_code_analysis_part4.md)

### 파일 위치·라인 수
| 파일 | 라인 수 | 분석 위치 |
|---|---|---|
| `dreamer.py` | 365 | 본 문서 |
| `models.py` | 441 | 005 |
| `networks.py` | 810 | 004 |
| `tools.py` | 1000 | 004(분포), 005(Optimizer), 006(데이터·포팅) |
| `exploration.py` | 135 | 006 |
| `configs.yaml` | 184 | 본 문서 §2 |
| `parallel.py`, `envs/`, `Dockerfile`, `requirements.txt`, `xvfb_run.sh` | — | **분석 대상 제외** |

### 컨벤션
- `_thinking/analysis/`는 append-only — 본 감사 작업 한정 1회 예외로 003~006 직접 수정 허용. 이후 다시 append-only 복귀.
- 한국어 응답.
- 코드 인용은 줄 링크만. 코드 블록 ``` ``` 발췌 금지. 표·산문·수식 인라인 허용.
- 분석 대상은 **dreamer-v3 코드 자체**. F1TENTH 통합 코멘트 금지.

---

## 1. 진입점·실행 흐름 (`dreamer.py`)

### 1-1. main 흐름 ([dreamer.py:206-339](../../../dreamerv3-torch/dreamer.py#L206-L339))

| 순서 | 단계 | 라인 |
|---|---|---|
| 1 | `set_seed_everywhere(config.seed)` + 옵션 `enable_deterministic_run` | [L207-209](../../../dreamerv3-torch/dreamer.py#L207-L209) |
| 2 | `traindir`/`evaldir` 생성, `steps`/`eval_every`/`log_every`/`time_limit`를 `action_repeat`로 나눔 | [L210-216](../../../dreamerv3-torch/dreamer.py#L210-L216) |
| 3 | `logger = tools.Logger(logdir, action_repeat·step)` (TensorBoard + JSONL) | [L224](../../../dreamerv3-torch/dreamer.py#L224) |
| 4 | `train_eps = load_episodes(directory, limit=dataset_size)`, `eval_eps = load_episodes(directory, limit=1)` | [L231-236](../../../dreamerv3-torch/dreamer.py#L231-L236) |
| 5 | `make_env × config.envs`개 생성 → `Parallel(env, 'process')` 또는 `Damy(env)` 래핑 | [L237-245](../../../dreamerv3-torch/dreamer.py#L237-L245) |
| 6 | **Prefill** — 랜덤 정책으로 `prefill=2500` 스텝 수집 (offline_traindir 미지정 시) | [L251-281](../../../dreamerv3-torch/dreamer.py#L251-L281) |
| 7 | `train_dataset = make_dataset(train_eps, config)` + 동일하게 eval | [L285-286](../../../dreamerv3-torch/dreamer.py#L285-L286) |
| 8 | `agent = Dreamer(obs_space, act_space, config, logger, train_dataset)`, `requires_grad_(False)` | [L287-294](../../../dreamerv3-torch/dreamer.py#L287-L294) |
| 9 | `latest.pt` 있으면 resume — `agent.load_state_dict` + `recursively_load_optim_state_dict` + `_should_pretrain._once = False` | [L295-299](../../../dreamerv3-torch/dreamer.py#L295-L299) |
| 10 | 메인 루프: `while agent._step < steps + eval_every:` — eval → train(`eval_every` 스텝) → 체크포인트 저장 반복 | [L302-334](../../../dreamerv3-torch/dreamer.py#L302-L334) |

**메인 루프 세부**:
- `eval_episode_num > 0`이면 `functools.partial(agent, training=False)`로 평가 (mode 액션). `video_pred_log=True`면 `video_pred` 호출 ([L316-318](../../../dreamerv3-torch/dreamer.py#L316-L318)).
- `simulate(agent, train_envs, train_eps, traindir, logger, limit=dataset_size, steps=eval_every, state=state)` — 학습 데이터 수집 + 학습 (`agent.__call__` 내부에서 train step 트리거).
- `items_to_save = {agent_state_dict, optims_state_dict}` 후 `torch.save(items_to_save, logdir / "latest.pt")` — **단일 파일 덮어쓰기**. 중간 스냅샷 없음.
- 루프 조건 `< steps + eval_every`로 마지막 한 번의 eval 보장.

### 1-2. `Dreamer` 클래스 ([dreamer.py:28-133](../../../dreamerv3-torch/dreamer.py#L28-L133))

**`__init__`** ([L29-56](../../../dreamerv3-torch/dreamer.py#L29-L56)):

| 속성 | 정의 |
|---|---|
| `_should_log = Every(log_every)` | log 주기 카운터 |
| `_should_train = Every(batch_size·batch_length / train_ratio)` | 기본 `(16·64)/512 = 2` → env step 2회마다 train 1회 |
| `_should_pretrain = Once()` | 첫 호출 True → `config.pretrain=100` 만큼 train step burst |
| `_should_reset = Every(reset_every)` | configs.yaml:15 reset_every=0이라 비활성 |
| `_should_expl = Until(int(expl_until / action_repeat))` | exploration 정책 사용 여부 (006 §1) |
| `_wm = models.WorldModel(...)` | [005 §1](005-dreamer_code_analysis_part3.md#1-worldmodel-modelspy) |
| `_task_behavior = models.ImagBehavior(config, self._wm)` | [005 §3](005-dreamer_code_analysis_part3.md#3-imagbehavior-modelspy) |
| `_expl_behavior` | `greedy`(`_task_behavior` 자체)/`random`(Random)/`plan2explore`(Plan2Explore). 006 §1 |

**torch.compile** ([L46-50](../../../dreamerv3-torch/dreamer.py#L46-L50)): `compile=True` (configs 기본) + Windows 아니면 `_wm`/`_task_behavior`를 `torch.compile`로 감쌈. 디버깅 시 끄는 게 편함.

**Plan2Explore용 reward 람다** ([L51](../../../dreamerv3-torch/dreamer.py#L51)): `reward = lambda f, s, a: self._wm.heads["reward"](f).mean()`. **`.mean()` 사용**. 입력 `f`(feat) 그대로. Plan2Explore가 `extr_scale > 0`일 때 보조 extrinsic 보너스로 사용. 자세한 흐름은 [005 §3-3](005-dreamer_code_analysis_part3.md#3-3-objective는-외부-주입-람다)와 006 §1 참조.

**`__call__(obs, reset, state, training)`** ([L58-84](../../../dreamerv3-torch/dreamer.py#L58-L84)):
1. `training=True`이면 train step 트리거:
   - 첫 호출은 `_should_pretrain()`이 True → `steps = pretrain = 100`.
   - 그 후 `_should_train(step)` (Every 카운터 — 006 §2-7).
   - `for _ in range(steps): self._train(next(self._dataset))`.
2. `_should_log(step)`이 True면 metric 기록 + `video_pred` (옵션) + `logger.write(fps=True)`.
3. `_policy(obs, state, training)` 호출.
4. `training=True`면 `self._step += len(reset)` (병렬 env 수만큼 누적).

**`_policy(obs, state, training)`** ([L86-115](../../../dreamerv3-torch/dreamer.py#L86-L115)):
1. `state = (latent, action)` 분해 (첫 호출이면 둘 다 None).
2. `obs = self._wm.preprocess(obs)` — image/255, cont 계산 ([005 §1-4](005-dreamer_code_analysis_part3.md#1-4-preprocessobs-modelspy177-192)).
3. `embed = self._wm.encoder(obs)`.
4. `latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs['is_first'])` — RSSM post 한 step ([004 §2-4](004-dreamer_code_analysis_part2.md#2-4-obs_step--post-한-스텝-networkspy174-206)).
5. **`eval_state_mean` 분기** ([L94-95](../../../dreamerv3-torch/dreamer.py#L94-L95)): `if eval_state_mean: latent['stoch'] = latent['mean']`. configs.yaml:81 기본 False. True면 continuous RSSM의 stoch을 sample 대신 mean으로 (deterministic eval). **discrete RSSM에는 `mean` 키가 없으므로 KeyError** — continuous 전용 옵션.
6. `feat = self._wm.dynamics.get_feat(latent)`.
7. **액션 선택**:
   - `training=False` (eval): `actor = self._task_behavior.actor(feat); action = actor.mode()` — deterministic.
   - `self._should_expl(self._step)` True: `actor = self._expl_behavior.actor(feat); action = actor.sample()` — exploration.
   - 그 외 (training + not explore): `actor = self._task_behavior.actor(feat); action = actor.sample()`.
8. `logprob = actor.log_prob(action)`.
9. `latent`/`action` detach (다음 호출 state로 저장).
10. **`onehot_gumble` 후처리** ([L109-112](../../../dreamerv3-torch/dreamer.py#L109-L112)): `if actor['dist'] == 'onehot_gumble': action = torch.one_hot(torch.argmax(action, dim=-1), num_actions)`. gumbel sample을 hard one-hot으로 변환. 기본 actor dist는 'normal'/'onehot'이라 평소 무시.
11. 반환 `({action, logprob}, (latent, action))`.

**`_train(data)`** ([L117-133](../../../dreamerv3-torch/dreamer.py#L117-L133)):
1. `post, context, mets = self._wm._train(data)` ([005 §1-2](005-dreamer_code_analysis_part3.md#1-2-traindata--단일-forward에서-모든-loss-계산-modelspy108-174)).
2. **task behavior reward 람다** ([L122-124](../../../dreamerv3-torch/dreamer.py#L122-L124)): `reward = lambda f, s, a: self._wm.heads["reward"](self._wm.dynamics.get_feat(s)).mode()`. **`.mode()` 사용**. 입력 state `s`에서 `get_feat`로 feat 계산 후 reward head. ([005 §3-3](005-dreamer_code_analysis_part3.md#3-3-objective는-외부-주입-람다)의 task behavior 람다.)
3. `self._task_behavior._train(start=post, reward)` 호출 — actor·value 학습.
4. `expl_behavior != 'greedy'`이면 `self._expl_behavior.train(start, context, data)` — Plan2Explore ensemble + explore actor 학습. metric에 `expl_` 접두어.
5. metric 누적.

**dreamer.py:51 vs L122-124 두 람다 비교**:
| 비교 | L51 (Plan2Explore용) | L122-124 (task behavior용) |
|---|---|---|
| 입력 | `f`(feat) 직접 | `s`(state)에서 `get_feat(s)`로 feat 계산 |
| 메서드 | `.mean()` | `.mode()` |
| 호출 컨텍스트 | `Plan2Explore.__init__`에 전달 | `_train`마다 새로 만들어 `_task_behavior._train`에 전달 |
| 결과값 | 동일 (`DiscDist.mean() == DiscDist.mode()` — [004 §1-2](004-dreamer_code_analysis_part2.md#1-2-discdist--twohot-인코딩-categorical-toolspy452-506)) | 동일 |

### 1-3. `make_env` ([dreamer.py:146-203](../../../dreamerv3-torch/dreamer.py#L146-L203))

`task = "suite_subtask"` 형식 (예: `dmc_walker_walk`) → `_`로 split → suite별 환경 클래스 분기.

| suite | 환경 모듈 | action wrapper |
|---|---|---|
| `dmc` | `envs.dmc.DeepMindControl` | `NormalizeActions` |
| `atari` | `envs.atari.Atari` | `OneHotAction` |
| `dmlab` | `envs.dmlab.DeepMindLabyrinth` | `OneHotAction` |
| `memorymaze` | `envs.memorymaze.MemoryMaze` | `OneHotAction` |
| `crafter` | `envs.crafter.Crafter` | `OneHotAction` |
| `minecraft` | `envs.minecraft.make_env` | `OneHotAction` + `RewardObs` |

공통 wrapper 체인 ([L198-200](../../../dreamerv3-torch/dreamer.py#L198-L200)):
- `TimeLimit(time_limit)` — 강제 종료.
- `SelectAction(key='action')` — dict action에서 'action' 키 추출.
- `UUID(env)` — 에피소드 식별자 (cache 키로 사용).

`envs/` 모듈 자체는 본 분석 대상 제외 (환경 어댑터).

### 1-4. `make_dataset` ([dreamer.py:140-143](../../../dreamerv3-torch/dreamer.py#L140-L143))

- `generator = tools.sample_episodes(episodes, batch_length)` — replay buffer 샘플 generator (006 §2-2).
- `dataset = tools.from_generator(generator, batch_size)` — batch 묶음 (006 §2-3).
- `Dreamer.__call__` 안에서 `next(self._dataset)`로 소비.

### 1-5. Config 상속·argparse ([dreamer.py:342-365](../../../dreamerv3-torch/dreamer.py#L342-L365))

흐름:
1. `--configs A B C ...` argparse로 받음.
2. `configs.yaml` 로드 → `defaults` 블록을 베이스로 깔고 `--configs` 순서대로 **재귀 머지** (`recursive_update`):
   - 값이 dict이고 base에도 같은 키가 dict면 재귀.
   - 그 외에는 base[key] = value (덮어쓰기).
   - 따라서 nested dict는 키 단위 부분 override 가능. 예: `encoder: {mlp_keys: '.*'}`만 적어도 encoder 전체가 사라지지 않고 mlp_keys만 변경.
3. 두 번째 argparse가 모든 키를 `--key value` 형식으로 받을 수 있게 동적 생성.
4. `args_type(default)` — 값 타입을 default 기반으로 추론 (006 §3 utility).

**suite별 override 예** (configs.yaml):
- `dmc_proprio`: vector-only — `encoder: {mlp_keys: '.*', cnn_keys: '$^'}`, `decoder` 동일.
- `dmc_vision`: 이미지 — `encoder: {mlp_keys: '$^', cnn_keys: 'image'}`, `decoder` 동일.
- `atari100k`: `actor: {dist: 'onehot', std: 'none'}`, `imag_gradient: 'reinforce'`, action_repeat=4.
- `crafter`: `actor: {layers: 5, dist: 'onehot', std: 'none'}`, value/reward/cont도 5 layers, dyn_hidden/deter=1024/4096.
- `minecraft`: 위 + parallel=True, train_ratio=16, time_limit=36000, `mlp_keys`에 다중 키 정규식.
- `memorymaze`: `actor: {dist: 'onehot', std: 'none'}`, `imag_gradient: 'reinforce'`.

### 1-6. 추가 관찰

- **체크포인트**: `latest.pt` 단일 파일만 저장/덮어쓰기. 중간 스냅샷 원하면 코드 수정 필요. `recursively_collect_optim_state_dict` / `recursively_load_optim_state_dict`로 nested 옵티마이저 상태 자동 수집·복원 (006 §4).
- **`compile=True` 기본** ([L47](../../../dreamerv3-torch/dreamer.py#L47)): `torch.compile` 켜져 있음. 디버깅 시 끄는 게 편함.
- **`_should_pretrain = Once()` 메커니즘** ([L36, L62-64](../../../dreamerv3-torch/dreamer.py#L36)): 첫 `__call__`에서 True 반환 → `pretrain=100` step 학습. 이후 False. resume 시 `_should_pretrain._once = False`로 강제 ([L299](../../../dreamerv3-torch/dreamer.py#L299)) — 재시작 후 pretrain 안 함.
- **`_should_train` 카운팅**: `Every` drift-free 구현이라 호출이 불규칙해도 평균 빈도 정확 (006 §2-7).
- **`_step`은 update step** ([L41](../../../dreamerv3-torch/dreamer.py#L41)): `logger.step // action_repeat`. log에 기록되는 step은 `action_repeat * self._step`.
- **`requires_grad_(False)`** ([L294](../../../dreamerv3-torch/dreamer.py#L294)): agent 생성 직후 모든 파라미터 grad 비활성. 학습 시 `RequiresGrad` 컨텍스트로 lane별 활성 ([005 §3-2](005-dreamer_code_analysis_part3.md#3-2-trainstart-objective--전체-흐름-modelspy290-349)) — actor/value/world model lane이 격리.

---

## 2. `configs.yaml` 주요 디폴트

| 카테고리 | 키 | 값 | 의미 |
|---|---|---|---|
| **일반** | `steps` | 1e6 | 총 환경 step (action_repeat 적용 전) |
| | `action_repeat` | 2 | env step 1회 = 시뮬 2회 |
| | `eval_every` | 1e4 | eval 주기 |
| | `eval_episode_num` | 10 | eval 당 에피소드 수 |
| | `log_every` | 1e4 | log 주기 |
| | `seed` | 0 | 글로벌 시드 |
| | `deterministic_run` | False | 옵션 |
| | `compile` | True | torch.compile |
| | `precision` | 32 | AMP는 16일 때만 |
| | `video_pred_log` | True | video_pred 호출 여부 |
| **환경** | `task` | dmc_walker_walk | 기본 환경 |
| | `size` | [64, 64] | 이미지 크기 |
| | `envs` | 1 | 병렬 환경 수 |
| | `time_limit` | 1000 | 에피소드 최대 길이 |
| | `prefill` | 2500 | 랜덤 정책 수집 |
| | `reward_EMA` | True | RewardEMA toggle (005 §2) |
| **RSSM** | `dyn_hidden` | 512 | MLP hidden |
| | `dyn_deter` | 512 | GRU hidden |
| | `dyn_stoch` | 32 | stoch 차원 |
| | `dyn_discrete` | 32 | discrete 클래스 수 (=0이면 continuous) |
| | `dyn_rec_depth` | 1 | (코드상 1만 동작) |
| | `dyn_mean_act` | 'none' | continuous mean 활성화 |
| | `dyn_std_act` | 'sigmoid2' | continuous std 활성화 (`2·sigmoid(std/2)`) |
| | `dyn_min_std` | 0.1 | std 하한 |
| | `unimix_ratio` | 0.01 | OneHotDist 균등 혼합 |
| | `initial` | 'learned' | RSSM 초기 deter |
| **공통 nn** | `units` | 512 | MLP 단위 |
| | `act` | SiLU | 활성화 |
| | `norm` | True | LayerNorm |
| **encoder** | `mlp_keys` | '$^' (빈 매치) | MLP 라우팅 정규식 |
| | `cnn_keys` | 'image' | CNN 라우팅 정규식 |
| | `cnn_depth` | 32 | 첫 stage out_dim |
| | `kernel_size` | 4 | conv kernel |
| | `minres` | 4 | 최종 해상도 |
| | `mlp_layers` | 5 | MLP 깊이 |
| | `mlp_units` | 1024 | MLP 단위 (encoder만 override) |
| | `symlog_inputs` | True | MLP 입력 symlog |
| **decoder** | (encoder와 동일 + 아래) | | |
| | `cnn_sigmoid` | False | False면 mean+=0.5 |
| | `image_dist` | mse | image head 분포 ([004 §1-4](004-dreamer_code_analysis_part2.md#1-4-msedist--image-reconstruction-toolspy509-529)) |
| | `vector_dist` | symlog_mse | vector head 분포 ([004 §1-3](004-dreamer_code_analysis_part2.md#1-3-symlogdist--vector-reconstruction-toolspy532-561)) |
| | `outscale` | 1.0 | decoder 마지막 layer 초기 스케일 |
| **actor** | `layers` | 2 | MLP 깊이 |
| | `dist` | 'normal' | continuous 기본. atari/crafter 등은 'onehot' |
| | `entropy` | 3e-4 | entropy bonus |
| | `unimix_ratio` | 0.01 | onehot용 |
| | `std` | 'learned' | std 학습 |
| | `min_std`/`max_std` | 0.1 / 1.0 | std 범위 |
| | `temp` | 0.1 | gumbel용 |
| | `lr`/`eps`/`grad_clip` | 3e-5 / 1e-5 / 100 | 옵티마이저 |
| | `outscale` | 1.0 | actor 마지막 layer 초기 스케일 |
| **critic** | `dist` | 'symlog_disc' | DiscDist 255-bin ([004 §1-2](004-dreamer_code_analysis_part2.md#1-2-discdist--twohot-인코딩-categorical-toolspy452-506)) |
| | `slow_target` | True | slow value 사용 |
| | `slow_target_update` | 1 | 매 step polyak |
| | `slow_target_fraction` | 0.02 | τ |
| | `lr`/`eps`/`grad_clip` | 3e-5 / 1e-5 / 100 | |
| | `outscale` | 0.0 | 초기 V≈0 |
| **reward_head** | `layers` | 2 | |
| | `dist` | 'symlog_disc' | DiscDist 255-bin |
| | `loss_scale` | 1.0 | |
| | `outscale` | 0.0 | 초기 reward≈0 |
| **cont_head** | `layers` | 2 | |
| | `loss_scale` | 1.0 | |
| | `outscale` | 1.0 | 초기 확률 0.5 |
| **WM 학습** | `dyn_scale`/`rep_scale` | 0.5 / 0.1 | KL balancing |
| | `kl_free` | 1.0 | free bits |
| | `weight_decay` | 0.0 | WD off |
| **데이터** | `batch_size`/`batch_length` | 16 / 64 | replay 샘플 |
| | `train_ratio` | 512 | env step당 train step 비율 — `Every(batch_steps/ratio) = Every(2)` |
| | `pretrain` | 100 | 첫 호출 burst step |
| | `model_lr` | 1e-4 | WorldModel lr |
| | `opt_eps`/`grad_clip` | 1e-8 / 1000 | WM 옵티마이저 |
| | `dataset_size` | 1e6 | replay 메모리 한도 (006 §2-6) |
| | `opt` | 'adam' | 옵티마이저 종류 |
| **behavior** | `discount` | 0.997 | γ |
| | `discount_lambda` | 0.95 | λ-return |
| | `imag_horizon` | 15 | imagination 길이 |
| | `imag_gradient` | 'dynamics' | continuous 기본. discrete는 'reinforce' |
| | `imag_gradient_mix` | 0.0 | 'both' 모드일 때 mix |
| | `eval_state_mean` | False | continuous 전용. True면 eval 시 stoch=mean |
| **exploration** | `expl_behavior` | 'greedy' | 'random'/'plan2explore' 가능 |
| | `expl_until` | 0 | 0이면 영구 explore (006 §1) |
| | `expl_extr_scale` | 0.0 | extrinsic 보너스 가중 |
| | `expl_intr_scale` | 1.0 | intrinsic 보너스 가중 |
| | `disag_target` | 'stoch' | ensemble target |
| | `disag_log` | True | log(disag) 보너스 |
| | `disag_models` | 10 | ensemble 크기 |
| | `disag_offset` | 1 | step offset |
| | `disag_layers`/`disag_units` | 4 / 400 | ensemble MLP |
| | `disag_action_cond` | False | action 조건 ensemble 여부 |

`grad_heads = ['decoder', 'reward', 'cont']` (configs.yaml:41) — WorldModel `_train`에서 어느 head로부터 representation까지 gradient를 흘릴지 ([005 §1-2](005-dreamer_code_analysis_part3.md#1-2-traindata--단일-forward에서-모든-loss-계산-modelspy108-174)).

---

## 3. 다음 단계 안내

5~6단계 (WorldModel/RewardEMA/ImagBehavior): [005](005-dreamer_code_analysis_part3.md).
2~4단계 (분포·RSSM·Encoder/Decoder): [004](004-dreamer_code_analysis_part2.md).
7~9단계 (Exploration·데이터·포팅·평가): [006](006-dreamer_code_analysis_part4.md).

---

## 4. 컨벤션 재확인

- `_thinking/analysis/`는 append-only — 본 감사 작업 한정 1회 예외.
- 한국어 응답.
- 코드 인용은 줄 링크만. 코드 블록 발췌 금지.
- 분석 대상은 **dreamer-v3 코드 자체**. F1TENTH 통합 코멘트 금지.
