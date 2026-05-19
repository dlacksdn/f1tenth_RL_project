# 004 - Dreamer-v3 코드 분석 (Part 2: RSSM + Encoder/Decoder)

> **목적**: `NM512/dreamerv3-torch` 코드 자체 이해. F1TENTH 통합 설계가 아니라 **알고리즘 코드 자체** 분석.
> **선행 문서**: [003-dreamer_code_analysis_part1.md](003-dreamer_code_analysis_part1.md) — 진입점·WorldModel·ImagBehavior 분석.
> **분석 대상**: `/home/dlacksdn/dreamerv3-torch/networks.py`
> **작성일**: 2026-05-20

---

## 0. 다음 세션 핸드오프 (READ FIRST)

### 현재까지 진행 상태
- ✅ 1단계: 진입점 + Config (`dreamer.py`, `configs.yaml`) — 003 문서
- ✅ 2단계: WorldModel + ImagBehavior (`models.py`) — 003 문서
- ✅ 3단계: **RSSM 동역학** (`networks.py`의 `RSSM`) — **이 문서 §1**
- ✅ 4단계: **Encoder/Decoder** (`MultiEncoder`/`MultiDecoder`/`ConvEncoder`/`ConvDecoder`/`MLP`) — **이 문서 §2**
- ⏭️ **5단계 (NEXT)**: ImagBehavior 세부 — λ-return, reward EMA, slow target self-distill. 003 §2-3에서 개요는 봤으니, 이번엔 `tools.lambda_return`/`models.RewardEMA`/`_update_slow_target`을 코드 레벨로 파고든다.

### 전체 로드맵 (003에서 정의, 변경 없음)
1. ✅ 진입점·실행 흐름 (`dreamer.py`)
2. ✅ WorldModel (`models.py`)
3. ✅ RSSM 동역학 (`networks.py`의 `RSSM`)
4. ✅ Encoder/Decoder (`networks.py`)
5. ⏭️ **ImagBehavior 세부** — λ-return, reward EMA, slow target self-distill, advantage 정규화
6. Heads & 분포 (`MLP` 본체는 이미 §2-5, 분포 종류는 `tools.py`의 `SymlogDist`/`DiscDist`/`OneHotDist`/twohot에서 깊이 보기)
7. Exploration (`exploration.py`) — `Plan2Explore`, `Random`
8. 데이터 파이프라인 (`tools.py`의 `load_episodes`/`sample_episodes`/`from_generator`/`simulate`)
9. 유틸: optimizer 래퍼, EMA, slow target, gradient clipping
- **제외**: `envs/`, `parallel.py`, `Dockerfile`, `requirements.txt`

### 파일 위치
```
/home/dlacksdn/dreamerv3-torch/
├── dreamer.py         365 lines  (✅ 003)
├── models.py          441 lines  (✅ 003)
├── networks.py        810 lines  (✅ 003+004)
├── tools.py          1000 lines  (5~9단계에서 부분별 분석)
├── exploration.py     135 lines  (7단계)
├── parallel.py        209 lines  (제외)
├── configs.yaml       184 lines  (✅ 003)
└── envs/                         (제외)
```

### 사용자 작업 컨벤션 (반드시 지킬 것)
- `_thinking/analysis/`는 **append-only**. 기존 파일 절대 수정 금지.
- 새 파일은 명시적 요청("N 문서로 저장")이 있을 때만 저장.
- **한글로 응답**.
- 코드 인용은 markdown link `[파일:라인](../../../dreamerv3-torch/파일#L라인)` 형식.
- 분석은 **dreamer-v3 코드 자체 이해**가 목적. F1TENTH 통합 얘기 섞지 말 것.
- 사용자 메시지 "진행해" = 다음 단계 진행 신호.

### 다음 에이전트 진입 명령 (그대로 사용)
> "`_thinking/analysis/004-dreamer_code_analysis_part2.md`의 핸드오프 섹션을 보고 5단계(ImagBehavior 세부)로 진행해. `tools.lambda_return`, `models.RewardEMA`, `ImagBehavior._update_slow_target`, `_compute_target`, `_compute_actor_loss`를 코드 레벨로 정리한 다음 사용자에게 결과를 보고하고 저장 요청을 기다려."

---

## 1. RSSM 동역학 (`networks.py`의 `RSSM`)

위치: [networks.py:13-290](../../../dreamerv3-torch/networks.py#L13-L290). Dreamer-v3의 심장 — observation을 latent state로 압축하고 action으로 미래를 예측하는 recurrent dynamics model.

### 1-1. State 구조와 차원

Latent state는 dict — `stoch`(확률적) + `deter`(결정적) 두 축으로 구성.

| 모드 | 키 | shape | 의미 |
|---|---|---|---|
| discrete (기본) | `stoch` | `[B, 32, 32]` | 32 카테고리 × 32 클래스 one-hot |
| discrete | `logit` | `[B, 32, 32]` | stoch 분포의 logit |
| continuous | `stoch` | `[B, 32]` | gaussian sample |
| continuous | `mean`, `std` | `[B, 32]` | Normal 파라미터 |
| 공통 | `deter` | `[B, 512]` | GRU hidden state |

`get_feat(state)` = `concat(stoch.flatten, deter)` ([L154-159](../../../dreamerv3-torch/networks.py#L154-L159)):
- discrete: `32*32 + 512 = 1536`
- continuous: `32 + 512 = 544`

### 1-2. `__init__`: 4개의 MLP + GRUCell ([L48-97](../../../dreamerv3-torch/networks.py#L48-L97))

```
_img_in_layers  : (prev_stoch ⊕ prev_action) → hidden        # img_step 진입
_cell (GRUCell) : (hidden, prev_deter)        → new_deter    # 결정적 진행
_img_out_layers : deter                       → hidden       # prior 분포 입력
_obs_out_layers : (deter ⊕ embed)             → hidden       # post  분포 입력
_imgs_stat_layer: hidden → logit or (mean,std)              # prior 분포 파라미터
_obs_stat_layer : hidden → logit or (mean,std)              # post  분포 파라미터
```

- 모든 Linear는 `bias=False` + 옵션 LayerNorm(`eps=1e-3`) + SiLU 활성.
- 백본은 `tools.weight_init`(Xavier-like), stat 레이어는 `uniform_weight_init(1.0)` — 초기 분포가 거의 균등해야 KL이 폭주하지 않음.
- **prior network와 post network는 별개의 stat_layer를 가짐** (공유 안 함).

**initial state** ([L99-125](../../../dreamerv3-torch/networks.py#L99-L125)):
- `initial='zeros'`: 전부 0.
- `initial='learned'`(기본): `self.W ∈ R^(1, deter)` 파라미터를 `tanh`하여 deter 초기값으로, stoch은 `get_stoch(deter)` (=prior network의 mode).

### 1-3. `img_step`: prior 한 스텝 ([L208-233](../../../dreamerv3-torch/networks.py#L208-L233))

관측 없이 (action만으로) 다음 latent 예측. **imagination rollout의 한 step**.

```
prev_stoch (flatten if discrete) ⊕ prev_action
  → _img_in_layers → hidden
  → GRUCell(hidden, prev_deter) → new_deter
  → _img_out_layers(new_deter) → hidden2
  → _imgs_stat_layer(hidden2) → stats (logit OR mean/std)
  → stoch = dist(stats).sample()  # straight-through 적용
prior = {stoch, deter=new_deter, **stats}
```

`rec_depth=1` 기본. 코드 코멘트 "rec depth is not correctly implemented" — 여러 step 굴리고 싶다면 주의.

**GRUCell** ([L742-768](../../../dreamerv3-torch/networks.py#L742-L768)):
- 3개 게이트(reset/cand/update)를 한 Linear로 합쳐 계산 + LayerNorm.
- `update_bias=-1` (초기에 state 보존 쪽으로 편향) → 학습 초기 안정성.

### 1-4. `obs_step`: post 한 스텝 ([L174-206](../../../dreamerv3-torch/networks.py#L174-L206))

관측 embed를 반영한 latent.

```
1. is_first 처리:
   - 전부 첫 스텝이면 prev_state = initial(B), prev_action = 0
   - 일부면 element-wise mask: prev_state[k] = val*(1-is_first) + initial[k]*is_first
   - prev_action도 마스킹 (첫 스텝의 action은 0)
2. prior = img_step(prev_state, prev_action)
3. x = concat(prior["deter"], embed)
   → _obs_out_layers → hidden
   → _obs_stat_layer → stats
   → stoch = dist(stats).sample()
4. post = {stoch, deter=prior["deter"], **stats}
   return post, prior
```

**핵심: prior와 post는 `deter`를 공유하고 `stoch`만 다름**. prior는 deter만으로 stoch 분포를, post는 (deter ⊕ embed)로 분포를 만든다 → post가 더 정확, prior는 환경 없이 굴릴 수 있어 imagination에 쓰임.

### 1-5. `observe`: 시퀀스 전체 ([L127-143](../../../dreamerv3-torch/networks.py#L127-L143))

배치 시퀀스 `[B,T,...]` → time-major로 swap → `tools.static_scan`으로 T step 펼치며 `obs_step` 반복. 각 step에서 `prev_state[0]` (= 직전 post)를 사용.

`tools.static_scan` ([tools.py:795](../../../dreamerv3-torch/tools.py#L795)): JAX `lax.scan`의 단순 구현 — 매 step 결과를 dict별로 `unsqueeze(0)` 후 `torch.cat` 누적. 컴파일 효율은 미흡하나 동작은 명료.

`imagine_with_action` ([L145-152](../../../dreamerv3-torch/networks.py#L145-L152)): action 시퀀스만 받아 `img_step`만 반복 → prior trajectory. 학습 imagination에서는 action을 매 step `actor(feat)`로 새로 뽑으므로 이 함수는 평가/시각화 용도.

### 1-6. 분포: `_suff_stats_layer` + `get_dist` ([L161-172, L241-270](../../../dreamerv3-torch/networks.py#L161-L270))

**discrete**:
- stat layer 출력 `[..., stoch*discrete]` → `[..., 32, 32]` reshape → `logit`.
- `OneHotDist(logit, unimix_ratio=0.01)` → `Independent(..., 1)` (32 카테고리를 독립).
- **unimix**: `probs = (1-α)·softmax + α/K` (균등 분포 1% 혼합) → 0 확률 방지, exploration 안정화.
- **straight-through**: `sample = onehot.detach() + probs - probs.detach()` ([tools.py:441-449](../../../dreamerv3-torch/tools.py#L441-L449)) → forward는 one-hot, backward는 probs로 gradient.

**continuous**:
- stat layer 출력 split → `mean`, raw_std.
- `mean_act`: `'none'` 또는 `'tanh5'` (=`5·tanh(x/5)` — 부드러운 클리핑).
- `std_act`: `'softplus'`(기본) / `'abs'` / `'sigmoid'` / `'sigmoid2'`.
- `std = std_act(raw) + min_std(=0.1)` — 분산 하한.
- `ContDist(Independent(Normal(mean,std), 1))` (32차원 독립 정규).

### 1-7. `kl_loss`: KL balancing 본체 ([L272-290](../../../dreamerv3-torch/networks.py#L272-L290))

```python
sg = stop-gradient
rep_loss = KL( q(post)        ||  q(sg(prior)) )   # post가 prior에 끌려감
dyn_loss = KL( q(sg(post))    ||  q(prior)     )   # prior가 post에 끌려감
rep_loss = clip(rep_loss, min=free)                # free bits (kl_free=1.0)
dyn_loss = clip(dyn_loss, min=free)
loss = dyn_scale·dyn_loss + rep_scale·rep_loss     # 0.5·dyn + 0.1·rep
```

- **두 방향을 분리 가중**: prior 학습(0.5) 강하게, representation 학습(0.1) 약하게 → encoder가 prior의 부정확함에 끌려가는 것을 방지.
- **free bits**: KL이 1.0 nat 미만이면 gradient 0 → "이미 충분히 가까우면 더 줄이지 마라".
- discrete면 `dist(x)` 자체에, continuous면 `_dist`(원본 Normal)에 PyTorch `kl_divergence`.

### 1-8. 설계 포인트

1. **prior와 post가 deter 공유** — GRU가 결정적 추세, stoch이 잔여 불확실성을 담당. 이 분리가 KL balancing을 의미있게 만듦.
2. **discrete latent(32×32) + unimix 0.01** — Dreamer-v3가 v2 대비 categorical로 전환한 핵심. 32^32 표현력 + 닫힌형 KL.
3. **Straight-through estimator** — discrete sample에 path gradient를 흘려 actor가 RSSM을 통해 backprop 받게 함.
4. **learned initial deter** — `tanh(W)`만 학습. 작아도 충분.
5. **is_first 마스킹이 obs_step 내부** — 시퀀스가 에피소드 경계를 넘으면 자동으로 state 초기화. batch_length=64로 잘라도 안전.
6. **kl_free=1.0** — v3의 안정화 기법.
7. **prior/post stat_layer 분리** — 가중치 공유 안 함. 입력이 다르므로 별도 학습.

---

## 2. Encoder/Decoder (`networks.py`)

### 2-1. MultiEncoder ([L293-357](../../../dreamerv3-torch/networks.py#L293-L357))

dict 형태의 obs를 받아 단일 embedding 벡터로 압축.

**shape 자동 분기** ([L309-322](../../../dreamerv3-torch/networks.py#L309-L322)):
- `excluded = {is_first, is_last, is_terminal, reward}` + `log_*` prefix → encoder 입력에서 제외.
- 나머지 키를 정규식 `cnn_keys`/`mlp_keys`로 매칭 (기본: `cnn_keys='image'`, `mlp_keys='$^'`=빈 매치).
- `len(shape)==3` 이고 cnn 매치 → CNN 경로
- `len(shape) in (1,2)` 이고 mlp 매치 → MLP 경로

**조립** ([L326-346](../../../dreamerv3-torch/networks.py#L326-L346)):
- CNN이 있으면 모든 cnn 입력을 채널축으로 concat → `ConvEncoder` 하나로 처리. 입력 shape는 `cnn_shapes` 중 아무 거나의 H×W 사용 (모두 같은 해상도 가정).
- MLP가 있으면 모든 mlp 입력을 마지막 축으로 concat → `MLP`(`shape=None`, `symlog_inputs=True`) 하나로 처리.
- 두 출력은 마지막 축에서 concat → `embed = [B, T, outdim]`.

**outdim**: `cnn.outdim + mlp_units` (MLP가 있으면 mlp_units=1024 추가). `cnn.outdim`은 ConvEncoder 마지막 stage 채널×최종 H×W.

### 2-2. ConvEncoder ([L448-496](../../../dreamerv3-torch/networks.py#L448-L496))

이미지 인코더. **stride-2 conv를 `log2(H/minres)`번 쌓아 H→minres**로 다운샘플.

```
input: (B, T, H, W, C)  # H=W=64 가정, minres=4 → 4 stages
- 채널 dim: depth(32) → 64 → 128 → 256
- 공간 dim: 64 → 32 → 16 → 8 → 4
- 각 stage: Conv2dSamePad(stride=2, k=4, bias=False) + ImgChLayerNorm + SiLU
```

세부:
- **`Conv2dSamePad`** ([L771-798](../../../dreamerv3-torch/networks.py#L771-L798)): TF의 `padding='SAME'`을 PyTorch에서 흉내. forward 시 입력 크기 기반으로 `F.pad` 후 conv. (PyTorch 내장 padding은 입력 의존이 안 됨.)
- **`ImgChLayerNorm`** ([L801-810](../../../dreamerv3-torch/networks.py#L801-L810)): `(B, C, H, W) → permute → LayerNorm(C) → permute` — 채널 축에 LayerNorm. 일반 BatchNorm 대신 사용 → 배치 통계 의존 제거.
- forward에서 **`obs -= 0.5`** ([L487](../../../dreamerv3-torch/networks.py#L487)): preprocess에서 `/255` 한 입력을 `[-0.5, 0.5]`로 중심화.
- 마지막에 `(B*T, C, h, w) → flatten → (B, T, outdim)`. 코드의 `out_dim // 2 * h * w`는 마지막 stage가 끝난 뒤 in_dim 기준(루프 끝에서 한 번 더 `*=2` 되었기 때문).

### 2-3. MultiDecoder ([L360-446](../../../dreamerv3-torch/networks.py#L360-L446))

feat → obs(dict) 재구성 분포.

- `excluded = {is_first, is_last, is_terminal}` (reward는 별도 head라서 여기 안 옴 — 정확히는 reward는 `is_first/last/terminal`과 달리 obs_space에 안 들어가 있을 수도 있는데, MultiDecoder는 어쨌든 `is_*`만 제외).
- cnn/mlp 키 분리는 Encoder와 동일.
- **CNN 경로**: `ConvDecoder(feat_size, shape=(sum_channels, H, W), ...)` 하나로 전체 cnn 키를 동시 디코드 → 마지막에 채널 축에서 split → 각 키별로 `_make_image_dist` 적용.
- **MLP 경로**: `MLP(feat_size, shape=mlp_shapes, dist=vector_dist, ...)` — `shape`이 dict면 MLP 내부에서 각 키별 mean_layer를 만들어 `dict[name → Dist]` 반환 ([L638-642, L665-674](../../../dreamerv3-torch/networks.py#L638-L674)).

**이미지 분포 종류** ([L438-445](../../../dreamerv3-torch/networks.py#L438-L445)):
- `'normal'`: `Independent(Normal(mean, 1), 3)` — 표준편차 1 고정.
- `'mse'` (기본): `tools.MSEDist(mean)` — `log_prob = -(x-mean)^2.sum`, 즉 가우시안 log-prob에서 상수 제외. **기본값이 mse**라서 reconstruction loss는 실질적으로 픽셀 MSE.

**configs 기본** ([configs.yaml:45-48](../../../dreamerv3-torch/configs.yaml#L45-L48)):
```yaml
encoder: {mlp_keys: '$^', cnn_keys: 'image', cnn_depth: 32,
          kernel_size: 4, minres: 4, mlp_layers: 5, mlp_units: 1024, symlog_inputs: True}
decoder: {mlp_keys: '$^', cnn_keys: 'image', cnn_depth: 32,
          kernel_size: 4, minres: 4, mlp_layers: 5, mlp_units: 1024,
          cnn_sigmoid: False, image_dist: mse, vector_dist: symlog_mse, outscale: 1.0}
```
- Atari/DMC/Crafter/Minecraft 같은 비전 환경은 위 정규식 그대로.
- 벡터 전용 환경(`dmc_proprio` 등)은 `mlp_keys: '.*'`, `cnn_keys: '$^'`로 override.

### 2-4. ConvDecoder ([L499-585](../../../dreamerv3-torch/networks.py#L499-L585))

ConvEncoder의 거울상. **feat → Linear → reshape → ConvTranspose2d 스택**.

```
feat (B, T, 1536)
  → Linear(1536 → minres² · depth · 2^(L-1))   # 예: 4²·32·2³ = 4096
  → reshape (B*T, minres, minres, embed/minres²)  # (B*T, 4, 4, 256)
  → permute → (B*T, 256, 4, 4)
  → L stages of ConvTranspose2d(stride=2, k=4) + ImgChLayerNorm + SiLU
    채널: 256 → 128 → 64 → out_ch(=image channels)
    공간: 4 → 8 → 16 → 32 → 64
  → permute & reshape → (B, T, H, W, C)
  → +0.5 (또는 sigmoid)
```

세부:
- `calc_same_pad` ([L562-566](../../../dreamerv3-torch/networks.py#L562-L566)): ConvTranspose 출력 크기를 `2*input`으로 맞추기 위한 padding/output_padding 계산.
- **마지막 stage만 다름** ([L530-534](../../../dreamerv3-torch/networks.py#L530-L534)): `out_dim = self._shape[0]`(채널 = sum of cnn_shape channels), `act=False`, `norm=False`, `bias=True`. → 최종 픽셀 출력은 raw.
- 마지막 layer는 `uniform_weight_init(outscale)`로 작은 초기값 → 디코더 출력이 0 근처에서 시작 (학습 초기 안정).
- `cnn_sigmoid=False`(기본)면 `mean += 0.5` (preprocess에서 뺐던 0.5 복구), `True`면 `sigmoid`.

### 2-5. MLP ([L588-739](../../../dreamerv3-torch/networks.py#L588-L739))

Dreamer-v3의 **만능 MLP** — encoder의 vector 입력, decoder의 vector 출력, reward head, cont head, value head, actor 모두 이 클래스 한 개로 처리.

**`__init__`**:
- 백본: `[Linear(bias=False) + LayerNorm + act] × layers`. 모든 Linear는 `tools.weight_init`.
- `shape` 파라미터로 출력 모드 분기:
  - `shape=None`: encoder MLP — `forward`가 `out` 그대로 반환 (분포 X). ([L663-664](../../../dreamerv3-torch/networks.py#L663-L664))
  - `shape=int or tuple`: 단일 출력 head — `mean_layer = Linear(inp, prod(shape))`, optional `std_layer`.
  - `shape=dict`: 여러 출력 head — 각 키마다 `mean_layer[name]` (decoder vector 경로).
- 출력 head Linear는 `uniform_weight_init(outscale)`. value/critic은 `outscale=0.0` 으로 거의 0 초기화 → 학습 시작 시 V≈0.

**`symlog_inputs=True`** ([L659-660](../../../dreamerv3-torch/networks.py#L659-L660)): encoder의 MLP 경로에서만 사용. 입력을 `symlog(x) = sign(x)·log(1+|x|)`로 압축 → 큰 값(예: F1TENTH LiDAR raw distance)도 안정.

**`dist` 종류** ([L683-739](../../../dreamerv3-torch/networks.py#L683-L739)) — Dreamer-v3가 동원하는 분포 풀세트:

| 이름 | 본체 | 용도 |
|---|---|---|
| `'normal'` | `Normal(tanh(mean), sigmoid_std)` + Independent | **actor (continuous)**. `tanh(mean)`으로 액션 범위 [-1,1] 강제 |
| `'normal_std_fixed'` | `Normal(mean, fixed_std)` + Independent | std 고정 |
| `'tanh_normal'` | `TransformedDist(Normal, TanhBijector)` + SampleDist | SAC 스타일 액션. 잘 안 씀 |
| `'trunc_normal'` | `SafeTruncatedNormal(tanh(mean), std, -1, 1)` | 잘림 정규 |
| `'onehot'` | `OneHotDist(mean, unimix=0.01)` | **actor (discrete)** |
| `'onehot_gumble'` | `Gumbel(mean, 1/temp)` | 사용 안 함 (기본) |
| `'huber'` | `Independent(UnnormalizedHuber(mean, std, 1.0), len(shape))` | huber loss로 학습 |
| `'binary'` | `Independent(Bernoulli(logits=mean), len(shape))` | **cont head** (continue 확률) |
| `'symlog_disc'` | `tools.DiscDist(logits=mean)` 255-bin twohot | **reward head, value head** |
| `'symlog_mse'` | `tools.SymlogDist(mean)` | **decoder vector head**. `log_prob = -(symlog(x) - mean)^2` |

**연결 요약** (003 §2 + 본 §2-3과 교차 확인):
- WorldModel 내부에서 [models.py:69-83](../../../dreamerv3-torch/models.py#L69-L83):
  - `heads['decoder'] = MultiDecoder(..., image_dist='mse', vector_dist='symlog_mse')`
  - `heads['reward'] = MLP(feat, [], 2 layers, dist='symlog_disc', outscale=0.0)`
  - `heads['cont']   = MLP(feat, [], 2 layers, dist='binary')`
- ImagBehavior [models.py:225-256](../../../dreamerv3-torch/models.py#L225-L256):
  - `actor = MLP(feat, num_actions, layers=2, dist='normal'|'onehot', std='learned')`
  - `value = MLP(feat, [], 2 layers, dist='symlog_disc', outscale=0.0)`

### 2-6. symlog 변환 ([tools.py:23-29](../../../dreamerv3-torch/tools.py#L23-L29))

```python
symlog(x)  = sign(x) * log(1 + |x|)
symexp(x)  = sign(x) * (exp(|x|) - 1)
```
- encoder MLP 입력에 적용 → 입력 스케일 정규화.
- `DiscDist`/`SymlogDist`에서 target을 symlog 공간으로 옮겨 학습, 추론 시 `symexp`로 복원.
- 결과: reward/value/관측이 작든 크든 균등하게 다뤄짐. Dreamer-v3가 환경 독립적 하이퍼파라미터를 가능하게 만든 핵심 트릭.

### 2-7. 데이터 흐름 한눈에

```
obs(dict) ──┬─ cnn_keys → ConvEncoder ──┐
            └─ mlp_keys → MLP(symlog) ──┴─→ embed [B,T,outdim]

embed + action → RSSM.observe → post, prior (003 §2 참고)

feat = get_feat(post) ──┬─→ MultiDecoder
                        │     ├─ ConvDecoder → image_dist (mse)
                        │     └─ MLP(dict)    → vector_dist (symlog_mse)
                        ├─→ reward MLP (symlog_disc, 255 bins)
                        └─→ cont   MLP (binary, Bernoulli)
```

### 2-8. 설계 포인트

1. **정규식으로 키 라우팅** (`cnn_keys`/`mlp_keys`) → 환경 obs space에 새 키가 들어와도 config만 바꾸면 인코더가 자동 분기.
2. **CNN/MLP 모두 LayerNorm** — 배치 통계 의존 제거, RL의 비정상 분포에 강함.
3. **`Conv2dSamePad` + `ImgChLayerNorm`** — TF 스타일 그대로 PyTorch에 이식. 마이그레이션 산물.
4. **image_dist 기본이 `mse`** — Normal(1) 대신 MSE만 쓰면 픽셀 단위에서 동일하지만 상수 제외로 약간 빠름.
5. **decoder MLP는 dict shape 지원** → 여러 vector head를 한 MLP body로 공유, 마지막만 키별 분기.
6. **MLP 클래스 하나가 거의 모든 head** — `dist` 인자로 11종을 스위치. 라이브러리화의 좋은 사례.
7. **symlog 입력 + symlog_mse/symlog_disc 출력** — Dreamer-v3가 "환경마다 튜닝 안 해도 됨"의 핵심 메커니즘. encoder MLP, vector decoder, reward, value 모두 symlog 공간에서 학습.
8. **value head `outscale=0.0`** — Linear 마지막 layer를 작게 초기화. 학습 초기 V≈0이라 부트스트랩이 불안하지 않음.

---

## 3. 다음 세션이 바로 시작할 작업 (5단계: ImagBehavior 세부)

003 §2-3에서 개요는 봤다. 이번 단계의 목표는 **숫자가 어디서 어떻게 흐르는지 코드 줄 단위로** 보기.

### 분석 항목
1. **`models.RewardEMA`** ([models.py:11-26](../../../dreamerv3-torch/models.py#L11-L26)) — 5/95 quantile EMA. 어떤 momentum으로 갱신되고 어디서 호출되는가.
2. **`ImagBehavior._update_slow_target`** — polyak EMA `mix=slow_target_fraction=0.02`. `slow_target_update=1` (매 step) vs 다른 환경 차이.
3. **`ImagBehavior._imagine`** ([models.py:351-369](../../../dreamerv3-torch/models.py#L351-L369)) — `feat.detach()` 위치, `static_scan` 사용, action sampling 디테일.
4. **`ImagBehavior._compute_target`** ([models.py:371-389](../../../dreamerv3-torch/models.py#L371-L389)) — `discount = γ · cont.mean`, `tools.lambda_return` 호출, `weights = cumprod(discount)`.
5. **`tools.lambda_return`** (`tools.py`에서 라인 찾을 것) + 보조 `static_scan_for_lambda_return` ([tools.py:671](../../../dreamerv3-torch/tools.py#L671)).
6. **`ImagBehavior._compute_actor_loss`** ([models.py:391-433](../../../dreamerv3-torch/models.py#L391-L433)) — `imag_gradient` 3가지 모드 (`dynamics`/`reinforce`/`both`), `RewardEMA` 적용 위치, entropy bonus 계산.
7. **value loss 두 항** ([models.py:324-332](../../../dreamerv3-torch/models.py#L324-L332)) — `-log_prob(target)` vs `-log_prob(slow.mode())`, 두 항이 동등하게 합산되는지 가중되는지.
8. **옵티마이저 호출 흐름** — `_actor_opt`/`_value_opt`가 `tools.Optimizer` 래퍼인지, gradient clipping/AMP/EMA 어떻게 묶이는지.

### 작업 흐름 (권장)
1. `models.py:218-441` 다시 한번 정독.
2. `tools.py`에서 `lambda_return`, `static_scan_for_lambda_return`, `Optimizer` 찾아 발췌.
3. 위 항목들을 본 문서와 동일 형식(섹션 헤더, 표, 코드 블록, [파일:라인](링크))으로 정리.
4. 사용자에게 결과 보고 → "005 문서로 저장" 요청 받으면 `005-dreamer_code_analysis_part3.md` 생성.

---

## 4. 컨벤션 재확인

- `_thinking/`은 append-only. 기존 문서 절대 수정 금지.
- 명시적 "N 문서로 저장" 요청 있을 때만 새 파일 작성.
- 한글 응답.
- "진행해" = 다음 단계 진행.
- 분석 대상은 **dreamer-v3 코드 자체**. F1TENTH 통합 얘기 섞지 말 것.
