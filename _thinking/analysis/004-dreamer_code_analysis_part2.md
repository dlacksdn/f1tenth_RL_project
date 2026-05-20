# 004 - Dreamer-v3 코드 분석 (Part 2: 분포 카탈로그 + RSSM + Encoder/Decoder)

> **Revision**: 2026-05-20 — 4차원 진정성 감사 결과 직접 반영. 005 §2(분포 카탈로그)를 본 문서 §1로 이전. RSSM·Encoder/Decoder 본문 검증·압축. 변경 내역은 [008](008-audit-changelog.md) 참조.
> **목적**: `NM512/dreamerv3-torch` 코드 자체 이해. F1TENTH 통합 설계가 아니라 **알고리즘 코드 자체** 분석.
> **선행 문서**: [003-dreamer_code_analysis_part1.md](003-dreamer_code_analysis_part1.md) — 진입점·Config.
> **분석 대상**: `/home/dlacksdn/dreamerv3-torch/networks.py`, `tools.py`.
> **인용 규약**: 줄 링크 `(파일#L<a>-L<b>)` 만. 코드 블록 발췌 없음.

---

## 0. 핸드오프

### 현재까지 진행 상태
- ✅ 1단계 진입점·Config — [003](003-dreamer_code_analysis_part1.md)
- ✅ 2단계 **분포 카탈로그** — 본 문서 §1
- ✅ 3단계 **RSSM** — 본 문서 §2
- ✅ 4단계 **Encoder/Decoder** — 본 문서 §3
- ⏭️ **다음**: [005](005-dreamer_code_analysis_part3.md) — WorldModel + ImagBehavior 통합

### 전체 로드맵 (감사 후 갱신)
1. ✅ 진입점·Config (`dreamer.py`, `configs.yaml`) — 003
2. ✅ 분포 카탈로그 (`tools.py`) — 본 문서 §1
3. ✅ RSSM 동역학 (`networks.py`의 `RSSM`) — 본 문서 §2
4. ✅ Encoder/Decoder (`networks.py`) — 본 문서 §3
5. ⏭️ WorldModel + RewardEMA + ImagBehavior 통합 (`models.py`) — 005
6. ⏭️ Exploration + 데이터 + 포팅 + 평가 — 006

### 컨벤션
- 한국어 응답.
- 코드 인용은 줄 링크만. 코드 블록 ``` ``` 발췌 금지. 의사코드·수식 인라인은 허용.
- 분석 대상은 **dreamer-v3 코드 자체**. F1TENTH 통합 코멘트 금지.
- 본 감사 작업 한정으로 003~006 직접 수정 허용 (1회 예외).

---

## 1. 분포 카탈로그 (`tools.py`)

Dreamer-v3가 동원하는 분포 클래스들. **이 §은 003·005에서 등장하는 모든 head 분포의 정식 정의 위치**다. 다른 § 본문은 "분포 X는 §1-Y 참조" 형식으로 단축한다.

### 1-1. symlog / symexp ([tools.py:23-28](../../../dreamerv3-torch/tools.py#L23-L28))

부호 보존 로그 변환.

| 함수 | 정의 | 성질 |
|---|---|---|
| `symlog(x)` | sign(x) · log(1 + \|x\|) | 미분 `1/(1+\|x\|)`, 큰 \|x\|에서 gradient 둔감 |
| `symexp(x)` | sign(x) · (exp(\|x\|) − 1) | `symexp(symlog(x)) = x` 정확 역함수 |

`symlog(0)=0`, 단조증가, 원점에서 매끄러움. Dreamer-v3는 세 곳에서 사용:
1. encoder MLP 입력 ([networks.py:659-660](../../../dreamerv3-torch/networks.py#L659-L660))
2. `SymlogDist` (vector decoder 출력) — §1-3
3. `DiscDist` (reward/value head) — §1-2

핵심 효과: 환경마다 다른 reward/value/관측 스케일을 단일 hyperparameter로 흡수. v3가 "환경 독립 hyperparameter"를 달성한 메커니즘의 핵심.

### 1-2. DiscDist — twohot 인코딩 categorical ([tools.py:452-506](../../../dreamerv3-torch/tools.py#L452-L506))

스칼라 회귀를 **255-bin 분류**로 바꾼다. reward head, value head 둘 다 이 분포.

**bucket 구성** ([tools.py:464-465](../../../dreamerv3-torch/tools.py#L464-L465)):
- `buckets = linspace(-20.0, 20.0, steps=255)` — **symlog 공간**의 [-20, 20].
- `width = (buckets[-1] - buckets[0]) / 255 = 40/255 ≈ 0.157`.
- symexp 복원 시 원래 스케일 [−exp(20)+1, exp(20)−1] ≈ ±4.85e8 커버.

**mode() / mean()** ([tools.py:469-475](../../../dreamerv3-torch/tools.py#L469-L475)):
- 두 메서드 본문 동일 — `expected_symlog = Σ_i probs_i · buckets_i` 후 `symexp(expected_symlog)`.
- 진짜 mode가 아니라 expectation을 점추정으로 사용. categorical 회귀의 "soft mode".

**log_prob(x) — twohot 인코딩 본체** ([tools.py:478-502](../../../dreamerv3-torch/tools.py#L478-L502)):
1. `x ← symlog(x)`: target을 symlog 공간으로.
2. `below = (buckets <= x).sum(-1) - 1`: x를 둘러싸는 두 bin 중 작은 쪽 인덱스.
3. `above = total - (buckets > x).sum(-1)`: 큰 쪽 인덱스. `equal = (below == above)` (x가 정확히 bucket 위에 있을 때).
4. `below`/`above`를 `[0, 254]`로 clip — out-of-range target은 양 끝 bin에 모임.
5. 거리 가중치 계산: `weight_below = dist_to_above / total`, `weight_above = dist_to_below / total` — **가까운 bin이 큰 가중치**.
6. `target = onehot(below)·w_below + onehot(above)·w_above` — 두 인접 bin에만 질량 분배.
7. `target.squeeze(-2)`: value/reward는 마지막 차원 1을 가지므로 `x[..., None]` broadcast 후 추가된 dummy 차원 제거.
8. `log_pred = log_softmax(logits)`.
9. 반환 `(target * log_pred).sum(-1)` — cross-entropy.

**왜 twohot인가**: 표준 one-hot은 가까운 두 bin 사이의 부드러운 회귀 신호를 못 준다. twohot은 거리 가중치로 정확히 두 bin에 질량을 나누어 회귀와 분류의 장점을 결합. C51(distributional RL)의 projection과 같은 아이디어.

**out-of-range 안전장치**: target이 [-20, 20] 밖이면 clip이 끝 bin에 몰아넣어 학습 발산 방지.

### 1-3. SymlogDist — vector reconstruction ([tools.py:532-561](../../../dreamerv3-torch/tools.py#L532-L561))

MultiDecoder vector head 전용. **분포가 아니라 MSE의 symlog 버전을 분포 인터페이스로 포장**.

- `mode() / mean() = symexp(self._mode)` — `_mode`는 Linear 출력(symlog 공간 raw).
- `log_prob(value)`: `distance = (mode - symlog(value))^2`, `where(distance < tol=1e-8, 0, distance)` (AMP underflow 방지), 마지막 차원들 sum/mean. `-loss` 반환.
- `dist='abs'`/`dist='mse'` 두 옵션 — 기본 `'mse'`.
- 정규화 상수·표준편차 없음 → maximum likelihood 동치라기보다 **scale-aware MSE 손실**의 분포 래퍼.

### 1-4. MSEDist — image reconstruction ([tools.py:509-529](../../../dreamerv3-torch/tools.py#L509-L529))

`SymlogDist`에서 symlog만 제거. `image_dist='mse'` (기본)에서 사용.
- `log_prob(value) = -((mode - value)^2).sum(축 ≥ 2)`.
- 이미지 `[B, T, H, W, C]`는 축 2,3,4 sum → `[B, T]` loss.
- 이미지는 ConvEncoder에서 `obs -= 0.5`로 이미 `[-0.5, 0.5]` 정규화 후라 symlog 불필요.

### 1-5. OneHotDist — straight-through categorical ([tools.py:425-449](../../../dreamerv3-torch/tools.py#L425-L449))

`torchd.OneHotCategorical` 상속. RSSM stoch(discrete), actor(discrete action)에서 사용.

**unimix** ([tools.py:427-431](../../../dreamerv3-torch/tools.py#L427-L431)):
- `unimix_ratio > 0`이면 `probs = softmax(logits)·(1-α) + α/K`, 다시 `logits = log(probs)`.
- α=0.01 기본 → 모든 카테고리에 최소 0.01/K 확률 보장. log(0) 회피 + dead exploration 차단 + KL divergence 안정.

**Straight-through estimator**:
- `sample`: `s = super().sample().detach(); s += probs - probs.detach()` — forward는 one-hot, backward gradient는 probs로.
- `mode`: 동일 패턴, `argmax → one_hot` 후 STE.
- `a + b - b.detach()` 트릭은 PyTorch STE 표준. discrete RSSM stoch에 path gradient를 흘리는 핵심.

**RSSM wrapping**: `Independent(OneHotDist(logit, unimix=0.01), 1)` ([networks.py:163-166](../../../dreamerv3-torch/networks.py#L163-L166)) — `[stoch=32, discrete=32]` 모양에서 마지막 축을 categorical로, 그 앞 32 그룹을 독립으로 reinterpret. KL/log_prob이 32개 카테고리의 합으로 계산.

### 1-6. ContDist — continuous actor 래퍼 ([tools.py:564-590](../../../dreamerv3-torch/tools.py#L564-L590))

`Normal` 또는 `Independent(Normal, k)`를 감싸 `absmax` 제약 추가.

- `mode = self._dist.mean`, `absmax`가 있으면 `mode *= (absmax / clip(|mode|, min=absmax)).detach()` — `|mode| > absmax`이면 부드럽게 스케일 다운, gradient는 mean으로만 흐름.
- `sample`은 `rsample` 사용 → reparameterization gradient 활성.
- 액션 박스 [-1, 1] 강제는 tanh 포화 없이 ContDist에서 처리: `absmax=1.0` ([models.py:239](../../../dreamerv3-torch/models.py#L239)).

### 1-7. Bernoulli — cont head 전용 래퍼 ([tools.py:593-617](../../../dreamerv3-torch/tools.py#L593-L617))

WorldModel의 continue head가 사용. continue 확률 = 에피소드 종료 안 함 확률.

- `mode = round(self._dist.mean)` + STE.
- `log_prob(x)`: `softplus` 기반 BCE — `log_p0 = -softplus(logits)`, `log_p1 = -softplus(-logits)`. 직접 `log(p)`를 쓰면 logits가 클 때 overflow. softplus 형식이 항상 안정.
- `.mean`은 `sigmoid(logits)` = 확률. ImagBehavior `_compute_target`이 `cont_head(inp).mean`을 discount 계수로 사용.

### 1-8. 기타 분포 (옵션 또는 보조)

| 분포 | 위치 | 용도 / 기본 사용 여부 |
|---|---|---|
| `UnnormalizedHuber` | [tools.py:620-631](../../../dreamerv3-torch/tools.py#L620-L631) | pseudo-Huber. `log_prob = -(√((x-μ)² + threshold²) - threshold)`. `dist='huber'` 옵션. 기본 config 미사용 |
| `SafeTruncatedNormal` | [tools.py:634-649](../../../dreamerv3-torch/tools.py#L634-L649) | `trunc_normal` actor. sample 후 `[low+ε, high-ε]` STE clip. `_mult` 스케일 조정. 기본 미사용 |
| `TanhBijector` | [tools.py:652-668](../../../dreamerv3-torch/tools.py#L652-L668) | SAC식 `tanh_normal` actor. inverse에서 `clamp(±0.99999997)`로 NaN 방지 |
| `SampleDist` | [tools.py:398-422](../../../dreamerv3-torch/tools.py#L398-L422) | analytic mean/mode 없는 분포(tanh_normal 등)용. N=100 샘플 추정. 기본 미사용 |

### 1-9. head ↔ 분포 매핑 표

`MLP.dist(...)` ([networks.py:683-738](../../../dreamerv3-torch/networks.py#L683-L738))가 dispatcher.

| Head | dist 키 | 분포 class | log_prob target | mode 출력 | 사용 위치 |
|---|---|---|---|---|---|
| image decoder | `'mse'` | `MSEDist` | raw pixel | raw + 0.5 | [networks.py:443-444](../../../dreamerv3-torch/networks.py#L443-L444) |
| vector decoder | `'symlog_mse'` | `SymlogDist` | symlog(x) | symexp(logits) | [networks.py:735-736](../../../dreamerv3-torch/networks.py#L735-L736) |
| reward head | `'symlog_disc'` | `DiscDist` | twohot(symlog(r)) | symexp(Σ p·b) | [networks.py:733-734](../../../dreamerv3-torch/networks.py#L733-L734) |
| value head | `'symlog_disc'` | `DiscDist` | twohot(symlog(V)) | symexp(Σ p·b) | 동일 |
| cont head | `'binary'` | `Bernoulli` | BCE(continue) | round + STE | [networks.py:727-732](../../../dreamerv3-torch/networks.py#L727-L732) |
| RSSM stoch (discrete) | — | `Independent(OneHotDist, 1)` | KL between post/prior | one_hot + STE | [networks.py:163-166](../../../dreamerv3-torch/networks.py#L163-L166) |
| RSSM stoch (continuous) | — | `ContDist(Independent(Normal, 1))` | KL Normal | mean (clipped) | [networks.py:167-171](../../../dreamerv3-torch/networks.py#L167-L171) |
| actor (continuous) | `'normal'` | `ContDist(Independent(Normal,1), absmax=1)` | log Normal | tanh(mean), absmax=1 | [networks.py:693-700](../../../dreamerv3-torch/networks.py#L693-L700) |
| actor (discrete) | `'onehot'` | `OneHotDist` | log onehot | argmax + STE | [networks.py:713-714](../../../dreamerv3-torch/networks.py#L713-L714) |
| (옵션) | `'tanh_normal'` | `SampleDist(TransformedDistribution)` | log Normal + tanh jacobian | sample est | [networks.py:684-692](../../../dreamerv3-torch/networks.py#L684-L692) |
| (옵션) | `'trunc_normal'` | `SafeTruncatedNormal` | log truncated | tanh(mean) | [networks.py:706-712](../../../dreamerv3-torch/networks.py#L706-L712) |
| (옵션) | `'huber'` | `UnnormalizedHuber` in ContDist | pseudo-Huber | mean | [networks.py:719-726](../../../dreamerv3-torch/networks.py#L719-L726) |
| (옵션) | `'normal_std_fixed'` | `Normal(mean, fixed_std)` in ContDist | log Normal | mean | [networks.py:701-705](../../../dreamerv3-torch/networks.py#L701-L705) |
| (옵션) | `'onehot_gumble'` | `ContDist(Gumbel)` | — | sample | [networks.py:715-718](../../../dreamerv3-torch/networks.py#L715-L718) |

### 1-10. 설계 포인트 (분포)

1. **회귀 → 분류 변환 (DiscDist twohot)**: reward·value 스케일이 환경마다 천차만별이라 가우시안 회귀는 분산 hyperparameter 튜닝이 필요. 255-bin twohot은 분산 hyperparameter를 제거하면서 분포적 표현을 유지.
2. **symlog 공간에서 분류**: bucket 도메인 [-20, 20] → symexp 복원 시 ±4.85e8. 단일 hyperparameter로 모든 환경 커버. **Dreamer-v3가 "환경 독립 hyperparameter"를 달성한 핵심**.
3. **STE 일관성**: `OneHotDist`(RSSM/discrete actor), `Bernoulli`(cont head) 모두 `a.detach() + b - b.detach()` 패턴. forward는 hard, backward는 soft.
4. **unimix 0.01**: discrete categorical 모든 곳에 적용 (`unimix_ratio` configs.yaml:61, actor.unimix_ratio:50). 0 probability 회피.
5. **SymlogDist는 진짜 분포가 아님**: log_prob이 -MSE에 symlog target만 더한 형태. 정규화 상수 없음. "loss 함수의 분포 인터페이스 래핑".
6. **ContDist.absmax**: tanh-saturation 없이 부드러운 액션 박스 제약. mean의 gradient는 살리되 크기만 detach.
7. **out-of-range는 clip**: DiscDist twohot이 [-20, 20] 밖이면 끝 bin에 몰림 → 발산 시에도 학습 무너지지 않음. 안전장치.
8. **`Independent(OneHotDist, 1)` wrapping**: RSSM의 32×32 latent에서 32개 카테고리를 독립으로 보게 만들어 KL/log_prob이 합 형태로 계산되게 함.

---

## 2. RSSM 동역학 (`networks.py`의 `RSSM`)

위치: [networks.py:13-290](../../../dreamerv3-torch/networks.py#L13-L290). Dreamer-v3의 심장 — observation을 latent state로 압축하고 action으로 미래를 예측하는 recurrent dynamics model.

### 2-1. State 구조와 차원

Latent state는 dict — `stoch`(확률적) + `deter`(결정적) 두 축.

| 모드 | 키 | shape (배치 B 기준) | 의미 |
|---|---|---|---|
| discrete (기본) | `stoch` | `[B, 32, 32]` | 32 카테고리 × 32 클래스 one-hot |
| discrete | `logit` | `[B, 32, 32]` | stoch 분포의 logit |
| continuous | `stoch` | `[B, 32]` | Gaussian sample |
| continuous | `mean`, `std` | `[B, 32]` | Normal 파라미터 |
| 공통 | `deter` | `[B, 512]` | GRU hidden state |

`get_feat(state)` ([networks.py:154-159](../../../dreamerv3-torch/networks.py#L154-L159)) = `concat(stoch.flatten, deter)`:
- discrete: 32·32 + 512 = **1536**
- continuous: 32 + 512 = **544**

### 2-2. `__init__` — 4 MLP + GRUCell ([networks.py:14-97](../../../dreamerv3-torch/networks.py#L14-L97))

| 모듈 | 입력 | 출력 | 역할 |
|---|---|---|---|
| `_img_in_layers` ([L48-57](../../../dreamerv3-torch/networks.py#L48-L57)) | prev_stoch ⊕ prev_action | hidden | img_step 진입 |
| `_cell = GRUCell` ([L59](../../../dreamerv3-torch/networks.py#L59)) | hidden, prev_deter | new_deter | 결정적 진행 |
| `_img_out_layers` ([L62-68](../../../dreamerv3-torch/networks.py#L62-L68)) | deter | hidden | prior 분포 입력 |
| `_obs_out_layers` ([L71-77](../../../dreamerv3-torch/networks.py#L71-L77)) | deter ⊕ embed | hidden | post 분포 입력 |
| `_imgs_stat_layer` ([L80-89](../../../dreamerv3-torch/networks.py#L80-L89)) | hidden | logit(discrete) 또는 (mean,std) | prior 분포 파라미터 |
| `_obs_stat_layer` ([L80-91](../../../dreamerv3-torch/networks.py#L80-L91)) | hidden | logit/Normal params | post 분포 파라미터 |

세부:
- 모든 Linear는 `bias=False` + 옵션 LayerNorm(`eps=1e-3`) + SiLU 활성.
- 백본은 `tools.weight_init` (Xavier 변형 truncated normal — 본체는 006 §3-5).
- **stat 레이어는 `uniform_weight_init(1.0)`** — 초기 분포가 거의 균등해야 KL이 폭주하지 않음.
- **prior network와 post network는 별개의 stat_layer를 가짐** (가중치 공유 X). 입력이 다르므로 별도 학습.
- `initial='learned'` (기본): `self.W ∈ ℝ^(1, deter)` 파라미터를 학습. `initial(B)` 호출 시 `tanh(W).repeat(B, 1)` → `state['deter']`, `state['stoch'] = get_stoch(deter)` (= prior network의 mode) ([networks.py:99-125](../../../dreamerv3-torch/networks.py#L99-L125)).
- `initial='zeros'` 옵션 시 stoch/deter 전부 0.

**GRUCell** ([networks.py:742-768](../../../dreamerv3-torch/networks.py#L742-L768)):
- 3 게이트(reset/cand/update)를 한 Linear로 합쳐 계산 + LayerNorm.
- `update_bias=-1` (초기에 state 보존 편향) → 학습 초기 안정성.
- forward: `parts = layers(cat([inputs, state], -1))` → split 3, `reset=sigmoid`, `cand=tanh(reset*cand)`, `update=sigmoid(update + update_bias)`, `out = update*cand + (1-update)*state`.

### 2-3. `img_step` — prior 한 스텝 ([networks.py:208-233](../../../dreamerv3-torch/networks.py#L208-L233))

관측 없이(action만으로) 다음 latent 예측. **imagination rollout의 한 step**.

흐름:
1. `prev_stoch` flatten (discrete면 [B, 32, 32] → [B, 1024]).
2. `x = cat([prev_stoch, prev_action], -1)` → `_img_in_layers` → hidden.
3. `_cell(x, [prev_deter])` → `new_deter`. GRUCell이 Keras 스타일로 state를 list로 감쌈 → `deter = deter[0]`로 풀음.
4. `_img_out_layers(new_deter)` → hidden2.
5. `_imgs_stat_layer(hidden2)` → stats (logit OR mean/std).
6. `stoch = dist(stats).sample()` (STE 적용 — §1-5).
7. `prior = {stoch, deter=new_deter, **stats}`.

`rec_depth=1` 기본 (configs.yaml:37). 코드 주석 "rec depth is not correctly implemented" ([networks.py:219](../../../dreamerv3-torch/networks.py#L219)) — 여러 step 굴리고 싶다면 주의.

### 2-4. `obs_step` — post 한 스텝 ([networks.py:174-206](../../../dreamerv3-torch/networks.py#L174-L206))

관측 embed를 반영한 latent.

흐름:
1. **is_first 처리** ([L176-193](../../../dreamerv3-torch/networks.py#L176-L193)):
   - `prev_state is None` 또는 **전부 첫 스텝**(`sum(is_first) == len(is_first)`) → `prev_state = initial(B)`, `prev_action = zeros`.
   - **일부**만 첫 스텝 → element-wise mask: `prev_state[k] = val·(1-is_first) + initial[k]·is_first`, `prev_action *= (1 - is_first)`.
   - `is_first_r`는 `(1,)`을 val.shape에 맞춰 broadcast 가능하도록 차원 확장.
2. `prior = img_step(prev_state, prev_action)` — 위 §2-3.
3. `x = cat([prior['deter'], embed], -1)` → `_obs_out_layers` → hidden.
4. `_obs_stat_layer(hidden)` → stats.
5. `stoch = dist(stats).sample()`.
6. `post = {stoch, deter=prior['deter'], **stats}`.

**핵심: prior와 post는 `deter`를 공유하고 `stoch`만 다름**. prior는 deter만으로 stoch 분포를, post는 (deter ⊕ embed)로 분포를 만든다 → post가 더 정확, prior는 환경 없이 굴릴 수 있어 imagination에 쓰임.

### 2-5. `observe` / `imagine_with_action` ([networks.py:127-152](../../../dreamerv3-torch/networks.py#L127-L152))

**`observe(embed, action, is_first, state=None)`**: 배치 시퀀스 `[B, T, ...]` → time-major swap → `tools.static_scan`으로 T step 펼치며 `obs_step` 반복. 각 step에서 `prev_state[0]` (= 직전 post)을 사용 (튜플 `(post, prior)`의 첫 원소). 결과는 time-major → 다시 swap → batch-major `[B, T, ...]`. post/prior 두 dict 반환.

**`imagine_with_action(action, state)`**: action 시퀀스만 받아 `img_step`만 반복 → prior trajectory. 학습 imagination에서는 `ImagBehavior._imagine`이 매 step action을 actor로 새로 뽑으므로 본 함수는 **평가/시각화 용도**. 실제로 `WorldModel.video_pred` ([models.py:206](../../../dreamerv3-torch/models.py#L206))가 호출 — 첫 5 step은 관측 기반 post 시퀀스, 6번째부터 imagine_with_action으로 미래 예측.

`tools.static_scan`은 dict/tuple 누적 구현 ([tools.py:795-839](../../../dreamerv3-torch/tools.py#L795-L839)) — 본체 분석은 006 §3-1.

### 2-6. 분포 (요약 참조)

`_suff_stats_layer` ([networks.py:241-270](../../../dreamerv3-torch/networks.py#L241-L270))가 hidden → 분포 파라미터.

- discrete: stat layer 출력 `[..., stoch·discrete]` → reshape `[..., 32, 32]` → `logit`.
- continuous: split → `mean`, `std`. 활성화:
  - `mean_act`: `'none'`(기본, configs.yaml:38) 또는 `'tanh5' = 5·tanh(x/5)` (부드러운 클리핑).
  - `std_act`: 코드 시그니처 기본 `'softplus'`이나 **configs.yaml:39이 `'sigmoid2'`로 오버라이드** → 실효 기본 `2·sigmoid(std/2)`. 다른 옵션: `'softplus'`, `'abs' = abs(std + 1)`, `'sigmoid'`.
  - `std = std_act(raw) + min_std (=0.1)` — 분산 하한.

`get_dist(state)` ([networks.py:161-172](../../../dreamerv3-torch/networks.py#L161-L172)):
- discrete: `Independent(OneHotDist(logit, unimix=0.01), 1)` (§1-5 참조).
- continuous: `ContDist(Independent(Normal(mean, std), 1))` (§1-6 참조).

`get_stoch(deter)` ([networks.py:235-239](../../../dreamerv3-torch/networks.py#L235-L239)): deter만으로 prior network 거쳐 stoch 분포의 **mode** 반환. `initial='learned'`에서 학습 가능한 deter 초기값으로부터 stoch 초기값 생성.

### 2-7. `kl_loss` — KL balancing 본체 ([networks.py:272-290](../../../dreamerv3-torch/networks.py#L272-L290))

KL을 두 방향으로 분리해 별도 가중.

- `sg(x) = {k: v.detach() for k,v in x.items()}` — stop gradient.
- `rep_loss = KL(q(post) || q(sg(prior)))` — post가 prior를 따라가는 방향. **representation 학습** (post network가 prior에 가까워짐).
- `dyn_loss = KL(q(sg(post)) || q(prior))` — prior가 post를 따라가는 방향. **prior 학습** (prior network가 post에 가까워짐).
- 각각 `clip(min=free)` — kl_free=1.0 기본 (configs.yaml:59). clip은 forward 값을 free로 고정하므로 임계 미만에서 gradient=0.
- `loss = dyn_scale·dyn_loss + rep_scale·rep_loss` — configs `dyn_scale=0.5`, `rep_scale=0.1` (configs.yaml:57-58).
- 반환 `(loss, value=rep_loss원본, dyn_loss, rep_loss)`. 세 번째·네 번째는 metric용 (clip 후).

**discrete vs continuous KL 구현 차이** ([networks.py:277-284](../../../dreamerv3-torch/networks.py#L277-L284)):
- discrete: `dist(x)` (=Independent(OneHotDist))에 PyTorch `kl_divergence`.
- continuous: `dist(x)._dist` (= Independent(Normal))에 직접 KL — `ContDist` 래퍼를 풀어 underlying distribution에 KL 적용.

**가중 비대칭의 의미**:
- `dyn_scale > rep_scale` (0.5 > 0.1) → prior 학습이 더 강하게 밀림.
- 직관: post는 관측을 봐서 정확하므로 "정답"에 가깝다. prior가 post에 따라가도록 더 세게 끌어당기고, post가 prior에 끌려가는 힘은 약하게 — encoder가 prior의 부정확함에 끌려가 표현이 망가지는 것을 방지.
- `kl_free=1.0`: 이미 충분히 가까우면 더 줄이지 않음.

### 2-8. 설계 포인트 (RSSM)

1. **prior와 post가 deter 공유** — GRU가 결정적 추세, stoch이 잔여 불확실성을 담당. 이 분리가 KL balancing을 의미있게 만듦.
2. **discrete latent (32×32) + unimix 0.01** — Dreamer-v3가 v2 대비 categorical로 전환한 핵심. 32^32 표현력 + 닫힌형 KL.
3. **Straight-through estimator** — discrete sample에 path gradient를 흘려 actor가 RSSM을 통해 backprop 받게 함.
4. **learned initial deter** — `tanh(W)`만 학습. 작은 자유도로 충분.
5. **is_first 마스킹이 obs_step 내부** — 시퀀스가 에피소드 경계를 넘으면 자동 state 초기화. `batch_length=64`로 잘라도 안전.
6. **kl_free=1.0** — v3의 안정화 기법.
7. **prior/post stat_layer 분리** — 가중치 공유 안 함. 입력이 다르므로 별도 학습.
8. **`update_bias=-1` in GRUCell** — 학습 초기에 state 보존 쪽으로 편향, 안정성.

---

## 3. Encoder/Decoder (`networks.py`)

### 3-1. MultiEncoder ([networks.py:293-357](../../../dreamerv3-torch/networks.py#L293-L357))

dict 형태의 obs를 받아 단일 embedding 벡터로 압축.

**shape 자동 분기** ([L309-322](../../../dreamerv3-torch/networks.py#L309-L322)):
- `excluded = {is_first, is_last, is_terminal, reward}` + `log_*` 접두어 → 인코더 입력에서 제외.
- 나머지 키를 정규식 `cnn_keys`/`mlp_keys`로 매칭. 기본 (configs.yaml:46): `cnn_keys='image'`, `mlp_keys='$^'` (빈 매치, 즉 MLP 비활성).
- `len(shape)==3` AND cnn 매치 → CNN 경로 입력.
- `len(shape) in (1, 2)` AND mlp 매치 → MLP 경로 입력.

**조립** ([L326-346](../../../dreamerv3-torch/networks.py#L326-L346)):
- CNN이 있으면 모든 cnn 입력을 **채널축으로 concat** (`input_ch = Σ v[-1]`) → `ConvEncoder` 하나로 처리. 입력 H×W는 `cnn_shapes` 중 첫 값 사용(모두 같은 해상도 가정).
- MLP가 있으면 모든 mlp 입력을 **마지막 축으로 concat** → `MLP(shape=None, symlog_inputs=True)`. `shape=None`이라 분포가 아니라 raw 벡터 반환.
- 두 출력을 마지막 축에서 concat → `embed = [B, T, outdim]`.

**outdim** ([L326, 333, 346](../../../dreamerv3-torch/networks.py#L326)): `cnn.outdim + mlp_units`. MLP가 없으면 cnn.outdim만, CNN이 없으면 mlp_units만 (=1024 기본).

**dmc_proprio config** (configs.yaml:96-103): `mlp_keys: '.*'`, `cnn_keys: '$^'`로 오버라이드 — 벡터 전용 환경에서 사용. **minecraft** (configs.yaml:148-170)는 `mlp_keys`에 다중 키 정규식 + `cnn_keys='image'` 동시 매칭.

### 3-2. ConvEncoder ([networks.py:448-496](../../../dreamerv3-torch/networks.py#L448-L496))

이미지 인코더. **stride-2 conv를 `log2(H/minres)`번 쌓아 H→minres**로 다운샘플.

기본 (depth=32, minres=4, H=W=64): stages = log2(64/4) = 4.

stage별 채널·공간 진행:
- 입력: in_dim=3 (RGB), out_dim=depth=32, h=w=64.
- 매 stage 끝에 `in_dim = out_dim; out_dim *= 2; h, w //= 2`.
- 최종 stage 직후 변수 상태: out_dim=512, h=w=4.

| stage | Conv2dSamePad in→out | 공간 출력 |
|---|---|---|
| 0 | 3 → 32 | 32×32 |
| 1 | 32 → 64 | 16×16 |
| 2 | 64 → 128 | 8×8 |
| 3 | 128 → 256 | 4×4 |

각 stage = `Conv2dSamePad(stride=2, k=4, bias=False)` + `ImgChLayerNorm` (옵션, 기본 ON) + SiLU.

**outdim 계산** ([L482](../../../dreamerv3-torch/networks.py#L482)): `out_dim // 2 * h * w`. 루프 끝에서 `out_dim *= 2`가 한 번 더 실행되었으므로 실제 마지막 stage 출력 채널은 `out_dim // 2`. 기본 = `512 // 2 * 4 * 4 = 4096`.

**forward** ([L486-496](../../../dreamerv3-torch/networks.py#L486-L496)):
1. `obs -= 0.5`: preprocess에서 `/255`로 [0, 1] 정규화된 입력을 [−0.5, 0.5]로 중심화.
2. `(B, T, H, W, C) → (B·T, H, W, C) → (B·T, C, H, W)` (PyTorch 채널 우선).
3. layers 통과.
4. `(B·T, C', h, w) → (B·T, C'·h·w) → (B, T, outdim)`.

**Conv2dSamePad** ([networks.py:771-798](../../../dreamerv3-torch/networks.py#L771-L798)): TF의 `padding='SAME'`을 PyTorch에서 구현. 입력 크기 기반으로 매 forward 시 `F.pad` → conv. (PyTorch 내장 padding은 입력 의존이 안 됨.) 포팅 디테일은 006 §3-3.

**ImgChLayerNorm** ([networks.py:801-810](../../../dreamerv3-torch/networks.py#L801-L810)): `(B, C, H, W) → permute(0, 2, 3, 1) → LayerNorm(C) → permute(0, 3, 1, 2)` — 채널 축 LayerNorm. BatchNorm 대신 사용 → 배치 통계 의존 제거 (RL의 비정상 분포에 강함). 포팅 디테일은 006 §3-4.

### 3-3. MultiDecoder ([networks.py:360-446](../../../dreamerv3-torch/networks.py#L360-L446))

feat → obs(dict) 재구성 분포.

**제외 키** ([L380](../../../dreamerv3-torch/networks.py#L380)): `excluded = {is_first, is_last, is_terminal}`. **`reward`는 excluded에 없음** — 그러나 obs_space에 `reward`가 들어가는 환경(minecraft `wrappers.RewardObs` 등)에서도 mlp_keys 정규식이 `'reward'`를 매칭하지 않으면 라우팅 안 됨. 일반 환경은 reward가 env.step 반환값이라 obs_space에 없어서 자동 제외.

**조립** ([L382-419](../../../dreamerv3-torch/networks.py#L382-L419)):
- cnn/mlp 키 분리는 Encoder와 동일 (정규식 + 차원 수).
- **CNN 경로**: `ConvDecoder(feat_size, shape=(sum_channels, H, W), ...)` 하나로 전체 cnn 키 동시 디코드 → 마지막에 채널 축 split → 각 키별 `_make_image_dist`.
- **MLP 경로**: `MLP(feat_size, shape=mlp_shapes, dist=vector_dist, ...)` — `shape`이 dict면 MLP 내부에서 각 키별 `mean_layer[name]`을 만들어 `dict[name → Dist]` 반환 ([networks.py:638-642, 665-674](../../../dreamerv3-torch/networks.py#L638-L642)).

**이미지 분포** ([L438-445](../../../dreamerv3-torch/networks.py#L438-L445)):
- `'normal'`: `Independent(Normal(mean, std=1), 3)` — std 1 고정.
- `'mse'` (configs.yaml:48 기본): `MSEDist(mean)` (§1-4). 픽셀 MSE.

### 3-4. ConvDecoder ([networks.py:499-585](../../../dreamerv3-torch/networks.py#L499-L585))

ConvEncoder의 거울상.

흐름 (기본 minres=4, depth=32, 출력 64×64×3):
1. `Linear(feat_size → minres²·depth·2^(L-1))` — 예: `4²·32·8 = 4096`.
2. reshape `(B·T, minres, minres, embed/minres²)` → permute `(B·T, C, minres, minres)`.
3. L stages of `ConvTranspose2d(stride=2, k=4)` + `ImgChLayerNorm` + SiLU.

stage 진행 (예: L=4):
| stage | in_dim → out_dim | 공간 출력 |
|---|---|---|
| 0 | 256 → 128 | 8×8 |
| 1 | 128 → 64 | 16×16 |
| 2 | 64 → 32 | 32×32 |
| 3 | 32 → out_ch (=image channels=3) | 64×64 |

마지막 stage 차별 ([L530-534](../../../dreamerv3-torch/networks.py#L530-L534)):
- `out_dim = self._shape[0]` (전체 cnn 채널 합).
- `act = False`, `norm = False`, `bias = True`.
- 최종 픽셀 출력은 raw.
- **`uniform_weight_init(outscale)`** ([L559](../../../dreamerv3-torch/networks.py#L559)): outscale=1.0 기본. value/critic의 `outscale=0.0`과 대비 — decoder는 1.0이라 출력이 0 근처에 머무르지 않음.

`calc_same_pad(k, s, d)` ([L562-566](../../../dreamerv3-torch/networks.py#L562-L566)): ConvTranspose 출력 크기를 `2·input`으로 맞추기 위한 padding/output_padding 수동 계산.

**forward 후처리** ([L581-584](../../../dreamerv3-torch/networks.py#L581-L584)):
- `cnn_sigmoid=False` (기본): `mean += 0.5` (preprocess에서 뺐던 0.5 복구).
- `cnn_sigmoid=True`: `F.sigmoid(mean)`.

### 3-5. MLP — 만능 head body ([networks.py:588-739](../../../dreamerv3-torch/networks.py#L588-L739))

Dreamer-v3의 모든 head — encoder vector 입력, decoder vector 출력, reward head, cont head, value head, actor — 가 이 클래스 하나로 처리.

**`__init__`** ([L589-655](../../../dreamerv3-torch/networks.py#L589-L655)):
- 백본: `[Linear(bias=False) + LayerNorm + act] × layers` ([L624-635](../../../dreamerv3-torch/networks.py#L624-L635)). 모든 Linear는 `tools.weight_init`.
- `shape` 파라미터로 출력 모드 분기:
  - `shape=None`: encoder MLP — `forward`가 `out` 그대로 반환 (분포 X). ([L662-664](../../../dreamerv3-torch/networks.py#L662-L664))
  - `shape=int 또는 tuple`: 단일 출력 head — `mean_layer = Linear(inp, prod(shape))`, 옵션 `std_layer`.
  - `shape=dict`: 여러 출력 head — 각 키마다 `mean_layer[name]` (decoder vector 경로).
- 출력 head Linear는 `uniform_weight_init(outscale)`. value/critic은 `outscale=0.0` 으로 거의 0 초기화 → 학습 시작 시 V≈0이라 부트스트랩 안정.

**`symlog_inputs=True`** ([L659-660](../../../dreamerv3-torch/networks.py#L659-L660)): encoder MLP 경로에서만 사용 (configs.yaml:46). 입력에 `symlog(x)` 적용.

**`forward(features, dtype=None)`** ([L657-681](../../../dreamerv3-torch/networks.py#L657-L681)): `dtype` 파라미터는 받지만 **본문에서 사용 안 함 (dead arg)** — JAX 원본의 dtype 변환 시그니처 잔재. 안전성·성능 영향 0. Plan2Explore의 `head(inputs, torch.float32)` 호출 ([exploration.py:112](../../../dreamerv3-torch/exploration.py#L112))도 이 dead arg에 의존.

**`dist(...)` dispatcher** ([L683-738](../../../dreamerv3-torch/networks.py#L683-L738)): dist 키별 분포 생성. 11종 전체 매핑은 §1-9 표 참조.

### 3-6. 설계 포인트 (Encoder/Decoder/MLP)

1. **정규식으로 키 라우팅** (`cnn_keys`/`mlp_keys`) — 환경 obs space에 새 키가 들어와도 config만 바꾸면 인코더가 자동 분기.
2. **CNN/MLP 모두 LayerNorm** — 배치 통계 의존 제거, RL의 비정상 분포에 강함.
3. **`Conv2dSamePad` + `ImgChLayerNorm`** — TF 스타일 그대로 PyTorch에 이식. 마이그레이션 산물. 포팅 디테일은 006 §3-3/§3-4.
4. **image_dist 기본이 `'mse'`** — Normal(1) 대신 MSE만 쓰면 픽셀 단위에서 동일하지만 상수 제외로 약간 빠름.
5. **decoder MLP는 dict shape 지원** → 여러 vector head를 한 MLP body로 공유, 마지막만 키별 분기.
6. **MLP 클래스 하나가 거의 모든 head** — `dist` 인자로 11종 스위치. 라이브러리화의 좋은 사례.
7. **symlog 입력 + symlog_mse/symlog_disc 출력** — Dreamer-v3가 "환경마다 튜닝 안 해도 됨"의 핵심 메커니즘. encoder MLP, vector decoder, reward, value 모두 symlog 공간에서 학습.
8. **value head `outscale=0.0`** — Linear 마지막 layer를 작게 초기화. 학습 초기 V≈0이라 부트스트랩이 불안하지 않음. `weight_init` 본체는 006 §3-5.
9. **dead `dtype` arg** — JAX 포팅 잔재. 인지하고 있으면 됨.

---

## 4. 다음 단계 안내

5단계: **WorldModel + ImagBehavior 통합** — [005](005-dreamer_code_analysis_part3.md) 본문.

확인 항목:
- WorldModel `_train`의 head 순회 + KL 계산 + 옵티마이저 흐름 (003 §2 흡수).
- RewardEMA 정식 위치 (005 §2).
- ImagBehavior `_imagine`/`_compute_target`/`_compute_actor_loss`/value loss/slow target.
- λ-return 본체 + tuple 반환 shape 정확한 해석 (사실 오류 정정 사항).
- Optimizer 래퍼 (AMP + grad clip + decoupled WD).
- 손실 흐름 한눈에.

---

## 5. 컨벤션

- 한국어 응답.
- 코드 인용은 줄 링크만. 코드 블록 발췌 금지. 표·산문·수식 인라인 허용.
- 분석 대상은 **dreamer-v3 코드 자체**. F1TENTH 통합 코멘트 금지.
- 본 감사 작업 한정으로 직접 수정 허용 (1회 예외). 이후 다시 append-only 복귀.
