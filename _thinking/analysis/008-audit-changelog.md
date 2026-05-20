# 008 - Dreamer-v3 분석 문서 진정성 감사·재구성 로그

> **목적**: 003~006의 4차원(사실/해석/자기일관성/과·미주장+누락) 전수 감사를 수행하고 그 결과를 직접 003~006에 반영한 내역 누적 기록.
> **트리거 컨벤션**: 사용자 명시 허용 (1회 예외) — 003~006 직접 수정 가능. 본 008은 append-only.
> **비교 기준**: 원본 코드 `/home/dlacksdn/dreamerv3-torch/`. v3 논문은 의심 항목 spot check만.
> **인용 규약**: `(../../../dreamerv3-torch/<file>#L<a>-L<b>)` 줄 링크만. 코드 블록 발췌 금지. 표·산문·수식은 허용.
> **작성일 시작**: 2026-05-20

---

## 0. 4차원 진정성 정의

| 차원 | 질문 |
|---|---|
| **사실** | 코드 라인·숫자·시그니처와 일치하는가 |
| **해석** | "왜 그렇게 설계됐다"는 인과 주장이 타당한가 |
| **자기일관성** | 003~006이 서로 충돌하지 않는가 |
| **과·미주장 + 누락** | 부풀림·축소·빠짐이 있는가 |

판정 기호: ✓ 사실 일치 / ✗ 사실 오류 / ⚠ 해석·과주장 / + 누락 / 🔗 자기모순

---

## 1. ToC 재구성 결정

**채택안**: 007 §C안 + 본 감사에서 발견된 누락 보강.

| 문서 | 새 구조 |
|---|---|
| **003** | 진입점·Config 전용. 기존 §2(WorldModel/ImagBehavior 개요) → 005로 이전 |
| **004** | §1 분포 카탈로그 (005 §2 이전) + §2 RSSM + §3 Encoder/Decoder |
| **005** | §1 WorldModel + §2 RewardEMA + §3 ImagBehavior + §4 Optimizer + §5 손실 흐름. 003 §2 흡수 + 분포 §2 제거 |
| **006** | §1 Exploration + §2 데이터 (정식 §) + §3 포팅 디테일 (신설) + §4 평가·체크포인트 (신설) |

---

## 2. 사실 오류 정정 내역 (전수)

### 2-1. 005 §1-3 reward lambda 라인 오기 + 두 람다 구분 누락
- **이전**: `dreamer.py L108-112`에 `reward = lambda f, s, a: heads['reward'](dynamics.get_feat(s)).mode()` 라고 인용.
- **실제**:
  - `dreamer.py:51` 에 `reward = lambda f, s, a: self._wm.heads["reward"](f).mean()` — Plan2Explore 생성용. **`.mean()` 사용**, 입력은 `f`(feat) 직접.
  - `dreamer.py:122-124` 에 `reward = lambda f, s, a: self._wm.heads["reward"](self._wm.dynamics.get_feat(s)).mode()` — `_train` 시 task behavior 호출용. **`.mode()` 사용**, 입력은 state `s`에서 `get_feat`.
- **차이의 의미**:
  - `DiscDist.mean()`과 `DiscDist.mode()`는 코드상 동일 (둘 다 `symexp(Σ p·b)`, [tools.py:469-475](../../../dreamerv3-torch/tools.py#L469-L475)). 그러므로 결과 값은 같다.
  - 그러나 `__init__` 람다는 feat을 그대로 받고, `_train` 람다는 state에서 feat을 계산하는 차이가 있다 — 호출 컨텍스트 다름.

### 2-2. 005 §1-5/§1-6 lambda_return 출력 shape
- **이전**: "tuple of T-1, 각 [N, 1]" / "torch.stack(target, dim=1) → [N, T-1, 1]"
- **실제**: `static_scan_for_lambda_return` ([tools.py:671-688](../../../dreamerv3-torch/tools.py#L671-L688))의 마지막 `torch.unbind(outputs, dim=0)`가 dim 0(=batch dim N) 기준 분해 → **tuple of N, 각 [T-1, 1]**.
- **stack 결과**: `torch.stack(target, dim=1)` ([models.py:325, 403](../../../dreamerv3-torch/models.py#L325)) — N개의 [T-1, 1] 텐서를 dim=1에 쌓으면 **[T-1, N, 1]** (time-major). N과 T-1 위치가 005의 묘사와 반대.
- 이후 모든 shape 인덱싱(actor loss, value loss)이 time-major 기준으로 정합.

### 2-3. 004 §1-6 std_act 기본값
- **이전**: "std_act: 'softplus'(기본) / 'abs' / 'sigmoid' / 'sigmoid2'"
- **실제**: 코드 시그니처 ([networks.py:23](../../../dreamerv3-torch/networks.py#L23)) `std_act="softplus"` 기본. 그러나 `configs.yaml:39` `dyn_std_act: 'sigmoid2'`로 오버라이드 → **실효 기본 sigmoid2**.
- mean_act 동일: 시그니처 `'none'`, config는 `'none'`이라 일치.

### 2-4. 003 §2-3 weights 공식
- **이전**: "weights = cumprod(discount)"
- **실제**: ([models.py:386-388](../../../dreamerv3-torch/models.py#L386-L388)) `weights = cumprod(cat([ones_like(d[:1]), d[:-1]], 0))`. 즉 **선두에 1 끼움**.
- 의미: `weights[0]=1` (시작 step의 가중치는 항상 1), `weights[t] = ∏_{i<t} discount_i`. 이 선두 1이 없으면 첫 step도 discount 곱이 들어가 actor loss가 부정확.

### 2-5. 005 §1-7 "actor_target[:-1]" 표기
- **이전**: "actor_loss = -weights[:-1] * actor_target[:-1]"
- **실제**: ([models.py:432](../../../dreamerv3-torch/models.py#L432)) `actor_loss = -weights[:-1] * actor_target`. **actor_target에는 [:-1] 없음**.
- 이유: `target = stack(target, dim=1)` 이미 [T-1, N, 1] shape이고, base = `value[:-1]`도 [T-1, N, 1]이라 actor_target 자체가 길이 T-1. weights는 [T, N, 1]이라 `weights[:-1]`로 길이 맞춤.

---

## 3. 과주장·규약 위반 정정

### 3-1. 003 §2-1 F1TENTH 통합 코멘트 제거
- 위반 구절: "image 키 없는 환경(예: F1TENTH LiDAR만)이면 KeyError 발생. 통합 시 수정 필요."
- 위반 구절: "image 키 하드코딩 → vector-only obs에선 비활성(video_pred_log: false) 필수"
- **조치**: 코드 사실(`obs["image"] = obs["image"] / 255.0` 무조건 실행, image 키 가정)만 남기고 F1TENTH 언급 삭제. 일반화하여 "vector-only obs space에서는 KeyError" 정도로 표현.

---

## 4. 누락 보강 항목

| # | 항목 | 추가 위치 |
|---|---|---|
| 1 | `weight_init` / `uniform_weight_init` 본체 (Xavier 변형, outscale=0.0의 의미) | 006 §3-5 |
| 2 | `_policy`의 `eval_state_mean` 분기 (stoch ← mean) | 003 §1-2 |
| 3 | `_policy`의 `onehot_gumble` action 변환 (argmax → one_hot) | 003 §1-2 |
| 4 | `_should_pretrain = Once()` + pretrain=100 메커니즘 | 003 §1-2 |
| 5 | `simulate` eval 모드 cache 정리 (`popitem(last=False)`) | 006 §2-4 |
| 6 | `recursively_collect/load_optim_state_dict` | 006 §4-5 |
| 7 | `video_pred` 동작 (5 recon + openl + error, 6 batch) | 005 §1-5 |
| 8 | `enable_deterministic_run`, `set_seed_everywhere` | 006 §4-4 |
| 9 | `Once`, `Until` (Every와 같은 카운터 패밀리) | 006 §2-7 |
| 10 | `imagine_with_action` 용도(video_pred에서 사용) 명시 | 004 §2-5 |
| 11 | reward_head/cont_head의 outscale 차이 (reward 0.0, cont 1.0) | 005 §1-1 |
| 12 | `MultiDecoder`의 reward 제외 메커니즘 (excluded vs regex 라우팅) | 004 §3-3 |
| 13 | `Conv2dSamePad`/`ImgChLayerNorm`/`GRUCell` 포팅 디테일 | 006 §3-3/§3-4/§3-6 |
| 14 | `static_scan` (네스티드 dict/tuple 처리) | 006 §3-1 |

---

## 5. 중복 통합 내역

| 컴포넌트 | 이전 위치 | 통합 후 위치 |
|---|---|---|
| RewardEMA | 003 §2-2 + 005 §1-1 | **005 §2** 단일 정의 |
| WorldModel 개요+세부 | 003 §2-1 + 005 (없음, 통합 안 됨) | **005 §1** |
| ImagBehavior 개요+세부 | 003 §2-3 + 005 §1 | **005 §3** |
| 손실 흐름 그래프 | 003 §2-4 + (005에 흩어짐) | **005 §5** |
| 분포 카탈로그 | 005 §2 (다른 문서에서 forward ref) | **004 §1** |

---

## 6. 작업 진행 로그

### 2026-05-20 — 인프라 구축 완료
- 원본 6개 파일(`dreamer.py`, `models.py`, `networks.py`, `tools.py`, `exploration.py`, `configs.yaml`) 전수 흡수.
- ToC 재구성안 확정.
- 사실 오류 5건, 과주장 2건, 누락 14건 목록 작성.

### 2026-05-20 — 004 재작성 완료
- 분포 카탈로그(§1) 신설: 005 §2 본문 전체 이전 + 정정. `symlog`/`symexp`, `DiscDist`(twohot), `SymlogDist`, `MSEDist`, `OneHotDist`(STE+unimix), `ContDist`(absmax), `Bernoulli`(softplus BCE), 기타 옵션 분포. head-dist 매핑 표 통합.
- RSSM(§2): State 구조·`__init__`·`img_step`·`obs_step`·`observe`·`kl_loss` 모두 line-by-line 재검증. `std_act` 실효 기본 `sigmoid2` 정정 반영. GRUCell 동작·`update_bias=-1` 의미 추가.
- Encoder/Decoder(§3): MultiEncoder regex 라우팅, ConvEncoder stage별 채널·공간 진행 표, ConvDecoder 거울상, MLP head body 11종 dist dispatcher. dead `dtype` arg 인지 사항 포함.
- 모든 코드 블록 발췌 제거. 줄 링크와 표·산문·수식만 사용.

### 2026-05-20 — 005 재작성 완료
- WorldModel(§1) 신설: 003 §2-1 본문 흡수 + 본 감사 발견사항 반영. `__init__` 구성요소 표, `_train` 알고리즘 line별 검증, `preprocess`·`video_pred` 상세 동작. reward outscale=0.0 vs cont outscale=1.0 차이 추가.
- RewardEMA(§2): 003 §2-2 + 005 §1-1 중복 제거, 단일 정의로 통합.
- ImagBehavior(§3): 003 §2-3 개요 + 기존 005 §1 세부 통합. **사실 오류 정정**:
  - `lambda_return` 출력 shape: tuple of N (각 [T-1, 1]) → stack(dim=1) → [T-1, N, 1] time-major. 이전 "[N, T-1, 1]"은 오류였음.
  - `actor_target[:-1]` 표기 제거. 실제는 `actor_target` 그대로 (이미 길이 T-1).
  - reward lambda 두 람다 구분: `dreamer.py:51` (.mean(), Plan2Explore용) vs `dreamer.py:122-124` (.mode(), task용). 라인 번호 정정.
- Optimizer(§4): 005 §1-9의 사용처 내장 정리를 정식 §로 격상. AMP scaler 정석 패턴 명시.
- 손실 흐름 통합(§5).
- 분포 카탈로그 본문 제거 → 004 §1 참조로 단축.

### 2026-05-20 — 003 재작성 완료
- §2 WorldModel/ImagBehavior 본문 제거 → 005로 이전.
- §1 진입점 본문 line-by-line 재검증, 누락 항목 보강:
  - `_should_pretrain = Once()` + pretrain=100 burst 메커니즘.
  - `_policy`의 `eval_state_mean` 분기 (continuous 전용, discrete에서 KeyError 경고).
  - `_policy`의 `onehot_gumble` action 변환.
  - dreamer.py:51 vs L122-124 두 reward 람다 비교 표.
  - `Damy`/`Parallel` 분기, `recursively_collect/load_optim_state_dict` 언급.
- §2 `configs.yaml` 디폴트 표 정리: 카테고리별 (일반/환경/RSSM/공통nn/encoder/decoder/actor/critic/reward_head/cont_head/WM학습/데이터/behavior/exploration).
- **F1TENTH 통합 코멘트 전부 제거** (규약 위반 정리). "image 키 하드코딩"은 코드 사실로만 표기.

### 2026-05-20 — 006 재작성 완료
- "보강" 명명 폐기 — §1·§2가 7·8단계의 정식 본문임을 명시.
- Exploration(§1): 기존 005 §7 시도와 006 §1 보강을 통합. ensemble head 분포 정정(ContDist Normal, std≈0.957 고정), dead dtype arg, ensemble 다양성 메커니즘, `_behavior`/`_task_behavior` 관계, `expl_until=0` 함의, 분포 불일치 OOD 자기강화, `disag_action_cond=False` 함의 모두 포함.
- 데이터(§2): `load_episodes`, `sample_episodes`(seed=0 하드코딩, is_first 강제, 잇기 경계 신호 비대칭), `from_generator`(단일 스레드), `simulate`(state packing, eval cache popitem), `add_to_cache`(zero-padding 함정), `save_episodes`(BytesIO atomic-like), `erase_over_episodes`/`dataset_size`(env.id 사전순), `Every`/`Once`/`Until` 카운터 패밀리.
- **신설 §3 JAX→PyTorch 포팅 디테일**:
  - `static_scan` 네스티드 dict/tuple 처리.
  - `static_scan_for_lambda_return` 역방향 스캔 + tensor concat.
  - `Conv2dSamePad` 동적 padding.
  - `ImgChLayerNorm` permute 트릭.
  - `weight_init`/`uniform_weight_init` 본체 — 0.879 상수 의미, outscale=0.0의 "가중치 0" 메커니즘.
  - `GRUCell` 3-gate 합성 + `update_bias=-1`.
  - AMP scaler 정석 패턴 (005 §4와 교차).
  - dead `dtype` arg.
- **신설 §4 평가·체크포인트**:
  - eval rollout (training=False, mode, is_eval=True cache 정리).
  - `video_pred` 두 위치 호출 (train log + eval 직후).
  - `latest.pt` 단일 파일 정책.
  - resume 시드 결정성 한계.
  - `recursively_collect/load_optim_state_dict` 메커니즘.
  - `set_seed_everywhere`/`enable_deterministic_run`.
- §5 분석 종료 선언 + 단계별 커버리지 표.

### 2026-05-20 — 최종 cross-doc 일관성 점검 완료
- 모든 cross-reference 링크 확인: 003 → 004/005/006, 004 → 005, 005 → 004/006, 006 → 003/004/005 모두 정합.
- 핸드오프 §0이 모든 문서에서 같은 단계 매핑 사용.
- 컨벤션 §이 모든 문서에서 동일 (한국어, 줄 링크, F1TENTH 코멘트 금지, append-only 1회 예외).
- 중복 정의 0건: RewardEMA(005 §2 단일), ImagBehavior(005 §3 단일), 분포(004 §1 단일).

---

## 7. 본 감사가 다루지 않은 영역 (의도적 제외)

- `envs/` 환경 어댑터
- `parallel.py` 멀티프로세스 환경
- `Dockerfile`, `requirements.txt`, `xvfb_run.sh`
- `tools.Logger` 내부 (TensorBoard/JSONL 입출력)
- `offline_traindir`/`offline_evaldir` 오프라인 학습 경로
- `debug` config

이들은 dreamer-v3 알고리즘 코드 자체에 대한 분석이 아니므로 제외. 005의 `assert kl_loss.shape == embed.shape[:2]` 같은 sanity assertion은 분석에 포함.

---

## 8. 본 감사의 한계

- v2 논문 대조는 비용 대비 효과 낮아 제외 (사용자 결정).
- JAX 정식 구현 대조는 포팅 충실성이 본 감사의 목적이 아니라 제외.
- atomic claim 수준의 100% 검증은 시간 제약으로 spot check 비율 존재. 핵심 알고리즘 경로(WorldModel `_train`, ImagBehavior `_train`, RSSM dynamics, 분포 클래스)는 line-by-line 검증, 보조 유틸은 시그니처 + 핵심 동작만 검증.
- "내가 모르는 dreamer-v3 메커니즘"의 unknown unknowns는 잔존. 단, configs.yaml의 모든 키를 분석에 매핑하는 방식으로 일부 보완.

---

## 9. 기획 단계 진입 전 잔존 리스크

본 감사 후 기획 단계로 넘어가도 손색은 없으나, 다음 두 지점은 **인지하고 진입**할 것.

### 9-1. 숫자 가정은 기본 config 기준

본 분석의 shape 수치는 모두 **`configs.yaml`의 `defaults` 블록 기준**으로 계산됨.

| 수치 | 본 분석 값 | 가정 |
|---|---|---|
| `feat_size` | 1536 (discrete) / 544 (continuous) | `dyn_stoch=32, dyn_discrete=32, dyn_deter=512` |
| ConvEncoder `outdim` | 4096 | `cnn_depth=32, minres=4, H=W=64` |
| mlp encoder `outdim` 기여 | 1024 | `mlp_units=1024` |
| imagination batch `N` | 1024 | `batch_size=16, batch_length=64` |
| imagination horizon `H` | 15 | `imag_horizon=15` |

**override가 잦은 환경**:
- `crafter`/`minecraft`: `dyn_hidden=1024`, `dyn_deter=4096`, `units=1024` → `feat_size = 32·32 + 4096 = 5120` (discrete). 거의 모든 차원 표 재계산 필요.
- `atari100k`: `train_ratio=1024`, `time_limit=108000`, `action_repeat=4`. 학습 빈도·step 환산 다름.
- F1TENTH는 새 config 신설할 텐데, 그 시점에 본 분석의 차원·step·학습 빈도 수치를 새 config로 재계산 필요. 기획 문서에서 "본 분석은 defaults 기준, F1TENTH config는 별도 계산" 한 줄로 명시하면 충분.

### 9-2. `envs/` 미커버 — 어댑터 contract spot check 필요

`envs/` 모듈은 본 감사에서 의도적 제외 (환경별 어댑터, 알고리즘 무관). 그러나 **F1TENTH gym → dreamer 어댑터를 신규 작성**할 때 다음 contract들의 정확한 형태는 기존 어댑터를 한 번 열어봐야 안전:

- `envs/dmc.py` 같은 기존 어댑터의 `reset()`/`step()` 반환 형식 — obs dict 키 구성, `done`/`info` 형태.
- `envs/wrappers.py`의 `TimeLimit`/`SelectAction`/`UUID`/`OneHotAction`/`NormalizeActions`/`RewardObs` 정확한 동작 — 특히 `UUID`가 env.id를 어떻게 생성하는지 (cache 키와 직결).
- `observation_space.spaces` 의 dict 구조 — `Box`/`Discrete` 등 gym space 객체로 어떻게 전달되는지 (`shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}` 호출이 [models.py:35](../../../dreamerv3-torch/models.py#L35)에 있음).
- `is_first`/`is_last`/`is_terminal`이 obs dict에 어떻게 주입되는지 — wrapper 어디서 어떤 시점에 채우는지.

기획 단계에서 어댑터 설계 § 진입 직전에 `envs/dmc.py` + `envs/wrappers.py` 두 파일만 정독하면 됨. 미리 전수 분석할 필요는 없음 — 핀포인트로 충분.

### 9-3. 운용 규칙

- "분석 한 줄 인용 = 원본 한 번 확인" 규칙 유지.
- 잔존 리스크 두 지점은 기획 문서 초반에 한 번 짚고 시작.
- 기획 중 더 깊은 분석이 필요한 항목이 나타나면 008 이후로 추가 감사 가능 (append-only).
