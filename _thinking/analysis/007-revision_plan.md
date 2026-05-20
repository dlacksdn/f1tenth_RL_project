# 007 - 003~006 분석 문서 수정 계획

> **목적**: 003·004·005·006의 구조적 문제(분포 위치 늦음, ImagBehavior 분할 중복, 9단계 유틸 부유, 포팅·평가 누락)를 해결하기 위한 직접 편집 계획.
> **전제**: 이번 한정 `_thinking/` append-only 컨벤션 예외 허용 — 003~006 직접 수정 가능. 001·002는 건드리지 않음.
> **작성일**: 2026-05-20

---

## 0. 동기 (현 003~006의 문제 요약)

1. **분포 카탈로그(005 §2)가 너무 늦다**. `symlog_disc`/`OneHotDist`/`MSEDist`/`Bernoulli`는 1~5단계 어디서나 등장하는데 6번째 단계에서 처음 정식 정의됨. 004 §2-5에서 표로 선반영해야 했고 그래도 분포 설명이 분산됨.
2. **ImagBehavior 개요(003 §2-3) ↔ 세부(005 §1) 분할**의 경계가 모호. 같은 컴포넌트가 두 문서에 등장.
3. **9단계 "유틸"이 독립 단계**로 잡혔지만 사실상 005 §1에 자연 흡수됨 — 처음부터 사용처 통합이 옳았음.
4. **5·6단계 무게 과함** (005 한 문서 ~1000줄).
5. **포팅 디테일**(`static_scan`, `Conv2dSamePad`, `weight_init`, AMP, dead `dtype` arg)이 산재하거나 미커버.
6. **평가·체크포인트 경로** (`video_pred`, eval rollout, `latest.pt`, generator 재시작 시드)가 단편적으로만 언급.

---

## 1. 골격 — 파일 수 유지(4개), 내용 재분배

| 문서 | 새 주제 | 주요 작업 |
|---|---|---|
| **003** | 진입점·Config 전용 | §2(WorldModel/ImagBehavior 개요) 삭제 → 005로 이전. §1만 남기고 핸드오프 로드맵 갱신 |
| **004** | **분포 카탈로그** + RSSM + Encoder/Decoder | 005 §2(분포) 통째로 이전해 새 §1로 배치. 기존 §1 RSSM·§2 Encoder/Decoder를 §2·§3으로 밀고, 둘 다 "분포는 §1 참조"로 단축 |
| **005** | WorldModel + ImagBehavior 통합 | §2(분포) 제거 → 004로 이동. 003 §2(WorldModel 개요·ImagBehavior 개요) 흡수. 기존 §1(ImagBehavior 세부) + 9단계 유틸(Opt/EMA/slow target/clip)을 사용처에 내장한 단일 통합 문서로 |
| **006** | Exploration + 데이터 파이프라인 + 포팅 + 평가 | 기존 §1·§2 보강 내용을 "1차 정리" 흡수해 단일 섹션으로 재작성(005 §7·§8의 정리본을 옮겨와 통합). 신설 §3 포팅 디테일, 신설 §4 평가·체크포인트 |

---

## 2. 각 문서의 구조 (편집 후)

### 2-1. 003 — 진입점·Config

- §0 핸드오프 (로드맵 갱신: 분포가 2번째 단계로)
- §1 진입점·실행 흐름 (`dreamer.py`)
- §1-X 신규: `Damy`/`Parallel` 분기, 체크포인트 save 흐름 보강
- §2 `configs.yaml` 주요 디폴트
- §3 다음 단계 안내

기존 §2(WorldModel·ImagBehavior 개요) 본문은 005로 이전 후 본 문서에서 제거.

### 2-2. 004 — 분포 + RSSM + Encoder/Decoder

- §0 핸드오프
- §1 **분포 카탈로그** (`tools.py`) ← 신규 위치 (005 §2 이전)
  - `symlog`/`symexp`
  - `DiscDist` (twohot 인코딩, 255-bin, symlog 공간 [-20,20])
  - `SymlogDist`, `MSEDist`
  - `OneHotDist` + STE + unimix 0.01
  - `ContDist` + absmax
  - `Bernoulli` (numerically stable BCE)
  - `UnnormalizedHuber`, `SafeTruncatedNormal`, `TanhBijector`, `SampleDist`
  - head ↔ 분포 매핑 표
- §2 RSSM 동역학 ← 기존 §1, 분포 부분은 §1 참조로 압축
- §3 Encoder/Decoder + MLP head ← 기존 §2, dist 11종 표는 §1 참조로 압축

### 2-3. 005 — WorldModel + ImagBehavior 통합

- §0 핸드오프
- §1 WorldModel (003 §2-1·§2-2 이전 + `_train` 손실 흐름 정밀화)
  - 구성요소 표, `_train` 알고리즘, KL balancing(dyn 0.5 / rep 0.1), `grad_heads`, `preprocess`의 `image` 키 하드코딩 경고
  - `RewardEMA` 정의 (5/95 quantile EMA)
- §2 ImagBehavior (003 §2-3 + 기존 005 §1 통합)
  - `_update_slow_target` (polyak τ=0.02)
  - `_imagine` (feat.detach 위치, horizon=15, 1024 시작점)
  - `_compute_target` (γ·cont, λ-return 인덱싱 가이드, weights cumprod)
  - `tools.lambda_return` + `static_scan_for_lambda_return` 본체 (참조: §3로 일부 이양 가능)
  - `_compute_actor_loss` 3모드 (`dynamics`/`reinforce`/`both`), RewardEMA 적용 위치, entropy bonus
  - Value loss 두 항 (λ-return + slow self-distill)
  - **사용처 내장**: Optimizer 래퍼(AMP scale→backward→unscale→clip→step→update), decoupled WD, `retain_graph=True` 이유
- §3 손실 흐름 한눈에 (그래프)
- §4 다음 단계 안내

### 2-4. 006 — Exploration + 데이터 파이프라인 + 포팅 + 평가

- §0 핸드오프 (분석 종료 선언은 §5로)
- §1 Exploration (Plan2Explore + Random) — 기존 005 §7 + 006 §1 보강 통합
  - ensemble head 분포 정정 (`ContDist(Independent(Normal,1))`, tanh(mean), std≈0.957 고정)
  - dead `dtype` arg
  - ensemble 다양성 = 초기화 차이만 (수렴 시 disag 감소 위험)
  - `_behavior` vs `_task_behavior` 관계
  - `expl_until=0` 함의 (plan2explore에서 task actor가 env 데이터 0)
  - ensemble 학습 분포 vs imagination 분포 불일치 → OOD 보너스 자기강화
  - `disag_action_cond=False`의 epistemic/aleatoric 미분리
  - `RequiresGrad` 분리, ensemble grad가 world model로 안 흐름
  - `Random.actor`의 dummy feat
- §2 데이터 파이프라인 — 기존 005 §8 + 006 §2 보강 통합
  - `load_episodes`
  - `sample_episodes` (seed=0 하드코딩, 글로벌 seed와 무관, is_first 강제 마킹, 잇기 경계의 is_last/is_terminal/discount 미처리)
  - `from_generator` (단일 스레드, prefetch 없음)
  - `simulate` (state packing, agent_state=(latent, action))
  - `Every` (drift-free 카운팅)
  - `save_episodes` (BytesIO 경유 atomic-like)
  - `erase_over_episodes` + `dataset_size` 실제 의미 (디스크 영구 / 메모리 FIFO, env.id 사전순)
  - stale degree 정량화
  - `add_to_cache` zero-padding 함정
- §3 **JAX→PyTorch 포팅 디테일** ← 신규
  - `static_scan` / `static_scan_for_lambda_return` 구현 (역방향 스캔, tuple 반환)
  - `Conv2dSamePad` (TF SAME padding 흉내)
  - `ImgChLayerNorm` (채널축 LayerNorm)
  - `tools.weight_init` / `uniform_weight_init` (Xavier-like, 마지막 layer outscale)
  - AMP scaler 흐름 (scale→backward→unscale→clip→step→update)
  - dead `dtype` arg in `MLP.forward`
- §4 **평가·체크포인트** ← 신규
  - `video_pred` (image 키 하드코딩, video_pred_log 토글)
  - eval rollout 흐름 (`task_behavior.actor.mode()`, training=False 분기)
  - `latest.pt` 단일 파일 정책 (중간 스냅샷 없음, 코드 수정 필요)
  - resume 시 episode generator 재시작과 시드 결정성 문제
- §5 분석 종료 선언

---

## 3. 편집 방식 (실무 규칙)

1. **파일 헤더에 리비전 표시 추가**:
   ```markdown
   > **Revision**: 2026-05-20 — 분포 카탈로그를 004로 이동, ImagBehavior 통합, 포팅·평가 단계 신설 (계획: [007](007-revision_plan.md))
   ```
2. **줄 번호 링크 유지** (`#L491-L494` 등) — 코드 자체가 안 바뀌었으므로 그대로.
3. **본문 이동 시 텍스트는 거의 그대로**, 단 다음 두 가지 압축:
   - 004 §2·§3에서 분포 관련 단락을 "→ §1 참조" 한 줄로 축약.
   - 005 본문에서 003 §2를 흡수할 때 중복 단락 제거.
4. **메타 섹션 재작성**: 각 문서의 핸드오프(§0)와 "다음 단계" 섹션은 새 구조에 맞게 다시 작성. 특히 006 §3-1의 "005 핸드오프 로드맵 대비" 표는 새 로드맵으로 교체.
5. **`_thinking/analysis/`는 이번 작업 한정으로 직접 수정 허용** (사용자 명시). 작업 후 다시 append-only로 복귀.

---

## 4. 작업 순서 권장

1. **004 편집** — 분포 카탈로그 신설(005 §2 본문 이전) + 본문 압축. 가장 큰 이동, 먼저 처리해 005·003이 참조할 위치 확보.
2. **005 편집** — §2(분포) 제거 + 003 §2(WorldModel·ImagBehavior 개요) 흡수 + ImagBehavior 통합. Optimizer/EMA/slow target/clip을 사용처에 내장.
3. **003 편집** — §2 삭제 + 핸드오프 갱신. (005가 §2 흡수를 끝낸 뒤에 안전하게 진행.)
4. **006 편집** — §1·§2를 단일 본문으로 재작성(보강 표를 본문에 통합) + §3 포팅 디테일 신설 + §4 평가·체크포인트 신설 + §5 분석 종료 선언.

---

## 5. 옵션 — 축소판

§3 포팅 디테일·§4 평가·체크포인트 신설을 별도 결정으로 미루고 우선 다음 둘만 처리:

- **A안 (최소 변경)**: 분포 카탈로그 이동(005 §2 → 004 §1)만. 003·006은 손대지 않음.
- **B안 (중간)**: A안 + ImagBehavior 통합(003 §2 → 005). 006은 손대지 않음.
- **C안 (전체)**: 본 문서 §1~§4 전부.

기본 권장은 **C안**. 사용자가 축소 원하면 A 또는 B로 다운스코프.

---

## 6. 컨벤션 메모

- 본 수정 작업은 사용자가 명시적으로 허용한 1회 예외.
- 작업 종료 후 003~006 핸드오프 섹션에 "리비전 완료" 표시를 남기고, 이후부터는 다시 append-only로 복귀.
- 본 007 문서는 append-only 원칙 하에 유지 (수정 계획 기록).
