# 007 — DreamerV3 fixed-HP 충실도 원칙 (override 허용 범위 SSOT)

> 2026-05-21. 노트북(`env/`, torch+cpu) 세션. Phase 1-4 진입 전 사용자 질의에서 파생.
> 선행: [005 v3 §2-3 config](./005-f1tenth_dreamerV3_version3.md), fork `dreamerv3-torch/configs.yaml` defaults.
> 본 문서는 **설계 무결성 원칙**이다. Phase 2-3(configs_f1tenth.yaml 작성)·Phase 5(학습)·Phase 6(발표)이 반드시 준수.

---

## 1. 원칙 (사용자 확정)

**DreamerV3의 "단일 고정 하이퍼파라미터로 모든 task 동작"은 논문의 핵심 기여이므로 반드시 보존한다.**
알고리즘 레벨 HP는 fork default를 **건드리지 않는다.** 변경이 허용되는 것은 오직:
1. **모델 크기** — 논문도 task가 아니라 *compute 예산*에 따라 고르는 부류 (XS~XL 프리셋). 우리는 8GB(집컴 RTX 4060 Ti)에 맞춰 조정.
2. **compute/memory 프로파일** — precision, batch_size, compile, dataset_size 등 하드웨어 종속.
3. **불가피한 환경/모달리티 인터페이스** — LiDAR 1D encoder/decoder, reward, action space, termination. 이건 "Dreamer HP"가 아니라 "환경 정의".

**task(map_easy3 ↔ Oschersleben) 간에는 동일 config를 쓴다** — 이게 우리 scope 안에서 no-per-task-tuning 원칙의 실제 시험이며, Stage 2는 `latest.pt` warm load + fresh optim으로 동일 config 유지(v3 §5).

---

## 2. fork default vs configs_f1tenth.yaml 대조 (2026-05-21 실측)

근거: `dreamerv3-torch/configs.yaml` `defaults:` 블록 vs v3 §2-3 config 블록(line 254~289).

### 2-A. ✅ 유지 — 진짜 universal HP (override 금지)

| HP | default = 우리값 |
|---|---|
| `dyn_scale` / `rep_scale` / `kl_free` | 0.5 / 0.1 / 1.0 |
| `unimix_ratio` | 0.01 |
| actor `entropy` / actor·critic `lr` / `model_lr` | 3e-4 / 3e-5 / 1e-4 |
| `discount` / `discount_lambda` / `imag_horizon` | 0.997 / 0.95 / 15 |
| two-hot(`symlog_disc` reward/critic), `reward_EMA`, slow target | default |
| `weight_decay` / `grad_clip`(model) / `opt` | 0.0 / 1000 / adam |
| `train_ratio` / `batch_length` / `action_repeat` | 512 / 64 / 2 |

→ v3 config 블록은 위 항목을 **하나도 override 하지 않는다.** 향후 config 작성 시에도 이 표의 값은 명시 override 금지(= default 상속).

### 2-B. ⚠️ override 허용 — compute/memory profile

| 항목 | default → 우리 | 사유 |
|---|---|---|
| `precision` | 32 → 16 | 8GB VRAM (C-N7) |
| `batch_size` | 16 → 8 | 8GB VRAM |
| `compile` | True → False | 결정 #26 호환성 |
| `dataset_size` | 1e6 → 200000 | RAM (F5-14) |
| `prefill` | 2500 → 0 | 결정 #23 (GapFollower 별도 collector) |
| `eval_state_mean` | False → True | 결정 #19 |
| `eval_episode_num` | 10 → 20 | 평가 통계 |
| `steps` | 1e6 → 5e5 | 24h 예산 (dmc_proprio도 5e5) |

→ 논문도 하드웨어별로 바꾸는 부류. "단일 HP" 주장 위반 아님.

### 2-C. 🔴 명백한 deviation — 발표 시 투명하게 명시할 것

**(1) 모델 크기: 공식 프리셋이 아니라 hand-mixed 12M.**

| dim | default(DMC size) | 우리(12M) |
|---|---|---|
| `dyn_hidden` | 512 | 256 |
| `dyn_deter` | 512 | 1024 (↑) |
| `dyn_stoch` | 32 | 32 |
| `dyn_discrete` | 32 | 16 |
| `units` | 512 | 256 |
| encoder/decoder `mlp_units` | 1024 | 256 |

논문은 XS/S/M/L/XL **정해진 프리셋** 중 예산에 맞는 걸 고른다. 우리는 deter는 키우고 나머지는 줄여 차원을 **직접 섞었다.** compute-driven이라 정당화되나 "논문 프리셋 그대로"는 아님.

**(2) Encoder HP 일부 변경 — universal HP 영역을 건드린 부분.**
- `mlp_units 1024 → 256` (state MLP 용량 축소, 12M 예산)
- `symlog_inputs True → False` (결정 #15: state를 wrapper에서 이미 정규화 → 중복 symlog 제거)
- **1D LiDAR ConvEncoder/Decoder 신규 추가** — 논문엔 image-CNN + vector-MLP 두 경로뿐. 불가피한 신규 모달리티 확장(환경 인터페이스, HP 튜닝 아님).

**(3) reward shaping: task별 R_lap 상이** (map_easy3=25 / Oschersleben=100, v3 §4-3).
- Dreamer HP 아님(환경 정의). 트랙 길이 비례 shaping.
- 사용자 판단(2026-05-21): "나중에 일반화 성능 올릴 때 바꾸면 되는 것 — 큰 상관 없음." → 현 단계 허용. 단 두 task 보상 신호가 완전 동일하진 않다는 점 인지.

---

## 3. 발표 포지셔닝 (v3 §6 Phase 6 보강)

❌ "DreamerV3의 task-invariance를 *재현*했다"
✅ **"DreamerV3의 고정 HP 레시피를 *유지한 채*, 8GB용 custom 12M + 1D LiDAR encoder로 F1Tenth에 적응시켰다"**

근거: §2-A 전체를 default로 보존(=핵심 기여 유지), §2-C는 compute·모달리티 불가피성으로 한정·명시.

---

## 4. 후속 phase 체크포인트

- **Phase 2-3** configs_f1tenth.yaml 작성 시: §2-A 항목 명시 override 금지(default 상속 확인). §2-B/2-C만 override. 작성 후 본 표와 대조 검증.
- **Phase 5** 학습: Stage 1 ↔ Stage 2 config diff = (task name, R_lap, L_track, warm-load only). 알고리즘 HP diff=0 확인.
- **Phase 6** 발표: §3 포지셔닝 + §2-C 3종 deviation 명시.
