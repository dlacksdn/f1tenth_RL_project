# 010 — ConvEncoder1D/Decoder1D 차원 정정 (#16 supersede, A10 기반)

> 2026-05-21. 노트북 세션, Phase 2-3 A10 측정에서 파생. 사용자 결정(2026-05-21): "DreamerV3 fixed-HP 핵심 기여를 망가뜨리지 않는 선에서 적절히".
> 선행: [005 v3 #3/#5/#16, A10(line175)](./005-f1tenth_dreamerV3_version3.md), [007 fixed-HP §2-A/§2-C](./007-fixed_hp_fidelity.md), [implementation/010](../implementation/010-phase2-1-networks-1d.md).
> 본 문서는 ConvEncoder1D/Decoder1D **차원의 신 SSOT** (#16 supersede). Phase 2-1 코드(networks_1d.py)·Phase 5 학습이 준수.

---

## 1. 문제 — #16과 12M 목표(#3)의 모순

v3 #16: ConvEncoder1D `flatten 256×34=8704 → Linear(8704,512)`, decoder 거울상 `Linear(feat→8704)`.
Phase 2-3 A10 실측 (CPU full instantiate, scripts/param_audit.py): **26.58M** — 목표 [10M,14M]의 2배.

근본 원인 (지배항):
- **decoder `Linear(1536→8704)` = 13.36M** (decoder 14.17M의 대부분)
- **encoder `Linear(8704→512)` = 4.46M**
- 두 Linear만 ~18M. #16의 flatten=8704가 1080-beam 재구성 거울상에서 거대 투영을 만듦.

#3의 12M은 RSSM(dyn_deter=1024)+heads 기준 sizing이었고, 1D LiDAR encoder/decoder Linear 누적(~18M)을 반영 못 함. A10/R4가 정확히 포착.

---

## 2. 결정 — 1D conv 차원 축소 (5→6 stage, channel cap 128)

| | #16 (원안) | **신 SSOT (본 문서)** |
|---|---|---|
| 스테이지 | 5 stride-2 | **6 stride-2** |
| 채널 | 1→16→32→64→128→256 | **1→16→32→64→128→128→128** (cap 128) |
| 길이 | 1080→540→270→135→68→34 | **1080→540→270→135→68→34→17** |
| flatten | 256×34 = 8704 | **128×17 = 2176** |
| encoder Linear | 8704→512 (4.46M) | 2176→512 (1.11M) |
| decoder Linear | 1536→8704 (13.36M) | 1536→2176 (3.34M) |
| encoder out_dim | 512 (유지) | 512 (유지) |
| embed (lidar+state) | 512+256=768 (유지) | 768 (유지) |

decoder는 거울상 유지 (depths (128,128,128,64,32,16), output ch=1). 길이 스케줄은 encoder forward 공식으로 역산(코드 자동).

---

## 3. fixed-HP 충실도 (007 준수 — 사용자 제약 충족)

- **§2-A 알고리즘 HP 무변경**: dyn_scale/rep_scale/kl_free/lr/model_lr/discount/imag_horizon/**train_ratio/batch_length/action_repeat** 전부 default 상속. 본 변경은 이들을 건드리지 않음 → **DreamerV3 fixed-HP 핵심 기여 보존**.
- **§2-C(1) 모델 크기**: RSSM(#3 dyn_deter=1024, dyn_discrete=16, units=256)·heads **무변경**. world-model 용량 보존.
- **§2-C(2) 1D LiDAR encoder/decoder = "불가피한 신규 모달리티 확장(HP 튜닝 아님, 환경 인터페이스)"**. 본 차원 변경은 이 범주 내부 → fixed-HP 위반 아님. #5의 "Linear(flatten_dim→512)"는 flatten_dim을 generic으로 둠(#16만 8704로 못박았음).

---

## 4. A10 결과 (정정 후)

| 컴포넌트 | params | 비율 |
|---|---|---|
| RSSM(dynamics) | 5,056,512 | 38.3% |
| decoder | 4,135,062 | 31.3% |
| encoder | 1,512,208 | 11.5% |
| reward/cont/actor/value | 1,971,459 | 14.9% |
| slow_value | 525,311 | 4.0% |
| **합계** | **13,200,552 (13.20M)** | — |
| enc+dec | 5.65M | 42.8% |
| RSSM | 5.06M | 38.3% |
| heads | 1.97M | 14.9% |

- **A10 [10M,14M]: PASS** (13.20M).
- 비율: enc+dec 42.8%(기대 ~50%), RSSM 38.3%(기대 ~30%), heads 14.9%(기대 ~20%). conv 채널 cap으로 enc+dec가 줄고 RSSM 비중↑ — RSSM-heavy는 world-model(LeWorldModel) 목적에 부합. 기대 비율은 가이드(hard 기준은 [10,14M]+비율 보고)이므로 수용.

---

## 5. 영향 / 후속
- networks_1d.py depths 기본값 변경 (Phase 2-1 코드 revision). test_networks_1d.py A7 stage_lengths/flat/final_ch 갱신.
- A8(MultiEncoder outdim=512+256=768) 무영향 — lidar_units=512 유지.
- Phase 2-4 A19 dry-run(집컴 GPU): 13.20M + replay + batch_size=8 @ precision16의 VRAM/wall-clock 측정. 26M였다면 batch 축소 위험이 컸으나 13.2M으로 여유 확보.
- 발표(Phase 6): 007 §3 포지셔닝 유지 + "custom 12M(실측 13.2M)" 명시.
