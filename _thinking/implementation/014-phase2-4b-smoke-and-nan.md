# 014 — Phase 2-4b 통합 스모크 + 학습 NaN 근본수정 (A19 후 de-risk)

> 2026-05-22. 집컴 GPU. A19(impl/013) 후 사용자 지시로 결정-무관(시나리오 A/B 공통) de-risk 수행.
> 실제 `dreamer.py main()`을 짧은 예산으로 돌려 vendor vector-only 잠복 블로커를 색출·수정하고, 학습 NaN을 근본 규명·수정.
> 상세 진단 로그·근본원인은 notes/smoke_findings.md(SSOT). 본 문서는 numbered 체인 요약.

## 1. 수정한 블로커 4건 (실학습이 즉시/조기에 깨졌을 것들, 시나리오 무관)

| # | 위치 | 증상 → 수정 | commit |
|---|---|---|---|
| 1 | `tools.simulate:208` | 에피소드 종료 시 무조건 `cache["image"]` → vector-only(#14) KeyError. `.get`+is_eval 가드 | 19bb188 |
| 2 | `networks_1d.py` ConvEncoder1D/Decoder1D forward | `assert dim()==3` → 정책 롤아웃 단일스텝 `(B,L)` crash. leading-dim flatten 일반화 | 19bb188 |
| 3 | `dreamer.py` `eval_state_mean` | `latent["mean"]` — discrete RSSM(dyn_discrete=16) "logit"만 존재 → KeyError. discrete면 `get_dist(latent).mode()` | 19bb188 |
| 4 | `f1tenth_env.py` (env) | **f110 ST dynamics 수치 발산 → vel_x float32 overflow → inf state → 버퍼 오염 → 인코더 overflow → RSSM logit inf → 학습 crash** | 180190b |

## 2. #4 근본원인 (단계적 격리, notes/smoke_findings.md 상세)

- 순수 WM-train(랜덤 고정버퍼)은 fp16/fp32·버퍼크기 무관 **안정** → AMP/버퍼/학습동역학 무죄.
- 정책-수집 경로에서만 ~100 update 후 crash. forward hook → 첫 비유한 = state MLP `Encoder_linear0`.
- 상류: `overflow in cast`(f1tenth_env:174) + 버퍼 state=inf. wrapper가 action을 이미 clip하는데도 f110 dynamics 자체가 발산.
- symlog_inputs=True는 crash 지연만(입력 이미 inf) → 원인 아님, #15/#16대로 False 유지.

## 3. 수정 #4 — wrapper 발산 가드 (환경 인터페이스 robustness, HP 아님)

- step(): `|vel|>_VEL_DIVERGE(1e3)` 또는 raw(vel/pose/scan) 비유한 → **최우선 종료 `cause='diverged'`**.
- `_build_obs`: `np.nan_to_num`+clip → obs **항상 finite·bounded**(lidar fp64 경유, state clip ±10). 버퍼 오염 원천 차단.
- 신규 상수 `_VEL_DIVERGE/_STATE_DIVERGE/_STATE_CLIP`. 테스트 `tests/test_diverge_guard.py`(2건).

## 4. 검증

- repro_mainloop.py: agent 루프 2000 step / **1100 update 완주 NaN 0**(이전 ~100서 crash).
- full `dreamer.py main()` smoke: 2 사이클(eval→train) 완주 + **latest.pt(154MB) 저장** + model_loss ~10-11 안정.
- **pytest 23/23**(+test_diverge_guard). diagnostics: scripts/{diag_wm_nan,repro_mainloop}.py.

## 5. 잔여 / 다음

- f110 ST 적분 불안정 자체는 미규명(가드로 영향 차단). base_classes.py:488 영역. Phase 4 reward에서 'diverged' 종료 페널티 검토.
- reward=0 skeleton 여전 → Phase 4(progress reward + arclength windowed lap, 005 §4-3 / 009).
- #32(Oschersleben 훈련 금지)는 PROVISIONAL(planning/014) — 교수님 확답 대기. Phase 4는 결정-무관이라 진행 가능.
- A19 게이트 PASS(batch16, 단일500K 20.9h, VRAM 3.3GB) → Phase 3/5 진입 자격(시나리오 확정 후).
