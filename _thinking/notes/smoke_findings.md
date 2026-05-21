# 통합 스모크런 결과 (dreamer.py main() train+eval+ckpt, 집컴 GPU)

> 2026-05-22. A19 후 de-risk: 실제 dreamer.py main()을 짧은 예산(steps 1200, eval 1ep, prefill 600)으로 돌려 vendor vector-only 잠복 버그 색출. 시나리오 A/B 무관(invariant).
> 실행: `cd vendor/dreamerv3-torch && python dreamer.py --configs f1tenth --logdir /tmp/... --steps 1200 --eval_every 600 --prefill 600 --eval_episode_num 1`

## 발견·수정 (3건, 모두 실제 학습을 즉시 깨뜨렸을 하드 블로커)

| # | 위치 | 증상 | 패치 |
|---|---|---|---|
| 1 | `tools.simulate:208` | 에피소드 종료 시 무조건 `cache[...]["image"]` → vector-only(#14) KeyError. video는 is_eval 분기에서만 사용 | `.get("image",None)` + line 236 `if video is not None` 가드 |
| 2 | `networks_1d.py:142,231` ConvEncoder1D/Decoder1D forward | `assert dim()==3 (B,T,L)` — 정책 롤아웃 단일스텝 `(B,L)`에서 AssertionError | leading-dim 전부 flatten→처리→복원 (표준 dreamerv3 인코더 패턴). 학습(B,T,L)·롤아웃(B,L) 모두 처리 |
| 3 | `dreamer.py:94` `eval_state_mean` | `latent["stoch"]=latent["mean"]` — discrete RSSM(dyn_discrete=16)은 "logit"만 있고 "mean" 없음 → KeyError | discrete면 `get_dist(latent).mode()` 사용(obs_step sample=False와 동일), continuous는 "mean" |

→ #1/#2/#3 패치 후 스모크가 prefill→agent build→**eval 1ep 통과**(eval_return 로깅, eval_length 231)→training 진입까지 도달.

## 해결된 이슈 (4번) — world model 학습 NaN → f110 dynamics 발산 (RESOLVED 2026-05-22)

**근본원인 (단계적 진단으로 확정)**:
1. 순수 WM-train(랜덤데이터 고정 버퍼)은 fp16/fp32·버퍼크기 무관하게 **안정**(diag_wm_nan.py: 300 update 완주, loss 371→6.5). → WM 학습 동역학·AMP·버퍼크기 모두 무죄.
2. main()/정책-수집 경로(repro_mainloop.py)에서만 ~100 update 후 crash. forward hook으로 **첫 비유한 출력 = `wm.encoder._mlp.layers.Encoder_linear0`(state MLP)** 핀포인트.
3. 더 추적: `f1tenth_env.py:174 RuntimeWarning: overflow encountered in cast` + 버퍼에 **state=inf 저장**. 즉 **f110 ST dynamics가 수치 발산해 vel_x가 float32 한계(>3.4e38) 초과 → state=inf → 인코더 overflow → RSSM logit inf → OneHotDist ValueError**.
4. wrapper는 action을 이미 actuator 한계로 clip하는데도 발산 → **f110 dynamics 자체의 수치 불안정**(특정 궤적; 랜덤 탐색은 회피, 학습된 정책의 일관 명령이 도달). symlog_inputs=True도 입력이 이미 inf라 무력(crash 지연만).

**수정 (환경 인터페이스 robustness, HP 아님, 시나리오 무관)**:
- `f1tenth_env.py` **발산 가드**: step()에서 `|vel|>1e3` 또는 raw(vel/pose/scan) 비유한 감지 → **최우선 종료(cause='diverged', is_terminal)**. `_build_obs`는 `np.nan_to_num` + clip으로 **obs를 항상 finite·bounded 보장**(lidar fp64 경유 정화, state clip ±10). → 버퍼 오염 원천 차단.
- symlog_inputs는 원인 아님 → #15/#16대로 False 유지(되돌림).

**검증**: repro_mainloop 2000 step / **1100 update 완주 NaN 0**(이전 ~100서 crash). full `dreamer.py main()` smoke 2 사이클 완주 + latest.pt(154MB) 저장 + model_loss ~10-11 안정. pytest 21/21 유지.

**잔여(별도)**: f110 dynamics가 왜 발산하는지(ST 적분 불안정, base_classes.py:488 영역)는 미규명 — 가드로 영향 차단됨. Phase 4 reward에서 'diverged' 종료에 페널티 부여 검토. reward=0 skeleton은 여전(Phase 4 대상).

---

## (구) 미해결 이슈 (4번) — world model 학습 NaN (~100 update 후 발산) [위에서 RESOLVED]

- **증상**: "Start training" 후 `wm.observe → obs_step → get_dist → OneHotDist` logits (16,32,16)에 invalid(NaN/Inf). metrics 보면 **update 100까지 정상**(model_loss=27.4, **model_grad_norm=104**, **lidar_loss=21.4가 지배**, state_loss=0.42, reward_loss=4.2) 후 다음 step에서 발산.
- **fp16 아님**: precision=32에서도 동일 NaN 재현 → AMP 무관. 학습 안정성/아키텍처 문제.
- **grad clip 무력**: model `grad_clip=1000`(default), grad_norm=104라 클립 미발동. NaN은 클립으로 안 막힘.
- **지배 가설**: 커스텀 1D 인코더/디코더(networks_1d). lidar_loss(symlog_mse, 1080-dim sum-agg)가 model_loss의 78%. ConvEncoder1D/Decoder1D의 init scale·LayerNorm·flatten Linear(2176)이 활성/그래디언트 폭주를 유발할 가능성.
- **교란요인(주의)**: 현재 reward=0 skeleton(Phase 4 미구현)이라 reward/value/actor는 degenerate(metrics ~1e-7). WM NaN 자체는 reward-independent지만, 실제 reward 투입 시 동역학이 달라질 수 있음.
- **다음 진단 후보**: (a) 순수 WM-train 루프 250+ step에서 model_grad_norm 추이 로깅(발산 onset 핀포인트), (b) lidar 입력 분포·정규화 확인(obs lidar [0,1] 스케일), (c) ConvEncoder1D/Decoder1D 출력 magnitude·LayerNorm 위치 점검, (d) decoder outscale, (e) 必要시 임시 grad_clip↓ 또는 입력 symlog — 단 007 fixed-HP와 충돌 여부 검토.
- **상태**: Phase 5(실학습) 진입 전 반드시 해소 필요. 단 별도 디버깅 분기로 다룬다.

## 부수 메모

- 모델 파라미터(로그): model 11.69M + actor 0.46M + value 0.53M = 12.68M (A10 13.20M권).
- L_track 갱신 확인: map_easy3=100.57m, Oschersleben=275.18m (centerline 재추출, implementation/008). planning/011/013의 312.61 인용은 stale → 후속 정정 대상.
- eval 경로 정상 작동(eval_state_mean mode 사용) 확인 = decision #19 + discrete 호환 검증됨.
