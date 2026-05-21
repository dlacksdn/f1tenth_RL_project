# A19 dry-run 측정 결과 (Phase 2-4, 집컴 RTX 4060 Ti 8GB)

> 2026-05-22. `scripts/dryrun_bench.py` 실측. planning/005 §2-A19·§6-3·§11-A / planning/013 §7 / planning/014 §3 기준.
> 환경: torch 2.4.1+cu124, CUDA True, RTX 4060 Ti 8188MiB. precision=16(AMP), compile=False(#26).
> 측정 절차: map_easy3 env로 1000 env-step 수집(warmup 200) → A. agent build 후 train warmup 10 + timed 100 → B/C.

## 측정값

| 항목 | batch_size=8 (현 configs) | **batch_size=16 (dreamerv3 기본)** |
|---|---|---|
| A env_step_avg_ms | 1.058 | 1.026 |
| B train_step_avg_ms | 571.975 | **599.175** |
| C max_memory_reserved (MB) | 1806 (alloc peak 1737) | **3264** (alloc peak 3130) |
| every = batch_steps/train_ratio | 512/512 = **1** | 1024/512 = **2** |
| trains / agent-step | 1.0 | **0.5** |
| D_A 단일500K (250K agent-step) | 2387.6 min = 39.8h | **1252.6 min = 20.9h** |
| D_B 2-stage 1M (500K agent-step) | 4775.3 min = 79.6h | 2505.1 min = 41.75h |
| VRAM Pass (≤6400MB) | ✅ | ✅ |
| D_A Pass (≤1440min/24h) | ❌ | **✅** |
| D_B Pass | ❌ | ❌ |

모델 파라미터(bench 로그): model_opt 11,689,126 + actor 460,804 + value 525,311 = **12.68M** (A10 13.20M과 정합권; A10은 CPU full instantiate 측정, 본 수치는 optimizer-tracked variable 수).

## 핵심 발견

1. **B(train)가 지배항, env는 무시 가능**: A≈1ms vs B≈600ms. wall-clock의 99.6%가 train. env(CPU 물리)는 빠르다.
2. **B는 batch_size에 거의 둔감(+5%/2배)**: 64-step 순차 RSSM rollout + imagination이 지배 → batch는 GPU에서 잘 병렬화. 따라서 batch_size↑가 step당 비용을 거의 안 늘리면서 **train 빈도를 낮춰** wall-clock을 줄인다.
3. **train 빈도 = batch_size에 결합**: `_should_train = Every(batch_size×batch_length / train_ratio)`. batch_size=8이면 Every(1)=매 step train(replay ratio 2배=over-training), batch_size=16이면 Every(2)=2 step당 1회 = **dreamerv3 의도 replay ratio**. → batch_size=8(8GB 프로파일)은 의도치 않게 2배 학습했고 007 fixed-HP에 어긋났다. batch_size=16이 fidelity-correct.
4. **VRAM 대폭 여유**: batch_size=16에서도 3.3GB/8GB. "8GB 프로파일=batch_size=8" 가정(005 §0-4)은 과보수. batch_size=16 안전.

## §6-3 분기 결정 (VRAM 우선 → wall-clock)

- VRAM: 두 경우 모두 PASS → VRAM 조정 불요.
- Wall-clock: batch_size=8 단일500K FAIL(39.8h) → **batch_size 8→16 조정**(VRAM 여유 활용 + replay ratio 정상화) → 단일500K **20.9h PASS**.
- **확정 권고**: `configs.yaml f1tenth batch_size: 8 → 16`. (batch_length=64 유지, train_ratio=512 유지, steps=5e5 유지.)
- 시나리오 B(2-stage, Oschersleben 훈련 허용 시)는 batch16에서도 41.75h FAIL → 2-stage 채택 시 추가 조정 필요(stage당 steps 축소 또는 2일 분할). 단 #32 provisional(014)에서 시나리오 A가 기본.

## 동반 발견 — vendor 버그 패치 (필수, 본 세션 적용)

`tools.simulate` [line 208] `video = cache[...]["image"]` 가 **에피소드 종료 시 무조건** image 키를 읽음 → vector-only obs(#14)는 KeyError. `video`는 is_eval 분기(line 236)에서만 사용되므로 training/eval 모두에서 실제 학습 루프가 첫 에피소드 종료 시 crash했을 것. → `.get("image", None)` 가드 + line 236 `if video is not None` 가드 적용. (models.py:182 image guard #14/A20와 동일 성격의 fork 패치.)

## 미커밋 작업 (본 세션)

- scripts/dryrun_bench.py (신규)
- vendor/dreamerv3-torch/tools.py (simulate image 가드 2곳)
- vendor/dreamerv3-torch/configs.yaml (line 192 주석 정정 — planning/011/013)
- batch_size 8→16은 **사용자 확인 후 적용 예정**(mandatory stop).
