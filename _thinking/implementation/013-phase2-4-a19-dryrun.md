# 013 — Phase 2-4 A19 dry-run (집컴 GPU, Stage 1 진입 게이트)

> 2026-05-22. 집컴 RTX 4060 Ti 8GB, torch 2.4.1+cu124. planning/005 §2-A19·§6-3·§11-A, planning/013 §7, planning/014 §3 기준.
> 산출물: scripts/dryrun_bench.py, notes/dryrun_results.md, notes/A19_estimate_derivation.md.

## 1. 결과 요약 (notes/dryrun_results.md SSOT)

| batch_size | A env(ms) | B train(ms) | C VRAM(MB) | trains/step | D_A 단일500K | D_B 2-stage |
|---|---|---|---|---|---|---|
| 8 (구 config) | 1.058 | 571.98 | 1806 | 1.0 | 39.8h ❌ | 79.6h ❌ |
| **16 (신 config)** | 1.026 | 599.18 | 3264 | 0.5 | **20.9h ✅** | 41.75h ❌ |

- **VRAM**: 두 경우 PASS(≤6400). 8GB 프로파일(batch8) 과보수 판명.
- **A19 게이트(시나리오 A 단일 500K)**: batch16에서 **20.9h PASS** → Phase 3 진입 가능.

## 2. §6-3 분기 결정 — batch_size 8→16

VRAM 우선 분기: VRAM 통과 → wall-clock 조정. **batch_size 8→16** 적용(configs.yaml f1tenth).
근거(3-in-1):
1. **fixed-HP 정상화(007)**: `_should_train=Every(batch_size×batch_length/train_ratio)`. batch8=`Every(512/512)=1`=매 step train=replay ratio 2배 over-training. batch16=`Every(2)`=dreamerv3 의도 replay ratio. → batch8이 오히려 007 위반이었음.
2. **VRAM 여유**: batch16도 3.3GB/8GB.
3. **wall-clock**: 단일500K 39.8h→20.9h.
- batch_length=64 / train_ratio=512 / steps=5e5 / precision=16 / compile=False 유지.

## 3. §11-A 식 정정 (notes/A19_estimate_derivation.md SSOT)

§11-A 원식의 `N/train_ratio`(train 1/512 비율 가정)는 **코드와 불일치**. 실제 train 횟수 = `N_agent × train_ratio/(batch_size×batch_length)`. 정정식:
```
D[min] = (N_agent×A + N_agent×(train_ratio/(batch_size×batch_length))×B)/1000/60
```
steps 단위(013 §7): configs steps=5e5(env/sim) →[dreamer.py:218 //action_repeat]→ 250K agent-step.

## 4. vendor fork 패치 (필수)

`tools.simulate` [line 208] `video=cache[...]["image"]`가 에피소드 종료 시 무조건 image 키 접근 → vector-only obs(#14) KeyError. `video`는 is_eval 분기(line 236)에서만 소비. → `.get("image",None)` + line 236 `if video is not None` 가드. (models.py:182 image guard #14/A20 동류 패치.) **실제 학습 루프의 첫 에피소드 종료 crash를 사전 차단.**

## 5. mandatory stop 상태 + 다음

- A19 PASS(batch16, 시나리오 A). Phase 3(train.py snapshot wrapper + counter ckpt, planning/008·005 §3) 진입 가능.
- ★ 단 #32(Oschersleben 훈련 금지)가 교수님 확답 대기 = PROVISIONAL(planning/014). 시나리오(A/B) 확정 전 Phase 3 본격 착수 보류. 시나리오 B면 2-stage 41.75h FAIL → stage당 steps 축소 재측정 필요.
- OPEN-U1(채점 trade-off "이따가"), OPEN-U2(도메인 랜덤화 미응답)도 Phase 5 게이트.

## 6. 변경 파일

- scripts/dryrun_bench.py (신규, DRYRUN_OVERRIDES 환경변수로 §6-3 재측정 지원)
- vendor/dreamerv3-torch/tools.py (simulate image 가드 2곳)
- vendor/dreamerv3-torch/configs.yaml (batch_size 8→16, line 192 task 주석 정정)
- _thinking/planning/011~014, _thinking/notes/{dryrun_results,A19_estimate_derivation}.md
