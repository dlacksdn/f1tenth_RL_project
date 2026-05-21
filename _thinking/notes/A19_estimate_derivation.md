# A19 wall-clock 추정식 유도 (실측 기반, §11-A 정정)

> 2026-05-22. planning/005 §11-A 식을 dreamer.py main loop 코드 정독 + 실측으로 정정.
> 관련: notes/dryrun_results.md, planning/013 §7(steps 단위), planning/005 §11-A.

## 1. 단위 정의 (dreamer.py main loop 정독으로 확정)

- **sim frame**: f110 100Hz 1틱.
- **agent-step (= adapter step = 정책 결정)**: 어댑터가 `action_repeat=2` sim frame을 내부 소비. dreamerv3 1 env.step() = 1 agent-step.
- [dreamer.py:218](../../vendor/dreamerv3-torch/dreamer.py#L218) `config.steps //= action_repeat`: configs `steps`(sim/env 단위)를 agent-step으로 환산.
- [dreamer.py:307](../../vendor/dreamerv3-torch/dreamer.py#L307) 메인 루프는 `agent._step`(agent-step) < config.steps 까지.
- ∴ **configs steps=5e5 → //2 → 250000 agent-step**. "500K 예산" = 250K agent-step = 250K env.step() 호출 = 500K sim frame.

## 2. train 빈도 — §11-A 원식 정정 (중대)

§11-A 원식: `D = (N·A + (N/train_ratio)·B)/1000/60`, "train_ratio=512면 평균 1/512 비율" 가정.
**코드 실제**: [dreamer.py:35](../../vendor/dreamerv3-torch/dreamer.py#L35) `_should_train = tools.Every(batch_size×batch_length / train_ratio)`. [tools.Every:847](../../vendor/dreamerv3-torch/tools.py#L847) `count = int((step - last)/every)`.

→ trains/agent-step = `train_ratio / (batch_size × batch_length)` = `1 / every`.

| batch_size | batch_steps | every=batch_steps/512 | trains/agent-step |
|---|---|---|---|
| 8 | 512 | 1 | **1.0** (매 step) |
| 16 | 1024 | 2 | **0.5** (2 step당 1회) |

§11-A의 "1/train_ratio=1/512"는 **틀렸다**(코드와 57~114배 차이). 올바른 train 횟수 = N_agent × trains/agent-step.

## 3. 정정 추정식

```
N_train      = N_agent × (train_ratio / (batch_size × batch_length))
D[min]       = ( N_agent × A  +  N_train × B ) / 1000 / 60
```
- A = env_step_avg_ms (≈1ms, 무시 가능), B = train_step_avg_ms (지배항).
- N_agent: 단일500K=250000, 2-stage=500000.

## 4. 실측 대입 (batch_size=16, 권고 config)

- A=1.026ms, B=599.175ms, trains/agent-step=0.5.
- 단일500K: N_agent=250000, N_train=125000.
  - env = 250000×1.026/60000 = **4.3 min**
  - train = 125000×599.175/60000 = **1248.3 min**
  - **D_A = 1252.6 min = 20.9 h ≤ 1440 min(24h) → PASS**
- 2-stage 1M: D_B = 2505.1 min = 41.75 h → FAIL (2-stage 채택 시 stage당 steps 축소 필요).

batch_size=8(현 configs) 대입 시: trains/agent-step=1.0, N_train=250000, D_A=2387.6min=39.8h → FAIL. (∴ §6-3 분기로 batch_size=16 채택.)

## 5. 결론

- §11-A의 wall-clock 식은 **train 빈도 항이 틀렸다**. 정정식(§3)이 SSOT.
- batch_size↑가 wall-clock을 줄이는 비직관적 결과의 원인: B가 batch에 둔감(+5%/2배) + train 빈도 = batch_steps/train_ratio에 반비례.
- 24h 게이트 통과 구성: **batch_size=16, batch_length=64, train_ratio=512, steps=5e5(단일 500K), precision=16, compile=False** → D=20.9h, VRAM=3.3GB.
