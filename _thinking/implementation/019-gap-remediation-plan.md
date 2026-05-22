# 019 — 구현 보완 계획서 (018 갭 감사 반영, 제3자 검수 대상)

> 2026-05-22. 본 문서는 [018 traceability gap audit](./018-traceability-gap-audit.md)가 색출한
> 갭(A-1~A-6)을 **구현 가능한 수준으로 정밀화한 계획서**다. 코드 미수정(계획 단계). 학습 비방해.
> **다음 단계**: 제3자 에이전트가 본 계획서를 검수 → 승인 후 별도 세션/분기에서 구현.
> **선행 SSOT**: 005(A1~A20, 결정 #1~#31), 007(fixed-HP), 008(snapshot), 009(arclength lap),
> 015(시나리오 B), 017(watchdog+Stage2 설계), 018(갭 감사).
> **기준선**: git `1f32804` + `configs.yaml` uncommitted(M, dead config). pytest 28 passed.
> 학습 Stage 1 `runs/stage1_map_easy3/` envs=8 진행 중(20.6%, step ~103K/500K).

---

## 0. 원칙 (구현 시 불가침)

1. **fixed-HP(007 §2-A) 불변**: train_ratio=512, batch_size=16, batch_length=64, precision=16,
   dyn_scale/kl_free/lr/discount 등 알고리즘 HP는 절대 변경 금지. 본 계획은 **학습 오케스트레이션
   계층(저장/재개/평가/샘플 혼합)만** 확장한다.
2. **환경 물리/차체/맵/reward/판정 불변**: f1tenth_env.py의 dynamics·reward 산술·lap 판정(009)·
   obs 인터페이스는 건드리지 않는다. 단 §1-1의 lap_time 진단 필드 추가는 **물리/판정/reward에
   영향 없는 info 신호 추가**이며 사용자 승인 포인트로 명시한다(§1-1 참조).
3. **dead config 재발 금지**: config 키 추가 시 반드시 `.py` 배선까지 완성. 018 §0의 교훈.
4. **분류 B(018 §2) 불가침**: 010 encoder dims, /2π 정규화, batch16, completion-only A11,
   held-out 무효화 등은 후속 SSOT와 일치하는 의도적 설계 → 건드리지 않는다.
5. **학습 비방해**: `runs/stage1_map_easy3/` 읽기만. 재시작은 사용자 결정("이따가") 후 1회.
6. **vendor 인플레이스(결정 #1)**: train.py 신설하지 않고 dreamer.py main()/save/resume +
   tools.py 샘플링을 인플레이스 확장.

## 0-1. ★ 018 정정 (코드 실측, 본 계획서 우선)

| 018 기술 | 실측 결과 | 출처 |
|---|---|---|
| §A-3 counters = train/log/**eval/vid**/reset 5개 | `_should_eval`/`_should_video` **객체 부재**. 실제 `tools.Every`는 `_should_log`/`_should_train`/`_should_reset` **3개**뿐. eval은 main while(`dreamer.py:313-326`)서 매 iteration 무조건 실행(`config.eval_episode_num>0`), Every 미사용 | `dreamer.py:33,35,37` grep |
| §A-1·§A-4 "lap_time SSOT는 arclength wrap(009)" | env info에 **`lap_time` 필드 없음**. `lap_count_arc`/`env_step`만 존재(`f1tenth_env.py:405-422`). lap_time은 별도 산출 필요(§1-1) | `f1tenth_env.py:405-422` read |

→ A-3 counter 목록은 **3개**로 구현. A-1/A-4의 lap_time은 §1-1 신설 신호로 일원화.

---

## 1. 공통 설계 결정 (A-1·A-4 선결)

### 1-1. lap_time SSOT 신설 — 사용자 승인 포인트 ★

A-1(bin 판정)·A-4(A13 median/best)가 모두 lap_time을 요구하나 env가 미제공.
**환산 SSOT**: f110 `timestep=0.01s`(100Hz) × `action_repeat=2` → **1 wrapper-step = 0.02s** =
`f1tenth_env.step`의 `self._env_step += 1` 단위(`f1tenth_env.py:290`). 즉 `lap_time_s = Δenv_step × 0.02`.

**옵션 A (권장) — env info에 진단 필드 1개 추가**:
- `f1tenth_env.py` reset에 `self._lap_start_step = 0` 추가. lap_increased(`:326`) 시
  `lap_time_s = (self._env_step - self._lap_start_step) * DT_WRAP` 계산, `self._lap_start_step = self._env_step` 갱신.
- info에 `"lap_time_s": <완주 시 값 / 그 외 None>` 추가. `DT_WRAP = 0.02`를 모듈 상수로(f110 timestep × action_repeat, 실측 주석).
- **물리/reward/lap 판정 무변경** — 순수 진단 신호. A-1/A-4가 동일 소스 소비(SSOT 일원화).
- 리스크: 환경 파일 수정 → 사용자 "물리 불변" 원칙. 단 reward/dynamics/판정 라인 무수정이라 원칙 내. **명시 승인 필요.**

**옵션 B — env 무수정, 소비처가 per-step 자체 계산**:
- A-4(독립 eval 스크립트)는 step 루프에서 `info['lap_count_arc']` 증가 순간 `info['env_step']`을
  잡아 Δ×0.02로 계산 가능 → env 무수정.
- 단 A-1(학습 중 dreamer.py eval은 `tools.simulate`라 per-step info 비노출)은 simulate 패치 필요 →
  더 침습적. **SSOT 이원화 위험.**

→ **권장: 옵션 A**(작고 명확, A-1/A-4 공유, 물리 불변). 검수자 판단 요청.

### 1-2. lap_time 정의 (검수 포인트)
A13은 "median lap_time ≤120 ∧ best ≤110"(005:187). 본 계획은 **per-lap**(lap_count_arc 증가
간 경과)으로 정의. 2-lap 완주(LAP_TARGET=2)에서 lap이 2회 증가 → lap별 lap_time_s 2개 산출.
A13 median/best는 **완주 에피소드들의 lap별 lap_time 모집단**에서 집계. 005/009 원문과의 정합은
검수에서 확인 요망(특히 "에피소드당 1 lap_time" vs "lap별" 해석).

---

## 2. A-1 🔴 snapshot 시스템 (008 전체)

### 2-1. config 키 (configs.yaml f1tenth 블록, .py 배선 필수)
```yaml
snapshot_lap_threshold: {map_easy3: 45.0, oschersleben: 110.0}  # 008 §2-1 트랙별 T
snapshot_interval_keep: True       # interval step_{N}k.pt 보존(A15)
snapshot_diversity_bins: 5         # bin 개수(008 §2-1)
```
- 008 §5: 단일 `snapshot_lap_threshold=110.0` 및 `snapshot_save_all_below_threshold` 폐기 →
  트랙별 dict + best-per-bin. task에서 trackname 추출(`config.task='f1tenth_map_easy3'` → `map_easy3`).

### 2-2. interval snapshot (A15·#10b)
- `dreamer.py:341-345` save 블록 확장: `eval_every` trigger마다 `latest.pt` 저장 **직후**,
  full ckpt를 `logdir/step_{agent._step//1000}k.pt`로 **별도 복사 저장**(덮어쓰기 아님).
- full = 현 `items_to_save`(agent+optim) 그대로. ~50개/stage, ~10GB(008 §4).

### 2-3. diversity snapshot (A14·#10a, 008 §2)
- bin 경계: `T/5` 폭 5등분. map_easy3 `(0,9],…,(36,45]`; oschersleben `(0,22],…,(88,110]`(008 §2-1).
- 학습 중 eval 에피소드의 lap_time_s(§1-1) 관측 → 해당 bin의 기존 best보다 빠르면 교체 저장.
- 파일명 `policy_lap{X:.1f}s_step{Y}k.pt`(005 A14). bin당 1개, ≤5/stage.
- **partial state_dict(008 §2-2)**: optimizer 제거. world_model + actor inference만.
  저장 함수 신설 `save_inference_only(agent, path)`: `agent.state_dict()`에서 `_wm.*`+`_task_behavior.actor.*`
  (+ encoder 등 추론 필요분)만 추려 저장. optims_state_dict 제외 → ~50MB(008 §4).
  ★ 검수 포인트: 추론에 필요한 정확한 키 집합(value/_slow_value 포함 여부 — rollout 생성엔 actor만
  필요하나 world_model heads 전부 필요). 키 화이트리스트를 코드 증거로 확정할 것.

### 2-4. lap_time 수집 경로 (A-4 의존)
- 학습 중 eval은 `dreamer.py:315-326`의 `tools.simulate(is_eval=True)`. 현재 lap_time 미집계.
- §1-1 옵션 A 채택 시: eval episode info의 `lap_time_s`가 cache에 들어감 → simulate 종료 후
  cache 또는 신규 집계 훅에서 bin 판정. simulate 로깅부(`tools.py:201-247`)에 lap_time/완주
  집계를 추가하거나, eval 후 dreamer.py가 eval episode들을 직접 순회.
- ★ 검수 포인트: 학습 중 eval(deterministic, eval_state_mean=True)이 bin 다양성을 만들 만큼
  lap_time 분포를 주는지(매 eval 유사 trajectory 가능성). 008 §3은 "interval이 주공급원, diversity는
  큐레이션"이라 했으므로 diversity 빈약은 허용 범위이나 명시할 것.

### 2-5. 테스트
- `test_snapshot_bins.py`: bin 경계 분류 함수(트랙별 T, 5등분, 경계 포함/배제) 순수 단위 테스트.
- `test_save_inference_only.py`: partial state_dict가 optim 키 제외 + 추론 키 포함 + reload 후
  forward 동작.

---

## 3. A-2 🔴 Stage 2 fine-tune (#21/#9/#31)

### 3-1. warm-load + fresh optim + lr scale (#21)
- config 키는 **이미 선언됨**(`configs.yaml:77-80`, dead). 본 작업이 배선.
- `dreamer.py:305-310` 분기 수정:
```python
_is_resume = (logdir / "latest.pt").exists()
_do_warm = (not _is_resume) and bool(config.warm_load_ckpt)
# (agent 생성 전) lr scale 1회 적용 — 옵티마이저가 scaled lr로 생성되도록
if _do_warm and config.warm_lr_scale != 1.0:
    config.model_lr   *= config.warm_lr_scale     # models.py:94
    config.actor["lr"] *= config.warm_lr_scale    # models.py:267
    config.critic["lr"]*= config.warm_lr_scale    # models.py:278
# (agent 생성·requires_grad_ 후)
if _is_resume:
    <기존 전체 load + optim load + _should_pretrain._once=False>   # dreamer.py:306-310 그대로
elif _do_warm:
    ckpt = torch.load(config.warm_load_ckpt, map_location=config.device)
    wm_state = {k: v for k, v in ckpt["agent_state_dict"].items() if k.startswith("_wm.")}
    missing, unexpected = agent.load_state_dict(wm_state, strict=False)  # actor/critic/optim fresh
    # optim 미로드, _should_pretrain 유지(world model warm이므로 _once=False 권장 — 검수 포인트)
```
- **#21 해석(017 §2 확정)**: world model **weights만** warm, actor/critic weights + 모든 optimizer fresh.
- compile=False(`configs.yaml:202`)라 state_dict 키에 `_orig_mod.` prefix 없음(017 확인) → `_wm.` 직접 매칭.
- **resume 우선**: latest.pt 있으면 warm-load 무시(전체 resume) → Stage 2 crash 시 watchdog 호환.
- ★ 검수: `_wm.` 키만으로 world model 전체가 커버되는지(encoder/decoder/dynamics/heads 모두 `_wm.` 하위인지) 코드 증거 확인.

### 3-2. joint replay (#9/#31)
- `dreamer.py:296` `make_dataset(train_eps, config)` 또는 `tools.sample_episodes`(`tools.py:327`) 확장.
- 신규 `make_joint_dataset(episodes, stage1_episodes, config)`:
```python
gen_new = tools.sample_episodes(episodes, config.batch_length)
gen_old = tools.sample_episodes(stage1_episodes, config.batch_length)
def joint_gen():
    rng = np.random.RandomState(config.seed)
    while True:
        yield next(gen_old) if rng.rand() < config.joint_replay_ratio else next(gen_new)
return tools.from_generator(joint_gen(), config.batch_size)
```
- main에서 `joint_replay_ratio>0 and joint_replay_dir`이면 `tools.load_episodes(joint_replay_dir, limit=dataset_size)`로
  Stage1 episodes 로드 후 `make_joint_dataset` 사용. 아니면 기존 `make_dataset`.
- #31: A16 미달 시 `joint_replay_ratio=0.5` — 운영 파라미터(재학습 시 CLI override), 코드 분기 불필요.
- ★ 검수: `sample_episodes`/`from_generator` 반환 element 형식이 joint_gen과 호환되는지(`tools.py:313-367` 시그니처).

### 3-3. 테스트
- `test_warm_load.py`: 더미 ckpt에서 `_wm.*`만 로드, actor/critic 파라미터는 초기값 유지, optim state 빈 상태 확인.
- `test_joint_replay.py`: ratio=0/1/0.3에서 두 풀 샘플 비율이 통계적으로 맞는지(시드 고정).

---

## 4. A-3 🟡 counter ckpt (C-N10/R7)

- **실제 Every 객체 3개만**(§0-1): `_should_log`/`_should_train`/`_should_reset`.
- save(`dreamer.py:341`)에 추가:
```python
items_to_save["counters"] = {
    "log":   agent._should_log._last,
    "train": agent._should_train._last,
    "reset": agent._should_reset._last,
}
```
- resume(`dreamer.py:306-310`)에서 복원:
```python
if "counters" in checkpoint:
    agent._should_log._last   = checkpoint["counters"]["log"]
    agent._should_train._last = checkpoint["counters"]["train"]
    agent._should_reset._last = checkpoint["counters"]["reset"]
```
- 하위호환: `counters` 키 없는 기존 latest.pt resume 시 건너뜀(현 동작 유지).
- 심각도 중(018 §A-3 실측): `Every._last=None` 재기준은 train burst가 아니라 주기 1회 어긋남 수준.
  watchdog resume이 실제 발생하므로 명세상 구현.
- 테스트 `test_counter_ckpt.py`: save→load 후 `_last` 일치.

---

## 5. A-4 🟠 평가 게이트 하니스 (A11/A12/A13/A16)

### 5-1. 신규 `scripts/eval_gate.py` (독립 실행, env 무수정 경로)
- 입력: `--ckpt <path>` `--task <f1tenth_map_easy3|f1tenth_oschersleben>` `--episodes 20`(005 A11/A13 20ep).
- watch_drive.py 패턴 재사용(make_env + Damy + Dreamer load). **eval_state_mean=True, 고정 pose**.
- step 루프에서 episode별 수집:
  - 완주 여부: `info['cause']=='lap_complete'`(2-lap 완주, LAP_TARGET=2).
  - lap_time: §1-1 `info['lap_time_s']`(옵션 A) 또는 per-step Δenv_step×0.02(옵션 B).
- 산출/판정:
  - **완주율** = 완주 ep / 20. A11(map_easy3)≥0.80, A12(osch)≥0.80, A16(map_easy3 재평가)≥0.70.
  - **lap_time**: 완주 ep들의 per-lap lap_time → median, best. A13(osch): median≤120 ∧ best≤110.
  - A11은 **completion-only**(009 결정 B) — median lap_time 게이트 없음. 005:185 GF×1.5 사용 금지(018 §A-4).
- 출력: JSON + stdout 표(완주율/median/best/판정 PASS|FAIL). `runs/<...>/eval_gate_{task}_{step}.json`.

### 5-2. (선택) 학습 중 로깅
- `dreamer.py:315-326` eval 경로에서 완주율/lap_time을 TB에 추가 로깅(A-1 bin 판정과 신호 공유).
  필수 아님(eval_gate.py가 SSOT). 중복 구현 지양.

### 5-3. 테스트
- `test_eval_gate.py`: 합성 episode info 시퀀스(완주/충돌/timeout 혼합)로 완주율·median·best·게이트
  판정 로직 검증(실모델·시뮬레이터 불요, 집계 함수 순수 테스트).

---

## 6. A-5 🟡 A17 reward component TB 로깅

- env가 `reward_{progress,collision,reverse,diverged,lap}`를 info에 정확 분리(`f1tenth_env.py:417-421`).
- `tools.simulate`는 `log_` 접두 키만 TB 기록(`tools.py:213-214`).
- **최소 침습 택1**:
  - (a) env info 키를 `log_reward_progress` 등 `log_` 접두로 rename — simulate가 자동 집계.
    단 info 키 rename은 env 출력 변경(test/소비처 영향 확인).
  - (b) simulate 로깅 수집부(`tools.py:201-247`)에 reward component 누적·기록 추가 — env 무변경.
- → **권장 (a)**(vendor simulate 무수정, env info는 진단 채널). 검수 판단 요청.
- 테스트: 기존 test_reward.py는 component 합=reward만 검증 → TB 경로용 키 존재 확인 테스트 추가.

## 7. A-6 🟢 GapFollower prefill 자동배선

- `prefill=0`(`configs.yaml:206`) 적용됨. `scripts/gf_prefill.py`는 dreamer.py 미연결(수동 선행 실행).
- **택1**: (a) 운영 절차 문서화(README/주석에 "학습 전 gf_prefill.py 실행" 명시) — 최소.
  (b) dreamer.py prefill 경로(`:262-292`)에 GF collector 옵션 배선 — 침습적.
- → **권장 (a)**(현 "별도 collector" 설계 #23 정신, 코드 리스크 0).

## 7-1. networks_1d.py stale 주석 (018 §2)
- `networks_1d.py:134-136` `# 256/# 34/# 8704` → 실제 128/17/2176(010 supersede). **주석만** 정정.

---

## 8. 의존 순서 / 재시작 전략

### 8-1. 의존 그래프
- **§1-1 lap_time 신설 → A-1, A-4** (둘 다 lap_time 소비. 선결).
- A-1(snapshot bin)은 A-4의 lap_time 산출 로직에 의존.
- A-2(fine-tune)는 Stage 1 재시작에 **dead**(latest.pt resume 경로 + warm_load_ckpt='' 기본) → Stage 1 무영향, Stage 2 진입 전 필요.
- A-3(counter)은 독립.

### 8-2. Stage 1 재시작에 즉시 작동 = §1-1 + A-1 + A-3 + (A-4 로깅분).
### 8-3. Stage 2 진입 전 필요 = A-2 + A-4(게이트 판정).
### 8-4. 재시작 불필요 = A-5, A-6, 주석.

### 8-5. 재시작 전략 — **미확정(사용자 "이따가")**
- latest resume(20%이후 snapshot) vs 처음부터(전구간, ~3h 손실). 008 §A-1 완충: train_eps는 0%부터 보존.
- 재시작 횟수 최소화: §1-1+A-1+A-3 일괄 구현→테스트→1회 재시작 권장.

---

## 9. 테스트 / 회귀 매트릭스

| 영역 | 신규 테스트 | 회귀 |
|---|---|---|
| A-1 | test_snapshot_bins, test_save_inference_only | pytest 28 |
| A-2 | test_warm_load, test_joint_replay | pytest 28 |
| A-3 | test_counter_ckpt | pytest 28 |
| A-4 | test_eval_gate(집계 순수함수) | pytest 28 |
| §1-1 | test_lap_time_signal(env info lap_time_s 산출) | test_reward/lap 28 유지 |

- 모든 보완 후 `pytest dreamer_f1tenth/tests/` **28 passed 유지** + 신규 통과.
- CPU 가능분만 자동화(world model forward는 소형이라 CPU 단위 테스트 가능).

---

## 10. 검수자 집중 포인트 (불확실·결정 필요)

1. **§1-1 lap_time를 env info에 추가(옵션 A)** 할지 — 환경 파일 수정 가부(물리 불변 원칙 vs SSOT 일원화).
2. **§1-2 lap_time 정의** — per-lap vs 에피소드당 1개. 005:187/009와 정합.
3. **§2-3 diversity partial state_dict 키 화이트리스트** — 추론에 필요한 정확한 키 집합.
4. **§2-4 학습 중 eval이 bin 다양성을 실제로 만드는지**(deterministic eval).
5. **§3-1 `_wm.` 키가 world model 전체를 커버**하는지(encoder/dynamics/heads).
6. **§3-2 joint_gen element 형식**이 from_generator와 호환되는지.
7. **A-1↔A-4 lap_time 신호 공유 경로**(simulate 집계 vs 독립 스크립트)의 SSOT 단일성.
8. 재시작 전략(§8-5) 및 현 Stage 1 run 처리 — 사용자 결정 대기.
