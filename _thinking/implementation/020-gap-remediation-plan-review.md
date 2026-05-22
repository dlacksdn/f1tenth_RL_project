# 020 — 019 구현 보완 계획서 제3자 검수 결과 (구현 에이전트용 지시서)

> 2026-05-22. [019 구현 보완 계획서](./019-gap-remediation-plan.md)를 코드로 독립 교차검증한 결과.
> **독자**: 019를 받아 코드를 보완할 다음 에이전트. 본 문서는 019의 **수정 지시 + 승인/보류 판정**이다.
> **기준선**: git `1f32804`, `configs.yaml`만 uncommitted(M). 학습 `runs/stage1_map_easy3/` 진행 중(읽기만). 코드 미수정.
> **선행 SSOT**: 005(A1~A20·결정), 007(fixed-HP), 008(snapshot), 009(arclength lap), 015(시나리오 B), 017(watchdog+Stage2), 018(갭 감사), 019(보완 계획).

---

## 0. 핵심 결론 (먼저 읽어라)

019는 018 갭 A-1~A-6을 **빠짐없이 커버**하고, §0-1 정정 2건·bin 정의·partial state_dict·`_wm.*` 커버리지·counter 3개 등 file:line 인용 대부분이 **코드와 일치**한다. 그러나 구현 착수 전 반드시 고쳐야 할 결함이 있다:

- 🔴 **B-1 (치명, 구현 차단)**: §1-1 옵션A·§2-4·A-5(a)가 모두 "env `info` → `tools.simulate` cache" 경로를 전제하나, **simulate는 info를 cache에 넣지 않고 버린다**. 이대로 구현하면 학습 중(dreamer.py) lap_time/reward 수집이 **조용히 실패**한다(테스트는 통과하는데 A-1/A-5가 작동 안 함).
- ⚠️ **B-3 (문구)**: §0 원칙1("lr 절대 변경 금지")과 §3-1(`model_lr *= warm_lr_scale`)이 표면상 충돌. 실질 위반 아님(Stage2 게이트).
- ⚠️ **줄번호 오기**: §3-1 `compile=False`는 `configs.yaml:202`가 아니라 **`:209`**.

**A-4 `eval_gate.py`(독립 스크립트)는 영향 없음 — 정상.** 문제는 학습 중(dreamer.py/simulate) 경로뿐이다.

---

## 1. 🔴 B-1 — info는 simulate cache에 도달하지 않는다 (최우선 수정)

### 증거
`tools.simulate`가 cache에 담는 transition은 **obs + action + reward + discount뿐**이다:

```python
# vendor/dreamerv3-torch/tools.py:190-199
o, r, d, info = result
transition = o.copy()                                  # obs만 복사
transition["action"] = a; transition["reward"] = r
transition["discount"] = info.get("discount", ...)     # ★ info에서 discount 1개만
add_to_cache(cache, env.id, transition)                # info 나머지 전부 폐기
```
- `tools.py:167` `obs = {k: ... for k in obs[0] if "log_" not in k}` — obs의 `log_` 키는 agent(encoder) 입력에서 strip.
- `grep -rn "log_" dreamer_f1tenth/envs/ vendor/.../wrappers.py` → **0건**. 현재 obs에 log_ 키 없음.
- `f1tenth_env.py:405-422`의 `reward_progress`/`lap_count_arc` 등은 전부 **info에만** 존재 → simulate가 폐기.

### 영향
- **§2-4 line 98** "eval info의 `lap_time_s`가 cache에 들어감" → **거짓**. cache·npz 어디에도 안 들어감.
- **§1-1 옵션A의 'A-1/A-4 SSOT 일원화'는 절반만 참**: 옵션A(info 필드)는 A-4(독립 스크립트, `env.step` 반환 info 직접 소비)엔 충분하나, **A-1(학습 중 dreamer.py eval via simulate)엔 무력**. A-1은 옵션 A/B와 무관하게 별도 배선이 필요.
- **A-5(a)** "info 키를 `log_reward_*`로 rename"만으론 TB 도달 불가.

### 수정 지시 (구현 시 이대로)
학습 중 신호 수집(A-1 lap_time, A-5 reward 분리)은 **둘 중 하나**로 구현하라:

**경로 ① (권장) — `_build_obs`에 `log_` 접두 키로 노출**
- `f1tenth_env.py`의 `_build_obs` 반환 obs dict에 `log_lap_time_s`, `log_reward_progress`/`collision`/`reverse`/`diverged`/`lap`, (필요 시 `log_lap_count_arc`/`log_completed`) 추가.
- 자동 동작: `tools.py:167`이 encoder 입력에서 strip(world model 무영향) → `:192 transition=o.copy()`가 cache에 보존 → `:205 save_episodes`로 **npz에 저장**(이게 `:213-219`의 log_ pop보다 **앞**이라 npz엔 남음) → `:213-217`이 TB에 합산 로깅.
- bin 큐레이션(A-1)은 TB 합산값이 아니라 **eval 종료 후 eval_eps npz를 직접 순회**해 per-lap `log_lap_time_s`를 읽어 판정(019 §2-4의 "dreamer.py가 eval episode 직접 순회"는 데이터가 obs에 있어야 비로소 성립).
- ⚠️ obs space 정의에 log_ 키를 추가하면 안 됨(observation_space는 5-key 유지). simulate가 obs[0].keys()만 보므로 step 반환 obs dict에만 넣으면 됨. is_first 초기 transition(`tools.py:157-163`)에도 log_ 키가 없으면 키 불일치 가능 → reset obs/`build_obs(is_first=True)`에도 동일 키(기본 0/None)를 채워 **모든 transition이 동일 키 집합**이 되게 할 것. (이 점이 경로①의 유일한 함정.)

**경로 ② — `tools.simulate` 수집부 패치**
- `:189-199` 루프에서 원하는 info 키를 transition에 합류(`transition[f"log_{k}"]=info[k]`). vendor 수정이라 더 침습적이나 obs space/키일관성 문제 없음. 결정 #1(vendor 인플레이스)과 충돌 없음.

→ **경로① 권장**(env는 진단 채널, vendor simulate 무수정). 단 "모든 transition 동일 키" 함정 반드시 처리.

### 테스트 보강 (필수)
- 019 §9의 `test_lap_time_signal`이 "env info 산출"만 검증하면 **A-1 도달은 무검증** → 통과해도 실패. **`simulate → cache/npz`에 `log_lap_time_s`가 보존되는지** 단위 테스트 추가(소형 dummy env + simulate 1 episode).

---

## 2. ✅ 코드 일치 확인 (그대로 구현 가능)

| 항목 | 검증 결과 | 증거 |
|---|---|---|
| §0-1 ① counter 3개 | `_should_log/_should_train/_should_reset`만 Every. eval/vid 객체 부재 | `dreamer.py:33,35,37`(Every), `:36`(Once), `:38`(Until), `:315` eval은 `eval_episode_num>0` 무조건(Every 미사용) |
| §0-1 ② info에 lap_time 없음 | 정확 | `f1tenth_env.py:405-422` info에 `lap_count_arc`(409)·`env_step`(413)만 |
| §1-1 환산 0.02s | 산술 정확 | `f1tenth_env.py:39` SIM_TIMESTEP=0.01, `:40`/`configs.yaml:201` action_repeat=2, `:290` `_env_step+=1` |
| §2-1 bin/threshold | 008과 일치 | `008:31-32` map_easy3 `(0,9]…(36,45]`/osch `(0,22]…(88,110]`, track-dict {45,110} `008:67` |
| §2-2 interval save | save 블록 일치 | `dreamer.py:341-345`, `eval_every=1e4`(`configs.yaml:12`)//2=5000 update step → ~50개 |
| §2-3 partial state_dict | 008과 일치 | `008:25,38` "world_model+actor inference만, optimizer 제거". value/_slow_value 불요 |
| §3-1 `_wm.*` 커버리지 | world model 전체 `_wm` 하위 | `models.py:38-88` encoder/dynamics/heads(decoder,reward,cont) 모두 `self._wm`=WorldModel |
| §3-1 lr 줄 94/267/278 | 정확 | `models.py:94`(model_lr), `:267`(actor lr), `:278`(critic lr) |
| §3-2 joint_gen 형식 | from_generator 호환 | `tools.py:327` sample_episodes는 dict yield, `:313` from_generator는 stack |
| §3-2 load_episodes | 시그니처 일치 | `tools.py:368` `(directory, limit=None, reverse=True)` |
| A-3 counter | Every._last 존재 | `tools.py:846-859`, resume 시 `_last=None`→재기준 |
| §5-1 완주 판정 | `cause=='lap_complete'` | `f1tenth_env.py:391` cause="lap_complete", LAP_TARGET=2(`:89`) |

---

## 3. ⚠️ 정정/주의 사항 (구현 시 반영)

### 3-1. §3-1 `compile=False` 줄번호 오기
- 019는 `configs.yaml:202`라 했으나 실제 **`:209`**(`:202`는 `time_limit: 18000`). f1tenth 블록(193~)에서 compile=False(`:209`)가 default compile:True(`:17`)를 override → **결론(`_orig_mod.` prefix 없음, `_wm.` 직접 매칭)은 맞음**. 줄 인용만 209로 정정.

### 3-2. §3-1 lr scale 삽입 위치
- 의사코드 라벨이 `dreamer.py:305-310`(agent 생성 후)이나, lr scale은 **반드시 agent 생성(`:298`) 이전**에 적용해야 Optimizer가 scaled lr로 생성됨. 의도는 맞으나 삽입 위치를 `:298` 앞으로 명확히.

### 3-3. §0 원칙1 ↔ §3-1 lr 충돌 (B-3)
- §0은 "lr 절대 변경 금지", §3-1은 `model_lr *= warm_lr_scale`. **실질 위반 아님**: `_do_warm = (not _is_resume) and warm_load_ckpt`로 게이트되어 Stage 1(resume/scratch, warm_lr_scale=1.0)은 무변경. lr scale은 005 #21·R3(forgetting 방어)의 **의도적 Stage 2 예외**. → §0에 "warm_lr_scale는 Stage 2 warm-load 한정 예외(005 #21)"를 명문화하라.

### 3-4. §1-1 `DT_WRAP` 하드코딩
- `DT_WRAP = 0.02` 모듈 상수보다 **`SIM_TIMESTEP * self.action_repeat`로 도출** 권장. action_repeat가 바뀌면 0.02 상수는 silent 오류. (현재 action_repeat=2 고정이라 값은 동일하나 결합도 제거.)

### 3-5. §2-3/§3-1 중복키 인지 (B-2)
- `models.py:226` `self._world_model = world_model` 때문에 `agent.state_dict()`에 world model이 **`_wm.*` + `_task_behavior._world_model.*` 두 경로로 중복**(동일 Parameter 공유).
  - **warm-load(§3-1)**: `_wm.*`만 load(strict=False)해도 공유 텐서라 동시 갱신 → 정상. **테스트에서 `_task_behavior._world_model.*`가 missing_keys로 뜨는 것은 정상**임을 인지.
  - **inference save(§2-3)**: `_wm.*` 화이트리스트로 충분(공유). value/_slow_value 제외 정당.

---

## 4. §10 검수 포인트 8개 — 확정 판정

1. **§1-1 env info 수정 가부**: **승인**(물리/reward/판정 라인 무수정 = 원칙 내). **단 §1(B-1)대로 학습 중 수집은 `log_` obs 노출/simulate 패치 병행 필수.** "옵션A=A-1/A-4 일원화" 문구는 수정.
2. **§1-2 lap_time 정의**: **per-lap 확정.** 005:596 baseline이 f110 `lap_times[0]`(per-lap) 사용, A13(005:187) "median ∧ best"는 lap 모집단 통계 전제. `Δenv_step×0.02`=lap당 실경과 sim 시간이라 정확. (A13은 절대값 120/110s라 arclength vs f110 검출 차이 무해.) 완주 ep당 lap 2개 산출 → 모집단에서 median/best.
3. **§2-3 inference 키**: `_wm.*` + `_task_behavior.actor.*`로 **확정**(008 §2-2). value/_slow_value 제외.
4. **§2-4 deterministic eval 다양성**: 정책 개선에 따라 lap_time 변화 → 시간축으로 bin 채움. 008 §3 "interval 주공급, diversity 큐레이션"상 빈약 허용. **단 B-1 수집 경로가 선결**.
5. **§3-1 `_wm.` 커버리지**: 확인 ✅. compile=False(`:209`)로 prefix 없음.
6. **§3-2 joint_gen 형식**: 호환 ✅.
7. **A-1↔A-4 SSOT 단일성**: env **산출식은 공유**, **소비 경로는 이원화 불가피**(A-4=스크립트 직접 / A-1=simulate 패치·log_ obs). "단일 경로" 주장은 "산출식 공유 + 소비 경로 2개"로 재정의.
8. **재시작 전략**: 사용자 결정 대기. §1-1+A-1+A-3 일괄 후 1회 재시작 권고 합리적.

---

## 5. 구현 순서 권고 (019 §8 보강)

1. **선결**: §1-1 env 신호 — **B-1 경로①(log_ obs 키, 전 transition 키 일관성 처리)** + 도달 테스트.
2. A-1(snapshot bin/interval, partial state_dict) — B-2 중복키 인지.
3. A-3(counter ckpt) — 독립, 즉시.
4. A-5(reward 분리 TB) — B-1 경로①에 흡수(`log_reward_*` obs 키).
5. (1~4 일괄 → pytest 28 + 신규 통과 → **1회 재시작**, 사용자 승인 후.)
6. Stage 2 진입 전: A-2(warm-load §3-1 + joint replay §3-2), A-4(`eval_gate.py` — 독립, env 무수정 경로라 영향 없음).
7. A-6 prefill 문서화, networks_1d 주석 정정 — 재시작 불요.

**완료 게이트**: `pytest dreamer_f1tenth/tests/` 28 passed 유지 + 신규(특히 §1 simulate 도달 테스트) 통과. fixed-HP(train_ratio=512/batch16/batch_length64/precision16) 무변경, 환경 물리/판정/reward 산술 무변경, 추가 config 키 .py 배선 완료(dead config 재발 0) 확인 후 종료.
