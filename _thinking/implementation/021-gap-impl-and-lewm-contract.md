# 021 — 갭 보완 구현(1~4단계) + LeWM 정본 data contract 분석

> 2026-05-22. 020(검수 지시서)을 1차 자료로 018 갭 A-1/A-3/A-5 + §1-1(env 신호)을 구현하고,
> 그 과정에서 LeWorldModel(LeWM) 정본 코드를 분석해 snapshot 정책을 사용자 결정대로 재정의한 기록.
> append-only. 기준선 git `d76d2ce`. 학습 `runs/stage1_map_easy3/` envs=8 진행 중(읽기만, 비방해).

---

## 0. 요약

- 020 §5 구현 순서 1~4 일괄 구현 완료. pytest **40→38 passed**(테스트 통합으로 개수 변동, 모두 통과).
- B-1(치명) 해결: env 진단 신호를 **`log_` obs 키(경로①)**로 노출 → `tools.simulate`가 cache/npz에 보존.
- LeWM 정본 코드(`lucas-maes/le-wm`, official) 분석 결과 **offline dataset = (pixels 이미지, action) 시퀀스**임을 확정 → snapshot policy 저장의 위상 재정의 + bin 정책을 사용자 결정대로 변경.
- 환경 물리/차체/마찰/reward 산술/lap 판정(009) **무변경**. fixed-HP(007) **무변경**. dead config **0**.

---

## 1. 구현 내역 (file:line 근거)

### 1-1. env `log_` 진단 신호 (B-1 경로①, A-1 lap_time + A-5 reward 분리)
[dreamer_f1tenth/envs/f1tenth_env.py](../../dreamer_f1tenth/envs/f1tenth_env.py):
- `__init__`: `self._dt_wrap = SIM_TIMESTEP * self.action_repeat`(0.02 하드코딩 회피, 020 §3-4),
  `self._lap_start_step = 0`.
- `_build_obs`: `_LOG_KEYS` 8개(`log_lap_time_s`, `log_reward_{progress,collision,reverse,diverged,lap}`,
  `log_lap_count_arc`, `log_completed`)를 obs에 추가. **observation_space는 5-key 유지**(log_ 미선언).
  `log_fields=None`(reset/is_first)이면 전부 0 → **모든 transition 동일 키 집합**(경로① 함정 처리).
- `step`: lap 증가 step에만 `lap_time_s = (env_step - lap_start_step) * dt_wrap` 산출(순수 진단,
  terminated/reward/판정에 **미피드백**). reward component·lap_time을 `log_fields`로 흡수.
- **불변**: F110Env 생성부(마찰/차체 params), reward 산술, lap 판정, reverse/diverge guard, 종료 우선순위.

도달 검증: `tools.simulate`는 `log_` 키를 encoder 입력에서 strip(tools.py:167)·`sample_episodes`도
strip(tools.py:347/360) → **world model 무영향**. `transition=o.copy()`(:192)로 cache 보존,
`save_episodes`(:205)가 log_ pop(:213-219)보다 **앞** → npz 보존.

### 1-2. A-1 snapshot (interval + diversity + global best)
[vendor/dreamerv3-torch/snapshot_utils.py](../../vendor/dreamerv3-torch/snapshot_utils.py) 신규(순수 함수):
- `inference_state_dict`/`save_inference_only`: partial = `_wm.*` + `_task_behavior.actor.*`만
  (008 §2-2, 020 §4-3). value/_slow_value/optimizer 제외. B-2(중복키 `_task_behavior._world_model.*`)
  는 공유 텐서라 `_wm.*`만으로 충분.
- `save_interval_snapshot`: `latest.pt` 직후 full ckpt를 `step_{N}k.pt`로 별도 보존(A15).
- `lap_time_bin`/`update_diversity_snapshots`: **§3 사용자 결정대로** 재작성.
[vendor/dreamerv3-torch/dreamer.py](../../vendor/dreamerv3-torch/dreamer.py):
- save 블록: `counters`(A-3) + interval snapshot 호출.
- eval 직후: diversity+best snapshot 호출(`collect_eval_lap_times`로 eval npz의 per-lap lap_time 순회).
[configs.yaml](../../vendor/dreamerv3-torch/configs.yaml) f1tenth 블록: `snapshot_bin_width: 10.0`,
  `snapshot_lap_max: 110.0`, `snapshot_interval_keep: True` (전부 .py 배선 = dead config 0).

### 1-3. A-3 counter ckpt
dreamer.py save에 `counters={log,train,reset}`(`_should_*._last`) 저장, resume에 복원(하위호환:
`"counters"` 부재 시 skip). watchdog resume 시 Every 주기 위상 보존.

### 1-4. A-5 reward 분리 TB
별도 작업 없이 1-1의 `log_reward_*` obs 키로 흡수 → simulate가 자동 TB 합산(tools.py:213-217).

### 테스트
- `test_log_signal.py`(4): env log_ 키 일관성, dt_wrap 도출, **simulate→npz 도달**, sample_episodes strip.
- `test_snapshot.py`(6): 10초 bin 경계, partial 화이트리스트, reload, lap 수집, diversity+best 교체, interval.

---

## 2. LeWM 정본 data contract 분석 (`~/le-wm`, official)

공식성 확정: GitHub repo description "**Official code base for LeWorldModel**", owner `lucas-maes`
= 논문 공동 제1저자, HF `quentinll`, `le-wm.github.io`, arXiv 2603.19312. 타 검색결과는 fork/mirror.

코드 레벨 contract([jepa.py:29-45 `encode`](file:///home/dlacksdn/le-wm/jepa.py), train.py, config/train):

| 항목 | LeWM 요구 | 출처 |
|---|---|---|
| observation | **RGB pixels(이미지)**, ViT(`vit_hf`, image_size 224, patch 14) cls token | jepa.py:34-38 |
| action | 임의 action_dim, `action_encoder.input_dim = frameskip(5)×action_dim` | train.py:68 |
| 포맷 | **HDF5(.h5)/LanceDB(.lance)**, `swm.data.load_dataset` | README §Data |
| 시퀀스 | `num_steps = num_preds(1)+history_size(3)=4`, frameskip 5 | data/*.yaml |
| proprio/state | keys_to_load엔 있으나 **`jepa.encode` 미사용**(pixels cls token만) | jepa.py:34-45 |
| 학습 | encoder/predictor를 **scratch 학습**("no pre-trained representations") | README abstract |

### 정합성 결론 (train_eps npz ↔ LeWM)
- **action**: npz `action (T,2)` → ✅ 그대로 사용 가능.
- **observation**: npz는 `lidar (T,1080)+state (T,5)` **벡터**, LeWM은 **pixels(이미지)** → ❌ 모달리티 불일치.
  현 f1tenth 학습은 decision #14 vector-only라 npz에 pixels 자체가 없음.
- ⇒ LeWM은 ① policy weights를 받지 않고(scratch), ② pixels 데이터를 받음 → **008 §2-2의 전제
  "policy(world_model+actor)=LeWM offline dataset 생성원"은 정본 코드와 부합하지 않음**. 진짜 입력은
  (이미지 프레임, action) 시퀀스의 .h5.

### 미결(별도 계획서 예정 — 사용자 지시 2026-05-22)
LeWM 연계 방향: **(a) f1tenth 렌더 이미지 수집 파이프라인** vs **(b) LeWM encoder를 lidar용으로 변형**.
중요 설계 분기 → 추후 각잡고 별도 계획서 작성. 본 분기에서는 미결정.

---

## 3. snapshot 정책 재정의 (사용자 결정 2026-05-22)

008 §2-1 "트랙별 임계 T를 5등분"을 **폐기**하고 다음으로 대체(사용자 지시가 008 SSOT보다 우선):
- **diversity bin = 10초 고정 폭**, 상한 110s. 110/100/90/…/10 구간별 **최단 lap policy 1개**.
  파일 `policy_lap{X:.1f}s_step{Y}k.pt`(bin당 1개, 더 빠르면 교체+옛 파일 삭제).
- **global best policy 1개** 별도 슬롯으로 계속 갱신. 파일 `policy_best_lap{X:.1f}s_step{Y}k.pt`.
- partial state_dict(`_wm.*`+actor.*)는 유지(추론·분석용). LeWM 입력 여부와 무관하게 Dreamer 자체
  평가/다양성 산출물로서 보존.

---

## 4. 검증 / 무결성
- pytest `dreamer_f1tenth/tests/` **38 passed**(28 기존 + log_signal 4 + snapshot 6, 통합으로 개수 변동).
- config 신규 키 파싱 정상(`snapshot_bin_width=10.0`/`snapshot_lap_max=110.0`/`interval_keep=True`),
  제거 키(`snapshot_lap_threshold`/`snapshot_diversity_bins`) 부재. **dead config 0**.
- fixed-HP·환경 물리·reward·lap 판정 무변경(diff는 전부 additive, 기존 계산 라인 무수정).

## 5. 다음 단계
- 재시작: latest-resume(사용자 결정). watchdog 정지→학습 정지→latest.pt resume→watchdog 재기동(1회).
- A-2(warm-load/joint replay), A-4(eval_gate.py)는 Stage 2 진입 전 별도 분기.
- LeWM 연계(이미지 vs 변형) 별도 계획서.
