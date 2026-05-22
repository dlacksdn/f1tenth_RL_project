# 018 — 계획↔구현 추적성 갭 감사 (Phase 3 train.py 누락 전수 색출)

> 2026-05-22. 제3자 독립 검수(코드 미수정, 검수 전용). 학습(`runs/stage1_map_easy3/`, PID 32361, `--envs 8`) 비방해 읽기.
> **목적**: 본 문서를 읽고 **코드를 보완할 다음 에이전트**를 위한 갭 명세서. 모든 판정은 file:line 코드 증거로 뒷받침. 추정 없음.
> **선행 SSOT**: [005 v3](../planning/005-f1tenth_dreamerV3_version3.md)(A1~A20, 결정 #1~#31), [007](../planning/007-fixed_hp_fidelity.md)(fixed-HP), [008](../planning/008-snapshot_policy_refinement.md)(snapshot SSOT), [009](../planning/009-lap-detection-and-A11.md)(arclength lap), [013](../planning/013-heldout-protocol-rev2.md)/[015](../planning/015-scenario-B-confirmed.md)(held-out), [017](./017-watchdog-and-stage2-design.md)(watchdog+Stage2 설계).
> **검수 기준선**: `git` 1f32804, `git status` = `vendor/dreamerv3-torch/configs.yaml`만 uncommitted(M). `pytest dreamer_f1tenth/tests/` = **28 passed**.

---

## 0. 구조적 근본 원인 (확정)

`train.py`는 리포 어디에도 **존재하지 않는다** (`find . -name train.py` 0건). 학습 진입점은 vendor 원본 `vendor/dreamerv3-torch/dreamer.py`이며, 메인 루프 `dreamer.py:341-345`는 `eval_every`마다 **`latest.pt` 단 하나만 덮어쓰는 vendor 스톡 로직 그대로**다:

```python
# dreamer.py:341-345 (현재)
items_to_save = {
    "agent_state_dict": agent.state_dict(),
    "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
}
torch.save(items_to_save, logdir / "latest.pt")
```

005/008/015/017이 **"Phase 3 train.py 구현 사양"**으로 명세한 학습 오케스트레이션 일체(snapshot 3종, counter ckpt, fresh-optim warm-load, joint replay, 평가 게이트 산출)가 이 한 줄짜리 저장부에 통합되지 않았다. 008/015/017이 `configs.yaml`에 추가한 키들은 **어떤 .py도 읽지 않는 dead config**다.

검증 명령:
```
grep -rn "warm_load_ckpt\|warm_lr_scale\|joint_replay_ratio\|joint_replay_dir" --include='*.py' vendor/ dreamer_f1tenth/ scripts/
# → NO .py REFERENCES (configs.yaml:77-80 선언만 존재)
grep -rn "snapshot\|policy_lap\|step_.*k\.pt\|counters\|fresh_optim" --include='*.py' vendor/ dreamer_f1tenth/ scripts/
# → 저장/배선 로직 0건
ls runs/stage1_map_easy3/   # → latest.pt만. step_*.pt / policy_lap*.pt 0개
```

**결론**: 갭은 환경/모델 계층이 아니라 **학습 오케스트레이션 계층(Phase 3)에 집중**되어 있다. 보완 작업의 본체는 "dreamer.py의 main() / save 경로 / resume 경로 / dataset 샘플링을 train.py 사양대로 확장"하는 것이다.

---

## 1. 분류 A — 진짜 추적성 갭 (보완 대상)

### A-1. 🔴 snapshot 시스템 (008 전체 통째 미구현) — 진행 중 데이터 소실

| 사양 출처 | 요구 | 현재 코드 | 분류 |
|---|---|---|---|
| 005 A15·#10(b), 008 §2 | interval `step_{N}k.pt` (`eval_every=1e4`마다 full ckpt 별도 보존, ~50개/stage) | `dreamer.py:345` 동일 경로 덮어쓰기 | 미구현 |
| 005 A14·#10(a), 008 §2 | diversity `policy_lap{X:.1f}s_step{Y}k.pt` | repo 전체 `policy_lap` 문자열·로직 0건 | 미구현 |
| 008 §2-1 | 트랙별 임계 T를 5등분한 lap-time bin마다 **fastest 1개**만 (최대 5개/stage) | bin 로직 부재 | 미구현 |
| 008 §2-2 | diversity는 **optimizer state 제거** partial state_dict (~50MB) | partial-save 함수 부재 | 미구현 |
| 008 §2-1, §5 | config 키 `snapshot_lap_threshold`를 **트랙별 dict**(`map_easy3=45.0, Oschersleben=110.0`), `snapshot_save_all_below_threshold` → **best-per-bin** 로직 | `configs.yaml` f1tenth 블록에 키 자체 없음(line 194 주석 언급뿐) | 미구현 |

**bin 정의 (008 §2-1, 그대로 구현)**:
- map_easy3 (T=45): `(0,9],(9,18],(18,27],(27,36],(36,45]`
- Oschersleben (T=110): `(0,22],(22,44],(44,66],(66,88],(88,110]`
- bin마다 관측된 최저 lap_time policy 1개만 유지(더 빠른 게 나오면 교체). bin당 5개 → diversity 총 ≤10개(2 stage).

**구현 위치**: `dreamer.py:341-345` save 블록을 확장. eval 결과에서 lap_time을 받아(→ A-4 평가 하니스에 의존) bin 판정. lap_time SSOT는 arclength wrap 시점(009). interval은 step 카운터로 trigger.

**⚠️ 진행 중 손실 경고**: 현재 Stage 1 학습은 `latest.pt`만 갱신하므로 interval/diversity 스냅샷이 0개씩 쌓이고 있다. LeWorldModel offline dataset의 다양성 주공급원(008 §3)이 학습 종료 시 최종 step 1개로 소실된다. **부분 완충**: `runs/stage1_map_easy3/train_eps/*.npz` replay 버퍼는 누적 중 → behavioral 데이터 일부 회수 가능(단 008이 명세한 "정책 스냅샷" 산출물은 아님). 현 run을 끝까지 두고 snapshot 코드 주입 후 resume할지 / Stage 1 다양성을 포기할지는 **사용자 결정 사항**(학습 비방해 원칙).

### A-2. 🔴 Stage 2 fine-tune 메커니즘 (#21 / #9 / #31 dead config)

| 사양 출처 | 요구 | 현재 코드 | 분류 |
|---|---|---|---|
| 005 #21, 015 §2, 017 §3 | `--fresh_optim` (또는 `warm_load_ckpt`): latest.pt 부재 시 지정 ckpt에서 **world model weights만 warm load**, actor/critic + 모든 optimizer는 fresh. lr×0.5(`warm_lr_scale`) | `fresh_optim` grep 0건. `dreamer.py:306-309`는 latest.pt 존재 시 agent+全 optim 무조건 strict load | 미구현 |
| 005 #9, 015 §2 | joint replay 30%: 매 배치 element를 `joint_replay_ratio` 확률로 `joint_replay_dir`(Stage1 train_eps)에서 샘플 | `tools.py:327` `sample_episodes`는 스톡 단일 디렉토리. 혼합 로직 전무 | 미구현 |
| 005 #31 | A16 미달 시 joint_replay 0.5로 강화 | #9 종속 → 경로 부재 | 미구현 |

**dead config 증거**: `configs.yaml:77-80`에 `warm_load_ckpt/warm_lr_scale/joint_replay_ratio/joint_replay_dir` 선언되어 있으나 .py 참조 0건. 즉 현재 `--warm_load_ckpt`를 CLI로 줘도 무시된다.

**구현 위치**:
- warm-load: `dreamer.py:306-309` resume 블록 수정. `warm_load_ckpt`가 비어있지 않고 `logdir/latest.pt`가 부재일 때, 지정 ckpt에서 `_wm.*` 키만 `strict=False`로 load, optimizer는 load 안 함. `warm_lr_scale`를 model/actor/value lr에 1회 곱.
- joint replay: `dreamer.py:296` `make_dataset` 또는 `tools.sample_episodes`(`tools.py:327`)를 확장 — `joint_replay_ratio>0`이면 배치 element를 확률적으로 두 디렉토리(현 train_eps / joint_replay_dir)에서 혼합 샘플.

### A-3. 🟡 counter ckpt (C-N10 / R7)

| 사양 출처 | 요구 | 현재 코드 |
|---|---|---|
| 005 §0-2 C-N10, Phase 3, R7 | `checkpoint['counters'] = {n: c._last for n,c in [('train',_should_train),('log',_should_log),('eval',_should_eval),('vid',_should_video),('reset',_should_reset)]}`, resume 시 복원 | `dreamer.py:341-345` `items_to_save`에 `counters` 없음. grep 0건 |

**구현 위치**: save 블록(`dreamer.py:341`)에 counters dict 추가, resume 블록(`dreamer.py:306-310`)에서 복원.
**실측 주의(과대평가 금지)**: `tools.Every.__call__`(`tools.py:849-858`)은 resume 시 `_last=None`을 현 step으로 재기준 → 005 R7이 우려한 "train burst"보다는 eval/log 주기 1회 어긋남 수준. watchdog resume이 실제 발생하므로 명세상 필요하나 심각도는 중.

### A-4. 🟠 평가 게이트 산출 하니스 (A11/A12/A13/A16)

| 사양 출처 | 요구 | 현재 코드 |
|---|---|---|
| 005 §2-3, 009 §3·§5 | A11 2-lap **완주율**(completion-only, ≥80% 잠정), A12 Osch 완주율 ≥80%, A13 Osch median lap_time ≤120 ∧ best ≤110, A16 easy3 재평가 완주율 ≥70% | eval은 `tools.simulate(is_eval=True)`가 `eval_return/eval_length`만 로깅. **완주율·median/best lap_time 집계 코드 부재.** `scripts/eval_*.py` 없음 |

**판정에 필요한 신호는 env가 이미 내보냄**: `info['lap_count_arc']`(SSOT, `f1tenth_env.py:409`), `info['cause']=='lap_complete'`, lap_time은 arclength wrap 시점 기반(009). **소비처(집계기)만 신설하면 된다.**
**구현 위치**: 신규 `scripts/eval_gate.py`(20 ep 고정 pose, eval_state_mean=True), 또는 dreamer.py eval 경로(`dreamer.py:315-326`)에서 episode info를 수집해 완주율/lap_time median·best 산출 후 게이트 판정·로깅.
**주의**: A11은 009 결정 B로 **completion-only**(median lap_time 게이트 제거). 005 line 185 원문(GF×1.5) 사용 금지.

### A-5. 🟡 A17 reward 분리 TensorBoard 로깅 (부분구현 — 경로 단절)

| 사양 출처 | 요구 | 현재 코드 |
|---|---|---|
| 005 A17 | `reward/progress`, `reward/collision`, `reward/lap_complete`, `reward/reverse` TensorBoard 분리 | env가 `reward_progress/collision/reverse/diverged/lap`를 `info`에 정확히 분리(`f1tenth_env.py:416-421`). 그러나 `tools.simulate`는 `log_` 접두 키만 TB 기록(`tools.py:213-214`) → **info 키가 TB에 안 감** |

**구현 위치**: env info 키를 `log_reward_progress` 등 `log_` 접두로 바꾸거나, `tools.simulate`의 로깅 수집부를 패치해 reward component를 누적·기록. test_reward.py는 component 합=reward만 검증하므로 TB 경로는 무검증.

### A-6. 🟢 GapFollower prefill 자동배선 (#23 부분구현)

`prefill=0`(`configs.yaml:206`)은 적용됨. 단 `scripts/gf_prefill.py`는 dreamer.py에 import 0건 — **사용자가 수동 선행 실행해야만** 첫 10K GF 수집이 일어난다(dreamer.py:278 random_agent는 prefill=0이라 미실행). "별도 collector" 문구상 허용 범위이나 자동 파이프라인 미연결. **조치**: 자동 배선 또는 운영 절차 문서화.

---

## 2. 분류 B — 문서로 갱신된 의도적 변경 (갭 아님, 보완 대상 아님)

005 원문과 다르나 **후속 SSOT가 존재하고 코드가 최신 문서와 일치**. 추적성 양호. **건드리지 말 것.**

| 005 원문 | 코드 현재값 | 갱신 근거 SSOT |
|---|---|---|
| #5/#16 ConvEncoder1D 5-stage, flatten 256×34=8704 | 6-stage, ch cap 128, flatten 128×17=**2176**, `Linear(2176,512)` (`networks_1d.py:106,136`) | [010](../planning/010-encoder1d-dims-A10-correction.md) (A10 26.58M→[10,14]M) |
| #15/A_norm `ang_vel_z/π` | `/2π` (`f1tenth_env.py:50`) | implementation/005 §2-1 (실측 99%=6.22) |
| A9 batch_size=8 | **16** (`configs.yaml:210`) | A19 dryrun 재정정(주석 명시) |
| A11 median lap_time 게이트 | completion-only 2-lap (`f1tenth_env.py:89` `LAP_TARGET=2`) | [009](../planning/009-lap-detection-and-A11.md) 결정 B |
| 013 held-out `eval_heldout.py` + A_heldout 가드 | 부재 | [015](../planning/015-scenario-B-confirmed.md) 시나리오 B 확정으로 **명시 무효화** → 구현 대상 아님 |

**⚠️ stale 주석 1건(코드 동작 무관, 정정 권장)**: `networks_1d.py:134-136`의 `# 256 / # 34 / # 8704` 주석은 실제 값(128/17/2176)과 어긋남. 010 supersede 미반영 잔재.

---

## 3. 충실히 구현됨 (재확인, 회귀 방지 — 보완 시 깨지 말 것)

증거 확인 완료: #1 vendor fork, #4 action 2-dim/affine/imag_gradient=dynamics, #6 state 5-dim, #14 preprocess image KeyError 패치(`models.py:184` `if "image" in obs`), #19 eval_state_mean(discrete RSSM 대응 `dreamer.py:94-101`), #22 wrapper 체인(`dreamer.py:202-211`, TimeLimit env-step 단위), #25 map 명명, #26 compile=False, #27 `base_classes.py:488` `vel*sin(slip_angle)`, #30 lap_times 시작값, #28 GapFollower 측정 스크립트, 009 arclength windowed-closest-point lap 검출(`f1tenth_env.py:235-328`), 007 §2-A universal-HP 미override(f1tenth 블록이 dyn_scale/kl_free/lr/train_ratio/discount 등 default 상속), 017 watchdog(`scripts/train_watchdog.sh`). **pytest 28 passed.**

---

## 4. 보완 우선순위 (다음 에이전트 작업 순서)

### (1) 학습 재시작 전 — 데이터 소실 진행 중
- **A-1 snapshot 시스템(008)**: 단, 현 Stage 1 run 처리 방침은 사용자 결정 후 착수(학습 비방해).
- **A-3 counter ckpt(C-N10)**: watchdog resume 정합. 심각도 중.

### (2) Stage 2 진입 전 필수
- **A-2 fresh-optim warm-load + joint replay + #31 rollback**: dead config 배선. forgetting 방어(R3)가 통째 미작동 상태.
- **A-4 평가 게이트 하니스(A11~A16)**: Stage 1 종료 게이트 판정에 필요. env 신호는 준비됨, 집계기만 신설.

### (3) 영향 적음
- **A-5** A17 reward TB 로깅 경로 연결.
- **A-6** GapFollower prefill 자동배선/문서화.
- `networks_1d.py:134-136` stale 주석 정정.

---

## 5. 작업 시 준수 사항
- 보완은 **vendor `dreamer.py` main()/save/resume + `tools.py` 샘플링** 확장이 본체(train.py 신설 또는 dreamer.py 인플레이스 — 결정 #1 vendor-in 정책상 인플레이스 권장).
- **분류 B 항목(§2)은 건드리지 말 것** — 후속 SSOT와 일치하는 의도적 설계.
- config 키를 추가하면 반드시 .py 배선까지 완성(dead config 재발 금지).
- 보완 후 `pytest dreamer_f1tenth/tests/` 28 passed 회귀 확인 + 신규 기능 테스트 추가.
- `runs/stage1_map_easy3/` 학습 산출물 읽기만, 삭제/clean/재시작 금지.
