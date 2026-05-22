# 024 — A-2 Stage 2 fine-tune 구현 (warm-load + joint replay)

> 1차 자료: 019 §3(SSOT 사양), 020 §3(검수 확정·중복키), 017(resume/Stage2 설계), 015(시나리오 B).
> 본 분기 = **코드+테스트만**. 실제 Stage2 실행은 Stage1 성숙(≈500k) 후 zero-shot 측정으로
> fine-tune 필요성 확정 뒤, 사용자 승인 하 별도 process(runs/stage2_oschersleben).

## 노선 컨텍스트 (혼동 방지)
- 015(scenario-B-confirmed)가 최신 SSOT: Oschersleben 훈련 허용, v3 curriculum 부활.
  zero-shot held-out(011/013 시나리오 A)은 폐기. A-2 = Oschersleben **의도적 적응(fine-tune)**.
- 목표 = "generalization 아니라 Oschersleben 주행시간"(사용자 합의). fine-tune 설계 승인됨.

## 구현 내역

### 신규 모듈: vendor/dreamerv3-torch/stage2_utils.py
순수 헬퍼(무거운 deps 없이 단위 테스트 + dead config 0 배선). snapshot_utils/eval_gate 패턴.
- `extract_warm_state(agent_state_dict)` → `{k:v ... if k.startswith("_wm.")}`.
  world model weights만(actor/critic/optim 제외). #21 해석(017 §2).
- `joint_episode_generator(gen_old, gen_new, ratio, seed)` → `rng.rand()<ratio`면 old(Stage1).
- `make_joint_dataset(episodes, stage1_episodes, config)` → make_dataset(dreamer.py:147)과
  동일 인터페이스(from_generator(*, batch_size)). sample_episodes가 `log_` strip → 두 풀 키 일관.

### dreamer.py 배선 (in-place, 결정 #1)
- import에 `stage2_utils` 추가 (dreamer.py:20).
- make_dataset 호출부(dreamer.py:298 부근) 직전에 분기 판정:
  - `_is_resume = (logdir/"latest.pt").exists()`
  - `_do_warm = (not _is_resume) and bool(config.warm_load_ckpt)`  ← **resume 우선**(crash 시 watchdog 호환)
- joint replay: `joint_replay_ratio>0 and joint_replay_dir`이면
  `tools.load_episodes(joint_replay_dir, limit=dataset_size)` → `make_joint_dataset`. 아니면 기존 make_dataset.
- warm_lr_scale: **agent(Optimizer) 생성 전**(dreamer.py:319, 020 §3-2 정정)에 1회 적용 —
  `config.model_lr/actor["lr"]/critic["lr"] *= warm_lr_scale` (models.py:94/267/278). scale=1.0/resume은 무변경.
- resume 분기(dreamer.py:332 `if (...).exists()` → `if _is_resume:`)에 `elif _do_warm:` 추가(dreamer.py:357~):
  `torch.load(warm_load_ckpt, map_location=device)` → `extract_warm_state` → `load_state_dict(strict=False)`
  → `_should_pretrain._once=False`(world model warm이므로, resume과 동일·020 검수 권장). missing/unexpected 로깅.

### A2-3 dead config 0
configs.yaml:77-80의 4키(warm_load_ckpt/warm_lr_scale/joint_replay_ratio/joint_replay_dir)가
이제 dreamer.py에서 사용됨(grep 증거: dreamer.py:302/305/319/362, stage2_utils.py:53). configs.yaml 무수정.

## 검수 포인트 (코드 증거)
- **_wm.* 커버리지**: models.py:38-88 encoder/dynamics/decoder/reward·cont heads 전부 `self._wm`=WorldModel 하위.
  compile=False(configs.yaml:209)라 `_orig_mod.` prefix 없음 → `_wm.` 직접 매칭(020 §3-1 #5).
- **B-2 중복키**: models.py:226 `self._world_model = world_model` 공유 참조 → world model이
  `_wm.*` + `_task_behavior._world_model.*` 두 경로 중복(동일 Parameter). warm-load는 `_wm.*`만 로드해도
  공유 텐서라 동시 갱신. `_task_behavior._world_model.*`가 missing_keys로 뜨는 것은 **정상**(테스트로 확인).
- **joint_gen 형식**: tools.sample_episodes(dict yield, log_ strip) → from_generator(stack) 호환(테스트로 확인).

## 테스트 (신규 10개, 총 61→71 passed)
- test_warm_load.py(4): _wm.* 만 추출, actor/critic 초기값 유지, optim 키 제외,
  B-2 중복키 missing 정상, 공유 텐서 동시 갱신. 소형 nn.Module(실모델 불요).
- test_joint_replay.py(6): ratio=0(전부 new)/1(전부 old)/0.3/0.5 통계적 비율(±0.02), 시드 재현성,
  make_joint_dataset 실 episode dict로 batch shape·log_ strip·연속 생성 호환.

## 제약 준수
- env 물리/판정/reward 무변경. fixed-HP(train_ratio=512/batch16/batch_length64/precision16) 무변경.
- Oschersleben을 Stage1(runs/stage1_map_easy3)에 미투입. Stage2는 별도 logdir/process(미실행).
- Stage1 학습 읽기 전용(pid 28319 유지, step 진행 확인). watchdog 3회 가동 = 의도적 재기동
  (train.log:649 "새 코드 resume 재기동 commit a448a6b"), crash 아님. grad_norm Inf/NaN 17/378
  ≈4.5% = precision16 GradScaler 정상 동작(발산 아님).

## 다음 단계
Stage1 성숙(≈500k) 후 그 모델로 Oschersleben **zero-shot 평가**:
`python scripts/eval_gate.py --ckpt runs/stage1_map_easy3/latest.pt --task f1tenth_Oschersleben --gate A12,A13 --episodes 20`
→ A12(완주율≥0.80)/A13(median≤120∧best≤110) 충족 시 fine-tune 불필요. 미달 시 A-2 정당화 →
사용자 승인 하 Stage2 실행(warm_load_ckpt + joint_replay_dir/ratio CLI override, 별도 logdir).
