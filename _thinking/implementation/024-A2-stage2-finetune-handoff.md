# 024 — A-2 Stage2 fine-tune 핸드오프 (새 세션 실행 지시서 + 노선 컨텍스트)

작성: 2026-05-22. 선행: 022(A-4/snapshot), 023(eval 철학·zero-shot 먼저). HEAD=f5ecdab.
용도: 이 세션 clear 후 **새 세션이 A-2를 정확히 이어받기 위한 지시서**. §1 노선 배경 숙지 후 §2 프롬프트로 진행.

---

## 1. 노선 컨텍스트 (반드시 먼저 이해 — SSOT 충돌 주의)

- **015(scenario-B-confirmed)가 최신 SSOT**: 교수님 확답으로 **Oschersleben 훈련 허용 확정**,
  v3 curriculum 부활. zero-shot held-out(011/013 시나리오 A)은 **폐기**. (SSOT 룰=높은 번호 우선)
- ★ 단 **현재 코드(configs.yaml:199 주석)·일부 메모리는 아직 011 held-out 인용(stale)**. 혼동 주의.
  A-2는 015 노선의 구현분 = Oschersleben에 **의도적 적응(fine-tune)**. 이게 정상이다.
- 사용자 합의(2026-05-22): "generalization 아니라 **Oschersleben 주행시간**이 목표". fine-tune 설계 승인.
- fine-tune 정의 합의: **새 logdir에서 task=Oschersleben으로 dreamer 학습을 또 500k 돌리되,
  world model(_wm.*)만 Stage1에서 복사(actor/critic/optim fresh), map_easy3 경험 30% 섞기.**
  산출물 runs/stage2_oschersleben/latest.pt = Oschersleben 적응 모델.

### ★ 선결 운영 결정 (023): zero-shot 먼저 → fine-tune 필요성 데이터로 확정
- A-2 코드는 지금 준비 가능(015:62)하나, **실제 Stage2 실행 여부는 측정으로 결정**:
  Stage1 성숙(≈500k) 후 그 모델로 Oschersleben **zero-shot 평가**
  (`python scripts/eval_gate.py --ckpt runs/stage1_map_easy3/latest.pt --task f1tenth_Oschersleben
    --gate A12,A13 --episodes 20`). A12/A13 충족 시 **fine-tune 불필요**, 미달 시 A-2 정당화.
- 이번 A-2 분기 범위 = **코드+테스트 구현(배선)**. 실제 실행은 별도(Stage1 완료 후, 사용자 승인).

---

## 2. 새 세션 프롬프트 (그대로 사용 — HEAD 갱신 시 부팅 체크 숫자만 정정)

```
[역할] 너는 구현 에이전트다. f1tenth_RL_project Dreamer 베이스라인의 갭보완 A-2
(Stage2 fine-tune: warm-load + joint replay)를 구현한다. 직전 분기(commit f5ecdab)까지
A-4 평가 게이트(scripts/eval_gate.py)·snapshot persist(B-1)·명칭(B-2)·return 플롯
(scripts/plot_returns.py)을 완료했다. 이번 분기는 A-2 코드+테스트를 준비한다(실제 Stage2
실행은 Stage1 학습 완료 후, 사용자 승인 하).

[모델/언어] Opus 4.7(1M context). Sonnet/Haiku 금지. OMC 에이전트 소환 시 model=opus 명시.
한글로 답하라. 사용자는 한국인.

[부팅]
1) cd /home/dlacksdn/f1tenth_RL_project ; git log --oneline -1 → HEAD가 f5ecdab 여야 함.
   git status -sb → clean 기대. ★ 자동 pull/fetch/reset/checkout/clean/force 금지.
2) cd /home/dlacksdn/f1tenth_RL_project && source .venv/bin/activate.
   ★ cwd 리셋 빈번 — 매 Bash에서 절대경로 확인.
3) pytest dreamer_f1tenth/tests/ -q → 61 passed 기준선(40 + eval_gate 17 + snapshot 4).
4) ★ 학습 보호: runs/stage1_map_easy3/ 에서 envs=8 detached 학습 진행 중(watchdog
   scripts/train_watchdog.sh 가동). 읽기만, 삭제/clean/임의 재시작/방해 금지.
   ★ 생존 확인 시 pgrep 패턴이 자기 명령 셸을 매칭하는 오탐 주의 — 고유 패턴 사용:
   ps -eo pid,etime,cmd | grep "dreamer.py --configs f1tenth" | grep -v grep.
   metrics.jsonl로 step/value_mean/train_return 추이 모니터. pid 변동 의심 시 watchdog.log로
   재시작/crash 기록 확인(없으면 정상, 감시만).

[노선 — 반드시 먼저 이해] (상세: _thinking/implementation/024 §1)
- 015(scenario-B-confirmed)=최신 SSOT: Oschersleben 훈련 허용 확정, curriculum 부활.
  zero-shot held-out(011/013)은 폐기. 코드/메모리의 011 인용은 stale. A-2=Oschersleben 적응(정상).
- ★ 선결: A-2 코드는 지금 준비하되, 실제 실행 여부는 Stage1 성숙 후 Oschersleben zero-shot
  평가(eval_gate --task f1tenth_Oschersleben --gate A12,A13)로 결정. 이번 분기=코드+테스트만.

[작업 A-2] (1차 자료: 019 §3=SSOT 사양, 020 §3=검수 확정, 017=resume/Stage2 설계)
  A2-1 warm-load + fresh optim + lr scale (019 §3-1):
  - dreamer.py:305~ resume 분기 수정:
      _is_resume = (logdir/"latest.pt").exists()
      _do_warm   = (not _is_resume) and bool(config.warm_load_ckpt)
      # agent 생성 前: lr scale 1회(옵티마이저가 scaled lr로 생성되도록)
      if _do_warm and config.warm_lr_scale != 1.0:
          config.model_lr    *= config.warm_lr_scale   # models.py:94
          config.actor["lr"] *= config.warm_lr_scale   # models.py:267
          config.critic["lr"]*= config.warm_lr_scale   # models.py:278
      # agent 생성·requires_grad_ 後:
      if _is_resume:  <기존 전체 resume 그대로 — counters/snapshot_state 복원 포함>
      elif _do_warm:
          ckpt = torch.load(config.warm_load_ckpt, map_location=config.device)
          wm_state = {k:v for k,v in ckpt["agent_state_dict"].items() if k.startswith("_wm.")}
          agent.load_state_dict(wm_state, strict=False)   # actor/critic/optim fresh
          # _should_pretrain._once: world model warm이므로 _once=False 권장(검수 포인트)
  - ★ resume 우선: latest.pt 있으면 warm 무시(전체 resume) → Stage2 crash 시 watchdog 호환.
  - ★ 검수(코드 증거 필수): _wm.* 키만으로 world model 전체(encoder/decoder/dynamics/
    reward·cont heads) 커버 확인. compile=False(configs.yaml:202)라 _orig_mod. prefix 없음
    (020 §3-1 #5 확인됨). B-2 중복키(_task_behavior._world_model.* 공유텐서)는 strict=False
    missing으로 떠도 정상(020 §3-5).
  A2-2 joint replay (019 §3-2):
  - 신규 make_joint_dataset(episodes, stage1_episodes, config):
      gen_new = tools.sample_episodes(episodes, config.batch_length)
      gen_old = tools.sample_episodes(stage1_episodes, config.batch_length)
      def joint_gen():
          rng = np.random.RandomState(config.seed)
          while True:
              yield next(gen_old) if rng.rand() < config.joint_replay_ratio else next(gen_new)
      return tools.from_generator(joint_gen(), config.batch_size)
  - main(dreamer.py:296 부근): joint_replay_ratio>0 and joint_replay_dir 이면
    tools.load_episodes(joint_replay_dir, limit=dataset_size)로 Stage1 episodes 로드 후
    make_joint_dataset 사용. 아니면 기존 make_dataset.
  - ★ 검수: sample_episodes/from_generator 반환 element 형식이 joint_gen과 호환되는지
    (tools.py:313~367 시그니처 직접 확인). #31(A16 미달시 ratio=0.5)=CLI override 운영
    파라미터, 코드 분기 불필요.
  A2-3 dead config 활성화:
  - configs.yaml:77-80 warm_load_ckpt/warm_lr_scale/joint_replay_ratio/joint_replay_dir는
    선언만 됨(.py 사용처 0=dead). 본 작업이 .py 배선 → dead config 0 달성.

[작업 B] 테스트 (019 §3-3):
  - test_warm_load.py: 더미 ckpt에서 _wm.* 만 로드, actor/critic 초기값 유지, optim 빈 상태.
  - test_joint_replay.py: ratio=0/1/0.3에서 두 풀 샘플 비율 시드 고정 통계 검증.

[제약 — 엄수]
- env 물리/판정/reward 무변경. fixed-HP(train_ratio=512/batch16/batch_length64/precision16) 무변경.
- ★ Oschersleben을 Stage1 학습(runs/stage1_map_easy3)에 절대 투입 금지. Stage2=별도
  logdir(runs/stage2_oschersleben)·별도 process. 이번 분기는 코드만, 실행 안 함.
- 진행 중 Stage1 학습 방해 금지(읽기 전용).

[1차 자료 — 명시 필요분만 정독. _thinking는 append-only]
★ 필수: implementation/019 §3, 020 §3, 017, planning/015. 핸드오프 배경: implementation/024.
★ 코드 재사용: dreamer.py:296(make_dataset)·305-330(resume/warm 분기, B-1 복원 포함),
   tools.py:313-367(sample_episodes/from_generator/load_episodes), models.py:94/267/278(lr 키),
   configs.yaml:77-80(dead 키).
배경(필요시): 022/023(직전 구현·eval 철학), 011/013(폐기된 held-out — 015가 supersede, 혼동 주의).

[완료 게이트] pytest 61 passed 유지 + 신규(test_warm_load/test_joint_replay) 통과.
env/reward/판정/fixed-HP 무변경, dead config 0(4키 배선). 각 단계 file:line 근거로 보고.

[git] 분기 종료 시 implementation/025에 구현 기록 append → add → commit(트레일러
"Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>") → push origin master.
자동 pull/fetch/reset/force 금지.

[mandatory stop] 종료 시 멈추고 보고: 작업 내용 + 테스트 검증 + 다음 단계
(=Stage1 성숙 후 Oschersleben zero-shot 측정 → fine-tune 실행 여부 결정).

이제 부팅하고 학습 상태 확인 후, 019 §3·020 §3 정독하고 A2-1 → A2-2 → A2-3 → 테스트 순 진행.
```

---

## 3. 이번 세션(022~024) 산출물 요약
- 022: A-4 eval_gate.py + snapshot persist(B-1)/명칭(B-2). pytest 61 passed.
- 023: eval_gate sanity 검증 + 평가 철학(고정 pose deterministic 유지, completion+laptime 리포트).
- plot_returns.py: 학습 return 곡선(ppt용, runs/<logdir>/return_curve.png).
- 024(본 문서): A-2 핸드오프 지시서.
- commit: a448a6b → 6f2ac63 → f5ecdab (+ 본 024 커밋 예정). 모두 push origin master.
- Stage1 학습: step ~184k/500k 안정 진행(28319, 재시작 없음).
