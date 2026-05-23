# 029 — Stage2 학습 정지 + snapshot bin 세분화 + watch_drive --best (2026-05-23)

> 목표 달성 판단으로 Stage2 학습 정지(나중에 resume 가능). 부수적으로 snapshot bin 2초 세분화,
> watch_drive에 --best(현재 run best 자동 시연) 추가. hang 수정 commit ffdafba 이후 작업.

## 1. Stage2 학습 정지 (2026-05-23 21:xx)
- dreamer/watchdog/poll 전부 종료(자기매칭 방지 ps+awk pid 추출 → kill). 종료 순서:
  **watchdog 먼저**(안 그러면 dreamer kill 시 watchdog가 resume) → poll → dreamer(SIGTERM).
- 정지 시점: **마지막 metrics step 171002** (eval_return 시작 -2.7 → peak 296(120k) → 변동).
  `runs/stage2_oschersleben/latest.pt`(step ~166k, 마지막 eval 저장본) 보존 → **resume 가능**.
- GPU 반환됨. 원본 runs/stage1_map_easy3 불변.

### ★ 나중에 이어서 학습(resume) 방법
```bash
cd /home/dlacksdn/f1tenth_RL_project
setsid nohup ./scripts/stage2_watchdog.sh \
  > runs/stage2_oschersleben/watchdog.log 2>&1 < /dev/null &
```
- latest.pt 존재 → watchdog/dreamer가 **resume**(warm-load 아님, scaled-lr 복원). joint_replay 계속 적용.
- hang 가드(027/028) 적용돼 있어 재시작 안전. train_eps 이미 성숙(npz 다수).
- 마일스톤 그래프 폴링 다시 원하면 /tmp/stage2_poll.sh 재기동(200k/300k/400k + 500k 트리거).

## 2. snapshot bin 세분화 (configs.yaml)
- 사용자 의도: 원래 "110초/10초"는 **2바퀴 완주 총시간** 기준이었으나, 시스템은 **1바퀴(per-lap)**
  기준으로 동작 중. 1바퀴 기준 그대로 두고 bin만 세분화하기로.
- 변경: `snapshot_bin_width.oschersleben` **10.0 → 2.0**. `snapshot_lap_max` 110.0 **유지**.
- 결과: (0,2],(2,4],…,(108,110] 55개 bin, 각 구간 최단 lap 1개 보존. 실측 1바퀴 완주 16~38초라
  실제론 (14,16]~(36,38]만 채워짐(40~110초 bin은 비어있어도 무해).
- ★ 적용은 재시작 시 1회 로드 → 정지 전 재시작(step 165994 resume)으로 이미 반영됨.

## 3. watch_drive.py — --best 옵션 + strict=False (시연 UX 개선)
- 문제: `latest.pt`는 "마지막 상태"지 best 아님(map_easy3는 224k 6.1초 best 후 250k+ 퇴보). best
  추적 파일은 `policy_best_lap*.pt`(run best, 더 빠른 lap마다 교체)인데 partial(48MB)이라
  strict 로드 거부됨.
- 추가:
  - `--best`: logdir의 `policy_best_lap*.pt` 중 mtime 최신 자동 선택(파일명 가변 대응 → 항상 현재 best).
  - `load_state_dict(strict=False)`: partial(=_wm+actor)은 critic/value 키 없음. 시연(actor.mode()+
    world model latent)은 critic 미사용이라 missing 무시 로드해도 추론 정상. full ckpt면 missing 0.
- 시연 명령(항상 현재 best 자동):
  ```bash
  # map_easy3 최고
  python scripts/watch_drive.py --logdir runs/stage1_map_easy3 --task f1tenth_map_easy3 --best --episodes 3
  # oschersleben 최고
  python scripts/watch_drive.py --logdir runs/stage2_oschersleben --task f1tenth_Oschersleben --best --episodes 3
  ```
- 고정 시점 시연은 `--ckpt <step_Xk.pt>`(full) 사용. partial `policy_*.pt`는 --ckpt로 줘도 이제
  strict=False라 로드 가능.

## 4. KEEP 백업 현황 (runs/stage2_oschersleben/KEEP/, .gitignore *.pt라 git 제외·로컬 보존)
- KEEP_oscher_lap17.4s_step80k.pt / KEEP_oscher_best_lap17.4s_step80k.pt (partial, 80k)
- KEEP_oscher_step80k_FULL.pt (full 154MB, 80k resume용)
- KEEP_policy_lap16.6s_step85k.pt / KEEP_policy_best_lap16.6s_step85k.pt (partial, 현재 oscher best)
- 상세: _thinking/KEEP/oscher_demo_policy_step80k.md

## 현재 best 기록
- map_easy3: 6.1초(per-lap), policy_best_lap6.1s_step224k.pt / full=step_224k.pt
- oschersleben: 16.6초(per-lap), policy_best_lap16.6s_step85k.pt / full=step_85k.pt

## 관련
- 027(hang 근본원인), 028(hang 수정+재시작), 026(warm 근거), 019 §2(snapshot 정책).
- 동영상 작업은 별도 핸드오프: _thinking/KEEP/VIDEO_HANDOFF.md (새 세션에서 진행).
