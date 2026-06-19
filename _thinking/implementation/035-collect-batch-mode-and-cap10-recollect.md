# 035 — collect 배치 모드 추가 + GPU 이득 없음 측정 + cap-10 완주/충돌 재수집

> 2026-06-19. [[034-p2-collection-done-gpu-batch-next]] 후속. 외부 Diffuser 프로젝트
> (`/home/dlacksdn/f1tenth_planning_with_diffusion`) 데이터 수집의 RL_project 측 기록.
> ★ Dreamer 본래 학습 무침범(차량물리/reward/_STATE_SCALE 무변경). append-only.

---

## 1. collect_crash_data.py 배치 모드 추가 (commit)
034가 예고한 "GPU envs=N 배치 collect"를 구현. **단일 경로 100% 보존**:
- `--envs N`: 1=기존 단일 직렬(`collect_episode` 루프, 불변), >1=`collect_batch`
  (`vendor/.../tools.py` `simulate` L150-205 배치 루프 미러: done별 per-env reset+새 UUID,
  obs N-stack 배치 추론, env.id별 cache, done ep cause 필터).
- `--device cpu|cuda`: 기존 하드코딩 `"cpu"`를 옵션화(배치 추론 GPU용).
- `--parallel`: 배치 env.step을 Parallel(spawn 프로세스)로 병렬화. 미지정=Damy(메인 직렬).
- pose는 배치에서 `_pose_req`(log_pose_* 필수, .env 체인 폴백 불가 = Parallel 프로세스 분리).
- **부수 버그 수정**: `--save-complete` help의 `%` 미이스케이프(`100% → 100%%`) — `--help`
  호출 시 argparse `format_help`가 크래시하던 잠재 버그. 실수집(정상 인자)엔 무영향이라 034때 미발견.

## 2. ★ GPU 배치 측정 = 이득 없음 (Diffuser 008 §1과 동일 결론)
cap-15 step_105k, episodes=12, overhead(episodes=0) 분리, Σstep/rollout-s:
| 구성 | step/s | vs CPU단일 |
|---|---|---|
| CPU envs=1 | 224.5 | 기준 |
| GPU envs=8 Damy | 185.5 | −17% |
| GPU envs=8 Parallel | 238.7 | +6% |
- 원인: behavior 모델(Dreamer actor)이 작아 추론 비병목, 병목=f110 `env.step`(Python),
  CPU torch BLAS 이미 멀티코어. → **GPU 배치 미채택**(코드 보존). 수집은 단일 CPU로 충분.
- ※ 이건 *데이터 수집* 한정. Dreamer **학습**은 기존대로 GPU(envs=8 parallel) 사용.

## 3. cap-10 완주+충돌 재수집 완료
- `--ckpt runs/cap10_oschersleben/step_45k.pt --v_max 10 --save-complete --max-env-steps 9000
  --out runs/crash_data/cap10_full --episodes 40` (단일 CPU).
- 결과: **완주 30 + 충돌 10**(수집률 1.0). stochastic 완주율 75% 실현.
- 완주 ep 검증: 22키, v_max=10, pose(2807,3), is_terminal[-1]=False·is_last[-1]=True·
  log_completed=1·lap_time_s>0 2개(2랩). 0-패딩 정렬 OK.
- **P2 데이터 4 tier 종료**: cap-5(완주22+충돌22) / cap-10(완주30+충돌31) /
  cap-15(충돌371) / cap-20(충돌291). (runs/는 gitignore라 데이터는 commit 안 됨.)

## 4. 무침범 보증 (재확인)
- 변경 = `scripts/collect_crash_data.py`(수집 하니스)뿐. **차량 물리·env reward·_STATE_SCALE·
  Dreamer 코어·V_MAX 파라미터화(031) 무변경.** runs/crash_data = 데이터 산출물.

## 5. 더보기
전체 계획·전략·P3 로더 사양 = Diffuser `_thinking/implementation/008-gpu-batch-measured-no-gain-and-p2-done.md`.
