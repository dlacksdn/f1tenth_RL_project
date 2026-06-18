# 032 — cap-5 학습 종료·baseline 확정 + cap10_watchdog 신설·cap-10 착수

> 2026-06-18. [[031-vmax-speedcap-param-for-diffuser]] 후속. 외부 Diffuser 프로젝트
> (`/home/dlacksdn/f1tenth_planning_with_diffusion`) 데이터용 속도캡 정책 작업의 RL_project 측 기록.
> ★ Dreamer 본래 학습 기능 무침범(V_MAX=backward-compat, 기본20) 그대로. append-only.

---

## 1. cap-5 종료 + baseline 확정
- `runs/cap5_oschersleben` 학습 metrics step 75,504(목표 300k의 25%)에서 **수동 종료**(watchdog→dreamer kill 순).
- **진동**(cap-15와 동일): 빠른 라인 시도→2랩째 크래시. **latest.pt(=step_35k)는 골짜기 → 쓰지 말 것.**
- deterministic eval(`scripts/eval_gate.py --v_max 5 --episodes 5`):
  - step_35k: 완주율 0(1랩도 실패) / step_30k: 0(1랩 52.58s만) / **step_25k: 1.0, 2랩 107.16s** / step_20k: 1.0, 2랩 121.86s.
  - **★ 채택 baseline = `runs/cap5_oschersleben/step_25k.pt` (2랩 107.16s, 완주율 1.0, A12/A13 PASS).**
- eval JSON: `runs/cap5_oschersleben/eval_gate_f1tenth_Oschersleben_{35k,30k}.json` 등 저장됨.

## 2. cap10_watchdog.sh 신설 (코드 추가)
- `scripts/cap10_watchdog.sh` = cap5/cap15_watchdog와 **동일 proven 패턴**(bracket-pgrep `[d]reamer\.py.*cap10_oschersleben`,
  사전 other_dreamer 안전점검, crash resume). 차이 = **V_MAX=10.0 + logdir `runs/cap10_oschersleben`**.
- 파라미터: warm=stage1_map_easy3/latest.pt, lr×0.5, joint 0.3, steps 300000, envs 8, parallel.
- **joint 제거 실험 보류**: cap-5/15 진동했어도 봉우리 스냅샷으로 baseline 확보 성공(proven) → cap-10도 동일 레시피.
- 가동: 2026-06-18 17:47, `runs/cap10_oschersleben/watchdog.log`. 예상 ~60-70s/2랩 중간 tier.

## 3. 무침범 보증 (재확인)
- 추가/수정한 것 = `scripts/cap10_watchdog.sh`(신규 운영 스크립트) 뿐. **차량 물리·env reward·_STATE_SCALE·
  Dreamer 코어 무변경.** V_MAX는 031의 backward-compat 파라미터(기본20) 그대로 사용.
- runs/cap10_oschersleben = 신규 디렉터리. 기존 학습(map_easy3/stage2) 무영향.

## 4. 더보기
전체 계획·데이터 설계 전환(실패 rollout + 완주 앵커 하이브리드)·평가 프로토콜 =
Diffuser 프로젝트 `_thinking/implementation/004-cap5-baseline-and-data-design-pivot.md`.
