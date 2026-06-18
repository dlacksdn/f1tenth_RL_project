# 033 — cap-10 학습 종료·step_45k baseline 확정

> 2026-06-19. [[032-cap5-done-cap10-start]] 후속. 외부 Diffuser 프로젝트 데이터용 cap-10 정책 확정.
> Dreamer 본래 학습 기능 무침범(V_MAX=backward-compat). append-only.

---

## 1. cap-10 종료 + baseline 확정
- `runs/cap10_oschersleben` metrics step 93k(목표 300k의 31%)에서 **수동 종료**(watchdog→dreamer kill).
  완주 봉우리 충분 확보 + cap-5처럼 latest 골짜기 위험 회피.
- deterministic eval(`scripts/eval_gate.py --v_max 10 --episodes 5`):
  - step_45k/40k/35k/30k **전부 완주율 1.0** — cap-5와 달리 **무진동**(전 스냅샷 안정 완주).
  - **★ 채택 baseline = `runs/cap10_oschersleben/step_45k.pt`(2랩 53.66s = 27.3+26.36, 완주율 1.0, A12/A13 PASS).**
- ★ **cap-10(V_MAX=10) = 학습 sweet spot**: cap-5(저속)·cap-15(거의 무캡 고속)는 진동했으나 cap-10은 무진동.
- eval JSON: `runs/cap10_oschersleben/eval_gate_f1tenth_Oschersleben_{45k,40k,35k,30k}.json`.

## 2. tier 정책 인벤토리 (Diffuser 데이터 수집용, 전부 확정)
| tier | 정책(ckpt) | eval(--v_max) | 2랩 |
|---|---|---|---|
| cap-5 | runs/cap5_oschersleben/step_25k.pt | 5 | 107.16s |
| cap-10 | runs/cap10_oschersleben/step_45k.pt | 10 | 53.66s |
| cap-15 | runs/cap15_oschersleben/step_105k.pt | 15 | 37.3s |
| cap-20 | runs/stage2_oschersleben/policy_best_lap16.6s_step85k.pt (+크래시 259) | 20 | ~36s |

## 3. 다음 (코드 작업 예고)
- `scripts/collect_crash_data.py` 신설 예정: stochastic(training=True+eval_state_mean=False) rollout으로
  **충돌 ep만** 수집, pose(env.unwrapped._raw_obs, env 무수정)+v_max 기록, 4 tier CPU 동시.
- 전체 전략·구현 사양 = Diffuser `_thinking/implementation/005-crash-only-data-collection-strategy.md`.
