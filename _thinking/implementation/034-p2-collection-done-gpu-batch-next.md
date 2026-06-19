# 034 — P2 충돌/완주 수집 1차 완료 + collect --save-complete + GPU 배치 예고

> 2026-06-19. [[033-cap10-baseline-confirmed]] 후속. 외부 Diffuser 프로젝트 데이터 수집 현황.
> Dreamer 본래 학습 무침범. append-only.

---

## 1. 수집 데이터 (runs/crash_data/, 1차 소스 검증)
| tier | 완주(2랩) | 충돌 | 상태 |
|---|---|---|---|
| cap-5 | 22 | 22 (cap5_full 9 + cap5 13) | 완료 |
| cap-10 | 0 | 21 (봉우리 step_45k, 중단) | **완주+충돌 재수집 필요** |
| cap-15 | 0 | 371 | 완료 |
| cap-20 | 0 | 291 | 완료 |
- 전략(사용자): 저속~중속(cap-5/10)=완주+충돌(진정성), 고속(cap-15/20)=충돌만.
- 수집 방식: stochastic(`training=True`+`eval_state_mean=False`), 충돌 정렬=tools.simulate 미러.

## 2. collect_crash_data.py 변경 (★ commit 대상, uncommitted)
- `--save-complete`: collision뿐 아니라 `lap_complete` ep도 저장(저속 tier 완주용). 미지정 시 충돌-only 불변.
- `--max-env-steps`: build_config 후 `config.time_limit` 오버라이드(배회 ep 조기 truncate; 완주 안 끊기게 9000).
- 검증: cap5_full로 완주 22+충돌 9 정상 저장 실측.

## 3. 다음 (GPU 배치 — 새 세션)
- 추론은 ep마다 독립 → **GPU envs=N 배치 collect 구현**(`make_env(envs=N)`+`device=cuda`+simulate 배치 루프).
  품질(정렬/pose/충돌필터) 동일 검증 + 속도 측정 후 채택. 단일 env GPU는 배치1이라 비효율.
- cap-10 완주+충돌 GPU 배치 재수집(step_45k --save-complete). 완주율 86%.
- 전체 계획·전략 = Diffuser `_thinking/implementation/007-p2-collection-state-and-gpu-batch-handoff.md`.

## 4. 규약
★ push는 사용자 지시 시만(자율 push 금지, 2026-06-19). commit+문서는 분기마다 자율.
