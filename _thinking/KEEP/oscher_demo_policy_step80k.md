# [KEEP] Oschersleben 시연용 보존 정책 — Stage2 step 80k (2026-05-23)

> Stage2(Oschersleben fine-tune) 학습 중 **80,000 step** 시점에 산출된, Oschersleben을
> **완주(2-lap)에 성공**한 정책. per-lap 최단 **17.4초**. 시연/보존 목적으로 백업함.
> 학습이 진행되며 더 빠른 lap이 나오면 원본 snapshot은 자동 삭제되므로 KEEP/에 별도 보존.

## 산출 시점 (훈련량)
- **학습 step = 80,000 (= 80k)**  ← 파일명 `step80k`가 이 값.
- 파일 생성 시각: 2026-05-23 20:42 (Stage2 시작 13:53로부터 약 6시간 49분 학습 후).
- 산출 맥락: step 80k eval(20 에피소드)에서 **평균은 미완주 많음**(eval_return 99.26,
  eval_length 410.5, 평균 log_completed=0.0 = eval 진동 성향). 그러나 그 중 **best 에피소드가
  per-lap 17.4초로 완주** → diversity/run-best snapshot 로직이 그 best lap을 포착해 저장.
  → 즉 "항상 완주"가 아니라 "이 시점 best 케이스가 17.4초 완주"인 정책임에 유의.
- per-lap 17.4초 / 완주(2-lap) 총시간 약 35초.

## ★ 시연에 사용할 정확한 파일 (경로 포함)

**★ watch_drive 시연은 반드시 FULL ckpt로 할 것** (2026-05-23 정정):
```
/home/dlacksdn/f1tenth_RL_project/runs/stage2_oschersleben/KEEP/KEEP_oscher_step80k_FULL.pt
```
이유: watch_drive(scripts/watch_drive.py:179)는 `agent.load_state_dict(..., strict=True)`로
로드한다. `policy_*.pt`(partial, 114키=_wm 104+actor 10)는 critic(value)/_slow_value 키가 없어
**strict 로드가 RuntimeError(Missing key)로 죽는다.** FULL ckpt(step_80k 기반, 전체 키)는 같은 80k
시점·같은 actor 가중치라 **주행 결과 동일**하면서 strict 로드 통과.

시연 명령 (Oschersleben 맵, CPU 추론이라 학습 GPU 무방해):
```bash
cd /home/dlacksdn/f1tenth_RL_project && source .venv/bin/activate
python scripts/watch_drive.py \
  --ckpt runs/stage2_oschersleben/KEEP/KEEP_oscher_step80k_FULL.pt \
  --task f1tenth_Oschersleben --episodes 3
```
- map_easy3에서 보려면 `--task f1tenth_map_easy3`로 바꾼다(cross-track/forgetting 확인용).
- `--ckpt` 생략 시 `latest.pt`(현재 학습 중 최신본)가 실행되니 **반드시 --ckpt 명시**할 것.
- partial `policy_*.pt`는 보존/배포용이며 watch_drive 직접 시연엔 부적합(strict 에러).

## 보존 파일 일람 (모두 /home/dlacksdn/f1tenth_RL_project/runs/stage2_oschersleben/ 하위)

| 경로 | 종류 | 크기 | 비고 |
|---|---|---|---|
| `KEEP/KEEP_oscher_lap17.4s_step80k.pt` | diversity bin best (partial: _wm+actor) | 48MB | **시연 권장** |
| `KEEP/KEEP_oscher_best_lap17.4s_step80k.pt` | run best (partial) | 48MB | 시연 가능(거의 동일) |
| `KEEP/KEEP_oscher_step80k_FULL.pt` | full ckpt (optimizer 포함) | 154MB | **이 시점에서 학습 재개**용 |
| `policy_lap17.4s_step80k.pt` (원본) | diversity bin best | 48MB | ⚠️ 더 빠른 lap 나오면 자동 삭제 가능 |
| `policy_best_lap17.4s_step80k.pt` (원본) | run best | 48MB | ⚠️ run 최단 갱신 시 자동 삭제 가능 |
| `step_80k.pt` (원본) | interval full ckpt | 154MB | interval은 덮어쓰기 X(원래 안전) |

- KEEP 백업은 원본과 md5 동일 검증 완료(`policy_lap17.4s_step80k.pt` md5=421f7be7...).
- KEEP/ 폴더는 snapshot 로직(`_unlink_if`, snapshot_utils.py)의 추적 대상이 아니므로
  학습이 17.4초보다 빠른 lap을 내도 **절대 자동삭제되지 않음**.

## partial vs full
- `policy_*`(48MB) = 추론 키만(_wm.* + _task_behavior.actor.*). watch_drive 시연엔 충분.
- `*_FULL.pt`/`step_80k.pt`(154MB) = optimizer/value 포함. 80k 지점에서 fine-tune 재개하려면 이것.

## 관련
- Stage2 fine-tune 현황·hang 수정: _thinking/implementation/027, 028.
- snapshot 정책(interval/diversity bin/run best): snapshot_utils.py, 008/019 §2.
