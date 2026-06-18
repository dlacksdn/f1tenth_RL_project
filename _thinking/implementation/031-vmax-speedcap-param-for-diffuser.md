# 031 — V_MAX 속도캡 파라미터화 + 속도캡 정책 학습 (외부 Diffuser 프로젝트용)

> 2026-06-18. **이 변경들은 RL_project 자체 목표가 아니라 별도 프로젝트
> `/home/dlacksdn/f1tenth_planning_with_diffusion`(Diffuser offline RL, 추가과제 주제1)을 위한 것.**
> 그쪽이 "속도캡 behavior policy로 데이터 수집"을 필요로 해, 여기 Dreamer 인프라로 V_MAX를 제한한
> 정책을 학습한다. ★ **Dreamer 본래 학습 기능은 무침범**(전부 backward-compatible). append-only.

---

## 0. 왜 (배경)
Diffuser 과제: 느린 behavior policy로 offline 데이터 수집 → 환경 추가상호작용 없이 더 빠른 정책 합성.
"느린 정책"을 만들기 위해 **action space의 속도 상한 V_MAX를 낮춰(15/10/5) warm-load 학습**한다.
정책이 그 속도에 맞게 조향을 학습 → 깨끗한 저속 완주 데이터. (rollout-clamp 아님 = 훈련시 제한.)

## 1. 변경 파일 (전부 backward-compatible)
**V_MAX 파라미터화** — 기본 20.0이면 기존 Dreamer 동작과 100% 동일(실측 검증):
- `vendor/dreamerv3-torch/configs.yaml`: `defaults`에 `v_max: 20.0` 1줄 추가(신규 config 키).
- `vendor/dreamerv3-torch/dreamer.py`: make_env의 `F1Tenth(task, action_repeat, seed=...)`
  → `F1Tenth(..., v_max=config.v_max)` (1줄).
- `vendor/dreamerv3-torch/envs/f1tenth.py`(어댑터): `F1Tenth.__init__`에 `v_max=None` 인자
  (None→모듈 V_MAX=20) + `self._v_max` 저장 + **`__getstate__`/`__setstate__`에 `_v_max` 포함**
  (병렬 워커 pickle 필수) + `action_space` property의 high를 `self._v_max`로.
  - ★ 어댑터-레벨만 손댐: NormalizeActions([wrappers.py:32-39])가 어댑터 action_space.high를 읽어
    [-1,1]→[V_MIN, v_max] 매핑하므로 정책이 ≤v_max만 명령. 내부 F110GymnasiumWrapper(f1tenth_env.py)·
    차량물리·`_STATE_SCALE`(=20 유지)·reward 전부 **무변경**. wrapper의 clip(action,−5,**20**)은
    ≤v_max 명령이 그대로 통과하는 무해 상한.
- 검증(실측): 인자없음 → action high [0.4189, **20.0**](Dreamer 불변) / v_max=15 → 15.0 / pickle 왕복 15 유지 /
  NormalizeActions 정책+1→raw 15. `--v_max`는 CLI 자동생성(dreamer.py:488-490, defaults 키마다 --key).

**시각화/평가 스크립트에 `--v_max` 추가**(캡 정책을 학습값과 일치시켜 봐야 정확):
- `scripts/watch_drive.py`: `--v_max` 옵션 + config.v_max 세팅 + speed 출력식 v_max 반영(RenderDamy).
- `scripts/eval_gate.py`: `--v_max` 옵션 + config.v_max 세팅.

**신규 watchdog**(stage2_watchdog 패턴 + v_max + ★ bracket-pgrep 오탐수정):
- `scripts/cap15_watchdog.sh`(V_MAX=15), `scripts/cap5_watchdog.sh`(V_MAX=5).
  warm-load=stage1_map_easy3/latest.pt, joint 0.3, lr×0.5, --steps 300000.
  - ★ is_alive는 `pgrep -f "[d]reamer\.py.*capN_oschersleben"` bracket-trick — pgrep -f 가 같은
    문자열 든 셸 명령(모니터링 등)까지 매칭하는 cross-match 오탐을 막는다(cap15 첫 launch 때 이 버그로
    start_train 지연됐다 자기수정된 이력 있음).

## 2. 학습 결과/상태
- **cap-15**(runs/cap15_oschersleben): 학습 **진동**(완주↔크래시 반복). latest.pt=크래시골짜기(쓰지말 것).
  봉우리=`step_105k.pt`(헤드리스 eval: 완주율1.0, 2랩~37.3s, lap best 18.1s, A12/A13 PASS). **학습 정지함.**
  ⚠️ V_MAX=15는 Oschersleben(평균실현속도~14.4)에선 사실상 무캡 → ~37s(near-expert).
- **cap-5**(runs/cap5_oschersleben): `cap5_watchdog.sh`로 **학습 진행 중**(2026-06-18 시작). ~100s대 목표.
- joint_replay(map_easy3 무캡 혼합)가 cap-15 진동의 유력 원인 추정 → cap-10에서 joint 제거 실험 예정.

## 3. 무침범 보증 (RL_project 본래 학습)
- `--v_max` 미지정 시 V_MAX=20 → **map_easy3/stage2 등 기존 학습 완전 동일.** config 기본키만 추가.
- 차량 동역학·맵·env reward·종료조건·`_STATE_SCALE` **무수정.** runs/cap{15,5}_oschersleben은 신규 디렉터리.
- 변경은 master에 commit됨. 되돌리려면 해당 커밋 revert(단 Diffuser 데이터수집이 이 기능에 의존).

## 4. 더보기
전체 Diffuser 계획·P1~P6·tier 설계 = Diffuser 프로젝트 `_thinking/plan_new/008-diffuser-plan-v4.md` +
`_thinking/implementation/002-p1-speedcap-policies-and-rlproject-changes.md`.
