# 027 — Stage2 첫 시작 hang 버그: 근본원인 + 수정안 (2026-05-23)

> Stage2 fine-tune(A-2 joint replay)을 13:07 시작했으나 무한 루프(hang)로 막혀 정리(SIGKILL).
> latest.pt(Stage1 516k)·원본 runs/stage1_map_easy3 보존. GPU 반환됨. **수정 미적용 상태.**

## 증상
- dreamer pid 6445: CPU 95%, GPU 4%, metrics step=0 고정 11분+, train_eps npz 0개 = 무한 루프.
- warm-load/joint 배선은 정상 로그 확인됨(joint 0.3=355eps, lr×0.5, warm 104 _wm.* keys unexpected=0).

## 근본 원인 (debugger opus 분석, 코드 증거)
- `tools.sample_episodes`(tools.py:339-341)의 `if total<2: continue`가 **모든 에피소드 len<2면 무한 루프**.
  재현 테스트 확정: 빈 dict→ValueError, **len<2만→HANG**, len≥2→OK.
- 첫 train 트리거: `Every.__call__`(tools.py:851) `_last is None→return 1` → train simulate **첫 step에서
  즉시 첫 _train**(dreamer.py:69). 그 시점 train_eps cache엔 reset 직후 transition 1개(len 1)만 존재
  (tools.py:144,163).
- **Stage1↔Stage2 차이(핵심)**: prefill=0(configs.yaml:206)이라 prefill은 무관(공통). 차이는 첫 train 시점
  train_eps 구성. Stage1은 traindir에 선존 npz(또는 운)로 len≥2 확보 → 통과. **Stage2는 신규 빈 traindir +
  fresh 랜덤 actor가 Oschersleben에서 즉시 충돌(len 1)** → 유효 에피소드 없음 → HANG.
- joint replay가 위험을 "발현": seed=0 첫 rng.rand()=0.5488>0.3 → `joint_episode_generator` 첫 yield가
  `gen_new=sample_episodes(train_eps)` → 빈/len1이라 hang. (gen_old=355eps는 항상 안전.)
  Stage1 `make_dataset`(dreamer.py:147)도 동일 잠재 위험이나 선존 npz로 통과(= 운).

## 권장 수정안 (b): joint_episode_generator에 train_eps 유효성 가드
**위치**: vendor/dreamerv3-torch/stage2_utils.py (joint_episode_generator + make_joint_dataset)
**골자**: make_joint_dataset이 `episodes`(=train_eps) ref를 generator에 전달. gen_new 분기일 때
train_eps에 len≥2 에피소드가 1개도 없으면 그 yield를 `next(gen_old)`로 대체(Stage1 풀은 항상 안전).
유효 에피소드 생기면 정상 ratio 복귀.
```python
def joint_episode_generator(gen_old, gen_new, ratio, seed, new_episodes):
    rng = np.random.RandomState(seed)
    while True:
        use_old = rng.rand() < ratio
        if not use_old:
            has_valid = any(len(next(iter(ep.values()))) >= 2 for ep in new_episodes.values())
            if not has_valid:
                use_old = True   # train_eps 미성숙 → 안전한 gen_old로 우회
        yield next(gen_old) if use_old else next(gen_new)
```
**근거**: ① hang 원천을 안전 풀로 우회(절대 hang 안 함) ② stage2_utils.py 1파일 격리 ③ **Stage1 경로
(make_dataset) 무변경 → 회귀 0** ④ 초기 과도기만 gen_old↑, train_eps 쌓이면 ratio 자동 복귀(의도 보존).
**회귀 위험 낮음**: 초기 ratio 일시 치우침(수렴 무영향), 매 yield any() 스캔(early-return O(1)~).
**차선(방어심화, 1차 제외 권장)**: sample_episodes(tools.py:341)에 "전체 len<2면 raise" fail-fast —
단 vendor 공통함수라 Stage1 영향 별도 검증 필요.

## 다음 단계 (새 세션)
1. 위 수정 구현(stage2_utils.py) + 엣지케이스 단위테스트(빈/len1 train_eps에서 hang 안 하고 gen_old fallback)
   + pytest 회귀(71 passed + 신규).
2. runs/stage2_oschersleben 잔여물 정리(첫 hang 시 생성된 빈 logdir/train_eps) 후 stage2_watchdog.sh 재시작.
3. 시작 검증: hang 안 하는지(metrics step 진행, train_eps npz 생성), warm/joint 로그.
4. 운영값 유지: joint 0.3, lr×0.5, warm=stage1/latest.pt. 원본 stage1 불변.
- 관련: 026(warm-load 근거), 024(A-2 검증), 019 §3(사양), 025(트리거).
