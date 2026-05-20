# 006 — F1Tenth DreamerV3 구현 운영 전략 (모델·에이전트 분담)

> 005 v3 문서 확정 직후 작성. "Sonnet 4.6으로 가능한가 / deep-interview·에이전트 소환이 빠른가"에 대한 결론.

---

## 1. 결론 요약

- **Sonnet 4.6 메인 + Opus 핀포인트** 조합이 cost/quality 최적.
- v3가 줄번호 단위로 변경 위치를 못박아 둬서 메인 드라이버의 추론 부담이 낮음. `executor` 에이전트(Sonnet)로 충분.
- **deep-interview는 불필요**. v3에서 이미 요구사항·결정이 수렴됨. deep-interview는 요구가 불명확할 때 가치가 있음.
- **에이전트 소환은 "선택적으로 빠름"**. 단순 패치 적용은 메인이 직접 하는 게 더 빠르고, 멀티파일 리팩터·검증 패스·디버깅은 전용 에이전트가 빠름.

---

## 2. 모델 분담

### 2-1. Sonnet 4.6 (메인 드라이버, ~80%)
- Phase 1-0/1-1: 의존성 설치, 트랙 측정(centerline·L_track), GapFollower 베이스라인 측정
- Phase 2: fork-patch 적용
  - `dreamerv3-torch/models.py:182` preprocess image 가드
  - `dreamerv3-torch/networks.py` MultiEncoder/MultiDecoder lidar branch (ConvEncoder1D)
  - `f110_gym/envs/base_classes.py:488` vel_y = vel * sin(slip_angle)
- Phase 3: gymnasium 5-tuple wrapper, reward shaping, TimeLimit·Damy 통합
- Phase 4: 학습 루프 실행, 로그·체크포인트 관리, 일상 모니터링

### 2-2. Opus (핀포인트 에스컬레이션, ~20%)
- A19 dry-run 게이트 결과 해석 후 batch_size/precision 조정 판단
- 100K step 시점 R11 수렴 진단(보상 곡선·KL·entropy 동시 해석)
- 예상 외 디버깅: RSSM NaN, posterior collapse, action_repeat 불일치 등 근본 원인 추적
- precision=16 ↔ NM512 `_use_amp` 매핑 검증(Phase 2-0 직전, 5분 작업이지만 잘못되면 학습 전체 영향)

### 2-3. 에스컬레이션 트리거
1. 학습 곡선 이상(보상 정체·발산, KL 스파이크)
2. dry-run VRAM/wall-clock 초과
3. v3에 명시되지 않은 결정이 필요한 분기
4. 동일 버그 2회 이상 재현 시(메인 디버깅이 헛돌고 있다는 신호)

---

## 3. 에이전트 분담 (oh-my-claudecode)

### 3-1. 적극 활용
- `executor` (Sonnet): Phase 2~3 패치 실제 적용 — 메인이 위임
- `explore` / `Explore`: 처음 진입하는 모듈 위치 확인 (예: NM512 `_use_amp` 호출처)
- `verifier`: Phase 1-1 측정 결과 / Phase 2-0 dry-run 결과 검증 (메인이 자체 승인 금지 원칙)
- `code-reviewer`: Phase 3 wrapper·reward 코드 리뷰 (1회)
- `debugger` (Opus): 학습 중 비정상 종료·NaN 발생 시
- `document-specialist`: dreamerv3-torch / f1tenth_gym 미확인 API 호출 직전

### 3-2. 사용하지 않음
- `planner`: v3가 이미 계획. 재계획 불필요
- `analyst` / `critic`: 005 직전에 004 critic까지 끝남
- `designer`: UI 작업 없음
- `architect`: 005 v3 자체가 아키텍처 결정의 산물

### 3-3. 메인이 직접 (위임 X)
- v3에 줄번호 명시된 단일 파일 1-2줄 패치
- 셸 명령(설치·실행·로그 확인)
- 작은 설정 파일 편집

---

## 4. 운영 절차 (Phase 1-0 진입 직전 체크리스트)

1. **부팅 필독**: `_thinking/planning/005`, `_thinking/planning/006`, `_thinking/env_setting/` (폴더 전체), `CLAUDE.md`
2. v3 §4-3 `L_track` Phase 1-1 측정값으로 확정 → §0 변경 요약 반영 여부 결정
3. `env_setting/` 보정·추가 사항은 **새 번호 문서**로 append (기존 문서 수정 금지)
4. Phase 2-0 직전 precision=16 ↔ `_use_amp` 매핑 5분 검증 (Opus 또는 explore)
5. Phase 2 패치 완료 후 dry-run 게이트 통과 → verifier 확인
6. Phase 4 진입 후 첫 100K step에서 1회 Opus 진단

---

## 5. 메모

- v3는 추가 critic 없이 구현 진입 권장(이전 세션 결론).
- 잔존 리스크 3건은 005 자체에 내포되어 있으며, 새 세션 부팅 시 분기 보고 규칙으로 처리.
- 새 세션 부팅 프롬프트는 1회용이므로 본 문서에 포함하지 않고 대화에서 별도 제시.
