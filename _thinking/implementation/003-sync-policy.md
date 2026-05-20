# 003 — 동기화 정책 (분기점 md 저장 + commit + push)

> 2026-05-20 합의. 본 문서 이후 모든 세션에 적용.
> 선행: [planning/005 §1-C 결정 #29](../planning/005-f1tenth_dreamerV3_version3.md), [implementation/002](./002-phase1-1-measurement.md).

---

## 1. 목적·범위

집컴(GPU) ↔ 노트북(CPU) 양 머신에서 클로드 세션이 교대로 작업한다. 동기화 누락 시 다른 머신 세션이 같은 일을 반복하거나 잘못된 가정 위에서 코드를 짠다.
본 정책은 **클로드 세션이 작업을 끝낸 측(push side)** 에서 분기점마다 **(a) 진행 기록 md를 `_thinking/implementation/` 에 새 번호로 append → (b) git commit → (c) git push** 를 자동 실행하는 절차를 정의한다.

> **pull은 본 정책 범위 밖.** 사용자가 머신을 옮겼을 때 직접 `git pull`을 지시한다. 클로드는 자동 pull / fetch / reset / clean 시도 금지.

---

## 2. 분기점 정의

각 분기 종료 시 본 §3 절차 실행. 이 외 임의 분기 추가 가능 (작은 단위는 환영, 너무 잦은 commit만 피하면 됨).

| # | 분기 | 종료 신호 | 매핑된 md |
|---|---|---|---|
| 1 | Phase 1-0 deps | 의존성 설치 + smoke test | [001](./001-phase1-0-deps.md) ✅ |
| 2 | Phase 1-1 centerline + GF baseline | A_centerline + A_gap | [002](./002-phase1-1-measurement.md) ✅ |
| 3 | Phase 1-2 wrapper | A1~A5, A_norm PASS | 004 (예정) |
| 4 | Phase 1-3 dynamic patch | A6 PASS (`base_classes.py:488`) | 005 (예정) |
| 5 | Phase 1-4 reverse guard | A18 PASS | 006 (예정) |
| 6 | Phase 4 reward | A17 PASS | 007 (예정) |
| 7 | Phase 2-0 fork-patch | A20 PASS (`models.py:182`) | 008 (예정) |
| 8 | Phase 2-1 Encoder/Decoder1D | A7, A7b PASS | 009 (예정) |
| 9 | Phase 2-2 MultiEncoder 패치 | A8 PASS | 010 (예정) |
| 10 | Phase 2-3 configs_f1tenth.yaml | A9, A10 PASS | 011 (예정) |
| 11 | **Phase 2-4 dry-run 직전 강제 게이트** (v3 #29) | 노트북 작업 완료 시 | 012 (예정) |
| 12 | Phase 2-4 dry-run 결과 | A19 PASS or 분기 결정 | 013 (예정) |
| 13 | Phase 3 train.py + counter ckpt | R7 검증 | 014 (예정) |
| 14 | Stage 1 학습 시작 / 종료 | snapshot 100초대 trigger | 015, 016 (예정) |
| 15 | Stage 2 학습 시작 / 종료 | A11~A16 평가 | 017, 018 (예정) |

본 표의 번호는 가이드. 실제 md 번호는 작성 순서대로 부여(append-only).

---

## 3. 분기 종료 시 절차 (push side)

순서 준수. 작업을 막 끝낸 머신에서 실행.

```bash
# 1. md append (CLAUDE.md 정책 — 신규 번호 부여, 기존 수정 금지)
#    파일명: _thinking/implementation/{NNN}-{phase-keyword}.md
#    내용: 결과·진단·다음 단계·미확정 항목·acceptance criteria 매핑

# 2. 변경 사항 확인
cd /home/dlacksdn/f1tenth_RL_project
git status -sb
git diff --stat

# 3. stage + commit (분기 단위. 무관 변경 섞지 말 것)
git add <변경 파일들>
git commit -m "Phase X-Y: <분기 키워드>
<본문: 결과·산출물·acceptance criteria>"

# 4. push (인증 fatal 시 사용자에게 보고 후 사용자가 직접 실행)
git push origin master
```

### 3-1. 메시지 컨벤션
- 1행: `Phase X-Y: <키워드>` 또는 `Add ...`, `Patch ...`
- 본문: 산출물(파일), 통과한 acceptance criteria, 다음 분기 진입 조건
- Co-Authored-By 트레일러: `Claude <model-id> <noreply@anthropic.com>`

### 3-2. 무엇을 stage하나
- 분기 작업 결과의 코드·md·산출물(centerline CSV 등)만.
- `.gitignore`에 의해 학습 산출물(logs, runs, ckpt, replay, wandb, *.pt, events.out.tfevents.*)은 자동 제외.

### 3-3. 금지 사항
- ❌ `git push --force`, `--force-with-lease` — 원격 history 손상.
- ❌ `git pull`, `git fetch`, `git reset --hard`, `git checkout .`, `git clean -fd` — pull side는 본 정책 범위 밖. 사용자가 직접 지시한 경우에만 실행.

---

## 4. v3 #29 강제 게이트와의 관계

v3 §1-C 결정 #29는 "Phase 2-4 dry-run 진입 전 노트북에서 `git commit && git push`, 집컴에서 `git pull` 후 dry-run 실행"을 강제 게이트로 명시.
본 정책의 분기 11 (Phase 2-4 직전)의 **push 부분이 v3 #29 충족**. pull은 사용자가 집컴에서 직접 실행.

---

## 5. dreamerv3-torch 별도 동기화

`/home/dlacksdn/dreamerv3-torch/` 는 본 repo 외부 디렉토리 (NM512 fork, vendor-in 대상). Phase 2-0 이후 in-place 패치를 적용하면 그 디렉토리도 동기화 필요.
- 그 디렉토리는 별도 git repo. 패치 적용 시 그 repo에서도 commit + push.
- 대안(낮은 우선순위): 본 repo의 `vendor/dreamerv3-torch/` 로 흡수하는 방안은 Phase 2-0 진입 시 결정.
- 본 정책 §3 절차는 본 repo만 다룬다. dreamerv3-torch는 별도 운영.

---

## 6. .gitignore 정책 요약

머신별·재생성 가능·대용량은 무시. 코드·문서·측정값은 commit.

**무시**: `.venv/`, `env/`, `__pycache__`, `*.pyc`, `logs/`, `runs/`, `checkpoints/`, `*.pt`, `*.ckpt`, `wandb/`, `events.out.tfevents.*`, `replay/`, `*.npz`, `.omc/`, IDE 설정.

**Commit**: `scripts/`, `dreamer_f1tenth/`, in-place 패치된 `gym/f110_gym/`, `pkg/`, `maps/*.csv`, `_thinking/` 전체 (planning, env_setting, implementation, notes, analysis).

상세는 `.gitignore` 본문 주석 참조.

---

## 7. 인증 처리

- HTTPS remote: PAT(Personal Access Token) 필요. `git push` 시 fatal 발생 가능.
- 클로드 세션이 자동 push를 시도해 fatal이 나면 **사용자에게 보고 후 사용자가 직접 push**. 자동 재시도 금지.
- 권장 대안: gh CLI 인증(`gh auth login`) 또는 SSH remote 전환. 양 머신에 동일 인증 설정 시 자동화 가능.

---

## 8. 운영 룰 요약 (TL;DR)

**작업을 끝낼 때 (클로드가 자동 실행)**
1. `_thinking/implementation/{다음 번호}-*.md` 새 작성.
2. `git add <변경 파일> && git commit -m "Phase X-Y: ..."`.
3. `git push origin master` (인증 fatal 시 사용자에게 보고).

**pull / 다른 머신 동기화**
사용자가 직접 지시. 클로드는 자동 실행 금지.

**금지**: `--force`, `pull`, `fetch`, `reset --hard`, `clean -fd`, `checkout .` — 사용자가 직접 명령할 때만.
