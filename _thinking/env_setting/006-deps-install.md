# 006 — DreamerV3 의존성 설치 실측 기록

> 2026-05-20. 005-spec-fix §6 지시에 따라 설치 완료 후 작성.
> 집컴 (.venv/) 기준. 노트북(env/) 설치 시점에 별도 문서 추가.

---

## 설치된 버전 (집컴, .venv/)

| 패키지 | 설치 버전 | spec-fix §4 요구 | 비고 |
|---|---|---|---|
| gymnasium | 0.29.1 | >=0.28,<1.0 | ✅ |
| scikit-image | 0.21.0 | — | Phase 1-1 용도 |
| tensorboard | 2.14.0 | 2.17.1 (미존재) | PyPI 최신 Python 3.8 호환 |
| ruamel.yaml | 0.17.4 | 0.17.4 | ✅ |
| einops | 0.3.0 | 0.3.0 | ✅ |
| moviepy | 1.0.3 | 1.0.3 | ✅ |
| imageio-ffmpeg | 0.5.1 | — | moviepy 백엔드 |
| protobuf | 3.20.0 | 3.20.0 | ✅ |

---

## 스킵한 dreamerv3 requirements 항목

mujoco, dm_control, memory_maze, crafter, opencv-python — f1tenth 미사용.

---

## Pillow 충돌 사항

scikit-image 0.21.0 설치로 pillow 7.2.0 → 10.4.0 업그레이드.
gym 0.18.0은 Pillow<=7.2.0 요구하나 f110_gym 헤드리스 동작에 영향 없음 (smoke test PASS).
상세 내용: implementation/001-phase1-0-deps.md 참조.

---

## 활성화 명령 (집컴)

```bash
cd /home/dlacksdn/f1tenth_RL_project
source .venv/bin/activate
```
