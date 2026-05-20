# 001 — Phase 1-0: 의존성 설치 완료

> 2026-05-20. 집컴 (.venv/, RTX 4060 Ti 8GB, torch 2.4.1+cu124) 기준.

---

## 설치 결과

| 패키지 | 버전 | 비고 |
|---|---|---|
| gymnasium | 0.29.1 | 5-tuple wrapper용 |
| scikit-image | 0.21.0 | Phase 1-1 centerline skeletonize |
| tensorboard | 2.14.0 | dreamerv3 requirements는 2.17.1이나 PyPI 미존재 → 2.14.0 대체 |
| ruamel.yaml | 0.17.4 | dreamerv3 config 로딩 |
| einops | 0.3.0 | dreamerv3 네트워크 |
| moviepy | 1.0.3 | dreamerv3 비디오 로깅 |
| imageio-ffmpeg | 0.5.1 | moviepy 백엔드 |
| protobuf | 3.20.0 | tensorboard 의존 |

기존 유지: torch 2.4.1+cu124, gym 0.18.0, scipy 1.10.1, PyYAML 6.0.3.

---

## 주의: Pillow 버전 충돌

scikit-image 0.21.0 설치 시 pillow 7.2.0 → 10.4.0 업그레이드됨.
`gym 0.18.0 requires Pillow<=7.2.0` 경고 발생.

**현재 영향 없음**: f110_gym smoke test PASS (reset/step 정상). gym 0.18.0의 Pillow 의존은 `env.render()` 내부의 이미지 저장 경로에만 영향. 본 프로젝트는 gym의 render 미사용 (집컴 pyglet GL 경로 / dreamerv3 자체 video logging 경로 사용).

만약 추후 문제 발생 시: `pip install scikit-image==0.19.3 pillow==7.2.0` 으로 다운그레이드.

---

## 검증

```
gymnasium 0.29.1      ✅
scikit-image 0.21.0   ✅
tensorboard 2.14.0    ✅
ruamel.yaml OK        ✅
einops 0.3.0          ✅
moviepy 1.0.3         ✅
imageio-ffmpeg 0.5.1  ✅
torch 2.4.1+cu124 cuda: True  ✅
gym 0.18.0            ✅
f110_gym smoke test (map_easy3, reset+3step) PASS  ✅
```

---

## Phase 1-0 체크리스트

- [x] 의존성 설치 (집컴 .venv/)
- [x] f110_gym smoke test PASS
- [ ] RAM 확인 → 005 §0-4 보정: WSL2 가용 15Gi (물리 32GB). 005 §0-4 이미 "32GB" 기록 — 보정 불필요
- [ ] env_setting/001 §4 `map_easy` 표기 보정 기록 → env_setting/006 참조
- [ ] 노트북 (.env/) 동일 설치 (노트북 접속 시점에 처리)

---

## 다음: Phase 1-1

centerline 추출 (`scripts/extract_centerline.py`) + GapFollower baseline 측정 (`scripts/measure_gap_follower.py`).
두 스크립트 모두 신규 작성 필요.
