# 환경·스펙 보정 (집컴 기준)

> 2026-05-20 작성. 001(PC 스펙), 002(세팅 스냅샷), 004(노트북 venv 재구축)에 누락·차이가 있어 보정. append-only 원칙대로 기존 문서는 손대지 않음.

---

## 1. 작성 배경

- 001 문서에 CPU/RAM이 명시되지 않았음 (GPU만 기재).
- 004 문서는 **노트북(WSL2, CPU)** 기준이라 venv 폴더명·torch 빌드·CUDA 상태가 집컴과 다름.
- 005 v3(planning) 구현 진입 직전 현장 상태를 확정해 둘 필요가 있음.

---

## 2. 하드웨어 보정 (001 보강)

집컴 = `DESKTOP-IP77VVV` (WSL2, Ubuntu).

| 항목 | 값 | 비고 |
|---|---|---|
| CPU | **AMD Ryzen 5 7500F** | 6-core 12-thread, base 3.70GHz / boost 4.34GHz, L3=32MB |
| RAM | **32GB DDR5 5200MT/s** | DIMM (2/4 슬롯 사용, 듀얼 채널) |
| GPU | **NVIDIA GeForce RTX 4060 Ti 8GB** | Driver 596.21 / CUDA 13.2 호환 |
| Disk | 2TB SSD | |
| OS | Linux 6.6.114.1 WSL2 | |

> 001 문서는 GPU만 기재되어 있었음 → 본 문서에서 CPU/RAM/Disk를 정식 기록.

---

## 3. venv 상태 보정 (004 보강)

004는 노트북 기준으로 `env/` 경로·CPU torch였음. 집컴은 다음과 같음 — **004 절차로 이미 동기화 완료 상태**.

| 항목 | 노트북 (004) | 집컴 (현재) | 상태 |
|---|---|---|---|
| venv 경로 | `env/` | `.venv/` | ⚠️ 이름만 다름 (기능 동일) |
| pip shebang | (재구축 후) 정상 | `.venv` 정상 | ✅ |
| Python | 3.8.10 | 3.8.10 | ✅ |
| pip / setuptools | 22.0.3 / <58 | 22.0.3 / 44.0.0 | ✅ |
| torch | 2.4.1+cpu | **2.4.1+cu124** | ✅ 의도된 GPU 변형 |
| `torch.cuda.is_available()` | False | **True** | ✅ |
| torchvision / torchaudio | 0.19.1+cpu / 2.4.1+cpu | 0.19.1+cu124 / 2.4.1+cu124 | ✅ |
| gym | 0.18.0 | 0.18.0 | ✅ |
| f110-gym | editable | editable (`/home/dlacksdn/f1tenth_RL_project/gym`) | ✅ |
| numpy 1.24.4 / numba 0.58.1 / llvmlite 0.41.1 | 동일 | 동일 | ✅ |
| scipy 1.10.1 / pyglet 1.5.0 / Pillow 7.2.0 / matplotlib 3.7.5 | 동일 | 동일 | ✅ |

활성화 명령:
```bash
cd /home/dlacksdn/f1tenth_RL_project
source .venv/bin/activate
```

> 004 문서의 "집컴으로 옮길 때" 절차(96-98행)는 본 시점 기준으로 **완료된 상태**로 간주.

---

## 4. DreamerV3 진입을 위한 미설치 의존성 (Phase 1-0에서 처리)

노트북·집컴 둘 다 아직 깔리지 않은 항목 — 005 v3 §1, §3 진입 전 설치 필요.

- `gymnasium` (5-tuple wrapper용, 005 §3 필수)
- `ruamel.yaml`, `einops`, `tensorboard`
- `moviepy`, `imageio-ffmpeg` (DreamerV3 비디오 로깅)
- (선택) `wandb` 또는 `mlflow` — 005 §6 로깅 전략 결정에 따라

설치 시 주의: `--index-url https://download.pytorch.org/whl/cu124`는 torch 계열에만, 나머지는 PyPI에서 별도 설치 (004 §2 주의 사항 동일).

---

## 5. 005 v3 영향 정정

- 005 v3 §0-4 표의 "32GB DDR5" 이미 반영됨 (이전 세션에서 편집 완료).
- venv 경로가 `env/`가 아니라 `.venv/`라는 사실 → Phase 1-0 명령 작성 시 반드시 `.venv` 사용.
- GPU 8GB / RTX 4060 Ti / CUDA 12.4 토치 빌드 → 005 v3 §4-5 (precision=16 AMP, batch_size=8) 전제와 일치.

---

## 6. 후속

- DreamerV3 의존성 설치 후 본 문서 §4에 실제 설치 버전을 새 문서(`006-deps-install.md` 등)로 append 기록.
- 노트북·집컴 동기화 차이가 다시 생기면(예: 새 패키지 추가) 새 번호 문서로 보정.
