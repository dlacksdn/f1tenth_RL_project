# 002 - Dreamer-v3 구현체 선택 분석

> **목적**: F1TENTH 강화학습 프로젝트에 사용할 Dreamer-v3 구현체를 결정한다.
> **결론(TL;DR)**: **`NM512/dreamerv3-torch`** 를 베이스라인으로 채택한다. 공식 JAX 구현체와 `r2dreamer`는 보조 참고용.
> **작성일**: 2026-05-19

---

## 1. 비교 대상

| # | 저장소 | 설명 | 언어/프레임워크 |
|---|--------|------|-----------------|
| 1 | [danijar/dreamerv3](https://github.com/danijar/dreamerv3) | 논문 저자(공식) 구현 | JAX |
| 2 | [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) | 커뮤니티 표준 포팅 | PyTorch |
| 3 | [NM512/r2dreamer](https://github.com/NM512/r2dreamer) | 같은 저자의 후속/실험적 버전 (R2D2 스타일 시퀀스 학습) | PyTorch |

---

## 2. 공식 JAX vs PyTorch 포팅 비교

| 항목 | 공식 JAX (danijar) | PyTorch (NM512) |
|------|-------------------|-----------------|
| **성능/재현성** | 논문 수치와 정확히 일치, 가장 안정적 | 대부분 재현되지만 일부 환경에서 미세 차이 가능 |
| **속도** | JIT + XLA로 매우 빠름 (TPU/대형 GPU 유리) | 일반 GPU에서는 비슷하거나 약간 느림 |
| **디버깅** | 함수형 패러다임, `jit`/`vmap` 추적 어려움 | eager 모드, breakpoint/print 자유로움 |
| **커스터마이징** | 추상화 깊고 수정 난이도 높음 | 구조가 평탄해 모델/리워드/관측 수정 용이 |
| **외부 생태계** | gym/gymnasium 래퍼만 제공 | F1TENTH gym, ROS2, SB3 등 통합 사례 풍부 |

### F1TENTH 관점 핵심 포인트
1. F1TENTH는 **LiDAR 관측 + 커스텀 리워드 + sim-to-real** 이 핵심 → 모델 수정과 디버깅 빈도가 매우 높음.
2. ROS2 / F1TENTH Gym과의 통합 사례는 PyTorch 진영이 압도적.
3. JAX의 속도 이점은 환경 step이 빠를 때 발현되는데, F1TENTH는 시뮬레이터 step 자체가 병목이라 JAX 이점이 줄어듦.

→ **PyTorch 포팅이 실용적으로 우세.**

---

## 3. 더 나은 PyTorch 구현체 존재 여부 (2026-05 기준)

- **`NM512/dreamerv3-torch`가 사실상 커뮤니티 표준.** 다수 논문/프로젝트의 베이스라인.
- 그 외 포크(예: `alec-tschantz/dreamerv3-pytorch` 등)는 유지보수/이슈 응답성이 떨어짐.
- Hugging Face/Stable-Baselines3 계열에는 Dreamer-v3의 정식 포팅 없음 (DreamerV2까지만 존재).

→ **`dreamerv3-torch`보다 명확히 우월한 PyTorch 구현체는 현재 없음.**

---

## 4. `dreamerv3-torch` vs `r2dreamer` 최종 선택

### 채택: **`NM512/dreamerv3-torch`**

| 판단 기준 | dreamerv3-torch | r2dreamer |
|-----------|-----------------|-----------|
| 검증 이력 | 다수 논문/프로젝트 베이스라인으로 인용 | 검증 사례 적음, 실험적 |
| 논문 하이퍼파라미터 매핑 | 1:1 매핑이 명확 | 변경/추가됨 |
| 짧은 에피소드 환경 적합성 | 안정적 | sequence 길이 증가로 잦은 reset 환경에서 이점 불명확 |
| 학회/졸업과제 인용 용이성 | 좋음 | 부담 있음 |

### 단계적 확장 경로
1. **1단계**: `dreamerv3-torch`로 베이스라인 성능 확보.
2. **2단계**: long-horizon 주행(랩타임 최적화) 한계가 보일 때 `r2dreamer`의 시퀀스 학습/리플레이 요소만 선택적으로 포팅.

---

## 5. F1TENTH 적용 시 실무 체크리스트

- [ ] **Encoder 교체**: LiDAR 1D 관측 → 기본 CNN encoder를 **1D Conv encoder**로 교체. `networks.py`에서 encoder 모듈이 분리되어 있어 수정 용이.
- [ ] **보상 스케일링**: dreamerv3는 `symlog` 변환을 사용하므로, reward 설계 시 `symlog` 켠 상태에서 정규화 확인.
- [ ] **Replay buffer 용량**: `--replay.size` 및 LiDAR 저장 dtype(예: float16) 점검. 디스크 사용량 큼.
- [ ] **환경 래퍼**: F1TENTH Gym → gymnasium 인터페이스 변환 래퍼 작성.
- [ ] **ROS2 통합**: 학습은 PyTorch / 추론 노드는 ROS2 → 가중치만 ONNX 또는 torchscript로 export.

---

## 6. 결정 요약 (다음 에이전트용 핸드오프)

- **사용 구현체**: `https://github.com/NM512/dreamerv3-torch`
- **이유 한 줄**: F1TENTH의 빈번한 코드 수정/디버깅/ROS2 통합 요구에 PyTorch가 적합하며, 그중 가장 검증된 표준 포팅이기 때문.
- **버리지 않는 자료**:
  - 공식 JAX 구현 → 알고리즘 정합성 검증(레퍼런스) 용도.
  - r2dreamer → 베이스라인 안정화 이후 long-horizon 개선 옵션.
- **다음 작업 후보**:
  1. `dreamerv3-torch` 클론 및 의존성 환경 구성.
  2. F1TENTH Gym ↔ gymnasium 래퍼 작성.
  3. LiDAR 1D Conv encoder 설계 및 통합.
