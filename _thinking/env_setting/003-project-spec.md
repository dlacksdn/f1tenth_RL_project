# F1TENTH 프로젝트 명세

> 출처: AIE4003_RL_F1TENTH.pdf (RILS LAB @ INHA UNIV, Woo-Jin Ahn)
> 환경 세팅이 아닌 **프로젝트 목표/평가/발표/추가과제** 명세

---

## 1. 프로젝트 개요

- **대회**: F1TENTH KOREA CHAMPIONSHIP
- **공식 홈페이지**: https://roboracer.ai/
- **성격**: F1TENTH Autonomous Racing — 국제 연구자/엔지니어 커뮤니티가 주관하는 semi-regular 대회
- **프로젝트 목표**: 강화학습을 활용한 주행 알고리즘 설계

---

## 2. 평가 기준 (팀 프로젝트)

| 항목 | 비중 |
|------|------|
| Map Easy 완주 여부 | 30% |
| 알고리즘 발표 | 60% |
| 팀별 Lap time | 10% |

### 알고리즘 완성도 세부 기준
- state, action, reward control
- Map Easy 완주 여부
- Oschersleben 완주 여부

### F1/10 Race Simulation 결과
- map: **Oschersleben**
- 순위별로 점수 차등 배점

---

## 3. 팀 구성

- 조원 **2~3명**의 팀으로 구성
- 발표: 조별로 한 명이 발표

---

## 4. 발표 템플릿

- **분량**: 10분 내외
- **내용**: Agent와 구성한 환경을 수강자가 이해할 수 있도록 정리하여 발표
  - state, action, reward, **episode 종료 조건**
  - 학습 방법
  - Idea 및 Algorithm Contribution
  - **실제 주행 영상을 포함한 2바퀴 lap time** 기재

---

## 5. 추가 프로젝트 (개인, 학점 + 부여)

아래 두 주제 중 선택. **성공적으로 수행할 경우 학점 + 부여**.

### 주제 1: Offline Reinforcement Learning for Autonomous Driving
- **목표**: 기존 주행 데이터를 활용하여 더 나은 주행 정책 학습
- **핵심 아이디어**
  - 약 **100초대 성능의 policy**를 이용하여 데이터 수집
  - 수집된 데이터를 기반으로 Offline RL 수행
  - 환경과의 추가 상호작용 없이 정책 개선
- **기대 결과**: 기존 policy 대비 더 빠른 lap time 달성

### 주제 2: Inverse Reinforcement Learning for Autonomous Driving
- **목표**: 주행 데이터를 기반으로 **보상 함수(reward function)를 학습**하고, 이를 통해 더 나은 주행 정책 학습
- **핵심 아이디어**
  - expert 주행 데이터를 활용하여 **숨겨진 보상 함수 추정**
  - 학습된 보상 함수를 기반으로 Reinforcement Learning 수행
  - 사람이 설계한 reward 없이 정책 학습
- **기대 결과**
  - 수작업 reward 대비 **더 일반화된 주행 성능 확보**
  - 새로운 트랙에서도 안정적인 주행 가능
  - 데이터로부터 학습된 reward의 효과 분석

### 개인 보고서 템플릿
- 내용: Agent와 구성한 환경을 수강자가 이해할 수 있도록 정리하여 발표
  - 문제 정의
  - 접근 방법
  - 학습 및 실험 방법
  - 성능 및 결과 분석 등
- 개인 보고서 제출

---

## 6. 시스템 아키텍처 (PDF p.28)

원활한 강화학습을 위해 **Custom env** 사용.

### 기본 F110-env 구조
```
Step → Observation(scan data, position) → Generate Action(speed, steer) → Step
       └─ Done 시 종료
```

### Custom env 적용 구조 (F110-env + Custom env + RL)
```
Step → Observation(scan data, position)
       ↓
   ┌───────────────────────────────────┐
   │ Custom env                        │
   │  ├─ Custom Reward                 │
   │  ├─ Custom Observation            │
   │  └─ Custom Terminal (Done 판단)   │
   └───────────────────────────────────┘
       ↓ Reward, State
   ┌─────────┐
   │   DQN   │ → Action → Step (Custom Dynamics 적용)
   └─────────┘
```

핵심:
- **Custom Reward**: `f110_env.py`의 `step()` 함수에서 reward 설계
- **Custom Observation**: obs 키를 가공하여 state 구성 (예: `preprocess_lidar`)
- **Custom Terminal**: 충돌/완주/타임아웃 등 episode 종료 조건 정의
- **Custom Dynamics**: 필요 시 차량 동역학 파라미터 수정

---

## 7. 트랙 정보

| 트랙 | 용도 |
|------|------|
| **Map Easy** | 초기 학습 / 알고리즘 완성도 확인 (평가 30%) |
| **Oschersleben** | 본 경기 트랙 / F1/10 Race Simulation (평가 10%) |

학습 순서 권장: **Map Easy → Oschersleben**

---

## 8. 참고 링크

- 대회 공식: https://roboracer.ai/
- 교수님 코드: https://oss.inha.ac.kr/project/detail/176
- F1tenth Gym: https://gitlab.com/acrome-colab/riders-poc/f1tenth-riders-quickstart
