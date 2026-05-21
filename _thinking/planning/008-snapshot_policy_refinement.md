# 008 — Snapshot 저장 정책 정밀화 (결정 #10 / A14 / A15 보강)

> 2026-05-21. 노트북 세션. Phase 1-4 작업 중 사용자 질의("전부 저장하면 너무 많지 않나")에서 파생.
> 선행: [005 v3 결정 #10(line 102), A14(line 188), A15(line 189), config(line 284-286)](./005-f1tenth_dreamerV3_version3.md).
> 본 문서는 **Phase 3(train.py) 구현 사양 SSOT**. 결정 #10/A14를 무한 증가 없는 형태로 정밀화.

---

## 1. 배경 / 문제

결정 #10/A14 원안: "eval lap_time ≤ 110s policy를 **전부** 저장 (`snapshot_save_all_below_threshold=True`)".

발견된 2가지 문제:
1. **무한 증가 + 중복**: eval 주기(`eval_every=1e4`) = interval snapshot 주기. 자격 policy가 interval 시점과 겹쳐 같은 step이 `step_{N}k.pt` + `policy_lap*.pt`로 **이중 저장**. full ckpt ~200MB(A15)라 stage당 +2~6GB.
2. **트랙별 임계 부재**: `snapshot_lap_threshold=110.0`은 **Oschersleben 기준**(A13 best ≤110s). map_easy3는 lap ~30~45s(A11 ≤45s)라 110s 임계로는 **모든 eval policy가 자격** → 변별력 0.

---

## 2. 확정 정책 (사용자 승인 2026-05-21)

| 종류 | 규칙 | 용량 |
|---|---|---|
| `latest.pt` | full ckpt (model+optimizer), resume용. 항상 1개. | ~200MB |
| interval `step_{N}k.pt` | A15 그대로 — `eval_every=1e4`마다 full ckpt. full skill-curve 다양성 + resume. **실질 bulk.** | ~10GB/stage |
| **diversity `policy_lap{X:.1f}s_step{Y}k.pt`** | **트랙별 임계 T를 5등분**한 lap-time bin마다 **fastest 1개**만. **optimizer state 제거**(world_model+actor inference만). interval과 step 중복 시 별도 저장하되 optimizer 없어 경량. | ~50MB × 최대 5 = ~250MB/stage |

### 2-1. 트랙별 임계 T 및 bin

| Track | T (임계) | 근거 | bin (T/5 폭, [0,T] 5등분) |
|---|---|---|---|
| map_easy3 | **45s** | A11 baseline (≤45s) | (0,9],(9,18],(18,27],(27,36],(36,45] |
| Oschersleben | **110s** | A13 best (≤110s) | (0,22],(22,44],(44,66],(66,88],(88,110] |

- bin마다 학습 중 관측된 **최저 lap_time policy 1개**만 유지(더 빠른 게 나오면 교체).
- bin 상한 = 5개/stage → diversity policy 총 **≤ 10개**(2 stage).

### 2-2. optimizer 제거 근거
- diversity policy 용도 = LeWorldModel offline dataset 생성(world_model rollout + actor 추론). **resume 불필요** → Adam optimizer state(파라미터 2배) 저장 안 함. full ~200MB → ~50MB.
- resume는 `latest.pt` + interval `step_*k.pt`(full ckpt)가 담당.

---

## 3. 다양성 충분성 (LeWorldModel)

- **behavioral diversity 주 공급원 = interval 50개/stage** (random→expert 전 skill 구간). offline RL은 skill 스펙트럼을 span하는 데이터에서 이득.
- diversity policy = 그 위에 **고품질 expert 구간을 lap-time 스펙트럼으로 큐레이션**(bin별 best).
- 합계: interval 100개(2 stage) + curated ≤10개 → offline dataset 다양성 충분 판단.

---

## 4. 총 용량 추정

| 항목 | 용량 |
|---|---|
| interval (bulk) | ~10GB/stage × 2 = **~20GB** |
| diversity policy | ~250MB/stage × 2 = **~0.5GB** |
| latest | ~200MB |
| **합계** | **~21GB** |

- 원안(≤110s 전부 full ckpt) 대비 무한 증가·중복 제거. diversity는 사실상 무시 가능.
- **2차 레버**(추후 disk 부담 시): interval을 5e4 간격(10개/stage, ~4GB)으로 thinning 가능. A15 기본값(1e4)은 유지, 필요 시 Phase 5에서 재결정.

---

## 5. Phase 3 구현 체크포인트 (train.py)

- config 키 정밀화: `snapshot_lap_threshold`를 **트랙별 dict** 또는 task별 override로 변경(map_easy3=45, Oschersleben=110). v3 config의 단일 `110.0`은 본 문서로 보강.
- `snapshot_save_all_below_threshold=True` → **best-per-bin(5 bin)** 로직으로 대체. (의미 변경: "전부" → "bin별 최고 1개")
- diversity 저장 시 optimizer state 제외한 partial state_dict 저장 함수 분리.
- interval/latest는 기존 full ckpt 유지.
