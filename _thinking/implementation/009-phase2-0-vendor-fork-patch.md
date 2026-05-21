# 009 — Phase 2-0: dreamerv3-torch vendor-in + fork-patch (A20) + f1tenth make_env

> 2026-05-21. **노트북**(`env/`, torch 2.4.1+cpu) 세션. Phase 2-0 종료, mandatory stop.
> 선행: [008-centerline-reextraction.md](./008-centerline-reextraction.md), [003-sync-policy.md §5](./003-sync-policy.md).
> 관련 결정: [planning/005 v3 §1 #1/#14/#22/#25/#26, §2 A20, §2-3 config](../planning/005-f1tenth_dreamerV3_version3.md), [planning/007 fixed-HP §2-A/2-B](../planning/007-fixed_hp_fidelity.md).

---

## 0. 세션/결정 요약

- 이번 세션 작업 선택: **A (Phase 2-0→2-3, 임계경로)** — 사용자 "작업량 많은 걸로".
- **fork 동기화 방식 결정: vendor-in** (사용자 승인 2026-05-21). 근거: 원본 `/home/dlacksdn/dreamerv3-torch/`의 `origin`이 upstream `NM512/dreamerv3-torch`(개인 fork 아님 → push 권한 없음). 본 repo는 SSH push 동작 → 단일 repo atomic 동기화(노트북↔집컴)가 깔끔. 003 §5 명시 대안.
- 노트북 `env/`에 dreamerv3 deps 누락(집컴 `.venv`에만 설치, env_setting/006) → 사용자가 직접 설치: `tensorboard==2.14.0 ruamel.yaml==0.17.4 einops==0.3.0 moviepy==1.0.3 imageio-ffmpeg==0.5.1 protobuf==3.20.0`.

---

## 1. 산출물

| 파일 | 변경 |
|---|---|
| `vendor/dreamerv3-torch/` (신규) | NM512 fork vendor-in. upstream commit `6ef8646`. rsync 시 `.git/.omc/imgs/__pycache__/*.pyc/logdir` 제외 |
| `vendor/dreamerv3-torch/models.py:182` | `preprocess`: `obs["image"] = obs["image"]/255.0` → `if "image" in obs:` 가드로 감쌈 (#14, A20) |
| `vendor/dreamerv3-torch/dreamer.py` make_env | `elif suite == "f1tenth":` 분기 추가 → `F1Tenth` 어댑터 + `wrappers.NormalizeActions` (#22/#25) |
| `vendor/dreamerv3-torch/envs/f1tenth.py` (신규) | gymnasium `F110GymnasiumWrapper` → dreamerv3 내부 규약 변환 어댑터 |
| `vendor/dreamerv3-torch/configs.yaml` | `f1tenth:` 블록 append (Phase 2-0 skeleton; model/encoder는 2-3) |
| `vendor/dreamerv3-torch/VENDOR.md` (신규) | provenance + 패치 표 |
| `_thinking/patches/dreamerv3_torch_phase2-0.diff` (신규) | pristine upstream 대비 diff (v3 §7 재현성) |
| `dreamer_f1tenth/tests/conftest.py` | sys.path에 `vendor/dreamerv3-torch` 추가 |
| `dreamer_f1tenth/tests/test_preprocess_patch.py` (신규) | A20 검증 (2 test) |

---

## 2. 어댑터 설계 (★ analysis/008 §9-2 spot check 완료)

dreamerv3 내부 어댑터(`envs/dmc.py`)·wrapper(`envs/wrappers.py`) 규약을 실측 확인:

| 항목 | dreamerv3 내부 규약 | 우리 F110GymnasiumWrapper |
|---|---|---|
| step 반환 | **구 gym 4-tuple** `(obs, reward, done, info)` | gymnasium 5-tuple `(obs, r, term, trunc, info)` |
| reset 시그니처 | `reset() → obs` (seed/options 없음, info 없음) | `reset(seed, options) → (obs, info)` |
| obs의 is_first/is_terminal | 런타임 obs dict에 주입, **observation_space엔 미선언** | obs dict에 is_first/is_terminal/is_last 모두 |
| action_repeat | **어댑터 내부 루프**로 소비(dmc), config.action_repeat은 카운터/time_limit 회계용 | wrapper가 이미 내부 action_repeat=2 |
| spaces | **구 gym** `gym.spaces` | gymnasium spaces |

→ 직접 연결 불가. `envs/f1tenth.py` 어댑터가 우리 wrapper를 재사용하되 변환:
- `observation_space`: 인코더 입력 키 **lidar+state만** 구 gym `spaces.Dict`로 선언. is_first/is_terminal/is_last는 런타임 obs에만(MultiEncoder `excluded`가 필터) — dmc와 동일.
- `action_space`: 구 gym Box raw scale `[s_min,v_min]..[s_max,v_max]`. make_env 체인의 `NormalizeActions`가 [-1,1]로.
- `step`: wrapper 5-tuple → `done = terminated or truncated`, obs 그대로(is_terminal로 cont 학습, is_first로 RSSM reset).
- `reset`: wrapper `(obs,info)` → obs만.
- **action_repeat**: wrapper가 내부 소비(dmc 패턴 동일). config.action_repeat=2.
- **TimeLimit 이중처리 방지**: wrapper 내부 `max_episode_steps=10**9`로 비활성 → dreamerv3 `wrappers.TimeLimit`이 truncation 소유(결정 #22). `time_limit: 18000` → main의 `//action_repeat` → TimeLimit=9000 env-step=180s.

---

## 3. 검증

### 3-1. A20 + pytest 회귀

```
$ python -m pytest dreamer_f1tenth/tests/ -q
............  12 passed in 28.35s
```

| Criterion | 기준 | 결과 |
|---|---|---|
| A20 | vector-only obs(lidar/state, no image) 100 train + 10 eval preprocess 무에러 + cont 키 생성 | **PASS** (test_preprocess_patch.py::test_a20_preprocess_vector_only_no_keyerror) |
| A20 회귀 | image 존재 시 여전히 /255 (vision 경로 무변경) | **PASS** (test_a20_preprocess_still_divides_image_when_present) |
| 기존 회귀 A1~A6/A18/A_norm | 무영향 | **PASS** (10/10 기존 유지, 총 12/12) |

A20 테스트는 실제 `models.WorldModel.preprocess`를 dummy `_config`(device='cpu', discount=0.997)에 바인딩해 패치된 코드 경로를 충실히 검증(전체 인스턴스화는 2-3 영역).

### 3-2. 어댑터 smoke (make_env 경로)

`F1Tenth('map_easy3') → NormalizeActions` 체인 직접 구동:
- `observation_space.spaces` = `['lidar','state']` (is_* 미선언 ✓), 구 gym `gym.spaces.dict` ✓
- action raw `[-0.4189,-5]..[0.4189,20]` → normalized `[-1,-1]..[1,1]` ✓
- `reset() → dict` (5키, is_first=True), lidar (1080,), state (5,) ✓
- `step(a) → 4-tuple` (obs/float/done/info), is_terminal ∈ obs ∧ ∉ obs_space ✓ (dmc 규약 일치)

---

## 4. v3 acceptance SSOT 누적

| Criterion | 본 분기 후 |
|---|---|
| A20 | **PASS** (models.py:182 fork-patch, vector-only preprocess 무에러) |
| #14 fork-patch | vendor/dreamerv3-torch/models.py 단일 가드 적용 |
| #22 wrapper 체인 | f1tenth 어댑터로 구현: F110GymnasiumWrapper(내부 ar=2, timeout disabled) → NormalizeActions → (TimeLimit/SelectAction/UUID는 make_env 공통) |
| #25 map 명명 | configs `task='f1tenth_map_easy3'`, task split → trackname → TRACK_CONFIGS 키 일치 |
| fork 동기화 | **vendor-in** (003 §5, 본 repo 단일 동기화) |

---

## 5. 미확정 / 다음 단계 (Phase 2-1 진입 조건)

- `time_limit: 18000` / `steps: 5e5` 회계: main의 `//action_repeat` 반영해 TimeLimit=9000 env-step=180s 의도. **최종 step 예산은 A19 dry-run(2-4) 후 확정** (#11, §6-3 분기).
- configs `f1tenth:` 블록의 **model/encoder/decoder/12M + lidar_keys 섹션은 Phase 2-3에서 확정** (A9/A10). ConvEncoder1D(2-1)·MultiEncoder lidar 분기(2-2) 선행 필요.
- 알고리즘 HP는 007 §2-A대로 default 상속(override 금지) — config 블록에서 명시 안 함.
- **다음 분기 Phase 2-1**: `dreamer_f1tenth/networks_1d.py` (ConvEncoder1D 1080→540→270→135→68→34 flatten8704→Linear512 / ConvDecoder1D SymlogDist). A7/A7b CPU shape test. ~500줄 신규.

---

## 6. 체크리스트

- [x] vendor-in (rsync, 원본 repo·맵 무변경 — git porcelain로 확인).
- [x] models.py:182 image guard 패치 (#14).
- [x] envs/f1tenth.py 어댑터 (구 gym 4-tuple 변환, dmc 규약 일치).
- [x] dreamer.py make_env f1tenth 분기.
- [x] configs.yaml f1tenth skeleton 블록.
- [x] conftest.py vendor 경로 + test_preprocess_patch.py (A20).
- [x] pytest 12/12 PASS (회귀 없음) + 어댑터 smoke.
- [x] VENDOR.md provenance + _thinking/patches/ diff.
- [ ] commit + push (003 §3).
- [ ] **mandatory stop**: 사용자 보고 (A20 + fork 동기화 방식).
