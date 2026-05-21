# 012 — Phase 2-3: configs_f1tenth 12M + A9/A10 param audit (★ mandatory stop)

> 2026-05-21. **노트북**(`env/`, torch 2.4.1+cpu) 세션. Phase 2-3 종료 = mandatory stop (A10 보고).
> 선행: [011-phase2-2-multiencoder-lidar.md](./011-phase2-2-multiencoder-lidar.md), [planning/010 (#16 정정)](../planning/010-encoder1d-dims-A10-correction.md).
> 관련: [planning/005 v3 §2 A9/A10, §2-3 config, #3](../planning/005-f1tenth_dreamerV3_version3.md), [planning/007 §2-A/2-B/2-C](../planning/007-fixed_hp_fidelity.md).

---

## 1. 산출물

| 파일 | 변경 |
|---|---|
| `vendor/dreamerv3-torch/configs.yaml` | f1tenth 블록에 12M 차원 추가: dyn_hidden=256, dyn_deter=1024, dyn_stoch=32, dyn_discrete=16, units=256, encoder/decoder mlp_units=256/mlp_layers=5 (v3 §2-3, #3) |
| `scripts/param_audit.py` (신규) | f1tenth config 빌드(defaults+f1tenth recursive merge + numeric coerce) → CPU WorldModel+ImagBehavior instantiate → 파라미터 카운트+비율 (A9/A10) |
| `dreamer_f1tenth/networks_1d.py` | **A10 정정**(planning/010): ConvEncoder1D depths (16,32,64,128,256)→(16,32,64,128,128,128); ConvDecoder1D (256,128,64,32,16)→(128,128,128,64,32,16). flatten 8704→2176 |
| `dreamer_f1tenth/tests/test_networks_1d.py` | A7 stage_lengths→[…,34,17], final_ch→128, flat→2176 |

## 2. A9 — configs_f1tenth (8GB profile + 12M)

f1tenth 블록 = defaults 위 override (007 §2-A 알고리즘 HP는 미override=상속):
- 12M 모델: dyn_hidden=256, dyn_deter=1024, dyn_stoch=32, dyn_discrete=16, units=256, enc/dec mlp_units=256/mlp_layers=5.
- profile(§2-B): precision=16, batch_size=8, batch_length=64, compile=False, dataset_size=200000, prefill=0, eval_state_mean=True, eval_episode_num=20, video_pred_log=False, steps=5e5, time_limit=18000(//ar=2→9000 env-step), action_repeat=2, envs=1, parallel=False.
- routing(A8): encoder/decoder lidar_keys='lidar', mlp_keys='state', cnn_keys='$^', symlog_inputs=False, vector_dist=symlog_mse.
- **A9 PASS** (config 빌드+CPU instantiate 성공).

### 2-1. config 빌드 quirk (param_audit)
PyYAML 6 (YAML 1.1)이 `1e-4`/`3e-5`/`1e6`(소수점 없는 지수)를 **string**으로 파싱 → dreamerv3 args_type는 str 기본값 coerce 안 함 → Adam(lr='1e-4') TypeError. param_audit의 `_coerce`가 numeric 문자열만 float 변환(비numeric/bool/int 통과). (실제 학습은 dreamer.py argparse 경로라 별개 — 단 동일 quirk 잠재. Phase 2-4/3에서 CLI override 또는 동일 coerce 필요할 수 있음 — 기록.)

## 3. ★ A10 — 파라미터수 (mandatory stop 보고)

### 3-1. 1차 측정 = 26.58M FAIL → #16 정정 (planning/010)
#16의 flatten=8704가 decoder `Linear(1536→8704)=13.4M`를 만들어 26.58M(목표 2배). 사용자 결정으로 1D conv 차원 축소(fixed-HP 핵심 보존, 007 §2-C(2) 모달리티 인터페이스 범주). 5→6 stage + channel cap 128 → flatten 2176.

### 3-2. 정정 후 = **13.20M PASS**
| 컴포넌트 | params | 비율 |
|---|---|---|
| RSSM(dynamics) | 5,056,512 | 38.3% |
| decoder | 4,135,062 | 31.3% |
| encoder | 1,512,208 | 11.5% |
| reward+cont+actor+value | 1,971,459 | 14.9% |
| slow_value | 525,311 | 4.0% |
| **TOTAL** | **13,200,552 (13.20M)** | |
| enc+dec / RSSM / heads | 42.8% / 38.3% / 14.9% | (기대 ~50/30/20) |

- **A10 [10M,14M]: PASS.** 비율 보고 완료. RSSM-heavy(채널 cap)는 world-model 목적 부합.
- CPU instantiate 성공이 **Phase 2-2 device fix**(MultiEncoder/Decoder에 config.device 전달)를 end-to-end 검증.

## 4. 검증
- `python scripts/param_audit.py` → 13.20M PASS (위 표).
- `pytest dreamer_f1tenth/tests/` → **21/21 PASS** (networks_1d A7 갱신 포함, 회귀 없음).

## 5. v3 acceptance SSOT 누적
| Criterion | 본 분기 후 |
|---|---|
| A9 | **PASS** (configs_f1tenth 12M+8GB profile, CPU instantiate) |
| A10 | **PASS** (13.20M ∈ [10,14M] + 비율 보고) |
| #16 | **supersede** by planning/010 (flatten 8704→2176, 5→6 stage, ch cap 128) |
| device 일관성 | CPU full instantiate 검증 (2-2 fix) |

## 6. ★ mandatory stop — 다음 = Phase 2-4 (집컴 GPU 전용)
- Phase 2-4 A19 dry-run = **강제 게이트(#29)**: 노트북 push → 집컴 pull → `python scripts/dryrun_bench.py`로 VRAM(`max_memory_reserved`, ≤6400MB)+wall-clock(≤1440min) 측정. **노트북에선 여기서 정지.**
- 노트북에서 가능한 Phase 2 작업은 2-3까지 완료. dryrun_bench.py 작성은 2-4 진입 시(또는 노트북에서 미리 작성 가능하나 실행은 GPU).
- 후속 노트북 작업 후보: Phase 4(reward+episode, arclength lap, 작업 B) — fork 동기화 무관, CPU 가능.

## 7. 체크리스트
- [x] configs.yaml f1tenth 12M 차원.
- [x] scripts/param_audit.py (config 빌드 + CPU instantiate + 카운트).
- [x] A10 1차 26.58M FAIL → planning/010 정정 → 13.20M PASS.
- [x] networks_1d.py depths 정정 + test_networks_1d.py A7 갱신.
- [x] pytest 21/21.
- [ ] commit + push (003 §3).
- [ ] mandatory stop 보고 (A10 13.20M).
