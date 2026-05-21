# 011 — Phase 2-2: MultiEncoder/MultiDecoder lidar_keys 분기 (A8)

> 2026-05-21. **노트북**(`env/`, torch 2.4.1+cpu) 세션. Phase 2-2 종료 (mandatory stop 아님).
> 선행: [010-phase2-1-networks-1d.md](./010-phase2-1-networks-1d.md).
> 관련: [planning/005 v3 §2 A8, §3 2-2, #5/#15/#16](../planning/005-f1tenth_dreamerV3_version3.md), [analysis/004 §3-1/§3-3 MultiEncoder/Decoder](../analysis/004-dreamer_code_analysis_part2.md).

---

## 1. 산출물

| 파일 | 변경 |
|---|---|
| `vendor/dreamerv3-torch/networks.py` | `_import_lidar_nets()` (lazy, 프로젝트 루트 path 삽입); MultiEncoder `__init__`/`forward`에 `lidar_keys`/`lidar_units`/`device` + lidar 분기; MultiDecoder 동일 + `device` |
| `vendor/dreamerv3-torch/models.py` | WorldModel가 `device=config.device`를 MultiEncoder/MultiDecoder에 전달 (L38, L62) |
| `vendor/dreamerv3-torch/configs.yaml` | f1tenth encoder/decoder 라우팅 키 (lidar_keys='lidar', mlp_keys='state', cnn_keys='$^', #15 symlog_inputs=False, vector_dist=symlog_mse) |
| `dreamer_f1tenth/tests/test_multiencoder_lidar.py` (신규) | A8 (4 test) |

## 2. 구현 (decision #5/#16, A8)

**라우팅**: `len(shape)==1 ∧ lidar_keys 매치` → ConvEncoder1D/ConvDecoder1D. mlp_shapes에서 `k not in lidar_shapes`로 lidar 제외(이중 라우팅 방지, mlp_keys='.*'에도 robust). cnn(3D)은 그대로.
- Encoder: `outdim = cnn + lidar_units(512) + mlp_units`. forward concat `[cnn, lidar, mlp]` (cnn 없으면 `[lidar, mlp]` = #16).
- Decoder: `dists[lidar_key] = ConvDecoder1D(feat).SymlogDist`. `dists` dict에 state MLP 머지.
- `lidar_keys` 기본 `'$^'`(매치 없음) → 타 suite 무영향(opt-in). `assert len(lidar_shapes)==1`로 다중 lidar 키는 명시적 실패.
- import: `_import_lidar_nets()`가 lazy(분기 활성 시만), `vendor/dreamerv3-torch/networks.py`→dirname×3=프로젝트 루트 삽입 후 `dreamer_f1tenth.networks_1d` import.

## 3. ★ device fix (CPU instantiate 차단 근본원인 — 리뷰어 확인)

- `MLP._std`(networks.py:686)는 **registered buffer가 아니라 plain attribute** → 생성 시 `device="cuda"`(MLP 기본)에 eager 텐서로 즉시 할당. agent-level `dreamer.py:298 .to(config.device)`가 **이 텐서를 이전하지 못함**.
- upstream은 reward/cont head MLP에만 `device=config.device`를 넘기고 **MultiEncoder/MultiDecoder 내부 MLP에는 안 넘김** → CPU-only torch에서 `MultiEncoder` 구성 즉시 "Torch not compiled with CUDA enabled" 실패(upstream 잠재 비호환).
- **fix**: MultiEncoder/MultiDecoder에 `device` 파라미터 추가(기본 "cuda", GPU 런 무영향) + models.py가 `config.device` 전달. → CPU instantiate 가능, GPU 런 일관(`config.device='cuda:0'`).
- ConvEncoder1D/ConvDecoder1D는 eager 텐서 없음(전부 registered Parameter) → device 파라미터 불필요, agent-level `.to()`로 충분.

## 4. 검증

### 4-1. A8 pytest (CPU, device='cpu')
```
test_multiencoder_lidar.py: 4 passed
전체: 21 passed (009의 12 + 010의 5 + 본 4, 회귀 없음)
```
| Criterion | 검증 | 결과 |
|---|---|---|
| A8 encoder | lidar→1D, state→mlp, cnn={}, outdim=512+256=768, embed (2,3,768) | **PASS** |
| A8 double-route 방지 | mlp_keys='.*'에도 lidar는 mlp 제외 | **PASS** |
| A8 decoder | dict {lidar:SymlogDist mode(2,3,1080) lp(2,3), state lp(2,3)} | **PASS** |
| A8 타 suite 격리 | lidar_keys 미지정 시 lidar_shapes={}, outdim=256 | **PASS** |

### 4-2. code-reviewer(opus) 별도 패스 — APPROVE
- 0 Critical/0 High. 실행 기반 검증: 라우팅·outdim(측정 576=512+64)·decoder dict·non-f1tenth 격리(lazy import 미발화)·**CPU instantiate+forward(encoder+decoder)**·device kwarg 충돌 없음(전 suite config에 device 키 없음 확인)·AMP autocast inherit.
- device fix 근본원인(`_std` non-buffer) 정확 진단 확인. Medium 1(주석 "cuda" vs "cuda:0" cosmetic, 런타임 무영향)·Low 3(vector_dist 공유/ print/ video_pred image 키-기존 가드됨) — 전부 조치 불필요.

## 5. v3 acceptance SSOT 누적
| Criterion | 본 분기 후 |
|---|---|
| A8 | **PASS** (MultiEncoder/Decoder lidar_keys 분기, "Encoder/Decoder LIDAR shapes" 출력) |
| #15 | f1tenth encoder symlog_inputs=False |
| device 일관성 | MultiEncoder/Decoder도 config.device 사용 (CPU instantiate 가능) |

## 6. 다음 단계 (Phase 2-3, ★ mandatory stop)
- f1tenth config의 **model/12M 차원 확정**: dyn_hidden=256, dyn_deter=1024, dyn_stoch=32, dyn_discrete=16, units=256, encoder/decoder mlp_units=256, mlp_layers=5 (v3 §2-3). 007 §2-A 알고리즘 HP는 default 상속(override 금지).
- **A9**: configs_f1tenth 완성 (precision=16/batch_size=8/compile=False/prefill=0/eval_state_mean=True 등 이미 skeleton).
- **A10**: CPU full instantiate → `sum(p.numel() for p in agent.parameters() if requires_grad)` ∈ [10M,14M] + 비율 보고(RSSM~30%, enc+dec~50%, heads~20%). 리뷰어가 CPU instantiate 가능 확인했으므로 device 막힘 해소됨.
- Phase 2-3 종료 = mandatory stop (A10 파라미터수 보고).

## 7. 체크리스트
- [x] networks.py MultiEncoder/MultiDecoder lidar 분기 + _import_lidar_nets + device.
- [x] models.py device=config.device 전달.
- [x] configs.yaml f1tenth encoder/decoder 라우팅.
- [x] test_multiencoder_lidar.py (A8, 4 test).
- [x] pytest 21/21 PASS.
- [x] code-reviewer(opus) APPROVE.
- [ ] commit + push (003 §3).
