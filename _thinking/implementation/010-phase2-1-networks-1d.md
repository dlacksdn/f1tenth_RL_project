# 010 — Phase 2-1: ConvEncoder1D / ConvDecoder1D (networks_1d.py, A7/A7b)

> 2026-05-21. **노트북**(`env/`, torch 2.4.1+cpu) 세션. Phase 2-1 종료 (mandatory stop 아님 — 2-0/2-3/2-4만 해당).
> 선행: [009-phase2-0-vendor-fork-patch.md](./009-phase2-0-vendor-fork-patch.md).
> 관련: [planning/005 v3 §1 #5/#16, §2 A7/A7b, §3 2-1](../planning/005-f1tenth_dreamerV3_version3.md), [analysis/004 §3-2/§3-4 Conv, §1-3 SymlogDist](../analysis/004-dreamer_code_analysis_part2.md), [planning/007 §2-A/2-C(1D encoder 신규)](../planning/007-fixed_hp_fidelity.md).

---

## 1. 산출물

| 파일 | 변경 |
|---|---|
| `dreamer_f1tenth/networks_1d.py` (신규, ~250줄) | `ConvEncoder1D`, `ConvDecoder1D`, `Ch1dLayerNorm`, Conv1d-aware `weight_init_1d`/`uniform_weight_init_1d` |
| `dreamer_f1tenth/tests/test_networks_1d.py` (신규, 5 test) | A7/A7b shape + backward + range |

## 2. 사양 구현 (decision #5/#16)

**ConvEncoder1D** `(B,T,1080) → (B,T,512)`:
- 5-stage Conv1d(k=3, s=2, p=1, bias=`not norm`) + Ch1dLayerNorm + SiLU. 채널 1→16→32→64→128→256.
- 길이 `(L-1)//2+1`: **1080→540→270→135→68→34** (실측 검증). flatten 256×34=**8704** → `Linear(8704, 512)`.

**ConvDecoder1D** feat `(B,T,F) → SymlogDist over (B,T,1080)`:
- `Linear(F, 256×34)` → reshape (B*T,256,34) → 5-stage ConvTranspose1d(k=3,s=2,p=1) 채널 256→128→64→32→16→1.
- 길이 `2L-1+output_padding`, op=[1,0,1,1,1]로 34→68→135→270→540→1080. `assert length==output_len`로 빌드타임 가드.
- 출력 `tools.SymlogDist(mean, dist="mse", agg="sum")` — vector_dist `symlog_mse` 동치. `log_prob(target)` → dim[2:] sum → **(B,T)**.

### 2-1. 충실도 (analysis/004 §3 대조)
- bias=False under norm, channel-axis LayerNorm(eps=1e-3), SiLU — fork ConvEncoder/ImgChLayerNorm 동일.
- **init**: fork `tools.weight_init`은 Conv2d만 처리(Conv1d 미지원) → 동일 공식(fan_avg, space=kernel_size[0], std=√(1/fan)/0.8796 / uniform limit=√(3·scale/fan))을 Conv1d용으로 로컬 복제. Linear/LayerNorm은 `tools` 위임. 디코더 backbone=weight_init, 최종 conv+linear=uniform_weight_init(outscale) — fork ConvDecoder `layers[:-1]`/`layers[-1]` 패턴 동일.
- 디코더 길이 스케줄을 인코더 forward 공식으로 역산(하드코딩 X) → enc/dec drift 불가.
- 1D LiDAR encoder/decoder는 007 §2-C(2) "불가피한 신규 모달리티 확장"(HP 튜닝 아님)에 해당.

## 3. 검증

### 3-1. A7/A7b pytest (CPU)
```
test_networks_1d.py: 5 passed
전체: 17 passed in 16.45s (009의 12 + 본 5, 회귀 없음)
```
| Criterion | 검증 | 결과 |
|---|---|---|
| A7 | encoder (2,3,1080)→(2,3,512), stage_lengths=[1080,540,270,135,68,34], flat=8704, backward 유한 | **PASS** |
| A7b | decoder SymlogDist mode (2,3,1080), log_prob (2,3)≤0, backward 유한, symexp 유한 | **PASS** |

### 3-2. code-reviewer(opus) 별도 검증 패스 (자체 승인 금지 원칙)
- **APPROVE**, 0 Critical/0 High, 1 Medium + 3 Low.
- 핵심 위험(init 복제)을 **실제 가중치 텐서 측정**으로 검증: 최종 ConvTranspose1d uniform max|w|=0.3317(expected 0.343), backbone trunc_normal cap=0.09474(==expected), LayerNorm ones/zeros, linear uniform==0.025516. 모두 정확.
- 우려했던 double-init(Ch1dLayerNorm 래퍼)·`layers[-1]` 슬라이싱(최종 conv 맞음)·AMP(bf16 mode + f32 target upcast)·reshape contiguity — 전부 non-issue 확인.
- **적용한 hardening**: (MEDIUM) decoder `depths` 계약 — `assert n==len(target_lens)` + bottleneck/terminal-ch=1 주석. (LOW) stride/padding 명명 로컬화 + assert `< stride`. (LOW) tools 위임 no-op 의존 주석. hardening 후 17/17 유지.

## 4. v3 acceptance SSOT 누적
| Criterion | 본 분기 후 |
|---|---|
| A7 | **PASS** (ConvEncoder1D 1080→512, 8704 flatten) |
| A7b | **PASS** (ConvDecoder1D SymlogDist over 1080) |

## 5. 다음 단계 (Phase 2-2)
- `vendor/dreamerv3-torch/networks.py` MultiEncoder/MultiDecoder에 **lidar_keys 분기** 추가 (A8): `len(shape)==1 ∧ lidar 매치` → ConvEncoder1D/ConvDecoder1D 경로. forward concat 순서 `[lidar_out, mlp_out]`, encoder outdim=512+mlp_units. configs encoder/decoder에 `lidar_keys: 'lidar'` 추가.
- import 경로: networks_1d는 `dreamer_f1tenth` 패키지 → networks.py(vendor)에서 import 시 프로젝트 루트 path 필요(어댑터 envs/f1tenth.py가 이미 sys.path 처리). 2-2에서 networks.py가 직접 import할지 검토.

## 6. 체크리스트
- [x] networks_1d.py (ConvEncoder1D/ConvDecoder1D/Ch1dLayerNorm/Conv1d init).
- [x] test_networks_1d.py (A7/A7b, 5 test).
- [x] pytest 17/17 PASS (회귀 없음).
- [x] code-reviewer(opus) APPROVE + hardening 적용.
- [ ] commit + push (003 §3).
