# Vendored: NM512/dreamerv3-torch

Vendored into this repo (decision: vendor-in, planning/003 §5) on 2026-05-21.

- **Upstream**: https://github.com/NM512/dreamerv3-torch
- **Upstream commit**: `6ef8646d807cd10ce0c88e10a7e943211e7fc44c` (main)
- **Excluded on copy**: `.git/`, `.omc/`, `imgs/`, `__pycache__/`, `*.pyc`, `logdir`
- **Why vendor-in**: upstream `origin` is the NM512 repo (no push access); main
  project repo has working SSH push → single-repo atomic notebook↔desktop sync.

## F1Tenth fork-patches (vs upstream, see _thinking/patches/dreamerv3_torch_phase2-0.diff)

| File | Change | Ref |
|---|---|---|
| `models.py:182` | `preprocess`: guard `if "image" in obs:` before `/255` (vector-only obs) | decision #14, A20 |
| `dreamer.py` make_env | `elif suite == "f1tenth"` branch → `F1Tenth` adapter + `NormalizeActions` | decision #22/#25 |
| `configs.yaml` | append `f1tenth:` block (Phase 2-0 skeleton; model/encoder finalized 2-3) | v3 §2-3 |
| `envs/f1tenth.py` | NEW adapter: gymnasium `F110GymnasiumWrapper` → dreamerv3 4-tuple convention | decision #22 |

All other algorithm code is untouched (007 fixed-HP fidelity §2-A).
