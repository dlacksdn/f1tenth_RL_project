"""Phase 2-3 A9/A10 — f1tenth 12M parameter audit (CPU instantiate).

Builds the merged f1tenth config exactly as dreamer.py does (defaults + f1tenth
recursive_update + args_type flatten), instantiates WorldModel + ImagBehavior on
CPU, and reports the trainable parameter count + per-component breakdown.

A10 (v3 §2 / planning/005:175): total trainable params in [10M, 14M] AND ratio
report (RSSM ~30%, encoder+decoder ~50%, heads ~20% expected).

Run:
    source env/bin/activate
    python scripts/param_audit.py
"""
import os
import sys
import types

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENDOR = os.path.join(_ROOT, "vendor", "dreamerv3-torch")
for _p in (_ROOT, _VENDOR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gym  # old gym (matches the f1tenth adapter's spaces)
import torch

import models  # vendor
import tools   # vendor


def _coerce(v):
    """Coerce YAML scientific-notation strings (e.g. '1e-4') to float.

    PyYAML 6 (YAML 1.1) parses `1e-4`/`3e-5`/`1e6` as *strings*; dreamerv3's
    args_type does not coerce string defaults, so lr/steps stay strings and break
    the optimizer. Replicate the intended numeric coercion. Non-numeric strings
    ('SiLU', 'adam', '$^', task names) and other types pass through unchanged.
    """
    if isinstance(v, dict):
        return {k: _coerce(x) for k, x in v.items()}
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def build_config(suite_blocks=("f1tenth",), device="cpu"):
    """Replicate dreamer.py's defaults+block recursive merge, then numeric coerce."""
    with open(os.path.join(_VENDOR, "configs.yaml")) as f:
        configs = yaml.safe_load(f)

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    merged = {}
    for name in ("defaults", *suite_blocks):
        recursive_update(merged, configs[name])
    merged = {k: _coerce(v) for k, v in merged.items()}

    config = types.SimpleNamespace(**merged)
    # Set by main() from the env in a real run; fixed here for the audit.
    config.device = device
    config.num_actions = 2  # (steer, speed)
    return config


def f1tenth_spaces():
    """obs/act spaces matching vendor/.../envs/f1tenth.py adapter."""
    obs_space = gym.spaces.Dict({
        "lidar": gym.spaces.Box(0.0, 1.0, (1080,), dtype="float32"),
        "state": gym.spaces.Box(-float("inf"), float("inf"), (5,), dtype="float32"),
    })
    act_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(2,), dtype="float32"  # normalized action space
    )
    return obs_space, act_space


def _count(params):
    return sum(p.numel() for p in params if p.requires_grad)


def main():
    config = build_config()
    obs_space, act_space = f1tenth_spaces()

    wm = models.WorldModel(obs_space, act_space, step=0, config=config)
    behavior = models.ImagBehavior(config, wm)

    # Dreamer.parameters() = WorldModel + ImagBehavior (expl_behavior=greedy is an
    # alias to task_behavior; nn.Module dedups). Aggregate the same way.
    container = torch.nn.Module()
    container.wm = wm
    container.behavior = behavior
    total = _count(container.parameters())  # remove_duplicate=True dedups

    comps = {
        "encoder": _count(wm.encoder.parameters()),
        "decoder": _count(wm.heads["decoder"].parameters()),
        "rssm(dynamics)": _count(wm.dynamics.parameters()),
        "reward_head": _count(wm.heads["reward"].parameters()),
        "cont_head": _count(wm.heads["cont"].parameters()),
        "actor": _count(behavior.actor.parameters()),
        "value": _count(behavior.value.parameters()),
        "slow_value": _count(behavior._slow_value.parameters()),
    }

    enc_dec = comps["encoder"] + comps["decoder"]
    rssm = comps["rssm(dynamics)"]
    heads = comps["reward_head"] + comps["cont_head"] + comps["actor"] + comps["value"]

    print("=" * 64)
    print("F1Tenth DreamerV3 12M parameter audit (A10) — CPU instantiate")
    print("=" * 64)
    print(f"config: dyn_deter={config.dyn_deter} dyn_stoch={config.dyn_stoch} "
          f"dyn_discrete={config.dyn_discrete} dyn_hidden={config.dyn_hidden} "
          f"units={config.units}")
    print(f"        encoder={config.encoder}")
    print(f"        decoder={config.decoder}")
    print("-" * 64)
    for k, v in comps.items():
        print(f"  {k:18s} {v:>12,d}  ({100.0 * v / total:5.1f}%)")
    print("-" * 64)
    print(f"  {'encoder+decoder':18s} {enc_dec:>12,d}  ({100.0 * enc_dec / total:5.1f}%)  [~50% expected]")
    print(f"  {'RSSM':18s} {rssm:>12,d}  ({100.0 * rssm / total:5.1f}%)  [~30% expected]")
    print(f"  {'heads(r+c+a+v)':18s} {heads:>12,d}  ({100.0 * heads / total:5.1f}%)  [~20% expected]")
    print("=" * 64)
    print(f"  TOTAL trainable params: {total:,d}  ({total / 1e6:.2f}M)")
    in_band = 10_000_000 <= total <= 14_000_000
    print(f"  A10 [10M, 14M]: {'PASS' if in_band else 'FAIL'}")
    print("=" * 64)
    return 0 if in_band else 1


if __name__ == "__main__":
    sys.exit(main())
