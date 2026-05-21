"""A8 — MultiEncoder/MultiDecoder lidar_keys branch (Phase 2-2, decision #5/#16).

CPU-only. Verifies the fork-patched MultiEncoder/MultiDecoder route the LiDAR
1-D key to ConvEncoder1D/ConvDecoder1D and the vector 'state' key to the MLP:
  - encoder: lidar -> 1D path, state -> mlp; outdim == lidar_units + mlp_units.
  - decoder: returns a dict {lidar: SymlogDist, state: <dist>} with correct shapes.
"""
import numpy as np
import pytest
import torch

import networks  # vendor/dreamerv3-torch (on sys.path via conftest)
import tools

B, T = 2, 3
LIDAR_LEN = 1080
FEAT = 1536  # 12M discrete RSSM feat = 32*16 + 1024
SHAPES = {"lidar": (LIDAR_LEN,), "state": (5,)}

ENC_KW = dict(
    mlp_keys="state", cnn_keys="$^", lidar_keys="lidar",
    act="SiLU", norm=True, cnn_depth=32, kernel_size=4, minres=4,
    mlp_layers=2, mlp_units=256, symlog_inputs=False, lidar_units=512,
    device="cpu",
)
DEC_KW = dict(
    mlp_keys="state", cnn_keys="$^", lidar_keys="lidar",
    act="SiLU", norm=True, cnn_depth=32, kernel_size=4, minres=4,
    mlp_layers=2, mlp_units=256, cnn_sigmoid=False,
    image_dist="mse", vector_dist="symlog_mse", outscale=1.0,
    device="cpu",
)


def test_a8_encoder_routing_and_outdim():
    enc = networks.MultiEncoder(SHAPES, **ENC_KW)
    assert set(enc.lidar_shapes) == {"lidar"}
    assert set(enc.mlp_shapes) == {"state"}
    assert enc.cnn_shapes == {}
    # outdim = lidar (512) + mlp_units (256) = 768  (#16)
    assert enc.outdim == 512 + 256

    obs = {
        "lidar": torch.rand(B, T, LIDAR_LEN),
        "state": torch.rand(B, T, 5),
    }
    embed = enc(obs)
    assert embed.shape == (B, T, 768)
    assert torch.isfinite(embed).all()


def test_a8_lidar_not_double_routed_to_mlp():
    # mlp_keys that would also match lidar must not double-route it.
    enc = networks.MultiEncoder(SHAPES, **{**ENC_KW, "mlp_keys": ".*"})
    assert set(enc.lidar_shapes) == {"lidar"}
    assert set(enc.mlp_shapes) == {"state"}  # lidar excluded from mlp


def test_a8_decoder_dict_and_shapes():
    dec = networks.MultiDecoder(FEAT, SHAPES, **DEC_KW)
    assert set(dec.lidar_shapes) == {"lidar"}
    assert set(dec.mlp_shapes) == {"state"}

    feat = torch.rand(B, T, FEAT)
    dists = dec(feat)
    assert set(dists) == {"lidar", "state"}

    lidar_dist = dists["lidar"]
    assert isinstance(lidar_dist, tools.SymlogDist)
    assert lidar_dist.mode().shape == (B, T, LIDAR_LEN)
    lp = lidar_dist.log_prob(torch.rand(B, T, LIDAR_LEN))
    assert lp.shape == (B, T)

    # state vector head log_prob reduces to (B, T)
    state_lp = dists["state"].log_prob(torch.rand(B, T, 5))
    assert state_lp.shape == (B, T)


def test_a8_other_suites_unaffected_default_lidar_keys():
    # Without lidar_keys, default '$^' matches nothing -> no lidar branch.
    enc = networks.MultiEncoder(
        {"state": (5,)},
        mlp_keys="state", cnn_keys="$^",
        act="SiLU", norm=True, cnn_depth=32, kernel_size=4, minres=4,
        mlp_layers=2, mlp_units=256, symlog_inputs=False, device="cpu",
    )
    assert enc.lidar_shapes == {}
    assert set(enc.mlp_shapes) == {"state"}
    assert enc.outdim == 256
