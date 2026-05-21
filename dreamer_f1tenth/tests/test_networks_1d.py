"""A7 / A7b — ConvEncoder1D / ConvDecoder1D shape tests (Phase 2-1, decision #5/#16).

CPU-only. Verifies the LiDAR 1D conv path produces the spec'd shapes:
  A7  : ConvEncoder1D (B,T,1080) -> (B,T,512); length schedule 1080->540->270
        ->135->68->34; flatten 8704.
  A7b : ConvDecoder1D feat (B,T,F) -> SymlogDist; mode (B,T,1080); log_prob (B,T).
"""
import numpy as np
import pytest
import torch

from dreamer_f1tenth.networks_1d import ConvEncoder1D, ConvDecoder1D

B, T = 2, 3
LIDAR_LEN = 1080
FEAT = 1536  # 12M discrete RSSM feat = dyn_stoch*dyn_discrete + dyn_deter = 32*16 + 1024


def test_a7_encoder_shape_and_schedule():
    enc = ConvEncoder1D(input_len=LIDAR_LEN, out_dim=512)
    assert enc.stage_lengths == [1080, 540, 270, 135, 68, 34]
    assert enc._final_ch == 256
    assert enc._flat_dim == 8704
    assert enc.outdim == 512

    x = torch.rand(B, T, LIDAR_LEN)
    out = enc(x)
    assert out.shape == (B, T, 512)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_a7_encoder_backward():
    enc = ConvEncoder1D(input_len=LIDAR_LEN, out_dim=512)
    x = torch.rand(B, T, LIDAR_LEN, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_a7b_decoder_symlogdist_shapes():
    dec = ConvDecoder1D(feat_size=FEAT, output_len=LIDAR_LEN)
    feat = torch.rand(B, T, FEAT)
    dist = dec(feat)
    # SymlogDist exposes mode()/log_prob()
    mode = dist.mode()
    assert mode.shape == (B, T, LIDAR_LEN)

    target = torch.rand(B, T, LIDAR_LEN)
    lp = dist.log_prob(target)
    assert lp.shape == (B, T)            # agg="sum" reduces the 1080 axis
    assert torch.isfinite(lp).all()
    assert (lp <= 0).all()               # -sum(squared) <= 0


def test_a7b_decoder_backward():
    dec = ConvDecoder1D(feat_size=FEAT, output_len=LIDAR_LEN)
    feat = torch.rand(B, T, FEAT, requires_grad=True)
    target = torch.rand(B, T, LIDAR_LEN)
    loss = -dec(feat).log_prob(target).mean()
    loss.backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()


def test_a7b_decoder_reconstructs_normalized_lidar_range():
    """LiDAR obs is clip(0,30)/30 in [0,1]; symexp(mode) should be finite there."""
    dec = ConvDecoder1D(feat_size=FEAT, output_len=LIDAR_LEN)
    feat = torch.zeros(B, T, FEAT)
    mode = dec(feat).mode()
    assert torch.isfinite(mode).all()
