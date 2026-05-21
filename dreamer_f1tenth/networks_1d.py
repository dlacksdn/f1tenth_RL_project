"""1D LiDAR encoder/decoder for DreamerV3 (Phase 2-1, decision #5/#16, A7/A7b).

The fork's MultiEncoder/MultiDecoder route image (3D) to ConvEncoder and vectors
(1D/2D) to an MLP. F1Tenth's LiDAR is a 1080-beam 1D signal whose spatial
(angular) structure an MLP throws away. This module adds the missing 1D-conv
path; Phase 2-2 wires it into MultiEncoder/MultiDecoder via ``lidar_keys``.

ConvEncoder1D (decision #16):
    (B, T, 1080) -> 5x [Conv1d stride2] -> (B*T, 256, 34) -> flatten 8704
                 -> Linear(8704, 512) -> (B, T, 512)
    lengths: 1080 -> 540 -> 270 -> 135 -> 68 -> 34   (k=3, s=2, p=1: (L-1)//2 + 1)
    channels: 1 -> 16 -> 32 -> 64 -> 128 -> 256

ConvDecoder1D (mirror, SymlogDist over the 1080 vector):
    feat -> Linear(feat, 256*34) -> (B*T, 256, 34)
         -> 5x [ConvTranspose1d stride2] -> (B*T, 1, 1080) -> (B, T, 1080)
    output_padding per stage chosen to hit 34->68->135->270->540->1080.

Conventions mirror the fork (analysis/004 §3): bias=False under norm, channel-axis
LayerNorm (eps=1e-3), SiLU, fan_avg trunc_normal backbone init, uniform output
init scaled by ``outscale``. ``tools.weight_init`` only covers Conv2d, so Conv1d
init is replicated here with the same formula.
"""
import numpy as np
import torch
from torch import nn

import tools  # vendor/dreamerv3-torch (on sys.path); symlog/SymlogDist + init helpers

_TRUNC_STD_CORRECTION = 0.87962566103423978  # std of trunc_normal[-2,2]; mirrors tools


# ---------------------------------------------------------------------------
# Init (Conv1d-aware; Linear/LayerNorm delegate to fork's tools)
# ---------------------------------------------------------------------------
def _conv1d_fan(m):
    space = m.kernel_size[0]
    return space * m.in_channels, space * m.out_channels


def weight_init_1d(m):
    """fan_avg truncated-normal init; Conv1d handled locally, rest via tools."""
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        in_num, out_num = _conv1d_fan(m)
        denoms = (in_num + out_num) / 2.0
        std = np.sqrt(1.0 / denoms) / _TRUNC_STD_CORRECTION
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if getattr(m, "bias", None) is not None and hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    else:
        # Relies on tools.weight_init being a no-op for unhandled types
        # (Ch1dLayerNorm wrapper, SiLU, Sequential); handles Linear/LayerNorm.
        tools.weight_init(m)


def uniform_weight_init_1d(given_scale):
    """uniform output init scaled by given_scale; Conv1d local, rest via tools."""
    fork_f = tools.uniform_weight_init(given_scale)

    def f(m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            in_num, out_num = _conv1d_fan(m)
            denoms = (in_num + out_num) / 2.0
            limit = np.sqrt(3 * given_scale / denoms)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if getattr(m, "bias", None) is not None and hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        else:
            fork_f(m)  # nn.Linear / nn.LayerNorm

    return f


# ---------------------------------------------------------------------------
# Channel-axis LayerNorm for (B, C, L) — 1D analogue of fork's ImgChLayerNorm
# ---------------------------------------------------------------------------
class Ch1dLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-3):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):  # (B, C, L)
        x = x.permute(0, 2, 1)        # (B, L, C)
        x = self.norm(x)
        return x.permute(0, 2, 1)     # (B, C, L)


def _conv1d_out_len(length, kernel_size=3, stride=2, padding=1, dilation=1):
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class ConvEncoder1D(nn.Module):
    """1D LiDAR encoder: (B, T, input_len) -> (B, T, out_dim). decision #5/#16, A7."""

    def __init__(
        self,
        input_len=1080,
        out_dim=512,
        # 6 stride-2 stages, channels capped at 128 (A10-driven, planning/010):
        # 1080->540->270->135->68->34->17, flatten 128*17=2176. The original
        # #16 spec (5 stages -> 8704 flatten) made the mirror decoder's
        # Linear(feat,8704)=13.4M and blew the 12M target (measured 26.6M).
        depths=(16, 32, 64, 128, 128, 128),
        kernel_size=3,
        act="SiLU",
        norm=True,
    ):
        super().__init__()
        Act = getattr(nn, act)
        self.input_len = int(input_len)
        self.outdim = int(out_dim)

        layers = []
        in_ch = 1
        length = self.input_len
        self.stage_lengths = [length]
        for out_ch in depths:
            layers.append(
                nn.Conv1d(
                    in_ch, out_ch, kernel_size, stride=2, padding=1, bias=not norm
                )
            )
            if norm:
                layers.append(Ch1dLayerNorm(out_ch))
            layers.append(Act())
            in_ch = out_ch
            length = _conv1d_out_len(length, kernel_size)
            self.stage_lengths.append(length)
        self.layers = nn.Sequential(*layers)

        self._final_ch = in_ch                 # 256
        self._final_len = length               # 34
        self._flat_dim = self._final_ch * self._final_len   # 8704
        self._linear = nn.Linear(self._flat_dim, self.outdim)

        self.layers.apply(weight_init_1d)
        self._linear.apply(weight_init_1d)

    def forward(self, lidar):
        # lidar: (B, T, input_len)
        assert lidar.dim() == 3, f"expected (B, T, L), got {tuple(lidar.shape)}"
        b, t, length = lidar.shape
        assert length == self.input_len, f"len {length} != {self.input_len}"
        x = lidar.reshape(b * t, 1, length)    # (B*T, 1, L)
        x = self.layers(x)                     # (B*T, 256, 34)
        x = x.reshape(b * t, self._flat_dim)   # (B*T, 8704)
        x = self._linear(x)                    # (B*T, 512)
        return x.reshape(b, t, self.outdim)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class ConvDecoder1D(nn.Module):
    """1D LiDAR decoder: feat (B, T, F) -> SymlogDist over (B, T, output_len). A7b."""

    def __init__(
        self,
        feat_size,
        output_len=1080,
        # mirror of ConvEncoder1D (6 stages, bottleneck 128). depths[0] is the
        # bottleneck channel; output channel is fixed at 1 (planning/010).
        depths=(128, 128, 128, 64, 32, 16),
        kernel_size=3,
        act="SiLU",
        norm=True,
        outscale=1.0,
        dist="symlog_mse",
    ):
        super().__init__()
        Act = getattr(nn, act)
        self.output_len = int(output_len)
        self._dist = "mse" if dist == "symlog_mse" else dist

        # Mirror the encoder length schedule: derive start_len + per-stage targets.
        enc_lens = [self.output_len]
        for _ in depths:
            enc_lens.append(_conv1d_out_len(enc_lens[-1], kernel_size))
        self._start_len = enc_lens[-1]              # 34
        target_lens = enc_lens[-2::-1]              # [68, 135, 270, 540, 1080]
        self._start_ch = depths[0]                  # 256

        self._linear = nn.Linear(feat_size, self._start_ch * self._start_len)

        # depths[0] is the bottleneck channel; the terminal output channel is
        # fixed at 1 (single LiDAR vector). Stage count must mirror the encoder.
        stride, padding = 2, 1
        out_channels = list(depths[1:]) + [1]       # [128, 64, 32, 16, 1]
        n = len(out_channels)
        assert n == len(target_lens), (
            f"decoder stages {n} != encoder stages {len(target_lens)} "
            f"(depths={depths})"
        )
        layers = []
        in_ch = self._start_ch
        length = self._start_len
        for i, out_ch in enumerate(out_channels):
            target = target_lens[i]
            # ConvTranspose1d out = (L-1)*s - 2p + d*(k-1) + 1 + output_padding
            base = (length - 1) * stride - 2 * padding + 1 * (kernel_size - 1) + 1
            output_padding = target - base
            assert 0 <= output_padding < stride, (
                f"stage {i}: out_pad {output_padding} out of range "
                f"(L={length} -> target={target})"
            )
            last = i == n - 1
            layers.append(
                nn.ConvTranspose1d(
                    in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                    output_padding=output_padding, bias=True if last else (not norm),
                )
            )
            if not last:
                if norm:
                    layers.append(Ch1dLayerNorm(out_ch))
                layers.append(Act())
            in_ch = out_ch
            length = target
        self.layers = nn.Sequential(*layers)
        assert length == self.output_len

        # Mirror fork's ConvDecoder init: backbone trunc_normal, final + linear uniform.
        for m in self.layers[:-1]:
            m.apply(weight_init_1d)
        self.layers[-1].apply(uniform_weight_init_1d(outscale))
        self._linear.apply(uniform_weight_init_1d(outscale))

    def forward(self, feat):
        # feat: (B, T, F) -> SymlogDist with mode (B, T, output_len)
        assert feat.dim() == 3, f"expected (B, T, F), got {tuple(feat.shape)}"
        b, t = feat.shape[:2]
        x = self._linear(feat)                                  # (B, T, 256*34)
        x = x.reshape(b * t, self._start_ch, self._start_len)   # (B*T, 256, 34)
        x = self.layers(x)                                      # (B*T, 1, 1080)
        mean = x.reshape(b, t, self.output_len)                 # (B, T, 1080)
        return tools.SymlogDist(mean, dist=self._dist, agg="sum")
