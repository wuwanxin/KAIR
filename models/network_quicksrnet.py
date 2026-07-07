"""
QuickSRNet — parameterized for Y-only / YUV / RGB SR training.

Variants:
    tiny:   16 channels,  1 intermediate layer,  no ITO
    small:  32 channels,  2 intermediate layers, no ITO
    medium: 32 channels,  5 intermediate layers, no ITO
    large:  64 channels, 11 intermediate layers, ITO anchor
    xlarge: 96 channels, 15 intermediate layers, ITO anchor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorOp(nn.Module):
    """信息直通 (ITO) 上采样模块。

    Conv2d(1→4, 1×1) 将每个输入像素映射到 PixelShuffle 所需的 4 个亚像素位置。
    理想情况下，训练收敛后权重全为正，偏置各不相同，以近似插值 pattern。
    """
    def __init__(self, scaling_factor, in_channels=1):
        super().__init__()
        self.net = nn.Conv2d(in_channels, in_channels * scaling_factor ** 2, 1)

    def forward(self, x):
        return self.net(x)


class QuickSRNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=11,
                 scale=2, act_mode='R'):
        """
        Args:
            in_nc:  number of input channels (1 for Y, 3 for RGB/YUV)
            out_nc: number of output channels (same as in_nc)
            nc:     number of intermediate channels (channel width)
            nb:     number of intermediate Conv layers (blocks)
            scale:  super-resolution factor (2, 3, or 4)
            act_mode: 'H' for Hardtanh (FP32训练/推理), 'R' for ReLU, 'L' for LeakyReLU
        """
        super(QuickSRNet, self).__init__()
        self.scale = scale
        self.use_ito = nc >= 64  # ITO for large/xlarge variants

        if act_mode == 'H':
            activation = nn.Hardtanh(0., 1.)
        elif act_mode == 'R':
            activation = nn.ReLU(inplace=True)
        elif act_mode == 'L':
            activation = nn.LeakyReLU(0.1, inplace=True)
        else:
            raise ValueError(f"Unsupported act_mode: {act_mode} (use 'R', 'L', or 'H')")

        # Intermediate conv layers
        layers = []
        for _ in range(nb):
            layers.extend([
                nn.Conv2d(nc, nc, 3, 1, 1),
                activation,
            ])

        self.cnn = nn.Sequential(
            nn.Conv2d(in_nc, nc, 3, 1, 1),
            activation,
            *layers,
        )

        self.conv_last = nn.Conv2d(nc, out_nc * scale ** 2, 3, 1, 1)
        self.depth_to_space = nn.PixelShuffle(scale)

        # Final output clipping — keeps values in [0,1] range
        self.clip_output = nn.Hardtanh(0., 1.)

        if self.use_ito:
            self.anchor = AnchorOp(scale, in_nc)

    def forward(self, x):
        feat = self.cnn(x)
        out = self.conv_last(feat)
        if self.use_ito:
            x_up = self.anchor(x)
            out = x_up + out
        out = self.depth_to_space(out)
        if not self.training:
            out = self.clip_output(out)
        return out
