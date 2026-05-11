# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

def get_haar_filters(in_channels):
    h = 1 / np.sqrt(2) * np.ones((1, 2))
    h_inv = 1 / np.sqrt(2) * np.ones((1, 2))
    h_inv[0, 0] = -1 * h_inv[0, 0]
    f_ll = torch.from_numpy(np.transpose(h) * h).unsqueeze(0)
    f_lh = torch.from_numpy(np.transpose(h) * h_inv).unsqueeze(0)
    f_hl = torch.from_numpy(np.transpose(h_inv) * h).unsqueeze(0)
    f_hh = torch.from_numpy(np.transpose(h_inv) * h_inv).unsqueeze(0)
    layers = []
    for f in [f_ll, f_lh, f_hl, f_hh]:
        conv = nn.Conv2d(in_channels, in_channels, 2, stride=2, groups=in_channels, bias=False)
        conv.weight.requires_grad = False
        conv.weight.data = f.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
        layers.append(conv)
    return layers

class WaveletDomainSplitter(nn.Module):
    """固定Haar小波分解 (频带拆分)"""
    def __init__(self, in_channels):
        super().__init__()
        self.filters = nn.ModuleList(get_haar_filters(in_channels))
    def forward(self, x):
        return [f(x) for f in self.filters]   # [LL, LH, HL, HH]

class SaliencyGateReconstruction(nn.Module):
    """
    频域SE (动态频带激发)
    - 对每个子带独立做全局平均池化 + 两层MLP + Sigmoid 得到通道权重
    - 加权子带后拼接，再经3x3卷积重建
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # 为4个子带各建一个SE模块
        self.se_blocks = nn.ModuleList()
        for _ in range(4):
            self.se_blocks.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),                     # [B, C, 1, 1]
                    nn.Flatten(),                                # [B, C]
                    nn.Linear(in_channels, in_channels // reduction, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_channels // reduction, in_channels, bias=False),
                    nn.Sigmoid()                                 # [B, C]
                )
            )

        self.reconstructor = nn.Conv2d(in_channels * 4, in_channels, 3, padding=1)

    def forward(self, ll, hl, lh, hh):
        """
        Args:
            ll, hl, lh, hh: 四个小波子带，每个形状 [B, C, H, W]
        Returns:
            fused: [B, C, H, W] 重建后的频域特征
        """
        components = [ll, hl, lh, hh]
        weighted = []
        for comp, se in zip(components, self.se_blocks):
            weight = se(comp).view(comp.size(0), comp.size(1), 1, 1)  # [B, C, 1, 1]
            weighted.append(comp * weight)
        return self.reconstructor(torch.cat(weighted, dim=1))