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
    """小波域物理拆解模块 - 对应 Proposed Method 中的频域分解部分"""
    def __init__(self, in_channels):
        super().__init__()
        self.filters = nn.ModuleList(get_haar_filters(in_channels))
    def forward(self, x):
        return [f(x) for f in self.filters]

class SaliencyGateReconstruction(nn.Module):
    """频域显著性门控重构模块 - 对应 Proposed Method 中的 AFDP 逻辑"""
    def __init__(self, in_channels):
        super().__init__()
        self.sal_eval = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.Sigmoid()) for _ in range(4)
        ])
        self.reconstructor = nn.Conv2d(in_channels * 4, in_channels, 3, padding=1)
    def forward(self, ll, hl, lh, hh):
        components = [ll, hl, lh, hh]
        gated = [c * self.sal_eval[i](c) for i, c in enumerate(components)]
        return self.reconstructor(torch.cat(gated, dim=1))