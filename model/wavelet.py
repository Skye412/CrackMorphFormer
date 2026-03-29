import torch
import torch.nn as nn
import numpy as np

def get_wav(in_channels, pool=True):
    """标准的 Haar 小波池化"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    net = nn.Conv2d if pool else nn.ConvTranspose2d

    LL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    LH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)

    for l in [LL, LH, HL, HH]: l.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

    return LL, LH, HL, HH

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

class WaveletAFDP_Fusion(nn.Module):
    """创新：方向感知小波频域门控 (Wavelet-AFDP)"""
    def __init__(self, in_channels):
        super(WaveletAFDP_Fusion, self).__init__()
        self.gate_ll = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.Sigmoid())
        self.gate_hl = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.Sigmoid())
        self.gate_lh = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.Sigmoid())
        self.gate_hh = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.Sigmoid())
        self.fusion = nn.Conv2d(in_channels * 4, in_channels, kernel_size=3, padding=1)

    def forward(self, ll, hl, lh, hh):
        ll_gated = ll * self.gate_ll(ll)
        hl_gated = hl * self.gate_hl(hl)
        lh_gated = lh * self.gate_lh(lh)
        hh_gated = hh * self.gate_hh(hh)
        return self.fusion(torch.cat([ll_gated, hl_gated, lh_gated, hh_gated], dim=1))