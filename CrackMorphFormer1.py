# -*- coding: utf-8 -*-
"""
CrackMorphFormer with Dynamic Frequency Excitation (频域SE)
and Adaptive Structure Tensor-guided Morphological Prototype (自适应结构张量MPP).
改进：融合管状置信度与Sobel边缘强度，提升Precision和F1稳定性。
"""

import os
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.pvtv2 import pvt_v2_b2   # 请确保路径正确


# ==================== 频域模块 (wavelet.py 内容) ====================

def get_haar_filters(in_channels):
    """固定Haar小波核"""
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
    """固定Haar小波分解"""
    def __init__(self, in_channels):
        super().__init__()
        self.filters = nn.ModuleList(get_haar_filters(in_channels))
    def forward(self, x):
        return [f(x) for f in self.filters]   # [LL, LH, HL, HH]

class SaliencyGateReconstruction(nn.Module):
    """
    动态频带激发 (Dynamic Frequency Excitation)
    对每个子带独立使用SE模块，加权融合后重建
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.se_blocks = nn.ModuleList()
        for _ in range(4):
            self.se_blocks.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(in_channels, in_channels // reduction, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_channels // reduction, in_channels, bias=False),
                    nn.Sigmoid()
                )
            )
        self.reconstructor = nn.Conv2d(in_channels * 4, in_channels, 3, padding=1)

    def forward(self, ll, hl, lh, hh):
        components = [ll, hl, lh, hh]
        weighted = []
        for comp, se in zip(components, self.se_blocks):
            weight = se(comp).view(comp.size(0), comp.size(1), 1, 1)
            weighted.append(comp * weight)
        return self.reconstructor(torch.cat(weighted, dim=1))


# ==================== 上下文显著性加权 (CSW) ====================

class ContextualSignificanceWeighting(nn.Module):
    """CSW: 结合局部token和全局上下文的门控"""
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.local_path = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.global_path = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_context = torch.mean(x, dim=1, keepdim=True)
        return self.gate(self.local_path(x) + self.global_path(global_context))


# ==================== 自适应结构张量增强的 MPP ====================

class MorphologicalPrototypePerception(nn.Module):
    """
    自适应结构张量 (Adaptive Structure Tensor) 原型感知模块
    - 融合结构张量的管状置信度与经典的Sobel边缘强度，生成自适应置信度
    - 提升对假阳性的抑制能力，改善Precision
    """
    def __init__(self, d_model: int, proto_size: int = 16, sigma: float = 1.0):
        super().__init__()
        self.proto_size = proto_size
        self.sigma = sigma

        # 空间特征增强
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, groups=d_model, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # 用于结构张量平滑的高斯核
        self.register_buffer("gauss_kernel", self._get_gaussian_kernel(sigma))

        # 自适应置信度融合模块：输入 [tubularity, edge_strength]，输出融合权重
        self.confidence_fusion = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.affinity_estimator = nn.Linear(d_model, proto_size, bias=False)
        self.alignment_gate = ContextualSignificanceWeighting(d_model)
        self.norm = nn.LayerNorm(d_model)

        # 可学习缩放因子（允许负值，自适应调节）
        self.topo_scale = nn.Parameter(torch.tensor(0.05), requires_grad=True)
        self.orient_scale = nn.Parameter(torch.tensor(0.05), requires_grad=True)

        # 原型方向（倍角表示，范围 [0, π)）
        angles = torch.linspace(0, math.pi, steps=proto_size + 1)[:-1]
        init_dirs = torch.stack([torch.cos(2.0 * angles), torch.sin(2.0 * angles)], dim=1)
        self.proto_orient = nn.Parameter(init_dirs.clone(), requires_grad=True)

    def _get_gaussian_kernel(self, sigma):
        size = int(2 * sigma * 3 + 1) // 2 * 2 + 1
        if size < 3:
            size = 3
        x = torch.arange(size, dtype=torch.float32) - size // 2
        y = torch.arange(size, dtype=torch.float32) - size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def _compute_structure_tensor(self, feat_map):
        """
        输入: [B, 1, H, W]
        返回: (tubularity, v2_x, v2_y, lambda1, edge_strength)
             tubularity: [B,1,H,W] 管状置信度 (λ1-λ2)/(λ1+λ2)
             v2_x, v2_y: [B,1,H,W] 沿裂缝方向的单位向量
             lambda1: [B,1,H,W]    较大特征值
             edge_strength: [B,1,H,W] Sobel梯度幅度
        """
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=feat_map.dtype, device=feat_map.device).view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=feat_map.dtype, device=feat_map.device).view(1,1,3,3)
        gx = F.conv2d(feat_map, sobel_x, padding=1)
        gy = F.conv2d(feat_map, sobel_y, padding=1)

        edge_strength = torch.sqrt(gx**2 + gy**2 + 1e-6)   # 原始边缘强度

        gx2 = gx * gx
        gy2 = gy * gy
        gxgy = gx * gy

        kernel = self.gauss_kernel.to(feat_map.device)
        pad = kernel.shape[-1] // 2
        Jxx = F.conv2d(gx2, kernel, padding=pad)
        Jxy = F.conv2d(gxgy, kernel, padding=pad)
        Jyy = F.conv2d(gy2, kernel, padding=pad)

        trace = Jxx + Jyy
        det = Jxx * Jyy - Jxy * Jxy
        sqrt_disc = torch.sqrt(torch.clamp((trace/2)**2 - det, min=1e-6))
        lambda1 = trace/2 + sqrt_disc
        lambda2 = trace/2 - sqrt_disc

        eps = 1e-6
        denom = torch.sqrt(Jxy**2 + (lambda2 - Jxx)**2 + eps)
        v2_x = -Jxy / denom
        v2_y = (lambda2 - Jxx) / denom

        tubularity = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)   # [0,1] 管状置信度
        return tubularity, v2_x, v2_y, lambda1, edge_strength

    def forward(self, feat_map: torch.Tensor, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat_map.shape

        # 空间增强特征
        feat_conv = self.conv_spatial(feat_map)
        feat_conv_tokens = feat_conv.flatten(2).transpose(1, 2)   # [B, HW, C]

        # 计算结构张量与边缘强度
        edge_input = feat_map.mean(dim=1, keepdim=True)            # [B,1,H,W]
        tubularity, v2_x, v2_y, lambda1, edge_strength = self._compute_structure_tensor(edge_input)

        # ---- 自适应置信度融合 ----
        # 将 tubularity 和 edge_strength (归一化) 作为两个通道
        edge_norm = (edge_strength - edge_strength.min()) / (edge_strength.max() - edge_strength.min() + 1e-6)
        conf_input = torch.cat([tubularity, edge_norm], dim=1)     # [B,2,H,W]
        confidence = self.confidence_fusion(conf_input)            # [B,1,H,W] 融合后的置信度

        # 将置信度用于方向偏置
        confidence_map = confidence.flatten(2).transpose(1, 2)      # [B, HW, 1]

        # 倍角方向编码 (使用结构张量的特征向量 v2)
        ori_x = v2_x * v2_x - v2_y * v2_y
        ori_y = 2.0 * v2_x * v2_y
        orient_tokens = torch.cat([ori_x, ori_y], dim=1)           # [B,2,H,W]
        orient_tokens = orient_tokens.flatten(2).transpose(1, 2)   # [B, HW, 2]

        # 方向匹配
        proto_dirs = F.normalize(self.proto_orient, p=2, dim=-1)    # [P,2]
        orient_scores = torch.einsum("blc,pc->blp", orient_tokens, proto_dirs)  # [B,HW,P]
        orient_bias = orient_scores * confidence_map                # [B,HW,P]

        # 边缘偏置 (使用 lambda1 作为边缘强度，并标准化)
        edge_weight = lambda1.flatten(2).transpose(1, 2)            # [B, HW, 1]
        edge_weight = (edge_weight - edge_weight.mean(dim=1, keepdim=True)) / (edge_weight.std(dim=1, keepdim=True) + 1e-5)

        # 亲和度
        raw_affinity = self.affinity_estimator(feat_conv_tokens)    # [B, HW, P]
        affinity_logits = raw_affinity + self.topo_scale * edge_weight + self.orient_scale * orient_bias
        affinity = F.softmax(affinity_logits, dim=1)                # [B, HW, P]

        # 原型聚合
        prototypes = affinity.transpose(-1, -2) @ key               # [B, P, C]

        # 门控融合
        attn_weights = self.alignment_gate(prototypes + query)
        out = self.norm(query * attn_weights + query)
        return out


# ==================== 解码器块 ====================

class FrequencyStructuralAlignment(nn.Module):
    """
    频域结构对齐块 (FSA)，集成 WDS 和 MPP
    """
    def __init__(self, d_model: int, h: int = 8, proto_size: int = 16):
        super().__init__()
        self.splitter = WaveletDomainSplitter(d_model)
        self.reconstructor = SaliencyGateReconstruction(d_model)

        self.freq_attn = nn.MultiheadAttention(d_model, h, batch_first=True)
        self.freq_weight = ContextualSignificanceWeighting(d_model)
        self.norm_freq = nn.LayerNorm(d_model)

        self.mpp_module = MorphologicalPrototypePerception(
            d_model=d_model,
            proto_size=proto_size,
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        b, n, c = key.size()
        hw = int(math.sqrt(n))
        if hw * hw != n:
            raise ValueError(f"Feature tokens not square: N={n}")
        feat_img = key.transpose(1, 2).view(b, c, hw, hw)

        # ----- WDS 频域分支 -----
        ll, hl, lh, hh = self.splitter(feat_img)
        fused_freq = self.reconstructor(ll, hl, lh, hh)           # [B,C,H,W]
        freq_tokens = fused_freq.flatten(2).transpose(1, 2)       # [B, HW, C]
        freq_tokens = freq_tokens * self.freq_weight(freq_tokens) + freq_tokens
        x_freq, _ = self.freq_attn(query, freq_tokens, freq_tokens)
        x_freq = self.norm_freq(x_freq + query)

        # ----- MPP 原型分支 -----
        x_mpp = self.mpp_module(feat_img, query, key)

        return x_freq + x_mpp


# ==================== 完整模型 ====================

class CrackMorphFormer(nn.Module):
    """
    CrackMorphFormer with Dynamic Frequency Excitation and Adaptive Structure Tensor MPP
    """
    def __init__(
        self,
        channel: int = 64,
        num_queries: int = 16,
        backbone_path: Optional[str] = None,
    ):
        super().__init__()
        self.channel = channel
        self.num_queries = num_queries

        self.backbone = pvt_v2_b2()
        self._load_backbone(backbone_path)

        self.input_projs = nn.ModuleList([
            nn.Conv2d(in_c, channel, kernel_size=1)
            for in_c in [64, 128, 320, 512]
        ])

        self.fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
            ) for _ in range(3)
        ])

        self.decoders = nn.ModuleList([
            FrequencyStructuralAlignment(d_model=channel, h=8, proto_size=num_queries)
            for _ in range(3)
        ])

        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(channel, 8, batch_first=True) for _ in range(3)
        ])

        self.query_embed = nn.Embedding(num_queries, channel)
        self.level_embed = nn.Embedding(3, channel)

        self.head = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1),
        )

    def _load_backbone(self, backbone_path: Optional[str] = None):
        candidate_paths = []
        if backbone_path is not None:
            candidate_paths.append(backbone_path)
        candidate_paths.extend([
            "model/pvt_v2_b2.pth",
            "/home/skye/data/Skye/CrackMorphFormer/model/pvt_v2_b2.pth",
        ])
        ckpt_path = None
        for path in candidate_paths:
            if path is not None and os.path.exists(path):
                ckpt_path = path
                break
        if ckpt_path is None:
            print("PVTv2-B2 pretrained weights not found. Backbone randomly initialized.")
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        backbone_state = self.backbone.state_dict()
        filtered = {k: v for k, v in ckpt.items()
                    if k in backbone_state and v.shape == backbone_state[k].shape}
        self.backbone.load_state_dict(filtered, strict=False)
        print(f"Loaded backbone weights from {ckpt_path}")

    def forward(self, x: torch.Tensor):
        h_in, w_in = x.shape[-2:]

        feats = self.backbone(x)
        projs = [self.input_projs[i](feats[i]) for i in range(4)]

        # Top-down fusion
        d3 = self.fusion[0](
            F.interpolate(projs[3], size=projs[2].shape[-2:], mode="bilinear", align_corners=False)
            + projs[2]
        )
        d2 = self.fusion[1](
            F.interpolate(d3, size=projs[1].shape[-2:], mode="bilinear", align_corners=False)
            + projs[1]
        )
        d1 = self.fusion[2](
            F.interpolate(d2, size=projs[0].shape[-2:], mode="bilinear", align_corners=False)
            + projs[0]
        )

        bs = x.size(0)
        queries = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        features = [projs[3], d3, d2]
        outputs = []

        for i in range(3):
            lvl_feat = features[i].flatten(2).transpose(1, 2)
            lvl_feat = lvl_feat + self.level_embed.weight[i]

            queries = self.decoders[i](queries, lvl_feat)
            queries = self.self_attns[i](queries, queries, queries)[0]

            q_map = queries.mean(dim=1).view(bs, -1, 1, 1)

            out_feat = d1 * q_map
            out_feat = F.interpolate(out_feat, size=(h_in, w_in), mode="bilinear", align_corners=False)
            outputs.append(self.head(out_feat))

        return outputs