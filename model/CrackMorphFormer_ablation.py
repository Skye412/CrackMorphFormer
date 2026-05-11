# -*- coding: utf-8 -*-
import os
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.pvtv2 import pvt_v2_b2


# ==================== 频域模块 ====================
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
    def __init__(self, in_channels):
        super().__init__()
        self.filters = nn.ModuleList(get_haar_filters(in_channels))
    def forward(self, x):
        return [f(x) for f in self.filters]

class SaliencyGateReconstruction(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
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
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.local_path = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model)
        )
        self.global_path = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model)
        )
        self.gate = nn.Sigmoid()
    def forward(self, x):
        global_context = torch.mean(x, dim=1, keepdim=True)
        return self.gate(self.local_path(x) + self.global_path(global_context))

# ==================== 自适应结构张量原型 (ASTP) ====================
class MorphologicalPrototypePerception(nn.Module):
    def __init__(self, d_model: int, proto_size: int = 16, sigma: float = 1.0):
        super().__init__()
        self.proto_size = proto_size
        self.sigma = sigma
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, 1, 1, groups=d_model, bias=False),
            nn.BatchNorm2d(d_model), nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model), nn.ReLU(inplace=True),
        )
        self.register_buffer("gauss_kernel", self._get_gaussian_kernel(sigma))
        self.confidence_fusion = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False), nn.Sigmoid()
        )
        self.affinity_estimator = nn.Linear(d_model, proto_size, bias=False)
        self.alignment_gate = ContextualSignificanceWeighting(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.topo_scale = nn.Parameter(torch.tensor(0.05))
        self.orient_scale = nn.Parameter(torch.tensor(0.05))
        angles = torch.linspace(0, math.pi, steps=proto_size + 1)[:-1]
        init_dirs = torch.stack([torch.cos(2.0 * angles), torch.sin(2.0 * angles)], dim=1)
        self.proto_orient = nn.Parameter(init_dirs.clone())

    def _get_gaussian_kernel(self, sigma):
        size = int(2 * sigma * 3 + 1) // 2 * 2 + 1
        if size < 3: size = 3
        x = torch.arange(size, dtype=torch.float32) - size // 2
        y = torch.arange(size, dtype=torch.float32) - size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def _compute_structure_tensor(self, feat_map):
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=feat_map.dtype, device=feat_map.device).view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=feat_map.dtype, device=feat_map.device).view(1,1,3,3)
        gx = F.conv2d(feat_map, sobel_x, padding=1)
        gy = F.conv2d(feat_map, sobel_y, padding=1)
        edge_strength = torch.sqrt(gx**2 + gy**2 + 1e-6)
        gx2, gy2, gxgy = gx*gx, gy*gy, gx*gy
        kernel = self.gauss_kernel.to(feat_map.device)
        pad = kernel.shape[-1] // 2
        Jxx = F.conv2d(gx2, kernel, padding=pad)
        Jxy = F.conv2d(gxgy, kernel, padding=pad)
        Jyy = F.conv2d(gy2, kernel, padding=pad)
        trace = Jxx + Jyy
        det = Jxx*Jyy - Jxy*Jxy
        sqrt_disc = torch.sqrt(torch.clamp((trace/2)**2 - det, min=1e-6))
        lambda1 = trace/2 + sqrt_disc
        lambda2 = trace/2 - sqrt_disc
        denom = torch.sqrt(Jxy**2 + (lambda2 - Jxx)**2 + 1e-6)
        v2_x = -Jxy / denom
        v2_y = (lambda2 - Jxx) / denom
        tubularity = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)
        return tubularity, v2_x, v2_y, lambda1, edge_strength

    def forward(self, feat_map, query, key):
        b, c, h, w = feat_map.shape
        feat_conv = self.conv_spatial(feat_map)
        feat_conv_tokens = feat_conv.flatten(2).transpose(1, 2)
        edge_input = feat_map.mean(dim=1, keepdim=True)
        tubularity, v2_x, v2_y, lambda1, edge_strength = self._compute_structure_tensor(edge_input)

        edge_norm = (edge_strength - edge_strength.min()) / (edge_strength.max() - edge_strength.min() + 1e-6)
        conf_input = torch.cat([tubularity, edge_norm], dim=1)
        confidence = self.confidence_fusion(conf_input)
        confidence_map = confidence.flatten(2).transpose(1, 2)

        ori_x = v2_x*v2_x - v2_y*v2_y
        ori_y = 2.0 * v2_x * v2_y
        orient_tokens = torch.cat([ori_x, ori_y], dim=1).flatten(2).transpose(1, 2)
        proto_dirs = F.normalize(self.proto_orient, p=2, dim=-1)
        orient_scores = torch.einsum("blc,pc->blp", orient_tokens, proto_dirs)
        orient_bias = orient_scores * confidence_map

        edge_weight = lambda1.flatten(2).transpose(1, 2)
        edge_weight = (edge_weight - edge_weight.mean(dim=1, keepdim=True)) / (edge_weight.std(dim=1, keepdim=True) + 1e-5)
        raw_affinity = self.affinity_estimator(feat_conv_tokens)
        affinity_logits = raw_affinity + self.topo_scale * edge_weight + self.orient_scale * orient_bias
        affinity = F.softmax(affinity_logits, dim=1)
        prototypes = affinity.transpose(-1, -2) @ key
        attn_weights = self.alignment_gate(prototypes + query)
        out = self.norm(query * attn_weights + query)
        return out

# ==================== 解码器块（支持模块开关）====================
class FrequencyStructuralAlignment(nn.Module):
    def __init__(self, d_model: int, h: int = 8, proto_size: int = 16,
                 use_dfe: bool = True, use_astp: bool = True):
        super().__init__()
        self.use_dfe = use_dfe
        self.use_astp = use_astp

        if self.use_dfe:
            self.splitter = WaveletDomainSplitter(d_model)
            self.reconstructor = SaliencyGateReconstruction(d_model)
            self.freq_attn = nn.MultiheadAttention(d_model, h, batch_first=True)
            self.freq_weight = ContextualSignificanceWeighting(d_model)
            self.norm_freq = nn.LayerNorm(d_model)
        if self.use_astp:
            self.mpp_module = MorphologicalPrototypePerception(d_model, proto_size)

    def forward(self, query, key):
        b, n, c = key.size()
        hw = int(math.sqrt(n))
        if hw * hw != n:
            raise ValueError(f"Feature tokens not square: N={n}")
        feat_img = key.transpose(1, 2).view(b, c, hw, hw)

        x_freq = query
        if self.use_dfe:
            ll, hl, lh, hh = self.splitter(feat_img)
            fused_freq = self.reconstructor(ll, hl, lh, hh)
            freq_tokens = fused_freq.flatten(2).transpose(1, 2)
            freq_tokens = freq_tokens * self.freq_weight(freq_tokens) + freq_tokens
            x_freq, _ = self.freq_attn(query, freq_tokens, freq_tokens)
            x_freq = self.norm_freq(x_freq + query)

        x_mpp = x_freq
        if self.use_astp:
            x_mpp = self.mpp_module(feat_img, x_freq, key)
        return x_mpp

# ==================== 完整模型 ====================
class CrackMorphFormer(nn.Module):
    def __init__(
        self,
        channel: int = 64,
        num_queries: int = 16,
        backbone_path: Optional[str] = None,
        use_dfe: bool = True,
        use_astp: bool = True,
    ):
        super().__init__()
        self.channel = channel
        self.num_queries = num_queries
        self.use_dfe = use_dfe
        self.use_astp = use_astp

        self.backbone = pvt_v2_b2()
        self._load_backbone(backbone_path)

        self.input_projs = nn.ModuleList([
            nn.Conv2d(in_c, channel, 1) for in_c in [64, 128, 320, 512]
        ])

        self.fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, channel, 3, padding=1),
                nn.BatchNorm2d(channel), nn.ReLU(inplace=True)
            ) for _ in range(3)
        ])

        self.decoders = nn.ModuleList([
            FrequencyStructuralAlignment(d_model=channel, h=8, proto_size=num_queries,
                                         use_dfe=use_dfe, use_astp=use_astp)
            for _ in range(3)
        ])

        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(channel, 8, batch_first=True) for _ in range(3)
        ])

        self.query_embed = nn.Embedding(num_queries, channel)
        self.level_embed = nn.Embedding(3, channel)

        self.head = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 1),
        )

    def _load_backbone(self, backbone_path):
        candidate_paths = []
        if backbone_path is not None:
            candidate_paths.append(backbone_path)
        candidate_paths.extend(["model/pvt_v2_b2.pth", "/home/skye/data/Skye/CrackMorphFormer/model/pvt_v2_b2.pth"])
        ckpt_path = None
        for path in candidate_paths:
            if path and os.path.exists(path):
                ckpt_path = path
                break
        if ckpt_path is None:
            print("PVTv2-B2 pretrained weights not found. Random init.")
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        backbone_state = self.backbone.state_dict()
        filtered = {k: v for k, v in ckpt.items() if k in backbone_state and v.shape == backbone_state[k].shape}
        self.backbone.load_state_dict(filtered, strict=False)
        print(f"Loaded backbone from {ckpt_path}")

    def forward(self, x):
        h_in, w_in = x.shape[-2:]
        feats = self.backbone(x)
        projs = [self.input_projs[i](feats[i]) for i in range(4)]

        d3 = self.fusion[0](F.interpolate(projs[3], size=projs[2].shape[-2:], mode='bilinear', align_corners=False) + projs[2])
        d2 = self.fusion[1](F.interpolate(d3, size=projs[1].shape[-2:], mode='bilinear', align_corners=False) + projs[1])
        d1 = self.fusion[2](F.interpolate(d2, size=projs[0].shape[-2:], mode='bilinear', align_corners=False) + projs[0])

        bs = x.size(0)
        queries = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        features = [projs[3], d3, d2]
        outputs = []

        for i in range(3):
            lvl_feat = features[i].flatten(2).transpose(1, 2) + self.level_embed.weight[i]
            queries = self.decoders[i](queries, lvl_feat)
            queries = self.self_attns[i](queries, queries, queries)[0]
            q_map = queries.mean(dim=1).view(bs, -1, 1, 1)
            out_feat = d1 * q_map
            out_feat = F.interpolate(out_feat, size=(h_in, w_in), mode='bilinear', align_corners=False)
            outputs.append(self.head(out_feat))
        return outputs