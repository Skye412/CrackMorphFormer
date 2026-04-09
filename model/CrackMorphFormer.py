# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from model.pvtv2 import pvt_v2_b2
from model import wavelet

class ContextualSignificanceWeighting(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.local_path = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.global_path = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.gate = nn.Sigmoid()
    def forward(self, x):
        pool = torch.mean(x, dim=1, keepdim=True)
        return self.gate(self.local_path(x) + self.global_path(pool))

class MorphologicalPrototypePerception(nn.Module):
    def __init__(self, d_model, proto_size=16):
        super().__init__()
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, 1, 1, 1, groups=d_model, bias=False),
            nn.BatchNorm2d(d_model), nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model), nn.ReLU(inplace=True)
        )
        self.affinity_estimator = nn.Linear(d_model, proto_size, bias=False)
        self.alignment_gate = ContextualSignificanceWeighting(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feat_map, query, key):
        feat_conv = self.conv_spatial(feat_map).flatten(2).transpose(1, 2)
        affinity = F.softmax(self.affinity_estimator(feat_conv), dim=1)
        prototypes = affinity.transpose(-1, -2) @ key
        attn_weights = self.alignment_gate(prototypes + query)
        return self.norm(query * attn_weights + query)

class FrequencyStructuralAlignment(nn.Module):
    def __init__(self, d_model, h=8):
        super().__init__()
        # 修正：使用 model/wavelet.py 中的新类名
        self.splitter = wavelet.WaveletDomainSplitter(d_model)
        self.reconstructor = wavelet.SaliencyGateReconstruction(d_model)
        
        self.freq_attn = nn.MultiheadAttention(d_model, h, batch_first=True)
        self.freq_weight = ContextualSignificanceWeighting(d_model)
        self.norm_freq = nn.LayerNorm(d_model)
        self.mpp_module = MorphologicalPrototypePerception(d_model)

    def forward(self, query, key):
        b, n, c = key.size(); hw = int(math.sqrt(n))
        feat_img = key.transpose(1, 2).view(b, c, hw, hw)
        
        # 1. 频域路径 (Wavelet-AFDP)
        ll, hl, lh, hh = self.splitter(feat_img)
        fused_freq_feat = self.reconstructor(ll, hl, lh, hh)
        freq_tokens = fused_freq_feat.flatten(2).transpose(1, 2)
        f_wei = self.freq_weight(freq_tokens)
        freq_tokens = freq_tokens * f_wei + freq_tokens
        x_freq, _ = self.freq_attn(query, freq_tokens, freq_tokens)
        x1 = self.norm_freq(x_freq + query)

        # 2. 空间路径 (MPP)
        x2 = self.mpp_module(feat_img, query, key)
        return x1 + x2

class CrackMorphFormer(nn.Module):
    def __init__(self, channel=64, num_queries=16):
        super().__init__()
        self.backbone = pvt_v2_b2()
        path = '/home/skye/data/Skye/CrackMorphFormer/model/pvt_v2_b2.pth'
        if os.path.exists(path):
            save_model = torch.load(path, map_location='cpu')
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
            print(f"✅ Successfully loaded backbone weights from {path}")

        self.input_projs = nn.ModuleList([nn.Conv2d(in_c, channel, 1) for in_c in [64, 128, 320, 512]])
        self.fusion = nn.ModuleList([nn.Sequential(nn.Conv2d(channel, channel, 3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True)) for _ in range(3)])
        self.decoders = nn.ModuleList([FrequencyStructuralAlignment(channel, h=8) for _ in range(3)])
        self.self_attns = nn.ModuleList([nn.MultiheadAttention(channel, 8, batch_first=True) for _ in range(3)])
        self.query_embed = nn.Embedding(num_queries, channel)
        self.level_embed = nn.Embedding(3, channel)
        self.head = nn.Sequential(nn.Conv2d(channel, channel, 3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True), nn.Conv2d(channel, 1, 1))

    def forward(self, x):
        h_in, w_in = x.shape[-2:]; feats = self.backbone(x)
        projs = [self.input_projs[i](feats[i]) for i in range(4)]
        
        d3 = self.fusion[0](F.interpolate(projs[3], size=projs[2].shape[-2:], mode='bilinear') + projs[2])
        d2 = self.fusion[1](F.interpolate(d3, size=projs[1].shape[-2:], mode='bilinear') + projs[1])
        d1 = self.fusion[2](F.interpolate(d2, size=projs[0].shape[-2:], mode='bilinear') + projs[0])
        
        bs = x.size(0); queries = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        features = [projs[3], d3, d2]; outputs = []
        for i in range(3):
            lvl_feat = features[i].flatten(2).transpose(1, 2) + self.level_embed.weight[i]
            queries = self.decoders[i](queries, lvl_feat)
            queries = self.self_attns[i](queries, queries, queries)[0]
            q_map = queries.mean(dim=1).view(bs, -1, 1, 1)
            out_feat = F.interpolate(d1 * q_map, size=(h_in, w_in), mode='bilinear')
            outputs.append(self.head(out_feat))
        return outputs