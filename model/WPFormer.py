import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from typing import Optional
from torch import Tensor # 修正：Tensor 从 torch 导入

from model.pvtv2 import pvt_v2_b2, pvt_v2_b4
from model.position_encoding import PositionEmbeddingSine
from model.transformer import Transformer, SelfAttentionLayer, MLP, _get_activation_fn
from model import wavelet

# ================= 基础组件 =================

class DDFusion(nn.Module):
    def __init__(self, in_channels, dct_h=8):
        super(DDFusion, self).__init__()

    def forward(self, x, y):
        bs, c, H, W = y.size()
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        out = x + y
        return out

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, stride, dilation, dilation, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.conv(x))

class MSCW(nn.Module):
    def __init__(self, d_model=64):
        super(MSCW, self).__init__()
        self.local_attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.global_attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pool = torch.mean(x, dim=1, keepdim=True)
        attn = self.local_attn(x) + self.global_attn(pool)
        return self.sigmoid(attn)

# ================= 核心：融入 AFDP 的注意力机制 =================

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.norm1 = nn.LayerNorm(d_model)
        
        # 创新：小波池化 + AFDP 融合
        self.pool = wavelet.WavePool(d_model)
        self.afdp_fusion = wavelet.WaveletAFDP_Fusion(d_model)

        self.self_attn1 = nn.MultiheadAttention(d_model, h, dropout=dropout, batch_first=True)
        self.mscw1 = MSCW(d_model=d_model)

        self.proto_size = 16
        self.conv3x3 = DSConv3x3(d_model, d_model)
        self.Mheads = nn.Linear(d_model, self.proto_size, bias=False)
        self.mscw2 = MSCW(d_model=d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, value, attn_mask=None):
        b, n1, c = key.size()
        hw = int(math.sqrt(n1))

        # 1. 频域处理：AFDP 方向门控过滤
        feat = key.transpose(1, 2).view(b, c, hw, hw)
        ll, hl, lh, hh = self.pool(feat)
        
        # 调用 wavelet.py 中的 AFDP 融合
        fused_feat = self.afdp_fusion(ll, hl, lh, hh)
        fre = fused_feat.flatten(2).transpose(1, 2)

        wei = self.mscw1(fre)
        fre = wei * fre + fre
        
        x1, _ = self.self_attn1(query=query, key=fre, value=fre)
        x1 = self.norm1(x1 + query)

        # 2. 空间特征聚合
        feat_conv = self.conv3x3(feat).flatten(2).transpose(1, 2)
        multi_heads_weights = F.softmax(self.Mheads(feat_conv), dim=1)
        protos = multi_heads_weights.transpose(-1, -2) @ key

        attn = self.mscw2(protos + query)
        x2 = self.norm2(query * attn + query)

        return x1 + x2

# ================= 主模型 =================

class WPFormer(nn.Module):
    def __init__(self, method="pvt_v2_b2", channel=64, num_queries=16):
        super(WPFormer, self).__init__()
        
        # 修正：服务器绝对路径
        if method == "pvt_v2_b2":
            self.backbone = pvt_v2_b2()
            path = '/home/skye/data/Skye/DA-WCA1/model/pvt_v2_b2.pth'
        else:
            self.backbone = pvt_v2_b4()
            path = '/home/skye/data/Skye/DA-WCA1/model/pvt_v2_b4.pth'
            
        if os.path.exists(path):
            save_model = torch.load(path, map_location='cpu')
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
            print(f"✅ Loaded backbone weights from {path}")
        else:
            print(f"⚠️ Warning: Backbone weight not found at {path}")

        self.input_projs = nn.ModuleList([
            nn.Conv2d(64, channel, 1),
            nn.Conv2d(128, channel, 1),
            nn.Conv2d(320, channel, 1),
            nn.Conv2d(512, channel, 1)
        ])

        self.fusions = nn.ModuleList([DDFusion(channel) for _ in range(3)])
        self.out_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channel, channel, 3, padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True)) 
            for _ in range(3)
        ])

        self.query_embed = nn.Embedding(num_queries, channel)
        self.pe_layer = PositionEmbeddingSine(channel // 2, normalize=True)
        self.level_embed = nn.Embedding(3, channel)
        
        self.transformer_cross_attn = nn.ModuleList([MultiheadAttention(channel, h=8) for _ in range(3)])
        self.transformer_self_attn = nn.ModuleList([nn.MultiheadAttention(channel, 8, batch_first=True) for _ in range(3)])
        
        self.mask_head = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, x):
        h_in, w_in = x.shape[-2:]
        pvt = self.backbone(x)

        projs = [self.input_projs[i](pvt[i]) for i in range(4)]
        d3 = self.out_convs[0](self.fusions[0](projs[3], projs[2]))
        d2 = self.out_convs[1](self.fusions[1](d3, projs[1]))
        d1 = self.out_convs[2](self.fusions[2](d2, projs[0]))

        bs = x.size(0)
        queries = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        features = [projs[3], d3, d2]
        preds = []

        for i in range(3):
            lvl_feat = features[i].flatten(2).transpose(1, 2) + self.level_embed.weight[i]
            queries = self.transformer_cross_attn[i](query=queries, key=lvl_feat, value=lvl_feat)
            queries = self.transformer_self_attn[i](queries, queries, queries)[0]
            
            q_feat = queries.mean(dim=1).view(bs, -1, 1, 1)
            out = F.interpolate(d1 * q_feat, size=(h_in, w_in), mode='bilinear', align_corners=False)
            preds.append(self.mask_head(out))

        return preds