# -*- coding: utf-8 -*-
import os
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pvtv2 import pvt_v2_b2
from model import wavelet


class ContextualSignificanceWeighting(nn.Module):
    """
    Contextual Significance Weighting (CSW).

    It produces a token-wise gate by combining:
      1. local token information
      2. global mean-pooled context
    """

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
        """
        Args:
            x: [B, N, C]

        Returns:
            gate: [B, N, C]
        """
        global_context = torch.mean(x, dim=1, keepdim=True)
        return self.gate(self.local_path(x) + self.global_path(global_context))


class MorphologicalPrototypePerception(nn.Module):
    """
    Orientation-aware Morphological Prototype Perception.

    This module extends edge-guided sparse MPP with a prototype-wise
    doubled-angle orientation prior.

    Key ideas:
      1. Keep the Sobel edge magnitude bias.
      2. Assign each prototype a learnable orientation vector.
      3. Use doubled-angle encoding:
            u = ((gx^2 - gy^2) / (gx^2 + gy^2 + eps),
                 2 * gx * gy / (gx^2 + gy^2 + eps))
         This maps theta and theta + pi to the same orientation representation,
         which is suitable for unoriented thin crack structures.
    """

    def __init__(self, d_model: int, proto_size: int = 16):
        super().__init__()

        self.proto_size = proto_size

        self.conv_spatial = nn.Sequential(
            nn.Conv2d(
                d_model,
                d_model,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=d_model,
                bias=False,
            ),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
        )

        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1, -2, -1],
                 [0,   0,  0],
                 [1,   2,  1]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
        )

        self.affinity_estimator = nn.Linear(d_model, proto_size, bias=False)
        self.alignment_gate = ContextualSignificanceWeighting(d_model)
        self.norm = nn.LayerNorm(d_model)

        # Edge magnitude bias scale.
        self.topo_scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)

        # Orientation prior bias scale.
        self.orient_scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)

        # Initialize learnable prototype directions uniformly in [0, pi).
        angles = torch.linspace(0, math.pi, steps=proto_size + 1)[:-1]
        init_dirs = torch.stack(
            [torch.cos(2.0 * angles), torch.sin(2.0 * angles)],
            dim=1,
        )
        self.proto_orient = nn.Parameter(init_dirs.clone(), requires_grad=True)

    def forward(
        self,
        feat_map: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat_map: [B, C, H, W]
            query:    [B, P, C]
            key:      [B, H*W, C]

        Returns:
            refined query: [B, P, C]
        """
        b, c, h, w = feat_map.shape

        # Spatially enhanced features for affinity estimation.
        feat_conv = self.conv_spatial(feat_map)
        feat_conv = feat_conv.flatten(2).transpose(1, 2)  # [B, HW, C]

        # Sobel edge prior.
        edge_input = feat_map.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        grad_x = F.conv2d(edge_input, self.sobel_x, padding=1)
        grad_y = F.conv2d(edge_input, self.sobel_y, padding=1)

        edge_strength = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        edge_weight = edge_strength.view(b, 1, -1).transpose(1, 2)  # [B, HW, 1]
        edge_weight = (edge_weight - edge_weight.mean(dim=1, keepdim=True)) / (
            edge_weight.std(dim=1, keepdim=True) + 1e-5
        )

        # Doubled-angle orientation encoding.
        gx2 = grad_x * grad_x
        gy2 = grad_y * grad_y
        denom = gx2 + gy2 + 1e-6

        ori_x = (gx2 - gy2) / denom
        ori_y = (2.0 * grad_x * grad_y) / denom

        orient_tokens = torch.cat([ori_x, ori_y], dim=1)          # [B, 2, H, W]
        orient_tokens = orient_tokens.flatten(2).transpose(1, 2)  # [B, HW, 2]

        # Positive edge confidence for suppressing weak-gradient noise.
        edge_conf = edge_strength.flatten(2).transpose(1, 2)  # [B, HW, 1]
        edge_conf = edge_conf / (edge_conf.mean(dim=1, keepdim=True) + 1e-6)
        edge_conf = edge_conf.clamp(0.0, 3.0)

        # Prototype-wise orientation matching.
        proto_dirs = F.normalize(self.proto_orient, p=2, dim=-1)  # [P, 2]
        orient_scores = torch.einsum(
            "blc,pc->blp",
            orient_tokens,
            proto_dirs,
        )  # [B, HW, P]

        orient_bias = orient_scores * edge_conf  # [B, HW, P]

        # Affinity logits with edge bias and orientation bias.
        raw_affinity = self.affinity_estimator(feat_conv)  # [B, HW, P]

        affinity_logits = (
            raw_affinity
            + self.topo_scale * edge_weight
            + self.orient_scale * orient_bias
        )

        # Each prototype attends over spatial positions.
        affinity = F.softmax(affinity_logits, dim=1)  # [B, HW, P]

        # Prototype aggregation and query alignment.
        prototypes = affinity.transpose(-1, -2) @ key  # [B, P, C]

        attn_weights = self.alignment_gate(prototypes + query)
        out = self.norm(query * attn_weights + query)

        return out


class FrequencyStructuralAlignment(nn.Module):
    """
    Frequency Structural Alignment block.

    It contains:
      1. Wavelet domain splitting and saliency-gated reconstruction.
      2. Frequency-token cross attention.
      3. Orientation-aware MPP.
    """

    def __init__(self, d_model: int, h: int = 8, proto_size: int = 16):
        super().__init__()

        self.splitter = wavelet.WaveletDomainSplitter(d_model)
        self.reconstructor = wavelet.SaliencyGateReconstruction(d_model)

        self.freq_attn = nn.MultiheadAttention(d_model, h, batch_first=True)
        self.freq_weight = ContextualSignificanceWeighting(d_model)
        self.norm_freq = nn.LayerNorm(d_model)

        self.mpp_module = MorphologicalPrototypePerception(
            d_model=d_model,
            proto_size=proto_size,
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, P, C]
            key:   [B, N, C], where N = H * W

        Returns:
            refined query: [B, P, C]
        """
        b, n, c = key.size()
        hw = int(math.sqrt(n))

        if hw * hw != n:
            raise ValueError(
                f"FrequencyStructuralAlignment expects square feature tokens, "
                f"but got N={n}."
            )

        feat_img = key.transpose(1, 2).view(b, c, hw, hw)

        # Wavelet frequency branch.
        ll, hl, lh, hh = self.splitter(feat_img)
        fused_freq_feat = self.reconstructor(ll, hl, lh, hh)

        freq_tokens = fused_freq_feat.flatten(2).transpose(1, 2)
        freq_tokens = freq_tokens * self.freq_weight(freq_tokens) + freq_tokens

        x_freq, _ = self.freq_attn(query, freq_tokens, freq_tokens)
        x_freq = self.norm_freq(x_freq + query)

        # Orientation-aware MPP branch.
        x_mpp = self.mpp_module(feat_img, query, key)

        return x_freq + x_mpp


class CrackMorphFormer(nn.Module):
    """
    CrackMorphFormer-A.

    Final structure:
      - PVTv2-B2 backbone
      - lightweight top-down feature fusion
      - three sequential query refinement stages
      - wavelet-based frequency structural alignment
      - orientation-aware morphological prototype perception
      - query-to-mask prediction head
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
            )
            for _ in range(3)
        ])

        self.decoders = nn.ModuleList([
            FrequencyStructuralAlignment(
                d_model=channel,
                h=8,
                proto_size=num_queries,
            )
            for _ in range(3)
        ])

        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(channel, 8, batch_first=True)
            for _ in range(3)
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
        """
        Load PVTv2-B2 pretrained weights if available.

        Search order:
          1. user-provided backbone_path
          2. model/pvt_v2_b2.pth
          3. /home/skye/data/Skye/CrackMorphFormer/model/pvt_v2_b2.pth
        """
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
            print("PVTv2-B2 pretrained weights not found. Backbone is randomly initialized.")
            return

        ckpt = torch.load(ckpt_path, map_location="cpu")

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        backbone_state = self.backbone.state_dict()

        filtered = {
            k: v
            for k, v in ckpt.items()
            if k in backbone_state and v.shape == backbone_state[k].shape
        }

        self.backbone.load_state_dict(filtered, strict=False)
        print(f"Successfully loaded backbone weights from {ckpt_path}")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            outputs: list of three prediction logits.
                     Each item has shape [B, 1, H, W].
        """
        h_in, w_in = x.shape[-2:]

        feats = self.backbone(x)
        projs = [self.input_projs[i](feats[i]) for i in range(4)]

        # Top-down fusion.
        d3 = self.fusion[0](
            F.interpolate(
                projs[3],
                size=projs[2].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            + projs[2]
        )

        d2 = self.fusion[1](
            F.interpolate(
                d3,
                size=projs[1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            + projs[1]
        )

        d1 = self.fusion[2](
            F.interpolate(
                d2,
                size=projs[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            + projs[0]
        )

        bs = x.size(0)
        queries = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        # Coarse-to-fine query refinement levels.
        features = [projs[3], d3, d2]
        outputs = []

        for i in range(3):
            lvl_feat = features[i].flatten(2).transpose(1, 2)
            lvl_feat = lvl_feat + self.level_embed.weight[i]

            queries = self.decoders[i](queries, lvl_feat)
            queries = self.self_attns[i](queries, queries, queries)[0]

            q_map = queries.mean(dim=1).view(bs, -1, 1, 1)

            out_feat = d1 * q_map
            out_feat = F.interpolate(
                out_feat,
                size=(h_in, w_in),
                mode="bilinear",
                align_corners=False,
            )

            outputs.append(self.head(out_feat))

        return outputs