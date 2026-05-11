# -*- coding: utf-8 -*-
"""
CrackMorphFormer.

Paper-level modules:
1. Dynamic Frequency Excitation (DFE)
   - includes Haar Frequency Decomposition (HFD)
   - enhances crack-related frequency details

2. Structure-Guided Morphological Prototype Perception (SG-MPP)
   - uses structure tensor cues: tubularity, orientation, edge confidence
   - performs morphology-aware prototype aggregation and query calibration

Stable computation path:
PVT-v2 backbone -> top-down fusion -> query refinement -> query-guided high-resolution prediction.

Important implementation changes in this stable version:
1. DFE and SG-MPP are fused as residual deltas instead of summing two full residual queries.
2. Query self-attention uses residual + LayerNorm.
3. Final query-guided modulation uses a bounded channel gate initialized near identity.
"""

import os
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pvtv2 import pvt_v2_b2


# ============================================================
# Haar filters
# ============================================================

def get_haar_filters(in_channels: int):
    """
    Build fixed depth-wise Haar filters.

    Return order:
        LL, LH, HL, HH
    """
    h = 1.0 / np.sqrt(2.0) * np.ones((1, 2))
    h_inv = 1.0 / np.sqrt(2.0) * np.ones((1, 2))
    h_inv[0, 0] = -1.0 * h_inv[0, 0]

    f_ll = torch.from_numpy(np.transpose(h) * h).unsqueeze(0)
    f_lh = torch.from_numpy(np.transpose(h) * h_inv).unsqueeze(0)
    f_hl = torch.from_numpy(np.transpose(h_inv) * h).unsqueeze(0)
    f_hh = torch.from_numpy(np.transpose(h_inv) * h_inv).unsqueeze(0)

    layers = []

    for f in [f_ll, f_lh, f_hl, f_hh]:
        conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2,
            groups=in_channels,
            bias=False,
        )

        conv.weight.requires_grad = False
        conv.weight.data = (
            f.float()
            .unsqueeze(0)
            .expand(in_channels, -1, -1, -1)
            .clone()
        )

        layers.append(conv)

    return layers


# ============================================================
# DFE: Dynamic Frequency Excitation
# ============================================================

class HaarFrequencyDecomposition(nn.Module):
    """
    Haar Frequency Decomposition (HFD).

    Internal step of DFE.
    It decomposes the input feature into four Haar sub-bands:
        LL, LH, HL, HH
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.filters = nn.ModuleList(get_haar_filters(in_channels))

    def forward(self, x: torch.Tensor):
        return [f(x) for f in self.filters]


class DynamicFrequencyExcitation(nn.Module):
    """
    Dynamic Frequency Excitation (DFE).

    Main idea:
    - Decompose feature into LL/LH/HL/HH using fixed Haar filters.
    - Apply independent channel excitation to each sub-band.
    - Concatenate weighted sub-bands.
    - Reconstruct a frequency-enhanced feature with 3x3 convolution.

    Output resolution is half of input resolution.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        hidden = max(in_channels // reduction, 4)

        self.hfd = HaarFrequencyDecomposition(in_channels)

        self.subband_exciters = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, in_channels, bias=False),
                nn.Sigmoid(),
            )
            for _ in range(4)
        ])

        self.reconstruction = nn.Sequential(
            nn.Conv2d(
                in_channels * 4,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C, H, W]

        Returns:
            frequency-enhanced feature: [B, C, H/2, W/2]
        """
        ll, lh, hl, hh = self.hfd(feat)
        subbands = [ll, lh, hl, hh]

        weighted_subbands = []

        for subband, exciter in zip(subbands, self.subband_exciters):
            weight = exciter(subband).view(
                subband.size(0),
                subband.size(1),
                1,
                1,
            )
            weighted_subbands.append(subband * weight)

        out = torch.cat(weighted_subbands, dim=1)
        out = self.reconstruction(out)

        return out


# ============================================================
# Internal calibration gate
# ============================================================

class ContextAwareCalibrationGate(nn.Module):
    """
    Context-aware calibration gate.

    Internal helper, not a paper-level module.

    It combines:
    - local token representation
    - global mean context

    Output:
        token-wise channel gate in [0, 1]
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
        gate = self.local_path(x) + self.global_path(global_context)
        return self.gate(gate)


# ============================================================
# SG-MPP: Structure-Guided Morphological Prototype Perception
# ============================================================

class StructureGuidedMorphologicalPrototypePerception(nn.Module):
    """
    Structure-Guided Morphological Prototype Perception (SG-MPP).

    Main idea:
    - Use structure tensor cues to estimate crack morphology.
    - Generate tubularity, edge confidence, and orientation cues.
    - Build topological and orientation biases for prototype affinity.
    - Aggregate morphology-aware prototypes.
    - Calibrate query tokens with morphological prototypes.
    """

    def __init__(
        self,
        d_model: int,
        proto_size: int = 16,
        sigma: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.proto_size = proto_size
        self.sigma = sigma
        self.eps = eps

        self.spatial_enhancement = nn.Sequential(
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
            nn.Conv2d(
                d_model,
                d_model,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        self.register_buffer(
            "gaussian_kernel",
            self._build_gaussian_kernel(sigma),
            persistent=False,
        )

        sobel_x = torch.tensor(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        self.confidence_fusion = nn.Sequential(
            nn.Conv2d(
                2,
                1,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

        self.affinity_estimator = nn.Linear(
            d_model,
            proto_size,
            bias=False,
        )

        self.query_calibration = ContextAwareCalibrationGate(d_model)
        self.norm = nn.LayerNorm(d_model)

        self.topological_scale = nn.Parameter(
            torch.tensor(0.05),
            requires_grad=True,
        )

        self.orientation_scale = nn.Parameter(
            torch.tensor(0.05),
            requires_grad=True,
        )

        angles = torch.linspace(0, math.pi, steps=proto_size + 1)[:-1]

        init_orientations = torch.stack(
            [
                torch.cos(2.0 * angles),
                torch.sin(2.0 * angles),
            ],
            dim=1,
        )

        self.prototype_orientations = nn.Parameter(
            init_orientations.clone(),
            requires_grad=True,
        )

    @staticmethod
    def _build_gaussian_kernel(sigma: float) -> torch.Tensor:
        size = int(2 * sigma * 3 + 1) // 2 * 2 + 1
        size = max(size, 3)

        x = torch.arange(size, dtype=torch.float32) - size // 2
        y = torch.arange(size, dtype=torch.float32) - size // 2

        xx, yy = torch.meshgrid(x, y, indexing="ij")

        kernel = torch.exp(
            -(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2)
        )

        kernel = kernel / kernel.sum()

        return kernel.view(1, 1, size, size)

    @staticmethod
    def _minmax_normalize_per_sample(
        x: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W]

        Returns:
            normalized x in [0, 1]
        """
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    @staticmethod
    def _standardize_tokens(
        x: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, HW, 1]

        Returns:
            standardized x
        """
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + eps)

    def _compute_structure_tensor(
        self,
        feat_gray: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_gray: [B, 1, H, W]

        Returns:
            tubularity:     [B, 1, H, W]
            dir_x:          [B, 1, H, W]
            dir_y:          [B, 1, H, W]
            lambda1:        [B, 1, H, W]
            edge_strength:  [B, 1, H, W]
        """
        dtype = feat_gray.dtype
        device = feat_gray.device

        sobel_x = self.sobel_x.to(device=device, dtype=dtype)
        sobel_y = self.sobel_y.to(device=device, dtype=dtype)

        gx = F.conv2d(feat_gray, sobel_x, padding=1)
        gy = F.conv2d(feat_gray, sobel_y, padding=1)

        edge_strength = torch.sqrt(gx ** 2 + gy ** 2 + self.eps)

        gx2 = gx * gx
        gy2 = gy * gy
        gxgy = gx * gy

        kernel = self.gaussian_kernel.to(device=device, dtype=dtype)
        pad = kernel.shape[-1] // 2

        jxx = F.conv2d(gx2, kernel, padding=pad)
        jxy = F.conv2d(gxgy, kernel, padding=pad)
        jyy = F.conv2d(gy2, kernel, padding=pad)

        trace = jxx + jyy

        delta = ((jxx - jyy) * 0.5) ** 2 + jxy ** 2
        sqrt_delta = torch.sqrt(torch.clamp(delta, min=self.eps))

        lambda1 = trace * 0.5 + sqrt_delta
        lambda2 = trace * 0.5 - sqrt_delta

        denom = torch.sqrt(
            jxy ** 2 + (lambda2 - jxx) ** 2 + self.eps
        )

        dir_x = -jxy / denom
        dir_y = (lambda2 - jxx) / denom

        tubularity = (lambda1 - lambda2) / (lambda1 + lambda2 + self.eps)
        tubularity = torch.clamp(tubularity, min=0.0, max=1.0)

        return tubularity, dir_x, dir_y, lambda1, edge_strength

    def forward(
        self,
        feat_map: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat_map: [B, C, H, W]
            query:    [B, Q, C]
            key:      [B, HW, C]

        Returns:
            refined query: [B, Q, C]
        """
        enhanced_feat = self.spatial_enhancement(feat_map)
        enhanced_tokens = enhanced_feat.flatten(2).transpose(1, 2)

        feat_gray = feat_map.mean(dim=1, keepdim=True)

        (
            tubularity,
            dir_x,
            dir_y,
            lambda1,
            edge_strength,
        ) = self._compute_structure_tensor(feat_gray)

        edge_confidence = self._minmax_normalize_per_sample(edge_strength)

        confidence_input = torch.cat(
            [
                tubularity,
                edge_confidence,
            ],
            dim=1,
        )

        morphology_confidence = self.confidence_fusion(confidence_input)
        confidence_tokens = morphology_confidence.flatten(2).transpose(1, 2)

        orient_x = dir_x * dir_x - dir_y * dir_y
        orient_y = 2.0 * dir_x * dir_y

        orientation_tokens = torch.cat(
            [
                orient_x,
                orient_y,
            ],
            dim=1,
        )

        orientation_tokens = orientation_tokens.flatten(2).transpose(1, 2)

        prototype_dirs = F.normalize(
            self.prototype_orientations,
            p=2,
            dim=-1,
        )

        orientation_score = torch.einsum(
            "bnc,pc->bnp",
            orientation_tokens,
            prototype_dirs,
        )

        orientation_bias = orientation_score * confidence_tokens

        topological_bias = lambda1.flatten(2).transpose(1, 2)
        topological_bias = self._standardize_tokens(topological_bias)

        raw_affinity = self.affinity_estimator(enhanced_tokens)

        affinity_logits = (
            raw_affinity
            + self.topological_scale * topological_bias
            + self.orientation_scale * orientation_bias
        )

        affinity = F.softmax(affinity_logits, dim=1)

        prototypes = affinity.transpose(-1, -2) @ key

        gate = self.query_calibration(prototypes + query)

        out = self.norm(
            query + gate * prototypes
        )

        return out


# ============================================================
# Query refinement block
# ============================================================

class CrackMorphQueryRefinementBlock(nn.Module):
    """
    CrackMorph query refinement block.

    Stable implementation:
    - DFE branch and SG-MPP branch are converted into residual deltas.
    - The two deltas are fused with learnable branch scales.
    - This avoids double-counting the original query when both branches are enabled.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        proto_size: int = 16,
        use_dfe: bool = True,
        use_sgmpp: bool = True,
    ):
        super().__init__()

        self.use_dfe = use_dfe
        self.use_sgmpp = use_sgmpp

        if self.use_dfe:
            self.dfe = DynamicFrequencyExcitation(d_model)

            self.frequency_token_calibration = ContextAwareCalibrationGate(d_model)

            self.frequency_cross_attention = nn.MultiheadAttention(
                d_model,
                num_heads,
                batch_first=True,
            )

            self.frequency_norm = nn.LayerNorm(d_model)

            # DFE tends to enhance weak high-frequency responses.
            # A conservative initial scale reduces early false positives.
            self.dfe_scale = nn.Parameter(torch.tensor(-2.0), requires_grad=True)

        if self.use_sgmpp:
            self.sgmpp = StructureGuidedMorphologicalPrototypePerception(
                d_model=d_model,
                proto_size=proto_size,
            )

            # SG-MPP is the more discriminative branch in current ablations.
            self.sgmpp_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=True)

        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, Q, C]
            key:   [B, N, C]

        Returns:
            refined query: [B, Q, C]
        """
        b, n, c = key.size()

        hw = int(math.sqrt(n))

        if hw * hw != n:
            raise ValueError(f"Feature tokens are not square: N={n}")

        feat_img = (
            key.transpose(1, 2)
            .contiguous()
            .view(b, c, hw, hw)
        )

        if (not self.use_dfe) and (not self.use_sgmpp):
            return query

        out = query

        if self.use_dfe:
            freq_feat = self.dfe(feat_img)
            freq_tokens = freq_feat.flatten(2).transpose(1, 2)

            freq_tokens = (
                freq_tokens * self.frequency_token_calibration(freq_tokens)
                + freq_tokens
            )

            freq_delta, _ = self.frequency_cross_attention(
                query,
                freq_tokens,
                freq_tokens,
            )

            freq_query = self.frequency_norm(query + freq_delta)
            freq_delta = freq_query - query

            alpha_dfe = torch.sigmoid(self.dfe_scale)
            out = out + alpha_dfe * freq_delta

        if self.use_sgmpp:
            morph_query = self.sgmpp(
                feat_map=feat_img,
                query=query,
                key=key,
            )

            morph_delta = morph_query - query

            alpha_sgmpp = torch.sigmoid(self.sgmpp_scale)
            out = out + alpha_sgmpp * morph_delta

        out = self.fusion_norm(out)

        return out


# ============================================================
# CrackMorphFormer
# ============================================================

class CrackMorphFormer(nn.Module):
    """
    CrackMorphFormer.

    Backbone:
        PVT-v2-B2

    Paper-level modules:
        1. Dynamic Frequency Excitation (DFE)
        2. Structure-Guided Morphological Prototype Perception (SG-MPP)

    Prediction:
        query-guided high-resolution feature modulation.
    """

    def __init__(
        self,
        channel: int = 64,
        num_queries: int = 16,
        backbone_path: Optional[str] = None,
        use_dfe: bool = True,
        use_sgmpp: bool = True,
    ):
        super().__init__()

        self.channel = channel
        self.num_queries = num_queries
        self.use_dfe = use_dfe
        self.use_sgmpp = use_sgmpp

        self.backbone = pvt_v2_b2()
        self._load_backbone(backbone_path)

        self.input_projs = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                channel,
                kernel_size=1,
            )
            for in_channels in [64, 128, 320, 512]
        ])

        self.topdown_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    channel,
                    channel,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
            )
            for _ in range(3)
        ])

        self.query_refiners = nn.ModuleList([
            CrackMorphQueryRefinementBlock(
                d_model=channel,
                num_heads=8,
                proto_size=num_queries,
                use_dfe=use_dfe,
                use_sgmpp=use_sgmpp,
            )
            for _ in range(3)
        ])

        self.query_self_attns = nn.ModuleList([
            nn.MultiheadAttention(
                channel,
                8,
                batch_first=True,
            )
            for _ in range(3)
        ])

        self.query_self_norms = nn.ModuleList([
            nn.LayerNorm(channel)
            for _ in range(3)
        ])

        self.query_embed = nn.Embedding(
            num_queries,
            channel,
        )

        self.level_embed = nn.Embedding(
            3,
            channel,
        )

        self.query_gate_proj = nn.Linear(channel, channel)

        # Initialize query-guided modulation as identity-like:
        # 2 * sigmoid(0) = 1, so pred_feat initially equals d1.
        nn.init.zeros_(self.query_gate_proj.weight)
        nn.init.zeros_(self.query_gate_proj.bias)

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(
                channel,
                channel,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channel,
                1,
                kernel_size=1,
            ),
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

        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
        )

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        backbone_state = self.backbone.state_dict()

        filtered = {
            k: v
            for k, v in ckpt.items()
            if k in backbone_state and v.shape == backbone_state[k].shape
        }

        self.backbone.load_state_dict(
            filtered,
            strict=False,
        )

        print(f"Loaded backbone weights from {ckpt_path}")

    def forward(self, x: torch.Tensor):
        h_in, w_in = x.shape[-2:]

        feats = self.backbone(x)

        projected_feats = [
            self.input_projs[i](feats[i])
            for i in range(4)
        ]

        # ----------------------------------------------------
        # Top-down feature fusion
        # D3 = Up(F4) + F3
        # D2 = Up(D3) + F2
        # D1 = Up(D2) + F1
        # ----------------------------------------------------

        d3 = self.topdown_fusion[0](
            F.interpolate(
                projected_feats[3],
                size=projected_feats[2].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            + projected_feats[2]
        )

        d2 = self.topdown_fusion[1](
            F.interpolate(
                d3,
                size=projected_feats[1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            + projected_feats[1]
        )

        d1 = self.topdown_fusion[2](
            F.interpolate(
                d2,
                size=projected_feats[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            + projected_feats[0]
        )

        batch_size = x.size(0)

        queries = (
            self.query_embed.weight
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        # Coarse-to-fine query refinement.
        # Stable setting:
        #   stage 1: projected F4
        #   stage 2: D3
        #   stage 3: D2
        refinement_features = [
            projected_feats[3],
            d3,
            d2,
        ]

        outputs = []

        for i in range(3):
            level_tokens = (
                refinement_features[i]
                .flatten(2)
                .transpose(1, 2)
            )

            level_tokens = (
                level_tokens
                + self.level_embed.weight[i].view(1, 1, -1)
            )

            queries = self.query_refiners[i](
                query=queries,
                key=level_tokens,
            )

            self_attn_out = self.query_self_attns[i](
                queries,
                queries,
                queries,
            )[0]

            queries = self.query_self_norms[i](
                queries + self_attn_out
            )

            # Query-guided high-resolution modulation.
            # This is a bounded channel gate in [0, 2], initialized around 1.
            query_gate = self.query_gate_proj(
                queries.mean(dim=1)
            )

            query_gate = (
                2.0 * torch.sigmoid(query_gate)
            ).view(batch_size, -1, 1, 1)

            pred_feat = d1 * query_gate

            pred_feat = F.interpolate(
                pred_feat,
                size=(h_in, w_in),
                mode="bilinear",
                align_corners=False,
            )

            outputs.append(
                self.segmentation_head(pred_feat)
            )

        return outputs