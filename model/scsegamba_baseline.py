# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSegambaBaseline(nn.Module):
    """
    Wrap official SCSegamba backbone+head into the same interface
    used by the existing CrackMorphFormer training framework.
    Returns a single-channel logit map.
    """
    def __init__(self, repo_root: str, upsample_to_input: bool = True):
        super().__init__()
        self.repo_root = str(repo_root)
        self.upsample_to_input = upsample_to_input

        if not os.path.isdir(self.repo_root):
            raise FileNotFoundError(f"SCSegamba repo not found: {self.repo_root}")

        # Make official repo importable.
        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)

        # Official implementation:
        #   backbone = SAVSS(arch='Crack', ...)
        #   head = MFS(8)
        from mmcls.SAVSS_dev.models.SAVSS.SAVSS import SAVSS
        from models.MFS import MFS

        self.backbone = SAVSS(
            arch='Crack',
            out_indices=(0, 1, 2, 3),
            drop_path_rate=0.2,
            final_norm=True,
            convert_syncbn=True
        )
        self.decode_head = MFS(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        feats = self.backbone(x)
        logits = self.decode_head(feats)  # [B,1,h,w]

        if self.upsample_to_input and logits.shape[-2:] != input_hw:
            logits = F.interpolate(
                logits,
                size=input_hw,
                mode="bilinear",
                align_corners=False
            )
        return logits