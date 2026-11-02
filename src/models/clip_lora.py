#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: src/models/clip_lora.py
ROLE: Asymmetric Dual-Encoder Hybrid LoRA-CLIP Model Architecture
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
"""

import copy
from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adapters import LowRankLinearAdapter, BottleneckResidualAdapter


class AsymmetricHybridLoRACLIP(nn.Module):
    """
    Asymmetric Dual-Encoder Architecture:
      - Photo Tower: 100% Frozen CLIP Visual Encoder + Bottleneck Residual Adapter.
      - Sketch Tower: Trainable LoRA on Attention Blocks + Unfrozen LayerNorm + Residual Adapter.
      - Text Alignment: Bottleneck Residual Adapter on Frozen CLIP Text Prototypes.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_targets: Tuple[str, ...] = ("out_proj",),
        embed_dim: int = 512,
        bottleneck_dim: int = 256,
        res_scale: float = 0.2,
        unfreeze_sketch_ln: bool = True,
    ):
        super().__init__()

        # 1. Nhánh Photo: Đóng băng hoàn toàn (Frozen Visual Encoder)
        self.frozen_visual = copy.deepcopy(clip_model.visual).eval()
        for p in self.frozen_visual.parameters():
            p.requires_grad = False

        # 2. Nhánh Sketch: Gắn LoRA và tùy chọn mở LayerNorm
        self.sketch_visual = clip_model.visual
        for p in self.sketch_visual.parameters():
            p.requires_grad = False

        self.lora_modules: nn.ModuleList = nn.ModuleList()

        if hasattr(self.sketch_visual, "transformer"):
            for block in self.sketch_visual.transformer.resblocks:
                if unfreeze_sketch_ln:
                    for m in block.modules():
                        if isinstance(m, nn.LayerNorm):
                            for p in m.parameters():
                                p.requires_grad = True

                # Hook LoRA into specified target projections
                if hasattr(block, "attn"):
                    attn = block.attn
                    for target_name in lora_targets:
                        if hasattr(attn, target_name):
                            orig_linear = getattr(attn, target_name)
                            lora_mod = LowRankLinearAdapter(
                                orig_linear, rank=lora_rank, alpha=lora_alpha
                            )
                            self.lora_modules.append(lora_mod)

                            def _make_hook(l_mod):
                                def _hook(module, args, output):
                                    x_in = args[0]
                                    delta = (
                                        x_in.to(l_mod.lora_A.dtype)
                                        @ l_mod.lora_A.t()
                                        @ l_mod.lora_B.t()
                                    ) * l_mod.scaling
                                    return output + delta.to(output.dtype)
                                return _hook

                            orig_linear.register_forward_hook(_make_hook(lora_mod))

        # 3. Bottleneck Residual Adapters
        self.sketch_adapter = BottleneckResidualAdapter(embed_dim, bottleneck_dim, res_scale)
        self.photo_adapter = BottleneckResidualAdapter(embed_dim, bottleneck_dim, res_scale)
        self.text_adapter = BottleneckResidualAdapter(embed_dim, bottleneck_dim, res_scale)

    @torch.no_grad()
    def encode_frozen_visual(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Raw feature extraction from 100% frozen CLIP visual backbone."""
        dtype = next(self.frozen_visual.parameters()).dtype
        feat = self.frozen_visual(img_tensor.to(dtype)).float()
        return F.normalize(feat, dim=-1)

    def encode_sketch(self, sketch_tensor: torch.Tensor) -> torch.Tensor:
        """Full trainable forward pass for sketch images."""
        dtype = next(self.sketch_visual.parameters()).dtype
        raw_feat = self.sketch_visual(sketch_tensor.to(dtype)).float()
        return self.sketch_adapter(F.normalize(raw_feat, dim=-1))

    def encode_photo(self, photo_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass for photos through frozen backbone + trained photo adapter."""
        raw_feat = self.encode_frozen_visual(photo_tensor)
        return self.photo_adapter(raw_feat)

    def encode_photo_from_cached(self, cached_photo_feat: torch.Tensor) -> torch.Tensor:
        """Fast-path forward pass bypassing visual backbone using pre-computed features."""
        norm_feat = F.normalize(cached_photo_feat.float(), dim=-1)
        return self.photo_adapter(norm_feat)

    def adapt_text_prototypes(self, raw_text_prototypes: torch.Tensor) -> torch.Tensor:
        """Projects frozen text embeddings through trained text adapter."""
        return self.text_adapter(F.normalize(raw_text_prototypes.float(), dim=-1))

    def count_parameters(self) -> Dict[str, Any]:
        """Calculates total, trainable, and percentage ratio of parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_ratio_pct": round(100.0 * trainable / max(total, 1), 4),
        }


@torch.no_grad()
def build_clip_text_prototypes(
    clip_model: nn.Module,
    tokenize_fn,
    class_names: List[str],
    device: torch.device,
    template: str = "a photo of a {}",
) -> torch.Tensor:
    """Pre-computes L2-normalized CLIP text prototypes for all classes."""
    prompts = [
        template.format(c.replace("_", " ").replace("(", "").replace(")", "").strip())
        for c in class_names
    ]
    tokens = tokenize_fn(prompts).to(device)
    text_feats = clip_model.encode_text(tokens).float()
    return F.normalize(text_feats, dim=-1)
