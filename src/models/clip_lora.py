#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: src/models/clip_lora.py
ROLE: Asymmetric Dual-Encoder Hybrid LoRA-CLIP Model Architecture
      with Spatial Patch-Token Attention Pooling
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
Features:
  - Asymmetric Dual-Encoder (100% Frozen Photo Tower + LoRA-adapted Sketch Tower).
  - LowRankLinearAdapter on Vision Attention & MLP Projections.
  - SpatialAttentionPool extracting both CLS and 49 Patch Tokens for fine spatial strokes.
  - BottleneckResidualAdapters with strict Zero-Initialization.
  - Multimodal Sketch + Text Duet Query Fusion (CVPR 2024 'You'll Never Walk Alone').
"""

import copy
from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adapters import (
    LowRankLinearAdapter,
    BottleneckResidualAdapter,
    SpatialAttentionPool,
)


class AsymmetricHybridLoRACLIP(nn.Module):
    """
    Asymmetric Dual-Encoder Architecture:
      - Photo Tower: 100% Frozen CLIP Visual Encoder + Bottleneck Residual Adapter.
      - Sketch Tower: Trainable LoRA on Target Projections + Unfrozen LayerNorm
                     + Spatial Patch-Token Attention Pooler + Bottleneck Residual Adapter.
      - Text Alignment: Bottleneck Residual Adapter on Frozen CLIP Text Prototypes.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_targets: Tuple[str, ...] = ("out_proj",),
        embed_dim: int = 512,
        bottleneck_dim: int = 256,
        res_scale: float = 0.2,
        unfreeze_sketch_ln: bool = True,
        enable_patch_pooling: bool = True,
        patch_pool_weight: float = 0.3,
    ):
        super().__init__()
        self.enable_patch_pooling = enable_patch_pooling
        self.patch_pool_weight = patch_pool_weight
        self.embed_dim = embed_dim

        # 1. Photo Tower: 100% Frozen Visual Encoder
        self.frozen_visual = copy.deepcopy(clip_model.visual).eval()
        for p in self.frozen_visual.parameters():
            p.requires_grad = False

        # 2. Sketch Tower: Trainable LoRA + Unfrozen LayerNorm
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

                # Search both attention and MLP blocks for target projections
                candidate_modules = []
                if hasattr(block, "attn"):
                    candidate_modules.append(block.attn)
                if hasattr(block, "mlp"):
                    candidate_modules.append(block.mlp)

                for parent_mod in candidate_modules:
                    for target_name in lora_targets:
                        if hasattr(parent_mod, target_name):
                            orig_linear = getattr(parent_mod, target_name)
                            if isinstance(orig_linear, nn.Linear):
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

        # 3. Spatial Patch Attention Pooler (Prevents Geometric Collapse in Freehand Sketches)
        if self.enable_patch_pooling:
            self.spatial_pooler = SpatialAttentionPool(embed_dim)
        else:
            self.spatial_pooler = None

        # 4. Bottleneck Residual Adapters (Zero-Initialized)
        self.sketch_adapter = BottleneckResidualAdapter(embed_dim, bottleneck_dim, res_scale)
        self.photo_adapter = BottleneckResidualAdapter(embed_dim, bottleneck_dim, res_scale)
        self.text_adapter = BottleneckResidualAdapter(embed_dim, bottleneck_dim, res_scale)

    @staticmethod
    def _extract_dual_tokens(visual: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Custom ViT forward pass extracting both [CLS] token and 49 Patch Tokens.
        CLS Token captures high-level holistic semantics.
        Patch Tokens capture fine-grained spatial stroke contours.
        """
        dtype = visual.conv1.weight.dtype
        x = x.to(dtype)
        x = visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        cls_token = visual.ln_post(x[:, 0, :])
        patch_tokens = visual.ln_post(x[:, 1:, :])

        if visual.proj is not None:
            cls_token = cls_token @ visual.proj
            patch_tokens = patch_tokens @ visual.proj

        return cls_token.float(), patch_tokens.float()

    @torch.no_grad()
    def encode_frozen_visual(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Raw feature extraction from 100% frozen CLIP visual backbone."""
        dtype = (
            self.frozen_visual.conv1.weight.dtype
            if hasattr(self.frozen_visual, "conv1")
            else next(self.frozen_visual.parameters()).dtype
        )
        feat = self.frozen_visual(img_tensor.to(dtype)).float()
        return F.normalize(feat, dim=-1)

    def encode_sketch(self, sketch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Trainable forward pass for sketch images:
        Fuses [CLS] semantic token with spatially pooled stroke patch tokens.
        """
        if self.enable_patch_pooling and self.spatial_pooler is not None and hasattr(self.sketch_visual, "conv1"):
            cls_feat, patch_feats = self._extract_dual_tokens(self.sketch_visual, sketch_tensor)
            pooled_patch = self.spatial_pooler(patch_feats)
            fused_feat = F.normalize(cls_feat + self.patch_pool_weight * pooled_patch, dim=-1)
            return self.sketch_adapter(fused_feat)

        # Fallback for standard architectures without patch pooling
        dtype = (
            self.sketch_visual.conv1.weight.dtype
            if hasattr(self.sketch_visual, "conv1")
            else next(self.sketch_visual.parameters()).dtype
        )
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

    def encode_multimodal_query(
        self,
        sketch_tensor: torch.Tensor,
        text_prompt: Optional[str] = None,
        clip_model: Optional[nn.Module] = None,
        tokenize_fn=None,
        device: Optional[torch.device] = None,
        alpha: float = 0.7,
    ) -> torch.Tensor:
        """
        Multimodal Sketch + Text Duet Retrieval (CVPR 2024 'You'll Never Walk Alone').
        Fuses visual sketch representation with optional textual descriptions.
        """
        f_s = self.encode_sketch(sketch_tensor)
        if text_prompt is None or not text_prompt.strip() or clip_model is None or tokenize_fn is None:
            return f_s

        dev = device if device is not None else sketch_tensor.device
        tokens = tokenize_fn([text_prompt.strip()]).to(dev)
        with torch.no_grad():
            raw_text = clip_model.encode_text(tokens).float()
        f_t = self.adapt_text_prototypes(raw_text)

        f_fused = alpha * f_s + (1.0 - alpha) * f_t
        return F.normalize(f_fused, dim=-1)

    def forward(
        self,
        sketch_clean: Optional[torch.Tensor] = None,
        photo_pos: Optional[torch.Tensor] = None,
        photo_neg: Optional[torch.Tensor] = None,
        sketch_pert: Optional[torch.Tensor] = None,
        photo_pert: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified multi-tensor forward pass supporting DataParallel across multiple GPUs.
        """
        out: Dict[str, torch.Tensor] = {}
        if sketch_clean is not None:
            out["f_s_clean"] = self.encode_sketch(sketch_clean)
        if photo_pos is not None:
            out["f_p_pos"] = self.encode_photo(photo_pos)
        if photo_neg is not None:
            out["f_p_neg"] = self.encode_photo(photo_neg)
        if sketch_pert is not None:
            out["f_s_pert"] = self.encode_sketch(sketch_pert)
        if photo_pert is not None:
            out["f_p_pert"] = self.encode_photo(photo_pert)
        return out

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
