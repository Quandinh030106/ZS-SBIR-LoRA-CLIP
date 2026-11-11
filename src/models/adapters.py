#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: src/models/adapters.py
ROLE: Parameter-Efficient Fine-Tuning (PEFT) Components:
      - LowRankLinearAdapter (LoRA)
      - BottleneckResidualAdapter (Zero-Init Residual)
      - SpatialAttentionPool (Spatial Patch-Token Attention Pooling)
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankLinearAdapter(nn.Module):
    """
    Low-Rank Adaptation (LoRA) module for Linear layers (Hu et al., ICLR 2022).
    Computes: y = W_0 x + (alpha / rank) * (x @ A.T @ B.T)
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / float(rank)

        # LoRA down-projection A (initialized Gaussian)
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features, dtype=base_linear.weight.dtype)
            * (1.0 / math.sqrt(rank))
        )
        # LoRA up-projection B (initialized to zeros so delta is 0 at start)
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=base_linear.weight.dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        delta = (
            x.to(self.lora_A.dtype) @ self.lora_A.t() @ self.lora_B.t()
        ) * self.scaling
        return base_out + delta.to(base_out.dtype)


class BottleneckResidualAdapter(nn.Module):
    """
    Bottleneck Residual Adapter with Zero-Initialization.
    Guarantees identity mapping at Epoch 0: out = x + res_scale * 0 = x
    """

    def __init__(
        self,
        embed_dim: int = 512,
        bottleneck_dim: int = 256,
        res_scale: float = 0.2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(bottleneck_dim, embed_dim)
        self.res_scale = res_scale

        # Kaiming init for down-projection, strict Zero-init for up-projection
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        residual = self.fc2(self.act(self.fc1(x_fp32)))
        out = x_fp32 + self.res_scale * residual
        return F.normalize(out, dim=-1)


class SpatialAttentionPool(nn.Module):
    """
    Learned Spatial Attention Pooling over ViT Patch Tokens (7x7=49 tokens in ViT-B/32).
    Weights fine-grained local stroke contours (wings, wheels, windows, handles)
    to prevent geometric collapse in freehand sketches.
    Zero-initialization on projection layer ensures zero degradation at Epoch 0.
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: [B, N, D] (e.g. [B, 49, 512])
        scores = (patch_tokens @ self.query.transpose(-1, -2)).squeeze(-1)  # [B, N]
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, N, 1]
        pooled = (patch_tokens * weights).sum(dim=1)  # [B, D]
        return self.proj(pooled)
