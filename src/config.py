#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: src/config.py
ROLE: Centralized Configuration, Dataset Split Protocols & System Hyperparameters
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
"""

import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK ZERO-SHOT SPLIT PROTOCOLS
# ─────────────────────────────────────────────────────────────────────────────

# Standard Split 2 (Yelamarthi et al., ECCV 2018): 104 Seen / 21 Unseen
# Unseen classes are semantically and morphologically distinct from seen classes.
SKETCHY_SPLIT2_UNSEEN_21: List[str] = [
    "cabin", "cow", "dolphin", "door", "giraffe",
    "helicopter", "mouse", "pear", "raccoon", "rhinoceros",
    "saw", "scissors", "seagull", "skyscraper", "songbird",
    "sword", "tree", "wading_bird", "wheelchair", "windmill",
    "window"
]

# Standard Split 1 (Shen et al., CVPR 2018): 100 Seen / 25 Unseen
SKETCHY_SPLIT1_UNSEEN_25: List[str] = [
    "airplane", "alarm_clock", "ant", "ape", "apple",
    "armor", "axe", "banana", "bat", "bear",
    "bee", "beetle", "bell", "bench", "bicycle",
    "blimp", "bread", "butterfly", "cabin", "camel",
    "candle", "cannon", "car_(sedan)", "castle", "cat"
]

# Standard TU-Berlin Zero-Shot Split (30 Unseen classes out of 250)
TUBERLIN_UNSEEN_30: List[str] = [
    "airplane", "alarm_clock", "apple", "arm", "banana",
    "baseball_bat", "basket", "bathtub", "bear", "bed",
    "bee", "bell", "bench", "bicycle", "binoculars",
    "blimp", "book", "bottle", "bowl", "bridge",
    "bus", "butterfly", "cactus", "calculator", "camel",
    "camera", "candle", "cannon", "canoe", "car_(sedan)"
]

# OpenAI CLIP ViT-B/32 Normalization Constants
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SBIRConfig:
    """Complete System & Hyperparameter Configuration for ZS-SBIR."""

    # Dataset & Storage
    dataset_name: str = "sketchy"  # 'sketchy', 'tuberlin', 'qmul_shoe'
    data_root: str = "data/sketchy"
    photo_subdir: str = "photo"
    sketch_subdir: str = "splitted_sketches"  # can be 'sketch' or 'splitted_sketches'
    canonical_tx: str = "tx_000000000000"
    split_protocol: str = "split2"  # 'split2' (104/21), 'split1' (100/25), or 'custom'

    # Preprocessor (CV Engine)
    image_size: int = 224
    pad_ratio: float = 0.08
    min_contour_area: float = 15.0       # Filters out single dust/noise pixels
    min_stroke_density: float = 0.001    # Detects empty or near-empty sketches
    enable_morph: bool = True
    morph_prob: float = 0.5
    enable_random_erase: bool = True
    erase_prob: float = 0.5
    erase_scale: Tuple[float, float] = (0.02, 0.15)
    erase_ratio: Tuple[float, float] = (0.3, 3.3)

    # Model & LoRA Architecture
    clip_backbone: str = "ViT-B/32"
    embed_dim: int = 512
    bottleneck_dim: int = 256
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_targets: Tuple[str, ...] = ("out_proj",)  # expandable to ("out_proj", "v_proj")
    res_scale: float = 0.2
    unfreeze_sketch_ln: bool = True

    # Multi-Objective Loss & Mining
    triplet_margin: float = 0.3
    temp_ntxent: float = 0.07
    temp_cls: float = 0.07
    temp_consist: float = 0.07
    lambda_cls: float = 0.3           # Reduced from 1.0 to prevent unseen class bias
    lambda_consist: float = 1.0
    mining_mode: str = "semi_hard"    # 'semi_hard' or 'random'

    # Acceleration & Caching
    use_cached_photos: bool = False
    photo_cache_path: str = "outputs/cache/photo_features_cache.pt"

    # Optimization
    batch_size: int = 64
    num_epochs: int = 15
    lr: float = 2e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    num_workers: int = 4
    seed: int = 42
    save_dir: str = "outputs/checkpoints"

    # Evaluation & OOD Detection
    eval_chunk_size: int = 512
    ood_alpha: float = 0.7
    ood_std_multiplier: float = 1.5


def set_reproducible_seed(seed: int = 42) -> None:
    """Ensures 100% deterministic reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
