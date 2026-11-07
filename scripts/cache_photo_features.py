#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT: scripts/cache_photo_features.py
ROLE: Offline Fast Feature Pre-computation for Gallery & Training Photos
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
Usage:
  python scripts/cache_photo_features.py --data-root data/sketchy --output outputs/cache/photo_features_cache.pt
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SBIRConfig, set_reproducible_seed, TORCH_AVAILABLE
from src.data.split_manager import UnifiedSplitManager
from src.data.transforms import build_transforms
from src.engine.trainer import precompute_photo_cache

if TORCH_AVAILABLE:
    import torch
    import clip
    from src.models.clip_lora import AsymmetricHybridLoRACLIP


def main():
    parser = argparse.ArgumentParser(description="Offline Photo Feature Caching Tool")
    parser.add_argument("--data-root", type=str, default="data/sketchy")
    parser.add_argument("--dataset", type=str, default="sketchy")
    parser.add_argument("--photo-subdir", type=str, default="photo")
    parser.add_argument("--sketch-subdir", type=str, default="splitted_sketches")
    parser.add_argument("--split", type=str, default="split2")
    parser.add_argument("--output", type=str, default="outputs/cache/photo_features_cache.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_reproducible_seed(42)
    device = torch.device(args.device)

    print(f"[CACHE] Initializing UnifiedSplitManager on {args.data_root}...")
    manager = UnifiedSplitManager(
        data_root=args.data_root,
        dataset_name=args.dataset,
        photo_subdir=args.photo_subdir,
        sketch_subdir=args.sketch_subdir,
        split_protocol=args.split,
    )
    seen_photos, _ = manager.collect_samples(manager.seen_classes, manager.seen_cls2idx)
    print(f"[CACHE] Found {len(seen_photos):,} seen photos to pre-compute.")

    print(f"[CACHE] Loading CLIP ViT-B/32 on {device}...")
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model = AsymmetricHybridLoRACLIP(clip_model=clip_model.float()).to(device)

    tfs = build_transforms()
    cache = precompute_photo_cache(
        seen_photos,
        model=model,
        transform=tfs["photo_clean"],
        device=device,
        batch_size=args.batch_size,
        cache_path=args.output,
    )
    print(f"[CACHE] Finished! Cached {len(cache):,} features into {args.output}")


if __name__ == "__main__":
    main()
