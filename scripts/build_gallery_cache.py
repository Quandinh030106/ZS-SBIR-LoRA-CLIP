#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT: scripts/build_gallery_cache.py
ROLE: Pre-computes 512D Normalized Embeddings for 12,500 Gallery Photos
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import clip

# Add project root to sys.path
proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))

from src.config import SKETCHY_SPLIT2_UNSEEN_21, CLIP_MEAN, CLIP_STD
from src.data.transforms import build_transforms
from src.data.datasets import SBIRLinearDataset


class GalleryPrecompDataset(torch.utils.data.Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, c_idx, c_name, is_unseen = self.items[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), c_idx, c_name, is_unseen, p


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CACHE BUILDER] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    photo_dir = proj_root / "data" / "sketchy" / "photo" / "tx_000000000000"
    if not photo_dir.exists():
        photo_dir = proj_root / "data" / "sketchy" / "photo"
    print(f"[CACHE BUILDER] Photo Directory: {photo_dir}")

    # Discover classes and valid images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    classes = sorted([d.name for d in photo_dir.iterdir() if d.is_dir()])
    unseen_set = set(SKETCHY_SPLIT2_UNSEEN_21)

    samples = []
    cls2idx = {c: i for i, c in enumerate(classes)}

    for c in classes:
        c_folder = photo_dir / c
        c_idx = cls2idx[c]
        for f in sorted(c_folder.iterdir()):
            if f.suffix.lower() in valid_exts:
                samples.append((str(f), c_idx, c, c in unseen_set))

    print(f"[CACHE BUILDER] Found {len(samples):,} photos across {len(classes)} classes.")

    # Load Frozen CLIP Backbone
    print("[CACHE BUILDER] Loading CLIP ViT-B/32...")
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    tfs = build_transforms(image_size=224)["photo_clean"]

    loader = DataLoader(
        GalleryPrecompDataset(samples, tfs),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    feats_list, paths_list, labels_list, class_names_list, unseen_mask_list = [], [], [], [], []

    print("[CACHE BUILDER] Extracting 512D representations...")
    with torch.no_grad():
        for imgs, c_idxs, c_names, is_unseens, paths in loader:
            imgs = imgs.to(device)
            feats = clip_model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_list.append(feats.cpu())
            paths_list.extend(paths)
            labels_list.extend(c_idxs.tolist())
            class_names_list.extend(c_names)
            unseen_mask_list.extend(is_unseens.tolist())

    gallery_matrix = torch.cat(feats_list, dim=0)
    unseen_mask = torch.tensor(unseen_mask_list, dtype=torch.bool)

    output_dir = proj_root / "outputs" / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gallery_index.pt"

    payload = {
        "features": gallery_matrix,               # (N, 512) float32 normalized
        "paths": paths_list,                     # List[str]
        "labels": labels_list,                   # List[int]
        "class_names": class_names_list,         # List[str]
        "is_unseen": unseen_mask,                # Tensor[bool] of shape (N,)
        "classes": classes,                      # List[str] of 125 class names
    }

    torch.save(payload, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print("=" * 70)
    print(f"SUCCESS: Saved gallery index ({gallery_matrix.shape[0]:,} photos, {size_mb:.2f} MB) to:")
    print(f"  {output_path}")
    print(f"Unseen photos: {unseen_mask.sum().item():,} | Seen photos: {(~unseen_mask).sum().item():,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
