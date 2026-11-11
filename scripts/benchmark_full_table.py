#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT: scripts/benchmark_full_table.py
ROLE: Computes Full Comprehensive Benchmark Table matching Academic Literature:
      - ZS mAP@all (%)
      - ZS mAP@200 (%)
      - ZS P@100 (%) (Standard & Normalized)
      - ZS P@200 (%) (Standard & Normalized)
      - ZS Recall@1 (%)
      - ZS Recall@5 (%)
      - ZS Recall@10 (%)
      - OOD AUROC (%)
================================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import clip

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SBIRConfig, set_reproducible_seed
from src.data.split_manager import UnifiedSplitManager
from src.data.transforms import build_transforms
from src.data.datasets import SBIRLinearDataset
from src.engine.trainer import extract_features_dataloader
from src.engine.evaluator import AdaptiveOODDetector
from src.models.clip_lora import AsymmetricHybridLoRACLIP


def compute_extended_category_metrics(
    query_feats: np.ndarray,
    query_labels: np.ndarray,
    gallery_feats: np.ndarray,
    gallery_labels: np.ndarray,
    chunk_size: int = 512,
):
    q_norm = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-8)
    g_norm = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8)

    num_q = q_norm.shape[0]
    num_g = g_norm.shape[0]
    k_100 = min(100, num_g)
    k_200 = min(200, num_g)

    ap_all_list, ap_200_list = [], []
    p100_std_list, p200_std_list = [], []
    p100_norm_list, p200_norm_list = [], []
    rec1_list, rec5_list, rec10_list = [], [], []

    ranks = np.arange(1, num_g + 1, dtype=np.float32)[None, :]

    for start in range(0, num_q, chunk_size):
        end = min(start + chunk_size, num_q)
        sim_matrix = q_norm[start:end] @ g_norm.T
        sorted_indices = np.argsort(-sim_matrix, axis=1)
        sorted_g_labels = gallery_labels[sorted_indices]

        q_lbl = query_labels[start:end, None]
        matches = (sorted_g_labels == q_lbl).astype(np.float32)
        total_rel = matches.sum(axis=1)
        valid_mask = total_rel > 0

        cum_matches = np.cumsum(matches, axis=1)
        prec_at_k = cum_matches / ranks

        # Standard Precision (denominator = k)
        p100_std = prec_at_k[:, k_100 - 1]
        p200_std = prec_at_k[:, k_200 - 1]

        # Normalized R-Precision (denominator = min(total_rel, k))
        p100_norm = cum_matches[:, k_100 - 1] / np.maximum(np.minimum(total_rel, float(k_100)), 1.0)
        p200_norm = cum_matches[:, k_200 - 1] / np.maximum(np.minimum(total_rel, float(k_200)), 1.0)

        # Recall@k (Hit@k: Is there at least 1 correct class match in Top-K?)
        rec1 = (matches[:, :1].sum(axis=1) > 0).astype(np.float32)
        rec5 = (matches[:, :min(5, num_g)].sum(axis=1) > 0).astype(np.float32)
        rec10 = (matches[:, :min(10, num_g)].sum(axis=1) > 0).astype(np.float32)

        # Average Precision
        ap_all = (prec_at_k * matches).sum(axis=1) / np.maximum(total_rel, 1.0)
        denom_200 = np.maximum(np.minimum(total_rel, float(k_200)), 1.0)
        ap_200 = (prec_at_k[:, :k_200] * matches[:, :k_200]).sum(axis=1) / denom_200

        ap_all_list.append(ap_all[valid_mask])
        ap_200_list.append(ap_200[valid_mask])
        p100_std_list.append(p100_std[valid_mask])
        p200_std_list.append(p200_std[valid_mask])
        p100_norm_list.append(p100_norm[valid_mask])
        p200_norm_list.append(p200_norm[valid_mask])
        rec1_list.append(rec1[valid_mask])
        rec5_list.append(rec5[valid_mask])
        rec10_list.append(rec10[valid_mask])

    return {
        "mAP@all": round(float(np.concatenate(ap_all_list).mean() * 100.0), 2),
        "mAP@200": round(float(np.concatenate(ap_200_list).mean() * 100.0), 2),
        "P@100_std": round(float(np.concatenate(p100_std_list).mean() * 100.0), 2),
        "P@200_std": round(float(np.concatenate(p200_std_list).mean() * 100.0), 2),
        "P@100_norm": round(float(np.concatenate(p100_norm_list).mean() * 100.0), 2),
        "P@200_norm": round(float(np.concatenate(p200_norm_list).mean() * 100.0), 2),
        "Recall@1": round(float(np.concatenate(rec1_list).mean() * 100.0), 2),
        "Recall@5": round(float(np.concatenate(rec5_list).mean() * 100.0), 2),
        "Recall@10": round(float(np.concatenate(rec10_list).mean() * 100.0), 2),
    }


def main():
    set_reproducible_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BENCHMARK] Running on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    proj_root = Path(__file__).resolve().parent.parent
    ckpt_path = proj_root / "outputs" / "checkpoints" / "best_seco_lora_clip.pt"
    if not ckpt_path.exists():
        ckpt_path = proj_root / "outputs" / "checkpoints" / "best_seco_lora_clip.zip"
    print(f"[BENCHMARK] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg: SBIRConfig = ckpt.get("config", SBIRConfig())

    data_root = str(proj_root / "data" / "sketchy")
    manager = UnifiedSplitManager(
        data_root=data_root,
        dataset_name="sketchy",
        split_protocol="split2",
    )
    manager.print_summary()

    # Load Model
    clip_model, _ = clip.load(cfg.clip_backbone, device=device, jit=False)
    model = AsymmetricHybridLoRACLIP(
        clip_model=clip_model.float(),
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_targets=cfg.lora_targets,
    ).to(device)

    if "model_state" in ckpt:
        model.sketch_adapter.load_state_dict(ckpt["model_state"]["sketch_adapter"])
        model.photo_adapter.load_state_dict(ckpt["model_state"]["photo_adapter"])
        model.text_adapter.load_state_dict(ckpt["model_state"]["text_adapter"])
        model.sketch_visual.load_state_dict(ckpt["model_state"]["sketch_visual"])
    model.eval()

    tfs = build_transforms(image_size=cfg.image_size)

    # 1. Unseen Samples (21 Classes)
    unseen_photos, unseen_sketches = manager.collect_samples(
        manager.unseen_classes, manager.unseen_cls2idx
    )
    print(f"[BENCHMARK] Extracting features for Unseen Gallery ({len(unseen_photos):,} photos)...")
    photo_loader = DataLoader(
        SBIRLinearDataset(unseen_photos, tfs["photo_clean"]),
        batch_size=128,
        shuffle=False,
        num_workers=2,
    )
    g_feats, g_lbls, _, _ = extract_features_dataloader(photo_loader, model.encode_photo, device)

    print(f"[BENCHMARK] Extracting features for Unseen Queries ({len(unseen_sketches):,} sketches)...")
    sketch_loader = DataLoader(
        SBIRLinearDataset(unseen_sketches, tfs["sketch_clean"]),
        batch_size=128,
        shuffle=False,
        num_workers=2,
    )
    q_feats, q_lbls, _, _ = extract_features_dataloader(sketch_loader, model.encode_sketch, device)

    # Calculate All Zero-Shot Metrics
    print("[BENCHMARK] Computing Zero-Shot Metrics...")
    metrics = compute_extended_category_metrics(q_feats, q_lbls, g_feats, g_lbls)

    # 2. Compute OOD AUROC (Seen as In-Distribution vs Unseen as Out-of-Distribution)
    print("[BENCHMARK] Computing OOD AUROC...")
    seen_photos, seen_sketches = manager.collect_samples(
        manager.seen_classes, manager.seen_cls2idx
    )
    # Subsample 1,000 seen sketches and 1,000 seen photos for fast OOD calculation
    np.random.seed(42)
    sub_seen_photos = [seen_photos[i] for i in np.random.choice(len(seen_photos), min(1000, len(seen_photos)), replace=False)]
    sub_seen_sketches = [seen_sketches[i] for i in np.random.choice(len(seen_sketches), min(1000, len(seen_sketches)), replace=False)]

    seen_p_loader = DataLoader(SBIRLinearDataset(sub_seen_photos, tfs["photo_clean"]), batch_size=128, shuffle=False)
    seen_s_loader = DataLoader(SBIRLinearDataset(sub_seen_sketches, tfs["sketch_clean"]), batch_size=128, shuffle=False)

    seen_g_feats, _, _, _ = extract_features_dataloader(seen_p_loader, model.encode_photo, device)
    seen_q_feats, _, _, _ = extract_features_dataloader(seen_s_loader, model.encode_sketch, device)

    detector = AdaptiveOODDetector(alpha=0.7)
    id_scores = detector.compute_scores(seen_q_feats, seen_g_feats)
    ood_scores = detector.compute_scores(q_feats[:1000], seen_g_feats)
    ood_metrics = detector.evaluate_ood_auroc(id_scores, ood_scores)

    auroc = ood_metrics["AUROC"]

    print("\n" + "=" * 80)
    print("  COMPREHENSIVE BENCHMARK RESULTS TABLE")
    print("=" * 80)
    print(f"  ZS mAP@all (%)    : {metrics['mAP@all']}")
    print(f"  ZS mAP@200 (%)    : {metrics['mAP@200']}")
    print(f"  ZS P@100 (%) (Norm): {metrics['P@100_norm']}  | (Std): {metrics['P@100_std']}")
    print(f"  ZS P@200 (%) (Norm): {metrics['P@200_norm']}  | (Std): {metrics['P@200_std']}")
    print(f"  ZS Recall@1 (%)   : {metrics['Recall@1']}")
    print(f"  ZS Recall@5 (%)   : {metrics['Recall@5']}")
    print(f"  ZS Recall@10 (%)  : {metrics['Recall@10']}")
    print(f"  OOD AUROC (%)     : {auroc}")
    print("=" * 80)


if __name__ == "__main__":
    main()
