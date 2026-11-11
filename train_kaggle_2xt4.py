#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: train_kaggle_2xt4.py
ROLE: Dedicated High-Performance Runner for Kaggle (2x NVIDIA T4 GPUs)
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
Execution on Kaggle:
  !python train_kaggle_2xt4.py --epochs 15 --batch-size 128

Features:
  - Auto-locates Sketchy dataset in /kaggle/input/ or local directories.
  - Multi-GPU DataParallel over 2x T4 with PyTorch Mixed Precision (AMP).
  - Spatial Attention Pooling over ViT patch tokens (local stroke contours).
  - LoRA Rank 16 (alpha=32.0) on vision attention projections.
  - In-batch Semi-Hard Negative Mining (margin 0.25).
  - Full Diagnostic Report Table printed EVERY epoch (12 academic metrics + 5 losses).
  - Automatic training curve generation (saved to outputs/training_curves.png).
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
import numpy as np
import torch

# Ensure repo root is on sys.path
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

from src.config import SBIRConfig, set_reproducible_seed
from src.engine.trainer import run_training_pipeline


def find_kaggle_dataset_path(preferred_name: str = "sketchy") -> str:
    """Intelligently detects dataset directory across Kaggle and local paths."""
    candidate_roots = [
        Path(f"/kaggle/input/{preferred_name}"),
        Path("/kaggle/input/sketchy-dataset/sketchy"),
        Path("/kaggle/input/sketchy-dataset"),
        Path("/kaggle/input/sketchy-benchmark/sketchy"),
        Path(f"data/{preferred_name}"),
        PROJ_ROOT / "data" / preferred_name,
    ]

    for cand in candidate_roots:
        if cand.exists() and (cand / "photo").exists():
            print(f"[DATA DETECTED] Found valid dataset at: {cand}")
            return str(cand)

    # Recursive search inside /kaggle/input if available
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for path in kaggle_input.rglob("photo"):
            parent = path.parent
            if (parent / "splitted_sketches").exists() or (parent / "sketch").exists():
                print(f"[DATA DETECTED] Found valid dataset via recursive search: {parent}")
                return str(parent)

    fallback = str(PROJ_ROOT / "data" / preferred_name)
    print(f"[WARNING] Could not auto-detect dataset. Defaulting to: {fallback}")
    return fallback


def plot_academic_curves(history_json_path: str, output_img_path: str) -> None:
    """Generates a publication-grade 3-panel figure showing training progression."""
    if not os.path.exists(history_json_path):
        return

    try:
        import matplotlib.pyplot as plt
        with open(history_json_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        if len(history) < 2:
            return

        epochs = [h["epoch"] for h in history]
        map_all = [h["unseen_metrics"].get("mAP@all", 0.0) for h in history]
        map_200 = [h["unseen_metrics"].get("mAP@200", 0.0) for h in history]
        rec1 = [h["unseen_metrics"].get("Recall@1", 0.0) for h in history]
        rec5 = [h["unseen_metrics"].get("Recall@5", 0.0) for h in history]

        acc1 = [h["unseen_metrics"].get("Acc@1", 0.0) for h in history]
        acc5 = [h["unseen_metrics"].get("Acc@5", 0.0) for h in history]
        acc10 = [h["unseen_metrics"].get("Acc@10", 0.0) for h in history]

        loss_tot = [h["losses"].get("loss_total", 0.0) for h in history]
        loss_tri = [h["losses"].get("loss_triplet", 0.0) for h in history]
        loss_ntx = [h["losses"].get("loss_ntxent", 0.0) for h in history]
        loss_cls = [h["losses"].get("loss_class", 0.0) for h in history]

        plt.figure(figsize=(18, 5))
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # Subplot 1: Zero-Shot Category Metrics
        plt.subplot(1, 3, 1)
        plt.plot(epochs, map_all, "o-", label="mAP@all", color="#1f77b4", linewidth=2)
        plt.plot(epochs, map_200, "s-", label="mAP@200", color="#ff7f0e", linewidth=2)
        plt.plot(epochs, rec1, "^--", label="Recall@1 (Hit@1)", color="#2ca02c", alpha=0.8)
        plt.plot(epochs, rec5, "v--", label="Recall@5 (Hit@5)", color="#d62728", alpha=0.8)
        plt.title("Zero-Shot Retrieval (21 Unseen Classes)", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Score (%)")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.6)

        # Subplot 2: Fine-Grained Instance Alignment
        plt.subplot(1, 3, 2)
        plt.plot(epochs, acc1, "o-", label="Acc@1 (Exact Photo)", color="#9467bd", linewidth=2)
        plt.plot(epochs, acc5, "s-", label="Acc@5", color="#8c564b", linewidth=2)
        plt.plot(epochs, acc10, "d-", label="Acc@10", color="#e377c2", linewidth=2)
        plt.title("Fine-Grained Instance Matching", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)

        # Subplot 3: Multi-Objective Loss Convergence
        plt.subplot(1, 3, 3)
        plt.plot(epochs[1:], loss_tot[1:], "k-", label="Loss Total", linewidth=2.5)
        plt.plot(epochs[1:], loss_tri[1:], "--", label="Loss Triplet (Mined)", color="#ff7f0e")
        plt.plot(epochs[1:], loss_ntx[1:], "--", label="Loss NT-Xent", color="#2ca02c")
        plt.plot(epochs[1:], loss_cls[1:], ":", label="Loss Class (Seen)", color="#d62728")
        plt.title("Multi-Objective Training Losses", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        plt.savefig(output_img_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[REPORT] Saved publication diagnostic curves to: {output_img_path}")
    except Exception as e:
        print(f"[WARNING] Could not plot training curves: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="CS331: Kaggle 2x NVIDIA T4 Multi-GPU Training Runner"
    )
    parser.add_argument("--data-root", type=str, default="",
                        help="Optional explicit dataset root (auto-detected if empty)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-targets", type=str, default="out_proj")
    parser.add_argument("--triplet-margin", type=float, default=0.25)
    parser.add_argument("--mining-mode", type=str, default="semi_hard")
    parser.add_argument("--save-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--use-cached-photos", action="store_true")
    parser.add_argument("--eval-baseline-only", action="store_true")

    args = parser.parse_args()

    # Detect dataset directory
    data_root = args.data_root if args.data_root else find_kaggle_dataset_path("sketchy")
    print(f"[SETUP] Target Dataset Root: {data_root}")

    targets = tuple([t.strip() for t in args.lora_targets.split(",") if t.strip()])

    config = SBIRConfig(
        dataset_name="sketchy",
        data_root=data_root,
        split_protocol="split2",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=targets,
        enable_patch_pooling=True,
        patch_pool_weight=0.3,
        triplet_margin=args.triplet_margin,
        mining_mode=args.mining_mode,
        save_dir=args.save_dir,
        metrics_history_path="outputs/training_metrics_history.json",
        use_cached_photos=args.use_cached_photos,
    )

    # Run the complete multi-GPU pipeline
    run_training_pipeline(config, eval_only_baseline=args.eval_baseline_only)

    # Plot final curves
    plot_academic_curves(
        history_json_path=config.metrics_history_path,
        output_img_path="outputs/training_curves.png",
    )


if __name__ == "__main__":
    main()
