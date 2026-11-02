#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT: scripts/evaluate.py
ROLE: Standalone Evaluation & Visualization for Checkpoint Check
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
Usage:
  python scripts/evaluate.py --checkpoint outputs/checkpoints/best_seco_lora_clip.pt --data-root data/sketchy
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
from src.data.datasets import SBIRLinearDataset
from src.engine.evaluator import (
    compute_category_sbir_metrics_np,
    compute_fine_grained_sbir_metrics_np,
    AdaptiveOODDetector,
)
from src.engine.trainer import extract_features_dataloader
from src.utils.visualization import plot_retrieval_results, plot_ood_distribution

if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader
    import clip
    from src.models.clip_lora import AsymmetricHybridLoRACLIP, build_clip_text_prototypes


def main():
    parser = argparse.ArgumentParser(description="Standalone Model Evaluation CLI")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained .pt checkpoint file")
    parser.add_argument("--data-root", type=str, default="data/sketchy")
    parser.add_argument("--dataset", type=str, default="sketchy")
    parser.add_argument("--split", type=str, default="split2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--viz-samples", type=int, default=5,
                        help="Number of query retrieval visualization plots to export")
    parser.add_argument("--output-dir", type=str, default="outputs/eval_results")
    args = parser.parse_args()

    set_reproducible_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[EVAL] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg: SBIRConfig = ckpt.get("config", SBIRConfig())

    manager = UnifiedSplitManager(
        data_root=args.data_root,
        dataset_name=args.dataset,
        split_protocol=args.split,
    )
    manager.print_summary()

    unseen_photos, unseen_sketches = manager.collect_samples(
        manager.unseen_classes, manager.unseen_cls2idx
    )
    print(f"[EVAL] Unseen Gallery: {len(unseen_photos):,} photos | Unseen Queries: {len(unseen_sketches):,} sketches")

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

    photo_loader = DataLoader(
        SBIRLinearDataset(unseen_photos, tfs["photo_clean"]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    sketch_loader = DataLoader(
        SBIRLinearDataset(unseen_sketches, tfs["sketch_clean"]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("[EVAL] Extracting gallery photo features...")
    g_feats, g_lbls, g_inst, g_paths = extract_features_dataloader(
        photo_loader, model.encode_photo, device
    )

    print("[EVAL] Extracting query sketch features...")
    q_feats, q_lbls, q_inst, q_paths = extract_features_dataloader(
        sketch_loader, model.encode_sketch, device
    )

    print("\n" + "=" * 70)
    print("  ZERO-SHOT EVALUATION RESULTS")
    print("=" * 70)
    cat_metrics = compute_category_sbir_metrics_np(q_feats, q_lbls, g_feats, g_lbls)
    fg_metrics = compute_fine_grained_sbir_metrics_np(q_feats, q_inst, g_feats, g_inst)

    for k, v in cat_metrics.items():
        print(f"  {k:12s}: {v}%")
    for k, v in fg_metrics.items():
        print(f"  {k:12s}: {v}%")
    print("=" * 70)

    # Plot sample visual retrievals
    if args.viz_samples > 0:
        print(f"[VIZ] Generating {args.viz_samples} visual retrieval plots...")
        sim_mat = q_feats @ g_feats.T
        for idx in range(min(args.viz_samples, len(q_paths))):
            top_indices = np.argsort(-sim_mat[idx])[:5]
            ret_paths = [g_paths[i] for i in top_indices]
            ret_labels = [int(g_lbls[i]) for i in top_indices]
            save_p = os.path.join(args.output_dir, f"query_sample_{idx+1}.png")
            plot_retrieval_results(
                sketch_path=q_paths[idx],
                retrieved_paths=ret_paths,
                retrieved_labels=ret_labels,
                query_label=int(q_lbls[idx]),
                top_k=5,
                save_path=save_p,
            )


if __name__ == "__main__":
    main()
