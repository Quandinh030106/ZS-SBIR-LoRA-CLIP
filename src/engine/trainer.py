#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: src/engine/trainer.py
ROLE: End-to-End Training & Evaluation Engine with Mixed Precision (AMP) & Photo Caching
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
"""

import os
import time
from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import SBIRConfig, set_reproducible_seed
from src.data.split_manager import UnifiedSplitManager
from src.data.transforms import build_transforms
from src.data.datasets import SBIRLinearDataset, SeCoLoRATripletDataset
from src.models.clip_lora import AsymmetricHybridLoRACLIP, build_clip_text_prototypes
from src.losses.multi_loss import SeCoLoRAMultiObjectiveLoss
from src.engine.evaluator import (
    compute_category_sbir_metrics_np,
    compute_fine_grained_sbir_metrics_np,
)


def extract_features_dataloader(
    loader: DataLoader, encode_fn, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Extracts L2-normalized representations across a DataLoader."""
    feats_list, labels_list, inst_list, paths_list = [], [], [], []
    for imgs, labels, inst_ids, paths in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.no_grad():
            feats = encode_fn(imgs)
        feats_list.append(feats.cpu().numpy())
        labels_list.append(labels.numpy())
        inst_list.extend(list(inst_ids))
        paths_list.extend(list(paths))

    return (
        np.concatenate(feats_list, axis=0),
        np.concatenate(labels_list, axis=0),
        inst_list,
        paths_list,
    )


def precompute_photo_cache(
    photos: List[Tuple[str, int, str]],
    model: AsymmetricHybridLoRACLIP,
    transform,
    device: torch.device,
    batch_size: int = 128,
    cache_path: str = "outputs/cache/photo_features_cache.pt",
) -> Dict[str, torch.Tensor]:
    """
    Offline pre-computes 512D representations for all photos using the frozen visual encoder.
    Saves and returns an in-memory dictionary {filepath: tensor_512}.
    """
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_path):
        print(f"[CACHE] Loading pre-computed photo features from: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f"[CACHE] Pre-computing features for {len(photos):,} photos...")
    loader = DataLoader(
        SBIRLinearDataset(photos, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    cache: Dict[str, torch.Tensor] = {}
    model.eval()

    with torch.no_grad():
        for imgs, _, _, paths in loader:
            imgs = imgs.to(device)
            feats = model.encode_frozen_visual(imgs).cpu()
            for p, f in zip(paths, feats):
                cache[p] = f

    torch.save(cache, cache_path)
    print(f"[CACHE] Saved {len(cache):,} photo features to: {cache_path}")
    return cache


def run_training_pipeline(cfg: SBIRConfig, eval_only_baseline: bool = False) -> None:
    """Full execution pipeline: Data prep -> Model init -> Train loop -> Eval."""
    import clip

    set_reproducible_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 1. Initialize Split Manager
    manager = UnifiedSplitManager(
        data_root=cfg.data_root,
        dataset_name=cfg.dataset_name,
        photo_subdir=cfg.photo_subdir,
        sketch_subdir=cfg.sketch_subdir,
        canonical_tx=cfg.canonical_tx,
        split_protocol=cfg.split_protocol,
    )
    manager.print_summary()

    seen_photos, seen_sketches = manager.collect_samples(
        manager.seen_classes, manager.seen_cls2idx
    )
    unseen_photos, unseen_sketches = manager.collect_samples(
        manager.unseen_classes, manager.unseen_cls2idx
    )

    print(f"[DATA] Seen ({len(manager.seen_classes)} classes):   {len(seen_photos):,} photos | {len(seen_sketches):,} sketches")
    print(f"[DATA] Unseen ({len(manager.unseen_classes)} classes): {len(unseen_photos):,} photos | {len(unseen_sketches):,} sketches")

    # 2. Initialize Model
    print(f"[MODEL] Loading CLIP Backbone: {cfg.clip_backbone} on {device}...")
    clip_model, _ = clip.load(cfg.clip_backbone, device=device, jit=False)
    clip_model = clip_model.float()

    model = AsymmetricHybridLoRACLIP(
        clip_model=clip_model,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_targets=cfg.lora_targets,
        embed_dim=cfg.embed_dim,
        bottleneck_dim=cfg.bottleneck_dim,
        res_scale=cfg.res_scale,
        unfreeze_sketch_ln=cfg.unfreeze_sketch_ln,
    ).to(device)

    param_stats = model.count_parameters()
    print(f"[MODEL] Total Parameters:     {param_stats['total_params']:,}")
    print(f"[MODEL] Trainable Parameters: {param_stats['trainable_params']:,} ({param_stats['trainable_ratio_pct']}%)")

    # 3. Build Text Prototypes & Transforms
    seen_text_proto_raw = build_clip_text_prototypes(
        clip_model, clip.tokenize, manager.seen_classes, device
    )
    tfs = build_transforms(
        image_size=cfg.image_size,
        pad_ratio=cfg.pad_ratio,
        min_contour_area=cfg.min_contour_area,
        morph_prob=cfg.morph_prob,
        erase_prob=cfg.erase_prob,
        partial_prob=cfg.partial_prob if cfg.enable_partial_mask else 0.0,
        noise_prob=cfg.noise_prob if cfg.enable_stroke_noise else 0.0,
    )

    # 4. Prepare Evaluation DataLoaders (Unseen Gallery & Query)
    unseen_photo_loader = DataLoader(
        SBIRLinearDataset(unseen_photos, tfs["photo_clean"]),
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    unseen_sketch_loader = DataLoader(
        SBIRLinearDataset(unseen_sketches, tfs["sketch_clean"]),
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 5. Baseline 0 Evaluation (Epoch 0 - Frozen Zero-Shot Baseline)
    print("\n" + "=" * 70)
    print("  EVALUATING BASELINE 0 (Zero-Shot Frozen CLIP ViT-B/32)")
    print("=" * 70)
    model.eval()
    g_feats_0, g_lbls_0, g_inst_0, _ = extract_features_dataloader(
        unseen_photo_loader, model.encode_frozen_visual, device
    )
    q_feats_0, q_lbls_0, q_inst_0, _ = extract_features_dataloader(
        unseen_sketch_loader, model.encode_frozen_visual, device
    )
    base0_cat = compute_category_sbir_metrics_np(
        q_feats_0, q_lbls_0, g_feats_0, g_lbls_0, chunk_size=cfg.eval_chunk_size
    )
    base0_fg = compute_fine_grained_sbir_metrics_np(
        q_feats_0, q_inst_0, g_feats_0, g_inst_0
    )
    print(f"[BASELINE 0] ZS-SBIR:  {base0_cat}")
    print(f"[BASELINE 0] FG-SBIR:  {base0_fg}")
    print("=" * 70)

    if eval_only_baseline:
        return

    # 6. Optional Photo Feature Caching
    cached_photo_dict = None
    if cfg.use_cached_photos:
        cached_photo_dict = precompute_photo_cache(
            seen_photos, model, tfs["photo_clean"], device, cache_path=cfg.photo_cache_path
        )

    # 7. Training DataLoader & Loss Setup
    train_dataset = SeCoLoRATripletDataset(
        seen_photos, seen_sketches, tfs, cached_photo_features=cached_photo_dict
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
    )

    criterion = SeCoLoRAMultiObjectiveLoss(
        margin=cfg.triplet_margin,
        temp_ntxent=cfg.temp_ntxent,
        temp_cls=cfg.temp_cls,
        temp_consist=cfg.temp_consist,
        lambda_cls=cfg.lambda_cls,
        lambda_consist=cfg.lambda_consist,
        mining_mode=cfg.mining_mode,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=1e-7
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_map200 = 0.0

    print("\n" + "=" * 70)
    print(f"  COMMENCING TRAINING: {cfg.num_epochs} Epochs | Batch Size: {cfg.batch_size}")
    print("=" * 70)

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            s_clean = batch["sketch_clean"].to(device, non_blocking=True)
            s_pert = batch["sketch_pert"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                f_s_clean = model.encode_sketch(s_clean)
                f_s_pert = model.encode_sketch(s_pert)

                if batch["is_cached"][0]:
                    p_pos_raw = batch["photo_pos_raw"].to(device, non_blocking=True)
                    p_neg_raw = batch["photo_neg_raw"].to(device, non_blocking=True)
                    f_p_pos = model.encode_photo_from_cached(p_pos_raw)
                    f_p_neg = model.encode_photo_from_cached(p_neg_raw)
                    f_p_pert = None
                else:
                    p_pos_c = batch["photo_pos_clean"].to(device, non_blocking=True)
                    p_pos_p = batch["photo_pos_pert"].to(device, non_blocking=True)
                    p_neg_c = batch["photo_neg_clean"].to(device, non_blocking=True)
                    f_p_pos = model.encode_photo(p_pos_c)
                    f_p_neg = model.encode_photo(p_neg_c)
                    f_p_pert = model.encode_photo(p_pos_p)

                text_proto = model.adapt_text_prototypes(seen_text_proto_raw)

                loss, loss_dict = criterion(
                    f_s_clean=f_s_clean,
                    f_p_pos_clean=f_p_pos,
                    f_p_neg_clean=f_p_neg,
                    f_s_pert=f_s_pert,
                    f_p_pert=f_p_pert,
                    text_prototypes=text_proto,
                    labels=labels,
                    current_epoch=epoch,
                    total_epochs=cfg.num_epochs,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss_dict["loss_total"]

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = running_loss / max(len(train_loader), 1)

        # Validation on Unseen Test Split
        model.eval()
        g_feats, g_lbls, g_inst, _ = extract_features_dataloader(
            unseen_photo_loader, model.encode_photo, device
        )
        q_feats, q_lbls, q_inst, _ = extract_features_dataloader(
            unseen_sketch_loader, model.encode_sketch, device
        )

        zs_metrics = compute_category_sbir_metrics_np(
            q_feats, q_lbls, g_feats, g_lbls, chunk_size=cfg.eval_chunk_size
        )
        fg_metrics = compute_fine_grained_sbir_metrics_np(
            q_feats, q_inst, g_feats, g_inst
        )

        print(
            f"Epoch [{epoch:02d}/{cfg.num_epochs:02d}] ({epoch_time:.1f}s) | "
            f"Loss: {avg_loss:.4f} | "
            f"mAP@200: {zs_metrics['mAP@200']}% | P@200: {zs_metrics['P@200']}% | "
            f"mAP@all: {zs_metrics['mAP@all']}% | Acc@1: {fg_metrics['Acc@1']}%"
        )

        if zs_metrics["mAP@200"] > best_map200:
            best_map200 = zs_metrics["mAP@200"]
            ckpt_path = os.path.join(cfg.save_dir, "best_seco_lora_clip.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": {
                        "sketch_adapter": model.sketch_adapter.state_dict(),
                        "photo_adapter": model.photo_adapter.state_dict(),
                        "text_adapter": model.text_adapter.state_dict(),
                        "sketch_visual": model.sketch_visual.state_dict(),
                    },
                    "zs_metrics": zs_metrics,
                    "fg_metrics": fg_metrics,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"  >>> Best Checkpoint Saved: {ckpt_path} (mAP@200: {best_map200}%)")

    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETED. HIGHEST ZERO-SHOT mAP@200: {best_map200}%")
    print("=" * 70)
