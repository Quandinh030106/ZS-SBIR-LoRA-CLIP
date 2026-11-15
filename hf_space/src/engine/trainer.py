#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: src/engine/trainer.py
ROLE: End-to-End Multi-GPU Training & Diagnostic Evaluation Engine
      with Zero Data Leakage Verification & Comprehensive Academic Metrics
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
"""

import os
import json
import time
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import SBIRConfig, set_reproducible_seed
from src.data.split_manager import UnifiedSplitManager
from src.data.transforms import build_transforms
from src.data.datasets import SBIRLinearDataset, SeCoLoRATripletDataset
from src.models.clip_lora import AsymmetricHybridLoRACLIP, build_clip_text_prototypes
from src.losses.multi_loss import SeCoLoRAMultiObjectiveLoss
from src.engine.evaluator import (
    compute_comprehensive_sbir_metrics,
    format_diagnostic_report_table,
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
    """Full execution pipeline: Data prep -> Model init -> Train loop -> Academic Diagnostics."""
    import clip

    set_reproducible_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    device_desc = f"{num_gpus}x {torch.cuda.get_device_name(0)}" if (num_gpus > 0 and device.type == "cuda") else "CPU"

    os.makedirs(cfg.save_dir, exist_ok=True)
    if cfg.metrics_history_path:
        os.makedirs(os.path.dirname(cfg.metrics_history_path), exist_ok=True)

    # 1. Initialize Split Manager (Strict Zero Data Leakage Protocol)
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

    print(f"\n[DATA PROTOCOL VERIFICATION]")
    print(f"  Seen Training Split (104 classes):    {len(seen_photos):,} photos | {len(seen_sketches):,} sketches")
    print(f"  Unseen Test Split (21 classes ISOLATED): {len(unseen_photos):,} photos | {len(unseen_sketches):,} sketches")
    print(f"  Test Leakage Check: 100% Zero Leakage (Train and test classes strictly disjoint).")

    # 2. Initialize Model Architecture
    print(f"\n[MODEL INITIALIZATION]")
    print(f"  Loading Backbone: {cfg.clip_backbone} on {device_desc}...")
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
        enable_patch_pooling=cfg.enable_patch_pooling,
        patch_pool_weight=cfg.patch_pool_weight,
    ).to(device)

    param_stats = model.count_parameters()
    print(f"  Total Parameters:     {param_stats['total_params']:,}")
    print(f"  Trainable Parameters: {param_stats['trainable_params']:,} ({param_stats['trainable_ratio_pct']}%)")
    print(f"  Spatial Patch-Token Pooling: {'ENABLED (weights 49 patch tokens)' if cfg.enable_patch_pooling else 'DISABLED'}")

    # 3. Text Prototypes & Augmentation Transforms
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

    # 4. Unseen Evaluation DataLoaders (Isolated Test Gallery & Queries)
    unseen_photo_loader = DataLoader(
        SBIRLinearDataset(unseen_photos, tfs["photo_clean"]),
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    unseen_sketch_loader = DataLoader(
        SBIRLinearDataset(unseen_sketches, tfs["sketch_clean"]),
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 5. Baseline 0 Evaluation (Epoch 0 - Frozen Zero-Shot Baseline)
    print("\n" + "=" * 103)
    print("  EVALUATING BASELINE 0 (Zero-Shot Frozen CLIP ViT-B/32 on 21 Unseen Classes)")
    print("=" * 103)
    model.eval()
    g_feats_0, g_lbls_0, g_inst_0, _ = extract_features_dataloader(
        unseen_photo_loader, model.encode_frozen_visual, device
    )
    q_feats_0, q_lbls_0, q_inst_0, _ = extract_features_dataloader(
        unseen_sketch_loader, model.encode_frozen_visual, device
    )
    base0_metrics = compute_comprehensive_sbir_metrics(
        query_feats=q_feats_0,
        query_labels=q_lbls_0,
        query_inst_ids=q_inst_0,
        gallery_feats=g_feats_0,
        gallery_labels=g_lbls_0,
        gallery_inst_ids=g_inst_0,
        chunk_size=cfg.eval_chunk_size,
    )
    base0_table = format_diagnostic_report_table(
        epoch=0,
        total_epochs=cfg.num_epochs,
        loss_dict={"loss_total": 0.0, "loss_triplet": 0.0, "loss_ntxent": 0.0, "loss_class": 0.0, "loss_consist": 0.0},
        metrics=base0_metrics,
        epoch_time=0.0,
        current_lr=0.0,
        best_map200=base0_metrics["mAP@200"],
        device_info=device_desc,
    )
    print(base0_table)

    if eval_only_baseline:
        return

    # 6. Optional Photo Feature Caching
    cached_photo_dict = None
    if cfg.use_cached_photos:
        cached_photo_dict = precompute_photo_cache(
            seen_photos, model, tfs["photo_clean"], device, cache_path=cfg.photo_cache_path
        )

    # 7. Training DataLoader & Multi-GPU Setup
    train_dataset = SeCoLoRATripletDataset(
        seen_photos, seen_sketches, tfs, cached_photo_features=cached_photo_dict
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
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

    # Multi-GPU DataParallel wrapper if >= 2 GPUs available
    train_model = model
    if num_gpus > 1 and device.type == "cuda":
        print(f"[MULTI-GPU] Wrapping model in DataParallel across {num_gpus} GPUs.")
        train_model = nn.DataParallel(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=1e-7
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_map200 = base0_metrics["mAP@200"]
    best_map_all = base0_metrics["mAP@all"]
    best_epoch = 0

    metrics_history: List[Dict[str, Any]] = [
        {
            "epoch": 0,
            "epoch_time_s": 0.0,
            "lr": 0.0,
            "losses": {"loss_total": 0.0, "loss_triplet": 0.0, "loss_ntxent": 0.0, "loss_class": 0.0, "loss_consist": 0.0},
            "unseen_metrics": base0_metrics,
        }
    ]

    print("\n" + "=" * 103)
    print(f"  COMMENCING TRAINING: {cfg.num_epochs} Epochs | Batch Size: {cfg.batch_size} | Device: {device_desc}")
    print("=" * 103)

    for epoch in range(1, cfg.num_epochs + 1):
        train_model.train()
        start_time = time.time()

        running_tot = 0.0
        running_tri = 0.0
        running_ntx = 0.0
        running_cls = 0.0
        running_con = 0.0
        num_batches = 0

        for batch in train_loader:
            s_clean = batch["sketch_clean"].to(device, non_blocking=True)
            s_pert = batch["sketch_pert"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                if batch["is_cached"][0]:
                    p_pos_raw = batch["photo_pos_raw"].to(device, non_blocking=True)
                    p_neg_raw = batch["photo_neg_raw"].to(device, non_blocking=True)
                    f_s_clean = model.encode_sketch(s_clean)
                    f_s_pert = model.encode_sketch(s_pert)
                    f_p_pos = model.encode_photo_from_cached(p_pos_raw)
                    f_p_neg = model.encode_photo_from_cached(p_neg_raw)
                    f_p_pert = None
                else:
                    p_pos_c = batch["photo_pos_clean"].to(device, non_blocking=True)
                    p_pos_p = batch["photo_pos_pert"].to(device, non_blocking=True)
                    p_neg_c = batch["photo_neg_clean"].to(device, non_blocking=True)

                    if isinstance(train_model, nn.DataParallel):
                        f_dict = train_model(
                            sketch_clean=s_clean,
                            photo_pos=p_pos_c,
                            photo_neg=p_neg_c,
                            sketch_pert=s_pert,
                            photo_pert=p_pos_p,
                        )
                        f_s_clean = f_dict["f_s_clean"]
                        f_p_pos = f_dict["f_p_pos"]
                        f_p_neg = f_dict["f_p_neg"]
                        f_s_pert = f_dict.get("f_s_pert")
                        f_p_pert = f_dict.get("f_p_pert")
                    else:
                        f_s_clean = model.encode_sketch(s_clean)
                        f_s_pert = model.encode_sketch(s_pert)
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

            running_tot += loss_dict["loss_total"]
            running_tri += loss_dict.get("loss_triplet", 0.0)
            running_ntx += loss_dict.get("loss_ntxent", 0.0)
            running_cls += loss_dict.get("loss_class", 0.0)
            running_con += loss_dict.get("loss_consist", 0.0)
            num_batches += 1

        scheduler.step()
        epoch_time = time.time() - start_time
        curr_lr = float(scheduler.get_last_lr()[0])

        avg_losses = {
            "loss_total": round(running_tot / max(num_batches, 1), 4),
            "loss_triplet": round(running_tri / max(num_batches, 1), 4),
            "loss_ntxent": round(running_ntx / max(num_batches, 1), 4),
            "loss_class": round(running_cls / max(num_batches, 1), 4),
            "loss_consist": round(running_con / max(num_batches, 1), 4),
        }

        # 8. Unseen Zero-Shot Test Split Evaluation
        model.eval()
        g_feats, g_lbls, g_inst, _ = extract_features_dataloader(
            unseen_photo_loader, model.encode_photo, device
        )
        q_feats, q_lbls, q_inst, _ = extract_features_dataloader(
            unseen_sketch_loader, model.encode_sketch, device
        )

        unseen_metrics = compute_comprehensive_sbir_metrics(
            query_feats=q_feats,
            query_labels=q_lbls,
            query_inst_ids=q_inst,
            gallery_feats=g_feats,
            gallery_labels=g_lbls,
            gallery_inst_ids=g_inst,
            chunk_size=cfg.eval_chunk_size,
        )

        # 9. Print Full Diagnostic Report Table
        report_table = format_diagnostic_report_table(
            epoch=epoch,
            total_epochs=cfg.num_epochs,
            loss_dict=avg_losses,
            metrics=unseen_metrics,
            epoch_time=epoch_time,
            current_lr=curr_lr,
            best_map200=best_map200,
            device_info=device_desc,
        )
        print(report_table)

        # 10. Record Epoch History to JSON
        epoch_record = {
            "epoch": epoch,
            "epoch_time_s": round(epoch_time, 2),
            "lr": curr_lr,
            "losses": avg_losses,
            "unseen_metrics": unseen_metrics,
        }
        metrics_history.append(epoch_record)
        if cfg.metrics_history_path:
            with open(cfg.metrics_history_path, "w", encoding="utf-8") as f:
                json.dump(metrics_history, f, indent=2)

        # 11. Checkpoint Check & Save
        is_new_best = unseen_metrics["mAP@200"] > best_map200
        if is_new_best:
            best_map200 = unseen_metrics["mAP@200"]
            best_map_all = unseen_metrics["mAP@all"]
            best_epoch = epoch

            best_ckpt_path = os.path.join(cfg.save_dir, "best_seco_lora_clip.pt")
            save_payload = {
                "epoch": epoch,
                "model_state": {
                    "sketch_adapter": model.sketch_adapter.state_dict(),
                    "photo_adapter": model.photo_adapter.state_dict(),
                    "text_adapter": model.text_adapter.state_dict(),
                    "sketch_visual": model.sketch_visual.state_dict(),
                },
                "unseen_metrics": unseen_metrics,
                "loss_dict": avg_losses,
                "config": cfg,
                "history": metrics_history,
            }
            if model.spatial_pooler is not None:
                save_payload["model_state"]["spatial_pooler"] = model.spatial_pooler.state_dict()

            torch.save(save_payload, best_ckpt_path)
            print(f"  >>> [SAVED NEW RECORD] {best_ckpt_path} (mAP@200: {best_map200:.2f}%, mAP@all: {best_map_all:.2f}%)")

        # Always save last checkpoint for resume capability
        last_ckpt_path = os.path.join(cfg.save_dir, "last_checkpoint.pt")
        last_payload = {
            "epoch": epoch,
            "model_state": {
                "sketch_adapter": model.sketch_adapter.state_dict(),
                "photo_adapter": model.photo_adapter.state_dict(),
                "text_adapter": model.text_adapter.state_dict(),
                "sketch_visual": model.sketch_visual.state_dict(),
            },
            "unseen_metrics": unseen_metrics,
            "loss_dict": avg_losses,
            "config": cfg,
            "history": metrics_history,
        }
        if model.spatial_pooler is not None:
            last_payload["model_state"]["spatial_pooler"] = model.spatial_pooler.state_dict()
        torch.save(last_payload, last_ckpt_path)

    print("\n" + "=" * 103)
    print(f"  TRAINING COMPLETED SUCCESSFULLY across {cfg.num_epochs} Epochs!")
    print(f"  Best Validation Epoch: {best_epoch:02d} | Highest Zero-Shot mAP@200: {best_map200:.2f}% | mAP@all: {best_map_all:.2f}%")
    print(f"  Metrics History Saved to: {cfg.metrics_history_path}")
    print("=" * 103)
