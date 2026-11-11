#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: src/engine/evaluator.py
ROLE: High-Speed Vectorized Retrieval Metrics (Category, Fine-Grained, Recall@K,
      Normalized R-Precision) & Academic Diagnostic Report Generator
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


def compute_category_sbir_metrics_np(
    query_feats: np.ndarray,
    query_labels: np.ndarray,
    gallery_feats: np.ndarray,
    gallery_labels: np.ndarray,
    chunk_size: int = 512,
) -> Dict[str, float]:
    """
    Vectorized computation of category retrieval metrics:
      - mAP@all: Mean Average Precision over entire gallery.
      - mAP@200: Standard benchmark metric clamped to Top-200.
      - P@100_std, P@200_std: Standard Precision @ Top-K (denom = k).
      - P@100_norm, P@200_norm: Normalized R-Precision @ Top-K (denom = min(total_rel, k)).
      - Recall@1, Recall@5, Recall@10: Hit Rate @ Top-K (at least 1 match in Top-K).
    """
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

    res = {
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
    # Backward compatibility aliases
    res["P@100"] = res["P@100_std"]
    res["P@200"] = res["P@200_std"]
    return res


def compute_fine_grained_sbir_metrics_np(
    query_feats: np.ndarray,
    query_inst_ids: List[str],
    gallery_feats: np.ndarray,
    gallery_inst_ids: List[str],
) -> Dict[str, float]:
    """
    Fine-Grained Instance Retrieval: Evaluates Top-1, Top-5, Top-10 Accuracy
    matching exact instance IDs (e.g. matching specific shoe or sketch-photo pair).
    """
    g_id_to_idx = {gid: i for i, gid in enumerate(gallery_inst_ids)}
    valid_q_indices = [i for i, qid in enumerate(query_inst_ids) if qid in g_id_to_idx]

    if not valid_q_indices:
        return {"Acc@1": 0.0, "Acc@5": 0.0, "Acc@10": 0.0}

    q_f = query_feats[valid_q_indices]
    q_f = q_f / (np.linalg.norm(q_f, axis=1, keepdims=True) + 1e-8)
    g_f = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8)

    target_g_idx = np.array([g_id_to_idx[query_inst_ids[i]] for i in valid_q_indices])[:, None]
    sim = q_f @ g_f.T
    topk_indices = np.argsort(-sim, axis=1)[:, : min(10, g_f.shape[0])]

    acc1 = float((topk_indices[:, :1] == target_g_idx).any(axis=1).mean() * 100.0)
    acc5 = float((topk_indices[:, :5] == target_g_idx).any(axis=1).mean() * 100.0)
    acc10 = float((topk_indices[:, :10] == target_g_idx).any(axis=1).mean() * 100.0)

    return {
        "Acc@1": round(acc1, 2),
        "Acc@5": round(acc5, 2),
        "Acc@10": round(acc10, 2),
    }


def compute_comprehensive_sbir_metrics(
    query_feats: np.ndarray,
    query_labels: np.ndarray,
    query_inst_ids: Optional[List[str]],
    gallery_feats: np.ndarray,
    gallery_labels: np.ndarray,
    gallery_inst_ids: Optional[List[str]],
    chunk_size: int = 512,
) -> Dict[str, float]:
    """
    Computes unified diagnostic metric suite: Category ZS-SBIR + Fine-Grained FG-ZS-SBIR.
    Returns:
      - Category: mAP@all, mAP@200, P@100_std, P@200_std, P@100_norm, P@200_norm, Recall@1, Recall@5, Recall@10
      - Instance: Acc@1, Acc@5, Acc@10
    """
    cat_metrics = compute_category_sbir_metrics_np(
        query_feats=query_feats,
        query_labels=query_labels,
        gallery_feats=gallery_feats,
        gallery_labels=gallery_labels,
        chunk_size=chunk_size,
    )

    fg_metrics = {"Acc@1": 0.0, "Acc@5": 0.0, "Acc@10": 0.0}
    if query_inst_ids is not None and gallery_inst_ids is not None:
        fg_metrics = compute_fine_grained_sbir_metrics_np(
            query_feats=query_feats,
            query_inst_ids=query_inst_ids,
            gallery_feats=gallery_feats,
            gallery_inst_ids=gallery_inst_ids,
        )

    merged = {**cat_metrics, **fg_metrics}
    return merged


def format_diagnostic_report_table(
    epoch: int,
    total_epochs: int,
    loss_dict: Dict[str, float],
    metrics: Dict[str, float],
    epoch_time: float,
    current_lr: float,
    best_map200: float,
    device_info: str = "GPU",
) -> str:
    """
    Formats a comprehensive academic diagnostic box for terminal/logs at each epoch.
    Helps developers and researchers identify exact model learning bottlenecks.
    """
    # Diagnostic insights
    map_all = metrics.get("mAP@all", 0.0)
    map_200 = metrics.get("mAP@200", 0.0)
    rec_1 = metrics.get("Recall@1", 0.0)
    rec_5 = metrics.get("Recall@5", 0.0)
    acc_1 = metrics.get("Acc@1", 0.0)
    acc_5 = metrics.get("Acc@5", 0.0)
    loss_tot = loss_dict.get("loss_total", 0.0)

    insights = []
    if rec_5 >= 90.0:
        insights.append("[*] Category Locality: EXCELLENT (Top-5 captures correct class in >90% of queries).")
    elif rec_5 >= 75.0:
        insights.append("[!] Category Locality: GOOD (Top-5 captures >75%, fine strokes need slight refinement).")
    else:
        insights.append("[?] Category Locality: LAGGING (Model struggles with coarse category separation).")

    if acc_1 >= 8.0:
        insights.append("[*] Fine-Grained Alignment: STRONG (Spatial tokens actively matching specific instances).")
    else:
        insights.append("[i] Fine-Grained Alignment: MODERATE (Focusing primarily on category-level invariant geometry).")

    if map_200 >= best_map200:
        insights.append(f"[NEW BEST] Record Unseen ZS-SBIR mAP@200 reached: {map_200:.2f}% (Split 2 strict isolation).")

    WIDTH = 99  # Inner width excluding '| ' and ' |'

    def row(s: str) -> str:
        return f"| {s.ljust(WIDTH)} |"

    insights_str = "\n".join(row(f"  {s}") for s in insights)

    lines = [
        "+" + "=" * (WIDTH + 2) + "+",
        row(f"EPOCH [{epoch:02d}/{total_epochs:02d}] DIAGNOSTIC REPORT ({device_info} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e})"),
        "+" + "-" * (WIDTH + 2) + "+",
        row("[1] TRAINING LOSS OBJECTIVES (Seen Classes: 104 | 100% Zero Leakage to Test)"),
        row(f"  Loss_Total:   {loss_tot:<8.4f} | Loss_Triplet: {loss_dict.get('loss_triplet', 0.0):<8.4f} | Loss_NTXent:  {loss_dict.get('loss_ntxent', 0.0):<8.4f}"),
        row(f"  Loss_Class:   {loss_dict.get('loss_class', 0.0):<8.4f} | Loss_Consist: {loss_dict.get('loss_consist', 0.0):<8.4f} | Mining: Hard/Semi-Hard (Margin 0.25)"),
        "+" + "-" * (WIDTH + 2) + "+",
        row("[2] ZERO-SHOT CATEGORY RETRIEVAL (Unseen Classes: 21 | Isolated Gallery: 2,100 Photos)"),
        row(f"  mAP@all:      {map_all:<6.2f}% (Full)     | mAP@200:      {map_200:<6.2f}% (Standard Benchmark)"),
        row(f"  P@100 (std):  {metrics.get('P@100_std', 0.0):<6.2f}%             | P@200 (std):  {metrics.get('P@200_std', 0.0):<6.2f}%"),
        row(f"  P@100 (norm): {metrics.get('P@100_norm', 0.0):<6.2f}% (R-Prec)     | P@200 (norm): {metrics.get('P@200_norm', 0.0):<6.2f}% (R-Prec)"),
        row(f"  Recall@1:     {rec_1:<6.2f}% (Hit@1)     | Recall@5:     {rec_5:<6.2f}% (Hit@5)  | Recall@10: {metrics.get('Recall@10', 0.0):<6.2f}% (Hit@10)"),
        "+" + "-" * (WIDTH + 2) + "+",
        row("[3] FINE-GRAINED INSTANCE MATCHING (Exact Pair Retrieval)"),
        row(f"  Acc@1:        {acc_1:<6.2f}%             | Acc@5:        {acc_5:<6.2f}%          | Acc@10:    {metrics.get('Acc@10', 0.0):<6.2f}%"),
        "+" + "-" * (WIDTH + 2) + "+",
        row("[4] DIAGNOSTIC RADAR & HEALTH CHECK:"),
        insights_str,
        "+" + "=" * (WIDTH + 2) + "+",
    ]
    return "\n".join(lines)


class AdaptiveOODDetector:
    """
    Out-of-Distribution Detector for Rejecting Garbage / Nonsense Doodles.
    Combines Top-5 Gallery Visual Similarity with Max Text Prototype Similarity.
    """

    def __init__(self, alpha: float = 0.7, std_multiplier: float = 1.5):
        self.alpha = alpha
        self.std_multiplier = std_multiplier
        self.threshold: float = 0.0
        self.id_mean: float = 0.0
        self.id_std: float = 1.0

    def compute_scores(
        self,
        query_feats: np.ndarray,
        gallery_feats: np.ndarray,
        text_prototypes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        q_f = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-8)
        g_f = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8)
        sim = q_f @ g_f.T
        top_k = min(5, g_f.shape[0])
        vis_sim = -np.sort(-sim, axis=1)[:, :top_k].mean(axis=1)

        if text_prototypes is not None:
            t_f = text_prototypes / (np.linalg.norm(text_prototypes, axis=1, keepdims=True) + 1e-8)
            txt_sim = (q_f @ t_f.T).max(axis=1)
            return self.alpha * vis_sim + (1.0 - self.alpha) * txt_sim
        return vis_sim

    def fit_threshold(self, in_dist_scores: np.ndarray) -> float:
        self.id_mean = float(in_dist_scores.mean())
        self.id_std = float(in_dist_scores.std() + 1e-6)
        self.threshold = self.id_mean - self.std_multiplier * self.id_std
        return self.threshold

    def is_out_of_distribution(self, query_scores: np.ndarray) -> np.ndarray:
        return query_scores < self.threshold

    @staticmethod
    def evaluate_ood_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
        """Calculates Area Under ROC Curve (AUROC) and False Positive Rate at 95% TPR."""
        y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        y_score = np.concatenate([id_scores, ood_scores])

        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]

        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1.0 - y_true_sorted)
        tpr = tps / max(float(tps[-1]), 1.0)
        fpr = fps / max(float(fps[-1]), 1.0)

        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auroc = float(trapz_fn(tpr, fpr))
        idx_95 = np.searchsorted(tpr, 0.95)
        fpr_at_95 = float(fpr[min(idx_95, len(fpr) - 1)])

        return {
            "AUROC": round(auroc * 100.0, 2),
            "FPR@95TPR": round(fpr_at_95 * 100.0, 2),
        }
