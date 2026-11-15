#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: src/utils/visualization.py
ROLE: Academic Visualization Utilities for Presentations & Paper Reporting
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
"""

import os
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_retrieval_results(
    sketch_path: str,
    retrieved_paths: List[str],
    retrieved_labels: List[int],
    query_label: int,
    top_k: int = 5,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots a query sketch alongside its top-k retrieved photos.
    Borders are colored green (correct class) or red (incorrect class).
    """
    top_k = min(top_k, len(retrieved_paths))
    fig, axes = plt.subplots(1, top_k + 1, figsize=((top_k + 1) * 3, 3.5))

    # Plot Sketch
    s_img = Image.open(sketch_path).convert("RGB")
    axes[0].imshow(s_img)
    axes[0].set_title(f"Query Sketch\n(Class {query_label})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Plot Retrieved Photos
    for i in range(top_k):
        ax = axes[i + 1]
        p_img = Image.open(retrieved_paths[i]).convert("RGB")
        ax.imshow(p_img)
        is_match = retrieved_labels[i] == query_label
        border_color = "green" if is_match else "red"
        title = f"Top {i+1}: {'Match' if is_match else 'Miss'}\n(Class {retrieved_labels[i]})"

        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3.5)

        ax.set_title(title, fontsize=10, color=border_color, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[VIZ] Retrieval visualization saved to: {save_path}")
    plt.close()


def plot_ood_distribution(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    threshold: float,
    save_path: Optional[str] = None,
) -> None:
    """Plots score distributions for In-Distribution vs OOD samples with decision boundary."""
    plt.figure(figsize=(8, 5))
    plt.hist(id_scores, bins=40, alpha=0.6, color="blue", label="In-Distribution (Valid Sketches)", density=True)
    plt.hist(ood_scores, bins=40, alpha=0.6, color="red", label="Out-of-Distribution (Garbage Doodles)", density=True)
    plt.axvline(threshold, color="black", linestyle="--", linewidth=2.0, label=f"Threshold tau = {threshold:.3f}")

    plt.title("Adaptive Out-of-Distribution (OOD) Confidence Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Confidence Score S(q)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[VIZ] OOD distribution plot saved to: {save_path}")
    plt.close()
