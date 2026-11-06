#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: src/data/transforms.py
ROLE: Transformation Pipelines for Sketch & Photo (Clean & Perturbed)
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
"""

from typing import Dict
import torch
import torchvision.transforms as T
from PIL import Image

from src.config import CLIP_MEAN, CLIP_STD
from src.data.preprocessor import SketchCVPreprocessor


def build_transforms(
    image_size: int = 224,
    pad_ratio: float = 0.08,
    min_contour_area: float = 15.0,
    morph_prob: float = 0.6,
    erase_prob: float = 0.5,
    partial_prob: float = 0.35,
    noise_prob: float = 0.25,
) -> Dict[str, T.Compose]:
    """
    Constructs clean and perturbed transformations for dual-stream training.
    Integrates Partial Sketching (CVPR 2020), Noise-Tolerance (CVPR 2022), and Wrist-Tilt Invariance.
    """
    normalize = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

    # 1. Clean Sketch Transform (Evaluation & Primary retrieval stream)
    sketch_clean = T.Compose([
        SketchCVPreprocessor(
            target_size=image_size,
            pad_ratio=pad_ratio,
            min_contour_area=min_contour_area,
            enable_morph=False,
            enable_random_erase=False,
            enable_partial_mask=False,
            enable_stroke_noise=False,
        ),
        T.ToTensor(),
        normalize,
    ])

    # 2. Perturbed Sketch Transform (Consistency regularization stream)
    sketch_pert = T.Compose([
        SketchCVPreprocessor(
            target_size=image_size,
            pad_ratio=pad_ratio,
            min_contour_area=min_contour_area,
            enable_morph=True,
            morph_prob=morph_prob,
            enable_random_erase=True,
            erase_prob=erase_prob,
            enable_partial_mask=True,
            partial_prob=partial_prob,
            enable_stroke_noise=True,
            noise_prob=noise_prob,
        ),
        T.RandomRotation(degrees=(-10, 10), fill=255),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalize,
    ])

    # 3. Clean Photo Transform (Evaluation & Gallery encoding)
    photo_clean = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalize,
    ])

    # 4. Perturbed Photo Transform (Training consistency stream)
    photo_pert = T.Compose([
        T.RandomResizedCrop(
            image_size, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.4, scale=(0.02, 0.15), value=0),
    ])

    return {
        "sketch_clean": sketch_clean,
        "sketch_pert": sketch_pert,
        "photo_clean": photo_clean,
        "photo_pert": photo_pert,
    }
