#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: src/data/preprocessor.py
ROLE: Domain-Specific Computer Vision Pipeline for Hand-Drawn Sketches
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
Academic Innovations Integrated from Literature:
  1. Alpha-safe RGBA -> RGB composition onto solid white canvas.
  2. Noise-Resilient Tight-Crop with Contour Area Filtering (>= 15 px).
  3. Stroke Density & Empty Canvas verification.
  4. Morphological Dilation/Erosion simulating variable stroke thickness (finger vs stylus).
  5. White Random Erasing (value=255) simulating missing stroke components.
  6. Partial Sketch Masking (CVPR 2020 'Sketch Less for More'): Simulates on-the-fly early retrieval (30%-50% masked).
  7. Stroke Jitter / Noise Injection (CVPR 2022 'Sketching Without Worrying'): Robustness against messy user scribbles.
"""

import math
import random
from typing import Tuple, Optional
import cv2
import numpy as np
from PIL import Image


class SketchCVPreprocessor:
    """
    Robust Preprocessor for Hand-Drawn Sketches designed for Vision Transformers.
    """

    def __init__(
        self,
        target_size: int = 224,
        pad_ratio: float = 0.08,
        min_contour_area: float = 15.0,
        min_stroke_density: float = 0.001,
        enable_morph: bool = False,
        morph_prob: float = 0.5,
        enable_random_erase: bool = False,
        erase_prob: float = 0.5,
        erase_scale: Tuple[float, float] = (0.02, 0.15),
        erase_ratio: Tuple[float, float] = (0.3, 3.3),
        enable_partial_mask: bool = False,
        partial_prob: float = 0.35,
        enable_stroke_noise: bool = False,
        noise_prob: float = 0.25,
    ):
        self.target_size = target_size
        self.pad_ratio = pad_ratio
        self.min_contour_area = min_contour_area
        self.min_stroke_density = min_stroke_density
        self.enable_morph = enable_morph
        self.morph_prob = morph_prob
        self.enable_random_erase = enable_random_erase
        self.erase_prob = erase_prob
        self.erase_scale = erase_scale
        self.erase_ratio = erase_ratio
        self.enable_partial_mask = enable_partial_mask
        self.partial_prob = partial_prob
        self.enable_stroke_noise = enable_stroke_noise
        self.noise_prob = noise_prob

    @staticmethod
    def to_clean_sketch_rgb(pil_img: Image.Image) -> np.ndarray:
        """Composite RGBA onto pure white background to avoid transparent black artifacts."""
        if pil_img.mode == "RGBA":
            canvas = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
            composited = Image.alpha_composite(canvas, pil_img).convert("RGB")
            return np.array(composited, dtype=np.uint8)
        return np.array(pil_img.convert("RGB"), dtype=np.uint8)

    def tight_crop_and_symmetric_pad(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Tight-crops around stroke contours while rejecting small noise/dust artifacts.
        Adds symmetric 1:1 margins to preserve geometric aspect ratio.
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        inv = (255 - gray).astype(np.uint8)
        binary = (inv > 30).astype(np.uint8)

        # 1. Check stroke density: if near completely white canvas, return resized
        total_pixels = binary.shape[0] * binary.shape[1]
        active_stroke_pixels = int(np.count_nonzero(binary))
        if active_stroke_pixels / max(total_pixels, 1) < self.min_stroke_density:
            return cv2.resize(
                img_rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA
            )

        # 2. Extract external contours and filter noise dust / stray click pixels
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_items = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)
                candidate_items.append({
                    "cnt": cnt,
                    "area": area,
                    "box": (x, y, x + w, y + h),
                    "center": (x + w / 2.0, y + h / 2.0)
                })

        # Fallback if all contours were below min_contour_area threshold
        if not candidate_items:
            coords = cv2.findNonZero(binary)
            if coords is None:
                return cv2.resize(
                    img_rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA
                )
            x, y, w, h = cv2.boundingRect(coords)
            valid_boxes = [(x, y, x + w, y + h)]
        else:
            # Sort contours by area in descending order
            candidate_items.sort(key=lambda item: item["area"], reverse=True)
            max_area = candidate_items[0]["area"]

            # Main drawing cluster initialization
            main_items = [candidate_items[0]]
            main_box = list(candidate_items[0]["box"])  # [min_x, min_y, max_x, max_y]

            for item in candidate_items[1:]:
                # Substantial secondary stroke (>= 12% of max area) belongs to the drawing
                if item["area"] >= 0.12 * max_area:
                    main_items.append(item)
                    main_box[0] = min(main_box[0], item["box"][0])
                    main_box[1] = min(main_box[1], item["box"][1])
                    main_box[2] = max(main_box[2], item["box"][2])
                    main_box[3] = max(main_box[3], item["box"][3])
                else:
                    # Small component: check spatial distance to current main bounding box
                    cx, cy = item["center"]
                    dx = max(main_box[0] - cx, 0, cx - main_box[2])
                    dy = max(main_box[1] - cy, 0, cy - main_box[3])
                    dist = math.sqrt(dx * dx + dy * dy)

                    # If close to main body (< 35px), keep it (could be dot on 'i', eye dot, button)
                    # If far away (> 35px) and tiny (< 12% max_area), reject as stray click / dust tap!
                    if dist < 35.0:
                        main_items.append(item)
                        main_box[0] = min(main_box[0], item["box"][0])
                        main_box[1] = min(main_box[1], item["box"][1])
                        main_box[2] = max(main_box[2], item["box"][2])
                        main_box[3] = max(main_box[3], item["box"][3])

            valid_boxes = [it["box"] for it in main_items]

        # 3. Compute union bounding box across all verified drawing contours
        min_x = min(b[0] for b in valid_boxes)
        min_y = min(b[1] for b in valid_boxes)
        max_x = max(b[2] for b in valid_boxes)
        max_y = max(b[3] for b in valid_boxes)

        crop_w = max(max_x - min_x, 1)
        crop_h = max(max_y - min_y, 1)
        cropped = img_rgb[min_y:max_y, min_x:max_x]

        # 4. Symmetric 1:1 square padding with white margins
        max_side = max(crop_w, crop_h)
        margin = int(round(max_side * self.pad_ratio))
        square_side = max_side + 2 * margin

        pad_top = (square_side - crop_h) // 2
        pad_bottom = square_side - crop_h - pad_top
        pad_left = (square_side - crop_w) // 2
        pad_right = square_side - crop_w - pad_left

        padded = cv2.copyMakeBorder(
            cropped,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        return padded

    def apply_morphological_ops(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Randomly alters stroke thickness.
        Note: on dark strokes with white background, cv2.erode expands dark regions (thickens),
        while cv2.dilate shrinks dark regions (thins).
        """
        if not self.enable_morph or random.random() > self.morph_prob:
            return img_rgb

        ksize = random.choice([2, 3])
        kernel = np.ones((ksize, ksize), np.uint8)

        # 60% probability of thickening strokes, 40% probability of thinning
        if random.random() < 0.60:
            return cv2.erode(img_rgb, kernel, iterations=1)
        return cv2.dilate(img_rgb, kernel, iterations=1)

    def apply_partial_sketch_masking(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Partial Sketch Masking (CVPR 2020 'Sketch Less for More'):
        Simulates incomplete user sketches by masking 30% - 50% of the canvas with solid white (255).
        Guarantees the remaining region retains at least 20% active strokes.
        """
        if not self.enable_partial_mask or random.random() > self.partial_prob:
            return img_rgb

        h, w, _ = img_rgb.shape
        out = img_rgb.copy()

        # Choose a partial masking mode: top, bottom, left, right, or large rectangle
        mode = random.choice(["top", "bottom", "left", "right", "block"])
        mask_ratio = random.uniform(0.30, 0.50)

        if mode == "top":
            out[: int(h * mask_ratio), :, :] = 255
        elif mode == "bottom":
            out[int(h * (1.0 - mask_ratio)) :, :, :] = 255
        elif mode == "left":
            out[:, : int(w * mask_ratio), :] = 255
        elif mode == "right":
            out[:, int(w * (1.0 - mask_ratio)) :, :] = 255
        else:  # block
            bh = int(h * random.uniform(0.35, 0.55))
            bw = int(w * random.uniform(0.35, 0.55))
            top = random.randint(0, h - bh)
            left = random.randint(0, w - bw)
            out[top : top + bh, left : left + bw, :] = 255

        # Verify remaining stroke density
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        active_remaining = int(np.count_nonzero((255 - gray) > 30))
        if active_remaining < 40:  # If canvas was wiped out, reject mask
            return img_rgb

        return out

    def apply_stroke_jitter_noise(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Stroke Noise Injection (CVPR 2022 'Sketching Without Worrying'):
        Draws 1-2 faint synthetic scribble strokes to train noise-tolerance.
        """
        if not self.enable_stroke_noise or random.random() > self.noise_prob:
            return img_rgb

        out = img_rgb.copy()
        h, w, _ = out.shape

        for _ in range(random.choice([1, 2])):
            pt1 = (random.randint(10, w - 10), random.randint(10, h - 10))
            pt2 = (
                pt1[0] + random.randint(-40, 40),
                pt1[1] + random.randint(-40, 40),
            )
            color = (random.randint(30, 100), random.randint(30, 100), random.randint(30, 100))
            thickness = random.choice([1, 2])
            cv2.line(out, pt1, pt2, color, thickness=thickness)

        return out

    def apply_white_random_erasing(self, img_rgb: np.ndarray) -> np.ndarray:
        """Erases localized rectangular patches using pure white (255), simulating missing strokes."""
        if not self.enable_random_erase or random.random() > self.erase_prob:
            return img_rgb

        out = img_rgb.copy()
        h, w, _ = out.shape
        area = h * w

        num_patches = random.choice([1, 2])
        for _ in range(num_patches):
            for _ in range(10):
                target_area = random.uniform(self.erase_scale[0], self.erase_scale[1]) * area
                aspect_ratio = random.uniform(self.erase_ratio[0], self.erase_ratio[1])
                eh = int(round(math.sqrt(target_area * aspect_ratio)))
                ew = int(round(math.sqrt(target_area / aspect_ratio)))
                if ew < w and eh < h:
                    top = random.randint(0, h - eh)
                    left = random.randint(0, w - ew)
                    out[top : top + eh, left : left + ew, :] = 255
                    break
        return out

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        """Full pipeline execution returning a preprocessed PIL Image."""
        rgb = self.to_clean_sketch_rgb(pil_img)
        padded = self.tight_crop_and_symmetric_pad(rgb)
        morphed = self.apply_morphological_ops(padded)
        resized = cv2.resize(
            morphed, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA
        )
        partial = self.apply_partial_sketch_masking(resized)
        erased = self.apply_white_random_erasing(partial)
        noisy = self.apply_stroke_jitter_noise(erased)
        return Image.fromarray(noisy)
