#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: src/data/split_manager.py
ROLE: Universal Dataset Scanner, Split Manager & Zero-Copy Virtual Merger
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
Solves Technical Risk 1:
  - Transparently handles both standard 2-folder layouts ('photo' & 'sketch')
    and official Sangkloy / Sketchy 256x256 layouts ('photo' & 'splitted_sketches'
    with train/test partitions).
  - Merges sketches across train/test splits per category without duplicating files.
  - Implements strict Zero-Shot Seen/Unseen partitioning without data leakage.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional

from src.config import (
    SKETCHY_SPLIT2_UNSEEN_21,
    SKETCHY_SPLIT1_UNSEEN_25,
    TUBERLIN_UNSEEN_30,
)


class UnifiedSplitManager:
    """
    Universal Split & Dataset Directory Manager.
    Supports Sketchy, TU-Berlin Extended, and Fine-Grained datasets (QMUL-Shoe).
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str = "sketchy",
        photo_subdir: str = "photo",
        sketch_subdir: str = "splitted_sketches",
        canonical_tx: str = "tx_000000000000",
        split_protocol: str = "split2",
    ):
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name.lower()
        self.split_protocol = split_protocol
        self.canonical_tx = canonical_tx

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root.resolve()}")

        # 1. Resolve photo directory
        self.photo_dir = self._resolve_photo_dir(photo_subdir)

        # 2. Resolve sketch directories (may be a single directory or a list of train/test dirs)
        self.sketch_dirs = self._resolve_sketch_dirs(sketch_subdir)

        # 3. Discover all available classes from photo directory
        self.all_classes = sorted([
            d.name for d in self.photo_dir.iterdir() if d.is_dir()
        ])
        if not self.all_classes:
            raise ValueError(f"No class folders found in {self.photo_dir}")

        # 4. Determine Unseen & Seen class sets
        self.unseen_classes = self._determine_unseen_classes()
        self.seen_classes = sorted([
            c for c in self.all_classes if c not in set(self.unseen_classes)
        ])

        # 5. Build class index mappings
        self.seen_cls2idx = {c: i for i, c in enumerate(self.seen_classes)}
        self.unseen_cls2idx = {c: i for i, c in enumerate(self.unseen_classes)}
        self.global_cls2idx = {c: i for i, c in enumerate(self.all_classes)}

    def _resolve_photo_dir(self, subdir_name: str) -> Path:
        """Finds the directory containing class subfolders for photos."""
        candidate = self.data_root / subdir_name
        if not candidate.exists():
            for alt_name in ["photo", "ImageResized", "images", "image", "photos"]:
                alt = self.data_root / alt_name
                if alt.exists():
                    candidate = alt
                    break
        if not candidate.exists():
            raise FileNotFoundError(f"Photo directory not found at: {self.data_root / subdir_name}")

        # Check for nested transform folder e.g., photo/tx_000000000000
        tx_candidate = candidate / self.canonical_tx
        if tx_candidate.exists() and any(d.is_dir() for d in tx_candidate.iterdir()):
            return tx_candidate

        # Deep search if candidate only has subfolders
        subdirs = [d for d in candidate.iterdir() if d.is_dir()]
        if len(subdirs) < 10:
            for sd in subdirs:
                if (sd / self.canonical_tx).exists():
                    return sd / self.canonical_tx
                inner_class_dirs = [d for d in sd.iterdir() if d.is_dir()]
                if len(inner_class_dirs) >= 10:
                    return sd

        return candidate

    def _resolve_sketch_dirs(self, subdir_name: str) -> List[Path]:
        """
        Discovers all sketch directories.
        If 'splitted_sketches' is detected, returns BOTH 'train' and 'test' dirs
        to enable complete zero-shot aggregation.
        """
        candidate = self.data_root / subdir_name
        dirs: List[Path] = []

        if not candidate.exists():
            for alt_name in ["splitted_sketches", "sketch", "sketches", "png"]:
                alt = self.data_root / alt_name
                if alt.exists():
                    candidate = alt
                    break
        if not candidate.exists():
            raise FileNotFoundError(f"Sketch directory not found at: {self.data_root / subdir_name}")

        # Case 1: splitted_sketches with train and test
        train_dir = candidate / "train" / self.canonical_tx
        test_dir = candidate / "test" / self.canonical_tx
        if train_dir.exists() and test_dir.exists():
            dirs.append(train_dir)
            dirs.append(test_dir)
            return dirs

        train_dir_alt = candidate / "train"
        test_dir_alt = candidate / "test"
        if train_dir_alt.exists() and test_dir_alt.exists():
            dirs.append(train_dir_alt)
            dirs.append(test_dir_alt)
            return dirs

        # Case 2: standard single sketch folder (or nested with canonical_tx)
        tx_candidate = candidate / self.canonical_tx
        if tx_candidate.exists() and any(d.is_dir() for d in tx_candidate.iterdir()):
            dirs.append(tx_candidate)
            return dirs

        dirs.append(candidate)
        return dirs

    def _determine_unseen_classes(self) -> List[str]:
        """Selects unseen classes according to benchmark protocol."""
        if self.split_protocol == "split2":
            reference = set(SKETCHY_SPLIT2_UNSEEN_21)
        elif self.split_protocol == "split1":
            reference = set(SKETCHY_SPLIT1_UNSEEN_25)
        elif self.split_protocol == "tuberlin":
            reference = set(TUBERLIN_UNSEEN_30)
        elif self.split_protocol in ("cross_dataset", "all"):
            return sorted(self.all_classes)
        elif self.split_protocol in ("all_seen", "none"):
            return []
        else:
            raise ValueError(f"Unsupported split protocol: {self.split_protocol}")

        # Filter against available classes
        unseen = [c for c in self.all_classes if c in reference]
        return sorted(unseen)

    @staticmethod
    def extract_instance_id(filepath: str, is_sketch: bool) -> str:
        """
        Extracts pairing instance id.
        Sketch: 'n02691156_10151-1.png' -> 'n02691156_10151'
        Photo:  'n02691156_10151.jpg'   -> 'n02691156_10151'
        """
        stem = Path(filepath).stem
        if is_sketch and "-" in stem:
            return stem.rsplit("-", 1)[0]
        return stem

    def collect_samples(
        self, class_list: List[str], cls2idx: Dict[str, int]
    ) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]]]:
        """
        Collects all photo and sketch sample tuples (filepath, class_idx, instance_id).
        Aggregates across all discovered sketch directories (e.g. train + test).
        """
        photos: List[Tuple[str, int, str]] = []
        sketches: List[Tuple[str, int, str]] = []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for cname in class_list:
            if cname not in cls2idx:
                continue
            c_idx = cls2idx[cname]

            # 1. Collect photos for class
            p_folder = self.photo_dir / cname
            if p_folder.exists():
                for f in sorted(p_folder.iterdir()):
                    if f.suffix.lower() in valid_exts:
                        inst_id = self.extract_instance_id(str(f), is_sketch=False)
                        photos.append((str(f), c_idx, inst_id))

            # 2. Collect sketches for class across all sketch directories (Virtual Union)
            for s_root in self.sketch_dirs:
                s_folder = s_root / cname
                if s_folder.exists():
                    for f in sorted(s_folder.iterdir()):
                        if f.suffix.lower() in valid_exts:
                            inst_id = self.extract_instance_id(str(f), is_sketch=True)
                            sketches.append((str(f), c_idx, inst_id))

        return photos, sketches

    def print_summary(self) -> None:
        """Prints diagnostic overview of discovered partitions."""
        print("=" * 70)
        print(f"  DATASET CONFIG: {self.dataset_name.upper()} | PROTOCOL: {self.split_protocol}")
        print("=" * 70)
        print(f"  Photo Root:        {self.photo_dir}")
        print(f"  Sketch Roots:      {[str(p) for p in self.sketch_dirs]}")
        print(f"  Total Classes:     {len(self.all_classes)}")
        print(f"  Seen Classes:      {len(self.seen_classes)}")
        print(f"  Unseen Classes:    {len(self.unseen_classes)}")
        print("=" * 70)
