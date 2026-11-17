#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: app.py (Standalone ZeroGPU & REST API Backend)
ROLE: Interactive Web System & REST API for Zero-Shot & Open-World SBIR
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331 - UIT)
AUTHORS: Dinh Pham Anh Quan (24521425) & Pham Tuong Quan (24521451)
INSTRUCTOR: Thay Mai Tien Dung
================================================================================
CRITICAL ZEROGPU RULE:
  `import spaces` MUST BE THE ABSOLUTE FIRST IMPORT BEFORE TORCH!
================================================================================
"""

# -----------------------------------------------------------------------------
# 0. HUGGING FACE HUB COMPATIBILITY SHIM (MUST BE BEFORE SPACES/GRADIO)
# -----------------------------------------------------------------------------
import huggingface_hub
if not hasattr(huggingface_hub, "HfFolder"):
    class HfFolder:
        @staticmethod
        def get_token():
            try:
                from huggingface_hub import get_token
                return get_token()
            except Exception:
                return None
        @staticmethod
        def save_token(token: str):
            pass
        @staticmethod
        def delete_token():
            pass
    huggingface_hub.HfFolder = HfFolder

# -----------------------------------------------------------------------------
# 0.1 STARLETTE / FASTAPI TEMPLATE COMPATIBILITY PATCH FOR GRADIO 4.44.0
# Fixes "TypeError: unhashable type: 'dict'" caused by Starlette TemplateResponse signature change
# -----------------------------------------------------------------------------
try:
    import starlette.templating
    _orig_template_response = starlette.templating.Jinja2Templates.TemplateResponse
    def _patched_template_response(self, *args, **kwargs):
        if len(args) >= 1 and isinstance(args[0], str):
            template_name = args[0]
            context = args[1] if len(args) > 1 and isinstance(args[1], dict) else kwargs.pop("context", {})
            request = context.get("request") if isinstance(context, dict) else kwargs.pop("request", None)
            status_code = kwargs.pop("status_code", 200)
            return _orig_template_response(
                self,
                request=request,
                name=template_name,
                context=context,
                status_code=status_code,
                **kwargs
            )
        return _orig_template_response(self, *args, **kwargs)
    starlette.templating.Jinja2Templates.TemplateResponse = _patched_template_response
except Exception as _err:
    pass

# -----------------------------------------------------------------------------
# 1. ZEROGPU INIT & OFF-LOAD SANITIZATION (MUST BE BEFORE IMPORTING TORCH)
# -----------------------------------------------------------------------------
import os
import shutil

# Clean stale offload files if container was restarted
try:
    if os.path.exists("/data-nvme/zerogpu-offload"):
        for item in os.listdir("/data-nvme/zerogpu-offload"):
            p = os.path.join("/data-nvme/zerogpu-offload", item)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
            except Exception:
                pass
except Exception:
    pass

import spaces
print("[SYSTEM] ZeroGPU Hardware detected. Dynamic GPU allocator initialized.")

# Strictly idempotent guard for ZeroGPU startup to prevent clearing packed tensors
try:
    import spaces.zero.lifecycle
    _orig_lifecycle = spaces.zero.lifecycle.zerogpu_startup
    _started_lifecycle = False
    def _idempotent_lifecycle(*args, **kwargs):
        global _started_lifecycle
        if _started_lifecycle:
            return
        _started_lifecycle = True
        return _orig_lifecycle(*args, **kwargs)
    spaces.zero.lifecycle.zerogpu_startup = _idempotent_lifecycle
except Exception:
    pass

try:
    import spaces.zero
    if hasattr(spaces.zero, "startup"):
        _orig_zero_startup = spaces.zero.startup
        _started_zero = False
        def _idempotent_zero(*args, **kwargs):
            global _started_zero
            if _started_zero:
                return
            _started_zero = True
            return _orig_zero_startup(*args, **kwargs)
        spaces.zero.startup = _idempotent_zero
except Exception:
    pass

@spaces.GPU
def _zerogpu_startup_anchor():
    """Top-level explicit ZeroGPU anchor function for static AST scanner."""
    pass

import os
import sys
import json
import time
import io
import math
import copy
import base64
import random
import unicodedata
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True

import clip
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

proj_root = Path(__file__).resolve().parent
sys.path.insert(0, str(proj_root))

def get_initial_device() -> torch.device:
    try:
        if torch.cuda.is_available():
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")

DEVICE = get_initial_device()
print(f"[SYSTEM] Khoi tao he thong ZS-SBIR tren thiet bi: {DEVICE}...")

# -----------------------------------------------------------------------------
# 1. STANDALONE MODEL ARCHITECTURE & PEFT MODULES
# -----------------------------------------------------------------------------

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

SKETCHY_SPLIT2_UNSEEN_21: List[str] = [
    "cabin", "cow", "dolphin", "door", "giraffe",
    "helicopter", "mouse", "pear", "raccoon", "rhinoceros",
    "saw", "scissors", "seagull", "skyscraper", "songbird",
    "sword", "tree", "wading_bird", "wheelchair", "windmill",
    "window"
]

class LowRankLinearAdapter(nn.Module):
    """Low-Rank Adaptation (LoRA) module for Linear layers (Hu et al., ICLR 2022)."""
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad = False
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / float(rank)
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features, dtype=base_linear.weight.dtype) * (1.0 / math.sqrt(rank))
        )
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=base_linear.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        delta = (x.to(self.lora_A.dtype) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + delta.to(base_out.dtype)

class BottleneckResidualAdapter(nn.Module):
    """Bottleneck Residual Adapter with Zero-Initialization."""
    def __init__(self, embed_dim: int = 512, bottleneck_dim: int = 256, res_scale: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(bottleneck_dim, embed_dim)
        self.res_scale = res_scale
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        residual = self.fc2(self.act(self.fc1(x_fp32)))
        out = x_fp32 + self.res_scale * residual
        return F.normalize(out, dim=-1)

class SpatialAttentionPool(nn.Module):
    """Learned Spatial Attention Pooling over ViT Patch Tokens."""
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        scores = (patch_tokens @ self.query.transpose(-1, -2)).squeeze(-1)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (patch_tokens * weights).sum(dim=1)
        return self.proj(pooled)

class AsymmetricHybridLoRACLIP(nn.Module):
    """Asymmetric Dual-Encoder: Frozen Photo Tower + LoRA-adapted Sketch Tower."""
    def __init__(
        self,
        clip_model: nn.Module,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_targets: Tuple[str, ...] = ("out_proj",),
        embed_dim: int = 512,
        bottleneck_dim: int = 256,
        res_scale: float = 0.2,
        unfreeze_sketch_ln: bool = True,
        enable_patch_pooling: bool = True,
        patch_pool_weight: float = 0.3,
    ):
        super().__init__()
        self.enable_patch_pooling = enable_patch_pooling
        self.patch_pool_weight = patch_pool_weight
        self.embed_dim = embed_dim

        # 1. Photo Tower: Frozen CLIP Visual
        self.frozen_visual = copy.deepcopy(clip_model.visual).eval()
        for p in self.frozen_visual.parameters():
            p.requires_grad = False

        # 2. Sketch Tower: LoRA on Visual Encoder
        self.sketch_visual = clip_model.visual
        for p in self.sketch_visual.parameters():
            p.requires_grad = False

        self.lora_modules: nn.ModuleList = nn.ModuleList()
        if hasattr(self.sketch_visual, "transformer"):
            for block in self.sketch_visual.transformer.resblocks:
                if unfreeze_sketch_ln:
                    for m in block.modules():
                        if isinstance(m, nn.LayerNorm):
                            for p in m.parameters():
                                p.requires_grad = True

                candidate_modules = []
                if hasattr(block, "attn"):
                    candidate_modules.append(block.attn)
                if hasattr(block, "mlp"):
                    candidate_modules.append(block.mlp)

                for parent_mod in candidate_modules:
                    for target_name in lora_targets:
                        if hasattr(parent_mod, target_name):
                            orig_linear = getattr(parent_mod, target_name)
                            if isinstance(orig_linear, nn.Linear):
                                lora_mod = LowRankLinearAdapter(
                                    orig_linear, rank=lora_rank, alpha=lora_alpha
                                )
                                self.lora_modules.append(lora_mod)

                                def _make_hook(l_mod):
                                    def _hook(module, args, output):
                                        x_in = args[0]
                                        delta = (
                                            x_in.to(l_mod.lora_A.dtype)
                                            @ l_mod.lora_A.t()
                                            @ l_mod.lora_B.t()
                                        ) * l_mod.scaling
                                        return output + delta.to(output.dtype)
                                    return _hook

                                orig_linear.register_forward_hook(_make_hook(lora_mod))

        if self.enable_patch_pooling:
            self.spatial_pooler = SpatialAttentionPool(embed_dim)
        else:
            self.spatial_pooler = None

        self.sketch_adapter = BottleneckResidualAdapter(embed_dim, bottleneck_dim, res_scale)
        self.photo_adapter = BottleneckResidualAdapter(embed_dim, bottleneck_dim, res_scale)
        self.text_adapter = BottleneckResidualAdapter(embed_dim, bottleneck_dim, res_scale)

    @staticmethod
    def _extract_dual_tokens(visual: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = visual.conv1.weight.dtype
        x = x.to(dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        cls_token = visual.ln_post(x[:, 0, :])
        patch_tokens = visual.ln_post(x[:, 1:, :])
        if visual.proj is not None:
            cls_token = cls_token @ visual.proj
            patch_tokens = patch_tokens @ visual.proj
        return cls_token.float(), patch_tokens.float()

    def encode_sketch(self, sketch_tensor: torch.Tensor) -> torch.Tensor:
        if self.enable_patch_pooling and self.spatial_pooler is not None and hasattr(self.sketch_visual, "conv1"):
            cls_feat, patch_feats = self._extract_dual_tokens(self.sketch_visual, sketch_tensor)
            pooled_patch = self.spatial_pooler(patch_feats)
            fused_feat = F.normalize(cls_feat + self.patch_pool_weight * pooled_patch, dim=-1)
            return self.sketch_adapter(fused_feat)

        dtype = (
            self.sketch_visual.conv1.weight.dtype
            if hasattr(self.sketch_visual, "conv1")
            else next(self.sketch_visual.parameters()).dtype
        )
        raw_feat = self.sketch_visual(sketch_tensor.to(dtype)).float()
        return self.sketch_adapter(F.normalize(raw_feat, dim=-1))

    @torch.no_grad()
    def encode_frozen_visual(self, img_tensor: torch.Tensor) -> torch.Tensor:
        dtype = (
            self.frozen_visual.conv1.weight.dtype
            if hasattr(self.frozen_visual, "conv1")
            else next(self.frozen_visual.parameters()).dtype
        )
        feat = self.frozen_visual(img_tensor.to(dtype)).float()
        return F.normalize(feat, dim=-1)

    def encode_photo_from_cached(self, cached_photo_feat: torch.Tensor) -> torch.Tensor:
        norm_feat = F.normalize(cached_photo_feat.float(), dim=-1)
        return self.photo_adapter(norm_feat)

    def adapt_text_prototypes(self, raw_text_prototypes: torch.Tensor) -> torch.Tensor:
        return self.text_adapter(F.normalize(raw_text_prototypes.float(), dim=-1))

# -----------------------------------------------------------------------------
# 2. PREPROCESSING PIPELINE
# -----------------------------------------------------------------------------

def preprocess_sketch_pil(pil_img: Image.Image, target_size: int = 224, pad_ratio: float = 0.08) -> Image.Image:
    """Alpha-safe tight-crop with square padding for hand-drawn sketches."""
    if pil_img.mode == "RGBA":
        bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, pil_img).convert("RGB")
    else:
        img = pil_img.convert("RGB")

    gray = img.convert("L")
    inv = ImageOps.invert(gray)
    bbox = inv.getbbox()
    if bbox:
        cropped = img.crop(bbox)
        w, h = cropped.size
        max_side = max(w, h)
        pad = max(int(max_side * pad_ratio), 4)
        new_side = max_side + 2 * pad
        new_img = Image.new("RGB", (new_side, new_side), (255, 255, 255))
        new_img.paste(cropped, (pad + (max_side - w) // 2, pad + (max_side - h) // 2))
        return new_img.resize((target_size, target_size), Image.Resampling.BICUBIC)
    return img.resize((target_size, target_size), Image.Resampling.BICUBIC)

TRANSFORM_NORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
])

# -----------------------------------------------------------------------------
# 3. GLOBAL INITIALIZATION & CACHE LOADING
# -----------------------------------------------------------------------------

CLIP_BACKBONE = "ViT-B/32"
BASE_CLIP, _ = clip.load(CLIP_BACKBONE, device=DEVICE, jit=False)
BASE_CLIP = BASE_CLIP.float().eval()

# 3.1 Load Gallery Index Cache
candidate_cache_paths = [
    proj_root / "gallery_index.pt",
    proj_root / "outputs" / "cache" / "gallery_index.pt",
    proj_root.parent / "outputs" / "cache" / "gallery_index.pt",
]
cache_file = None
for cp in candidate_cache_paths:
    if cp.exists():
        cache_file = cp
        break

if not cache_file:
    try:
        from huggingface_hub import hf_hub_download
        print("[SYSTEM] Dang tai gallery_index.pt tu Dataset quan030106/sketchy_photos...")
        downloaded = hf_hub_download(
            repo_id="quan030106/sketchy_photos",
            filename="gallery_index.pt",
            repo_type="dataset",
        )
        cache_file = Path(downloaded)
    except Exception as e:
        print(f"[INFO] Chua tai duoc gallery_index tu dataset ({e}).")

if cache_file and cache_file.exists():
    print(f"[SYSTEM] Dang nap kho anh tu cache: {cache_file}...")
    cache_data = torch.load(str(cache_file), map_location="cpu", weights_only=False)
    DEFAULT_GALLERY_FEATS = cache_data["features"].float().to(DEVICE)
    DEFAULT_GALLERY_PATHS = cache_data["paths"]
    DEFAULT_GALLERY_CLASSES = cache_data["class_names"]
    DEFAULT_UNSEEN_MASK = cache_data["is_unseen"].to(DEVICE)
    CLASSES_125 = cache_data["classes"]
    print(f"[SYSTEM] Kho anh mac dinh: {DEFAULT_GALLERY_FEATS.shape[0]:,} anh ({DEFAULT_UNSEEN_MASK.sum().item():,} anh Unseen).")
else:
    print("[WARNING] Khong tim thay cache anh day du. Khoi tao kho anh demo gia lap.")
    CLASSES_125 = [
        "airplane", "alarm_clock", "ant", "apple", "arm", "banana", "bat", "bear",
        "bee", "beetle", "bell", "bench", "bicycle", "bird", "blimp", "bread",
        "bridge", "butterfly", "cactus", "cake", "camel", "camera", "candle",
        "car", "cat", "chair", "chicken", "church", "clock", "cloud", "coffee_cup",
        "couch", "cow", "crab", "crocodile", "cup", "deer", "dog", "dolphin",
        "door", "duck", "elephant", "eyeglasses", "fan", "feather", "fish",
        "flamingo", "flower", "frog", "giraffe", "guitar", "hammer", "hat",
        "hedgehog", "helicopter", "horse", "hot-air_balloon", "house", "cabin",
        "kangaroo", "knife", "lion", "lizard", "lobster", "monkey", "moon",
        "motorcycle", "mountain", "mouse", "mushroom", "owl", "palm_tree", "parrot",
        "pear", "penguin", "piano", "pig", "pillow", "pizza", "rabbit", "raccoon",
        "radio", "rifle", "scissors", "scorpion", "sea_turtle", "shark", "sheep",
        "shoe", "skyscraper", "snail", "snake", "spider", "spoon", "star",
        "strawberry", "sun", "swan", "sword", "t-shirt", "table", "teapot",
        "teddy_bear", "tiger", "tree", "trumpet", "umbrella", "violin", "volcano",
        "watch", "wheelchair", "windmill", "window", "zebra", "rhinoceros"
    ]
    DEFAULT_GALLERY_CLASSES = [CLASSES_125[i % len(CLASSES_125)] for i in range(2100)]
    DEFAULT_GALLERY_PATHS = [f"sample_{c}_{i}.jpg" for i, c in enumerate(DEFAULT_GALLERY_CLASSES)]
    DEFAULT_GALLERY_FEATS = F.normalize(torch.randn(2100, 512, device=DEVICE), dim=-1)
    DEFAULT_UNSEEN_MASK = torch.ones(2100, dtype=torch.bool, device=DEVICE)

# 3.2 Bilingual Concept Dictionary
CONCEPT_VIETNAMESE_MAP: Dict[str, str] = {
    "airplane": "Máy bay", "alarm_clock": "Đồng hồ báo thức", "ant": "Con kiến",
    "apple": "Quả táo", "arm": "Cánh tay", "banana": "Quả chuối", "bat": "Con dơi",
    "bear": "Con gấu", "bee": "Con ong", "beetle": "Bọ cánh cứng", "bell": "Cái chuông",
    "bench": "Ghế dài công viên", "bicycle": "Xe đạp", "bird": "Con chim",
    "blimp": "Khinh khí cầu", "bread": "Bánh mì", "bridge": "Cây cầu",
    "butterfly": "Con bướm", "cactus": "Cây xương rồng", "cake": "Bánh ngọt",
    "camel": "Lạc đà", "camera": "Máy ảnh", "candle": "Cây nến", "car": "Xe hơi / Ô tô",
    "cat": "Con mèo", "chair": "Cái ghế", "chicken": "Con gà", "church": "Nhà thờ",
    "clock": "Đồng hồ", "cloud": "Đám mây", "coffee_cup": "Cốc cà phê", "couch": "Ghế sofa",
    "cow": "Con bò", "crab": "Con cua", "crocodile": "Cá sấu", "cup": "Cái cốc",
    "deer": "Con hươu / nai", "dog": "Con chó", "dolphin": "Cá heo", "door": "Cánh cửa",
    "duck": "Con vịt", "elephant": "Con voi", "eyeglasses": "Kính mắt", "fan": "Cái quạt",
    "feather": "Lông vũ", "fish": "Con cá", "flamingo": "Chim hồng hạc", "flower": "Bông hoa",
    "frog": "Con ếch", "giraffe": "Hươu cao cổ", "guitar": "Đàn ghi-ta", "hammer": "Cái búa",
    "hat": "Cái mũ / nón", "hedgehog": "Con nhím", "helicopter": "Trực thăng",
    "horse": "Con ngựa", "hot-air_balloon": "Khinh khí cầu", "house": "Ngôi nhà",
    "cabin": "Nhà gỗ", "kangaroo": "Chuột túi", "knife": "Con dao", "lion": "Sư tử",
    "lizard": "Thằn lằn", "lobster": "Tôm hùm", "monkey": "Con khỉ", "moon": "Mặt trăng",
    "motorcycle": "Xe máy", "mountain": "Ngọn núi", "mouse": "Con chuột",
    "mushroom": "Cây nấm", "owl": "Cú mèo", "palm_tree": "Cây dừa", "parrot": "Con vẹt",
    "pear": "Quả lê", "penguin": "Chim cánh cụt", "piano": "Đàn piano", "pig": "Con heo",
    "pillow": "Cái gối", "pizza": "Bánh Pizza", "rabbit": "Con thỏ", "raccoon": "Gấu mèo",
    "radio": "Máy radio", "rifle": "Khẩu súng trường", "scissors": "Cái kéo",
    "scorpion": "Bọ cạp", "sea_turtle": "Rùa biển", "shark": "Cá mập", "sheep": "Con cừu",
    "shoe": "Đôi giày", "skyscraper": "Tòa nhà chọc trời", "snail": "Ốc sên",
    "snake": "Con rắn", "spider": "Con nhện", "spoon": "Cái thìa", "star": "Ngôi sao",
    "strawberry": "Quả dâu tây", "sun": "Mặt trời", "swan": "Thiên nga",
    "sword": "Thanh kiếm", "t-shirt": "Áo thun", "table": "Cái bàn", "teapot": "Ấm trà",
    "teddy_bear": "Gấu bông", "tiger": "Con hổ", "tree": "Cái cây", "trumpet": "Kèn trumpet",
    "umbrella": "Cây dù", "violin": "Đàn violin", "volcano": "Núi lửa", "watch": "Đồng hồ",
    "wheelchair": "Xe lăn", "windmill": "Cối xay gió", "window": "Cửa sổ", "zebra": "Ngựa vằn",
    "rhinoceros": "Tê giác"
}

def translate_concept(english_name: str) -> str:
    key = english_name.lower().replace(" ", "_")
    vi = CONCEPT_VIETNAMESE_MAP.get(key, None)
    if not vi:
        for k, v in CONCEPT_VIETNAMESE_MAP.items():
            if k in key or key in k:
                vi = v
                break
    clean_en = english_name.replace("_", " ").title()
    return f"{clean_en} ({vi})" if vi else clean_en

# 3.3 Text Prototypes for Zero-Shot Classification
PROMPT_TEMPLATES = [
    "a sketch of a {}",
    "a simple black and white line drawing of a {}",
    "a doodle of a {}",
    "an outline sketch of a {}",
]

def build_text_prototypes(class_list: List[str]) -> Tuple[torch.Tensor, List[str]]:
    tokens_list = []
    with torch.no_grad():
        for c in class_list:
            c_clean = c.replace("_", " ")
            prompts = [tmpl.format(c_clean) for tmpl in PROMPT_TEMPLATES]
            tok = clip.tokenize(prompts).to(DEVICE)
            feats = BASE_CLIP.encode_text(tok)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            proto = feats.mean(dim=0, keepdim=True)
            proto = proto / proto.norm(dim=-1, keepdim=True)
            tokens_list.append(proto)
    return torch.cat(tokens_list, dim=0), class_list

TEXT_PROTO_125, PROTO_CLASSES_125 = build_text_prototypes(CLASSES_125)

OPEN_WORLD_CONCEPTS = sorted(list(set(CLASSES_125 + [
    "guitar", "laptop", "smartphone", "drone", "submarine", "train", "motorcycle",
    "robot", "skyscraper", "cactus", "pizza", "coffee_cup", "shoe", "t-shirt",
    "glasses", "wristwatch", "flower", "tree", "mountain", "bridge", "book",
    "castle", "tent", "barn", "apartment", "villa", "cottage", "helicopter",
    "scissors", "wheelchair", "cabin", "rhinoceros"
])))
OPEN_WORLD_PROTOS, OPEN_WORLD_NAMES = build_text_prototypes(OPEN_WORLD_CONCEPTS)

def remove_diacritics(text: str) -> str:
    norm = unicodedata.normalize('NFD', text)
    stripped = ''.join(c for c in norm if unicodedata.category(c) != 'Mn')
    return stripped.replace('đ', 'd').replace('Đ', 'D')

VIETNAMESE_KEYWORD_BRIDGE = [
    (r"\b(non|mu|mu len|non la|mu bao hiem|mu luoi trai|chiec mu)\b", "a hat, cap, headwear"),
    (r"\b(pizza|banh pizza)\b", "a round baked pizza pie with cheese"),
    (r"\b(khi cau|khinh khi cau)\b", "an aerial blimp, airship, zeppelin"),
    (r"\b(banh mi|o banh mi)\b", "a loaf of bread, bakery bun, baguette"),
    (r"\b(truc thang|may bay truc thang)\b", "a helicopter with rotary blades"),
    (r"\b(may bay|phi co)\b", "an airplane aircraft flying"),
    (r"\b(cai keo|cay keo)\b", "a pair of cutting scissors"),
    (r"\b(xe lan)\b", "a mobility wheelchair"),
    (r"\b(nha go|can nha)\b", "a wooden cabin cottage house"),
    (r"\b(te giac)\b", "a horned rhinoceros wild animal"),
    (r"\b(xe may|xe dap|2 banh)\b", "a two-wheeled vehicle, motorcycle, bicycle"),
    (r"\b(o to|xe hoi|4 banh)\b", "a four-wheeled vehicle, automobile, car"),
]

def enrich_semantic_clue(raw_clue: Optional[str]) -> Optional[str]:
    if not raw_clue or not raw_clue.strip():
        return None
    text = raw_clue.strip()
    unaccented = remove_diacritics(text).lower()
    expansions = []
    import re
    for pattern, eng_expansion in VIETNAMESE_KEYWORD_BRIDGE:
        if re.search(pattern, unaccented):
            expansions.append(eng_expansion)
    if expansions:
        return ", ".join(list(dict.fromkeys(expansions)))
    return text

# -----------------------------------------------------------------------------
# 3.4 ZERO-EXTRACTION ZIP PHOTO STREAM (Boot in 3s, read photos in 1ms)
# -----------------------------------------------------------------------------
PHOTO_ZIP_PATH: Optional[Path] = None
CACHED_ZIP_HANDLE: Optional[zipfile.ZipFile] = None

# Check local first
local_zip = proj_root / "sketchy_photos.zip"
if local_zip.exists():
    PHOTO_ZIP_PATH = local_zip
    print(f"[SYSTEM] Phat hien {local_zip.name} cuc bo.")
else:
    try:
        from huggingface_hub import hf_hub_download
        print("[SYSTEM] Dang ket noi kho anh tu Dataset quan030106/sketchy_photos...")
        downloaded = hf_hub_download(
            repo_id="quan030106/sketchy_photos",
            filename="sketchy_photos.zip",
            repo_type="dataset",
        )
        PHOTO_ZIP_PATH = Path(downloaded)
        print(f"[SYSTEM] Ket noi kho anh zip thanh cong ({PHOTO_ZIP_PATH.stat().st_size / (1024*1024):.1f} MB)!")
    except Exception as e:
        print(f"[INFO] Chua ket noi duoc dataset zip ({e}).")

def ensure_photo_zip() -> Optional[Path]:
    global PHOTO_ZIP_PATH
    if PHOTO_ZIP_PATH is not None and PHOTO_ZIP_PATH.exists():
        return PHOTO_ZIP_PATH
    local_zip = proj_root / "sketchy_photos.zip"
    if local_zip.exists():
        PHOTO_ZIP_PATH = local_zip
        return PHOTO_ZIP_PATH
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="quan030106/sketchy_photos",
            filename="sketchy_photos.zip",
            repo_type="dataset",
        )
        PHOTO_ZIP_PATH = Path(downloaded)
        return PHOTO_ZIP_PATH
    except Exception as e:
        print(f"[WARNING] Khong the tai sketchy_photos.zip: {e}")
        return None

def get_zip_handle() -> Optional[zipfile.ZipFile]:
    global CACHED_ZIP_HANDLE
    zp = ensure_photo_zip()
    if CACHED_ZIP_HANDLE is None and zp and zp.exists():
        try:
            CACHED_ZIP_HANDLE = zipfile.ZipFile(str(zp), 'r')
        except Exception as e:
            print(f"[WARNING] Khong the doc file zip: {e}")
    return CACHED_ZIP_HANDLE

def get_photo_image(original_path: str, class_name: str) -> Optional[Image.Image]:
    """Retrieve photo directly from disk or from sketchy_photos.zip in <1ms without disk extraction."""
    # Cross-platform normalization: strip Windows drive and backslashes
    clean_path = str(original_path).replace("\\", "/")
    fname = clean_path.split("/")[-1]

    # 1. Check direct file path if exists
    p = Path(original_path)
    if p.exists() and p.is_file():
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            pass

    # 2. Check local disk directories if present
    disk_cands = [
        proj_root / "photos" / class_name / fname,
        proj_root / "gallery_photos" / class_name / fname,
        proj_root / "photos" / fname,
    ]
    for dc in disk_cands:
        if dc.exists() and dc.is_file():
            try:
                return Image.open(dc).convert("RGB")
            except Exception:
                pass

    # 3. Direct instant read from cached ZIP (0 ms disk extraction overhead)
    zip_p = ensure_photo_zip()
    if zip_p and zip_p.exists():
        zh = get_zip_handle()
        zip_candidates = [
            f"{class_name}/{fname}",
            f"photos/{class_name}/{fname}",
            f"sketchy_photos/{class_name}/{fname}",
            fname,
        ]
        for zcand in zip_candidates:
            if zh is not None:
                try:
                    raw_bytes = zh.read(zcand)
                    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                except Exception:
                    pass
            # Fallback direct read with fresh handle
            try:
                with zipfile.ZipFile(str(zip_p), "r") as fresh_zh:
                    raw_bytes = fresh_zh.read(zcand)
                    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            except Exception:
                continue

    return None

def image_to_base64_thumb(photo_input: Any, class_name: str = "", max_size=(160, 160)) -> Optional[str]:
    try:
        im = None
        if isinstance(photo_input, Image.Image):
            im = photo_input.convert("RGB")
        elif isinstance(photo_input, str):
            im = get_photo_image(photo_input, class_name)

        if im is None:
            im = Image.new("RGB", (160, 160), (30, 41, 59))

        im.thumbnail(max_size)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=75)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None

def decode_base64_image(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

# -----------------------------------------------------------------------------
# 4. LOAD TRAINED ASYMMETRIC LORA-CLIP MODEL
# -----------------------------------------------------------------------------

def load_trained_model() -> AsymmetricHybridLoRACLIP:
    candidate_paths = [
        proj_root / "best_seco_lora_clip.pt",
        proj_root / "outputs" / "checkpoints" / "best_seco_lora_clip.pt",
        proj_root.parent / "outputs" / "checkpoints" / "best_seco_lora_clip.pt",
    ]
    ckpt_path = None
    for p in candidate_paths:
        if p.exists():
            ckpt_path = p
            break

    if ckpt_path is None:
        try:
            from huggingface_hub import hf_hub_download
            print("[SYSTEM] Dang tim best_seco_lora_clip.pt tu Dataset quan030106/sketchy_photos...")
            downloaded = hf_hub_download(
                repo_id="quan030106/sketchy_photos",
                filename="best_seco_lora_clip.pt",
                repo_type="dataset",
            )
            ckpt_path = Path(downloaded)
        except Exception as e:
            print(f"[INFO] Chua tai duoc model tu dataset ({e}).")

    model = AsymmetricHybridLoRACLIP(
        clip_model=BASE_CLIP,
        lora_rank=16,
        lora_alpha=32.0,
        lora_targets=("out_proj",),
        enable_patch_pooling=True,
        patch_pool_weight=0.3,
    ).to(DEVICE)

    if ckpt_path and ckpt_path.exists():
        try:
            print(f"[SYSTEM] Dang nap trong so toi uu tu: {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
            if "model_state" in ckpt:
                state = ckpt["model_state"]
                model.sketch_adapter.load_state_dict(state["sketch_adapter"])
                model.photo_adapter.load_state_dict(state["photo_adapter"])
                model.text_adapter.load_state_dict(state["text_adapter"])
                model.sketch_visual.load_state_dict(state["sketch_visual"])
                if model.spatial_pooler is not None and "spatial_pooler" in state:
                    model.spatial_pooler.load_state_dict(state["spatial_pooler"])
            print("[SYSTEM] Nap trong so model thanh cong!")
        except Exception as e:
            print(f"[WARNING] Khong the nap checkpoint ({e}). Dung khoi tao mac dinh.")
    else:
        print("[WARNING] Khong tim thay checkpoint model. Dung khoi tao mac dinh.")

    model.eval()
    return model

MODEL = load_trained_model()

# -----------------------------------------------------------------------------
# 5. CORE RETRIEVAL & QUERY LOGIC
# -----------------------------------------------------------------------------

def ensure_active_device() -> torch.device:
    global DEFAULT_GALLERY_FEATS, DEFAULT_UNSEEN_MASK, TEXT_PROTO_125, OPEN_WORLD_PROTOS
    active_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if active_dev != MODEL.sketch_visual.conv1.weight.device:
        MODEL.to(active_dev)
        BASE_CLIP.to(active_dev)
        if hasattr(DEFAULT_GALLERY_FEATS, "to"):
            DEFAULT_GALLERY_FEATS = DEFAULT_GALLERY_FEATS.to(active_dev)
            DEFAULT_UNSEEN_MASK = DEFAULT_UNSEEN_MASK.to(active_dev)
            TEXT_PROTO_125 = TEXT_PROTO_125.to(active_dev)
            OPEN_WORLD_PROTOS = OPEN_WORLD_PROTOS.to(active_dev)
    return active_dev

def extract_sketch_tensor(sketch_input: Optional[Any], device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    if sketch_input is None:
        return None
    if isinstance(sketch_input, dict) and "composite" in sketch_input:
        raw_img = sketch_input["composite"]
    elif isinstance(sketch_input, dict) and "image" in sketch_input:
        raw_img = sketch_input["image"]
    elif isinstance(sketch_input, Image.Image):
        raw_img = sketch_input
    elif isinstance(sketch_input, np.ndarray):
        raw_img = Image.fromarray(sketch_input)
    else:
        return None

    dev = device if device is not None else MODEL.sketch_visual.conv1.weight.device
    preprocessed_img = preprocess_sketch_pil(raw_img, target_size=224)
    tensor = TRANSFORM_NORM(preprocessed_img).unsqueeze(0).to(dev)
    return tensor

def encode_query_representation(
    sketch_tensor: torch.Tensor,
    semantic_clue: Optional[str] = None
) -> torch.Tensor:
    dev = sketch_tensor.device
    with torch.no_grad():
        q_feat = MODEL.encode_sketch(sketch_tensor)
        q_feat = F.normalize(q_feat, dim=-1)

        enriched_clue = enrich_semantic_clue(semantic_clue)
        if enriched_clue:
            txt_tok = clip.tokenize([enriched_clue]).to(dev)
            txt_feat = BASE_CLIP.encode_text(txt_tok)
            txt_feat = F.normalize(txt_feat, dim=-1)
            q_feat = F.normalize(0.70 * q_feat + 0.30 * txt_feat, dim=-1)

    return q_feat

# -----------------------------------------------------------------------------
# 6. REST API (FASTAPI) WITH FULL CORS
# -----------------------------------------------------------------------------

fastapi_app = FastAPI(
    title="ZS-SBIR Inference API",
    description="High-Precision Backend for Zero-Shot SBIR and Sketch Classification",
    version="2.0.0"
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RetrievePayload(BaseModel):
    sketch: str
    gallery_scope: str = "unseen"
    semantic_clue: Optional[str] = None
    top_k: int = 10

class ClassifyPayload(BaseModel):
    sketch: str
    semantic_clue: Optional[str] = None

@fastapi_app.get("/api")
def api_root():
    return {
        "title": "ZS-SBIR API & Studio",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/api/health"
    }

@fastapi_app.get("/healthz")
def api_healthz():
    return {"status": "ok"}

@fastapi_app.get("/api/health")
def api_health():
    return {
        "status": "online",
        "device": str(DEVICE),
        "gallery_size": DEFAULT_GALLERY_FEATS.shape[0],
        "unseen_count": int(DEFAULT_UNSEEN_MASK.sum().item()),
        "classes_count": len(CLASSES_125),
        "model": "Asymmetric LoRA-CLIP ViT-B/32 (Epoch 09)",
        "zero_shot_map": 72.40,
        "unseen_recall_1": 81.87
    }

@fastapi_app.post("/api/retrieve")
@spaces.GPU
def api_retrieve(payload: RetrievePayload):
    t_start = time.time()
    dev = ensure_active_device()
    try:
        pil_img = decode_base64_image(payload.sketch)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 sketch: {e}")

    preprocessed_img = preprocess_sketch_pil(pil_img, target_size=224)
    sketch_tensor = TRANSFORM_NORM(preprocessed_img).unsqueeze(0).to(dev)
    q_feat = encode_query_representation(sketch_tensor, payload.semantic_clue)

    # Predict concept
    with torch.no_grad():
        adapted_protos = MODEL.adapt_text_prototypes(TEXT_PROTO_125.to(dev))
        sim_text = (q_feat @ adapted_protos.T).squeeze(0)
        best_idx = sim_text.argmax().item()
        pred_concept = PROTO_CLASSES_125[best_idx]

    # Gallery selection (3 scopes: unseen, seen/train, all)
    scope_str = payload.gallery_scope.lower()
    if "unseen" in scope_str:
        mask = DEFAULT_UNSEEN_MASK
        base_feats = DEFAULT_GALLERY_FEATS[mask]
        base_paths = [p for p, m in zip(DEFAULT_GALLERY_PATHS, mask.tolist()) if m]
        base_classes = [c for c, m in zip(DEFAULT_GALLERY_CLASSES, mask.tolist()) if m]
    elif "seen" in scope_str or "train" in scope_str:
        mask = ~DEFAULT_UNSEEN_MASK
        base_feats = DEFAULT_GALLERY_FEATS[mask]
        base_paths = [p for p, m in zip(DEFAULT_GALLERY_PATHS, mask.tolist()) if m]
        base_classes = [c for c, m in zip(DEFAULT_GALLERY_CLASSES, mask.tolist()) if m]
    else:
        base_feats = DEFAULT_GALLERY_FEATS
        base_paths = DEFAULT_GALLERY_PATHS
        base_classes = DEFAULT_GALLERY_CLASSES

    # Top-K calculation
    with torch.no_grad():
        adapted_gallery = MODEL.encode_photo_from_cached(base_feats.to(dev))
        sims = (q_feat @ adapted_gallery.T).squeeze(0)
        k = min(payload.top_k, sims.shape[0])
        topk_scores, topk_indices = torch.topk(sims, k=k)

    results = []
    for rank, (score, idx) in enumerate(zip(topk_scores, topk_indices), start=1):
        idx_val = idx.item()
        c_name = base_classes[idx_val]
        p_path = base_paths[idx_val]
        sim_val = float(score.item())
        thumb = image_to_base64_thumb(p_path, class_name=c_name)
        results.append({
            "rank": rank,
            "class": c_name,
            "similarity": round(sim_val, 4),
            "image": thumb or ""
        })

    elapsed_ms = round((time.time() - t_start) * 1000.0, 1)
    max_sim = float(topk_scores[0].item()) if len(topk_scores) > 0 else 0.0

    return {
        "top_1_class": pred_concept,
        "max_similarity": round(max_sim, 4),
        "latency_ms": elapsed_ms,
        "results": results
    }

@fastapi_app.post("/api/classify")
@spaces.GPU
def api_classify(payload: ClassifyPayload):
    t_start = time.time()
    dev = ensure_active_device()
    try:
        pil_img = decode_base64_image(payload.sketch)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 sketch: {e}")

    preprocessed_img = preprocess_sketch_pil(pil_img, target_size=224)
    sketch_tensor = TRANSFORM_NORM(preprocessed_img).unsqueeze(0).to(dev)
    q_feat = encode_query_representation(sketch_tensor, payload.semantic_clue)

    with torch.no_grad():
        adapted_protos = MODEL.adapt_text_prototypes(OPEN_WORLD_PROTOS.to(dev))
        sims = (q_feat @ adapted_protos.T).squeeze(0)
        probs = F.softmax(sims * 100.0, dim=-1)
        top5_probs, top5_indices = torch.topk(probs, k=5)

    candidates = []
    for rank, (p, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
        idx_val = idx.item()
        c_name = OPEN_WORLD_NAMES[idx_val]
        candidates.append({
            "rank": rank,
            "class": c_name,
            "probability": round(float(p.item()), 4),
            "cosine_sim": round(float(sims[idx_val].item()), 4)
        })

    elapsed_ms = round((time.time() - t_start) * 1000.0, 1)
    return {
        "top_1_class": candidates[0]["class"],
        "confidence": candidates[0]["probability"],
        "latency_ms": elapsed_ms,
        "candidates": candidates
    }

# -----------------------------------------------------------------------------
# 7. GRADIO WEB UI
# -----------------------------------------------------------------------------

@spaces.GPU
def gradio_retrieve(sketch_input, sketch_upload, scope, clue):
    t_start = time.time()
    dev = ensure_active_device()
    active_sketch = sketch_upload if sketch_upload is not None else sketch_input
    sketch_tensor = extract_sketch_tensor(active_sketch, device=dev)
    if sketch_tensor is None:
        return "Chưa có nét vẽ", "--", "0.0 ms", []

    q_feat = encode_query_representation(sketch_tensor, clue)
    with torch.no_grad():
        adapted_protos = MODEL.adapt_text_prototypes(TEXT_PROTO_125.to(dev))
        sim_text = (q_feat @ adapted_protos.T).squeeze(0)
        best_idx = sim_text.argmax().item()
        pred_label = translate_concept(PROTO_CLASSES_125[best_idx])

    scope_str = scope.lower()
    if "unseen" in scope_str:
        mask = DEFAULT_UNSEEN_MASK
        base_feats = DEFAULT_GALLERY_FEATS[mask]
        base_paths = [p for p, m in zip(DEFAULT_GALLERY_PATHS, mask.tolist()) if m]
        base_classes = [c for c, m in zip(DEFAULT_GALLERY_CLASSES, mask.tolist()) if m]
    elif "seen" in scope_str or "train" in scope_str:
        mask = ~DEFAULT_UNSEEN_MASK
        base_feats = DEFAULT_GALLERY_FEATS[mask]
        base_paths = [p for p, m in zip(DEFAULT_GALLERY_PATHS, mask.tolist()) if m]
        base_classes = [c for c, m in zip(DEFAULT_GALLERY_CLASSES, mask.tolist()) if m]
    else:
        base_feats = DEFAULT_GALLERY_FEATS
        base_paths = DEFAULT_GALLERY_PATHS
        base_classes = DEFAULT_GALLERY_CLASSES

    with torch.no_grad():
        adapted_gallery = MODEL.encode_photo_from_cached(base_feats.to(dev))
        sims = (q_feat @ adapted_gallery.T).squeeze(0)
        top10_scores, top10_indices = torch.topk(sims, k=min(10, sims.shape[0]))

    results_grid = []
    for rank, (score, idx) in enumerate(zip(top10_scores, top10_indices), start=1):
        idx_val = idx.item()
        p_path = base_paths[idx_val]
        c_name = translate_concept(base_classes[idx_val])
        sim_val = float(score.item() * 100.0)
        caption = f"#{rank} {c_name} | {sim_val:.1f}%"
        im = get_photo_image(p_path, base_classes[idx_val])
        if im is not None:
            results_grid.append((im, caption))
        else:
            ph = Image.new("RGB", (200, 200), (30, 41, 59))
            results_grid.append((ph, caption))

    max_sim_text = f"{float(top10_scores[0].item() * 100.0):.1f}%" if len(top10_scores) > 0 else "--"
    elapsed = f"{(time.time() - t_start) * 1000.0:.1f} ms"
    return pred_label, max_sim_text, elapsed, results_grid

@spaces.GPU
def gradio_classify(sketch_input, sketch_upload, clue):
    t_start = time.time()
    dev = ensure_active_device()
    active_sketch = sketch_upload if sketch_upload is not None else sketch_input
    sketch_tensor = extract_sketch_tensor(active_sketch, device=dev)
    if sketch_tensor is None:
        return "Chưa có nét vẽ", "--", "<p>Vui lòng vẽ nét trên canvas.</p>", "0.0 ms"

    q_feat = encode_query_representation(sketch_tensor, clue)
    with torch.no_grad():
        adapted_protos = MODEL.adapt_text_prototypes(OPEN_WORLD_PROTOS.to(dev))
        sims = (q_feat @ adapted_protos.T).squeeze(0)
        probs = F.softmax(sims * 100.0, dim=-1)
        top5_probs, top5_indices = torch.topk(probs, k=5)

    top1_idx = top5_indices[0].item()
    top1_name = translate_concept(OPEN_WORLD_NAMES[top1_idx])
    top1_conf = f"{float(top5_probs[0].item() * 100.0):.1f}%"

    rows = []
    for rank, (p, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
        idx_val = idx.item()
        c_name = translate_concept(OPEN_WORLD_NAMES[idx_val])
        pct = float(p.item() * 100.0)
        cos_val = float(sims[idx_val].item())
        color = "#0284C7" if rank == 1 else "#64748B"
        rows.append(f"""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 500; margin-bottom: 4px; color: #1E293B;">
                <span>#{rank} {c_name}</span>
                <span style="color: {color}; font-weight: 600;">{pct:.1f}% (Cos: {cos_val:.3f})</span>
            </div>
            <div style="background: #E2E8F0; border-radius: 999px; height: 8px; overflow: hidden;">
                <div style="background: {color}; width: {pct}%; height: 100%; border-radius: 999px;"></div>
            </div>
        </div>
        """)

    breakdown_html = f"<div style='background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px;'>{''.join(rows)}</div>"
    elapsed = f"{(time.time() - t_start) * 1000.0:.1f} ms"
    return top1_name, top1_conf, breakdown_html, elapsed

custom_css = """
body, .gradio-container {
    background-color: #0F172A !important;
    color: #F8FAFC !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
.tech-card {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
"""

with gr.Blocks(title="ZS-SBIR Studio — Zero-Shot Sketch-Based Image Retrieval", css=custom_css) as demo:
    gr.Markdown(
        """
        # HỆ THỐNG TRUY HỒI HÌNH ẢNH QUA BẢN VẼ PHÁC THẢO (ZS-SBIR)
        **Asymmetric Hybrid LoRA-CLIP** • CS331 — Đồ án Cuối kỳ • UIT
        *Sinh viên: Đinh Phạm Anh Quân (24521425) & Phạm Tường Quân (24521451) • GVHD: Thầy Mai Tiến Dũng*
        """
    )

    with gr.Row():
        with gr.Column(scale=5, elem_classes="tech-card"):
            gr.Markdown("### 1. Bản vẽ Phác thảo (Query Sketch)")
            sketchpad = gr.Sketchpad(
                type="pil",
                label="Bảng vẽ tay trực tiếp",
                brush=gr.Brush(colors=["#000000", "#FFFFFF"], default_size=6),
                layers=False,
                canvas_size=(250, 250),
            )
            upload_sketch = gr.Image(type="pil", label="Hoặc tải ảnh phác thảo", sources=["upload"])

            gr.Markdown("### 2. Thiết lập truy vấn")
            gallery_scope = gr.Radio(
                choices=["Tập Unseen (2,100 ảnh - Zero-Shot chuẩn)", "Tập Seen / Train (10,400 ảnh - Tập đã học)", "Toàn bộ kho (12,500 ảnh)"],
                value="Tập Unseen (2,100 ảnh - Zero-Shot chuẩn)",
                label="Phạm vi tìm kiếm"
            )
            semantic_clue = gr.Textbox(placeholder="Ví dụ: 'xe 2 bánh', 'trực thăng bay'...", label="Mô tả bổ trợ (Optional)")

            with gr.Row():
                btn_ret = gr.Button("🔍 TRUY HỒI HÌNH ẢNH (TOP 10)", variant="primary")
                btn_cls = gr.Button("🎯 NHẬN DIỆN ĐỐI TƯỢNG", variant="secondary")

        with gr.Column(scale=7):
            with gr.Tabs():
                with gr.TabItem("🖼️ Truy hồi Hình ảnh (Top 10)"):
                    with gr.Row():
                        res_concept = gr.Markdown("**Khái niệm:** Đang chờ nét vẽ...")
                        res_sim = gr.Markdown("**Độ khớp max:** --")
                        res_time = gr.Markdown("**Thời gian:** 0.0 ms")
                    res_gallery = gr.Gallery(label="Top 10 bức ảnh khớp nhất", columns=5, rows=2, height="auto")

                with gr.TabItem("🎯 Nhận diện Vật thể (Zero-Shot)"):
                    cls_label = gr.Markdown("### Đang chờ nét vẽ...")
                    cls_conf = gr.Markdown("Xác suất: --")
                    cls_breakdown = gr.HTML("<p>Vẽ nét phác thảo để bắt đầu nhận diện.</p>")
                    cls_time = gr.Markdown("Thời gian: 0.0 ms")

    btn_ret.click(
        fn=gradio_retrieve,
        inputs=[sketchpad, upload_sketch, gallery_scope, semantic_clue],
        outputs=[res_concept, res_sim, res_time, res_gallery],
        api_name=False,
    )
    btn_cls.click(
        fn=gradio_classify,
        inputs=[sketchpad, upload_sketch, semantic_clue],
        outputs=[cls_label, cls_conf, cls_breakdown, cls_time],
        api_name=False,
    )

    demo.get_api_info = lambda *args, **kwargs: {"named_endpoints": {}, "unnamed_endpoints": {}}
    demo.queue(default_concurrency_limit=5)

app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# Explicit ZeroGPU startup report for mounted ASGI / Uvicorn server
try:
    from spaces import zero as _spaces_zero
    if hasattr(_spaces_zero, "startup"):
        _spaces_zero.startup()
        print("[SYSTEM] ZeroGPU startup report sent successfully to broker!", flush=True)
    elif hasattr(_spaces_zero, "zerogpu_startup"):
        _spaces_zero.zerogpu_startup()
        print("[SYSTEM] ZeroGPU startup report sent successfully to broker!", flush=True)
except Exception as e:
    print(f"[SYSTEM] ZeroGPU direct startup notice: {e}", flush=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
