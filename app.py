#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: app.py
ROLE: Interactive Web System & REST API for Zero-Shot & Open-World SBIR
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331 - UIT)
AUTHORS: Dinh Pham Anh Quan (24521425) & Pham Tuong Quan (24521451)
INSTRUCTOR: Thay Mai Tien Dung
================================================================================
Core Capabilities:
  1. Sketch-to-Image Retrieval (Top-10 Cosine Similarity across 12,500 Gallery).
  2. Zero-Shot Sketch Classification (Direct object guessing via CLIP text prototypes).
  3. Custom Session Gallery Upload (RAM-based Ephemeral Indexing).
  4. Multimodal Semantic Clue Fusion (Cross-Modal Text Adapter conditioning).
  5. High-Performance REST API with CORS for Vercel Web Deployment.
================================================================================
"""

import os
import sys
import json
import time
import io
import base64
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import clip
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
# ZeroGPU Support
try:
    import spaces
    gpu_decorator = spaces.GPU
    print('[SYSTEM] ZeroGPU Hardware detected. GPU dynamic decorator enabled.')
except Exception:
    gpu_decorator = lambda f: f


# Project root setup
proj_root = Path(__file__).resolve().parent
sys.path.insert(0, str(proj_root))

try:
    from src.config import SKETCHY_SPLIT2_UNSEEN_21
    from src.data.preprocessor import SketchCVPreprocessor
    from src.data.transforms import build_transforms
    from src.models.clip_lora import AsymmetricHybridLoRACLIP
except ImportError:
    # Fallback if running directly in hf_space
    from configs import *
    from src.data.transforms import build_transforms
    from src.models.clip_lora import AsymmetricHybridLoRACLIP


# -----------------------------------------------------------------------------
# 1. GLOBAL INITIALIZATION & CACHE LOADING
# -----------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Khoi tao he thong ZS-SBIR tren thiet bi: {DEVICE}...")

# 1.1 Load Frozen Base CLIP
CLIP_BACKBONE = "ViT-B/32"
BASE_CLIP, _ = clip.load(CLIP_BACKBONE, device=DEVICE, jit=False)
BASE_CLIP = BASE_CLIP.float()
BASE_CLIP.eval()

# 1.2 Load Pre-computed Default Gallery Cache
candidate_cache_paths = [
    proj_root / "outputs" / "cache" / "gallery_index.pt",
    proj_root / "gallery_index.pt",
    proj_root.parent / "outputs" / "cache" / "gallery_index.pt",
]
cache_file = None
for cp in candidate_cache_paths:
    if cp.exists():
        cache_file = cp
        break

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
    print("[WARNING] Khong tim thay cache anh day du. Khoi tao kho anh demo gia lap tren RAM.")
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

# 1.3 Bilingual Dictionary
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

# 1.4 Text Prototypes for Zero-Shot Classification
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

TFS = build_transforms(image_size=224)


# -----------------------------------------------------------------------------
# 2. LOAD TRAINED ASYMMETRIC LORA-CLIP MODEL
# -----------------------------------------------------------------------------

def load_trained_model() -> AsymmetricHybridLoRACLIP:
    candidate_paths = [
        proj_root / "outputs" / "checkpoints" / "best_seco_lora_clip.pt",
        proj_root / "best_seco_lora_clip.pt",
        proj_root.parent / "outputs" / "checkpoints" / "best_seco_lora_clip.pt",
    ]
    ckpt_path = None
    for p in candidate_paths:
        if p.exists():
            ckpt_path = p
            break

    rank, alpha, targets = 16, 32.0, ("out_proj",)
    enable_patch_pooling = True

    model = AsymmetricHybridLoRACLIP(
        clip_model=BASE_CLIP,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_targets=targets,
        enable_patch_pooling=enable_patch_pooling,
        patch_pool_weight=0.3,
    ).to(DEVICE)

    if ckpt_path is not None:
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
            print("[SYSTEM] Nap trong so thanh cong!")
        except Exception as e:
            print(f"[WARNING] Khong the nap checkpoint ({e}). Dung khoi tao mac dinh.")

    model.eval()
    return model

MODEL = load_trained_model()


# -----------------------------------------------------------------------------
# 3. CORE PROCESSING LOGIC
# -----------------------------------------------------------------------------

def extract_sketch_tensor(sketch_input: Optional[Any]) -> Optional[torch.Tensor]:
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

    if isinstance(raw_img, Image.Image) and raw_img.mode == "RGBA":
        white_bg = Image.new("RGBA", raw_img.size, (255, 255, 255, 255))
        raw_img = Image.alpha_composite(white_bg, raw_img).convert("RGB")
    else:
        raw_img = raw_img.convert("RGB")

    tensor = TFS["sketch_clean"](raw_img).unsqueeze(0).to(DEVICE)
    return tensor

def encode_query_representation(
    sketch_tensor: torch.Tensor,
    semantic_clue: Optional[str] = None
) -> torch.Tensor:
    with torch.no_grad():
        q_feat = MODEL.encode_sketch(sketch_tensor)
        q_feat = F.normalize(q_feat, dim=-1)

        enriched_clue = enrich_semantic_clue(semantic_clue)
        if enriched_clue:
            txt_tok = clip.tokenize([enriched_clue]).to(DEVICE)
            txt_feat = BASE_CLIP.encode_text(txt_tok)
            txt_feat = F.normalize(txt_feat, dim=-1)
            q_feat = F.normalize(0.70 * q_feat + 0.30 * txt_feat, dim=-1)

    return q_feat

def resolve_photo_path(original_path: str, class_name: str) -> Optional[str]:
    p = Path(original_path)
    if p.exists():
        return str(p)
    cand_dir = proj_root / "gallery_photos" / class_name
    if cand_dir.exists():
        cand_file = cand_dir / p.name
        if cand_file.exists():
            return str(cand_file)
        all_photos = list(cand_dir.glob("*.jpg"))
        if all_photos:
            return str(all_photos[0])
    return None

def image_to_base64_thumb(path_or_img: Any, max_size=(160, 160)) -> Optional[str]:
    try:
        if isinstance(path_or_img, str):
            p = Path(path_or_img)
            if p.exists():
                im = Image.open(p).convert("RGB")
            else:
                return None
        elif isinstance(path_or_img, Image.Image):
            im = path_or_img.convert("RGB")
        else:
            return None
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
# 4. REST API (FASTAPI) WITH FULL CORS
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

@fastapi_app.get("/api/health")
def api_health():
    return {
        "status": "online",
        "device": str(DEVICE),
        "gallery_size": DEFAULT_GALLERY_FEATS.shape[0],
        "unseen_count": int(DEFAULT_UNSEEN_MASK.sum().item()),
        "classes_count": len(CLASSES_125),
        "model": "Asymmetric LoRA-CLIP ViT-B/16 (Epoch 09)",
        "zero_shot_map": 72.40,
        "unseen_recall_1": 81.87
    }

@fastapi_app.post("/api/retrieve")
@gpu_decorator
def api_retrieve(payload: RetrievePayload):
    t_start = time.time()
    try:
        pil_img = decode_base64_image(payload.sketch)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 sketch: {e}")

    sketch_tensor = TFS["sketch_clean"](pil_img).unsqueeze(0).to(DEVICE)
    q_feat = encode_query_representation(sketch_tensor, payload.semantic_clue)

    # Predict concept
    with torch.no_grad():
        adapted_protos = MODEL.adapt_text_prototypes(TEXT_PROTO_125)
        sim_text = (q_feat @ adapted_protos.T).squeeze(0)
        best_idx = sim_text.argmax().item()
        pred_concept = PROTO_CLASSES_125[best_idx]

    # Gallery selection
    if "unseen" in payload.gallery_scope.lower():
        mask = DEFAULT_UNSEEN_MASK
        base_feats = DEFAULT_GALLERY_FEATS[mask]
        base_paths = [p for p, m in zip(DEFAULT_GALLERY_PATHS, mask.tolist()) if m]
        base_classes = [c for c, m in zip(DEFAULT_GALLERY_CLASSES, mask.tolist()) if m]
    else:
        base_feats = DEFAULT_GALLERY_FEATS
        base_paths = DEFAULT_GALLERY_PATHS
        base_classes = DEFAULT_GALLERY_CLASSES

    # Top-K calculation
    with torch.no_grad():
        adapted_gallery = MODEL.encode_photo_from_cached(base_feats)
        sims = (q_feat @ adapted_gallery.T).squeeze(0)
        k = min(payload.top_k, sims.shape[0])
        topk_scores, topk_indices = torch.topk(sims, k=k)

    results = []
    for rank, (score, idx) in enumerate(zip(topk_scores, topk_indices), start=1):
        idx_val = idx.item()
        c_name = base_classes[idx_val]
        p_path = base_paths[idx_val]
        sim_val = float(score.item())
        real_path = resolve_photo_path(p_path, c_name)
        thumb = image_to_base64_thumb(real_path or p_path)
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
@gpu_decorator
def api_classify(payload: ClassifyPayload):
    t_start = time.time()
    try:
        pil_img = decode_base64_image(payload.sketch)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 sketch: {e}")

    sketch_tensor = TFS["sketch_clean"](pil_img).unsqueeze(0).to(DEVICE)
    q_feat = encode_query_representation(sketch_tensor, payload.semantic_clue)

    with torch.no_grad():
        adapted_protos = MODEL.adapt_text_prototypes(OPEN_WORLD_PROTOS)
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
# 5. GRADIO APPLICATION INTERFACE
# -----------------------------------------------------------------------------

@gpu_decorator
def gradio_retrieve(sketch_input, sketch_upload, scope, clue):
    t_start = time.time()
    active_sketch = sketch_upload if sketch_upload is not None else sketch_input
    sketch_tensor = extract_sketch_tensor(active_sketch)
    if sketch_tensor is None:
        return "Chưa có nét vẽ", "--", "0.0 ms", []

    q_feat = encode_query_representation(sketch_tensor, clue)
    with torch.no_grad():
        adapted_protos = MODEL.adapt_text_prototypes(TEXT_PROTO_125)
        sim_text = (q_feat @ adapted_protos.T).squeeze(0)
        best_idx = sim_text.argmax().item()
        pred_label = translate_concept(PROTO_CLASSES_125[best_idx])

    if "Unseen" in scope:
        mask = DEFAULT_UNSEEN_MASK
        base_feats = DEFAULT_GALLERY_FEATS[mask]
        base_paths = [p for p, m in zip(DEFAULT_GALLERY_PATHS, mask.tolist()) if m]
        base_classes = [c for c, m in zip(DEFAULT_GALLERY_CLASSES, mask.tolist()) if m]
    else:
        base_feats = DEFAULT_GALLERY_FEATS
        base_paths = DEFAULT_GALLERY_PATHS
        base_classes = DEFAULT_GALLERY_CLASSES

    with torch.no_grad():
        adapted_gallery = MODEL.encode_photo_from_cached(base_feats)
        sims = (q_feat @ adapted_gallery.T).squeeze(0)
        top10_scores, top10_indices = torch.topk(sims, k=min(10, sims.shape[0]))

    results_grid = []
    for rank, (score, idx) in enumerate(zip(top10_scores, top10_indices), start=1):
        idx_val = idx.item()
        p_path = base_paths[idx_val]
        c_name = translate_concept(base_classes[idx_val])
        sim_val = float(score.item() * 100.0)
        caption = f"#{rank} {c_name} | {sim_val:.1f}%"
        if Path(p_path).exists():
            results_grid.append((p_path, caption))
        else:
            # Generate placeholder if photo not stored locally
            ph = Image.new("RGB", (200, 200), (30, 41, 59))
            results_grid.append((ph, caption))

    max_sim_text = f"{float(top10_scores[0].item() * 100.0):.1f}%" if len(top10_scores) > 0 else "--"
    elapsed = f"{(time.time() - t_start) * 1000.0:.1f} ms"
    return pred_label, max_sim_text, elapsed, results_grid

@gpu_decorator
def gradio_classify(sketch_input, sketch_upload, clue):
    t_start = time.time()
    active_sketch = sketch_upload if sketch_upload is not None else sketch_input
    sketch_tensor = extract_sketch_tensor(active_sketch)
    if sketch_tensor is None:
        return "Chưa có nét vẽ", "--", "<p>Vui lòng vẽ nét trên canvas.</p>", "0.0 ms"

    q_feat = encode_query_representation(sketch_tensor, clue)
    with torch.no_grad():
        adapted_protos = MODEL.adapt_text_prototypes(OPEN_WORLD_PROTOS)
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
                choices=["Tập Unseen (2,100 ảnh - Zero-Shot chuẩn)", "Toàn bộ kho (12,500 ảnh)"],
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
        outputs=[res_concept, res_sim, res_time, res_gallery]
    )
    btn_cls.click(
        fn=gradio_classify,
        inputs=[sketchpad, upload_sketch, semantic_clue],
        outputs=[cls_label, cls_conf, cls_breakdown, cls_time]
    )

# Mount Gradio onto FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
