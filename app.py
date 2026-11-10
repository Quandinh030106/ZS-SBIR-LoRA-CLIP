#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: app.py
ROLE: Interactive Web System for Zero-Shot & Open-World SBIR (Apple HIG Edition)
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
Features:
  1. Apple Human Interface Guidelines (HIG) Minimalist Clean Design.
  2. Dual Studio Mode:
     - Mode 1: Visual Retrieval (Top 10 photos from 12,500 gallery).
     - Mode 2: Concept Guesser (Pictionary AI - guesses object from sketch + optional clues).
  3. Ephemeral In-Memory Session Gallery (zero disk bloat, multi-user isolation).
  4. Multimodal Semantic Clue Fusion (supports Vietnamese semantic descriptions).
  5. Hot-swappable model architectures & private .pt checkpoint registration.
================================================================================
"""

import os
import sys
import json
import shutil
import time
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import clip
import gradio as gr

# Project root setup
proj_root = Path(__file__).resolve().parent
sys.path.insert(0, str(proj_root))

from src.config import SKETCHY_SPLIT2_UNSEEN_21
from src.data.preprocessor import SketchCVPreprocessor
from src.data.transforms import build_transforms
from src.models.clip_lora import AsymmetricHybridLoRACLIP


# ─────────────────────────────────────────────────────────────────────────────
# 1. GLOBAL INITIALIZATION & CACHE LOADING
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Initializing ZS-SBIR Studio Engine on {DEVICE}...")

# 1.1 Load Frozen Base CLIP
CLIP_BACKBONE = "ViT-B/32"
BASE_CLIP, _ = clip.load(CLIP_BACKBONE, device=DEVICE, jit=False)
BASE_CLIP = BASE_CLIP.float()
BASE_CLIP.eval()

# 1.2 Load Pre-computed Default Gallery Cache
CACHE_FILE = proj_root / "outputs" / "cache" / "gallery_index.pt"
if not CACHE_FILE.exists():
    raise FileNotFoundError(
        f"Gallery index cache not found at {CACHE_FILE}. Run 'python scripts/build_gallery_cache.py' first!"
    )

print(f"[SYSTEM] Loading gallery cache from {CACHE_FILE}...")
cache_data = torch.load(str(CACHE_FILE), map_location="cpu", weights_only=False)

DEFAULT_GALLERY_FEATS = cache_data["features"].float().to(DEVICE)      # (12500, 512) Float32
DEFAULT_GALLERY_PATHS = cache_data["paths"]                     # List[str]
DEFAULT_GALLERY_CLASSES = cache_data["class_names"]             # List[str]
DEFAULT_UNSEEN_MASK = cache_data["is_unseen"].to(DEVICE)        # Tensor[bool] (12500,)
CLASSES_125 = cache_data["classes"]                             # 125 class names

print(f"[SYSTEM] Default Gallery: {DEFAULT_GALLERY_FEATS.shape[0]:,} photos ({DEFAULT_UNSEEN_MASK.sum().item():,} Unseen).")

# 1.3 Bilingual Dictionary for Concepts (English -> Vietnamese)
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
    "horse": "Con ngựa", "hot-air_balloon": "Khinh khí cầu", "house": "Ngôi nhà / Tòa nhà",
    "kangaroo": "Chuột túi", "knife": "Con dao", "lion": "Sư tử", "lizard": "Thằn lằn",
    "lobster": "Tôm hùm", "monkey": "Con khỉ", "moon": "Mặt trăng", "motorcycle": "Xe máy / Mô tô",
    "mountain": "Ngọn núi", "mouse": "Con chuột", "mushroom": "Cây nấm", "owl": "Cú mèo",
    "palm_tree": "Cây cọ / Cây dừa", "parrot": "Con vẹt", "pear": "Quả lê", "penguin": "Chim cánh cụt",
    "piano": "Đàn piano", "pig": "Con heo / lợn", "pillow": "Cái gối", "pizza": "Bánh Pizza",
    "rabbit": "Con thỏ", "raccoon": "Gấu mèo", "radio": "Máy radio", "rifle": "Khẩu súng trường",
    "scissors": "Cái kéo", "scorpion": "Bọ cạp", "sea_turtle": "Rùa biển", "shark": "Cá mập",
    "sheep": "Con cừu", "shoe": "Đôi giày", "skyscraper": "Tòa nhà chọc trời", "snail": "Ốc sên",
    "snake": "Con rắn", "spider": "Con nhện", "spoon": "Cái thìa / muỗng", "star": "Ngôi sao",
    "strawberry": "Quả dâu tây", "sun": "Mặt trời", "swan": "Thiên nga", "sword": "Thanh kiếm",
    "t-shirt": "Áo thun", "table": "Cái bàn", "teapot": "Ấm trà", "teddy_bear": "Gấu bông",
    "tiger": "Con hổ", "tree": "Cái cây", "trumpet": "Kèn trumpet", "umbrella": "Cây ô / dù",
    "violin": "Đàn violin", "volcano": "Núi lửa", "watch": "Đồng hồ đeo tay", "wheelchair": "Xe lăn",
    "windmill": "Cối xay gió", "window": "Cửa sổ", "zebra": "Ngựa vằn",
}

def translate_concept(english_name: str) -> str:
    """Returns bilingual string 'English (Vietnamese)'."""
    key = english_name.lower().replace(" ", "_")
    vi = CONCEPT_VIETNAMESE_MAP.get(key, None)
    if not vi:
        # Check subword or split
        for k, v in CONCEPT_VIETNAMESE_MAP.items():
            if k in key or key in k:
                vi = v
                break
    clean_en = english_name.replace("_", " ").title()
    return f"{clean_en} ({vi})" if vi else clean_en

# 1.4 Pre-compute Text Prototypes for Zero-Shot Recognition
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

# Broad Open-World concepts vocabulary
EXTENDED_CONCEPTS = sorted(list(set(CLASSES_125 + [
    "guitar", "laptop", "smartphone", "drone", "submarine", "train", "motorcycle",
    "robot", "skyscraper", "cactus", "pizza", "coffee_cup", "shoe", "t-shirt",
    "glasses", "wristwatch", "flower", "tree", "mountain", "bridge", "book",
    "castle", "tent", "barn", "apartment", "villa", "cottage", "helicopter"
])))
OPEN_WORLD_PROTOS, OPEN_WORLD_NAMES = build_text_prototypes(EXTENDED_CONCEPTS)

def remove_diacritics(text: str) -> str:
    """Strip Vietnamese accent marks to handle unaccented queries like 'mot loai non'."""
    norm = unicodedata.normalize('NFD', text)
    stripped = ''.join(c for c in norm if unicodedata.category(c) != 'Mn')
    return stripped.replace('đ', 'd').replace('Đ', 'D')

# 1.5 Semantic Clue Translation Bridge (Vietnamese -> Multimodal English Prompting)
VIETNAMESE_KEYWORD_BRIDGE = [
    (r"\b(non|mu|mu len|non la|mu bao hiem|mu luoi trai|mu cao boi|chiec non|cai non|chiec mu|cai mu)\b", "a hat, cap, headwear, fedora, bonnet"),
    (r"\b(pizza|banh pizza)\b", "a round baked pizza pie with cheese and savory toppings"),
    (r"\b(khi cau|khinh khi cau)\b", "an aerial blimp, airship, zeppelin, dirigible"),
    (r"\b(banh mi|o banh mi|banh)\b", "a loaf of bread, bakery bun, baguette"),
    (r"\b(thuyen|thuyen buom|tau thuy|thuyen rong)\b", "a sailboat, sailing boat, vessel on ocean water"),
    (r"\b(nui lua|nui|ngon nui)\b", "a volcano with crater, mountain peak"),
    (r"\b(trai le|qua le|le)\b", "a fresh yellow pear fruit"),
    (r"\b(noi nguoi o|cho o|nha|chung cu|toa nha|ngoi nha)\b", "a residential house, apartment building where people live"),
    (r"\b(rat lon|to lon|khong lo|do so)\b", "huge, massive, very large size, monumental structure"),
    (r"\b(2 banh|hai banh|xe may|xe dap)\b", "a two-wheeled vehicle, bicycle, motorcycle"),
    (r"\b(4 banh|bon banh|o to|xe hoi|xe o to)\b", "a four-wheeled vehicle, automobile, car"),
    (r"\b(dong vat|con vat|thu cung|cho|meo)\b", "an animal, mammal creature, domestic pet"),
    (r"\b(cay|cay coi|thuc vat|hoa qua|trai cay)\b", "a tree, botanical plant, natural flora"),
    (r"\b(chim|loai chim|bay|canh)\b", "a bird with wings, flying animal, songbird"),
    (r"\b(may bay|phi co)\b", "an airplane, jetliner, passenger aircraft"),
    (r"\b(dong ho|dong ho bao thuc|dong ho deo tay)\b", "an alarm clock, wristwatch, timepiece"),
    (r"\b(ghe|cai ghe|ghe ngoi|sofa)\b", "furniture to sit on, chair, armchair, bench"),
    (r"\b(ban|cai ban)\b", "a wooden table, desk furniture"),
    (r"\b(keo|cai keo|dao|cat)\b", "cutting tool, sharp scissors, knife"),
    (r"\b(ly|coc|tach|uong nuoc|tach tra)\b", "a cup, coffee mug, tea cup, drinking container"),
]

def enrich_semantic_clue(raw_clue: Optional[str]) -> Optional[str]:
    """Expands Vietnamese semantic descriptions into rich multimodal English prompts."""
    if not raw_clue or not raw_clue.strip():
        return None
    text = raw_clue.strip().lower()
    unaccented = remove_diacritics(text)

    expansions = []
    for pattern, eng_desc in VIETNAMESE_KEYWORD_BRIDGE:
        if re.search(pattern, text) or re.search(pattern, unaccented):
            expansions.append(eng_desc)

    if expansions:
        unique_expansions = list(dict.fromkeys(expansions))
        return f"{', '.join(unique_expansions)}"

    # If pure English (ASCII only), keep as is
    try:
        text.encode('ascii')
        return text
    except UnicodeEncodeError:
        return unaccented

# 1.6 Preprocessor & Transforms
PREPROCESSOR = SketchCVPreprocessor(target_size=224, min_contour_area=15.0, enable_morph=False, enable_random_erase=False)
TFS = build_transforms(image_size=224)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL REGISTRY & CHECKPOINT MANAGER
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY_FILE = proj_root / "outputs" / "checkpoints" / "models_registry.json"

def get_registry() -> Dict[str, Dict[str, str]]:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_registry(reg: Dict[str, Dict[str, str]]) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)

LOADED_MODELS: Dict[str, AsymmetricHybridLoRACLIP] = {}

def load_model_from_checkpoint(ckpt_path: Path) -> AsymmetricHybridLoRACLIP:
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", None)
    rank = cfg.lora_rank if cfg else 8
    alpha = cfg.lora_alpha if cfg else 16.0
    targets = cfg.lora_targets if cfg else ("out_proj",)

    model = AsymmetricHybridLoRACLIP(
        clip_model=BASE_CLIP,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_targets=targets,
    ).to(DEVICE)

    if "model_state" in ckpt:
        model.sketch_adapter.load_state_dict(ckpt["model_state"]["sketch_adapter"])
        model.photo_adapter.load_state_dict(ckpt["model_state"]["photo_adapter"])
        model.text_adapter.load_state_dict(ckpt["model_state"]["text_adapter"])
        model.sketch_visual.load_state_dict(ckpt["model_state"]["sketch_visual"])
    model.eval()
    return model

# Initialize Best Model (Model B)
BEST_CKPT_PATH = proj_root / "outputs" / "checkpoints" / "best_seco_lora_clip.pt"
if not BEST_CKPT_PATH.exists():
    BEST_CKPT_PATH = proj_root / "outputs" / "checkpoints" / "best_seco_lora_clip.zip"

OUR_MODEL_KEY = "Robust Semi-Hard (72.18% mAP)"
BASELINE_KEY = "Standard Baseline (69.68% mAP)"
ENSEMBLE_KEY = "Dual Ensemble (73.05% mAP)"

if BEST_CKPT_PATH.exists():
    print(f"[SYSTEM] Loading Best Model: {OUR_MODEL_KEY}...")
    LOADED_MODELS[OUR_MODEL_KEY] = load_model_from_checkpoint(BEST_CKPT_PATH)

reg = get_registry()
if OUR_MODEL_KEY not in reg:
    reg[OUR_MODEL_KEY] = {
        "path": str(BEST_CKPT_PATH),
        "method": "Asymmetric LoRA (Rank 8, Vision Cross-Attn)",
        "loss": "Batch Semi-Hard InfoNCE (Margin 0.20)",
        "metrics": "72.18% mAP@all, 68.62% mAP@200",
        "description": "In-Batch Semi-Hard Mining + Partial Masking + Stroke Jitter + Zero-Init Adapter"
    }
    save_registry(reg)

for name, meta in reg.items():
    if name not in LOADED_MODELS and Path(meta["path"]).exists():
        try:
            LOADED_MODELS[name] = load_model_from_checkpoint(Path(meta["path"]))
            print(f"[SYSTEM] Registered from disk: {name}")
        except Exception as e:
            print(f"[WARNING] Could not load {name}: {e}")

# Technical specifications dictionary for live UI display
MODEL_SPECS_DATA = {
    OUR_MODEL_KEY: {
        "method": "Asymmetric LoRA (Rank 8, Vision Cross-Attn)",
        "loss": "Batch Semi-Hard InfoNCE (Margin 0.20)",
        "metrics": "72.18% mAP@all (Sketchy) • 57.51% (TU-Berlin)"
    },
    BASELINE_KEY: {
        "method": "Symmetric LoRA (Rank 8, Standard Text+Vision)",
        "loss": "Classic Triplet Margin Ranking Loss",
        "metrics": "69.68% mAP@all"
    },
    ENSEMBLE_KEY: {
        "method": "Dual-Expert Late Fusion (0.6 Robust + 0.4 Baseline)",
        "loss": "Multi-Task Representation Ensemble",
        "metrics": "73.05% mAP@all"
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. CORE INFERENCE, CONCEPT GUESSER & EPHEMERAL GALLERY
# ─────────────────────────────────────────────────────────────────────────────

def extract_sketch_tensor(sketch_input: Optional[Any]) -> Optional[torch.Tensor]:
    """Helper to convert sketchpad input or uploaded file to preprocessed normalized tensor."""
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

    # Handle RGBA with transparent alpha by compositing on solid white
    if isinstance(raw_img, Image.Image) and raw_img.mode == "RGBA":
        white_bg = Image.new("RGBA", raw_img.size, (255, 255, 255, 255))
        raw_img = Image.alpha_composite(white_bg, raw_img).convert("RGB")
    else:
        raw_img = raw_img.convert("RGB")

    clean_sketch = PREPROCESSOR(raw_img)
    tensor = TFS["sketch_clean"](clean_sketch).unsqueeze(0).to(DEVICE)
    return tensor


def encode_query_representation(
    sketch_tensor: torch.Tensor,
    selected_model_name: str,
    semantic_clue: Optional[str] = None
) -> torch.Tensor:
    """Encodes sketch + optional semantic clue with multimodal fusion."""
    with torch.no_grad():
        if selected_model_name == ENSEMBLE_KEY:
            m_ours = LOADED_MODELS.get(OUR_MODEL_KEY, next(iter(LOADED_MODELS.values())))
            alt_key = next((k for k in LOADED_MODELS.keys() if k != OUR_MODEL_KEY), None)
            m_alt = LOADED_MODELS.get(alt_key, m_ours)

            f_ours = m_ours.encode_sketch(sketch_tensor)
            f_alt = m_alt.encode_sketch(sketch_tensor)
            q_feat = 0.6 * f_ours + 0.4 * f_alt
            q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
        else:
            active_model = LOADED_MODELS.get(selected_model_name, next(iter(LOADED_MODELS.values())))
            q_feat = active_model.encode_sketch(sketch_tensor)
            q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)

        # Optional Semantic Clue Fusion
        enriched_clue = enrich_semantic_clue(semantic_clue)
        if enriched_clue:
            txt_tok = clip.tokenize([enriched_clue]).to(DEVICE)
            txt_feat = BASE_CLIP.encode_text(txt_tok)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            # Multimodal Fusion: 65% sketch geometry + 35% text prior
            q_feat = 0.65 * q_feat + 0.35 * txt_feat
            q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)

    return q_feat


# --- 3.1 EPHEMERAL IN-MEMORY SESSION GALLERY ---

def append_ephemeral_gallery(
    files: Optional[List[Any]],
    session_state: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Encodes newly uploaded photos directly in RAM.
    NEVER writes to server disk. Isolated to current browser session.
    """
    if session_state is None:
        session_state = {"feats": [], "paths": [], "classes": []}

    if not files:
        count = len(session_state.get("paths", []))
        return session_state, f"Index: 12,500 base photos • + {count} custom photos in session"

    tfs = TFS["photo_clean"]
    new_count = 0

    with torch.no_grad():
        for file_obj in files:
            path_str = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
            try:
                img = Image.open(path_str).convert("RGB")
                tensor = tfs(img).unsqueeze(0).to(DEVICE)
                feat = BASE_CLIP.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                stem = Path(path_str).stem.split("_")[0].replace("-", " ")

                session_state["feats"].append(feat.squeeze(0).cpu())
                session_state["paths"].append(path_str)
                session_state["classes"].append(stem)
                new_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to encode session photo {path_str}: {e}")

    total_custom = len(session_state["paths"])
    status_msg = f"Index: 12,500 base photos • + {total_custom} custom photos in session (Encoded {new_count} new in RAM)"
    return session_state, status_msg


def reset_ephemeral_gallery() -> Tuple[Dict[str, Any], str]:
    """Instantly clears all in-memory custom photos for this session."""
    empty_state = {"feats": [], "paths": [], "classes": []}
    return empty_state, "Index: 12,500 base photos • + 0 custom photos in session (Reset clean)"


# --- 3.2 VISUAL RETRIEVAL PIPELINE (MODE 1) ---

def run_visual_retrieval(
    sketch_input: Optional[Any],
    sketch_upload: Optional[Any],
    selected_model: str,
    gallery_scope: str,
    semantic_clue: Optional[str],
    session_gallery: Optional[Dict[str, Any]] = None,
    retrieval_mode: str = "🤝 Dual-Expert Consensus (AI Hội chẩn)"
) -> Tuple[str, str, str, str, List[Tuple[str, str]]]:
    """Executes Top 10 Visual Retrieval with Dual-Expert Consensus Architecture."""
    t_start = time.time()
    active_sketch = sketch_upload if sketch_upload is not None else sketch_input
    sketch_tensor = extract_sketch_tensor(active_sketch)

    if sketch_tensor is None:
        empty_badge = "<div style='color:#86868B;font-size:12px;'>Vui lòng vẽ nét phác thảo trên canvas để bắt đầu!</div>"
        return (
            "Vui lòng vẽ nét phác thảo trên canvas!",
            "Chưa có dữ liệu",
            empty_badge,
            "0 ms",
            []
        )

    # 1. Encode query
    active_model = LOADED_MODELS.get(selected_model, next(iter(LOADED_MODELS.values())))
    q_feat = encode_query_representation(sketch_tensor, selected_model, semantic_clue)

    # 2. Zero-Shot Concept Prediction (using trained text adapter)
    with torch.no_grad():
        adapted_text_protos = active_model.adapt_text_prototypes(TEXT_PROTO_125)
        sim_text = (q_feat @ adapted_text_protos.T).squeeze(0)
        best_idx = sim_text.argmax().item()
        best_en = PROTO_CLASSES_125[best_idx]
        best_conf = float(sim_text[best_idx].cpu().item() * 100.0)
        bilingual_label = translate_concept(best_en)

    if best_conf >= 25.0:
        conf_badge = f"Rất cao ({best_conf:.1f}%)"
    elif best_conf >= 20.0:
        conf_badge = f"Khá ({best_conf:.1f}%)"
    else:
        conf_badge = f"Trung bình ({best_conf:.1f}%)"

    pred_display = f"{bilingual_label}"

    # 3. Assemble Active Gallery (Default + Session in-memory)
    if gallery_scope == "Unseen Only (2,100 photos - Zero-Shot)":
        mask = DEFAULT_UNSEEN_MASK
        base_feats = DEFAULT_GALLERY_FEATS[mask]
        base_paths = [p for p, m in zip(DEFAULT_GALLERY_PATHS, mask.tolist()) if m]
        base_classes = [c for c, m in zip(DEFAULT_GALLERY_CLASSES, mask.tolist()) if m]
    else:
        base_feats = DEFAULT_GALLERY_FEATS
        base_paths = DEFAULT_GALLERY_PATHS
        base_classes = DEFAULT_GALLERY_CLASSES

    # Append ephemeral session photos if any
    if session_gallery and session_gallery.get("feats"):
        custom_feats = torch.stack(session_gallery["feats"], dim=0).to(DEVICE)
        raw_gallery_feats = torch.cat([base_feats, custom_feats], dim=0)
        active_gallery_paths = base_paths + session_gallery["paths"]
        active_gallery_classes = base_classes + session_gallery["classes"]
    else:
        raw_gallery_feats = base_feats
        active_gallery_paths = base_paths
        active_gallery_classes = base_classes

    # 4. Top 10 Cosine Ranking through Photo Adapter with Dual-Expert Consensus
    with torch.no_grad():
        adapted_gallery_feats = active_model.encode_photo_from_cached(raw_gallery_feats)
        raw_sims = (q_feat @ adapted_gallery_feats.T).squeeze(0)

        # Concept Probabilities from Trained Text Prototypes
        concept_probs = F.softmax(sim_text * 25.0, dim=-1)
        prob_map = {cls_name.lower(): float(p.item()) for cls_name, p in zip(PROTO_CLASSES_125, concept_probs)}
        max_p = max(prob_map.values()) if prob_map else 1.0

        if "Dual-Expert" in retrieval_mode:
            # Map each gallery photo to its concept probability
            class_priors = torch.tensor(
                [prob_map.get(str(c).lower(), 1e-4) for c in active_gallery_classes],
                device=raw_sims.device,
                dtype=raw_sims.dtype
            )
            # Relative confidence ratio
            rel_prior = (class_priors / max_p).clamp(min=0.01)
            # Gated ranking score: Instance similarity modulated by concept prior
            final_sims = raw_sims * torch.pow(rel_prior, 0.40)
            consensus_badge = f"""
            <div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;padding:10px 14px;margin-top:10px;font-size:12.5px;color:#166534;'>
                <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
                    <span style='background:#16A34A;color:#FFFFFF;padding:2px 7px;border-radius:6px;font-size:10.5px;font-weight:700;'>AI 1 (Doodle/Concept)</span>
                    <span>Đoán ý niệm chủ đạo: <b>{bilingual_label}</b> ({best_conf:.1f}%)</span>
                </div>
                <div style='display:flex;align-items:center;gap:8px;'>
                    <span style='background:#2563EB;color:#FFFFFF;padding:2px 7px;border-radius:6px;font-size:10.5px;font-weight:700;'>AI 2 (Sketchy/Instance)</span>
                    <span>So khớp hình học vi mô & loại bỏ ảnh rác khác lớp trong kho 12.500 ảnh</span>
                </div>
            </div>
            """
        else:
            final_sims = raw_sims
            consensus_badge = f"""
            <div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:10px 14px;margin-top:10px;font-size:12px;color:#475569;'>
                <span style='background:#64748B;color:#FFFFFF;padding:2px 7px;border-radius:6px;font-size:10.5px;font-weight:700;'>Single Model</span>
                <span>Chế độ đơn mô hình: Xếp hạng thuần túy khoảng cách Cosine hình học (không hội chẩn).</span>
            </div>
            """

        top10_scores, top10_indices = torch.topk(final_sims, k=min(10, final_sims.shape[0]))

    results_grid: List[Tuple[str, str]] = []
    for rank, (idx, score) in enumerate(zip(top10_indices.tolist(), top10_scores.tolist()), start=1):
        p = active_gallery_paths[idx]
        c = translate_concept(active_gallery_classes[idx])
        sim_val = float(score)
        tag = f"#{rank} • {sim_val:.3f} | {c}"
        results_grid.append((p, tag))

    elapsed_ms = f"{(time.time() - t_start) * 1000.0:.1f} ms (PyTorch)"
    return pred_display, conf_badge, consensus_badge, elapsed_ms, results_grid


# --- 3.3 CONCEPT GUESSER PIPELINE (MODE 2 - PICTIONARY AI) ---

def run_concept_guesser(
    sketch_input: Optional[Any],
    sketch_upload: Optional[Any],
    selected_model: str,
    semantic_clue: Optional[str]
) -> Tuple[str, str, str, str]:
    """
    Pictionary AI Mode:
    Predicts what object is being drawn from sketch + optional text clue.
    Completely independent of gallery photos.
    Returns:
      1. Hero Top 1 Prediction
      2. Confidence Badge (Vietnamese)
      3. HTML breakdown table for Top 5 candidates with progress bars
      4. Elapsed time
    """
    t_start = time.time()
    active_sketch = sketch_upload if sketch_upload is not None else sketch_input
    sketch_tensor = extract_sketch_tensor(active_sketch)

    if sketch_tensor is None:
        empty_html = "<div style='color:#86868B;padding:16px;text-align:center;'>Vui lòng vẽ nét phác thảo trên canvas để AI bắt đầu đoán!</div>"
        return "Chưa có nét vẽ", "Đang chờ...", empty_html, "0 ms"

    # Encode query with multimodal fusion
    active_model = LOADED_MODELS.get(selected_model, next(iter(LOADED_MODELS.values())))
    q_feat = encode_query_representation(sketch_tensor, selected_model, semantic_clue)

    # Score against Extended Concepts using Trained Text Adapter
    with torch.no_grad():
        adapted_open_protos = active_model.adapt_text_prototypes(OPEN_WORLD_PROTOS)
        sim_scores = (q_feat @ adapted_open_protos.T).squeeze(0)
        # Calibrated temperature scaling for confident, clean probabilities
        probs = F.softmax(sim_scores * 35.0, dim=-1)
        top5_probs, top5_indices = torch.topk(probs, k=5)

    top5_probs = top5_probs.cpu().tolist()
    top5_indices = top5_indices.cpu().tolist()

    top1_en = OPEN_WORLD_NAMES[top5_indices[0]]
    top1_label = translate_concept(top1_en)
    top1_pct = top5_probs[0] * 100.0

    if top1_pct >= 60.0:
        conf_text = f"Rất cao ({top1_pct:.1f}%)"
    elif top1_pct >= 35.0:
        conf_text = f"Cao ({top1_pct:.1f}%)"
    elif top1_pct >= 18.0:
        conf_text = f"Trung bình ({top1_pct:.1f}%)"
    else:
        conf_text = f"Thấp ({top1_pct:.1f}%)"

    # Build Apple-style HTML ranking card
    clue_note = f"<div style='font-size:12px;color:#86868B;margin-top:4px;'>Gợi ý áp dụng: <em>{semantic_clue}</em></div>" if semantic_clue and semantic_clue.strip() else ""

    html_rows = []
    for rank, (idx, prob) in enumerate(zip(top5_indices, top5_probs), start=1):
        c_en = OPEN_WORLD_NAMES[idx]
        bilingual = translate_concept(c_en)
        pct = prob * 100.0
        bar_color = "#34C759" if rank == 1 else "#0071E3" if rank == 2 else "#86868B"
        row_html = f"""
        <div style="margin-bottom: 12px;">
          <div style="display: flex; justify-content: space-between; font-size: 13.5px; margin-bottom: 4px;">
            <span style="font-weight: 600; color: #1D1D1F;">
              <span style="color: #86868B; width: 20px; display: inline-block;">#{rank}</span>
              {bilingual}
            </span>
            <span style="font-weight: 600; color: #1D1D1F; font-feature-settings: 'tnum';">{pct:.1f}%</span>
          </div>
          <div style="height: 6px; background: #EBEBED; border-radius: 999px; overflow: hidden;">
            <div style="height: 100%; width: {pct:.1f}%; background: {bar_color}; border-radius: 999px; transition: width 0.3s ease;"></div>
          </div>
        </div>
        """
        html_rows.append(row_html)

    breakdown_html = f"""
    <div style="background: #FFFFFF; border: 1px solid rgba(0,0,0,0.08); border-radius: 14px; padding: 18px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
      <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #86868B; margin-bottom: 14px;">
        Top 5 Khái niệm dự đoán (Xác suất mở rộng)
      </div>
      {''.join(html_rows)}
      {clue_note}
    </div>
    """

    elapsed_ms = f"{(time.time() - t_start) * 1000.0:.1f} ms"
    return top1_label, conf_text, breakdown_html, elapsed_ms


# --- 3.4 CHECKPOINT REGISTRATION ---

def register_new_checkpoint(
    model_name: str,
    tech_description: str,
    ckpt_file: Optional[Any],
    save_permanently: bool,
) -> Tuple[str, gr.Dropdown]:
    """Registers a new checkpoint with mandatory name & technical description."""
    if not model_name or not model_name.strip():
        return "Lỗi: Vui lòng nhập Tên mô hình duy nhất!", gr.Dropdown()

    model_name = model_name.strip()
    reg = get_registry()

    if model_name in LOADED_MODELS or model_name in reg:
        return f"Lỗi: Tên mô hình '{model_name}' đã tồn tại! Vui lòng chọn tên khác biệt.", gr.Dropdown()

    if not tech_description or not tech_description.strip():
        return "Lỗi: Vui lòng nhập Mô tả kỹ thuật (Technical Description) của mô hình!", gr.Dropdown()

    if ckpt_file is None:
        return "Lỗi: Vui lòng tải lên file checkpoint .pt hoặc .zip!", gr.Dropdown()

    file_path = ckpt_file.name if hasattr(ckpt_file, "name") else str(ckpt_file)

    try:
        loaded_model = load_model_from_checkpoint(Path(file_path))
        LOADED_MODELS[model_name] = loaded_model

        if save_permanently:
            safe_filename = "".join(c for c in model_name if c.isalnum() or c in ("-", "_")).lower() + ".pt"
            dest_path = proj_root / "outputs" / "checkpoints" / safe_filename
            shutil.copyfile(file_path, str(dest_path))

            reg[model_name] = {
                "path": str(dest_path),
                "method": "Custom Checkpoint",
                "loss": "User Configured",
                "metrics": "User Uploaded",
                "description": tech_description.strip(),
            }
            save_registry(reg)
            msg = f"Thành công: Đã đăng ký & LƯU VĨNH VIỄN '{model_name}'!"
        else:
            msg = f"Thành công: Đã nạp TẠM THỜI '{model_name}' vào phiên hiện tại!"

        all_choices = list(LOADED_MODELS.keys()) + [ENSEMBLE_KEY]
        return msg, gr.Dropdown(choices=all_choices, value=model_name)

    except Exception as e:
        return f"Lỗi nạp checkpoint: {str(e)}", gr.Dropdown()


# ─────────────────────────────────────────────────────────────────────────────
# 4. GRADIO WEB INTERFACE (APPLE HIG AESTHETICS)
# ─────────────────────────────────────────────────────────────────────────────

apple_css = """
:root {
    --bg-canvas: #F5F5F7;
    --surface-primary: #FFFFFF;
    --text-primary: #1D1D1F;
    --text-secondary: #6E6E73;
    --accent-system: #0071E3;
}

body, .gradio-container {
    background-color: #F5F5F7 !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif !important;
    max-width: 1360px !important;
    margin: 0 auto !important;
}

/* Minimalist Nav Header */
.apple-header {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 16px;
    padding: 14px 24px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Apple Cards */
.apple-card {
    background: #FFFFFF !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03) !important;
    padding: 20px !important;
}

/* Action Primary Button */
.btn-apple {
    background-color: #0071E3 !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.btn-apple:hover {
    background-color: #0077ED !important;
    box-shadow: 0 4px 12px rgba(0, 113, 227, 0.3) !important;
}

/* Quiet Secondary Buttons */
.btn-quiet {
    border-radius: 8px !important;
    font-weight: 500 !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
}
"""

with gr.Blocks(title="ZS-SBIR Studio — Apple HIG Edition") as demo:

    # Ephemeral Session State (in-memory custom gallery photos)
    session_gallery_state = gr.State({"feats": [], "paths": [], "classes": []})

    # ── HEADER ──
    with gr.Row(elem_classes="apple-header"):
        with gr.Column(scale=8):
            gr.Markdown(
                """
                <div style="display: flex; align-items: baseline; gap: 12px;">
                    <span style="font-size: 20px; font-weight: 700; letter-spacing: -0.4px; color: #1D1D1F;">ZS-SBIR Studio</span>
                    <span style="font-size: 13px; color: #86868B; border-left: 1px solid rgba(0,0,0,0.12); padding-left: 10px;">Asymmetric LoRA-CLIP (ViT-B/32)</span>
                </div>
                """
            )
        with gr.Column(scale=4):
            gr.Markdown(
                f"""
                <div style="text-align: right; font-size: 12.5px; font-weight: 500; color: #6E6E73;">
                    <span style="color: #34C759; margin-right: 4px;">●</span> Engine Ready ({DEVICE.type.upper()})
                </div>
                """
            )

    # ── MAIN 2-COLUMN STUDIO ──
    with gr.Row():

        # ── LEFT COLUMN: Model Switcher & Canvas Controls ──
        with gr.Column(scale=5, elem_classes="apple-card"):
            
            # 1. Model Selector
            gr.Markdown("<div style='font-size:12px;font-weight:600;text-transform:uppercase;color:#86868B;letter-spacing:0.5px;'>Model Architecture</div>")
            model_choices = list(LOADED_MODELS.keys()) + [ENSEMBLE_KEY]
            model_dropdown = gr.Dropdown(
                choices=model_choices,
                value=OUR_MODEL_KEY if OUR_MODEL_KEY in model_choices else model_choices[0],
                label="Selected Architecture (Zero-Reload Hot-Swap)",
                interactive=True,
            )

            # Live Model Specs Box
            spec_box = gr.Markdown(
                f"""
                <div style="background:#FBFBFD;border:1px solid rgba(0,0,0,0.06);border-radius:10px;padding:10px 14px;font-size:12px;margin-bottom:12px;">
                    <div style="color:#6E6E73;">Method: <strong>{MODEL_SPECS_DATA.get(OUR_MODEL_KEY, {}).get('method', 'LoRA')}</strong></div>
                    <div style="color:#6E6E73;">Objective: <strong>{MODEL_SPECS_DATA.get(OUR_MODEL_KEY, {}).get('loss', 'Semi-Hard')}</strong></div>
                    <div style="color:#34C759;font-weight:600;margin-top:2px;">Metric: {MODEL_SPECS_DATA.get(OUR_MODEL_KEY, {}).get('metrics', '72.18% mAP')}</div>
                </div>
                """
            )

            def update_model_specs(chosen_name):
                info = MODEL_SPECS_DATA.get(chosen_name, {
                    "method": "Custom Registered Checkpoint",
                    "loss": "Custom Objective",
                    "metrics": "Evaluated on demand"
                })
                return f"""
                <div style="background:#FBFBFD;border:1px solid rgba(0,0,0,0.06);border-radius:10px;padding:10px 14px;font-size:12px;margin-bottom:12px;">
                    <div style="color:#6E6E73;">Method: <strong>{info['method']}</strong></div>
                    <div style="color:#6E6E73;">Objective: <strong>{info['loss']}</strong></div>
                    <div style="color:#34C759;font-weight:600;margin-top:2px;">Metric: {info['metrics']}</div>
                </div>
                """

            model_dropdown.change(fn=update_model_specs, inputs=[model_dropdown], outputs=[spec_box])

            # 2. PencilKit-Style Sketchpad
            gr.Markdown("<div style='font-size:12px;font-weight:600;text-transform:uppercase;color:#86868B;letter-spacing:0.5px;margin-top:8px;'>Sketch Canvas (224 × 224)</div>")
            sketchpad = gr.Sketchpad(
                type="pil",
                label="Drawing Pad (Black = Pen, White = Eraser)",
                brush=gr.Brush(colors=["#000000", "#FFFFFF"], default_size=6),
            )

            upload_sketch_tab = gr.Image(
                type="pil",
                label="Or Upload Pre-drawn Sketch File",
                sources=["upload"],
            )

            # 3. Optional Semantic Clue Input
            semantic_clue_input = gr.Textbox(
                placeholder="e.g. 'nơi người ở, rất lớn' hoặc 'two-wheel motorized vehicle' (optional prior bias)",
                label="Optional Semantic Clues / Description",
                lines=1,
            )

            # 4. Mode-specific action buttons
            with gr.Row():
                btn_run_retrieval = gr.Button("Retrieve Similar Images (Top 10)", variant="primary", elem_classes="btn-apple")
                btn_run_guesser = gr.Button("Guess What I'm Drawing (AI)", variant="secondary", elem_classes="btn-quiet")

            # 5. Checkpoint Registration Accordion
            with gr.Accordion("Register New .pt Checkpoint", open=False):
                reg_name = gr.Textbox(label="Model Unique Name *", placeholder="e.g. Asymmetric-LoRA-Rank16")
                reg_desc = gr.Textbox(label="Technical Architecture & Loss *", placeholder="Rank, targeted attention layers, loss formula...", lines=2)
                reg_file = gr.File(label="Upload Weights (.pt)", file_types=[".pt", ".zip"])
                reg_perm = gr.Checkbox(value=False, label="Persist to disk (leave unchecked for temporary session)")
                btn_register = gr.Button("Register & Hot-Swap", variant="secondary", size="sm")
                reg_status = gr.Textbox(label="Registration Status", interactive=False)

        # ── RIGHT COLUMN: Studio Modes & Results ──
        with gr.Column(scale=7):

            with gr.Tabs() as studio_tabs:

                # ── TAB 1: VISUAL RETRIEVAL STUDIO ──
                with gr.TabItem("Visual Retrieval (10 Photos)", id="tab_retrieval"):
                    
                    # Hero status
                    with gr.Row(elem_classes="apple-card", elem_id="retrieval_hero"):
                        with gr.Column(scale=8):
                            gr.Markdown("<div style='font-size:11px;font-weight:600;text-transform:uppercase;color:#86868B;'>Predicted Concept (Zero-Shot)</div>")
                            retrieval_pred_box = gr.Markdown("<div style='font-size:22px;font-weight:600;color:#1D1D1F;'>Đang chờ nét vẽ...</div>")
                        with gr.Column(scale=4):
                            retrieval_conf_box = gr.Markdown("<div style='text-align:right;font-size:13px;color:#34C759;font-weight:600;'>Độ tin cậy: --</div>")
                            retrieval_time_box = gr.Markdown("<div style='text-align:right;font-size:11.5px;color:#86868B;'>0 ms</div>")

                    # Dual-Expert Consensus Feedback Card
                    retrieval_consensus_box = gr.HTML(
                        value="<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:8px 14px;font-size:12px;color:#64748B;'>Hệ thống Hai AI: AI 1 (Doodle/Concept) đoán ý niệm & AI 2 (Sketchy) so khớp hình học chi tiết.</div>"
                    )

                    # Gallery Grid Header & Engine Mode Controls
                    with gr.Row():
                        retrieval_mode_selector = gr.Radio(
                            choices=[
                                "🤝 Dual-Expert Consensus (AI Hội chẩn)",
                                "⚡ Single Model (Raw Geometric)",
                            ],
                            value="🤝 Dual-Expert Consensus (AI Hội chẩn)",
                            label="Retrieval Engine Mode",
                            scale=6,
                        )
                        gallery_scope_selector = gr.Radio(
                            choices=[
                                "Unseen Only (2,100 photos - Zero-Shot)",
                                "Full Gallery (12,500 photos)",
                            ],
                            value="Unseen Only (2,100 photos - Zero-Shot)",
                            label="Search Scope",
                            scale=6,
                        )

                    # Top 10 Photos Grid (5 cols x 2 rows)
                    retrieval_gallery = gr.Gallery(
                        label="Top 10 Retrieved Images",
                        show_label=False,
                        columns=5,
                        rows=2,
                        height="auto",
                        object_fit="cover",
                    )

                    # Ephemeral Session Gallery Box
                    with gr.Group(elem_classes="apple-card", elem_id="gallery_manager"):
                        gr.Markdown("<div style='font-size:13px;font-weight:600;color:#1D1D1F;'>Append Photos to Session Gallery</div>")
                        gr.Markdown("<div style='font-size:12px;color:#86868B;'>Upload image files to encode on-the-fly. Stored in RAM only (never written to disk).</div>")
                        
                        custom_photo_uploader = gr.File(
                            file_count="multiple",
                            file_types=["image"],
                            label="Select or Drop Images Here",
                        )
                        with gr.Row():
                            btn_append = gr.Button("Append to Session (In-Memory)", variant="secondary", size="sm")
                            btn_reset = gr.Button("Reset to Default (Clear RAM)", variant="stop", size="sm")
                        
                        gallery_status_display = gr.Textbox(
                            value="Index: 12,500 base photos • + 0 custom photos in session",
                            label="Session Gallery Status",
                            interactive=False,
                        )

                # ── TAB 2: CONCEPT GUESSER (PICTIONARY AI) ──
                with gr.TabItem("Concept Guesser (Pictionary AI)", id="tab_guesser"):
                    
                    with gr.Group(elem_classes="apple-card"):
                        gr.Markdown("<div style='font-size:11px;font-weight:600;text-transform:uppercase;color:#86868B;'>Top Guessed Concept (Khái niệm dự đoán hàng đầu)</div>")
                        with gr.Row():
                            with gr.Column(scale=8):
                                guesser_hero_text = gr.Markdown("<div style='font-size:24px;font-weight:700;color:#1D1D1F;'>Vẽ nét bất kỳ để AI đoán...</div>")
                            with gr.Column(scale=4):
                                guesser_conf_text = gr.Markdown("<div style='text-align:right;font-size:14px;color:#34C759;font-weight:600;'>Độ tin cậy: --</div>")

                        guesser_breakdown_html = gr.HTML(
                            value="<div style='color:#86868B;padding:16px;text-align:center;'>Vui lòng vẽ nét phác thảo trên canvas để AI bắt đầu đoán!</div>"
                        )
                        guesser_time_text = gr.Markdown("<div style='font-size:11.5px;color:#86868B;text-align:right;margin-top:8px;'>0 ms</div>")

    # ── EVENT BINDINGS ──

    # Event 1: Visual Retrieval Execution (with Dual-Expert Consensus)
    def on_click_retrieval(pad, up, model, scope, clue, mode, state):
        pred_label, conf, consensus_html, t_el, grid = run_visual_retrieval(pad, up, model, scope, clue, mode, state)
        p_html = f"<div style='font-size:22px;font-weight:600;color:#1D1D1F;'>{pred_label}</div>"
        c_html = f"<div style='text-align:right;font-size:13px;color:#34C759;font-weight:600;'>Độ tin cậy: {conf}</div>"
        t_html = f"<div style='text-align:right;font-size:11.5px;color:#86868B;'>{t_el}</div>"
        return p_html, c_html, consensus_html, t_html, grid

    btn_run_retrieval.click(
        fn=on_click_retrieval,
        inputs=[sketchpad, upload_sketch_tab, model_dropdown, gallery_scope_selector, semantic_clue_input, retrieval_mode_selector, session_gallery_state],
        outputs=[retrieval_pred_box, retrieval_conf_box, retrieval_consensus_box, retrieval_time_box, retrieval_gallery],
    )

    # Event 2: Concept Guesser Execution
    def on_click_guesser(pad, up, model, clue):
        hero, conf, b_html, t_el = run_concept_guesser(pad, up, model, clue)
        h_html = f"<div style='font-size:24px;font-weight:700;color:#1D1D1F;'>{hero}</div>"
        c_html = f"<div style='text-align:right;font-size:14px;color:#34C759;font-weight:600;'>Độ tin cậy: {conf}</div>"
        t_html = f"<div style='font-size:11.5px;color:#86868B;text-align:right;margin-top:8px;'>Thời gian xử lý: {t_el}</div>"
        return h_html, c_html, b_html, t_html

    btn_run_guesser.click(
        fn=on_click_guesser,
        inputs=[sketchpad, upload_sketch_tab, model_dropdown, semantic_clue_input],
        outputs=[guesser_hero_text, guesser_conf_text, guesser_breakdown_html, guesser_time_text],
    )

    # Event 3: Append Ephemeral Photos to Session Gallery
    btn_append.click(
        fn=append_ephemeral_gallery,
        inputs=[custom_photo_uploader, session_gallery_state],
        outputs=[session_gallery_state, gallery_status_display],
    )

    # Event 4: Reset Ephemeral Gallery (Purge RAM)
    btn_reset.click(
        fn=reset_ephemeral_gallery,
        inputs=[],
        outputs=[session_gallery_state, gallery_status_display],
    )

    # Event 5: Register Checkpoint (.pt)
    btn_register.click(
        fn=register_new_checkpoint,
        inputs=[reg_name, reg_desc, reg_file, reg_perm],
        outputs=[reg_status, model_dropdown],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
    )
