#!/usr/bin/env python3
"""
Data Validation Script for SBIR Project
========================================
Kiểm định toàn bộ dataset trước khi huấn luyện.
Hỗ trợ: Sketchy Database (256x256), TU-Berlin, QMUL-Shoe/Chair

Chạy lệnh:
    python scripts/validate_data.py --dataset sketchy --data_root data

"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Thiếu thư viện. Chạy: pip install Pillow tqdm")
    sys.exit(1)


# ─────────────────────────────────────────────────────────
# CẤU HÌNH CHO TỪNG DATASET
# ─────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "sketchy": {
        "photo_dirs": [
            "256x256/photo/tx_000000000000",
        ],
        "sketch_dirs": [
            "256x256/splitted_sketches/train/tx_000000000000",
            "256x256/splitted_sketches/test/tx_000000000000",
        ],
        "photo_ext": ".jpg",
        "sketch_ext": ".png",
        "pair_separator": "-",  # sketch: {photo_id}-{num}.png -> photo: {photo_id}.jpg
        "total_classes_expected": 125,
    },
    "tuberlin": {
        "photo_dirs": ["ImageResized"],
        "sketch_dirs": ["png"],
        "photo_ext": ".jpg",
        "sketch_ext": ".png",
        "pair_separator": None,  # Category-level (không ghép cặp instance)
        "total_classes_expected": 250,
    },
    "qmul_shoe": {
        "photo_dirs": ["photo"],
        "sketch_dirs": ["sketch/tx_000000000000"],
        "photo_ext": ".jpg",
        "sketch_ext": ".png",
        "pair_separator": "-",
        "total_classes_expected": 1,  # Single category: shoe
    },
    "qmul_chair": {
        "photo_dirs": ["photo"],
        "sketch_dirs": ["sketch/tx_000000000000"],
        "photo_ext": ".jpg",
        "sketch_ext": ".png",
        "pair_separator": "-",
        "total_classes_expected": 1,  # Single category: chair
    },
}


# ─────────────────────────────────────────────────────────
# HÀM KIỂM TRA TÍNH HỢP LỆ CỦA ẢNH
# ─────────────────────────────────────────────────────────

def check_image(path: Path) -> dict:
    """
    Kiểm tra một file ảnh:
    - File tồn tại & không rỗng
    - Mở được bằng PIL (không bị corrupt)
    - Kích thước và số kênh màu
    """
    result = {
        "path": str(path),
        "ok": False,
        "size_bytes": 0,
        "width": None,
        "height": None,
        "channels": None,
        "mode": None,
        "error": None,
    }

    if not path.exists():
        result["error"] = "FILE_NOT_FOUND"
        return result

    result["size_bytes"] = path.stat().st_size

    if result["size_bytes"] == 0:
        result["error"] = "EMPTY_FILE"
        return result

    try:
        with Image.open(path) as img:
            img.verify()  # Kiểm tra header
        with Image.open(path) as img:
            result["width"], result["height"] = img.size
            result["mode"] = img.mode
            result["channels"] = len(img.getbands())
            result["ok"] = True
    except Exception as e:
        result["error"] = f"CORRUPT: {str(e)[:80]}"

    return result


def check_image_fast(path: Path) -> tuple:
    """Phiên bản nhanh, chỉ trả về (path, ok, error)"""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return (str(path), False, "MISSING_OR_EMPTY")
        with Image.open(path) as img:
            img.verify()
        return (str(path), True, None)
    except Exception as e:
        return (str(path), False, f"CORRUPT: {str(e)[:60]}")


# ─────────────────────────────────────────────────────────
# BƯỚC 1: QUÉT & THU THẬP DANH SÁCH FILE
# ─────────────────────────────────────────────────────────

def collect_files(data_root: Path, cfg: dict) -> dict:
    """Thu thập toàn bộ file ảnh và sketch theo config."""
    print("\n" + "═" * 60)
    print("  BƯỚC 1: Quét cấu trúc thư mục dataset")
    print("═" * 60)

    collected = {
        "photos": {},       # {photo_id: Path}
        "sketches": [],     # [{"path": Path, "photo_id": str, "class": str, "split": str}]
        "classes": set(),
        "class_photo_count": defaultdict(int),
        "class_sketch_count": defaultdict(int),
        "missing_dirs": [],
    }

    # === Thu thập ảnh photo ===
    print("\n[PHOTO] Đang quét thư mục ảnh...")
    for photo_dir_rel in cfg["photo_dirs"]:
        photo_dir = data_root / photo_dir_rel
        if not photo_dir.exists():
            print(f"  ⚠ Không tìm thấy: {photo_dir}")
            collected["missing_dirs"].append(str(photo_dir))
            continue

        class_dirs = sorted([d for d in photo_dir.iterdir() if d.is_dir()])
        print(f"  Tìm thấy: {photo_dir}")
        print(f"  └── {len(class_dirs)} thư mục lớp (classes)")

        for class_dir in class_dirs:
            class_name = class_dir.name
            collected["classes"].add(class_name)

            photo_files = sorted(class_dir.glob(f"*{cfg['photo_ext']}"))
            for pf in photo_files:
                photo_id = pf.stem  # Tên file không có extension
                collected["photos"][photo_id] = pf
                collected["class_photo_count"][class_name] += 1

    print(f"\n  ✔ Tổng ảnh photo: {len(collected['photos']):,}")
    print(f"  ✔ Tổng số lớp:    {len(collected['classes'])}")

    # === Thu thập sketch ===
    print("\n[SKETCH] Đang quét thư mục sketch...")
    for sketch_dir_rel in cfg["sketch_dirs"]:
        sketch_dir = data_root / sketch_dir_rel

        # Xác định split (train/test) từ tên đường dẫn
        split = "train" if "train" in sketch_dir_rel else \
                "test"  if "test"  in sketch_dir_rel else "all"

        if not sketch_dir.exists():
            print(f"  ⚠ Không tìm thấy: {sketch_dir}")
            collected["missing_dirs"].append(str(sketch_dir))
            continue

        class_dirs = sorted([d for d in sketch_dir.iterdir() if d.is_dir()])
        print(f"\n  Tìm thấy [{split.upper()}]: {sketch_dir}")
        print(f"  └── {len(class_dirs)} thư mục lớp")

        for class_dir in class_dirs:
            class_name = class_dir.name
            sketch_files = sorted(class_dir.glob(f"*{cfg['sketch_ext']}"))

            for sf in sketch_files:
                # Trích xuất photo_id từ tên sketch
                # Ví dụ: "n02691156_10151-1.png" -> photo_id = "n02691156_10151"
                if cfg["pair_separator"] and cfg["pair_separator"] in sf.stem:
                    photo_id = sf.stem.rsplit(cfg["pair_separator"], 1)[0]
                else:
                    photo_id = sf.stem

                collected["sketches"].append({
                    "path": sf,
                    "photo_id": photo_id,
                    "class": class_name,
                    "split": split,
                })
                collected["class_sketch_count"][class_name] += 1

    total_train = sum(1 for s in collected["sketches"] if s["split"] == "train")
    total_test  = sum(1 for s in collected["sketches"] if s["split"] == "test")
    print(f"\n  ✔ Tổng sketch (train): {total_train:,}")
    print(f"  ✔ Tổng sketch (test):  {total_test:,}")
    print(f"  ✔ Tổng sketch (all):   {len(collected['sketches']):,}")

    return collected


# ─────────────────────────────────────────────────────────
# BƯỚC 2: KIỂM TRA GHÉP CẶP SKETCH-PHOTO
# ─────────────────────────────────────────────────────────

def validate_pairs(collected: dict) -> dict:
    """Kiểm tra xem mỗi sketch có tìm được ảnh photo tương ứng không."""
    print("\n" + "═" * 60)
    print("  BƯỚC 2: Kiểm tra ghép cặp Sketch ↔ Photo")
    print("═" * 60)

    valid_pairs = []
    orphan_sketches = []  # Sketch không tìm được ảnh tương ứng

    for item in collected["sketches"]:
        photo_id = item["photo_id"]
        if photo_id in collected["photos"]:
            valid_pairs.append({
                "sketch_path":  str(item["path"]),
                "photo_path":   str(collected["photos"][photo_id]),
                "class":        item["class"],
                "split":        item["split"],
                "photo_id":     photo_id,
            })
        else:
            orphan_sketches.append({
                "sketch_path": str(item["path"]),
                "photo_id":    photo_id,
                "class":       item["class"],
            })

    print(f"\n  ✔ Cặp hợp lệ (sketch ↔ photo):  {len(valid_pairs):,}")
    print(f"  ✗ Sketch mồ côi (không có photo): {len(orphan_sketches):,}")

    if orphan_sketches:
        print(f"\n  [CẢNH BÁO] {len(orphan_sketches)} sketch không tìm được ảnh gốc!")
        print("  5 ví dụ đầu tiên:")
        for item in orphan_sketches[:5]:
            print(f"    - {item['sketch_path']}")
            print(f"      → Tìm photo_id: '{item['photo_id']}' nhưng không thấy")

    return {
        "valid_pairs": valid_pairs,
        "orphan_sketches": orphan_sketches,
    }


# ─────────────────────────────────────────────────────────
# BƯỚC 3: KIỂM TRA TÍNH TOÀN VẸN ẢNH (INTEGRITY CHECK)
# ─────────────────────────────────────────────────────────

def validate_integrity(valid_pairs: list, max_workers: int = 8) -> dict:
    """
    Kiểm tra tất cả file ảnh (cả sketch và photo) có mở được không.
    Dùng đa luồng để tăng tốc.
    """
    print("\n" + "═" * 60)
    print("  BƯỚC 3: Kiểm tra toàn vẹn file ảnh (Integrity Check)")
    print("═" * 60)

    # Thu thập tất cả đường dẫn cần kiểm tra (loại bỏ trùng lặp)
    all_paths_to_check = set()
    for pair in valid_pairs:
        all_paths_to_check.add(pair["sketch_path"])
        all_paths_to_check.add(pair["photo_path"])

    all_paths = [Path(p) for p in all_paths_to_check]
    print(f"\n  Đang kiểm tra {len(all_paths):,} file ảnh với {max_workers} luồng...")

    corrupt_files = []
    ok_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_image_fast, p): p for p in all_paths}
        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="  Integrity check", ncols=70):
            path_str, ok, error = future.result()
            if ok:
                ok_count += 1
            else:
                corrupt_files.append({"path": path_str, "error": error})

    print(f"\n  ✔ File hợp lệ:    {ok_count:,}")
    print(f"  ✗ File lỗi/hỏng:  {len(corrupt_files):,}")

    if corrupt_files:
        print(f"\n  [CẢNH BÁO] Phát hiện {len(corrupt_files)} file lỗi:")
        for item in corrupt_files[:10]:
            print(f"    [{item['error']}] {item['path']}")

    return {"corrupt_files": corrupt_files, "ok_count": ok_count}


# ─────────────────────────────────────────────────────────
# BƯỚC 4: KIỂM TRA KỸ THUẬT ẢNH (CHANNEL & SIZE CHECK)
# ─────────────────────────────────────────────────────────

def validate_image_specs(collected: dict, sample_size: int = 500) -> dict:
    """
    Lấy mẫu ngẫu nhiên để kiểm tra kích thước và số kênh màu.
    Không cần quét toàn bộ (tốn thời gian).
    """
    print("\n" + "═" * 60)
    print("  BƯỚC 4: Kiểm tra thông số kỹ thuật ảnh (lấy mẫu)")
    print("═" * 60)

    import random

    photo_paths = random.sample(
        [p for p in collected["photos"].values()],
        min(sample_size, len(collected["photos"]))
    )
    sketch_paths = random.sample(
        [s["path"] for s in collected["sketches"]],
        min(sample_size, len(collected["sketches"]))
    )

    def analyze_batch(paths, label):
        size_counter = defaultdict(int)
        mode_counter = defaultdict(int)
        channel_counter = defaultdict(int)

        for path in tqdm(paths, desc=f"  Phân tích {label}", ncols=70):
            try:
                with Image.open(path) as img:
                    size_counter[img.size] += 1
                    mode_counter[img.mode] += 1
                    channel_counter[len(img.getbands())] += 1
            except:
                pass

        return size_counter, mode_counter, channel_counter

    print(f"\n  [PHOTO] Phân tích {len(photo_paths)} ảnh mẫu:")
    p_sizes, p_modes, p_channels = analyze_batch(photo_paths, "Photo")
    print(f"    Kích thước phổ biến: {dict(sorted(p_sizes.items(), key=lambda x: -x[1])[:3])}")
    print(f"    Mode màu:            {dict(p_modes)}")
    print(f"    Số kênh màu:         {dict(p_channels)}")

    print(f"\n  [SKETCH] Phân tích {len(sketch_paths)} sketch mẫu:")
    s_sizes, s_modes, s_channels = analyze_batch(sketch_paths, "Sketch")
    print(f"    Kích thước phổ biến: {dict(sorted(s_sizes.items(), key=lambda x: -x[1])[:3])}")
    print(f"    Mode màu:            {dict(s_modes)}")
    print(f"    Số kênh màu:         {dict(s_channels)}")

    # Cảnh báo nếu sketch không phải 1 hoặc 3 kênh
    unusual_channels = {k: v for k, v in s_channels.items() if k not in [1, 3]}
    if unusual_channels:
        print(f"\n  [⚠ CẢNH BÁO] Phát hiện sketch có số kênh màu bất thường: {unusual_channels}")
        print("    → Cần kiểm tra và convert về RGB khi preprocessing!")

    return {
        "photo": {"sizes": dict(p_sizes), "modes": dict(p_modes), "channels": dict(p_channels)},
        "sketch": {"sizes": dict(s_sizes), "modes": dict(s_modes), "channels": dict(s_channels)},
    }


# ─────────────────────────────────────────────────────────
# BƯỚC 5: THỐNG KÊ PHÂN BỐ LỚP
# ─────────────────────────────────────────────────────────

def validate_class_distribution(collected: dict, cfg: dict) -> dict:
    """Thống kê số lượng ảnh và sketch theo từng lớp."""
    print("\n" + "═" * 60)
    print("  BƯỚC 5: Thống kê phân bố lớp (Class Distribution)")
    print("═" * 60)

    p_counts = collected["class_photo_count"]
    s_counts = collected["class_sketch_count"]

    print(f"\n  Tổng số lớp tìm thấy: {len(collected['classes'])}")
    print(f"  Số lớp kỳ vọng:       {cfg['total_classes_expected']}")

    if len(collected["classes"]) != cfg["total_classes_expected"]:
        print(f"  [⚠ CẢNH BÁO] Số lớp thực tế KHÁC kỳ vọng!")

    # Thống kê min/max
    if p_counts:
        min_p = min(p_counts.values()); max_p = max(p_counts.values())
        avg_p = sum(p_counts.values()) / len(p_counts)
        print(f"\n  [PHOTO] Số ảnh mỗi lớp: min={min_p}, max={max_p}, avg={avg_p:.1f}")

        # Liệt kê các lớp có ít ảnh bất thường
        low_photo_classes = {k: v for k, v in p_counts.items() if v < avg_p * 0.5}
        if low_photo_classes:
            print(f"  [⚠] Lớp có quá ít ảnh (< 50% trung bình): {low_photo_classes}")

    if s_counts:
        min_s = min(s_counts.values()); max_s = max(s_counts.values())
        avg_s = sum(s_counts.values()) / len(s_counts)
        print(f"  [SKETCH] Số sketch mỗi lớp: min={min_s}, max={max_s}, avg={avg_s:.1f}")

    return {
        "class_photo_count": dict(p_counts),
        "class_sketch_count": dict(s_counts),
        "all_classes": sorted(list(collected["classes"])),
    }


# ─────────────────────────────────────────────────────────
# XUẤT BÁO CÁO & FILE METADATA
# ─────────────────────────────────────────────────────────

def export_results(
    dataset_name: str,
    valid_pairs: list,
    orphan_sketches: list,
    corrupt_files: list,
    class_stats: dict,
    output_dir: Path,
):
    """Xuất báo cáo và file metadata CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Lưu file metadata_pairs.json (dùng cho DataLoader) ---
    meta_path = output_dir / f"{dataset_name}_metadata_pairs.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(valid_pairs, f, ensure_ascii=False, indent=2)
    print(f"\n  ✔ Metadata pairs đã lưu: {meta_path}")
    print(f"    ({len(valid_pairs):,} cặp hợp lệ)")

    # --- 2. Lưu file báo cáo validation ---
    report = {
        "dataset": dataset_name,
        "summary": {
            "total_valid_pairs": len(valid_pairs),
            "total_orphan_sketches": len(orphan_sketches),
            "total_corrupt_files": len(corrupt_files),
            "total_classes": len(class_stats["all_classes"]),
            "train_pairs": sum(1 for p in valid_pairs if p["split"] == "train"),
            "test_pairs": sum(1 for p in valid_pairs if p["split"] == "test"),
        },
        "all_classes": class_stats["all_classes"],
        "class_photo_count": class_stats["class_photo_count"],
        "class_sketch_count": class_stats["class_sketch_count"],
        "orphan_sketches": orphan_sketches[:50],   # Chỉ lưu 50 ví dụ đầu
        "corrupt_files": corrupt_files,
    }

    report_path = output_dir / f"{dataset_name}_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  ✔ Báo cáo validation đã lưu: {report_path}")

    # --- 3. Lưu file danh sách class (dùng cho Zero-Shot split) ---
    classes_path = output_dir / f"{dataset_name}_classes.txt"
    with open(classes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(class_stats["all_classes"]))
    print(f"  ✔ Danh sách 125 class đã lưu: {classes_path}")

    return report


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Data Validation Script - SBIR Project"
    )
    parser.add_argument(
        "--dataset", type=str, default="sketchy",
        choices=list(DATASET_CONFIGS.keys()),
        help="Tên dataset cần kiểm định"
    )
    parser.add_argument(
        "--data_root", type=str, default="data",
        help="Đường dẫn tới thư mục gốc chứa dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/metadata",
        help="Thư mục lưu file báo cáo và metadata"
    )
    parser.add_argument(
        "--skip_integrity", action="store_true",
        help="Bỏ qua bước kiểm tra toàn vẹn file (nếu dataset rất lớn)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Số luồng song song khi kiểm tra integrity"
    )
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    cfg        = DATASET_CONFIGS[args.dataset]

    print(f"\n{'═' * 60}")
    print(f"  DATA VALIDATION - SBIR Project")
    print(f"{'═' * 60}")
    print(f"  Dataset:   {args.dataset.upper()}")
    print(f"  Data Root: {data_root.resolve()}")

    if not data_root.exists():
        print(f"\n[LỖI] Không tìm thấy thư mục data: {data_root.resolve()}")
        sys.exit(1)

    # Bước 1: Quét thư mục
    collected = collect_files(data_root, cfg)

    # Bước 2: Kiểm tra ghép cặp
    pair_result = validate_pairs(collected)

    # Bước 3: Kiểm tra integrity (tùy chọn)
    corrupt_files = []
    if not args.skip_integrity:
        integrity_result = validate_integrity(pair_result["valid_pairs"], args.workers)
        corrupt_files = integrity_result["corrupt_files"]
    else:
        print("\n  [BỎ QUA] Integrity check (--skip_integrity)")

    # Bước 4: Kiểm tra thông số kỹ thuật ảnh
    validate_image_specs(collected)

    # Bước 5: Thống kê phân bố lớp
    class_stats = validate_class_distribution(collected, cfg)

    # Xuất kết quả
    print("\n" + "═" * 60)
    print("  XUẤT KẾT QUẢ")
    print("═" * 60)
    report = export_results(
        dataset_name=args.dataset,
        valid_pairs=pair_result["valid_pairs"],
        orphan_sketches=pair_result["orphan_sketches"],
        corrupt_files=corrupt_files,
        class_stats=class_stats,
        output_dir=output_dir,
    )

    # Tóm tắt cuối
    print("\n" + "═" * 60)
    print("  KẾT QUẢ VALIDATION TỔNG QUAN")
    print("═" * 60)
    s = report["summary"]
    status = "✅ PASS" if s["total_corrupt_files"] == 0 and s["total_orphan_sketches"] == 0 \
             else "⚠ PASS (có cảnh báo)" if s["total_corrupt_files"] == 0 \
             else "❌ FAIL (có file lỗi)"

    print(f"\n  Trạng thái tổng thể:       {status}")
    print(f"  Tổng cặp (sketch, photo):  {s['total_valid_pairs']:,}")
    print(f"    └─ Tập train:            {s['train_pairs']:,}")
    print(f"    └─ Tập test:             {s['test_pairs']:,}")
    print(f"  Số lớp (classes):          {s['total_classes']}")
    print(f"  Sketch mồ côi:             {s['total_orphan_sketches']:,}")
    print(f"  File lỗi:                  {s['total_corrupt_files']:,}")
    print(f"\n  Báo cáo chi tiết tại: outputs/metadata/")
    print("═" * 60)


if __name__ == "__main__":
    main()
