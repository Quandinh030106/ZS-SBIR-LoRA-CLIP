#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload complete ZS-SBIR application, weights, and all 12,500 photos to Hugging Face Space.
"""

import sys
import os
from pathlib import Path
from huggingface_hub import HfApi, login

def main():
    repo_id = "quan030106/zs-sbir-api"
    print("=" * 70)
    print(f"  HUGGING FACE SPACE UPLOADER: {repo_id}")
    print("=" * 70)

    token = os.environ.get("HF_TOKEN")
    if not token and len(sys.argv) > 1:
        token = sys.argv[1].replace("--token=", "").strip()

    if not token:
        print("\nĐể upload tự động toàn bộ thư mục và 12,500 ảnh lên Space:")
        print("1. Vào https://huggingface.co/settings/tokens tạo một Token với quyền 'Write'.")
        token = input("2. Dán token Hugging Face của bạn vào đây: ").strip()

    if not token:
        print("[ERROR] Token không được để trống!")
        return

    api = HfApi(token=token)
    proj_root = Path(__file__).resolve().parent.parent

    # 1. Upload hf_space folder (app.py, requirements.txt, README.md, src/, configs/)
    hf_space_dir = proj_root / "hf_space"
    print(f"\n[1/4] Đang tải mã nguồn và cấu hình từ {hf_space_dir}...")
    api.upload_folder(
        folder_path=str(hf_space_dir),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["sketchy_photos.zip"], # upload separately with progress
    )
    print("  -> Tải mã nguồn thành công!")

    # 2. Upload model weights (best_seco_lora_clip.pt)
    ckpt_path = proj_root / "outputs" / "checkpoints" / "best_seco_lora_clip.pt"
    if ckpt_path.exists():
        print(f"\n[2/4] Đang tải trọng số mô hình tốt nhất ({ckpt_path.stat().st_size / (1024*1024):.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(ckpt_path),
            path_in_repo="best_seco_lora_clip.pt",
            repo_id=repo_id,
            repo_type="space",
        )
        print("  -> Tải trọng số mô hình thành công!")
    else:
        print(f"[WARNING] Không tìm thấy file trọng số tại {ckpt_path}")

    # 3. Upload gallery vector index (gallery_index.pt)
    cache_path = proj_root / "outputs" / "cache" / "gallery_index.pt"
    if cache_path.exists():
        print(f"\n[3/4] Đang tải kho vector đặc trưng 12,500 ảnh ({cache_path.stat().st_size / (1024*1024):.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(cache_path),
            path_in_repo="gallery_index.pt",
            repo_id=repo_id,
            repo_type="space",
        )
        print("  -> Tải kho vector thành công!")
    else:
        print(f"[WARNING] Không tìm thấy file cache tại {cache_path}")

    # 4. Upload all 12,500 photos (sketchy_photos.zip)
    zip_path = proj_root / "hf_space" / "sketchy_photos.zip"
    if zip_path.exists():
        print(f"\n[4/4] Đang tải toàn bộ kho 12,500 ảnh ({zip_path.stat().st_size / (1024*1024):.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(zip_path),
            path_in_repo="sketchy_photos.zip",
            repo_id=repo_id,
            repo_type="space",
        )
        print("  -> Tải toàn bộ kho ảnh thành công!")
    else:
        print(f"[WARNING] Không tìm thấy file nén tại {zip_path}")

    print("\n" + "=" * 70)
    print(f"  HOÀN TẤT ĐẨY MÃ NGUỒN, TRỌNG SỐ VÀ 12,500 ẢNH LÊN HUGGING FACE SPACE!")
    print(f"  Space URL: https://huggingface.co/spaces/{repo_id}")
    print("=" * 70)

if __name__ == "__main__":
    main()
