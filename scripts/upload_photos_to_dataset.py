#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tải file nén 12,500 ảnh gốc (sketchy_photos.zip) lên Hugging Face Dataset quan030106/sketchy-photos
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

def main():
    repo_id = "quan030106/sketchy-photos"
    print("=" * 70)
    print(f"  TẢI 12,500 ẢNH GỐC LÊN HUGGING FACE DATASET: {repo_id}")
    print("=" * 70)

    token = os.environ.get("HF_TOKEN")
    if not token and len(sys.argv) > 1:
        token = sys.argv[1].replace("--token=", "").strip()

    if not token:
        token = input("Nhập token Hugging Face của bạn (quyền Write): ").strip()

    if not token:
        print("[ERROR] Token không được để trống!")
        return

    proj_root = Path(__file__).resolve().parent.parent
    zip_path = proj_root / "hf_space" / "sketchy_photos.zip"

    if not zip_path.exists():
        print(f"[ERROR] Không tìm thấy file {zip_path}")
        return

    print(f"\nĐang tải {zip_path.name} ({zip_path.stat().st_size / (1024*1024):.1f} MB) lên Dataset {repo_id}...")
    api = HfApi(token=token)

    api.upload_file(
        path_or_fileobj=str(zip_path),
        path_in_repo="sketchy_photos.zip",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print("\n" + "=" * 70)
    print(f"  TẢI THÀNH CÔNG TOÀN BỘ 12,500 ẢNH GỐC LÊN DATASET!")
    print(f"  Dataset URL: https://huggingface.co/datasets/{repo_id}")
    print("=" * 70)

if __name__ == "__main__":
    main()
