---
title: ZS-SBIR LoRA-CLIP API & Studio
emoji: 🎨
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Zero-Shot Sketch-Based Image Retrieval (ZS-SBIR) — Asymmetric LoRA-CLIP

Hệ thống Inference API & Web Demo cho bài toán **Truy hồi hình ảnh dựa trên bản vẽ phác thảo (Zero-Shot SBIR)** và **Nhận diện vật thể phác thảo (Zero-Shot Sketch Classification)**.

- **Đồ án môn học:** CS331 — Thị giác Máy tính Nâng cao (Trường ĐH Công nghệ Thông tin — ĐHQG-HCM)
- **Giảng viên hướng dẫn:** Thầy Mai Tiến Dũng
- **Sinh viên thực hiện:**
  - Đinh Phạm Anh Quân (MSSV: 24521425)
  - Phạm Tường Quân (MSSV: 24521451)

---

## 🚀 Tính năng Hệ thống
1. **Truy hồi hình ảnh (Sketch-to-Image Retrieval):** Tìm kiếm Top 10 bức ảnh tương đồng nhất dựa trên Cosine Similarity trong không gian 512-D L2-normalized.
2. **Nhận diện vật thể (Zero-Shot Sketch Classification):** Dự đoán danh mục đối tượng bản vẽ qua CLIP Text Prototypes (125 lớp Sketchy + tập từ vựng mở).
3. **Dung hợp đa phương thức (Multimodal Text Conditioning):** Kết hợp nét vẽ (70%) cùng gợi ý ngôn ngữ (30%) với từ điển cầu nối Việt - Anh.
4. **REST API phục vụ Frontend Vercel:** Cung cấp các endpoint `/api/health`, `/api/retrieve`, `/api/classify` tích hợp sẵn CORS.

---

## 📡 REST API Documentation

### 1. `GET /api/health`
Kiểm tra trạng thái hệ thống, cấu hình thiết bị và kích thước kho dữ liệu.

### 2. `POST /api/retrieve`
Truy hồi Top 10 ảnh khớp nhất từ bản vẽ base64.
```json
{
  "sketch": "data:image/png;base64,...",
  "gallery_scope": "unseen",
  "semantic_clue": "red helicopter flying",
  "top_k": 10
}
```

### 3. `POST /api/classify`
Dự đoán danh mục vật thể và xác suất Top 5 ứng viên.
```json
{
  "sketch": "data:image/png;base64,...",
  "semantic_clue": ""
}
```
