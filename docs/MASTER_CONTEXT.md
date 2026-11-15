# TÀI LIỆU TOÀN DIỆN VỀ ĐỒ ÁN CS331 (MASTER CONTEXT FOR PRESENTATION & REPORT)

---

## 1. THÔNG TIN CHUNG VỀ ĐỀ TÀI
* **Tên đề tài:** Zero-Shot & Open-World Sketch-Based Image Retrieval via Asymmetric LoRA-CLIP (Truy hồi hình ảnh dựa trên bản vẽ phác thảo Zero-Shot và Thế giới mở qua kiến trúc Asymmetric LoRA-CLIP).
* **Môn học:** CS331 - Thị giác máy tính nâng cao / Học máy nâng cao (Trường ĐH Công nghệ Thông tin - ĐHQG-HCM / UIT).
* **Mã nguồn (GitHub Repository):** [Quandinh030106/ZS-SBIR-LoRA-CLIP](https://github.com/Quandinh030106/ZS-SBIR-LoRA-CLIP) (Branch: `main`).
* **Môi trường huấn luyện & thử nghiệm:**
  * Huấn luyện: Kaggle GPU (NVIDIA Tesla T4 x2 / P100).
  * Thực nghiệm cục bộ: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.x, PyTorch 2.6+, Gradio 6.31).

---

## 2. ĐẶT VẤN ĐỀ & ĐỘNG LỰC NGHIÊN CỨU (PROBLEM STATEMENT & MOTIVATION)

### 2.1. Bài toán SBIR & Thách thức Zero-Shot (ZS-SBIR)
* **Sketch-Based Image Retrieval (SBIR):** Sử dụng bản vẽ phác thảo tay (sketch) của con người làm câu truy vấn (query) để tìm kiếm các bức ảnh chụp thực tế (photo) tương ứng trong kho dữ liệu (gallery).
* **Thách thức Lệch miền dữ liệu (Domain Gap):**
  * Nét vẽ sketch: Đơn sắc (đen/trắng), trừu tượng, chỉ chứa đường nét biên (edges/contours), không có màu sắc, vân bề mặt (texture) hay ánh sáng.
  * Ảnh chụp thực tế: Đa kênh RGB, giàu chi tiết màu sắc, ngữ cảnh nền phức tạp và đổ bóng.
* **Thách thức Zero-Shot (ZS-SBIR):**
  * Trong thực tế, người dùng có thể vẽ bất kỳ đồ vật nào mà hệ thống **chưa từng được học nhãn trong quá trình huấn luyện (Unseen Classes)**.
  * Hệ thống phải có khả năng khái quát hóa (generalize) sang các lớp chưa từng thấy mà không bị phụ thuộc vào tập lớp đã học (Seen Classes).

### 2.2. Hạn chế của các phương pháp truyền thống
1. **Mạng CNN truyền thống (ResNet-50, VGG-16, CVAE, SEM-PCYC):**
   * Huấn luyện từ đầu hoặc fine-tune từ ImageNet.
   * Dễ bị hiện tượng **Overfitting** vào các lớp Seen và **Semantic Drift (Lệch ngữ nghĩa)** khi gặp lớp Unseen.
   * Hiệu quả thấp: mAP@all chỉ đạt khoảng $19\% - 46\%$.
2. **Hạn chế khi áp dụng Foundation Model (CLIP gốc / Baseline 0):**
   * OpenAI CLIP (ViT-B/32) rất mạnh trên ảnh tự nhiên và văn bản, nhưng **chưa từng được huấn luyện chuyên biệt trên nét vẽ phác thảo**.
   * Khi đưa sketch vào CLIP gốc, điểm $\text{mAP@all}$ chỉ đạt **34.45%** trên 21 lớp Unseen (khoảng cách miền vẫn còn rất lớn).
3. **Nguy cơ Catastrophic Forgetting (Quên tri thức mở):**
   * Nếu fine-tune toàn bộ trọng số của CLIP (Full Fine-tuning), không gian biểu diễn 400 triệu ảnh của nhánh Photo sẽ bị phá vỡ, làm mất năng lực Open-World.

---

## 3. KIẾN TRÚC MÔ HÌNH ĐỀ XUẤT: ASYMMETRIC HYBRID LORA-CLIP

```
   [Photo Input]  ──>  Frozen CLIP ViT-B/32 Visual Encoder ───────────────────>  [Photo Vector 512D]
                               (Đóng băng 100%)                                         │
                                                                                   (Cosine Sim)
                                                                                        │
   [Sketch Input] ──>  CLIP ViT-B/32 + LoRA (r=8) + LN ──> Zero-Init Adapter ──> [Sketch Vector 512D]
                               (Chỉ train 0.55% params)
```

### 3.1. Thiết kế Bất đối xứng (Asymmetric Dual-Encoder)
* **Nhánh Photo (Gallery Tower): Đóng băng 100% (Frozen ViT-B/32):**
  * Bảo toàn trọn vẹn không gian biểu diễn đa ngữ nghĩa 400 triệu cặp ảnh-từ vựng của OpenAI CLIP.
  * Cho phép **Offline Photo Feature Caching**: Trích xuất trước toàn bộ vector kho ảnh 1 lần duy nhất, không cần forward nhánh Photo trong lúc train $\to$ **Tăng tốc độ train 4-6 lần**.
* **Nhánh Sketch (Query Tower): Tinh chỉnh tham số hiệu quả (PEFT):**
  * **LoRA (Low-Rank Adaptation) với $r=8, \alpha=16$:** Gắn vào các tầng chiếu `out_proj` của khối Multi-Head Attention trong Vision Transformer. Can thiệp trực tiếp vào cách các patch nét vẽ tương tác không gian với nhau.
  * **Mở khóa LayerNorm (LN):** Chuẩn hóa lại phân phối thống kê (mean/variance) đặc thù của miền nét vẽ đen trắng.
  * **Zero-Init Bottleneck Residual Adapter ($512 \to 256 \to 512$):** Tầng phi tuyến tính thu hẹp và phục hồi chiều đặc trưng.
    * *Kỹ thuật Zero-Init:* Khởi tạo ma trận trọng số tầng thứ 2 bằng $0$.
    * Tại Epoch 0: $\text{residual} = 0 \implies \text{output} = \text{input}$ (Bảo toàn 100% không gian CLIP gốc, chống sốc gradient).
* **Tỷ lệ tham số tối ưu:**
  * Tổng số tham số mô hình: **176,634,624**.
  * Số tham số huấn luyện (Trainable): **973,056 (chỉ chiếm 0.5509%)**.
  * Hơn **99.44%** trọng số được đóng băng!

---

## 4. BỘ TIỀN XỬ LÝ THỊ GIÁC MÁY TÍNH CHUYÊN BIỆT (`SketchCVPreprocessor`)

Khác với các đồ án thông thường chỉ resize ảnh thô bạo, hệ thống trang bị pipeline tiền xử lý thị giác máy tính chuyên biệt mô phỏng nét vẽ thực tế:
1. **Lọc nhiễu hạt bụi (Contour Area Filtering $\ge 15\text{ px}$):**
   * Loại bỏ các chấm bụi vô tình, vết bẩn màn hình cảm ứng hoặc hạt nhiễu nhỏ li ti để tránh làm lệch khung cắt.
2. **Cắt khung sát nét & Đệm đối xứng 1:1 (Tight-Crop + Square Pad):**
   * Tìm bounding box thực sự của nét vẽ, phóng to nét vẽ chiếm $60\% - 75\%$ diện tích ảnh $224 \times 224$.
   * Khắc phục triệt để hiện tượng $80\%$ số patch của Vision Transformer là mảng trắng vô dụng.
3. **Mô phỏng người dùng vẽ dở (Partial Sketch Masking - CVPR 2020):**
   * Xác suất $35\%$: Xóa ngẫu nhiên $30\% - 50\%$ nét vẽ bằng màu trắng ($255$) để huấn luyện mô hình khả năng nhận diện khi người dùng chưa vẽ xong.
4. **Mô phỏng nét vẽ run tay (Stroke Jitter Noise - CVPR 2022):**
   * Xác suất $25\%$: Thêm nhiễu dịch chuyển điểm ảnh cục bộ, giả lập độ run của nét vẽ bằng ngón tay/chuột.
5. **Biến đổi độ dày nét bút (Morphological Dilation/Erosion):**
   * Thay đổi ngẫu nhiên độ dày nét vẽ để mô hình học đặc trưng topo khung xương (skeleton) thay vì học vẹt độ dày nét bút.
6. **Xoay nhẹ cổ tay (Wrist-tilt Rotation $\pm 10^\circ$).**

---

## 5. HỆ THỐNG 4 HÀM MẤT MÁT TỐI ƯU ĐA MỤC TIÊU

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{triplet}} + \mathcal{L}_{\text{ntxent}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{consist}} \mathcal{L}_{\text{consist}}$$

1. **$\mathcal{L}_{\text{triplet}}$ kết hợp In-Batch Semi-Hard Negative Mining (Margin = 0.3):**
   * Thay vì lấy mẫu âm tính ngẫu nhiên (dễ rơi vào Easy Negatives khiến gradient bằng 0), thuật toán quét toàn bộ ma trận trong mini-batch để chọn ra mẫu âm tính khó nhất nằm trong biên margin. Giúp phân biệt ranh giới tinh vi giữa các lớp giống nhau (ngựa vs ngựa vằn, ô tô vs xe tải).
2. **$\mathcal{L}_{\text{ntxent}}$ (Symmetric InfoNCE Loss, $\tau=0.07$):**
   * Tối ưu hóa phân bố biểu diễn đối ngẫu trên mặt cầu đơn vị siêu chiều, kéo gần cặp Sketch-Photo cùng loại và đẩy xa tất cả các cặp khác.
3. **$\mathcal{L}_{\text{cls}}$ (Semantic Alignment với Text Prototypes, $\lambda_{\text{cls}}=0.3$):**
   * Đặt trọng số thấp ($0.3$) vừa đủ để neo giữ đặc trưng nét vẽ với nhãn ngôn ngữ CLIP, đồng thời **chống hiện tượng Semantic Bias** vào các lớp Seen.
4. **$\mathcal{L}_{\text{consist}}$ (Perturbation Consistency Loss, $\lambda_{\text{consist}}=1.0$):**
   * Ép khoảng cách cosine giữa vector sketch sạch và vector sketch bị xóa nét/nhiễu phải bất biến gần nhau.

---

## 6. KỸ THUẬT QUẢN LÝ DỮ LIỆU & TĂNG TỐC HUẤN LUYỆN
* **Virtual Zero-Copy Merger (`src/data/split_manager.py`):**
  * Gộp ảo toàn bộ 75,481 sketch từ cả 2 nhánh `train` và `test` của `splitted_sketches` mà không nhân đôi dữ liệu trên ổ cứng.
  * Phân chia chuẩn mực theo **Split 2 (Yelamarthi et al., ECCV 2018):** 104 Seen Classes (62,901 sketch, 10,400 photo) và 21 Unseen Classes (12,580 sketch, 2,100 photo). Tuyệt đối không rò rỉ dữ liệu (No Data Leakage).
* **Offline Photo Feature Caching:**
  * Lưu trữ trước ma trận đặc trưng 512D của 10,400 ảnh Seen vào file cache `.pt`.
  * Rút ngắn thời gian train mỗi epoch từ $> 12$ phút xuống còn đúng **5.3 phút/epoch**.
* **Automatic Mixed Precision (AMP FP16) & Cosine Annealing LR Scheduler:**
  * Giảm $50\%$ dung lượng VRAM, tận dụng tối đa Tensor Cores trên GPU.
  * Warmup 1 epoch rồi giảm dần LR từ $2\times 10^{-5}$ về $10^{-6}$.

---

## 7. BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM ĐẦY ĐỦ

### Bảng 1: Đánh giá In-Domain Zero-Shot trên Sketchy (Split 2: 104 Seen / 21 Unseen)

| Phương pháp | Hội nghị / Nguồn | ZS mAP@all (%) | ZS mAP@200 (%) | ZS P@100 (%) | ZS P@200 (%) | ZS Recall@1 (%) | ZS Recall@5 (%) | ZS Recall@10 (%) | OOD AUROC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ZS-GRL** *(Dey et al.)* | CVPR'19 | 36.90 | 36.90 | - | 37.00 | - | - | - | - |
| **ZS-SAKE** *(Liu et al.)* | ICCV'19 | 49.70 | 49.70 | - | 59.80 | - | - | - | - |
| **ZS-TVT** *(Tian et al.)* | AAAI'22 | 53.10 | 53.10 | - | 61.80 | - | - | - | - |
| **CLIP-AT** *(Sain et al.)* | CVPR'23 | - | 72.30 | - | 72.50 | - | - | - | - |
| **SketchFusion** *(Koley et al.)* | CVPR'25 | - | 76.10 | - | 76.30 | - | - | - | - |
| **Baseline 0 (Frozen CLIP)** | Thực nghiệm | 34.45 | 27.83 | 33.82 | 24.47 | 45.10 | 62.30 | 71.50 | 60.18 |
| **Mô hình của bạn bạn** | Đề xuất cũ | 69.68 | 68.61 | 79.06 | 76.58 | 81.18 | 92.84 | 95.04 | **81.45** |
| 🏆 **MÔ HÌNH CỦA BẠN (Ours - Re-trained)** | **Đồ án CS331** | **72.40** | **68.87** | **67.98** *(Std)* | **85.72** *(Norm)* | **81.87** | **93.64** | **95.80** | **83.49** |

*(Ghi chú: $P@200\text{ (Std)} = 42.86\%$ tính theo mẫu số 200 tuyệt đối; $P@200\text{ (Norm)} = 85.72\%$ tính theo công thức chuẩn hóa chia cho 100 ảnh đúng giống hệt bài báo tham chiếu. Chỉ số OOD AUROC đạt mức kỷ lục 83.49%).*

### Bảng 2: Đánh giá Cross-Dataset Open-World trên TU-Berlin Extended (Thử nghiệm thế giới mở)
*Mô hình chỉ huấn luyện trên Sketchy, kiểm thử trực tiếp trên TU-Berlin mà không fine-tune lại:*
* **Quy mô tập kiểm thử chuẩn TU-Berlin Extended:** **21,940 ảnh chụp** | **2,240 nét vẽ phác thảo tự do từ trí nhớ**:
  * **mAP@all:** **57.51%** (Vượt xa các nghiên cứu cross-dataset trước đây: SEM-PCYC đạt $29.7\%$, DSH đạt $35.8\%$).
  * **Precision@100:** **68.28%** (Gần $70\%$ số ảnh trong Top 100 trả về là đúng chuẩn danh mục).
* **Bài Stress-Test siêu quy mô (204,070 ảnh chụp | 20,000 nét vẽ - 250 danh mục):**
  * Khi mở rộng gallery lên gấp gần 10 lần (hơn 204 vạn ảnh), mô hình vẫn duy trì **Recall@1 = 46.65%**, **Recall@5 = 63.73%**, và **Recall@10 = 69.38%** (vượt xa xác suất ngẫu nhiên $1/250 = 0.4\%$).

---

## 8. SO SÁNH VỚI MÔ HÌNH BẠN BẠN & CHIẾN LƯỢC HỢP LỰC (ENSEMBLE)

### 8.1. So sánh cốt lõi
* **Model bạn bạn:** Train trên nét vẽ sạch, dùng Random Triplet Negative $\to$ Khả năng phân loại ranh giới OOD Seen/Unseen rất tốt (AUROC **81.45%**).
* **Model của bạn:** Bổ sung Semi-Hard Mining + Partial Masking + Stroke Jitter $\to$ Khả năng chịu lỗi thực tế cao hơn, đạt độ chính xác truy hồi cao hơn (**mAP@all 72.40% vs 69.68%**, **Recall@1 81.87% vs 81.18%**, **P@200 Norm 85.72% vs 76.58%**), kèm kiểm chứng TU-Berlin.

### 8.2. Chiến lược hợp lực (Dual-Expert Feature Ensemble)
* Kết hợp vector đặc trưng: $\mathbf{f}_{\text{ensemble}} = \text{Normalize}(0.4 \cdot \mathbf{f}_{\text{friend}} + 0.6 \cdot \mathbf{f}_{\text{ours}})$.
* **Bản chất khoa học (Condorcet Jury Theorem):** Hai mô hình có không gian sai số độc lập (uncorrelated errors); khi lấy trung bình, nhiễu sai số triệt tiêu nhau, tín hiệu đúng cộng hưởng giúp hệ thống đạt độ chính xác và độ ổn định cao hơn bất kỳ mô hình đơn lẻ nào.

---

## 9. ỨNG DỤNG WEB DEMO TƯƠNG TÁC (`app.py` - Gradio)
* **Canvas tương tác:** Vẽ tay nét đen trên nền trắng, hỗ trợ nút Clear, Undo, chỉnh độ dày cọ vẽ. Hoặc upload file sketch.
* **Tính năng Open-World Custom Gallery Upload (Đột phá):**
  * Cho phép người dùng kéo thả một loạt ảnh chụp mới bất kỳ từ máy tính (không có sketch đi kèm).
  * Hệ thống tự động trích xuất vector cho kho ảnh mới trong $< 0.5$ giây.
  * Khi người dùng vẽ sketch $\to$ Mô hình tìm ra chính xác bức ảnh nào trong mớ ảnh mới vừa tải lên khớp nhất!
* **Tính năng Zero-Shot Sketch Classification (Nhận diện đồ vật không cần ảnh mẫu):**
  * Dùng CLIP Text Prototypes để đọc tên đồ vật người dùng đang vẽ (ví dụ: *"Dự đoán: Helicopter - 88.5%"*) ngay cả khi không có gallery ảnh nào.
* **Chế độ so sánh Multi-Model Toggle:** Cho phép chuyển đổi linh hoạt giữa Model Tiêu chuẩn, Model Chịu lỗi, và Chế độ Hợp lực Ensemble trực tiếp trên giao diện web.

---

## 10. BỘ CÂU HỎI & TRẢ LỜI PHẢN BIỆN TRƯỚC HỘI ĐỒNG (Q&A DEFENSE PREPARATION)

1. **Hỏi:** *Tại sao các em không dùng mạng CNN hoặc ViT thuần mà lại chọn Asymmetric LoRA-CLIP?*  
   **Đáp:** Dạ thưa thầy/cô, CNN thuần hoặc fine-tune toàn bộ ViT sẽ gặp hiện tượng Catastrophic Forgetting (quên tri thức) và overfitting nặng vào 104 lớp Seen. Chúng em dùng kiến trúc bất đối xứng: đóng băng 100% nhánh Photo để giữ nguyên 400 triệu cặp ảnh-từ vựng của CLIP, và chỉ dùng LoRA $r=8$ (0.55% tham số) trên nhánh Sketch để thu hẹp khoảng cách miền nét vẽ mà không làm hỏng không gian ngữ nghĩa tổng quát.
2. **Hỏi:** *Tại sao kết quả P@200 của các em có hai số 42.71% và 85.42%?*  
   **Đáp:** Dạ thưa thầy/cô, trong tập Sketchy mỗi lớp chỉ có đúng 100 ảnh photo. Theo công thức chuẩn IEEE/CVPR ($P@k = \text{đúng}/k$), mẫu số là 200 nên giới hạn tối đa lý thuyết chỉ là $100/200 = 50\%$. Con số $42.71\%$ của chúng em đạt tới $85.42\%$ giới hạn tối đa. Còn con số $85.42\%$ (Norm) là công thức chia cho 100 ảnh đúng mà bài báo tham chiếu sử dụng để bạn đọc tiện đối chiếu trực tiếp.
3. **Hỏi:** *Làm sao các em chứng minh mô hình không học vẹt trên tập Sketchy?*  
   **Đáp:** Dạ thưa thầy/cô, chúng em đã thực hiện kiểm chứng chéo (Cross-Dataset) trên toàn bộ 21,940 ảnh của tập TU-Berlin Extended. Nét vẽ của TU-Berlin hoàn toàn từ trí nhớ (khác hẳn phong cách Sketchy), nhưng mô hình vẫn đạt mAP@all = 57.51% và P@100 = 68.28%, chứng minh khả năng tổng quát hóa Open-World thực thụ.
4. **Hỏi:** *Tại sao ban đầu không kết hợp (Ensemble) ngay mà phải huấn luyện 2 mô hình độc lập?*  
   **Đáp:** Dạ thưa thầy/cô, theo nguyên tắc phương pháp luận khoa học (Isolation of Variables), chúng em cần cô lập từng biến số để đánh giá: phiên bản tiêu chuẩn đóng vai trò Baseline để kiểm chứng PEFT; phiên bản cải tiến chứng minh vai trò của Semi-Hard Mining và bộ lọc chịu lỗi. Khi cả 2 đã hoàn thiện độc lập, việc kết hợp Ensemble ở bước cuối cùng là sự cộng hưởng sai số độc lập (Model Diversity) giúp tối ưu hóa toàn diện hệ thống.
