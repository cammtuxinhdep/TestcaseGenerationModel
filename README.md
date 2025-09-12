# TestcaseGenerationModel

Dự án huấn luyện mô hình **viT5-base** với **LoRA** để sinh test case tiếng Việt từ yêu cầu ngôn ngữ tự nhiên (natural language) của người dùng.  

## Dataset
- Dataset được tự tổng hợp để train mô hình sinh test case.  
- Lấy cảm hứng và tham khảo từ **SoftwareTestingHelp, Katalon, TestSigma**.  
- Được chỉnh sửa/bổ sung thủ công để phù hợp ngữ cảnh tiếng Việt.  

## Nội dung
- **code/train.py**: Code huấn luyện + merge + đóng gói (không chứa token).  
- **model/**: Thư mục chứa LoRA adapter phục vụ nộp đồ án/tái lập.  
- Model trực tiếp trên HuggingFace: [cammtuxinhdep/vit5_base](https://huggingface.co/cammtuxinhdep/vit5_base)  

## Training Details
- **Base model**: `VietAI/vit5-base`  
- **LoRA config**: r=32, α=64, dropout=0.05 (target modules: q,k,v,o)  
- **Environment**: Kaggle GPU (NVIDIA A100)  
- **Epochs**: 12  
- **Final Loss**: 1.2364 (at step 400)  

## Pipeline tổng quát
1. **Input**: Người dùng nhập yêu cầu bằng ngôn ngữ tự nhiên (ví dụ: "Sinh test case cho chức năng đăng nhập web thương mại điện tử").  
2. **NLU & Intent parsing**: Chuẩn hóa câu hỏi (không dấu, mapping platform/domain/feature/aspect).  
3. **Retriever**:  
   - Sử dụng **SentenceTransformer (SBERT)** + **FAISS (cosine similarity)** để tìm test case tương tự từ corpus.  
   - Nếu FAISS không khả dụng → fallback sang TF-IDF.  
4. **Generator (T5-LoRA)**: Nếu thiếu dữ liệu, mô hình viT5-base fine-tune sinh mới test case theo format 4 mục:  
   - Mô tả (description)  
   - Inputs (dữ liệu kiểm thử)  
   - Các bước chính (steps)  
   - Đầu ra mong muốn (expected output)  
5. **Post-process**: Chuẩn hóa, suy luận thêm feature/aspect nếu cần.  
6. **UI**: Giao diện Gradio, bảng test case có thể export ra Excel.  

## Mục đích
- Nghiên cứu, học tập.  
- Minh họa khả năng ứng dụng **LLM + Retrieval** để hỗ trợ **sinh bộ test case tự động**.  
