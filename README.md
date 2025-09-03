# TestcaseGenerationModel
Dự án huấn luyện mô hình viT5-base với LoRA để sinh test case tiếng Việt từ yêu cầu tự nhiên (natural language) của người dùng.
## Dataset
Dataset tự tổng hợp để train mô hình sinh test case, lấy cảm hứng từ SoftwareTestingHelp, Katalon, TestSigma và được chỉnh sửa/bổ sung bởi tôi.

## Nội dung
- code/train.py: Code huấn luyện + merge + đóng gói (không chứa token).
- model: Thư mục chứa LoRA adapter phục vụ nộp đồ án/tái lập.
- Model trực tiếp trên HuggingFace: [cammtuxinhdep/vit5_base](https://huggingface.co/cammtuxinhdep/vit5_base)

## Training Details
- Model: VietAI/vit5-base (LoRA r=32, α=64, dropout=0.05 (target modules: q,k,v,o))
- Training Environment: Kaggle GPU (NVIDIA A100)
- Final Loss: 1.3540 (at step 1250)

## Mục đích phục vụ
Nghiên cứu, học tập.
