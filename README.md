# TestcaseGenerationModel
Dự án huấn luyện mô hình DistilGPT2 với LoRA để sinh test case từ ngữ cảnh của người dùng nhập.

## Dataset
Dataset tự tổng hợp để train mô hình sinh test case, lấy cảm hứng từ SoftwareTestingHelp, Katalon, TestSigma và được chỉnh sửa/bổ sung bởi tôi.

## Nội dung
- code/train.py: Code huấn luyện.
- model/Kaggle: Fine-tuned model zipped từ Kaggle output.

## Training Details
- Model: DistilGPT2 with LoRA (r=16, alpha=32)
- Training Environment: Run on Kaggle with a GPU 100 (NVIDIA A100, ~40GB/80GB VRAM)
- Final Loss: 1.4739 (at step 900)

## Mục đích phục vụ
Nghiên cứu, học tập.
