# Import necessary libraries
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model and tokenizer (adjust path to your local model directory)
model_dir = "D:/TestcaseGenerationModel/model/kaggle/working/distilgpt2_finetuned"
base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = PeftModel.from_pretrained(base_model, model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Sample prompt dataset for FAISS (replace with your actual dataset)
sample_prompts = [
    "Viết một hàm tính tổng hai số.",
    "Tạo một test case cho đăng nhập thành công.",
    "Kiểm tra lỗi nhập liệu trống."
]
embeddings = embedder.encode(sample_prompts, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))


# Function to generate test cases
def generate_test_cases(user_input, max_cases=10):
    if not user_input.strip():
        return "Vui lòng nhập một prompt hợp lệ!", [], []

    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=768, padding=True).to(device)

    # Generate test cases
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_return_sequences=min(max_cases, 10),
            do_sample=True,
            temperature=0.7,
            top_k=50
        )

    test_cases = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return "Đã tạo thành công!", test_cases, []


# Function to check similarity and suggest
def check_similarity(input_text):
    if not input_text.strip():
        return "Không có đầu vào để kiểm tra.", []

    input_embedding = embedder.encode([input_text], convert_to_numpy=True)
    distances, indices = index.search(input_embedding, k=1)

    if distances[0][0] > 1.0:  # Threshold for unmatched (adjust as needed)
        with open("unmatched_inputs.txt", "a", encoding="utf-8") as f:
            f.write(f"{input_text}\n")
        return "Không khớp với bất kỳ prompt nào. Đề xuất: Kiểm tra lại cú pháp hoặc thử:\n- {sample_prompts[indices[0][0]]}", []
    return "Đầu vào hợp lệ!", []


# Function to export test cases to file
def export_test_cases(test_cases, file_format="txt"):
    if not test_cases:
        return "Không có trường hợp kiểm thử để xuất."

    filename = f"test_cases.{file_format}"
    with open(filename, "w", encoding="utf-8") as f:
        if file_format == "txt":
            f.write("\n".join(test_cases))
        elif file_format == "csv":
            f.write("Test Case\n")
            for case in test_cases:
                f.write(f"{case}\n")
    return f"Đã xuất thành công vào {filename}!"


# Gradio interface
with gr.Blocks(title="Tạo Trường Hợp Kiểm Thử") as demo:
    gr.Markdown("# Giao Diện Tạo Trường Hợp Kiểm Thử")

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Nhập Prompt", placeholder="Ví dụ: Tạo test case cho đăng nhập...")
            generate_btn = gr.Button("Tạo Trường Hợp Kiểm Thử")
            check_btn = gr.Button("Kiểm Tra Tương Thích")

        with gr.Column():
            output_message = gr.Textbox(label="Thông Báo", interactive=False)
            test_cases_output = gr.Textbox(label="Trường Hợp Kiểm Thử", lines=10, interactive=False)
            suggestions_output = gr.Textbox(label="Gợi Ý", interactive=False)

    with gr.Row():
        export_btn = gr.Button("Xuất File")
        format_dropdown = gr.Dropdown(["txt", "csv"], value="txt", label="Định Dạng File")
        export_message = gr.Textbox(label="Thông Báo Xuất File", interactive=False)

    # Event handlers
    generate_btn.click(
        fn=generate_test_cases,
        inputs=[input_text],
        outputs=[output_message, test_cases_output, suggestions_output]
    )

    check_btn.click(
        fn=check_similarity,
        inputs=[input_text],
        outputs=[output_message, suggestions_output]
    )

    export_btn.click(
        fn=export_test_cases,
        inputs=[test_cases_output, format_dropdown],
        outputs=[export_message]
    )

# Launch the interface
demo.launch()