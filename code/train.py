# =========================
# Training Script for DistilGPT2 with LoRA
# This script fine-tunes DistilGPT2 using LoRA for text generation.
# Configuration optimized for performance with specific hyperparameters.
# =========================

import os, time, random, numpy as np, warnings, torch

# Disable unnecessary logging and warnings
os.environ["WANDB_DISABLED"] = "true"   # Disable wandb logging completely
os.environ["WANDB_SILENT"] = "true"     # Silence wandb output

# Set environment for quiet operation and GPU optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass

# Uninstall TensorFlow/keras to avoid CUDA logging (optional, no internet required)
try:
    os.system("python -m pip -q uninstall -y tensorflow tensorflow-cpu tensorflow-gpu keras keras-preprocessing keras-nightly > /dev/null 2>&1")
except Exception:
    pass

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
from huggingface_hub import notebook_login
import importlib.metadata as im

# Set random seed for reproducibility
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# Login to Hugging Face Hub (push will be skipped if not logged in)
notebook_login()

# ===== 1) Data Preparation: Combine main dataset with variants =====
# Load and merge main dataset with variant prompts, creating multiple instances
dataset_main = load_dataset("your_dataset_repo/main_dataset", split="train")  # Replace with your dataset repository
dataset_variants = load_dataset("your_dataset_repo/variant_dataset", split="train")  # Replace with your dataset repository
df_main = dataset_main.to_pandas()
df_variants = dataset_variants.to_pandas()

rows = []
for _, r in df_main.iterrows():
    p = str(r['prompt']).strip()
    c = str(r['completion']).strip()
    vrows = df_variants[df_variants['prompt'] == r['prompt']]
    if not vrows.empty:
        v = vrows.iloc[0]
        v1 = (str(v['variant_1']).strip() if pd.notna(v['variant_1']) else "")
        v2 = (str(v['variant_2']).strip() if pd.notna(v['variant_2']) else "")
    else:
        v1 = v2 = ""
    rows.append({"prompt": p, "completion": c})
    rows.append({"prompt": (v1 if v1 else p), "completion": c})
    rows.append({"prompt": (v2 if v2 else p), "completion": c})

merged_df = pd.DataFrame(rows)
dataset = Dataset.from_pandas(merged_df, preserve_index=False)

# Build text with Instruction/Response format
SEP_INST = "### Instruction:"
SEP_RESP = "### Response:"
def build_text(ex):
    return {"text": f"{SEP_INST}\n{ex['prompt'].strip()}\n{SEP_RESP}\n{ex['completion'].strip()}"}
dataset = dataset.map(build_text)

# ===== 2) Tokenizer and Tokenization =====
# Initialize tokenizer and set padding token
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
pad_id = tokenizer.eos_token_id

max_length = 768  # Maximum sequence length for tokenization
def tok_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
tokenized = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names)

# Use standard data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

# ===== 3) Model Configuration =====
# Check for bitsandbytes availability for QLoRA, fallback to LoRA FP16
def has_bnb(min_version="0.39.0"):
    try:
        v = im.version("bitsandbytes"); from packaging import version as V
        return V.parse(v) >= V.parse(min_version)
    except Exception:
        return False
BNB_AVAILABLE = has_bnb("0.39.0")

if BNB_AVAILABLE and BitsAndBytesConfig is not None:
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_cfg, device_map="auto")
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.half().to("cuda")

# Clean up model configuration warnings
try:
    model.config.__dict__.pop("loss_type", None)
except Exception:
    pass
model.config.pad_token_id = pad_id
model.config.use_cache = False

# Apply LoRA configuration
lora_cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["c_attn", "c_fc", "c_proj"],
    lora_dropout=0.1, bias="none",
    fan_in_fan_out=True
)
model = get_peft_model(model, lora_cfg)

# Enable memory-saving techniques
model.gradient_checkpointing_enable()
try:
    model.enable_input_require_grads()
except Exception:
    pass

# ===== 4) Training Configuration =====
# Training arguments optimized for performance and stability
training_args = TrainingArguments(
    output_dir="/kaggle/working/distilgpt2_finetuned",
    run_name="distilgpt2-lora-ep4-bs16-lr5e5-cosine",
    num_train_epochs=4,
    per_device_train_batch_size=8,  # Reduced to avoid memory issues
    gradient_accumulation_steps=2,  # Effective batch size ≈ 16
    save_steps=200,
    logging_steps=100,
    learning_rate=5e-5,
    fp16=True,
    max_grad_norm=1.0,
    warmup_steps=200,
    lr_scheduler_type="cosine",
    weight_decay=0.1,  # Added to prevent overfitting
    label_names=["labels"],
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

print("Starting training...")
start = time.time()
trainer.train()
print(f"Training time: {(time.time() - start)/60:.2f} minutes")

# ===== 5) Save and Push to Hugging Face Hub =====
# Save model and tokenizer locally
save_dir = "/kaggle/working/distilgpt2_finetuned"
print(f"Saving model to: {save_dir}")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("Model and tokenizer saved successfully!")

# Compress model for download
print("Zipping model...")
#!zip -r /kaggle/working/distilgpt2_finetuned.zip /kaggle/working/distilgpt2_finetuned
print("Model zipped successfully!")

from huggingface_hub import HfApi, HfFolder, whoami
try:
    token = HfFolder.get_token() or os.getenv("HF_TOKEN")
    if token:
        me = whoami(token=token)
        # Fixed repository ID for Hugging Face
        repo_id = "cammtuxinhdep/distilgpt2_generate_testcase"
        if "/" in repo_id:
            HfApi().create_repo(repo_id=repo_id, private=False, exist_ok=True, token=token)
            model.push_to_hub(repo_id, token=token)
            tokenizer.push_to_hub(repo_id, token=token)
            print("Model and tokenizer uploaded successfully!")
        else:
            print("Skip push: invalid repo ID format.")
    else:
        print("Skip push: no HF token found (not logged in).")
except Exception as e:
    print(f"Skip push: unable to push (check token/permissions). {e}")