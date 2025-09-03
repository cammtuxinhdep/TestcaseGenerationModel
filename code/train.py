#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train viT5-base with LoRA to generate Vietnamese test cases.
- Trains on JSONL main dataset + CSV variants (không loại trùng)
- Normalizes target to exactly 3 headings
- Saves LoRA adapter + merged full model
- Zips artifacts for submission/report
- (Optional) Uploads merged model & ZIP artifacts to Hugging Face

© 2025 - for academic/reporting use.

USAGE (examples):
  # Basic (local/Kaggle paths)
  python train_vit5_testcase_vit5_lora.py \
    --main_jsonl /kaggle/input/generate-testcase-dataset-jsonl/dataset2.jsonl \
    --variant_csv /kaggle/input/testcase-prompt-variants/prompt_variants.csv

  # Upload merged model & ZIP artifacts to HF (requires env HF_TOKEN)
  HF_TOKEN=hf_xxxxx python train_vit5_testcase_vit5_lora.py \
    --main_jsonl /kaggle/input/generate-testcase-dataset-jsonl/dataset2.jsonl \
    --variant_csv /kaggle/input/testcase-prompt-variants/prompt_variants.csv \
    --upload_hf --repo_id cammtuxinhdep/vit5_base

Notes on sensitive info:
- Do NOT hardcode tokens. Use environment variable HF_TOKEN.
- This script never prints token; it only reads from env.
"""

from __future__ import annotations
import os
import re
import ast
import sys
import time
import json
import math
import shutil
import random
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------
# Reproducibility & CUDA tweaks
# ----------------------------
SEED_DEFAULT = 42
def set_seed(seed: int = SEED_DEFAULT):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

# ----------------------------
# Data loading utilities
# ----------------------------
def robust_read_jsonl(path: str) -> pd.DataFrame:
    """
    Robust JSONL reader:
    - Accepts slightly malformed lines by trimming to {...}
    - Uses ast.literal_eval to avoid code execution
    - Requires columns: 'prompt', 'completion'
    """
    rows, good, bad = [], 0, 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            if not s.startswith("{"):
                i, j = s.find("{"), s.rfind("}")
                if i != -1 and j != -1 and j > i:
                    s = s[i : j + 1]
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, dict):
                    rows.append(obj)
                    good += 1
                else:
                    bad += 1
            except Exception:
                bad += 1
    df = pd.DataFrame(rows)
    if not {"prompt", "completion"}.issubset(df.columns):
        raise ValueError(f"JSONL missing required columns. Found: {list(df.columns)}")
    print(f"[Main] Parsed OK: {good}, skipped: {bad}")
    return df.dropna(subset=["prompt", "completion"]).reset_index(drop=True)

def read_variants_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Reads prompt variant CSV with schema:
      - required: 'prompt'
      - optional: columns starting with 'variant' (variant_1, variant_2, ...)
    Returns DataFrame of (canonical_prompt, variant) or None if no variants.
    """
    if not path or not os.path.exists(path):
        print("[Variants] CSV not provided or not found. Using only canonical prompts.")
        return None
    df = pd.read_csv(path)
    if "prompt" not in df.columns:
        print("[Variants] Missing 'prompt' column. Ignoring variants.")
        return None
    var_cols = [c for c in df.columns if c.lower().startswith("variant")]
    if not var_cols:
        print("[Variants] No variant_* columns. Ignoring variants.")
        return None

    rows = []
    for _, r in df.iterrows():
        p = str(r["prompt"]).strip()
        for c in var_cols:
            v = str(r[c]).strip() if pd.notna(r[c]) else ""
            if v:
                rows.append({"canonical_prompt": p, "variant": v})
    out = pd.DataFrame(rows)
    if out.empty:
        print("[Variants] Empty variants after parsing.")
        return None
    return out

# ----------------------------
# Target normalization (3 headings)
# ----------------------------
HEAD_DESC  = "Mô tả (description):"
HEAD_STEPS = "Các bước chính (main steps):"
HEAD_EXP   = "Đầu ra mong muốn (expected output):"

def ensure_three_headings(text: str) -> str:
    s = (text or "").strip()
    if HEAD_DESC not in s:
        s = HEAD_DESC + "\n" + s
    if HEAD_STEPS not in s:
        s += "\n" + HEAD_STEPS
    if HEAD_EXP not in s:
        s += "\n" + HEAD_EXP
    return s

def normalize_target(t: str) -> str:
    s = ensure_three_headings(t)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ----------------------------
# Model training
# ----------------------------
@dataclass
class TrainConfig:
    model_name: str = "VietAI/vit5-base"
    out_adapter_dir: str = "./vit5_testcase_lora"
    out_merged_dir: str = "./vit5_base_merged"
    epochs: int = 12
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    per_device_train_batch_size: int = 8
    grad_accum: int = 4
    src_max: int = 384
    tgt_max: int = 256
    seed: int = SEED_DEFAULT
    fp16: Optional[bool] = None  # auto by torch.cuda.is_available() if None

def train_and_merge(df_all: pd.DataFrame, cfg: TrainConfig) -> Tuple[str, str, Dict]:
    """
    Trains viT5-base with LoRA on df_all[["text","target","source"]] and merges LoRA into base.
    Returns (adapter_dir, merged_dir, final_eval_metrics).
    """
    set_seed(cfg.seed)
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq, Trainer, TrainingArguments
    )
    from peft import LoraConfig, get_peft_model

    # Build HF dataset
    assert {"text", "target", "source"}.issubset(df_all.columns)
    hf_all = Dataset.from_pandas(df_all[["text", "target", "source"]], preserve_index=False)
    splits = hf_all.train_test_split(test_size=0.1, seed=cfg.seed)
    print("Train/Eval:", len(splits["train"]), len(splits["test"]))

    # Tokenizer & base model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    if cfg.fp16 is None:
        cfg.fp16 = torch.cuda.is_available()
    if torch.cuda.is_available():
        base = base.half().to("cuda")
    base.config.use_cache = False
    if getattr(base.config, "decoder_start_token_id", None) is None:
        base.config.decoder_start_token_id = tokenizer.pad_token_id

    # LoRA
    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q", "k", "v", "o"],
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(base, lora)
    model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    # Tokenization (no padding at map stage)
    def tok_fn(batch):
        mi = tokenizer(batch["text"], max_length=cfg.src_max, truncation=True, padding=False)
        with tokenizer.as_target_tokenizer():
            lb = tokenizer(batch["target"], max_length=cfg.tgt_max, truncation=True, padding=False)
        mi["labels"] = lb["input_ids"]
        return mi

    tok_train = splits["train"].map(tok_fn, batched=True, remove_columns=splits["train"].column_names)
    tok_eval  = splits["test"].map(tok_fn,  batched=True, remove_columns=splits["test"].column_names)
    collator  = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    # Training arguments (compatible with TF 4.52.x)
    training_args = TrainingArguments(
        output_dir=cfg.out_adapter_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="constant_with_warmup",
        optim="adafactor",            # T5-friendly
        max_grad_norm=1.0,
        logging_steps=50,
        save_steps=500, save_total_limit=2,
        report_to="none",
        label_names=["labels"],
        fp16=bool(cfg.fp16),
        group_by_length=True,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_eval,
        data_collator=collator,
    )

    print("Starting training...")
    t0 = time.time()
    trainer.train()
    print(f"Training time: {(time.time() - t0)/60:.2f} minutes")

    # Evaluation (final)
    metrics = {}
    try:
        metrics = trainer.evaluate(tok_eval)
        loss = metrics.get("eval_loss")
        if loss is not None:
            ppl = math.exp(loss)
            metrics["eval_perplexity"] = ppl
        print("Eval:", metrics)
    except Exception as e:
        print("Skip evaluate:", e)

    # Save adapter + tokenizer
    Path(cfg.out_adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.out_adapter_dir)
    tokenizer.save_pretrained(cfg.out_adapter_dir)
    lora.save_pretrained(cfg.out_adapter_dir)
    print("Saved adapter to:", cfg.out_adapter_dir)

    # Merge LoRA into base and save
    print("Merging LoRA into base...")
    merged_dir = cfg.out_merged_dir
    Path(merged_dir).mkdir(parents=True, exist_ok=True)

    # Prefer AutoPeftModelForSeq2SeqLM if available
    try:
        from peft import AutoPeftModelForSeq2SeqLM
        merged = AutoPeftModelForSeq2SeqLM.from_pretrained(
            cfg.out_adapter_dir, torch_dtype=(None if not cfg.fp16 else "auto"), low_cpu_mem_usage=True
        )
        merged = merged.merge_and_unload()
    except Exception:
        # Fallback: create fresh base and load adapter, then merge
        from transformers import AutoModelForSeq2SeqLM
        from peft import PeftModel
        base2 = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
        peft_model = PeftModel.from_pretrained(base2, cfg.out_adapter_dir)
        merged = peft_model.merge_and_unload()

    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print("Merged model saved to:", merged_dir)

    return cfg.out_adapter_dir, merged_dir, metrics

# ----------------------------
# Build dataset (main + variants, no dedup)
# ----------------------------
def build_training_dataframe(main_jsonl: str, variant_csv: Optional[str]) -> pd.DataFrame:
    df_main = robust_read_jsonl(main_jsonl)
    df_variants = read_variants_csv(variant_csv)

    # Rows: include canonical + all variants, mapping to same completion; NO DEDUP
    rows = []
    if df_variants is not None and len(df_variants):
        from collections import defaultdict
        var_map = defaultdict(list)
        for _, r in df_variants.iterrows():
            var_map[str(r["canonical_prompt"]).strip()].append(str(r["variant"]).strip())

        for _, r in df_main.iterrows():
            p = str(r["prompt"]).strip()
            c = str(r["completion"]).strip()
            rows.append({"prompt": p, "completion": c, "source": "main"})
            for v in var_map.get(p, []):
                rows.append({"prompt": v, "completion": c, "source": "variant"})
    else:
        for _, r in df_main.iterrows():
            rows.append({
                "prompt": str(r["prompt"]).strip(),
                "completion": str(r["completion"]).strip(),
                "source": "main"
            })

    df_all = pd.DataFrame(rows)
    print(f"Tổng mẫu: {len(df_all)}  (main: {(df_all['source']=='main').sum()} , variant: {(df_all['source']=='variant').sum()})")

    # Normalize target to 3 headings
    df_all["completion"] = df_all["completion"].map(normalize_target)

    # Instruction template (simple & consistent)
    instr = (
        "YÊU CẦU:\n{prompt}\n\n"
        "Hãy tạo test case theo ĐÚNG và CHỈ format 3 mục sau (không thêm mục khác, giữ đúng thuật ngữ):\n"
        f"{HEAD_DESC}\n{HEAD_STEPS}\n{HEAD_EXP}"
    )
    df_all = df_all.assign(
        text=df_all["prompt"].map(lambda p: instr.format(prompt=p)),
        target=df_all["completion"]
    )
    return df_all[["text", "target", "source"]]

# ----------------------------
# Zipping & checksums
# ----------------------------
def zip_dir(src_dir: str, out_zip: str):
    out_zip = out_zip if out_zip.endswith(".zip") else out_zip + ".zip"
    if os.path.exists(out_zip):
        os.remove(out_zip)
    base, _ = os.path.splitext(out_zip)
    shutil.make_archive(base, "zip", src_dir)
    print("Zipped:", out_zip)

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ----------------------------
# Upload to HF (optional)
# ----------------------------
def upload_to_hf(merged_dir: str, adapter_zip: str, merged_zip: str, repo_id: str):
    """
    Uploads merged model folder + ZIP artifacts to Hugging Face.
    Requires env HF_TOKEN; token value is not printed.
    """
    from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder, upload_file

    token = os.getenv("HF_TOKEN", "").strip()
    if not token.startswith("hf_"):
        print("⚠️  HF_TOKEN is not set. Skipping Hugging Face upload.")
        return

    api = HfApi()
    create_repo(repo_id=repo_id, private=False, exist_ok=True, token=token)

    print("Uploading merged model (folder)...")
    upload_folder(
        repo_id=repo_id,
        folder_path=merged_dir,
        repo_type="model",
        token=token,
        commit_message="Upload merged vit5-base (LoRA merged) for testcase generation"
    )

    if os.path.exists(merged_zip):
        print("Uploading merged ZIP artifact...")
        upload_file(
            path_or_fileobj=merged_zip,
            path_in_repo="artifacts/vit5_base_merged.zip",
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

    if os.path.exists(adapter_zip):
        print("Uploading adapter ZIP artifact...")
        upload_file(
            path_or_fileobj=adapter_zip,
            path_in_repo="artifacts/vit5_testcase_lora.zip",
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

    print(f"✅ Upload complete. Repo: https://huggingface.co/{repo_id}")

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Train viT5-base (LoRA) for Vietnamese test case generation.")
    p.add_argument("--main_jsonl", type=str, required=True, help="Path to main JSONL (with 'prompt' & 'completion').")
    p.add_argument("--variant_csv", type=str, default=None, help="Path to prompt variants CSV (optional).")
    p.add_argument("--out_dir", type=str, default="./outputs", help="Base output dir for artifacts.")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=8, help="per_device_train_batch_size")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--src_max", type=int, default=384)
    p.add_argument("--tgt_max", type=int, default=256)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--upload_hf", action="store_true", help="Upload merged + ZIP artifacts to HF (requires env HF_TOKEN).")
    p.add_argument("--repo_id", type=str, default="cammtuxinhdep/vit5_base", help="HF repo to upload to (owner/name).")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # Prepare output dirs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = str(out_dir / "vit5_testcase_lora")
    merged_dir  = str(out_dir / "vit5_base_merged")

    # Build dataframe from data sources
    print("Main JSONL:", args.main_jsonl)
    print("Variant CSV:", args.variant_csv)
    df_all = build_training_dataframe(args.main_jsonl, args.variant_csv)

    # Train and merge
    cfg = TrainConfig(
        out_adapter_dir=adapter_dir,
        out_merged_dir=merged_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch,
        grad_accum=args.grad_accum,
        src_max=args.src_max,
        tgt_max=args.tgt_max,
        seed=args.seed,
    )
    adapter_dir, merged_dir, metrics = train_and_merge(df_all, cfg)

    # Zip artifacts
    adapter_zip = str(out_dir / "vit5_testcase_lora.zip")
    merged_zip  = str(out_dir / "vit5_base_merged.zip")
    zip_dir(adapter_dir, adapter_zip)
    zip_dir(merged_dir,  merged_zip)

    # Print checksums (useful in reports)
    print("\nArtifacts:")
    print(" -", adapter_zip, "sha256:", sha256(adapter_zip))
    print(" -", merged_zip,  "sha256:", sha256(merged_zip))

    # Optional upload to HF
    if args.upload_hf:
        upload_to_hf(merged_dir=merged_dir, adapter_zip=adapter_zip, merged_zip=merged_zip, repo_id=args.repo_id)
    else:
        print("\nℹ️ Skipping HF upload. To enable, pass --upload_hf and export HF_TOKEN=hf_xxx")

if __name__ == "__main__":
    # Defensive import of torch to show device early (non-fatal if missing CUDA libs)
    try:
        import torch
        print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print("Torch import warning:", e)
    sys.exit(main())
