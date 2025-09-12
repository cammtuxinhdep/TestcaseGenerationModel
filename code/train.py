from __future__ import annotations
import os, re, json, math, time, argparse, shutil, hashlib, random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model

SEED = 42
random.seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

HEAD_DESC   = "Mô tả (description):"
HEAD_INPUTS = "Inputs (dữ liệu kiểm thử):"
HEAD_STEPS  = "Các bước chính (main steps):"
HEAD_EXP    = "Đầu ra mong muốn (expected output):"

def ensure_4h(s: str) -> str:
    """Đảm bảo có đủ 4 heading và đúng thứ tự mô tả→inputs→steps→expected."""
    s = (s or "").strip()
    # nếu lẫn lộn thứ tự, cắt lại theo tên heading
    def take(a, b):
        ia = s.find(a); ib = s.find(b)
        if ia != -1 and ib != -1 and ib > ia:
            return s[ia+len(a):ib].strip()
        if ia != -1 and (ib == -1):
            return s[ia+len(a):].strip()
        return ""
    d  = take(HEAD_DESC,   HEAD_INPUTS) or take(HEAD_DESC,   HEAD_STEPS) or take(HEAD_DESC,   HEAD_EXP)
    ip = take(HEAD_INPUTS, HEAD_STEPS)  or take(HEAD_INPUTS, HEAD_EXP)
    st = take(HEAD_STEPS,  HEAD_EXP)
    ex = s.split(HEAD_EXP,1)[1].strip() if HEAD_EXP in s else ""

    # fallback nếu không parse được -> giữ nguyên
    if not any([d, ip, st, ex]):
        d = s

    parts = [
        f"{HEAD_DESC}\n{d.strip()}",
        f"{HEAD_INPUTS}\n{ip.strip()}",
        f"{HEAD_STEPS}\n{st.strip()}",
        f"{HEAD_EXP}\n{ex.strip()}",
    ]
    txt = "\n".join(parts).strip()
    # dọn trắng
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt

def robust_read_jsonl(path: str) -> pd.DataFrame:
    rows=[]
    with open(path,"r",encoding="utf-8",errors="replace") as f:
        for ln, line in enumerate(f, 1):
            line=line.strip()
            if not line: continue
            try:
                obj=json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict): continue
            if not {"prompt","completion"}.issubset(obj):
                continue
            rows.append({"prompt": str(obj["prompt"]).strip(),
                         "completion": ensure_4h(str(obj["completion"]))})
    if not rows:
        raise RuntimeError("train.jsonl rỗng hoặc thiếu field.")
    return pd.DataFrame(rows)

def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def zip_dir(src_dir: str, out_zip: str):
    if os.path.exists(out_zip): os.remove(out_zip)
    base, _ = os.path.splitext(out_zip)
    shutil.make_archive(base, "zip", src_dir)
    print("Zipped:", out_zip)

def train(args):
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    df = robust_read_jsonl(args.train_jsonl)
    print("Rows:", len(df))

    INSTR = (
        "YÊU CẦU:\n{prompt}\n\n"
        "Hãy trả lời theo ĐÚNG 4 heading sau, không thêm mục khác, giữ nguyên tiếng Việt:\n"
        f"{HEAD_DESC}\n{HEAD_INPUTS}\n{HEAD_STEPS}\n{HEAD_EXP}"
    )
    df = df.assign(
        text   = df["prompt"].map(lambda p: INSTR.format(prompt=p)),
        target = df["completion"]
    )
    ds = Dataset.from_pandas(df[["text","target"]], preserve_index=False)
    splits = ds.train_test_split(test_size=0.1, seed=SEED)
    print("Train/Eval:", len(splits["train"]), len(splits["test"]))

    model_name = "VietAI/vit5-base"
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if torch.cuda.is_available(): base = base.half().to("cuda")
    base.config.use_cache = False
    if getattr(base.config,"decoder_start_token_id",None) is None:
        base.config.decoder_start_token_id = tok.pad_token_id

    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q","k","v","o"],
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(base, lora)
    model.gradient_checkpointing_enable()
    try: model.enable_input_require_grads()
    except Exception: pass

    SRC_MAX, TGT_MAX = args.src_max, args.tgt_max
    def tok_fn(b):
        mi = tok(b["text"], max_length=SRC_MAX, truncation=True, padding=False)
        with tok.as_target_tokenizer():
            lb = tok(b["target"], max_length=TGT_MAX, truncation=True, padding=False)
        mi["labels"] = lb["input_ids"]
        return mi

    tr  = splits["train"].map(tok_fn, batched=True, remove_columns=splits["train"].column_names)
    ev  = splits["test"].map(tok_fn,  batched=True, remove_columns=splits["test"].column_names)
    coll= DataCollatorForSeq2Seq(tokenizer=tok, model=model, pad_to_multiple_of=8)

    out_adapter = Path(args.out_dir) / "vit5_testcase_lora"
    out_merged  = Path(args.out_dir) / "vit5_base_merged"
    out_adapter.mkdir(parents=True, exist_ok=True)
    out_merged.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out_adapter),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio, lr_scheduler_type="constant_with_warmup",
        optim="adafactor",
        logging_steps=50, save_steps=500, save_total_limit=2,
        report_to="none", fp16=torch.cuda.is_available(),
        group_by_length=True, seed=SEED,
    )

    trainer = Trainer(model=model, args=targs,
                      train_dataset=tr, eval_dataset=ev,
                      data_collator=coll)
    print("Start training...")
    t0=time.time()
    trainer.train()
    print(f"Training time: {(time.time()-t0)/60:.2f} minutes")

    try:
        metrics = trainer.evaluate(ev)
        if "eval_loss" in metrics:
            metrics["eval_ppl"] = float(math.exp(metrics["eval_loss"]))
        print("Eval:", metrics)
    except Exception as e:
        print("Skip evaluate:", e)

    # save adapter
    model.save_pretrained(str(out_adapter))
    tok.save_pretrained(str(out_adapter))
    print("Saved adapter to:", out_adapter)

    # merge
    print("Merging LoRA into base...")
    from peft import AutoPeftModelForSeq2SeqLM, PeftModel
    try:
        merged = AutoPeftModelForSeq2SeqLM.from_pretrained(
            str(out_adapter),
            torch_dtype=(torch.float16 if torch.cuda.is_available() else None),
            low_cpu_mem_usage=True
        ).merge_and_unload()
    except Exception:
        base2 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        peft = PeftModel.from_pretrained(base2, str(out_adapter))
        merged = peft.merge_and_unload()

    merged.save_pretrained(str(out_merged), safe_serialization=True)
    tok.save_pretrained(str(out_merged))
    print("Merged saved to:", out_merged)

    # zip
    azip = str(Path(args.out_dir) / "vit5_testcase_lora.zip")
    mzip = str(Path(args.out_dir) / "vit5_base_merged.zip")
    zip_dir(str(out_adapter), azip)
    zip_dir(str(out_merged),  mzip)
    print("SHA256:")
    print(" adapter:", sha256(azip))
    print(" merged :", sha256(mzip))

    # upload HF optional
    if args.upload_hf:
        from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder, upload_file
        token = os.getenv("HF_TOKEN", "")
        if not token.startswith("hf_"):
            print("HF_TOKEN chưa có, bỏ qua upload.")
        else:
            api = HfApi()
            create_repo(repo_id=args.repo_id, private=False, exist_ok=True, token=token)
            print("Uploading merged folder ...")
            upload_folder(repo_id=args.repo_id, folder_path=str(out_merged),
                          repo_type="model", token=token,
                          commit_message="Upload merged viT5 LoRA (testcase)")
            upload_file(path_or_fileobj=mzip, path_in_repo="artifacts/vit5_base_merged.zip",
                        repo_id=args.repo_id, repo_type="model", token=token)
            upload_file(path_or_fileobj=azip, path_in_repo="artifacts/vit5_testcase_lora.zip",
                        repo_id=args.repo_id, repo_type="model", token=token)
            print(f"Done: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True, help="Đường dẫn train.jsonl (prompt + completion 4 heading)")
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--src_max", type=int, default=384)
    ap.add_argument("--tgt_max", type=int, default=320)
    ap.add_argument("--upload_hf", action="store_true")
    ap.add_argument("--repo_id", default="yourname/vit5_testcase")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train(args)
