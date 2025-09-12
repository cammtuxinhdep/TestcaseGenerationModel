# eval_metrics.py
import os, re, json, numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from typing import List, Tuple
from tqdm import tqdm

# ========= Config =========
CORPUS_CSV = "data/processed/result/corpus.csv"
HF_MODEL_REPO = "cammtuxinhdep/vit5_base"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOPK_RETRIEVE = 5   # k ứng viên dùng làm ngữ cảnh cho Generator
MAX_NEW_TOKENS = 220

# ========= Utils =========
def norm_text(s: str) -> str:
    s = (s or "").replace("\r","").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def join_blocks(d, i, s, e) -> str:
    return f"Mô tả:\n{d}\n\nInputs:\n{i}\n\nSteps:\n{s}\n\nExpected:\n{e}".strip()

def parse_4blocks(txt: str) -> Tuple[str,str,str,str]:
    # parse theo 4 heading (VI) bạn đã chuẩn hoá
    HEAD_DESC   = "Mô tả (description):"
    HEAD_INPUTS = "Inputs (dữ liệu kiểm thử):"
    HEAD_STEPS  = "Các bước chính (main steps):"
    HEAD_EXP    = "Đầu ra mong muốn (expected output):"
    t = (txt or "").replace("\r","")
    def take(a,b):
        la, lb = t.lower().find(a.lower()), t.lower().find(b.lower())
        if la>=0 and lb>la: return t[la+len(a):lb].strip()
        if la>=0 and lb<0:  return t[la+len(a):].strip()
        return ""
    d  = take(HEAD_DESC,   HEAD_INPUTS)
    i_ = take(HEAD_INPUTS, HEAD_STEPS)
    s  = take(HEAD_STEPS,  HEAD_EXP)
    e  = t.split(HEAD_EXP,1)[-1].strip() if HEAD_EXP.lower() in t.lower() else ""
    def clean(x):
        x = re.sub(r"^\s*[-•*]\s*", "", x, flags=re.M)
        x = re.sub(r"[ \t]+", " ", x)
        return "\n".join([ln.strip() for ln in x.splitlines() if ln.strip()])
    return clean(d), clean(i_), clean(s), clean(e)

# ========= Load data =========
df = pd.read_csv(CORPUS_CSV).fillna("")
REQ_COLS = ["description","inputs","steps","expected_output"]
for c in REQ_COLS + ["prompt"]:
    if c not in df.columns: df[c] = ""
df["reference_text"] = df.apply(lambda r: join_blocks(r["description"], r["inputs"], r["steps"], r["expected_output"]), axis=1)

# ========= Embedding & FAISS =========
from sentence_transformers import SentenceTransformer
EMB = SentenceTransformer(EMB_MODEL_NAME)
def embed_texts(texts: List[str]):
    return EMB.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

CORPUS_BAG = (df["description"].astype(str) + "\n" + df["steps"].astype(str) + "\n" + df["expected_output"].astype(str)).tolist()
CORPUS_VEC = embed_texts(CORPUS_BAG)

import faiss
INDEX = faiss.IndexFlatIP(CORPUS_VEC.shape[1])
INDEX.add(CORPUS_VEC.astype(np.float32))

def retrieve_topk(prompt: str, k: int) -> List[int]:
    vq = embed_texts([prompt]).astype(np.float32)
    scores, idxs = INDEX.search(vq, min(k, len(df)))
    return idxs[0].tolist()

# ========= Generator (ViT5 LoRA merged) =========
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
MODEL = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_REPO).to(DEVICE); MODEL.eval()

HEAD_DESC   = "Mô tả (description):"
HEAD_INPUTS = "Inputs (dữ liệu kiểm thử):"
HEAD_STEPS  = "Các bước chính (main steps):"
HEAD_EXP    = "Đầu ra mong muốn (expected output):"

def make_instruction(query: str, contexts: List[str]) -> str:
    ctx_blob = "\n\n".join([f"- {c}" for c in contexts]) if contexts else ""
    return (
        f"YÊU CẦU:\n{query.strip()}\n\n"
        f"Ngữ cảnh tham khảo (nếu có):\n{ctx_blob}\n\n"
        "Hãy tạo test case THEO ĐÚNG và CHỈ 4 mục sau (đúng thứ tự & heading):\n"
        f"{HEAD_DESC}\n{HEAD_INPUTS}\n{HEAD_STEPS}\n{HEAD_EXP}"
    ).strip()

def generate_4blocks(query: str, contexts: List[str]) -> Tuple[str,str,str,str]:
    instr = make_instruction(query, contexts)
    inp = TOKENIZER(instr, return_tensors="pt", truncation=True, max_length=384).to(DEVICE)
    with torch.no_grad():
        out = MODEL.generate(
            **inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
            top_p=0.92, temperature=0.7, num_return_sequences=1,
            pad_token_id=TOKENIZER.pad_token_id
        )[0]
    txt = TOKENIZER.decode(out, skip_special_tokens=True)
    return parse_4blocks(txt)

# ========= Hai chế độ =========
def run_retrieval_only(prompt: str) -> Tuple[str,str,str,str,str]:
    idx = retrieve_topk(prompt, 1)[0]
    d_, i_, s_, e_ = df.loc[idx, ["description","inputs","steps","expected_output"]].tolist()
    hyp = join_blocks(d_, i_, s_, e_)
    return hyp, d_, i_, s_, e_

def run_retrieval_plus_generator(prompt: str) -> Tuple[str,str,str,str,str]:
    idxs = retrieve_topk(prompt, TOPK_RETRIEVE)
    ctxs = []
    for i in idxs:
        r = df.loc[i]
        ctxs.append(join_blocks(r["description"], r["inputs"], r["steps"], r["expected_output"]))
    d, i_, s, e = generate_4blocks(prompt, ctxs)
    hyp = join_blocks(d, i_, s, e)
    return hyp, d, i_, s, e

# ========= Chỉ số =========
def exact_match(refs, hyps):
    return float(np.mean([norm_text(a)==norm_text(b) for a,b in zip(refs,hyps)]))

def cosine_mean(refs, hyps):
    ref_vec = embed_texts(refs)
    hyp_vec = embed_texts(hyps)
    sims = (ref_vec * hyp_vec).sum(axis=1)
    return float(np.mean(sims))

from rouge_score import rouge_scorer
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
def rougeL_f1_mean(refs, hyps):
    vals = [rouge.score(r,h)["rougeL"].fmeasure for r,h in zip(refs,hyps)]
    return float(np.mean(vals))

import sacrebleu
def bleu_mean(refs, hyps):
    return float(sacrebleu.corpus_bleu(hyps, [refs]).score / 100.0)

from bert_score import score as bertscore
def bert_f1_mean(refs, hyps):
    P,R,F1 = bertscore(hyps, refs, lang="vi", rescale_with_baseline=True)
    return float(F1.mean().item())

# ========= Chạy & ghi kết quả =========
def evaluate_mode(mode_name: str, runner):
    refs, hyps, rows = [], [], []
    for idx, r in tqdm(df.iterrows(), total=len(df), desc=mode_name):
        prompt = r.get("prompt","") or "Tạo test case đăng nhập"
        ref = r["reference_text"]
        hyp, d, i_, s, e = runner(prompt)
        refs.append(ref); hyps.append(hyp)
        rows.append({
            "mode": mode_name, "row_id": idx,
            "prompt": prompt,
            "reference": ref, "hypothesis": hyp,
            "d_gen": d, "i_gen": i_, "s_gen": s, "e_gen": e
        })
    # tính metrics
    summ = {
        "mode": mode_name,
        "exact": round(exact_match(refs, hyps), 4),
        "cosine": round(cosine_mean(refs, hyps), 4),
        "rougeL": round(rougeL_f1_mean(refs, hyps), 4),
        "bleu": round(bleu_mean(refs, hyps), 4),
        "bertscore_f1": round(bert_f1_mean(refs, hyps), 4),
    }
    return rows, summ

all_rows = []
summary = []

rows_ret, summ_ret = evaluate_mode("retrieval_only",
                                   lambda q: run_retrieval_only(q))
all_rows += rows_ret; summary.append(summ_ret)

rows_rg,  summ_rg  = evaluate_mode("retrieval_plus_generator",
                                   lambda q: run_retrieval_plus_generator(q))
all_rows += rows_rg; summary.append(summ_rg)

pd.DataFrame(all_rows).to_csv("eval_detailed.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(summary).to_csv("eval_summary.csv", index=False, encoding="utf-8-sig")

print("\n=== eval_summary.csv ===")
print(pd.DataFrame(summary))
print("\nĐã ghi eval_detailed.csv & eval_summary.csv")
