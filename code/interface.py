import os, re, io, json, time, unicodedata, warnings
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import gradio as gr

warnings.filterwarnings("ignore")

# ---------------- Config ----------------
CORPUS_CSV     = os.getenv("CORPUS_CSV", "data/processed/result/corpus.csv")
HF_MODEL_REPO  = os.getenv("HF_MODEL_REPO", "cammtuxinhdep/vit5_base")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

TOPK_INITIAL = int(os.getenv("TOPK_INITIAL", "300"))
BATCH_RETURN = int(os.getenv("BATCH_RETURN", "10"))

COL_DESC, COL_INPUTS, COL_STEPS, COL_EXPECT = "description", "inputs", "steps", "expected_output"
DISPLAY_COLS = [COL_DESC, COL_INPUTS, COL_STEPS, COL_EXPECT, "feature", "aspect"]

CONTENT_COLS = [COL_DESC, COL_INPUTS, COL_STEPS, COL_EXPECT]
TAG_COLS     = ["platform", "domain", "feature", "aspect", "scope"]

# ---- NLU ----
NLU = {
    "platform": {
        "web": ["web", "trinh duyet", "trình duyệt", "chrome", "edge", "firefox"],
        "ứng dụng": ["app", "ung dung", "ứng dụng", "android", "ios", "mobile"],
        "cả hai": ["ca hai", "cả hai", "da nen tang", "đa nền tảng", "cross-platform", "cross platform", "đa nền tàng"],
    },
    "domain": {
        "tổng quát": ["tong quat", "tổng quát", "common", "chung"],
        "ngân hàng": ["ngan hang", "ngân hàng", "bank"],
        "thương mại điện tử": ["tmdt", "tmđt", "ecommerce", "thuong mai dien tu", "thương mại điện tử", "mua sam", "mua sắm", "shopping"],
        "máy tính": ["desktop", "may tinh", "máy tính"],
        "salesforce": ["salesforce"],
        "notepad": ["notepad"],
    },
    "feature": {
        "chung": ["tat ca", "tất cả", "day du", "đầy đủ", "toan bo", "toàn bộ"],
        "đăng nhập": ["dang nhap", "đăng nhập", "login", "sign in", "signin"],
        "đăng ký": ["dang ky", "đăng ký", "register", "signup", "sign up"],
        "đăng xuất": ["dang xuat", "đăng xuất", "logout", "sign out"],
        "tìm kiếm": ["tim kiem", "tìm kiếm", "search"],
        "phân trang": ["phan trang", "phân trang", "pagination"],
        "tải lên hình ảnh": ["tai len hinh", "tải lên hình", "upload anh", "upload ảnh", "image upload", "upload image"],
        "lọc": ["loc", "lọc", "filter"],
        "email": ["email", "mail"],
        "quản trị": ["quan tri", "quản trị", "admin"],
        "quản lý chi nhánh": ["chi nhanh", "chi nhánh", "branch"],
        "quản lý tài khoản": ["tai khoan", "tài khoản", "account"],
        "quản lý giao dịch": ["giao dich", "giao dịch", "transaction"],
        "yêu cầu dịch vụ": ["yeu cau dich vu", "yêu cầu dịch vụ", "service request"],
        "xử lý giao dịch": ["xu ly giao dich", "xử lý giao dịch", "process transaction"],
        "quản lý kho": ["kho", "inventory", "warehouse"],
        "danh mục & sản phẩm": ["danh muc", "sản phẩm", "san pham", "catalog", "product"],
        "giỏ hàng": ["gio hang", "giỏ hàng", "cart"],
        "thanh toán": ["thanh toan", "thanh toán", "checkout", "payment"],
        "vận chuyển & giao hàng": ["van chuyen", "giao hang", "vận chuyển", "giao hàng", "shipping", "delivery"],
        "xuất excel": ["xuat excel", "xuất excel", "export excel", "xlsx"],
        "lịch": ["lich", "lịch", "calendar", "ics"],
        "tính toán": ["tinh toan", "tính toán", "calculator"],
        "phi chức năng": ["phi chuc nang", "phi chức năng", "non-functional"],
    },
    "aspect": {
        "chức năng": ["chuc nang", "chức năng", "functional", "feature"],
        "hiệu năng": ["hieu nang", "hiệu năng", "performance", "tai", "tải", "do tre", "độ trễ", "latency", "toi uu", "tối ưu", "throughput"],
        "bảo mật": ["bao mat", "bảo mật", "security", "xss", "csrf", "sql injection", "phan quyen", "phân quyền", "ma hoa", "mã hóa"],
        "trải nghiệm": ["trai nghiem", "trải nghiệm", "usability", "ux", "ui", "giao dien", "giao diện", "hien thi", "hiển thị"],
        "dữ liệu": ["du lieu", "dữ liệu", "validation", "dinh dang", "định dạng", "format", "database"],
        "api": ["api", "webhook"],
        "đa nền tảng": ["da nen tang", "đa nền tảng", "cross-platform"],
    },
}

# --------- Utils ----------
def strip_accents(s: str) -> str:
    s = (s or "").lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def ensure_content_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in CONTENT_COLS + TAG_COLS:
        if c not in df.columns: df[c] = ""
    return df

def load_corpus(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy corpus: {path}")
    df = pd.read_csv(path).fillna("")
    df = ensure_content_cols(df)
    for c in TAG_COLS:
        df[f"_{c}_norm"] = df[c].map(strip_accents)
    df["_bag"] = (
        df[COL_DESC].astype(str) + "\n" +
        df[COL_STEPS].astype(str) + "\n" +
        df[COL_EXPECT].astype(str)
    ).map(strip_accents)
    return df

CORPUS = load_corpus(CORPUS_CSV)

# ---- Embedding + FAISS ----
USE_FAISS = False
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMB = SentenceTransformer(EMB_MODEL_NAME)
    VEC = EMB.encode(CORPUS["_bag"].tolist(), normalize_embeddings=True, convert_to_numpy=True)
    INDEX = faiss.IndexFlatIP(VEC.shape[1]); INDEX.add(VEC.astype(np.float32))
    USE_FAISS = True
except Exception:
    EMB = None; VEC = None; INDEX = None

# --------- Intent parsing ----------
def parse_intent(text: str) -> Dict[str, Optional[List[str]]]:
    t_norm = strip_accents(text or "")
    out = {"platform": [], "domain": [], "feature": [], "aspect": []}
    for group in ["platform","domain","feature","aspect"]:
        hits = []
        for label, kws in NLU[group].items():
            if any(strip_accents(kw) in t_norm for kw in kws):
                hits.append(label)
        out[group] = list(dict.fromkeys(hits)) if hits else None
    return out

def filter_by_tags(df: pd.DataFrame, intent: Dict[str, Optional[List[str]]]) -> pd.DataFrame:
    d = df.copy()

    # platform
    plats = intent.get("platform") or []
    if plats:
        norm = [strip_accents(x) for x in plats]
        d = d[d["_platform_norm"].isin(norm) | (d["_platform_norm"] == strip_accents("cả hai"))]
    else:
        d = d[d["_platform_norm"] == strip_accents("cả hai")]

    # domain
    doms = intent.get("domain") or []
    if doms:
        norm = [strip_accents(x) for x in doms]
        d = d[d["_domain_norm"].isin(norm) | (d["_scope_norm"] == strip_accents("dùng chung"))]
    else:
        d = d[d["_scope_norm"] == strip_accents("dùng chung")]

    # aspect
    asps = intent.get("aspect") or []
    if asps:
        norm = set(strip_accents(x) for x in asps)
        d = d[d["_aspect_norm"].isin(norm)]

    # feature
    feats = intent.get("feature") or []
    if feats:
        norm = [strip_accents(x) for x in feats]
        if strip_accents("chung") not in norm:
            d = d[d["_feature_norm"].isin(norm)]

    desired_feat = set(strip_accents(x) for x in (intent.get("feature") or []))
    desired_asp  = set(strip_accents(x) for x in (intent.get("aspect") or []))

    def _boost_row(r):
        b = 0
        if desired_feat and r["_feature_norm"] in desired_feat: b += 2
        if desired_asp  and r["_aspect_norm"]  in desired_asp:  b += 1
        return b

    d["_match_boost"] = d.apply(_boost_row, axis=1)
    return d

def retrieve_indices(query: str, pool_index: np.ndarray, topk: int) -> List[int]:
    q = strip_accents(query)
    if USE_FAISS and EMB is not None:
        qv = EMB.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        scores, idxs = INDEX.search(qv, min(topk, len(CORPUS)))
        return idxs[0].tolist()
    # fallback TF-IDF cosine (trên pool)
    from sklearn.feature_extraction.text import TfidfVectorizer
    bags = CORPUS.loc[pool_index, "_bag"].tolist()
    vec = TfidfVectorizer(min_df=1).fit(bags + [q])
    M = vec.transform(bags); vq = vec.transform([q])
    sims = (M @ vq.T).toarray().ravel()
    order = np.argsort(-sims)
    return pool_index[order][:topk].tolist()

# --------- HF generator + parser (4 heading) ----------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
MODEL = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_REPO).to(DEVICE); MODEL.eval()

HEAD_DESC   = "Mô tả (description):"
HEAD_INPUTS = "Inputs (dữ liệu kiểm thử):"
HEAD_STEPS  = "Các bước chính (main steps):"
HEAD_EXP    = "Đầu ra mong muốn (expected output):"

def make_instruction(user_query: str) -> str:
    return (
        "YÊU CẦU:\n" + (user_query or "").strip() +
        "\n\nHãy tạo test case THEO ĐÚNG và CHỈ 4 mục sau, đúng thứ tự & heading:\n" +
        f"{HEAD_DESC}\n{HEAD_INPUTS}\n{HEAD_STEPS}\n{HEAD_EXP}"
    )

def parse_4blocks(txt: str) -> Tuple[str,str,str,str]:
    t = txt.replace("\r","")
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

# --------- SUY LUẬN feature/aspect từ văn bản ----------
def infer_tags_from_text(txt: str) -> Tuple[Optional[str], Optional[str]]:
    t = strip_accents(txt or "")
    feat_hit, asp_hit = None, None

    for label, kws in NLU["feature"].items():
        if any(strip_accents(k) in t for k in kws):
            feat_hit = label
            break

    for label, kws in NLU["aspect"].items():
        if any(strip_accents(k) in t for k in kws):
            asp_hit = label
            break

    return feat_hit, asp_hit

def generate_cases(user_query: str, n: int, *, intent=None) -> pd.DataFrame:
    instr = make_instruction(user_query)
    inp = TOKENIZER(instr, return_tensors="pt", truncation=True, max_length=384).to(DEVICE)
    with torch.no_grad():
        outs = MODEL.generate(
            **inp, max_new_tokens=220, do_sample=True, top_p=0.92, temperature=0.7,
            num_return_sequences=n, pad_token_id=TOKENIZER.pad_token_id
        )

    rows, feats, asps = [], [], []
    for o in outs:
        txt = TOKENIZER.decode(o, skip_special_tokens=True)
        d, i_, s, e = parse_4blocks(txt)
        rows.append([d, i_, s, e])

        # 1) thử suy luận từ mô tả/steps
        feat, asp = infer_tags_from_text(d + "\n" + s)

        # 2) không có thì lấy từ intent (nếu user có nêu)
        if not feat and intent and intent.get("feature"):
            feat = intent["feature"][0]
        if not asp and intent and intent.get("aspect"):
            asp = intent["aspect"][0]

        feats.append(feat or "")
        asps.append(asp or "")

    df = pd.DataFrame(rows, columns=CONTENT_COLS)
    df["feature"] = feats
    df["aspect"]  = asps
    return df

# --------- Orchestrator ----------
def build_related(user_query: str, already: int) -> Tuple[str, pd.DataFrame]:
    uq = (user_query or "").strip()
    if not uq:
        return "Vui lòng nhập ngữ cảnh.", pd.DataFrame(columns=DISPLAY_COLS)

    intent  = parse_intent(uq)

    # Lọc theo platform/domain (+optional aspect/feature)
    pool_df = filter_by_tags(CORPUS, intent)
    pool_idx = pool_df.index.to_numpy() if len(pool_df) else CORPUS.index.to_numpy()

    # Tìm theo ngữ nghĩa
    idxs = retrieve_indices(uq, pool_idx, topk=min(TOPK_INITIAL, len(pool_idx)))
    df_cands = CORPUS.loc[idxs]
    df_cands = filter_by_tags(df_cands, intent).sort_values("_match_boost", ascending=False)

    # Lấy batch từ corpus
    take_df = df_cands[DISPLAY_COLS].head(already + BATCH_RETURN).tail(BATCH_RETURN)

    # Kiểm tra: user có yêu cầu feature (không phải 'chung') nhưng trong pool không có?
    feat_req = (intent.get("feature") or [])
    feat_req_norm = [strip_accents(x) for x in feat_req if strip_accents(x) != strip_accents("chung")]
    have_feature_in_pool = False
    if feat_req_norm and len(pool_df):
        pool_feats = set(pool_df["_feature_norm"].tolist())
        have_feature_in_pool = any(f in pool_feats for f in feat_req_norm)

    # Nếu thiếu -> sinh mới; nếu feature lạ và không suy luận được thì gán 'phi chức năng'
    lack = BATCH_RETURN - len(take_df)
    if lack > 0:
        gen_df = generate_cases(uq, lack, intent=intent)

        if feat_req_norm and not have_feature_in_pool:
            # fallback cuối: chưa gán được feature -> gán 'phi chức năng'
            empty_mask = gen_df["feature"].astype(str).str.strip() == ""
            gen_df.loc[empty_mask, "feature"] = "phi chức năng"

        take_df = pd.concat([take_df, gen_df[DISPLAY_COLS]], ignore_index=True)

    msg = f"Đã sinh {len(take_df)} test case liên quan."
    return msg, take_df

def export_visible(df_state: pd.DataFrame) -> Tuple[str, str]:
    if df_state is None or df_state.empty:
        return "❌ Chưa có dữ liệu để xuất.", ""
    out = f"testcases_visible_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df_state.to_excel(out, index=False)
    return f"✅ Đã xuất {out}", out

def export_all(user_query: str, stash_df: pd.DataFrame) -> Tuple[str, str]:
    uq = (user_query or "").strip()
    if not uq:
        return "❌ Bạn cần nhập ngữ cảnh trước khi xuất toàn bộ.", ""
    intent = parse_intent(uq)
    df_all = filter_by_tags(CORPUS, intent)[DISPLAY_COLS]
    if stash_df is not None and not stash_df.empty:
        df_all = pd.concat([df_all, stash_df[DISPLAY_COLS]], ignore_index=True)
    df_all = df_all.drop_duplicates()
    if df_all.empty:
        return "❌ Không có dữ liệu phù hợp để xuất.", ""
    out = f"testcases_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df_all.to_excel(out, index=False)
    return f"✅ Đã xuất {len(df_all)} test case vào {out}", out

# --------- UI ----------
CUSTOM_CSS = """
body, .gradio-container { background:#eaf2ff !important; }
.gradio-container footer, footer { display:none !important; }
.panel { background:#dbe7ff; border:1px solid #b5ccff; border-radius:14px; padding:16px; box-shadow:0 2px 10px rgba(0,0,0,.05); }
.panel-title { display:inline-block; background:#cfe0ff; color:#1b3f8a; padding:6px 12px; border-radius:10px; font-weight:700; margin-bottom:10px; }
#tbl { max-height:520px; overflow:auto; background:#fff; border-radius:10px; border:1px solid #cfe1fd; }
#tbl th{ position:sticky; top:0; background:#f6f9ff; z-index:1; }
"""

try:
    THEME = gr.themes.Soft()
except Exception:
    THEME = None

with gr.Blocks(title="Mô hình Tự động sinh test case theo ngữ cảnh", theme=THEME, css=CUSTOM_CSS) as demo:
    gr.Markdown("<h1 style='text-align:center'>Mô hình Tự động sinh test case theo ngữ cảnh</h1>")

    with gr.Row():
        with gr.Column(scale=5, elem_classes=["panel"]):
            gr.Markdown('<span class="panel-title">Nhập ngữ cảnh</span>')
            qbox = gr.Textbox(placeholder="Ví dụ: Sinh test case cho chức năng đăng nhập web thương mại điện tử...", lines=6, show_label=False)

            gen_btn = gr.Button("Sinh 10 test case liên quan")

            with gr.Row():
                export_vis_btn = gr.Button("Xuất test case đang hiển thị")
                export_all_btn = gr.Button("Xuất toàn bộ test case")

            note = gr.Textbox(show_label=False, interactive=False, placeholder="Thông báo")

        with gr.Column(scale=7, elem_classes=["panel"]):
            gr.Markdown('<span class="panel-title">Bảng test case</span>')
            table = gr.Dataframe(
                headers=DISPLAY_COLS,
                datatype=["str","str","str","str","str","str"],
                interactive=False, wrap=True, row_count=(0,"dynamic"),
                elem_id="tbl", show_label=False
            )

    # states
    last_query  = gr.State("")
    shown_state = gr.Number(value=0, visible=False, precision=0)
    table_state = gr.State(pd.DataFrame(columns=DISPLAY_COLS))
    stash_state = gr.State(pd.DataFrame(columns=DISPLAY_COLS))   # cộng dồn cho export-all

    downloader = gr.DownloadButton("Tải xuống", visible=False)

    # handlers
    def _gen(q, last_q, shown, current_table, stash):
        q = (q or "").strip()
        # nếu ngữ cảnh đổi -> reset
        if q != (last_q or "").strip():
            msg, df_batch = build_related(q, 0)
            new_table = df_batch.copy()
            new_shown = len(df_batch)
            new_stash = df_batch.copy()
            return msg, new_table, new_shown, new_table, new_stash, q, gr.update(visible=False, value=None)

        # giữ nguyên ngữ cảnh -> cộng dồn
        msg, df_batch = build_related(q, int(shown))
        new_table = pd.concat([current_table, df_batch], ignore_index=True)
        new_shown = int(shown) + len(df_batch)
        new_stash = pd.concat([stash, df_batch], ignore_index=True)
        return msg, new_table, new_shown, new_table, new_stash, last_q, gr.update(visible=False, value=None)

    def _export_visible(df_state):
        msg, path = export_visible(df_state)
        return msg, (gr.update(value=path, visible=True) if path else gr.update(visible=False, value=None))

    def _export_all(q, stash):
        msg, path = export_all(q, stash)
        return msg, (gr.update(value=path, visible=True) if path else gr.update(visible=False, value=None))

    gen_btn.click(_gen,
                  inputs=[qbox, last_query, shown_state, table_state, stash_state],
                  outputs=[note, table, shown_state, table_state, stash_state, last_query, downloader])

    export_vis_btn.click(_export_visible, inputs=[table_state], outputs=[note, downloader])
    export_all_btn.click(_export_all, inputs=[qbox, stash_state], outputs=[note, downloader])

if __name__ == "__main__":
    demo.launch()
