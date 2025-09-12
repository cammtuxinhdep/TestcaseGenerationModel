from __future__ import annotations
import argparse, re, json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# Headings (VI)
H_DESC   = "Mô tả (description):"
H_INPUTS = "Inputs (dữ liệu kiểm thử):"
H_STEPS  = "Các bước chính (main steps):"
H_EXP    = "Đầu ra mong muốn (expected output):"
H_ORDER  = [H_DESC, H_INPUTS, H_STEPS, H_EXP]

# Từ khoá bắt cột linh hoạt (tiếng Việt/Anh)
CANDIDATE_COLS = {
    "platform": ["platform", "nền tảng", "nen tang"],
    "domain":   ["domain", "lĩnh vực", "linh vuc"],
    "feature":  ["feature", "chức năng", "chuc nang"],
    "aspect":   ["aspect", "khía cạnh", "khia canh"],
    "scope":    ["scope", "phạm vi", "pham vi"],
    "context":  ["context", "ngữ cảnh", "ngu canh", "tiêu đề", "tieu de", "title", "name"],
    "prompt":   ["prompt", "yêu cầu", "yeu cau"],
    # 4 khối nội dung
    "desc":     ["mô tả", "mo ta", "description", "mo_ta", "desc"],
    "inputs":   ["inputs", "input", "dữ liệu", "du lieu", "test data", "dau vao"],
    "steps":    ["các bước", "cac buoc", "steps", "procedure", "bước", "buoc"],
    "expected": ["expected", "đầu ra", "dau ra", "kỳ vọng", "ky vong", "output", "expected output"],
}

def _norm(s: str) -> str:
    if pd.isna(s): return ""
    return str(s).replace("\r", "").strip()

def _match_col(df_cols, aliases):
    df_lc = {str(c).strip().lower(): c for c in df_cols}
    for a in aliases:
        if a in df_lc: return df_lc[a]
    for a in aliases:
        for k, v in df_lc.items():
            if a in k:
                return v
    return None

def _make_prompt(row: pd.Series, has_prompt_col: bool) -> str:
    if has_prompt_col and row.get("__col_prompt", ""):
        p = _norm(row["__col_prompt"])
        if p: return p

    plat = _norm(row.get("__col_platform", ""))
    dom  = _norm(row.get("__col_domain", ""))
    feat = _norm(row.get("__col_feature", ""))
    asp  = _norm(row.get("__col_aspect", ""))
    ctx  = _norm(row.get("__col_context", ""))

    parts = []
    if ctx: parts.append(ctx)
    base = "Tạo test case"
    if feat and asp:
        base += f" cho {feat} - {asp}"
    elif feat:
        base += f" cho {feat}"
    elif asp:
        base += f" - {asp}"
    if dom:
        base += f" ({dom})"
    if plat:
        base += f" [{plat}]"
    parts.append(base)
    return " - ".join([p for p in parts if p]).strip()

def _default_desc(prompt: str) -> str:
    return "Mô tả ngắn gọn mục tiêu kiểm thử theo yêu cầu."

def _default_inputs(prompt: str) -> str:
    p = prompt.lower()
    if "đăng nhập" in p or "login" in p:
        return "email=test@example.com; password=Valid@123"
    if "tìm kiếm" in p or "search" in p:
        return "keyword='áo thun'; filter=size:M;color:Đen"
    if "tải lên" in p or "upload" in p:
        return "file=image.jpg (1.2MB, JPG)"
    return "Không có"

def _default_steps(prompt: str) -> str:
    p = prompt.lower()
    if "đăng nhập" in p or "login" in p:
        return "1) Mở trang Đăng nhập\n2) Nhập email & mật khẩu theo kịch bản\n3) Nhấn Đăng nhập\n4) Quan sát điều hướng/thông báo"
    if "tìm kiếm" in p or "search" in p:
        return "1) Mở trang có ô tìm kiếm\n2) Nhập từ khoá/điều kiện\n3) Thực thi tìm kiếm\n4) Quan sát danh sách kết quả"
    if "phân trang" in p or "pagination" in p:
        return "1) Mở danh sách có phân trang\n2) Chuyển trang (tiếp/trước/số cụ thể)\n3) Quan sát dữ liệu hiển thị\n4) Kiểm tra trạng thái nút"
    if "tải lên" in p or "upload" in p:
        return "1) Mở màn hình tải lên\n2) Chọn tệp theo kịch bản\n3) Thực hiện tải lên\n4) Quan sát thông báo/kết quả"
    return "1) Mở màn hình liên quan\n2) Thao tác theo kịch bản\n3) Quan sát hiển thị/kết quả\n4) Đối chiếu kỳ vọng"

def _default_expected(prompt: str) -> str:
    p = prompt.lower()
    if "đăng nhập" in p or "login" in p:
        return "Đăng nhập thành công hoặc bị từ chối với thông báo phù hợp."
    if "tìm kiếm" in p or "search" in p:
        return "Kết quả trả về đúng theo từ khoá/điều kiện, không lỗi."
    if "phân trang" in p or "pagination" in p:
        return "Trang hiển thị đúng dữ liệu, trạng thái nút phân trang chính xác."
    if "tải lên" in p or "upload" in p:
        return "Tệp hợp lệ được tải thành công; không hợp lệ bị chặn với thông báo rõ ràng."
    return "Hệ thống phản hồi đúng yêu cầu, không lỗi."

def _clean_block(s: str) -> str:
    s = _norm(s)
    s = re.sub(r"^\s*[-•*]\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _compose_target(desc: str, inputs: str, steps: str, exp: str) -> str:
    return (
        f"{H_DESC}\n{desc}\n\n"
        f"{H_INPUTS}\n{inputs}\n\n"
        f"{H_STEPS}\n{steps}\n\n"
        f"{H_EXP}\n{exp}"
    ).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Đường dẫn dataset4.xlsx")
    ap.add_argument("--sheet", default=0, help="Tên/index sheet")
    ap.add_argument("--out_dir", default="data/processed", help="Thư mục output")
    ap.add_argument("--train_name", default="train.jsonl")
    ap.add_argument("--corpus_name", default="corpus.csv")
    ap.add_argument("--debug_name", default="debug_view.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.input, sheet_name=args.sheet)
    cols = {k: _match_col(df.columns, v) for k, v in CANDIDATE_COLS.items()}
    for k, c in cols.items():
        df[f"__col_{k}"] = df[c] if c is not None else ""

    rows_train = []
    rows_corpus = []
    rows_debug = []

    for i, r in df.iterrows():
        prompt = _make_prompt(r, has_prompt_col=(cols["prompt"] is not None))
        desc   = _clean_block(r.get("__col_desc", ""))
        inputs = _clean_block(r.get("__col_inputs", ""))
        steps  = _clean_block(r.get("__col_steps", ""))
        exp    = _clean_block(r.get("__col_expected", ""))

        synthesized = {"desc":0,"inputs":0,"steps":0,"exp":0}
        if not desc:   desc   = _default_desc(prompt);   synthesized["desc"]=1
        if not inputs: inputs = _default_inputs(prompt); synthesized["inputs"]=1
        if not steps:  steps  = _default_steps(prompt);  synthesized["steps"]=1
        if not exp:    exp    = _default_expected(prompt); synthesized["exp"]=1

        target = _compose_target(desc, inputs, steps, exp)
        # Huấn luyện: text/target
        text = (
            "YÊU CẦU:\n"
            f"{prompt}\n\n"
            "Hãy tạo test case THEO ĐÚNG và CHỈ 4 mục (giữ đúng heading tiếng Việt, đúng thứ tự):\n"
            f"{H_DESC}\n{H_INPUTS}\n{H_STEPS}\n{H_EXP}"
        )

        rows_train.append({"text": text, "target": target})

        rows_corpus.append({
            "prompt": prompt,
            "description": desc,
            "inputs": inputs,
            "steps": steps,
            "expected_output": exp,
            "platform": _norm(r.get("__col_platform","")),
            "domain":   _norm(r.get("__col_domain","")),
            "feature":  _norm(r.get("__col_feature","")),
            "aspect":   _norm(r.get("__col_aspect","")),
            "scope":    _norm(r.get("__col_scope","")),
        })

        rows_debug.append({
            "row_id": i,
            "has_desc": int(bool(desc)),
            "has_inputs": int(bool(inputs)),
            "has_steps": int(bool(steps)),
            "has_expected": int(bool(exp)),
            "synth_desc": synthesized["desc"],
            "synth_inputs": synthesized["inputs"],
            "synth_steps": synthesized["steps"],
            "synth_expected": synthesized["exp"],
            "len_desc": len(desc),
            "len_inputs": len(inputs),
            "len_steps": len(steps),
            "len_expected": len(exp),
            "prompt_preview": (prompt[:120] + "…") if len(prompt) > 120 else prompt,
            "target_len": len(target),
        })

    # write files
    train_path  = out_dir / args.train_name
    corpus_path = out_dir / args.corpus_name
    debug_path  = out_dir / args.debug_name

    with open(train_path, "w", encoding="utf-8") as f:
        for row in rows_train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pd.DataFrame(rows_corpus).to_csv(corpus_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_debug).to_csv(debug_path, index=False, encoding="utf-8-sig")

    print("Done.")
    print(" -", train_path.resolve())
    print(" -", corpus_path.resolve())
    print(" -", debug_path.resolve())

if __name__ == "__main__":
    main()