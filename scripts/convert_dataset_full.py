import argparse
from pathlib import Path
import pandas as pd
import json
import re

# ---------- Tiện ích ----------
def norm(s):
    return "" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s).strip()

def clean_text(s):
    s = norm(s)
    # Chuẩn hóa xuống dòng
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    return s

def is_unnamed(x):
    return x is None or (isinstance(x, str) and x.lower().startswith("unnamed"))

def to_pair(col):
    """Trả về (lvl0, lvl1) dạng chuỗi thường (lower) để so khớp dễ."""
    if isinstance(col, tuple) and len(col) >= 2:
        return norm(col[0]).lower(), norm(col[1]).lower()
    # Trường hợp chỉ 1 level
    return norm(col).lower(), ""

def pick_completion_cols(all_cols):
    """
    Từ danh sách cột thuộc nhóm 'completion', chọn theo thứ tự:
    - mô tả/description
    - bước/steps
    - expected/output
    Fallback: lấy tối đa 3 cột đầu còn lại.
    """
    def find_any(keys):
        for c in all_cols:
            _, l1 = to_pair(c)
            if any(k in l1 for k in keys):
                return c
        return None

    desc = find_any(["mô tả", "description", "mo ta"])
    steps = find_any(["bước", "steps", "procedure", "quy trình"])
    exp   = find_any(["đầu ra", "expected", "output", "kỳ vọng"])

    ordered = []
    for c in (desc, steps, exp):
        if c and c not in ordered:
            ordered.append(c)
    for c in all_cols:
        if c not in ordered:
            ordered.append(c)
    return ordered[:3]

def safe_get(row, col):
    try:
        return clean_text(row[col])
    except Exception:
        return ""

def first_nonempty(series):
    # chọn giá trị đầu tiên khác rỗng (khi prompt có nhiều sub-col)
    for v in series:
        v = clean_text(v)
        if v:
            return v
    return ""

# ---------- Chuyển đổi ----------
def convert_excel(
    xlsx_path: str,
    sheet=0,
    out_csv="data/processed/merged_prompt_completion.csv",
    out_jsonl=None,
    include_tags=True,
    keep_duplicates=True,
    add_instr=False,
):
    """
    include_tags: đưa 5 cột tag vào output.
    keep_duplicates: True -> giữ nguyên trùng lặp; False -> drop_duplicates.
    add_instr: nếu True, tạo thêm cột 'input_text' cho kiểu T5 (ghép prompt + tags vào input).
    """
    # Đọc 2 dòng header
    df = pd.read_excel(xlsx_path, sheet_name=sheet, header=[0, 1])

    # Loại cột 'Unnamed'
    df = df.loc[:, [not (is_unnamed(c[0]) and is_unnamed(c[1])) for c in df.columns]]

    # Tìm nhóm cột
    prompt_cols = [c for c in df.columns if to_pair(c)[0] == "prompt"]
    comp_cols   = [c for c in df.columns if to_pair(c)[0] == "completion"]

    if not prompt_cols:
        raise RuntimeError("Không tìm thấy nhóm cột level-0 = 'prompt'.")
    if not comp_cols:
        raise RuntimeError("Không tìm thấy nhóm cột level-0 = 'completion'.")

    # Cột completion (tối đa 3)
    comp_sel = pick_completion_cols(comp_cols)

    # Nhóm tag (nếu có)
    tag_names = ["platform", "domain", "feature", "aspect", "scope"]
    tag_cols = {}
    for name in tag_names:
        cand = [c for c in df.columns if to_pair(c)[0] == name]
        tag_cols[name] = cand[0] if cand else None  # có thể None

    # Build từng dòng
    records = []
    for _, row in df.iterrows():
        # prompt có thể là nhiều sub-cột -> lấy cái đầu tiên có nội dung
        p_vals = [safe_get(row, c) for c in prompt_cols]
        prompt_text = first_nonempty(p_vals)

        # completion = ghép 1-3 cột theo nhãn con
        parts = []
        for c in comp_sel:
            _, l1 = to_pair(c)
            title = norm(c[1]) if isinstance(c, tuple) and len(c) > 1 else "Completion"
            val = safe_get(row, c)
            if val:
                # chuẩn hoá label tiếng Việt hay gặp
                lab = title
                lab_low = l1
                if "mô tả" in lab_low or "description" in lab_low:
                    lab = "Mô tả (description)"
                elif "bước" in lab_low or "steps" in lab_low:
                    lab = "Các bước chính (main steps)"
                elif "đầu ra" in lab_low or "expected" in lab_low or "output" in lab_low:
                    lab = "Đầu ra mong muốn (expected output)"
                parts.append(f"{lab}: {val}")
        completion_text = "\n".join(parts).strip()

        # tags
        tags = {}
        if include_tags:
            for name in tag_names:
                c = tag_cols.get(name)
                tags[name] = safe_get(row, c) if c is not None else ""

        rec = {"prompt": prompt_text, "completion": completion_text}
        if include_tags:
            rec.update(tags)

        # input_text (style T5) – hữu ích nếu bạn muốn cho model biết tags
        if add_instr:
            tag_str = " | ".join([f"{k}: {v}" for k, v in tags.items() if v])
            prefix = f"[NGỮ CẢNH] {tag_str}\n" if tag_str else ""
            rec["input_text"] = f"{prefix}{prompt_text}"
            rec["target_text"] = completion_text

        records.append(rec)

    out_df = pd.DataFrame(records)

    # Lọc rỗng
    out_df = out_df[(out_df["prompt"] != "") & (out_df["completion"] != "")]
    # Lọc trùng (tuỳ chọn)
    if not keep_duplicates:
        dedup_keys = ["prompt", "completion"] + ([*tag_names] if include_tags else [])
        out_df = out_df.drop_duplicates(subset=dedup_keys).reset_index(drop=True)

    # Lưu
    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv_path, index=False, encoding="utf-8")
    print(f"[OK] Saved CSV -> {out_csv_path.resolve()}  | rows: {len(out_df)}")

    if out_jsonl:
        out_jsonl_path = Path(out_jsonl)
        out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                # Chỉ ghi các key có giá trị
                obj = {k: v for k, v in r.items() if v != ""}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[OK] Saved JSONL -> {out_jsonl_path.resolve()}")

    return out_df


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Chuyển Excel 2 header (prompt/completion/tags) -> CSV & JSONL cho huấn luyện"
    )
    ap.add_argument("--input", required=True, help="Đường dẫn .xlsx (2 dòng header).")
    ap.add_argument("--sheet", default=0, help="Tên hoặc index sheet (mặc định 0).")
    ap.add_argument("--out_csv", default="data/processed/merged_prompt_completion.csv")
    ap.add_argument("--out_jsonl", default="data/processed/merged_prompt_completion.jsonl")
    ap.add_argument("--no_tags", action="store_true", help="Không xuất 5 cột tag vào output.")
    ap.add_argument("--drop_dups", action="store_true", help="Loại trùng lặp.")
    ap.add_argument("--add_instr", action="store_true", help="Thêm input_text/target_text (style T5).")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_excel(
        xlsx_path=args.input,
        sheet=args.sheet,
        out_csv=args.out_csv,
        out_jsonl=args.out_jsonl,
        include_tags=not args.no_tags,
        keep_duplicates=not args.drop_dups,
        add_instr=args.add_instr,
    )
