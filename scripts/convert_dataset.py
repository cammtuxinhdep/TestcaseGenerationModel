import argparse
from pathlib import Path
import pandas as pd

def clean(s):
    if pd.isna(s):
        return ""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n").strip()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Đường dẫn .xlsx có 2 dòng header")
    ap.add_argument("--sheet", default=0, help="Tên hoặc index sheet (mặc định 0)")
    ap.add_argument("--out_csv", default="data/processed/merged_prompt_completion.csv")
    ap.add_argument("--out_jsonl", default=None)
    args = ap.parse_args()

    # 1) Đọc 2 dòng header → MultiIndex columns
    df = pd.read_excel(args.input, header=[0, 1], sheet_name=args.sheet)

    # 2) Xác định các cột
    #    - prompt: level-0 == "prompt" (lấy cột đầu tiên thuộc nhóm này)
    #    - completion: 3 cột con dưới nhóm "completion"
    prompt_cols = [c for c in df.columns if str(c[0]).strip().lower() == "prompt"]
    if not prompt_cols:
        raise RuntimeError("Không tìm thấy cột level-0 = 'prompt'")
    prompt_col = prompt_cols[0]  # (level0='prompt', level1='...')

    comp_cols = [c for c in df.columns if str(c[0]).strip().lower() == "completion"]
    if len(comp_cols) < 3:
        # vẫn cho chạy nếu ít hơn 3, sẽ ghép những gì có
        pass

    # Ưu tiên đúng thứ tự mô tả/steps/expected nếu có
    def pick(label_keywords):
        for c in comp_cols:
            lvl1 = str(c[1]).lower()
            if any(k in lvl1 for k in label_keywords):
                return c
        return None

    desc_col = pick(["mô tả", "description"])
    steps_col = pick(["bước", "steps", "procedure"])
    exp_col   = pick(["đầu ra", "expected", "output"])

    # Fallback: nếu chưa tìm đủ, chọn theo vị trí
    ordered = []
    if desc_col: ordered.append(desc_col)
    if steps_col: ordered.append(steps_col)
    if exp_col:   ordered.append(exp_col)
    for c in comp_cols:
        if c not in ordered:
            ordered.append(c)
    # Lấy tối đa 3 cột đầu tiên cho completion
    ordered = ordered[:3]

    # 3) Xây prompt & completion
    def build_completion(row):
        parts = []
        for c in ordered:
            title = str(c[1]).strip()
            parts.append(f"{title}: {clean(row[c])}")
        return "\n".join(parts).strip()

    out = pd.DataFrame({
        "prompt": df[prompt_col].map(clean),
        "completion": df.apply(build_completion, axis=1)
    })

    # 4) Lọc rỗng / trùng
    out = out[(out["prompt"] != "") & (out["completion"] != "")]
    out = out.drop_duplicates(subset=["prompt", "completion"]).reset_index(drop=True)

    # 5) Lưu
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"Saved CSV -> {Path(args.out_csv).resolve()}  | rows: {len(out)}")

    if args.out_jsonl:
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for _, r in out.iterrows():
                f.write(
                    f'{{"prompt": {r["prompt"]!r}, "completion": {r["completion"]!r}}}\n'
                )
        print(f"Saved JSONL -> {Path(args.out_jsonl).resolve()}")

if __name__ == "__main__":
    main()
