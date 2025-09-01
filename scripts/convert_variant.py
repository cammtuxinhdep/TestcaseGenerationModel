import argparse
from pathlib import Path
import pandas as pd

def clean(s):
    if pd.isna(s):
        return ""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n").strip()
    return s

def main():
    # Thiết lập argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Đường dẫn file .xlsx chứa 3 cột: prompt, variant_1, variant_2")
    ap.add_argument("--sheet", default=0, help="Tên hoặc index sheet (mặc định 0)")
    ap.add_argument("--out_csv", default="data/processed/prompt_variants.csv", help="Đường dẫn file CSV đầu ra")
    args = ap.parse_args()

    # 1) Đọc file Excel (1 dòng header)
    df = pd.read_excel(args.input, sheet_name=args.sheet)

    # 2) Kiểm tra các cột
    expected_cols = ["prompt", "variant_1", "variant_2"]
    if not all(col in df.columns for col in expected_cols):
        missing = [col for col in expected_cols if col not in df.columns]
        raise RuntimeError(f"Không tìm thấy cột: {missing}")

    # 3) Làm sạch dữ liệu
    out = pd.DataFrame({
        "prompt": df["prompt"].map(clean),
        "variant_1": df["variant_1"].map(clean),
        "variant_2": df["variant_2"].map(clean)
    })

    # 4) Lọc dòng rỗng và trùng lặp
    out = out[(out["prompt"] != "") & (out["variant_1"] != "") & (out["variant_2"] != "")]
    out = out.drop_duplicates(subset=["prompt", "variant_1", "variant_2"]).reset_index(drop=True)

    # 5) Lưu file CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"Saved CSV -> {Path(args.out_csv).resolve()} | rows: {len(out)}")

if __name__ == "__main__":
    main()