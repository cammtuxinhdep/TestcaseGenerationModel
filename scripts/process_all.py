import argparse, re, os
from pathlib import Path
from typing import Optional
from src.format_dataset import process_test_cases

def norm(s: str) -> str:
    # chuẩn hoá để so khớp: lower + bỏ đuôi .csv + thay nhiều khoảng trắng/underscore bằng 1 dấu -
    s = s.lower().strip()
    s = re.sub(r"\.csv$", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    return s

def find_like(raw: Path, preferred_names) -> Optional[Path]:
    # Tìm file trong raw khớp gần đúng với 1 trong các tên gợi ý.
    if not raw.exists():
        return None
    cand_norm = [norm(x) for x in preferred_names]
    for p in raw.iterdir():
        if p.is_file():
            np = norm(p.name)
            for cn in cand_norm:
                # khớp chính xác hoặc bắt đầu bằng
                if np == cn or np.startswith(cn):
                    return p
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--translate", type=str, default="false")
    args = parser.parse_args()

    translate = args.translate.strip().lower() == "true"
    raw = Path(args.data_dir) / "raw"
    proc = Path(args.data_dir) / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    print("cwd =", os.getcwd())
    print("raw =", raw.resolve())
    print("raw exists?", raw.exists())
    if raw.exists():
        print("raw listing:", [p.name for p in raw.iterdir()])

    # Tự dò file CSV
    f1 = find_like(raw, ["Test_cases.csv", "Test cases.csv"])

    o1 = proc / "test_cases_vi.csv"

    if f1 and f1.exists():
        print(f"Processing: {f1.resolve()}")
        process_test_cases(f1, o1, translate=translate)
    else:
        print("WARNING: missing Test_cases.csv (hoặc tên gần đúng)")

    print(f"Done. Merged -> {o1.resolve()}")

if __name__ == "__main__":
    main()