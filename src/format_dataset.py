import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

try:
    from googletrans import Translator
except ImportError:
    Translator = None

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip())

def _maybe_translate(texts, do_translate: bool):
    if not do_translate or Translator is None:
        return texts
    translator = Translator()
    result = []
    for t in tqdm(texts, desc="Translating"):
        try:
            result.append(translator.translate(t, src="en", dest="vi").text)
        except:
            result.append(t)
    return result

def process_test_cases(input_csv: Path, output_csv: Path, translate=False):
    df = pd.read_csv(input_csv)
    out = pd.DataFrame({
        "prompt": df.iloc[:,0].map(_clean_text),
        "completion": df.iloc[:,1].map(_clean_text)
    })
    if translate:
        out["prompt"] = _maybe_translate(out["prompt"], True)
        out["completion"] = _maybe_translate(out["completion"], True)
    out.to_csv(output_csv, index=False)
    return out
