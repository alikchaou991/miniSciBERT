# preprocessing.py
from pathlib import Path
import re
from loader import load_dataset_local

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_corpus(dataset_path: str, out_path: str) -> None:
    ds = load_dataset_local(dataset_path)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for ex in ds:
            title = ex.get("title", "") or ""
            abstract = ex.get("abstract", "") or ""
            text = f"{title} {abstract}".strip()

            if not text:
                continue

            norm = normalize_text(text)

            if len(norm.split()) < 3:
                continue

            fout.write(norm + "\n")