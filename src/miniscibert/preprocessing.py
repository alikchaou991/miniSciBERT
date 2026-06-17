from pathlib import Path
import re
from datasets import load_from_disk

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_corpus(dataset_path: str, out_path: str) -> None:
    ds = load_from_disk(dataset_path)
    train_ds = ds["train"]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for ex in train_ds:
            title = ex.get("title", "") or ""
            abstract = ex.get("abstract", "") or ""
            text = f"{title} {abstract}".strip()

            if not text:
                continue

            norm = normalize_text(text)

            if len(norm.split()) < 3:
                continue

            fout.write(norm + "\n")
            count += 1

    print(f"Saved {count} lines to {out_path}")