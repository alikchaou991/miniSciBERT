from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from miniscibert.preprocessing import preprocess_corpus

if __name__ == "__main__":
    preprocess_corpus(
        dataset_path="data/raw/research_paper2023",
        out_path="data/processed/abstracts.txt"
    )