# scripts/preprocess_data.py
from miniscibert.preprocessing import preprocess_corpus

if __name__ == "__main__":
    preprocess_corpus(
        raw_dir="data/raw",
        out_path="data/clean/corpus.txt"
    )