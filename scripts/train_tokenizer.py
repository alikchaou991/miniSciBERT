from pathlib import Path
from tokenizers import BertWordPieceTokenizer

def main():
    corpus_file = "data/processed/abstracts.txt"   # change if your file has another name
    save_dir = Path("models/my_tokenizer")
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertWordPieceTokenizer(lowercase=True)
    tokenizer.train(
        files=[corpus_file],
        vocab_size=30000,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    tokenizer.save_model(str(save_dir))
    tokenizer.save(str(save_dir / "tokenizer.json"))

if __name__ == "__main__":
    main()