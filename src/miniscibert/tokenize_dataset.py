from datasets import load_from_disk
from transformers import BertTokenizerFast

def main():
    ds = load_from_disk("data/raw/research_paper2023")["train"]
    tokenizer = BertTokenizerFast.from_pretrained("models/my_tokenizer")

    def tokenize_batch(batch):
        texts = [
            f"{t} {a}".strip()
            for t, a in zip(batch["title"], batch["abstract"])
        ]
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized_ds = ds.map(tokenize_batch, batched=True)
    tokenized_ds.save_to_disk("data/tokenized/research_paper2023")

if __name__ == "__main__":
    main()