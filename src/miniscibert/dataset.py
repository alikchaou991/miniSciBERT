from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from tokenizers import BertWordPieceTokenizer


class MLMTextDataset(Dataset):
    def __init__(
        self,
        corpus_file: str | Path,
        vocab_file: str | Path,
        block_size: int = 128,
        lowercase: bool = True,
    ):
        self.corpus_file = Path(corpus_file)
        self.vocab_file = Path(vocab_file)
        self.block_size = block_size

        if not self.corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_file}")
        if not self.vocab_file.exists():
            raise FileNotFoundError(f"Vocab file not found: {self.vocab_file}")

        self.tokenizer = BertWordPieceTokenizer(str(self.vocab_file), lowercase=lowercase)

        text = self.corpus_file.read_text(encoding="utf-8")
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids

        usable_length = (len(token_ids) // block_size) * block_size
        token_ids = token_ids[:usable_length]

        self.examples: List[torch.Tensor] = [
            torch.tensor(token_ids[i:i + block_size], dtype=torch.long)
            for i in range(0, len(token_ids), block_size)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return {"input_ids": self.examples[idx]}