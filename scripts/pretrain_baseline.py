from pathlib import Path
import math
import random

import torch
from torch.utils.data import DataLoader, random_split
from tokenizers import BertWordPieceTokenizer

from src.miniscibert.config import get_config
from src.miniscibert.dataset import MLMTextDataset
from src.miniscibert.collator import MLMCollator
from src.miniscibert.model import MiniBertForMLM


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_loss += outputs["loss"].item()
            total_batches += 1

    if total_batches == 0:
        return None

    return total_loss / total_batches


def main():
    cfg = get_config()
    set_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertWordPieceTokenizer(
        str(cfg.paths.vocab_file),
        lowercase=True,
    )

    cfg.model.vocab_size = tokenizer.get_vocab_size()

    dataset = MLMTextDataset(
        corpus_file=cfg.paths.corpus_file,
        vocab_file=cfg.paths.vocab_file,
        block_size=cfg.model.max_position_embeddings,
        lowercase=True,
    )

    train_size = int(len(dataset) * cfg.train.train_val_split)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    collator = MLMCollator(
        tokenizer=tokenizer,
        mlm_probability=cfg.train.mlm_probability,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    model = MiniBertForMLM(cfg.model).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    global_step = 0
    best_val_loss = float("inf")

    model.train()
    while global_step < cfg.train.max_steps:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in 