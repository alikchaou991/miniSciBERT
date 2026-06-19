from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathsConfig:
    corpus_file: Path = Path("data/processed/abstracts.txt")
    tokenizer_dir: Path = Path("models/my_tokenizer")
    vocab_file: Path = Path("models/my_tokenizer/vocab.txt")
    tokenizer_file: Path = Path("models/my_tokenizer/tokenizer.json")
    output_dir: Path = Path("outputs/pretrain_baseline")
    checkpoint_dir: Path = Path("outputs/pretrain_baseline/checkpoints")


@dataclass
class ModelConfig:
    vocab_size: int = 30000
    max_position_embeddings: int = 128
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    mlm_probability: float = 0.15
    train_val_split: float = 0.9
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda"


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def get_config() -> Config:
    cfg = Config()
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return cfg