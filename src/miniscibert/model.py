import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniBertForMLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.embedding_layer_norm = nn.LayerNorm(config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_hidden_layers,
        )

        self.mlm_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlm_layer_norm = nn.LayerNorm(config.hidden_size)
        self.mlm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.mlm_decoder.bias = self.mlm_bias

        self.mlm_decoder.weight = self.token_embeddings.weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.embedding_layer_norm(x)
        x = self.embedding_dropout(x)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        x = self.mlm_dense(x)
        x = F.gelu(x)
        x = self.mlm_layer_norm(x)

        logits = self.mlm_decoder(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
        }