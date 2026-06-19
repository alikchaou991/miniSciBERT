import torch


class MLMCollator:
    def __init__(self, tokenizer, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.unk_token_id = tokenizer.token_to_id("[UNK]")
        self.cls_token_id = tokenizer.token_to_id("[CLS]")
        self.sep_token_id = tokenizer.token_to_id("[SEP]")
        self.mask_token_id = tokenizer.token_to_id("[MASK]")

        self.special_token_ids = {
            self.pad_token_id,
            self.unk_token_id,
            self.cls_token_id,
            self.sep_token_id,
            self.mask_token_id,
        }

        self.vocab_size = tokenizer.get_vocab_size()

    def __call__(self, examples):
        input_ids = torch.stack([example["input_ids"] for example in examples])
        labels = input_ids.clone()

        attention_mask = torch.ones_like(input_ids)

        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.special_token_ids:
            special_tokens_mask |= input_ids.eq(token_id)

        probability_matrix.masked_fill_(special_tokens_mask, 0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            low=0,
            high=self.vocab_size,
            size=labels.shape,
            dtype=torch.long,
        )
        input_ids[indices_random] = random_words[indices_random]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }