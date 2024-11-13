import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import pickle
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.optimization import get_linear_schedule_with_warmup
import os
from tqdm import tqdm


class PTBXLEncodedDataset(Dataset):

    def __len__(self) -> int:
        return len(self.report_tokens)

    def pad_tokens(self, index: int):
        tokens = self.report_tokens[index]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.report_tokens[index] = tokens
        elif padding < 0:
            tokens = tokens[: self.max_seq_len]
            self.report_tokens[index] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat(
            (torch.ones(self.prefix_length), mask), dim=0
        )  # adding prefix mask
        return tokens, mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(index)
        encoder_embedding = self.encoder_embeddings[index]
        if self.normalize_prefix:
            encoder_embedding = encoder_embedding.float()
            encoder_embedding = encoder_embedding / encoder_embedding.norm(2, -1)
        return tokens, mask, encoder_embedding

    def __init__(
        self,
        data_path: str,
        prefix_length: int,
        gpt2_type: str = "gpt2",
        normalize_prefix=False,
    ):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, "rb") as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data))
        self.encoder_embeddings = []
        self.ecg_ids = []
        self.report_tokens = []

        for data in tqdm(all_data):
            self.encoder_embeddings.append(
                torch.tensor(data["embedding"], dtype=torch.float32)
            )
            self.ecg_ids.append(data["ecg_id"])
            report_tokenized = torch.tensor(
                self.tokenizer.encode(data["report"]), dtype=torch.int64
            )
            self.report_tokens.append(report_tokenized)

        self.max_seq_len = max([len(tokens) for tokens in self.report_tokens])


path = "data/ptb-xl/parsed_ptb_train.pkl"

dataset = PTBXLEncodedDataset(
    data_path=path, prefix_length=10, gpt2_type="gpt2", normalize_prefix=False
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
for i, (tokens, mask, encoder_embedding) in tqdm(enumerate(dataloader)):
    pass
