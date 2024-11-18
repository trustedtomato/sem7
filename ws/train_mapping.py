import argparse
import os
from math import isnan
from typing import Optional, Tuple

import config
import torch
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as nnf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from utils import Logger, pkl_load, pkl_save


def ddp_setup() -> int:
    init_process_group(backend="nccl")
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    return device


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
        gpt2_type: str = "gpt2-medium",
        normalize_prefix=False,
    ):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        all_data = pkl_load(data_path)
        print("Data size is %0d" % len(all_data))
        self.encoder_embeddings = []
        self.ecg_ids = []
        self.report_tokens = []

        for data in tqdm(all_data):
            # filter out nan reports
            if type(data["report"]) == float:
                continue

            self.encoder_embeddings.append(
                torch.tensor(data["embedding"], dtype=torch.float32)
            )
            self.ecg_ids.append(data["ecg_id"])
            report_tokenized = torch.tensor(
                self.tokenizer.encode(data["report"]), dtype=torch.int64
            )
            self.report_tokens.append(report_tokenized)

        self.max_seq_len = max([len(tokens) for tokens in self.report_tokens])


class MlpTransformer(nn.Module):
    def __init__(
        self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.0
    ):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim**-0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(
            b, m, 2, self.num_heads, c // self.num_heads
        )
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum("bnhd,bmhd->bnmh", queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum("bnmh,bmhd->bnhd", attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):
    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(
        self,
        dim_self,
        dim_ref,
        num_heads,
        mlp_ratio=4.0,
        bias=False,
        dropout=0.0,
        act=nnf.relu,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(
            dim_self, dim_ref, num_heads, bias=bias, dropout=dropout
        )
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(
            dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout
        )


class Transformer(nn.Module):
    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:  # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(
        self,
        dim_self: int,
        num_heads: int,
        num_layers: int,
        dim_ref: Optional[int] = None,
        mlp_ratio: float = 2.0,
        act=nnf.relu,
        norm_layer: nn.Module = nn.LayerNorm,
        enc_dec: bool = False,
    ):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(
                    TransformerLayer(
                        dim_self,
                        dim_ref,
                        num_heads,
                        mlp_ratio,
                        act=act,
                        norm_layer=norm_layer,
                    )
                )
            elif enc_dec:  # self
                layers.append(
                    TransformerLayer(
                        dim_self,
                        dim_self,
                        num_heads,
                        mlp_ratio,
                        act=act,
                        norm_layer=norm_layer,
                    )
                )
            else:  # self or cross
                layers.append(
                    TransformerLayer(
                        dim_self,
                        dim_ref,
                        num_heads,
                        mlp_ratio,
                        act=act,
                        norm_layer=norm_layer,
                    )
                )
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):
    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        # print("x shape: ", x.shape)
        prefix = self.prefix_const.unsqueeze(0)
        # print("prefix shape: ", prefix.shape)
        prefix = prefix.expand(x.shape[0], *self.prefix_const.shape)
        # print("prefix shape after expand: ", prefix.shape)
        prefix = torch.cat((x, prefix), dim=1)
        # print("prefix shape after cat: ", prefix.shape)
        out = self.transformer(prefix)
        # print("out shape: ", out.shape)
        out = out[:, self.clip_length :]
        # print("out shape after slicing: ", out.shape)
        return out

    def __init__(
        self,
        dim_clip: int,
        dim_embedding: int,
        prefix_length: int,
        clip_length: int,
        num_layers: int = 8,
    ):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(
            torch.randn(prefix_length, dim_embedding), requires_grad=True
        )


class ClipCaptionModel(nn.Module):
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self,
        tokens: torch.Tensor,
        prefix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(
        self,
        prefix_length: int,
        clip_length: int,
        prefix_size: int = 320,
        num_layers: int = 8,
        gpt2_type: str = "gpt2-medium"
    ):
        super().__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt2_type)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = TransformerMapper(
            prefix_size, self.gpt_embedding_size, prefix_length, clip_length, num_layers
        )


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.gpt.eval()
        return self


def train(
    dataset: PTBXLEncodedDataset,
    model: nn.Module,
    device: int,
    args,
    lr: float = 2e-5,
    warmup_steps: int = 5000,
):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    snapshot_path = os.path.join(args.out_dir, f"{args.prefix}_snapshot.pt")

    print(f"Using device {device}")
    model = model.to(device)
    model = DDP(model, device_ids=[device])
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    sampler = DistributedSampler(dataset)
    train_dataloader = DataLoader(
        dataset, batch_size=args.bs, shuffle=False, drop_last=True, sampler=sampler
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.epochs * len(train_dataloader),
    )

    # define losses type
    def save_snapshot(epoch: int, losses: list[float] = []):
        snapshot = {
            "state": model.state_dict(),
            "current_epoch": epoch,
            "losses": losses,
        }
        torch.save(snapshot, snapshot_path)

    # Load snapshot if exists
    start_epoch = 0
    losses: list[float] = []
    if os.path.exists(snapshot_path):
        snapshot = torch.load(snapshot_path, weights_only=True)
        model.load_state_dict(snapshot["state"])
        start_epoch = snapshot["current_epoch"] + 1
        losses = snapshot["losses"]

    is_master = device == 0

    for epoch in range(start_epoch, args.epochs):
        print(f">>> Training epoch {epoch} on device {device}")
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = (
                tokens.to(device),
                mask.to(device),
                prefix.to(device, dtype=torch.float32),
            )
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1 : -1]
            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens.flatten(),
                ignore_index=0,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if is_master:
                progress = f"{str(idx).rjust(len(str(len(train_dataloader))))}/{len(train_dataloader)}"
                print(f"Epoch {epoch}, {progress}, loss: {loss.item()}")
                losses.append(loss.item())
        if is_master:
            print("Saving snapshot")
            save_snapshot(epoch)

    return model


def main(args):
    device = ddp_setup()

    dataset = PTBXLEncodedDataset(
        data_path=args.data,
        prefix_length=args.prefix_length,
        gpt2_type="openai-community/gpt2-medium",
        normalize_prefix=args.normalize_prefix,
    )

    prefix_dim = 320
    if args.only_prefix:
        model = ClipCaptionPrefix(
            args.prefix_length,
            clip_length=args.prefix_length_clip,
            prefix_size=prefix_dim,
            num_layers=args.num_layers,
            gpt2_type="openai-community/gpt2-medium",
        )
        print("Train only prefix")
    else:
        model = ClipCaptionModel(
            args.prefix_length,
            clip_length=args.prefix_length_clip,
            prefix_size=prefix_dim,
            num_layers=args.num_layers,
            gpt2_type="openai-community/gpt2-medium",
        )
        print("Train both prefix and GPT")

    train(dataset, model, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data/ptb-xl/parsed_ptb_train.pkl")
    parser.add_argument("--out_dir", default="./data/tscap")
    parser.add_argument("--prefix", default="tscap", help="prefix for saved filenames")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--prefix_length", type=int, default=config.prefix_length)
    parser.add_argument(
        "--prefix_length_clip", type=int, default=config.ts_embedding_length
    )
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--only_prefix", dest="only_prefix", action="store_true")
    parser.add_argument("--num_layers", type=int, default=config.num_layers)
    parser.add_argument(
        "--normalize_prefix", dest="normalize_prefix", action="store_true"
    )
    args = parser.parse_args()
    main(args)
