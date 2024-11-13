from json import encoder
from typing import Optional

import PIL.Image
from torch import nn
import torch
import torch.nn.functional as nnf
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from typing_extensions import override

from parse_ptb import load_encoder, preprocess
from train_mapping import ClipCaptionModel, ClipCaptionPrefix, PTBXLEncodedDataset
from torch.utils.data import DataLoader, Dataset
from utils import pkl_load

T = torch.Tensor


def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().view(-1).cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


def main():
    # setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prefix_length = 10
    clip_length = 10
    batch_size = 40

    weights_path = "results/checkpoints/"
    model = ClipCaptionModel(prefix_length, clip_length)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    model = model.eval()
    model = model.to(device)

    data_path = "data/ptb-xl/parsed_ptb_test.pkl"
    dataset = PTBXLEncodedDataset(data_path, prefix_length)
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
        tokens = tokens.to(device)
        mask = mask.to(device)
        prefix = prefix.to(device)
        with torch.no_grad():
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        # do we actually need this generate2 function?
        x = generate2(model, tokenizer, embed=prefix_embed)
        print(x)


if __name__ == "__main__":
    main()
