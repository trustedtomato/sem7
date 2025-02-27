import argparse
import os

import config
import pandas as pd
import torch
import torch.nn.functional as nnf
from ail_parser import Parser, parse_intermixed_args
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_mapping import PTBXLEncodedDataset, TsCaptionModel, TsCaptionPrefix
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from TSCapMetrics import TSCapMetrics
from typing_extensions import override
from utils import pkl_load, pkl_save

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

            if tokens is not None:
                output_list = list(tokens.squeeze().view(-1).cpu().numpy())
                output_text = tokenizer.decode(output_list)
                generated_list.append(output_text)

    return generated_list[0]


def get_test_loss(model, dataloader, device):
    model.eval()
    torch.set_grad_enabled(False)
    sum_test_loss = 0
    for idx, (tokens, mask, prefix) in enumerate(dataloader):
        tokens, mask, prefix = (
            tokens.to(device),
            mask.to(device),
            prefix.to(device, dtype=torch.float32),
        )
        logits = model(tokens, prefix, mask)
        loss = nnf.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens.flatten(),
            ignore_index=0,
        )
        sum_test_loss += loss.item()
    return sum_test_loss / len(dataloader)


def main(args):
    # setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    data_path = args.test_path
    dataset = PTBXLEncodedDataset(data_path, args.prefix_length)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    snapshot_path = args.snapshot_path
    model = TsCaptionModel(
        args.prefix_length,
        ts_embedding_length=args.ts_embedding_length,
        ts_embedding_dim=dataset.embedding_dim,
        num_layers=args.mapper_num_layers,
    )
    state = torch.load(
        snapshot_path, map_location=torch.device("cpu"), weights_only=True
    )["state"]
    # remove module. prefix
    state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.eval()
    model = model.to(device)

    hypotheses = []
    references = []
    print("Generating predictions...")
    test_loss = get_test_loss(model, test_dataloader, device)
    for _ in range(args.n_iter):
        hyp = []
        ref = []
        for idx, (tokens, mask, prefix) in tqdm(enumerate(test_dataloader)):
            tokens = tokens.to(device)
            mask = mask.to(device)
            prefix = prefix.to(device)
            with torch.no_grad():
                prefix_embed = model.clip_project(prefix).reshape(
                    1, args.prefix_length, -1
                )
            x = generate2(model, tokenizer, embed=prefix_embed)
            hyp.append(x.replace("\n", ""))
            ref.append(
                tokenizer.decode(tokens.squeeze().cpu().numpy())
                .rstrip("!")
                .replace("\n", "")
            )
            # if idx % 100 == 0:
            #     print("\ngener", x)
            #     print(
            #         "truth",
            #         tokenizer.decode(tokens.squeeze().cpu().numpy()).rstrip("!"),
            #     )
        hypotheses.append(hyp)
        if len(references) == 0:
            references = ref
    if args.save_examples:
        hyp = hypotheses[0]
        hyp_ref_zip = list(zip(hyp, references))
        df = pd.DataFrame(hyp_ref_zip, columns=["Hyp", "Ref"])
        df.to_csv(f"./{args.out_dir}/{args.out_name}.csv")

    if args.metrics:
        ts_cap_metrics = TSCapMetrics()
        ts_cap_metrics.calculate_metrics(
            references,
            hypotheses,
            test_loss,
            out_dir=args.out_dir,
            out_name=f"{args.out_name}.csv",
        )


def modify_parser(parser: Parser):
    parser.add_argument(
        "--test_path",
        required=True,
        type=str,
        help="The path to the test dataset",
    )
    parser.add_argument(
        "--snapshot_path",
        required=True,
        type=str,
        help="The path to the snapshot file",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="The path to the output folder",
    )
    parser.add_argument(
        "--out_name",
        required=True,
        type=str,
        help="The name of the output file without extension",
    )
    parser.add_argument(
        "--n_iter",
        required=True,
        type=int,
        help="The number of iterations for metrics",
    )
    parser.add_argument(
        "--metrics",
        dest="metrics",
        action="store_true",
    )
    parser.add_argument(
        "--save_examples",
        dest="save_examples",
        action="store_true",
    )

    parser.add_argument(
        "--ts_embedding_length", type=int, default=config.ts_embedding_length
    )
    parser.add_argument(
        "--mapper_num_layers", type=int, default=config.mapper_num_layers
    )
    parser.add_argument("--prefix_length", type=int, default=config.prefix_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modify_parser(parser)
    args = parser.parse_intermixed_args()
    main(args)
