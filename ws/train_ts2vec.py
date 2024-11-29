import argparse
import datetime
import json
import os
import time

import config
import pandas as pd
import torch
from parse_ptb import load_raw_data, preprocess
from ts2vec import TS2Vec
from utils import data_dropout, init_dl_program, name_with_datetime, pkl_save
from ws.ail_parser import Parser


def load_ptb_data(data_path="data/ptb-xl/"):
    sampling_rate = 100
    train_fold_size = 8
    val_fold_size = 1
    ptb_df = pd.read_csv(data_path + "ptbxl_database_translated.csv")
    # For testing purposes uncomment the line below
    # ptb_df = ptb_df[0:100]
    # strat_fold goes from 1-10 and is used to split the data into train, validation and test sets
    ptb_df_train = ptb_df[ptb_df.strat_fold <= train_fold_size]
    ptb_df_val = ptb_df[
        (train_fold_size < ptb_df.strat_fold)
        & (ptb_df.strat_fold <= train_fold_size + val_fold_size)
    ]
    ptb_df_test = ptb_df[train_fold_size + val_fold_size < ptb_df.strat_fold]
    train_data = load_raw_data(ptb_df_train, sampling_rate, data_path)
    val_data = load_raw_data(ptb_df_val, sampling_rate, data_path)
    preprocessed_train_data = preprocess(train_data, sampling_rate)
    preprocessed_val_data = preprocess(val_data, sampling_rate)
    return preprocessed_train_data.copy(), preprocessed_val_data.copy()


def main(args):
    print("Arguments:", str(args))
    print("Loading data... ", end="")

    ptb_path = "data/ptb-xl/"
    train_data, val_data = load_ptb_data(data_path=ptb_path)

    print("Done")

    output_dir = "data/ts2vec"
    os.makedirs(output_dir, exist_ok=True)
    t = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TS2Vec(
        input_dim=train_data.shape[-1],
        depth=args.tsencoder_depth,
        hidden_dim=args.tsencoder_hidden_dim,
        batch_size=config.tsencoder_batch_size,
        device=device,
        lr=args.lr,
        output_dim=args.ts_embedding_dim,
        max_train_length=args.max_train_length,
    )

    # This saves the model on it's own too when it's done training
    model.fit(
        train_data,
        val_data,
        settled=args.settled,
        model_name=args.model_name,
        n_epochs=args.epochs,
        verbose=True,
    )
    # #model.save(f"{output_dir}/{args.model_name}.pkl")

    t = time.time() - t
    snapshot_path = f"data/ts2vec/{args.model_name}_snapshot.pt"
    snapshot = torch.load(snapshot_path, weights_only=True)
    snapshot["training_time"] = t
    torch.save(snapshot, snapshot_path)
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    print("Finished.")


def modify_parser(parser: Parser):
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        default="model",
        help="The name of the pkl file for the model without the extension (defaults to model)",
    )
    parser.add_argument(
        "--settled",
        type=int,
        default=3,
        help="number of epochs to wait before early stopping if no improvement",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="The learning rate (defaults to 0.001)"
    )
    parser.add_argument(
        "--max-train-length",
        type=int,
        default=3000,
        help="For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="The number of maximum epochs"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save the checkpoint every <save_every> iterations/epochs",
    )
    parser.add_argument(
        "--tsencoder_hidden_dim",
        type=int,
        default=config.tsencoder_hidden_dim,
        help="If not using config set the hidden dimension of the encoder",
    )
    parser.add_argument(
        "--tsencoder_depth",
        type=int,
        default=config.tsencoder_depth,
        help="If not using config set the depth of the encoder",
    )
    parser.add_argument(
        "--ts_embedding_dim",
        type=int,
        default=config.ts_embedding_dim,
        help="If not using config set the embedding dimension of the encoder",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modify_parser(parser)
    args = parser.parse_intermixed_args()
    main(args)
