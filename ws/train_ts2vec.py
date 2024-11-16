import argparse
import datetime
import os
import sys
import time

import datautils
import numpy as np
import pandas as pd
import scipy.signal
import tasks
import torch
import wfdb
from parse_ptb import load_raw_data, preprocess
from tqdm import tqdm
from ts2vec import TS2Vec
from utils import data_dropout, init_dl_program, name_with_datetime, pkl_save


def load_ptb_data(data_path="data/ptb-xl/"):
    sampling_rate = 100
    train_fold_size = 8
    val_fold_size = 1
    ptb_df = pd.read_csv(data_path + "ptbxl_database_translated.csv")
    # For testing purposes uncomment the line below
    #ptb_df = ptb_df[0:100]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", type=int, default=8, help="The batch size (defaults to 8)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="The learning rate (defaults to 0.001)"
    )
    parser.add_argument(
        "--repr-dims",
        type=int,
        default=320,
        help="The representation dimension (defaults to 320)",
    )
    parser.add_argument(
        "--max-train-length",
        type=int,
        default=3000,
        help="For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)",
    )
    parser.add_argument(
        "--iters", type=int, default=None, help="The number of iterations"
    )
    parser.add_argument("--epochs", type=int, default=None, help="The number of epochs")
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save the checkpoint every <save_every> iterations/epochs",
    )
    args = parser.parse_args()

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
        input_dims=train_data.shape[-1],
        device=device,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
    )
    loss_log = model.fit(
        train_data, val_data, n_epochs=args.epochs, n_iters=args.iters, verbose=True
    )
    model.save(f"{output_dir}/model.pkl")

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    print("Finished.")
