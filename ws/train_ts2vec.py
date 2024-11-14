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
from tqdm import tqdm
from utils import data_dropout, init_dl_program, name_with_datetime, pkl_save

from ts2vec import TS2Vec


def save_checkpoint_callback(save_every=1, unit="epoch"):
    assert unit in ("epoch", "iter")

    def callback(model, loss):
        n = model.n_epochs if unit == "epoch" else model.n_iters
        if n % save_every == 0:
            model.save(f"{run_dir}/model_{n}.pkl")

    return callback


def apply_highpass_filter(data, lowcut=0.05, sampling_rate=100, axis=-1):
    b, a = scipy.signal.butter(5, lowcut, btype="highpass", fs=sampling_rate)
    return scipy.signal.filtfilt(b, a, data, axis=axis)


# Data is of shape (n_batch, n_samples, n_channels)
def preprocess(data, sampling_rate=100):
    # Filter the data. Commonly done with bandpass filter from 0.05 Hz to 150 Hz, but here we only use highpass filter because of sampling rate
    data_filtered = apply_highpass_filter(
        data, lowcut=0.05, sampling_rate=sampling_rate, axis=1
    )
    # Normalize the data to have zero mean and unit variance along the time axis
    # data_normalized = (data_filtered - np.mean(data_filtered, axis=1)) / np.std(data_filtered, axis=1)
    return data_filtered


def load_raw_data(df, sampling_rate, path):
    data = [
        wfdb.rdsamp(path + f)
        for f in tqdm((df.filename_lr if sampling_rate == 100 else df.filename_hr))
    ]
    return np.array([signal for signal, meta in data])


def load_ptb_data(data_path="data/ptb-xl/"):
    sampling_rate = 100
    train_fold_size = 8
    val_fold_size = 1
    ptb_df = pd.read_csv(data_path + "ptbxl_database_translated.csv")
    ptb_df = ptb_df[0:100]
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
    parser.add_argument("dataset", help="The dataset name")
    parser.add_argument(
        "run_name",
        help="The folder name used to save model, output and evaluation metrics. This can be set to any word",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="The gpu no. used for training and inference (defaults to 0)",
    )
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
    parser.add_argument("--seed", type=int, default=None, help="The random seed")
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="The maximum allowed number of threads used by this process",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Whether to perform evaluation after training",
    )
    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    # device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    print("Loading data... ", end="")

    ptb_path = "data/ptb-xl/"
    train_data, val_data = load_ptb_data(data_path=ptb_path)

    print("done")

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
    )

    if args.save_every is not None:
        unit = "epoch" if args.epochs is not None else "iter"
        config[f"after_{unit}_callback"] = save_checkpoint_callback(
            args.save_every, unit
        )

    run_dir = "training/" + args.dataset + "__" + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    t = time.time()

    model = TS2Vec(input_dims=train_data.shape[-1], device="cpu", **config)
    loss_log = model.fit(
        train_data, val_data, n_epochs=args.epochs, n_iters=args.iters, verbose=True
    )
    model.save(f"{run_dir}/model.pkl")

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    print("Finished.")
