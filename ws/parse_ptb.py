import os
import pickle

import numpy as np
import pandas as pd
import scipy.signal
import torch
import wfdb
from tqdm import tqdm

from ts2vec import TS2Vec


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
        for f in (df.filename_lr if sampling_rate == 100 else df.filename_hr)
    ]
    return np.array([signal for signal, meta in data])


def load_encoder(path=None):
    encoder = TS2Vec(input_dims=12, device="cuda")
    if path is not None:
        assert os.path.exists(path), "Encoder model path does not exist"
        encoder.load(path)
    return encoder


# Load the encoder from file if supplied and the model exists
encoder = load_encoder(path=None)
sampling_rate = 100
# Number of samples to process at once
n_samples = 100
data_folder = "data/ptb-xl/"
out_folder = "data/ptb-xl/"

train_fold_size = 8
val_fold_size = 1
ptb_df = pd.read_csv(data_folder + "ptbxl_database_translated.csv")
# strat_fold goes from 1-10 and is used to split the data into train, validation and test sets
ptb_df_train = ptb_df[ptb_df.strat_fold <= train_fold_size]
ptb_df_val = ptb_df[
    (train_fold_size < ptb_df.strat_fold)
    & (ptb_df.strat_fold <= train_fold_size + val_fold_size)
]
ptb_df_test = ptb_df[train_fold_size + val_fold_size < ptb_df.strat_fold]

# Encodes the data row by row and saves the embeddings and reports in a pickle file

for out_name, dataset in [
    ("parsed_ptb_train.pkl", ptb_df_train),
    ("parsed_ptb_test.pkl", ptb_df_test),
    ("parsed_ptb_val.pkl", ptb_df_val),
]:
    embedding_report_list = []
    for i in tqdm(range(len(dataset))):
        # dataset_slice is a dataframe with n_batch rows
        dataset_slice = dataset[
            i : i + 1
        ]  # Get a single row as a dataframe instead of a series
        # ecg_leads_data is of shape (n_batch, n_samples, n_channels)
        ecg_leads_data = load_raw_data(dataset_slice, sampling_rate, data_folder)
        preprocessed_data = preprocess(ecg_leads_data)
        # Copy is a workaround for the preprocessing putting a negative stride on the nparray
        # encoded_data is of shape (n_batch, n_embed)
        encoded_data = encoder.encode(
            preprocessed_data.copy(), encoding_window="full_series"
        )
        embedding_report_list.append(
            {
                "embedding": encoded_data[0],
                "report": dataset_slice["report"].array[0],
                "ecg_id": dataset_slice["ecg_id"].array[0],
            }
        )
    with open(out_folder + out_name, "wb") as f:
        pickle.dump(embedding_report_list, f)


# This code is for processing the data in batches of n_samples but it's not that much faster

# for out_name, dataset in [("parsed_ptb_train.pkl", ptb_df_train), ("parsed_ptb_test.pkl", ptb_df_test)]:
#     embedding_report_list = []
#     for i in tqdm(range(len(dataset)//n_samples)):
#         dataset_slice = dataset[i*n_samples:min((i+1)*n_samples, len(dataset))]
#         data = load_raw_data(dataset_slice, sampling_rate, data_folder)
#         preprocessed_data = preprocess(data)
#         encoded_data = encoder.encode(preprocessed_data, encoding_window="full_series")
#         for i in range(n_samples):
#             embedding_report_list.append({"embedding": encoded_data[i], "report": dataset_slice.iloc[i].report})
#     with open(out_folder+out_name, "wb") as f:
#         pickle.dump(embedding_report_list, f)
