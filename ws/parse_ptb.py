import os
import pickle
import time

import numpy as np
import pandas as pd
import scipy.signal
import torch
import wfdb
from tqdm import tqdm
from ts2vec import TS2Vec
from utils import pkl_load, pkl_save


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


def main():
    # Load the encoder from file if supplied and the model exists
    encoder = load_encoder(path="data/ts2vec/model.pkl")
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

    t = time.time()
    for out_name, dataset in [
        ("train", ptb_df_train),
        ("test", ptb_df_test),
        ("val", ptb_df_val),
    ]:
        print(f"Loading {out_name} data")
        ecg_leads_data = load_raw_data(dataset, sampling_rate, data_folder)
        print(f"Preprocessing {out_name} data")
        preprocessed_data = preprocess(ecg_leads_data)
        print(f"Encoding {out_name} data")
        encoded_data = encoder.encode(
            preprocessed_data.copy(), encoding_window="full_series"
        )
        print(f"Saving parsed {out_name} data")

        filtered_trace = 0
        filtered_unconfirmed = 0
        parsed_data = []
        for i in range(len(dataset)):
            embedding = encoded_data[i]
            report = dataset.iloc[i].report
            ecg_id = dataset.iloc[i].ecg_id
            try:
                if "trace only" in report:
                    filtered_trace += 1
                    continue
                if "unconfirmed report" in report:
                    filtered_unconfirmed += 1
                    report = report.replace("unconfirmed report", "")
            except TypeError:
                print('Error in report:', report)
                print('Error in ecg_id:', ecg_id)

            parsed_data.append({"embedding": embedding, "report": report, "ecg_id": ecg_id})

        pkl_save(
            out_folder + f"parsed_ptb_{out_name}.pkl",
            parsed_data
        )
        print(f"Filtered {filtered_trace} trace only reports and {filtered_unconfirmed} unconfirmed reports")
    print(f"Time taken: {time.time()-t}")

    # This code is for processing the data in batches of n_samples if the memory
    # is not enough to process all the data at once

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


if __name__ == "__main__":
    main()
