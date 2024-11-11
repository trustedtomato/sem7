import pandas as pd
import numpy as np
import wfdb
from ts2vec import TS2Vec
from tqdm import tqdm
import pickle
import neurokit2 as nk
import os

def preprocess(data, sampling_rate=100):
    # Data is of shape (n_batch, n_samples, n_channels)
    # Filter the data
    for i in range(data.shape[0]):
        # Clean each channel of the data
        for j in range(data.shape[-1]):
            data[i, :, j] = nk.ecg_clean(data[i, :, j], sampling_rate=sampling_rate, method="neurokit")
    # Normalize the data to have zero mean and unit variance along the time axis
    data = (data - np.mean(data, axis=1)) / np.std(data, axis=1)
    return data

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def load_encoder(path = None):
    encoder = TS2Vec(input_dims=12, device="cuda")
    if path is not None and os.path.exists(path):
        encoder.load(path)
    return encoder

# Load the encoder from file if supplied
encoder = load_encoder(path="results/encoder.pt")
sampling_rate = 100
# Number of samples to process at once
n_samples = 100
data_path = "data/ptb-xl/"
out_path = "data/ptb-xl/"

test_fold = 10
ptb_df = pd.read_csv(data_path+'ptbxl_database_translated.csv')
ptb_df_train = ptb_df[ptb_df.strat_fold < test_fold]
ptb_df_test = ptb_df[ptb_df.strat_fold == test_fold]

# Encodes the data row by row and saves the embeddings and reports in a pickle file

for out_name, dataset in [("parsed_ptb_train.pkl", ptb_df_train), ("parsed_ptb_test.pkl", ptb_df_test)]:
    embedding_report_list = []
    for i in tqdm(range(len(dataset))):
        row = dataset[i:i+1]
        data = load_raw_data(row, sampling_rate, data_path)
        #print(data[0,:,0].shape)
        print(np.mean(data[0,:,0]))
        print(np.mean(data, axis=1))
        preprocessed_data = data#preprocess(data)
        encoded_data = encoder.encode(preprocessed_data, encoding_window="full_series")
        embedding_report_list.append({"embedding": encoded_data[0], "report": row.report})
    with open(out_path+out_name, "wb") as f:
        pickle.dump(embedding_report_list, f)


# This code is for processing the data in batches of n_samples but it's not that much faster

# for out_name, dataset in [("parsed_ptb_train.pkl", ptb_df_train), ("parsed_ptb_test.pkl", ptb_df_test)]:
#     embedding_report_list = []
#     for i in tqdm(range(len(dataset)//n_samples)):
#         dataset_slice = dataset[i*n_samples:min((i+1)*n_samples, len(dataset))]
#         data = load_raw_data(dataset_slice, sampling_rate, data_path)
#         preprocessed_data = preprocess(data)
#         encoded_data = encoder.encode(preprocessed_data, encoding_window="full_series")
#         for i in range(n_samples):
#             embedding_report_list.append({"embedding": encoded_data[i], "report": dataset_slice.iloc[i].report})
#     with open(out_path+out_name, "wb") as f:
#         pickle.dump(embedding_report_list, f)

    

    
    




    









