import json
import subprocess

with open("ts2vec_experiments.json", "r") as f:
    experiments = json.load(f)

processes = []

for experiment in experiments:
    model_name = experiment["model_name"]
    tsencoder_depth = experiment["tsencoder_depth"]
    tsencoder_hidden_dim = experiment["tsencoder_hidden_dim"]
    ts_embedding_dim = experiment["ts_embedding_dim"]
    command = [
        "srun",
        "--gres=gpu:1",
        "--mem-per-gpu=30G",
        "singularity",
        "exec",
        "--nv",
        "/ceph/container/pytorch/pytorch_24.09.sif",
        "bash ./main.sh",
        "-n",
        "train_ts2vec.py",
        "--model_name",
        model_name,
        "--tsencoder_depth",
        tsencoder_depth,
        "--tsencoder_hidden_dim",
        tsencoder_hidden_dim,
        "--ts_embedding_dim",
        ts_embedding_dim,
        "--epochs",
        1000,
    ]
    processes.append(subprocess.Popen(command))

for process in processes:
    process.wait()
