import json
import os
import subprocess

with open("ts2vec_experiments.json", "r") as f:
    experiments = json.load(f)

script_dir = ".experiment_scripts/"
script_name = "ts2vec_experiments.slurm"
output_name = "ts2vec_experiments.out"
error_name = "ts2vec_experiments.err"
os.makedirs(script_dir, exist_ok=True)


with open(os.path.join(script_dir, script_name), "w") as fh:
    fh.write("#!/bin/bash\n")
    fh.write("#SBATCH --job-name=ts2vec_exp\n")
    fh.write(f"#SBATCH --output={os.path.join(script_dir, output_name)}\n")
    fh.write(f"#SBATCH --error={os.path.join(script_dir, error_name)}\n")
    fh.write("#SBATCH --mem-per-gpu=30G\n")
    fh.write("#SBATCH --nodes=1\n")
    fh.write(f"#SBATCH --ntasks={len(experiments)}\n")
    fh.write("#SBATCH --cpus-per-task=4\n")
    fh.write(f"#SBATCH --array=1-{len(experiments)}\n")
    fh.write(f"#SBATCH --gpus={len(experiments)}\n")
    fh.write("#SBATCH --gpus-per-task=1\n")
    fh.write("echo $CUDA_VISIBLE_DEVICES\n")
    fh.write("srun -l --ntasks=1 --gpus=1 echo $CUDA_VISIBLE_DEVICES\n")
    # for experiment in experiments:
    #     model_name = experiment["model_name"]
    #     tsencoder_depth = experiment["tsencoder_depth"]
    #     tsencoder_hidden_dim = experiment["tsencoder_hidden_dim"]
    #     ts_embedding_dim = experiment["ts_embedding_dim"]
    #     command = [
    #         "srun",
    #         "--ntasks=1",
    #         "singularity",
    #         "exec",
    #         "--nv",
    #         "/ceph/container/pytorch/pytorch_24.09.sif",
    #         "bash ./main.sh",
    #         "-n",
    #         "train_ts2vec.py",
    #         "--model_name",
    #         f'"{model_name}"',
    #         "--tsencoder_depth",
    #         f"{tsencoder_depth}",
    #         "--tsencoder_hidden_dim",
    #         f"{tsencoder_hidden_dim}",
    #         "--ts_embedding_dim",
    #         f"{ts_embedding_dim}",
    #         "--epochs",
    #         "1000",
    #     ]
    #     fh.write(" ".join(command) + "\n")

subprocess.run(["sbatch", ".experiment_scripts/ts2vec_experiments.slurm"])
