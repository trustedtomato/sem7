import argparse
import json

from ail_fe_main_scmds import SCmd


def modify_parser(parser: argparse._ArgumentGroup):
    pass


def get_scmd(experiment):
    hd = experiment["tsencoder_hidden_dim"]
    d = experiment["tsencoder_depth"]
    ed = experiment["ts_embedding_dim"]
    name = f"ts2vec_hd{hd}_d{d}_ed{ed}"
    return SCmd(
        program="sbatch",
        opts=["-J", name, f"--gres=gpu:1", "--mem-per-gpu=30G"],
        python_module="train_ts2vec",
        python_args=[
            "--model_name",
            name,
            "--tsencoder_depth",
            d,
            "--tsencoder_hidden_dim",
            hd,
            "--ts_embedding_dim",
            ed,
            "--epochs",
            1000,
        ],
    )


def get_scmds(args: argparse.Namespace):
    with open("ts2vec_experiments.json", "r") as f:
        experiments = json.load(f)
    return [get_scmd(experiment) for experiment in experiments]
