import argparse
from typing import Literal


class SCmd:
    def __init__(
        self,
        program: Literal["srun", "sbatch"],
        opts: list[str],
        python_args: list[str | int] = [],
    ):
        self.program = program
        self.opts = opts
        self.python_args = python_args


def get_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--gpu",
        "-g",
        type=int,
        required=True,
        help="FE: The number of GPUs to request in Slurm.",
    )
    return parser


def get_scmds(args: argparse.Namespace) -> list[SCmd]:
    return [SCmd(program="srun", opts=[f"--gres=gpu:{args.gpu}", "--mem-per-gpu=30G"])]


# run.sh --keep-jobs --scmds-from=main_run_scmds.py -g -n
