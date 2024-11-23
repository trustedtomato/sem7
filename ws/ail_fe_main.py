import argparse
from importlib.util import find_spec
import os
import subprocess
import sys

from ail_fe_main_scmds import SCmd
from ail_parser import parse_intermixed_args


def main(args: argparse.Namespace):
    if not os.path.isdir(".venv"):
        subprocess.run(["python3", "-m", "venv", ".venv"])

    if not args.keep_jobs:
        result = subprocess.run(
            ["squeue", "--me", "--nohead", "--format", "%F"],
            capture_output=True,
            text=True,
        )
        jobs = set(result.stdout.split())
        for job in jobs:
            subprocess.run(["scancel", job])

    # result = subprocess.run(["sinfo"], capture_output=True, text=True)
    # free_nodes = ""
    # for line in result.stdout.splitlines():
    #     if "idle" in line:
    #         free_nodes = line.split("idle ")[1]
    #         break

    # print(f"Idle nodes: {free_nodes}")

    scmds: list[SCmd] = __import__(args.scmds_from).get_scmds(args)
    python_file_spec = find_spec(args.file)
    if python_file_spec is None:
        raise ImportError(f"Module {args.f} not found")
    python_file = python_file_spec.origin
    if python_file is None:
        raise ImportError(f"Module {args.f} not found (origin)")

    for scmd in scmds:
        command = [
            scmd.program,
            *scmd.opts,
            # "-w",
            # free_nodes,
            "singularity",
            "exec",
            "--nv",
            "/ceph/container/pytorch/pytorch_24.09.sif",
            "bash",
            "ail_slurm_main.sh",
            python_file,
            *sys.argv[1:],
        ]
        print("Running command", command)
        subprocess.run(command)


if __name__ == "__main__":
    args = parse_intermixed_args(uninstalled_requirements=True)
    main(args)
