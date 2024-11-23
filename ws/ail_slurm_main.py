import subprocess
import sys

import torch

from ail_parser import parse_intermixed_args

if __name__ == "__main__":
    args = parse_intermixed_args()

    if not args.no_install:
        subprocess.call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )

    gpu_count = torch.cuda.device_count()
    subprocess.run(
        ["torchrun", "--nproc_per_node=" + str(gpu_count), args.file, *sys.argv[1:]]
    )
