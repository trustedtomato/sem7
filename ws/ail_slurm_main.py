import subprocess
import sys

from ail_parser import parse_intermixed_args

if __name__ == "__main__":
    # cut off the first argument, which is the Python file path
    args = parse_intermixed_args(sys_args=sys.argv[2:], uninstalled_requirements=True)

    if not args.no_install:
        subprocess.call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )

    import torch

    gpu_count = torch.cuda.device_count()
    command = (
        ["torchrun", "--nproc_per_node=" + str(gpu_count)] + sys.argv[1:]
        if args.torchrun
        else ["python3"] + sys.argv[1:]
    )
    print("Running command: ", command)
    subprocess.run(command)
