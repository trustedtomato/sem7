from itertools import takewhile
import subprocess
import sys
import torch

args = sys.argv[1:]

# get options for run.sh
opts = list(takewhile(lambda x: x.startswith("-"), args))

no_install = "-n" in opts
use_gpu = "-g" in opts

args = args[len(opts) :]

# get script
script = (args[0:1] or [""])[0]
if not ".py" in script:
    print("Usage: ./run.sh [-n] <script.py> [args]")
    exit(1)

if not no_install:
    subprocess.call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

if use_gpu:
    gpu_count = torch.cuda.device_count()
    subprocess.run(["torchrun", "--nproc_per_node=" + str(gpu_count), *args])
else:
    subprocess.run([sys.executable, "-u", *args])
