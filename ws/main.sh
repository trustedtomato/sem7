#!/bin/bash

set -e

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

srun --gres=gpu:1 singularity exec --nv /ceph/container/pytorch/pytorch_24.09.sif bash ./main_run.sh