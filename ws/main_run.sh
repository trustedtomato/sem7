#!/bin/bash

set -e

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Cancel all our jobs before starting a new one
squeue --me --nohead --format %F | uniq | xargs -I %s -- scancel %s

srun --gres=gpu:6 --mem=60G singularity exec  --nv /ceph/container/pytorch/pytorch_24.09.sif bash ./main.sh "$@"