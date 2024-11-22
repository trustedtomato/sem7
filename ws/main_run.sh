#!/bin/bash

set -e

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Cancel all our jobs before starting a new one
squeue --me --nohead --format %F | uniq | xargs -I %s -- scancel %s
FREE_NODES="$(sinfo | grep -oP '(?<=idle )(ailab-l4-\[.+?\])')"
echo Idle nodes: $FREE_NODES
srun --gres=gpu:8 --mem=60G -w "${FREE_NODES}" singularity exec  --nv /ceph/container/pytorch/pytorch_24.09.sif bash ./main.sh "$@"
