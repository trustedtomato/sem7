#!/bin/bash

set -e

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Cancel all our jobs before starting a new one
squeue --me --nohead --format %F | uniq | xargs -I %s -- scancel %s
OCCUPIED_NODES="$(sinfo | grep -oP '(?<=mix )(ailab-l4-\[.+?\])')"
echo Occupied nodes: $OCCUPIED_NODES
srun --gres=gpu:8 --mem=60G --exclude="${OCCUPIED_NODES}" singularity exec  --nv /ceph/container/pytorch/pytorch_24.09.sif bash ./main.sh "$@"
