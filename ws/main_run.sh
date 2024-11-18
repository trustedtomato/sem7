#!/bin/bash

set -e

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

OCCUPIED_NODES="$(sinfo | grep -oP '(?<=mix )(ailab-l4-\[.+?\])')"
echo "${OCCUPIED_NODES}"
srun --gres=gpu:6 --mem=60G --exclude="${OCCUPIED_NODES}" singularity exec  --nv /ceph/container/pytorch/pytorch_24.09.sif bash ./main.sh "$@"