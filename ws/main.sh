#!/bin/bash

set -e

srun --gres=gpu:1 singularity exec --nv /ceph/container/pytorch_* python3 main.py