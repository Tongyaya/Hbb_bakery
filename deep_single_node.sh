#!/bin/bash

NPROC_PER_NODE=2

CUDA_VISIBLE_DEVICES=2,4 python -m torch.distributed.run \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes 1 \
    --standalone \
    src/train.py myscripts/sft.yaml
