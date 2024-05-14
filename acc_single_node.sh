#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file myscripts/single_config.yaml \
    src/train.py myscripts/sft.yaml
