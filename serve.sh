#!/bin/bash

vllm serve meta-llama/Meta-Llama-3.1-8B \
    --enable-lora \
    --lora-modules fa=$1 \
    --max-lora-rank 256 \
    --dtype "auto" \
    --max-model-len 2048 \
    --tensor-parallel-size 2 \
    --swap-space 8 \
    --gpu-memory-utilization 0.8
