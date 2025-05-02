#!/bin/bash

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)

pip install vllm
python scripts/run_herald.py \
  --model $MODEL \
  --dataset $DATASET \
  --output_path $OUTPUT_PATH
