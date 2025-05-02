#!/bin/bash

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)

uv pip install --system --no-build-isolation vllm
python scripts/run_herald.py \
  --model $MODEL \
  --dataset $DATASET \
  --output_path $OUTPUT_PATH
