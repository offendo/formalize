#!/bin/bash

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)

python scripts/run_herald.py \
  --model $MODEL \
  --dataset $HERALD_INPUT \
  --output_path $HERALD_OUTPUT \
  --num_samples $NUM_SAMPLES  \
  --generations $GENERATIONS \
  --temperature $TEMPERATURE \
  --top_p $TOP_P
