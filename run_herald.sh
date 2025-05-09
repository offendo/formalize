#!/bin/bash

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)

python scripts/run_herald.py \
  --model $MODEL \
  --dataset $DATASET \
  --output_path $OUTPUT_PATH \
  --num_samples $NUM_SAMPLES 
