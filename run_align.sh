#!/bin/bash

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

export NUM_DEVICES=$(nvidia-smi  -L | wc -l)

# Do alignment scoring
accelerate launch --num_processes=$NUM_DEVICES src/formalize/align.py predict-herald \
  --model-name "/volume/formal_align/formal_align_qwen_3b_instruct/" \
  --dataset "$ALIGN_INPUT" \
  --output-json "$ALIGN_OUTPUT" \
  --batch-size $ALIGN_BATCH_SIZE
echo "finished scoring"
