#!/bin/bash

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

# Do alignment scoring
python src/formalize/align.py predict-herald \
  --model-name "/volume/formal_align/formal_align_qwen_3b_instruct/" \
  --dataset "$DATASET" \
  --output-json "$OUTPUT_PATH" \
  --batch-size 8
echo "finished scoring"
