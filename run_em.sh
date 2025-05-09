#!/bin/bash

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

# Do training
axolotl fetch deepspeed_configs
axolotl train $CONFIG
echo "finished training"

# merge the lora weights
axolotl merge-lora $CONFIG
echo "merged"

# Do inference
python scripts/run_herald.py \
  --model $MODEL \
  --dataset $DATASET \
  --output_path $OUTPUT_PATH \
  --num_samples $NUM_SAMPLES \
  --generations $GENERATIONS
echo "finished inference"
