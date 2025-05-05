#!/bin/bash

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

# Install cuda
apt install -y wget
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt -y install cuda-toolkit-12-4


# Do training
axolotl fetch deepspeed_configs
axolotl train $CONFIG

# merge the lora weights
axolotl merge-lora $CONFIG

# Do inference
python scripts/run_herald.py \
  --model $MODEL \
  --dataset $DATASET \
  --output_path $OUTPUT_PATH \
  --num_samples $NUM_SAMPLES \
  --generations $GENERATIONS

# Do alignment scoring
python scripts/run_align.py \
  --model $MODEL \
  --dataset $OUTPUT_PATH \
  --output_path "$OUTPUT_PATH.scored" \
  --batch_size 2
