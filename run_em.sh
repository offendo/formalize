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

# Do alignment scoring
python src/formalize/align.py predict_herald \
  --model_name $MODEL \
  --dataset "$OUTPUT_PATH" \
  --output_json "$OUTPUT_PATH.json" \
  --batch_size 2
echo "finished scoring"
