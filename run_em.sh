#!/bin/bash

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

# Do training
axolotl fetch deepspeed_configs
axolotl train $CONFIG
echo "finished training"

# merge the lora weights
# axolotl merge-lora $CONFIG
# echo "merged"
