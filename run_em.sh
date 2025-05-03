#!/bin/bash

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

# Install xformers
pip3 install xformers --index-url https://download.pytorch.org/whl/cu124

axolotl fetch deepspeed_configs
axolotl train $CONFIG

# # merge the lora weights
# axolotl merge-lora $CONFIG
