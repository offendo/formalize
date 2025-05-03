#!/bin/bash

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit

axolotl fetch deepspeed_configs
axolotl train $CONFIG

# merge the lora weights
axolotl merge-lora $CONFIG
