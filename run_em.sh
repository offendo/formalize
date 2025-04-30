#!/bin/bash

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

axolotl fetch deepspeed_configs
axolotl train $CONFIG
axolotl merge-lora $CONFIG
