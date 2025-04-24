#!/bin/bash

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)

pip install -U vllm torch flashinfer
python run_herald.py \
  --model FrenzyMath/Herald_translator \
  --dataset offendo/math-atlas-titled-theorems \
  --output_path /volume/math_atlas_herald_predictions.json
