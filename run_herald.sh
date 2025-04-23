#!/bin/bash

pip install -U vllm torch
python run_herald.py \
  --model FrenzyMath/Herald_translator \
  --dataset offendo/math-atlas-titled-theorems \
  --output_path /volume/math_atlas_herald_predictions.json
