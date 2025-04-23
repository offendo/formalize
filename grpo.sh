#!/bin/bash

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

pip install -U torchao more_itertools icecream 
if [[ $OPTIMIZER =~ .*lomo ]]; then
  pip install lomo-optim
fi;

export WANDB_PROJECT='grpo-autoformalization'

export SEED=1234

python src/formalize/grpo.py \
  --model-name "offendo/lean-alignment" \
  --alignment-model-path "offendo/lean-alignment" \
  --dataset $DATASET \
  --output-dir "/volume/grpo_$WANDB_RUN" \
  --num-epochs $EPOCHS \
  --learning-rate 1e-6 \
  --max-prompt 256 \
  --max-completion 256 \
  --batch-size $BATCH_SIZE --num-generations $NUM_GENERATIONS --gradient-accumulation $GRAD_ACC \
  --quantize-alignment-model \
  --optimizer $OPTIMIZER --scheduler $SCHEDULER
