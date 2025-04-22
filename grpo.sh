#!/bin/bash

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

pip install -U torchao more_itertools trl==0.15.2 icecream
if [[ $OPTIMIZER =~ .*lomo ]]; then
  pip install lomo-optim
fi;

export WANDB_PROJECT='grpo-autoformalization'

export SEED=1234

python src/formalize/grpo.py \
  --model-name "offendo/lean-alignment" \
  --alignment-model-path "offendo/lean-alignment" \
  --dataset "AI-MO/minif2f_test" \
  --output-dir "/volume/grpo_$WANDB_RUN" \
  --epochs $EPOCHS \
  --max-prompt 256 \
  --max-completion 256 \
  --batch-size $BATCH_SIZE --num-generations $BATCH_SIZE \
  --quantize-alignment-model \
  --optimizer $OPTIMIZER --scheduler $SCHEDULER \
  --eval-steps $EVAL_STEPS
