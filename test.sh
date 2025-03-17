#!/usr/bin/env sh

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

export WANDB_PROJECT='formal-align'
export WANDB_RUN='v1'

python src/formalize/align.py test \
    --model-name "meta-llama/Meta-Llama-3.1-8B" \
    --adapter-name "/volume/formal_align_$WANDB_RUN" \
    --dataset "offendo/formal-align-redux-test" \
    --output-dir "/volume/formal_align_$WANDB_RUN" \
    --max-tokens 2048 \
    --seed 1234 \
    --batch-size 2 --unsloth
