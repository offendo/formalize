#!/usr/bin/env sh

huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

export WANDB_PROJECT='formal-align'

if [[ $ADAPTER_NAME = "" ]]; then
  export ADAPTER=""
else;
  export ADAPTER="--adapter-name /volume/formal_align_$ADAPTER_NAME"
fi

python src/formalize/align.py test \
    --model-name "$MODEL_NAME" \
    $ADAPTER \
    --dataset "offendo/formal-align-redux-test" \
    --output-dir "/volume/formal_align_$WANDB_RUN" \
    --max-tokens 2048 \
    --seed 1234 \
    --batch-size 2
