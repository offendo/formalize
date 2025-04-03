huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

pip install torchao

export WANDB_PROJECT='formal-align'

export SEED=1234

python src/formalize/align.py train \
    --model-name $MODEL_NAME \
    --dataset "offendo/formal-align-redux" \
    --eval-dataset "offendo/formal-align-redux-test" \
    --output-dir "/volume/formal_align_$WANDB_RUN" \
    --max-tokens 2048 \
    --seed $SEED \
    --learning-rate "$LR" --scheduler "$SCHEDULER" --optimizer "$OPTIMIZER" \
    --num-epochs $EPOCHS \
    --batch-size $BATCH_SIZE --gradient-accumulation $GRAD_ACC \
    $GRAD_CKPT \
    $ADD_SPECIAL_REPRESENTATION \
    --lora-rank $LORA_RANK --eval-steps $EVAL_STEPS \
