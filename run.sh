# huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
# wandb login $(cat /etc/api-tokens/wandb-token)
# 
# pip install torchao
# 
# export WANDB_PROJECT='formal-align'
# export WANDB_RUN='with_eval_bs32ga16'
# 
# export SEED=1234

if [[ $GRAD_CKPT = "False" ]]; then
  export GRAD_CKPT=""
else
  export GRAD_CKPT="--gradient-checkpointing"
fi
python src/formalize/align.py train \
    --model-name "meta-llama/Meta-Llama-3.1-8B" \
    --dataset "offendo/formal-align-redux" \
    --eval-dataset "offendo/formal-align-redux-test" \
    --output-dir "/volume/formal_align_$WANDB_RUN" \
    --max-tokens 2048 \
    --seed $SEED \
    --learning-rate "$LR" --scheduler "$SCHEDULER" --optimizer "$OPTIMIZER" \
    --num-epochs $EPOCHS \
    --batch-size $BATCH_SIZE --gradient-accumulation $GRAD_ACC \
    $GRAD_CKPT \
    --lora-rank $LORA_RANK
