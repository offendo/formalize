huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

export WANDB_PROJECT='formal-align'
export WANDB_RUN='full_bs128_higher_lr'

python src/formalize/align.py \
    --model-name "meta-llama/Meta-Llama-3.1-8B" \
    --dataset "offendo/formal-align-redux" \
    --output-dir "/volume/formal_align_$WANDB_RUN" \
    --max-tokens 2048 \
    --seed 1234 \
    --learning-rate "2e-5" --scheduler "cosine" --optimizer "paged_adamw_8bit" \
    --num-epochs 6 \
    --batch-size 1 --gradient-accumulation 128
