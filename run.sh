huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

export WANDB_PROJECT='formal-align'
export WANDB_RUN='debug'

python src/formalize/align.py train \
    --model-name "meta-llama/Meta-Llama-3.1-8B" \
    --dataset "offendo/formal-align-redux" \
    --eval-dataset "offendo/formal-align-redux-test" \
    --output-dir "/volume/formal_align_$WANDB_RUN" \
    --max-tokens 2048 \
    --seed 1234 \
    --learning-rate "2e-5" --scheduler "cosine" --optimizer "paged_adamw_8bit" \
    --num-epochs 1000 \
    --batch-size 2
