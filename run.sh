huggingface-cli login --token $(cat /etc/api-tokens/hf-token)
wandb login $(cat /etc/api-tokens/wandb-token)

python src/formalize/align.py \
    --model-name "unsloth/Meta-Llama-3.1-8B" \
    --dataset "offendo/formal-align-redux" \
    --output-dir "./formal_align" \
    --max-tokens 2048 \
    --seed 1234 \
    --learning-rate "2e-6" --scheduler "constant" --optimizer "paged_adamw_8bit" \
    --num-epochs 1 \
    --batch-size 4 --gradient-accumulation 16 \
    --lora-rank 64
