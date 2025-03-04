#!/usr/bin/env python3
from typing import Annotated
from typer import Argument, Option, run as typer_run
from datasets import load_dataset, Dataset, DatasetDict

from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer


def load_model(
    model_name: str,
    max_length: int,
    lora_rank: int,
    gpu_memory_utilization: float,
    seed: int,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        load_in_4bit=lora_rank != -1,  # if we're using LoRA, load in 4 bit mode
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,  # Reduce if out of memory
    )

    if lora_rank != -1:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_rank,
            use_gradient_checkpointing="unsloth",  # Enable long context finetuning
            random_state=seed,
        )
    return model, tokenizer


def load_data(dataset_name: str):
    return load_dataset(dataset_name)


def get_reward_fns(fn_names: list[str]):
    REWARD_FNS = {}
    return [REWARD_FNS[name] for name in fn_names]


def train(
    # fmt:off
    model_name: Annotated[str, Option(help="path to model to train", rich_help_panel="Model Config")],
    dataset: Annotated[str, Option(help="path to datasets to train", rich_help_panel="Data Config")],
    output_dir: Annotated[str, Option(help="gradient accumulation", rich_help_panel="Data Config")],
    max_prompt: Annotated[int, Option(help="max input tokens", rich_help_panel="Training Config")],
    max_completion: Annotated[int, Option(help="max output tokens", rich_help_panel="Training Config")],
    lora_rank: Annotated[int, Option(help="lora rank to train (-1 for no lora)", rich_help_panel="Model Config")] = -1,

    gpu_memory_utilization: Annotated[float, Option(help="percent of GPU to give to unsloth", rich_help_panel="Training Config")] = 0.6,
    seed: Annotated[int, Option(help="random seed", rich_help_panel="Training Config")] = 1234,
    reward: Annotated[list[str], Option(help="list of reward functions to use", rich_help_panel="Training Config")] = ["xml"],
    learning_rate: Annotated[float, Option(help="learning rate", rich_help_panel="Training Config")] = 5e-5,
    scheduler: Annotated[str, Option(help="learning rate scheduler", rich_help_panel="Training Config")] = "cosine",
    optimizer: Annotated[str, Option(help="optimizer", rich_help_panel="Training Config")] = "paged_adamw_8bit",
    num_generations: Annotated[int, Option(help="number of GRPO generations", rich_help_panel="Training Config")] = 5,
    num_epochs: Annotated[int, Option(help="number of training epochs", rich_help_panel="Training Config")] = 5,
    batch_size: Annotated[int, Option(help="batch size", rich_help_panel="Training Config")] = 4,
    gradient_accumulation: Annotated[int, Option(help="gradient accumulation", rich_help_panel="Training Config")] = 1,
    # fmt:on
):
    model, tokenizer = load_model(
        model_name,
        max_length=max_prompt + max_completion,
        lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
    )

    training_args = GRPOConfig(
        use_vllm=True,  # use vLLM for fast inference!
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type=scheduler,
        optim=optimizer,
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,  # Increase to 4 for smoother training
        num_generations=num_generations,  # Decrease if out of memory
        max_prompt_length=max_prompt,
        max_completion_length=max_completion,
        num_train_epochs=num_epochs,  # Set to 1 for a full training run
        save_steps=500,
        max_grad_norm=0.1,
        report_to="wandb",  # Can use Weights & Biases
        output_dir=output_dir,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=get_reward_fns(reward),
        args=training_args,
        train_dataset=load_data(dataset),
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    typer_run(train)
