#!/usr/bin/env python3
from typing import Annotated
from typer import Argument, Option, run as typer_run

from transformers import DataCollator, DefaultDataCollator, PreTrainedTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from datasets import load_dataset, Dataset, DatasetDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_model(
    model_name: str,
    max_length: int,
    lora_rank: int,
    gpu_memory_utilization: float,
    seed: int,
):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        load_in_4bit=lora_rank != -1,  # if we're using LoRA, load in 4 bit mode
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=lora_rank if lora_rank != -1 else 64,
        gpu_memory_utilization=gpu_memory_utilization,  # Reduce if out of memory
    )

    if lora_rank != -1:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_rank,
            use_gradient_checkpointing="unsloth",  # Enable long context finetuning
            use_rslora=False,
            bias="none",
            lora_dropout=0,
            random_state=seed,
        )
    return model, tokenizer


class FastLanguageTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Cross entropy loss (autoformalization loss)
        ce_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        # Contrastive loss
        hidden_states = outputs["hidden_states"]
        input_length = inputs["input_length"]

        # Get the states for the end of the input (NL) and end out of the output (FL)
        fl_state = hidden_states[-1]
        nl_state = hidden_states[input_length]

        # Do mean Log Softmax over the cosine similarity
        cos = F.cosine_similarity(fl_state, nl_state, dim=-1)
        cl_loss = torch.mean(F.log_softmax(cos))

        # loss = cross entropy + contrastive loss
        loss = ce_loss + cl_loss
        return (loss, outputs) if return_outputs else loss


def load_data(dataset_name: str, tokenizer: PreTrainedTokenizer) -> DatasetDict:
    EOS: str = tokenizer.eos_token  # type:ignore

    def get_input_length(examples):
        input_prompts = ["\n".join(ex.split("\n")[:-1]) for ex in examples["input"]]
        inputs = tokenizer(input_prompts, add_special_tokens=False).input_ids
        return {"input_length": [len(i) for i in inputs]}

    def apply_template(examples):
        prompts = []
        for input, output in zip(examples["input"], examples["output"]):
            prompts.append(
                f"Statement in natural language:\n{input}\nTranslate the statement in natural language to Lean:\n{output}"
                + EOS
            )
        return {"text": prompts}

    dataset: DatasetDict = load_dataset(dataset_name)  # type:ignore
    if "input_length" not in dataset.column_names["train"]:
        dataset = dataset.map(get_input_length, batched=True)

    dataset = dataset.map(apply_template, batched=True)
    return dataset  # type:ignore


def train(
    # fmt:off
    model_name: Annotated[str, Option(help="path to model to train", rich_help_panel="Model Config")],
    dataset: Annotated[str, Option(help="path to datasets to train", rich_help_panel="Data Config")],
    output_dir: Annotated[str, Option(help="path to output directory", rich_help_panel="Data Config")],
    max_tokens: Annotated[int, Option(help="max tokens", rich_help_panel="Training Config")] = 2048,
    lora_rank: Annotated[int, Option(help="lora rank to train (-1 for no lora)", rich_help_panel="Model Config")] = -1,
    gpu_memory_utilization: Annotated[float, Option(help="percent of GPU to give to unsloth", rich_help_panel="Training Config")] = 0.6,
    seed: Annotated[int, Option(help="random seed", rich_help_panel="Training Config")] = 1234,
    learning_rate: Annotated[float, Option(help="learning rate", rich_help_panel="Training Config")] = 5e-5,
    scheduler: Annotated[str, Option(help="learning rate scheduler", rich_help_panel="Training Config")] = "cosine",
    optimizer: Annotated[str, Option(help="optimizer", rich_help_panel="Training Config")] = "paged_adamw_8bit",
    num_epochs: Annotated[int, Option(help="number of training epochs", rich_help_panel="Training Config")] = 1,
    batch_size: Annotated[int, Option(help="batch size", rich_help_panel="Training Config")] = 4,
    gradient_accumulation: Annotated[int, Option(help="gradient accumulation", rich_help_panel="Training Config")] = 1,
    # fmt:on
):
    model, tokenizer = load_model(
        model_name,
        max_length=max_tokens,
        lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
    )

    from unsloth import is_bfloat16_supported

    training_args = SFTConfig(
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type=scheduler,
        optim=optimizer,
        logging_steps=10,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,  # Increase to 4 for smoother training
        num_train_epochs=num_epochs,  # Set to 1 for a full training run
        save_steps=500,
        report_to="wandb",  # Can use Weights & Biases
        output_dir=output_dir,
        max_seq_length=max_tokens,
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template="Translate the statement in natural language to Lean:", tokenizer=tokenizer
    )
    data = load_data(dataset, tokenizer)
    trainer = FastLanguageTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=data["train"],
        data_collator=collator,
        eval_dataset=data["validation"],
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    typer_run(train)
