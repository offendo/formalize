#!/usr/bin/env python3
import os
from typing import Annotated
from typer import Argument, Option, run as typer_run

from transformers import (
    DataCollator,
    DataCollatorForLanguageModeling,
    DefaultDataCollator,
    PreTrainedTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
DEBUG = bool(os.environ.get("DEBUG", "") != "")


def load_model(
    model_name: str,
    max_length: int,
    lora_rank: int,
    gpu_memory_utilization: float,
    seed: int,
    unsloth: bool = False,
):
    if unsloth:
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
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=lora_rank != -1,  # if we're using LoRA, load in 4 bit mode
            trust_remote_code=True,
            device_map="auto",
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


class FastLanguageTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Cross entropy loss (autoformalization loss)
        inputs["output_hidden_states"] = True
        ce_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        # Last hidden state for contrastive loss
        hidden_states = outputs.hidden_states[-1]  # type:ignore

        # Get the index of the end of the prompt, so we can get the representation of the natural language
        # FIXME: for some reason, we overestimated by 3, so we have to subtract 3 here
        nl_index = inputs["input_length"] - 3
        fl_index = torch.sum(inputs["attention_mask"], dim=1) - 1

        # Get the states for the end of the input (NL) and end out of the output (FL)
        fl_state = hidden_states[:, fl_index]
        nl_state = hidden_states[:, nl_index]

        # Do mean Log Softmax over the cosine similarity
        cos = F.cosine_similarity(fl_state, nl_state, dim=-1)
        cl_loss = torch.mean(F.log_softmax(cos, dim=-1))

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

    def tokenize(examples):
        batch = tokenizer(examples["text"], padding=False)
        batch["input_length"] = examples["input_length"]
        return batch

    dataset: DatasetDict = load_dataset(dataset_name)  # type:ignore
    if "input_length" not in dataset.column_names["train"]:
        dataset = dataset.map(get_input_length, batched=True)

    dataset = dataset.map(apply_template, batched=True)
    dataset = dataset.map(tokenize, batched=True)
    return dataset  # type:ignore


class CustomCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples, *args, **kwargs):
        if isinstance(examples, dict):
            input_length = examples["input_length"]
        else:
            input_length = torch.tensor([ex["input_length"] for ex in examples], dtype=torch.long)
        texts = [ex.pop("text") for ex in examples]
        inputs = [ex.pop("input") for ex in examples]
        outputs = [ex.pop("output") for ex in examples]
        batch = super().__call__(examples, *args, **kwargs)
        batch["input_length"] = input_length
        return batch


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
    unsloth: Annotated[bool, Option("--unsloth", help="enable unsloth", rich_help_panel="Training Config")] = False,
    # fmt:on
):
    model, tokenizer = load_model(
        model_name,
        max_length=max_tokens,
        lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        unsloth=unsloth,
    )

    training_args = SFTConfig(
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type=scheduler,
        optim=optimizer,
        logging_steps=10,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,  # Increase to 4 for smoother training
        num_train_epochs=num_epochs,  # Set to 1 for a full training run
        save_steps=500,
        report_to="wandb",  # Can use Weights & Biases
        output_dir=output_dir,
        max_seq_length=max_tokens,
        remove_unused_columns=False,
    )
    collator = CustomCollator(tokenizer=tokenizer, mlm=False)
    data = load_data(dataset, tokenizer)
    trainer = FastLanguageTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        data_collator=collator,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    typer_run(train)
