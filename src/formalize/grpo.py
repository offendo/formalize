#!/usr/bin/env python3
# pyright: reportPrivateImportUsage=false
import re
import torch
from pathlib import Path
from typing import Annotated
from typer import Argument, Option, run as typer_run
from datasets import load_dataset, Dataset, DatasetDict
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from formalize.align import CustomCollator, compute_formal_align_score, load_model as load_align_model, tokenize_chat


def load_model(
    model_name: str | Path,
    lora_rank: int,
    adapter_name: str | None = None,
    debug: bool = False,
    quantize: bool = False,
):
    if debug:
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_size = 32
        config.intermediate_size = 128
        config.num_hidden_layers = 4
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        if adapter_name:
            model = PeftModel.from_pretrained(model, adapter_name, is_trainable=False)
        elif lora_rank != -1:
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
    else:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True) if quantize else None
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=lora_rank != -1,  # if we're using LoRA, load in 4 bit mode
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
        )
        if adapter_name:
            model = PeftModel.from_pretrained(model, adapter_name, is_trainable=False)
        elif lora_rank != -1:
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


def load_data(dataset_name: str):
    ds = load_dataset(dataset_name, split="train")
    if "minif2f" in dataset_name.lower():
        ds = ds.rename_column("informal_prefix", "natural_language")

    assert "natural_language" in ds.column_names  # type:ignore
    return ds


def make_formal_align_reward_fn(formal_align_model_path: str | Path):
    align_model, align_tokenizer = load_align_model(
        formal_align_model_path,
        max_length=2048,
        lora_rank=-1,
        gpu_memory_utilization=0.3,
        unsloth=False,
        debug=False,
        seed=1234,
    )
    chat_marker = align_tokenizer("<|im_start|>assistant", add_special_tokens=False).input_ids
    collator = CustomCollator(tokenizer=align_tokenizer, mask_inputs=False)

    def align_reward_fn(completions: list[list[dict]], natural_language: list[str], **kwargs):
        # completions will be in message format, so a list of [{'role': ..., 'content': ...}]
        # the inner list has length equal to the number of completions asked for (?)
        completion_content = [comp[0]["content"] for comp in completions]
        examples = [{"input": nl, "output": fl} for nl, fl in zip(completion_content, natural_language)]
        batch = tokenize_chat(examples, chat_marker, align_tokenizer)

        # Convert the dict[list] into list[dict] because that's what the collator expects
        uncollated = [{} for _ in examples]
        for key, val in batch.items():
            for i, v in enumerate(val):
                uncollated[i][key] = v

        # Now we collate it, which converts it back into dict[list], but with some extra processing
        model_inputs = collator(uncollated)

        # And finally convert it all over to the device
        for key, val in model_inputs:
            model_inputs[key] = val.to(align_model.device)
        model_output = align_model(**model_inputs)
        scores = compute_formal_align_score(model_inputs, model_output)

        return list(scores["certainty_score"] + scores["similarity_score"])

    return align_reward_fn


def make_max_thinking_length_reward_fn(max_length):
    def max_thinking_length_reward_fn(completions: list[list[dict]], **kwargs):
        completion_content = [comp[0]["content"] for comp in completions]
        pattern = r"^<think>(.*?)</think><answer>.*?</answer>$"
        matches = [re.match(pattern, content) for content in completion_content]
        lengths = [len(m.group(0)) if m else 0.0 for m in matches]
        return [0.25 if ls <= max_length else 0.0 for ls in lengths]

    return max_thinking_length_reward_fn


def get_reward_fns(alignment_model_path: str | None = None, max_thinking_length: int = -1):
    fns = []
    if alignment_model_path is not None:
        fns.append(make_formal_align_reward_fn(alignment_model_path))
    if max_thinking_length > 0:
        fns.append(make_max_thinking_length_reward_fn(max_thinking_length))

    return fns


def train(
    # fmt:off
    model_name: Annotated[str, Option(help="path to model to train", rich_help_panel="Model Config")],
    dataset: Annotated[str, Option(help="path to datasets to train", rich_help_panel="Data Config")],
    output_dir: Annotated[str, Option(help="gradient accumulation", rich_help_panel="Data Config")],
    max_prompt: Annotated[int, Option(help="max input tokens", rich_help_panel="Training Config")],
    max_completion: Annotated[int, Option(help="max output tokens", rich_help_panel="Training Config")],
    lora_rank: Annotated[int, Option(help="lora rank to train (-1 for no lora)", rich_help_panel="Model Config")] = -1,

    seed: Annotated[int, Option(help="random seed", rich_help_panel="Training Config")] = 1234,
    alignment_model_path: Annotated[str, Option(help="name/path of alignment reward model", rich_help_panel="Training Config")] = "offendo/lean-alignment",
    max_thinking_length: Annotated[int, Option(help="max thinking length for reward", rich_help_panel="Training Config")] = -1,
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
        lora_rank=lora_rank,
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
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
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
        seed=seed,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=get_reward_fns(alignment_model_path=alignment_model_path, max_thinking_length=max_thinking_length),
        args=training_args,
        train_dataset=load_data(dataset)["train"],
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    typer_run(train)
