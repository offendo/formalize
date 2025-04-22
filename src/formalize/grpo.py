#!/usr/bin/env python3
# pyright: reportPrivateImportUsage=false
import re
import torch
import typer
import logging
from icecream import ic

from pathlib import Path

from typing import Annotated
from typer import Argument, Option
from datasets import load_dataset, Dataset, DatasetDict
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from align import CustomCollator, compute_formal_align_score, load_model as load_align_model, tokenize_chat
from more_itertools import chunked

typer.core.rich = None
app = typer.Typer(pretty_exceptions_short=False, pretty_exceptions_show_locals=False)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
            attn_implementation="sdpa",
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logging.info(f"Loaded model {model_name} in DEBUG mode.")
    else:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True) if quantize else None
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=lora_rank != -1,  # if we're using LoRA, load in 4 bit mode
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
            # attn_implementation="flash_attention_2",
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logging.info(f"Loaded model {model_name} in PRODUCTION mode.")

    tokenizer.padding_side = "left"
    logging.info("Using left-padding for generation model.")
    return model, tokenizer


def load_data(dataset_name: str):
    ds = load_dataset(dataset_name, split="train")
    if "minif2f" in dataset_name.lower():
        ds = ds.rename_column("informal_prefix", "natural_language")

    assert "natural_language" in ds.column_names  # type:ignore

    def apply_prompt(examples):
        prompts = [
            [
                {"role": "system", "content": "You are an expert mathematician."},
                {"role": "user", "content": f"Translate the following statement in natural language to Lean:\n{nl}"},
            ]
            for nl in examples["natural_language"]
        ]
        return {"prompt": prompts}

    logging.info(f"Loaded dataset {dataset_name}")
    ds = ds.map(apply_prompt, batched=True)
    logging.info(f"Formatted data into message lists.")
    logging.info(f"Example: {ds[0]['prompt']}")
    return ds


def make_formal_align_reward_fn(formal_align_model_path: str | Path, quantize: bool = False, align_batch_size: int = 1):
    align_model, align_tokenizer = load_align_model(
        formal_align_model_path,
        max_length=2048,
        lora_rank=-1,
        gpu_memory_utilization=0.3,
        unsloth=False,
        debug=False,
        seed=1234,
        quantize=quantize,
    )
    align_model.eval()
    logging.info(f"Loaded alignment model from {formal_align_model_path}")
    chat_marker = align_tokenizer("<|im_start|>assistant", add_special_tokens=False).input_ids
    collator = CustomCollator(tokenizer=align_tokenizer, mask_inputs=False, mlm=False)

    def align_reward_fn(completions: list[list[dict]], natural_language: list[str], **kwargs):
        # completions will be in message format, so a list of [{'role': ..., 'content': ...}]
        # the inner list has length equal to the number of completions asked for (?)
        completion_content = [comp[0]["content"] for comp in completions]

        # Lop off everything after the :=
        completion_content = [comp.split(":=")[0] for comp in completion_content]

        examples = {"input": natural_language, "output": completion_content}
        batch = tokenize_chat(examples, chat_marker, align_tokenizer)

        # Convert the dict[list] into list[dict] because that's what the collator expects
        uncollated = [{} for _ in natural_language]
        for key, val in batch.items():
            for i, v in enumerate(val):
                uncollated[i][key] = v

        ex = align_tokenizer.decode(uncollated[0]["input_ids"])
        logging.debug(f"Formatted and tokenized reward inputs: {ex}")

        # Now we collate it, which converts it back into dict[list], but with some extra processing
        cert_scores = []
        sim_scores = []
        for batch in chunked(uncollated, align_batch_size):
            model_inputs = collator(batch)

            # And finally convert it all over to the device
            for key, val in model_inputs.items():
                val.requires_grad = False
                model_inputs[key] = val.to(align_model.device)

            # Pass it to the model & compute scores
            model_inputs["output_hidden_states"] = True
            with torch.no_grad():
                model_output = align_model(**model_inputs)
            scores = compute_formal_align_score(model_inputs, model_output)

            cert_scores.extend(scores["certainty_score"].view(-1).tolist())
            sim_scores.extend(scores["similarity_score"].view(-1).tolist())

        return [c + s for c, s in zip(cert_scores, sim_scores)]

    return align_reward_fn


def make_max_thinking_length_reward_fn(max_length):
    def max_thinking_length_reward_fn(completions: list[list[dict]], **kwargs):
        completion_content = [comp[0]["content"] for comp in completions]
        pattern = r"^<think>(.*?)</think><answer>.*?</answer>$"
        matches = [re.match(pattern, content) for content in completion_content]
        lengths = [len(m.group(0)) if m else 0.0 for m in matches]
        return [0.25 if ls <= max_length else 0.0 for ls in lengths]

    return max_thinking_length_reward_fn


def theorem_format_reward_fn(completions: list[list[dict]], **kwargs):
    completion_content = [comp[0]["content"] for comp in completions]
    pattern = r"^<think>(.*?)</think><answer>.*?</answer>$"
    matches = [re.match(pattern, content) for content in completion_content]
    lengths = [len(m.group(0)) if m else 0.0 for m in matches]
    return [0.25 if ls <= max_length else 0.0 for ls in lengths]


def get_reward_fns(
    alignment_model_path: str | None = None, max_thinking_length: int = -1, quantize_alignment_model: bool = False
):
    fns = []
    if alignment_model_path is not None:
        fns.append(make_formal_align_reward_fn(alignment_model_path, quantize=quantize_alignment_model))
    if max_thinking_length > 0:
        fns.append(make_max_thinking_length_reward_fn(max_thinking_length))

    return fns


@app.command()
def train(
    # fmt:off
    # Required stuff
    model_name: Annotated[str, Option(help="path to model to train", rich_help_panel="Model Config")],
    dataset: Annotated[str, Option(help="path to datasets to train", rich_help_panel="Data Config")],
    output_dir: Annotated[str, Option(help="gradient accumulation", rich_help_panel="Data Config")],
    max_prompt: Annotated[int, Option(help="max input tokens", rich_help_panel="Training Config")],
    max_completion: Annotated[int, Option(help="max output tokens", rich_help_panel="Training Config")],
    # Model stuff
    lora_rank: Annotated[int, Option(help="lora rank to train (-1 for no lora)", rich_help_panel="Model Config")] = -1,
    # Reward stuff
    alignment_model_path: Annotated[str, Option(help="name/path of alignment reward model", rich_help_panel="Reward Config")] = "offendo/lean-alignment",
    max_thinking_length: Annotated[int, Option(help="max thinking length for reward", rich_help_panel="Reward Config")] = -1,
    quantize_alignment_model: Annotated[bool, Option("--quantize-alignment-model", help="quantize alignment model", rich_help_panel="Reward Config")] = False,
    # Training stuff
    num_epochs: Annotated[int, Option(help="number of training epochs", rich_help_panel="Training Config")] = 5,
    batch_size: Annotated[int, Option(help="batch size", rich_help_panel="Training Config")] = 4,
    learning_rate: Annotated[float, Option(help="learning rate", rich_help_panel="Training Config")] = 5e-5,
    scheduler: Annotated[str, Option(help="learning rate scheduler", rich_help_panel="Training Config")] = "cosine",
    optimizer: Annotated[str, Option(help="optimizer", rich_help_panel="Training Config")] = "paged_adamw_8bit",
    num_generations: Annotated[int, Option(help="number of GRPO generations", rich_help_panel="Training Config")] = 5,
    seed: Annotated[int, Option(help="random seed", rich_help_panel="Training Config")] = 1234,
    gradient_accumulation: Annotated[int, Option(help="gradient accumulation", rich_help_panel="Training Config")] = 1,
    # Debugging
    debug: Annotated[bool, Option("--debug", help="enable debug mode", rich_help_panel="Training Config")] = False,
    # fmt:on
):
    model, tokenizer = load_model(
        model_name,
        lora_rank=lora_rank,
        debug=debug,
    )

    training_args = GRPOConfig(
        use_vllm=False,  # use vLLM for fast inference!
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type=scheduler,
        optim=optimizer,
        logging_steps=10,
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
        log_completions=True,
    )
    rewards = get_reward_fns(
        alignment_model_path=alignment_model_path,
        max_thinking_length=max_thinking_length,
        quantize_alignment_model=quantize_alignment_model,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=rewards,
        args=training_args,
        train_dataset=load_data(dataset),
    )
    trainer.generation_config.stop_strings = [":=", " :=", " := "]
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    app()
