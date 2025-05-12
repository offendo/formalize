#!/usr/bin/env python3
# pyright: reportPrivateImportUsage=false
import json
import os
import pickle
import string
import math
from dataclasses import dataclass

from pathlib import Path
from pprint import pprint
from typing import Annotated, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
import accelerate
from accelerate import Accelerator, PartialState
from accelerate.utils import gather
from more_itertools import chunked
from tqdm import tqdm
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EvalPrediction,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import CausalLMOutput
from transformers.utils import ModelOutput
from trl import SFTConfig, SFTTrainer
from typer import Option, Typer

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
DEBUG = bool(os.environ.get("DEBUG", "") != "")
typer.core.rich = None
app = Typer(pretty_exceptions_short=False, pretty_exceptions_show_locals=False)


@dataclass
class FormalAlignOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    predictions: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = None


def load_model(
    model_name: str | Path,
    max_length: int,
    lora_rank: int,
    gpu_memory_utilization: float,
    seed: int,
    unsloth: bool = False,
    adapter_name: str | None = None,
    debug: bool = False,
    quantize: bool = False,
    device_map: str | None = "auto",
):
    if unsloth and debug:
        raise NotImplementedError("Can't debug in unsloth mode because the model sizes are fixed.")
    if unsloth:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_name or model_name,
            max_seq_length=max_length,
            load_in_8bit=True,  # if we're using LoRA, load in 4 bit mode
            fast_inference=False,  # Enable vLLM fast inference
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
    elif debug:
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
            quantization_config=quantization_config,
            device_map=device_map,
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


def score_example(nl: str, fl: str, model, tokenizer):
    collator = CustomCollator(tokenizer=tokenizer, mask_inputs=False, mlm=False)
    if "Qwen" in tokenizer.name_or_path:
        chat_marker = tokenizer("<|im_start|>assistant", add_special_tokens=False).input_ids
    elif "Llama" in tokenizer.name_or_path:
        chat_marker = tokenizer("<|start_header_id|>assistant", add_special_tokens=False).input_ids
    else:
        raise NotImplementedError(f"tokenizer type {tokenizer.name_or_path} not supported for chat models yet")
    example = {"input": [nl], "output": [fl]}
    prompt = tokenize_chat(example, chat_marker, tokenizer)
    batch = collator([{key: val[0] for key, val in prompt.items()}])
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.to(model.device)

    batch["output_hidden_states"] = True
    model_out = model(**batch)
    scores = compute_formal_align_score(batch, model_out)
    scores["mean"] = (scores["certainty_score"] + scores["similarity_score"]) / 2
    return tokenizer.batch_decode(prompt["input_ids"]), scores


def compute_formal_align_score(inputs: dict, model_outputs: CausalLMOutput):
    # Last hidden state for similarity computation
    assert model_outputs.hidden_states is not None
    hidden_states = model_outputs.hidden_states[-1]

    # Get the index of the end of the prompt, so we can get the representation of the natural language
    # nl_index should be the " \n" token; to get start of FL you need to add 1
    nl_index = inputs["nl_end_idx"]
    fl_index = torch.sum(inputs["attention_mask"], dim=1) - 1 - 1  # -1 to zero index, then -1 to get := token

    # Create input mask to mask out the prompt & padding for certainty score computation
    B, N, V = model_outputs.logits.shape
    fl_mask = inputs["fl_mask"][:, 1:]
    log_probs = torch.log_softmax(model_outputs.logits, dim=-1)[:, :-1]  # predictions of ids 1->end
    idxs = inputs["input_ids"][:, 1:]
    token_log_probs = torch.gather(log_probs.reshape(-1, V), 1, idxs.reshape(-1, 1)).view(B, -1)
    fl_token_log_probs = token_log_probs * fl_mask  # zero out the ones we don't care about
    mean_log_probs = fl_token_log_probs.sum(dim=-1) / fl_mask.sum(dim=-1)
    certainty_score = torch.exp(mean_log_probs)

    nl_state = hidden_states[torch.arange(B), nl_index]
    fl_state = hidden_states[torch.arange(B), fl_index]

    # Do cosine similarity between pairs, and compare against the labels
    cos = torch.cosine_similarity(nl_state, fl_state, dim=-1)

    # Use sigmoid instead
    similarity_score = (cos + 1) / 2

    return dict(certainty_score=certainty_score, similarity_score=similarity_score)


class FastLanguageTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        B, N = inputs["input_ids"].shape

        # Cross entropy loss (autoformalization loss)
        inputs["output_hidden_states"] = True
        ce_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        ce_loss = ce_loss * inputs["aligned"]  # only want to count the positive examples
        ce_loss = ce_loss.sum() / inputs["aligned"].sum() if inputs["aligned"].sum() else 0.0

        scores = compute_formal_align_score(inputs, outputs)
        certainty_score = scores["certainty_score"]
        similarity_score = scores["similarity_score"]

        cl_loss = F.mse_loss(similarity_score, inputs["aligned"].to(similarity_score.dtype))

        # loss = cross entropy + contrastive loss
        loss = ce_loss + cl_loss
        self._metrics["ce_loss"].append(float(ce_loss))
        self._metrics["cl_loss"].append(cl_loss.item())

        new_outputs = FormalAlignOutput(
            loss=loss,
            logits=outputs.logits,  # type:ignore
            predictions=(certainty_score, similarity_score),
        )
        return (loss, new_outputs) if return_outputs else loss


def tokenize_chat(examples, marker: list[int], tokenizer: PreTrainedTokenizer):
    messages = []
    prompts = []
    nl_end_idx = []
    for inp, output in zip(examples["input"], examples["output"]):
        # Format input/output as a prompt
        format_user = f"Translate the following statement in natural language to Lean:\n{inp}"
        format_assistant = f"{output}"
        ex_messages = [
            {"role": "system", "content": "You are an expert mathematician."},
            {"role": "user", "content": format_user},
            {"role": "assistant", "content": format_assistant},
        ]
        messages.append(ex_messages)

    prompts = tokenizer.apply_chat_template(messages, tokenize=True)
    for prompt in prompts:
        assert isinstance(prompt, list)
        end_idx = [idx for idx, _ in enumerate(prompt) if prompt[idx : idx + len(marker)] == marker]
        nl_end_idx.append(end_idx[0] - 1)

    # Do tokenization here
    batch = {}

    batch["input_ids"] = prompts
    batch["attention_mask"] = [[1] * len(p) for p in prompts]
    batch["nl_end_idx"] = nl_end_idx
    batch["fl_start_idx"] = [nl + len(marker) + 2 for nl in nl_end_idx]
    # batch["text"] = tokenizer.batch_decode(prompts)
    if "label" in examples:
        batch["aligned"] = examples["label"]

    return batch


def tokenize(examples, marker: list[int], tokenizer: PreTrainedTokenizer):
    prompts = []
    for inp, output in zip(examples["input"], examples["output"]):
        # Format input/output as a prompt
        format_input = f"Translate the following statement in natural language to Lean:\n {inp}"
        format_output = f"\nLean statement:\n {output}"
        format_prompt = format_input + format_output + EOS
        prompts.append(format_prompt)

    # Do tokenization here
    batch = tokenizer(prompts, padding=False)
    nl_end_idx = []
    for prompt in batch.input_ids:
        start_index = [idx for idx, ids in enumerate(prompt) if prompt[idx : idx + len(marker)] == marker]
        input_len = start_index[0] - 1
        nl_end_idx.append(input_len)
    batch["nl_end_idx"] = nl_end_idx
    batch["fl_start_idx"] = [nl + len(marker) + 2 for nl in nl_end_idx]
    batch["text"] = prompts
    if "label" in examples:
        batch["aligned"] = examples["label"]

    return batch


def load_data(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int = 2048,
    use_chat_template: bool = False,
    subset: float = 1.0,
) -> DatasetDict:
    # Completion markers for chat models/non-chat models
    if "Qwen" in tokenizer.name_or_path:
        chat_marker = tokenizer("<|im_start|>assistant", add_special_tokens=False).input_ids
    elif "Llama" in tokenizer.name_or_path:
        chat_marker = tokenizer("<|start_header_id|>assistant", add_special_tokens=False).input_ids
    else:
        raise NotImplementedError(f"tokenizer type {tokenizer.name_or_path} not supported for chat models yet")
    marker = tokenizer("Lean statement", add_special_tokens=False).input_ids

    # Load the data and filter to a small subset if desired
    dataset: DatasetDict = load_dataset(dataset_name)  # type:ignore
    if subset < 1.0:
        dataset = DatasetDict({key: val.select(range(int(len(val) * subset))) for key, val in dataset.items()})

    # Tokenize using either chat or non-chat tokenization function
    if use_chat_template:
        dataset = dataset.map(lambda batch: tokenize_chat(batch, chat_marker, tokenizer), batched=True)
    else:
        dataset = dataset.map(lambda batch: tokenize(batch, marker, tokenizer), batched=True)
    dataset = dataset.filter(lambda ex: len(ex["input_ids"]) <= max_tokens and len(ex["input_ids"]) >= ex["nl_end_idx"])
    return dataset


class CustomCollator(DataCollatorForLanguageModeling):
    def __init__(self, *args, mask_inputs: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_inputs = mask_inputs

    def __call__(self, examples, *args, **kwargs):
        if "aligned" in examples[0]:
            labels = [ex.pop("aligned") for ex in examples]
        else:
            labels = None

        # Remove the stuff we don't need
        texts = [ex.pop("text") for ex in examples if "text" in ex]
        inputs = [ex.pop("input") for ex in examples if "input" in ex]
        outputs = [ex.pop("output") for ex in examples if "output" in ex]
        misalign_types = [ex.pop("misalign_type") for ex in examples if "misalign_type" in ex]

        batch = super().__call__(examples, *args, **kwargs)
        batch["nl_end_idx"] = torch.tensor([ex["nl_end_idx"] for ex in examples], dtype=torch.long)

        if self.mask_inputs:
            # Mask out the input part so the model only trains on completions
            for label, length in zip(batch["labels"], batch["nl_end_idx"]):
                label[:length] = -100

        # Mask out the input part
        fl_mask = torch.zeros_like(batch["input_ids"], dtype=torch.long)
        for i, length in enumerate(batch["fl_start_idx"]):
            fl_mask[i, length:] = 1
            padding_mask = batch["input_ids"][i] != self.tokenizer.pad_token_id
            fl_mask[i] = fl_mask[i] * padding_mask

        batch["fl_mask"] = fl_mask

        if labels is not None:
            batch["aligned"] = torch.tensor(labels, dtype=torch.long)
        return batch


def compute_metrics(evals: EvalPrediction):
    # FormalAlign recommends 0.7, but I think 0.5 is better for our model
    CUTOFF = 0.5

    (cert_score, sim_score), (_, labels), inputs, losses = evals
    scores = (cert_score + sim_score) / 2
    preds = (scores >= CUTOFF).astype(int)  # type:ignore
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    roc_auc = roc_auc_score(labels, scores)
    results = {}
    # print("=" * 100)
    for name, val in {"cert": cert_score, "sim": sim_score, "mean": scores}.items():
        score_p = [c for c, l in zip(val, labels) if l == 1]
        score_p_mean = sum(score_p) / len(score_p)
        score_n = [c for c, l in zip(val, labels) if l == 0]
        score_n_mean = sum(score_n) / len(score_n)
        # print(f"Positive {name} scores:  {score_p_mean}")
        # print(f"Negative {name} scores:  {score_n_mean}")
        results[f"{name}_score_pos"] = score_p_mean
        results[f"{name}_score_neg"] = score_n_mean
    # print("=" * 100)
    # get scores of each type
    return {"precision": p, "recall": r, "f1": f1, "roc_auc": roc_auc, **results}


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits, scores = logits
        return scores
    return logits


@app.command()
def train(
    # fmt:off
    model_name: Annotated[str, Option(help="path to model to train", rich_help_panel="Model Config")],
    dataset: Annotated[str, Option(help="path to datasets to train", rich_help_panel="Data Config")],
    eval_dataset: Annotated[str, Option(help="path to datasets to eval", rich_help_panel="Data Config")],
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
    negative_sample_ratio: Annotated[float, Option(help="% of positive dataset to sample as negative examples", rich_help_panel="Training Config")] = 1.0,
    gradient_accumulation: Annotated[int, Option(help="gradient accumulation", rich_help_panel="Training Config")] = 1,
    gradient_checkpointing: Annotated[bool, Option("--gradient-checkpointing", help="enable gradient checkpointing", rich_help_panel="Training Config")] = False,
    eval_steps: Annotated[int, Option(help="eval steps", rich_help_panel="Training Config")] = 500,
    unsloth: Annotated[bool, Option("--unsloth", help="enable unsloth", rich_help_panel="Training Config")] = False,
    mask_inputs: Annotated[bool, Option("--mask-inputs", help="train on completions only", rich_help_panel="Training Config")] = False,
    debug: Annotated[bool, Option("--debug", help="enable debug mode", rich_help_panel="Training Config")] = False,
    # fmt:on
):
    model, tokenizer = load_model(
        model_name,
        max_length=max_tokens,
        lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        unsloth=unsloth,
        debug=debug,
    )

    training_args = SFTConfig(
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.0,
        warmup_ratio=0.03,
        label_smoothing_factor=0.0,
        lr_scheduler_type=scheduler,
        optim=optimizer,
        logging_steps=20,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,  # Increase to 4 for smoother training
        gradient_checkpointing=gradient_checkpointing,
        num_train_epochs=num_epochs,  # Set to 1 for a full training run
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_minif2f_test_f1",
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        report_to="wandb",  # Can use Weights & Biases
        output_dir=output_dir,
        max_seq_length=max_tokens,
        remove_unused_columns=False,
        include_for_metrics=["loss", "inputs"],
        label_names=["label", "aligned"],
    )
    collator = CustomCollator(tokenizer=tokenizer, mask_inputs=mask_inputs, mlm=False)
    is_chat_model = "instruct" in model_name.lower()
    data = load_data(dataset, tokenizer, max_tokens=max_tokens, use_chat_template=is_chat_model)
    data = data.shuffle(seed=seed)
    positives = data.filter(lambda ex: ex["aligned"] == 1)
    negatives = data.filter(lambda ex: ex["aligned"] == 0)
    print("Positive dataset: ", positives)
    print("Negatives dataset: ", negatives)

    # Only select some of the negative examples to make training more balanaced
    num_neg_examples = int(negative_sample_ratio * len(positives["train"]))
    train_data = concatenate_datasets([positives["train"], negatives["train"].select(range(num_neg_examples))])
    train_data = train_data.shuffle(seed=seed)

    test_data = load_data(eval_dataset, tokenizer, max_tokens=max_tokens, use_chat_template=is_chat_model)
    test_data = test_data.shuffle(seed=seed)
    eval_data = {key: val.select(range(200)) for key, val in test_data.items()}
    trainer = FastLanguageTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        data_collator=collator,
        train_dataset=train_data,
        compute_metrics=compute_metrics,
        eval_dataset=eval_data,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
    for split in test_data.keys():
        split_data = test_data[split].shuffle(seed).select(range(1000))
        outputs = trainer.predict(split_data, metric_key_prefix=split)
        (cert_score, sim_score), (_, labels), metrics = outputs  # type:ignore
        pprint(metrics)
        with open(Path(output_dir, f"{split}_metrics.json"), "w") as f:
            json.dump(metrics, f)
        with open(Path(output_dir, f"{split}_outputs.pkl"), "wb") as f:
            pickle.dump({"label": labels, "cert_score": cert_score, "sim_score": sim_score}, f)
        df = pd.DataFrame({"label": labels, "cert_score": cert_score, "sim_score": sim_score})
        df.to_json(Path(output_dir, f"{split}_preds.json"))


@app.command()
def test(
    # fmt:off
    model_name: Annotated[str, Option(help="path to model to test", rich_help_panel="Model Config")],
    dataset: Annotated[str, Option(help="path to datasets to test", rich_help_panel="Data Config")],
    output_dir: Annotated[str, Option(help="path to output directory", rich_help_panel="Data Config")],
    adapter_name: Annotated[str, Option(help="path to adapter to test", rich_help_panel="Model Config")] = None,
    max_tokens: Annotated[int, Option(help="max tokens", rich_help_panel="Training Config")] = 2048,
    gpu_memory_utilization: Annotated[float, Option(help="percent of GPU to give to unsloth", rich_help_panel="Training Config")] = 0.6,
    seed: Annotated[int, Option(help="random seed", rich_help_panel="Training Config")] = 1234,
    batch_size: Annotated[int, Option(help="batch size", rich_help_panel="Training Config")] = 4,
    unsloth: Annotated[bool, Option("--unsloth", help="enable unsloth", rich_help_panel="Training Config")] = False,
    # fmt:on
):
    model, tokenizer = load_model(
        model_name,
        max_length=max_tokens,
        lora_rank=-1,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        unsloth=unsloth,
        adapter_name=adapter_name,
    )

    training_args = SFTConfig(
        logging_steps=10,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        report_to="wandb",  # Can use Weights & Biases
        output_dir=output_dir,
        max_seq_length=max_tokens,
        remove_unused_columns=False,
        include_for_metrics=["loss", "inputs"],
        label_names=["label", "aligned"],
    )
    collator = CustomCollator(tokenizer=tokenizer, mlm=False)
    is_chat_model = "instruct" in model_name.lower()
    test_data = load_data(dataset, tokenizer, max_tokens=max_tokens, use_chat_template=is_chat_model)
    trainer = FastLanguageTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        data_collator=collator,
        compute_metrics=compute_metrics,
        train_dataset=test_data[list(test_data.keys())[0]],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    for split in test_data.keys():
        split_data = test_data[split].shuffle(seed).select(range(1000))
        outputs = trainer.predict(split_data, metric_key_prefix=split)
        (cert_score, sim_score), (_, labels), metrics = outputs  # type:ignore
        pprint(metrics)
        with open(Path(output_dir, f"{split}_metrics.json"), "w") as f:
            json.dump(metrics, f)
        with open(Path(output_dir, f"{split}_outputs.pkl"), "wb") as f:
            pickle.dump({"label": labels, "cert_score": cert_score, "sim_score": sim_score}, f)
        df = pd.DataFrame({"label": labels, "cert_score": cert_score, "sim_score": sim_score})
        df.to_json(Path(output_dir, f"{split}_preds.json"))


@app.command()
def predict_herald(
    model_name: Annotated[str, Option(help="path to model to test", rich_help_panel="Model Config")],
    dataset: Annotated[str, Option(help="path to datasets to test", rich_help_panel="Data Config")],
    output_json: Annotated[str, Option(help="path to output json", rich_help_panel="Data Config")],
    adapter_name: Annotated[str, Option(help="path to adapter to test", rich_help_panel="Model Config")] = None,
    max_tokens: Annotated[int, Option(help="max tokens", rich_help_panel="Training Config")] = 2048,
    gpu_memory_utilization: Annotated[
        float, Option(help="percent of GPU to give to unsloth", rich_help_panel="Training Config")
    ] = 0.6,
    seed: Annotated[int, Option(help="random seed", rich_help_panel="Training Config")] = 1234,
    batch_size: Annotated[int, Option(help="batch size", rich_help_panel="Training Config")] = 4,
    num_samples: Annotated[int, Option(help="num samples", rich_help_panel="Training Config")] = -1,
):
    model, tokenizer = load_model(
        model_name,
        max_length=max_tokens,
        lora_rank=-1,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        unsloth=False,
        adapter_name=adapter_name,
        device_map=None,
    )

    # Everything here is herald specific stuff
    if dataset.endswith(".json"):
        df = pd.read_json(dataset)
    else:
        df = load_from_disk(dataset).to_pandas()

    df = df[["informal_statement", "formal_statement", "name"]]
    if num_samples > 0:
        df = df[:num_samples]

    def split_off_name(text):
        splitted = text.split("**", maxsplit=2)
        if len(splitted) == 3:
            _, name, theorem = splitted
        elif len(splitted) == 2:
            _, theorem = splitted
            name = "Theorem"
        else:
            name = None
            theorem = None
        return {"id": name, "informal_statement": theorem}

    def format_example(example):
        nl = split_off_name(example["informal_statement"])["informal_statement"].strip()
        nl = nl.replace(r"\(", "$").replace(r"\)", "$")  # convert from \( to $
        nl = nl.lstrip(string.punctuation)

        statements = example["formal_statement"]
        if not isinstance(statements, list):
            statements = [statements]
        examples = []
        for stm in statements:
            fl = stm.split("-/")[-1].split("sorry")[0].strip()
            examples.append({"index": example["name"], "input": nl, "output": fl})
        return examples

    df["examples"] = df.apply(lambda row: format_example(row), axis=1)
    df = df.explode(["examples", "formal_statement"]).reset_index(names=["group_index"])
    cols = pd.json_normalize(df["examples"])
    df = pd.merge(left=df, right=cols, left_index=True, right_index=True, how="right")
    df = df.drop("examples", axis=1)

    chat_marker = tokenizer("<|im_start|>assistant", add_special_tokens=False).input_ids
    collator = CustomCollator(tokenizer, mlm=False)

    hf_dataset = Dataset.from_list(df.to_dict(orient="records"))
    hf_dataset = hf_dataset.map(
        lambda batch: tokenize_chat(batch, chat_marker, tokenizer), batched=True, remove_columns=[*df.columns]
    )

    trainer = FastLanguageTrainer(
        model=model,
        processing_class=tokenizer,
        args=SFTConfig(
            logging_steps=10,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            per_device_eval_batch_size=batch_size,
            report_to="wandb",  # Can use Weights & Biases
            output_dir=str(Path(output_json).parent),
            max_seq_length=max_tokens,
            remove_unused_columns=False,
            include_for_metrics=["loss", "inputs"],
            label_names=["label", "aligned"],
        ),
        data_collator=collator,
        train_dataset=hf_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    certs = []
    sims = []
    model = model.eval()
    accelerator = Accelerator()
    with accelerator.split_between_processes(hf_dataset.to_list()) as inputs:
        dataloader = trainer.get_test_dataloader(inputs)
        model, dataloader = accelerator.prepare(model, dataloader)
        pbar = tqdm(dataloader, total=math.ceil(len(inputs) / batch_size), position=accelerator.process_index)
        for batch in pbar:
            with torch.no_grad():
                model_outputs = model(**batch, output_hidden_states=True)
                scores = compute_formal_align_score(batch, model_outputs)
                certs.extend(scores["certainty_score"].cpu())
                sims.extend(scores["similarity_score"].cpu())

    # Accelerate gather
    certs = [certs]
    sims = [sims]
    certs_gathered = gather(certs)
    sims_gathered = gather(sims)

    certs_flat = [c.item() for cert in certs_gathered for c in cert]
    sims_flat = [c.item() for sim in sims_gathered for c in sim]

    df["certainty_score"] = certs_flat
    df["similarity_score"] = sims_flat
    df["score"] = (df["certainty_score"] + df["similarity_score"]) / 2
    df["aligned"] = df["score"] > 0.5

    df = df.groupby("informal_statement").agg(list).reset_index(names=["informal_statement"])
    df = df[
        ["informal_statement", "name", "formal_statement", "certainty_score", "similarity_score", "score", "aligned"]
    ]

    df.to_json(output_json)


if __name__ == "__main__":
    app()
