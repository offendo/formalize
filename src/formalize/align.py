#!/usr/bin/env python3
import os
import json
import pickle
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Annotated, Optional
from typer import Argument, Option, run as typer_run, Typer
import typer

from transformers import (
    DataCollator,
    DataCollatorForLanguageModeling,
    DefaultDataCollator,
    EvalPrediction,
    LlamaForCausalLM,
    PreTrainedTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
)
from transformers.utils import ModelOutput
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from datasets import concatenate_datasets, load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
DEBUG = bool(os.environ.get("DEBUG", "") != "")
typer.core.rich = None
app = Typer(pretty_exceptions_short=False, pretty_exceptions_show_locals=False)


def cosine_similarity_matrix(matrix1, matrix2):
    """
    Computes the cosine similarity between all pairs of rows in two matrices.

    Args:
      matrix1: A PyTorch tensor of shape (N, D).
      matrix2: A PyTorch tensor of shape (M, D).

    Returns:
      A PyTorch tensor of shape (N, M) where each element (i, j) is the cosine
      similarity between row i of matrix1 and row j of matrix2.
    """

    # Normalize rows to unit length
    matrix1_normalized = F.normalize(matrix1, p=2, dim=1)
    matrix2_normalized = F.normalize(matrix2, p=2, dim=1)

    # Compute cosine similarity using matrix multiplication
    similarity_matrix = torch.matmul(matrix1_normalized, matrix2_normalized.transpose(0, 1))

    return similarity_matrix


@dataclass
class FormalAlignOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    predictions: torch.FloatTensor = None


def load_model(
    model_name: str,
    max_length: int,
    lora_rank: int,
    gpu_memory_utilization: float,
    seed: int,
    unsloth: bool = False,
    adapter_name: str | None = None,
    debug: bool = False,
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
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=lora_rank != -1,  # if we're using LoRA, load in 4 bit mode
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
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


class FastLanguageTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        B, N = inputs["input_ids"].shape

        # Cross entropy loss (autoformalization loss)
        inputs["output_hidden_states"] = True
        ce_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        ce_loss = ce_loss * inputs["aligned"]  # only want to count the positive examples
        ce_loss = ce_loss.sum() / inputs["aligned"].sum() if inputs["aligned"].sum() else 0.0

        # Last hidden state for contrastive loss
        hidden_states = outputs.hidden_states[-1]  # type:ignore

        if "nl_token_index" in inputs:
            nl_index = inputs["nl_token_index"]
            fl_index = inputs["fl_token_index"]
        else:
            # Get the index of the end of the prompt, so we can get the representation of the natural language
            # nl_index should be the " \n" token; to get start of FL you need to add 1
            nl_index = inputs["input_length"]
            fl_index = torch.sum(inputs["attention_mask"], dim=1) - 1 - 1  # -1 to zero index, then -1 to get := token

        # Create input mask to mask out the prompt & padding for certainty score computation
        B, N, V = outputs.logits.shape
        input_mask = inputs["input_mask"][:, 1:]
        log_probs = torch.log_softmax(outputs.logits, dim=-1)[:, :-1]
        idxs = inputs['input_ids'][:, 1:]
        token_log_probs = torch.gather(log_probs.reshape(-1, V), 1, idxs.reshape(-1, 1)).view(B, -1)
        fl_token_log_probs = token_log_probs * input_mask
        mean_log_probs = fl_token_log_probs.sum(dim=-1) / input_mask.sum(dim=-1)
        certainty_score = torch.exp(mean_log_probs)

        # Directly optimize the certainty score to match the aligned labels
        cert_loss = F.mse_loss(certainty_score, inputs['aligned'].float())

        # certainty_score = torch.zeros(B, dtype=outputs.logits.dtype, device=outputs.logits.device)
        # for i, (sequence, logits, start, stop) in enumerate(
        #     zip(inputs["input_ids"], outputs.logits, nl_index + 2, fl_index)
        # ):
        #     log_probs = torch.log_softmax(logits[start:stop], dim=-1)
        #     token_probs = log_probs[torch.arange(len(log_probs)), sequence[start + 1 : stop + 1]]
        #     certainty_score[i] = torch.exp(torch.mean(token_probs, dim=-1))

        nl_state = hidden_states[torch.arange(B), nl_index]
        fl_state = hidden_states[torch.arange(B), fl_index]

        # Do cosine similarity between pairs, and compare against the labels
        cos = torch.cosine_similarity(nl_state, fl_state, dim=-1)

        # Use sigmoid instead
        similarity_score = cos
        cl_loss = F.mse_loss((cos + 1) / 2, inputs["aligned"].float())

        # loss = cross entropy + contrastive loss
        loss = ce_loss + cl_loss + cert_loss
        self._metrics["ce_loss"].append(float(ce_loss))
        self._metrics["cert_loss"].append(float(cert_loss))
        self._metrics["cl_loss"].append(cl_loss.item())

        new_outputs = FormalAlignOutput(
            loss=loss, logits=outputs.logits, predictions=(certainty_score, similarity_score)
        )
        return (loss, new_outputs) if return_outputs else loss


def load_data(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int = 2048,
    add_special_representation: bool = False,
) -> DatasetDict:
    EOS: str = tokenizer.eos_token  # type:ignore
    END_OF_NL = ""
    END_OF_FL = ""
    END_OF_NL_ID = None
    END_OF_FL_ID = None
    if add_special_representation:
        END_OF_NL = "<|end_of_nl|>"
        END_OF_FL = "<|end_of_fl|>"
        tokenizer.add_tokens([END_OF_NL, END_OF_FL])
        END_OF_NL_ID = tokenizer.convert_tokens_to_ids(END_OF_NL)
        END_OF_FL_ID = tokenizer.convert_tokens_to_ids(END_OF_FL)

    def tokenize(examples):
        prompts = []
        input_lengths = []
        for input, output in zip(examples["input"], examples["output"]):
            # Format input/output as a prompt
            format_input = f"Statement in natural language:\n{input} \n"
            format_output = f"Translate the statement in natural language to Lean:\n{output}"
            format_prompt = format_input + END_OF_NL + format_output + END_OF_FL + EOS
            prompts.append(format_prompt)

            input_len = len(tokenizer(format_input, add_special_tokens=False).input_ids)
            input_lengths.append(input_len)

        # Do tokenization here
        batch = tokenizer(prompts, padding=False)
        batch["input_length"] = input_lengths
        batch["text"] = prompts
        if "label" in examples:
            batch["aligned"] = examples["label"]

        if add_special_representation:
            batch["nl_token_index"] = [example.index(END_OF_NL_ID) for example in batch["input_ids"]]
            batch["fl_token_index"] = [example.index(END_OF_FL_ID) for example in batch["input_ids"]]

        return batch

    dataset: DatasetDict = load_dataset(dataset_name)  # type:ignore
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.filter(
        lambda ex: len(ex["input_ids"]) <= max_tokens and len(ex["input_ids"]) >= ex["input_length"]
    )
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
        texts = [ex.pop("text") for ex in examples]
        inputs = [ex.pop("input") for ex in examples]
        outputs = [ex.pop("output") for ex in examples]
        misalign_types = [ex.pop("misalign_type") if "misalign_type" in ex else None for ex in examples]

        batch = super().__call__(examples, *args, **kwargs)
        batch["input_length"] = torch.tensor([ex["input_length"] for ex in examples], dtype=torch.long)

        if self.mask_inputs:
            # Mask out the input part so the model only trains on completions
            for label, length in zip(batch["labels"], batch["input_length"]):
                label[:length] = -100

        # Mask out the input part
        input_mask = torch.zeros_like(batch['input_ids'], dtype=torch.long)
        for i, length in enumerate(batch["input_length"]):
            input_mask[i, length:] = 1
            padding_mask = batch['input_ids'][i] != self.tokenizer.pad_token_id
            input_mask[i] = input_mask[i] * padding_mask

        batch['input_mask'] = input_mask

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
    add_special_representation: Annotated[bool, Option("--add-special-representation", help="add <|end_of_nl/fl|> tokens", rich_help_panel="Training Config")] = False,
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
    data = load_data(dataset, tokenizer, max_tokens=max_tokens)
    data = data.shuffle(seed=seed)
    positives = data.filter(lambda ex: ex["aligned"] == 1)
    negatives = data.filter(lambda ex: ex["aligned"] == 0)
    print("Positive dataset: ", positives)
    print("Negatives dataset: ", negatives)

    # Only select some of the negative examples to make training more balanaced
    num_neg_examples = int(negative_sample_ratio * len(positives["train"]))
    train_data = concatenate_datasets([positives["train"], negatives["train"].select(range(num_neg_examples))])
    train_data = train_data.shuffle(seed=seed)

    test_data = load_data(eval_dataset, tokenizer, max_tokens=max_tokens)
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
    test_data = load_data(dataset, tokenizer, max_tokens=max_tokens)
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


if __name__ == "__main__":
    app()
