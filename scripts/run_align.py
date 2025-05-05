#!/usr/bin/env python3

import os
import sys
import pandas as pd
import torch
import string
from more_itertools import chunked
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

from argparse import ArgumentParser
from src.formalize.align import CustomCollator, compute_formal_align_score, tokenize_chat

if __name__ == "__main__":

    parser = ArgumentParser("inference")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--output_dataset_iteration", type=str, required=False, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)

    args = parser.parse_args()

    llm = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    llm = llm.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ds = pd.read_json(args.dataset)
    if args.num_samples:
        ds = ds[: args.num_samples]

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
        nl = split_off_name(example["text"])["informal_statement"].strip()
        nl = nl.replace(r"\(", "$").replace(r"\)", "$")  # convert from \( to $
        nl = nl.lstrip(string.punctuation)
        fl = example["formal"].split("-/")[-1].split("sorry")[0].strip()
        return {"index": example.name, "input": nl, "output": fl}

    thms = ds.apply(lambda row: format_example(row), axis=1)
    ds = pd.merge(left=ds, right=pd.json_normalize(thms), left_index=True, right_on="index", how="left")

    chat_marker = tokenizer("<|im_start|>assistant", add_special_tokens=False).input_ids
    collator = CustomCollator(tokenizer, mlm=False)

    certs = []
    sims = []
    for batch in chunked(tqdm(ds.to_dict(orient="records")), args.batch_size):
        size = len(batch)
        batch = {key: [batch[i][key] for i in range(size)] for key in batch[0].keys()}
        batch = tokenize_chat(batch, chat_marker, tokenizer)
        batch = [{key: val[i] for key, val in batch.items()} for i in range(size)]
        batch = collator(batch)
        batch = {key: val.to(llm.device) for key, val in batch.items()}
        with torch.no_grad():
            model_outputs = llm(**batch, output_hidden_states=True)
            scores = compute_formal_align_score(batch, model_outputs)
            certs.extend(scores["certainty_score"].tolist())
            sims.extend(scores["similarity_score"].tolist())

    ds["certainty_score"] = certs
    ds["similarity_score"] = sims

    ds.to_json(args.output_path)
