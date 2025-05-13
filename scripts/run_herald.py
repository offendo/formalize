import os
import sys
import pandas as pd
import torch
import re
from pathlib import Path
from more_itertools import chunked
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from argparse import ArgumentParser


def chat_template_to_prompt(messages: list[dict], model: str) -> str:
    """
    Chat template for deepseek and internlm
    """
    result = ""
    total_step = len(messages)
    for i, message in enumerate(messages):
        if model == "internlm":
            result += "<|im_start|>" + message["role"] + "\n" + message["content"]
            if i + 1 != total_step:
                result += "<|im_end|>\n"
            elif message["role"] == "user":
                result += "<|im_end|>\n<|im_start|>assistant\n"

        elif model == "deepseek":
            if message["role"] == "user":
                result += "User:" + message["content"] + "\n\n"
            elif message["role"] == "assistant":
                result += "Assistant" + message["content"] + "<｜end▁of▁sentence｜>"
            elif message["role"] == "system":
                result += message["content"] + "\n\n"
            if i + 1 == total_step and message["role"] == "user":
                result += "Assistant:"
        else:
            raise NotImplementedError
    return result


def herald_format_prompt(informal_name, informal_statement, tokenizer=None):
    template = "Please translate the natural language statement to Lean4 code with the header\n**Name**\n{informal_name}\n**Informal statement**\n{informal_statement}\n"
    msgs = [
        {"role": "system", "content": "You are an expert at Lean 4 and Mathematics."},
        {
            "role": "user",
            "content": template.format(informal_name=informal_name, informal_statement=informal_statement),
        },
    ]
    return chat_template_to_prompt(msgs, "deepseek")


def kimina_format_prompt(informal_name, informal_statement, tokenizer):
    prompt = "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n"
    prompt += informal_statement

    messages = [
        {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return text

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
    return {"name": name, "informal_statement": text}


def format_example(example: dict):
    name = example["name"]
    nl = example["informal_statement"]
    fl = example["formal_statement"]
    system = "You are an expert at Lean 4 and Mathematics."
    instruction = f"Please translate the natural language statement to Lean4 code with the header\n**Name**\n{name}\n**Informal statement**\n{nl}\n"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": fl},
    ]
    return {"conversation": messages}


def format_reverse_example(example: dict):
    name = example["name"]
    nl = example["informal_statement"]
    # Need to get the formal language but without the comment
    fl = "".join(re.split(r"\n(theorem|lemma)", example["formal_statement"], re.MULTILINE)[-2:])
    system = "You are an expert at Lean 4 and Mathematics."
    instruction = (
        f"Please translate the formal language statement in Lean 4 into natural language\n**Formal statement**\n{fl}\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": nl},
    ]
    return {"conversation": messages}


if __name__ == "__main__":

    parser = ArgumentParser("inference")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=['kimina', 'herald'])
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--top_p", type=float, default=0.95)

    args = parser.parse_args()

    # Save previous output if needed
    # ==============================
    if Path(args.output_path).exists():
        Path(args.output_path).rename(args.output_path + ".old-backup")
        print(f"Output path {args.output_path} already exists - moving to backup")

    # Load the model/tokenizer
    # ========================
    llm = LLM(
        args.model,
        dtype=torch.bfloat16,
        tensor_parallel_size=torch.cuda.device_count(),
        task="generate",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Format the dataset
    # ==================
    ds = load_dataset(args.dataset, split="train")
    if args.num_samples > 0:
        ds = ds.take(args.num_samples)

    ds = ds.map(lambda x: split_off_name(x["informal_statement"]), batched=False)
    ds = ds.filter(lambda ex: ex["name"] != None)

    batch = [dict(id=ex["name"], informal_statement=ex["informal_statement"]) for ex in ds]
    match args.model_type:
        case "herald":
            prompts = [herald_format_prompt(i["id"], i["informal_statement"]) for i in batch]
        case "kimina":
            prompts = [kimina_format_prompt(i["id"], i["informal_statement"], tokenizer) for i in batch]
        case _:
            raise ValueError(f"We don't support model type {args.model_type}")

    # Do the generation
    # =================
    params = SamplingParams(
        n=args.generations,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )
    outputs = llm.generate(prompts, sampling_params=params)  # type: ignore
    all_outputs = [[o.text for o in output.outputs] for output in outputs]

    # Create a new dataset with the new outputs
    # =========================================
    new_ds = pd.DataFrame.from_records(
        [
            dict(
                informal_statement=ex["informal_statement"],
                formal_statement=all_outputs[idx],
                name=ex["name"],
            )
            for idx, ex in enumerate(ds)
        ]
    )

    # Save to disk
    new_ds.to_json(Path(args.output_path).with_suffix(".json"))
