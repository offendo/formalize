import os
import sys
import pandas as pd
import torch
import re
from pathlib import Path
from more_itertools import chunked
from datasets import load_dataset, load_from_disk, Dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm


if not os.path.exists(os.getcwd() + "/herald_translator"):
    os.system("git clone https://github.com/frenzymath/herald_translator.git")

sys.path.append(os.getcwd() + "/herald_translator")
from worker.translator import Translator
from argparse import ArgumentParser


def format_example(example: dict):
    name = example["informal_name"]
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
    name = example["informal_name"]
    nl = example["informal_statement"]
    # Need to get the formal language but without the comment
    fl = ''.join(re.split(r"\n(theorem|lemma)", example["formal_statement"], re.MULTILINE)[-2:])
    system = "You are an expert at Lean 4 and Mathematics."
    instruction = f"Please translate the formal language statement in Lean 4 into natural language\n**Formal statement**\n{fl}\n"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": nl},
    ]
    return {"conversation": messages}

if __name__ == "__main__":

    parser = ArgumentParser("inference")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)

    args = parser.parse_args()

    if Path(args.output_path).exists():
        raise ValueError(f"Output path {args.output_path} already exists. Need a new directory.")

    translator = Translator(args.model, gpus=torch.cuda.device_count())
    llm = LLM(
        args.model,
        dtype=torch.bfloat16,
        tensor_parallel_size=torch.cuda.device_count(),
        task="generate",
        trust_remote_code=True,
    )
    translator.model = llm

    if Path(args.dataset).exists():
        ds = load_from_disk(args.dataset)
    else:
        ds = load_dataset(args.dataset, split="train")
    if args.num_samples:
        ds = ds.take(args.num_samples)

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
        return {"informal_name": name, "informal_statement": theorem}

    ds = ds.map(lambda x: split_off_name(x["informal_statement"]), batched=False)
    ds = ds.filter(lambda ex: ex["informal_name"] != None)

    batch = [dict(id=ex["informal_name"], informal_statement=ex["informal_statement"]) for ex in ds]
    out = translator.batch_generate(
        batch, sampling_params=dict(temperature=args.temperature, max_tokens=args.max_tokens)
    )
    all_outputs = [ex[0] for ex in out]

    # Create a new dataset with the new outputs
    new_ds = Dataset.from_list(
        [
            dict(
                informal_statement=ex["informal_statement"],
                formal_satement=all_outputs[idx],
                name=ex["name"],
            )
            for idx, ex in enumerate(ds)
        ]
    )

    # Add the `conversation` key
    new_ds = new_ds.map(format_example, batched=False)

    # Save to disk
    new_ds.save_to_disk(args.output_path)
