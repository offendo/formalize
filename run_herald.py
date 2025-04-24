import os
import sys
import pandas as pd
import torch
from more_itertools import chunked
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm


if not os.path.exists(os.getcwd() + "/herald_translator"):
    os.system("git clone https://github.com/frenzymath/herald_translator.git")

sys.path.append(os.getcwd() + "/herald_translator")
from worker.translator import Translator
from argparse import ArgumentParser

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

    translator = Translator("FrenzyMath/Herald_translator", gpus=torch.cuda.device_count())
    llm = LLM(
        "FrenzyMath/Herald_translator",
        dtype=torch.bfloat16,
        tensor_parallel_size=torch.cuda.device_count(),
        task="generate",
        trust_remote_code=True,
    )
    translator.model = llm

    ds = load_dataset("offendo/math-atlas-titled-theorems", split="train")
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
        return {"id": name, "informal_statement": theorem}

    ds = ds.map(lambda x: split_off_name(x["text"]), batched=False)
    ds = ds.filter(lambda ex: ex["id"] != None)

    batch = [dict(id=ex["id"], informal_statement=ex["informal_statement"]) for ex in ds]
    out = translator.batch_generate( batch, sampling_params=dict(temperature=args.temperature, max_tokens=args.max_tokens))
    all_outputs = [ex[0] for ex in out]

    df = pd.DataFrame({"text": ds["text"], "name": ds["id"], "formal": all_outputs})
    df.to_json(args.output_path)
