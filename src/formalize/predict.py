import pandas as pd
import sys
from pathlib import Path
from argparse import ArgumentParser
from openai import OpenAI
from datasets import load_from_disk, Dataset

from formalize.mathatlas import MathAtlas

SYSTEM = "You are an expert mathematician who is fluent in Lean 4."

PROMPT = """Given a mathematical {tag}, formalize it into Lean 4.

-- {tag} to formalize
{text}"""

PROMPT_WITH_REFS = """Given a mathematical {tag}, formalize it into Lean 4. You may use definitions found in Mathlib. \
You are also given the {tag}'s name(s) and a list of referenced objects.

-- Names
{names}

-- References
{refs}

--- {tag} to formalize
{text}"""


def find_refs(row: dict, database: MathAtlas):
    refs = []
    for idn, tag, links in zip(row['text'], row['tag'], row['links']):
        if tag != 'reference':
            continue
        if len(links) == 0:
            refs.append((idn, None, None))
            continue
        name = database.pandas.loc[links[0]['file_id'], links[0]['start'], links[0]['end'], links[0]['tag']]
        target = name.parent_text.iloc[0]
        target_tag = name.parent_tag.iloc[0]
        refs.append((idn, target_tag, target))
    return refs

def find_names(row: dict):
    return [idn for idn, tag in zip(row['text'], row['tag']) if tag == 'name']

def format_example(example: dict, database: MathAtlas, use_references: bool = True):
    text = example['parent_text'][0]
    if use_references:
        refs = find_refs(example, database)
        ref_text = '\n\n'.join([f"# ID: {ref}\n# Type: {reftag}\n# Text: {reftext}" for ref, reftag, reftext in refs])
        name_text = ', '.join(find_names(example))
        prompt = PROMPT_WITH_REFS.format(tag=example['tag'], names=name_text, text=text, refs=ref_text)
    else:
        prompt = PROMPT.format(tag=example['parent_tag'], text=text)

    return prompt

def format_conversation(prompt: str, system: str):
    return [
        dict(role="system", content=system),
        dict(role="user", content=prompt),
    ]

if __name__ == "__main__":
    parser = ArgumentParser('formalize')

    subparser = parser.add_subparsers(help='command', dest='command')

    process = subparser.add_parser('process_data')
    process.add_argument('--json', '-j', type=str, help='path to MathAtlas json')
    process.add_argument('--save_dataset', '-s', type=str, default="./mathatlas/", help='path where to save dataset')

    formalize = subparser.add_parser('formalize')
    formalize.add_argument('--dataset', '-d', type=str, help='path to dataset')
    formalize.add_argument('--output', '-o', type=str, help='path to output json')
    formalize.add_argument('--model', '-m', type=str, help='name of model')
    formalize.add_argument('--num_examples', '-n', type=int, default=None, help='number of examples to process')
    formalize.add_argument('--shuffle', action='store_true', help='shuffle the dataset')
    formalize.add_argument('--system', '-s', type=str, default=SYSTEM, help='system instruction')

    args = parser.parse_args()

    match args.command:
        case 'process_data':
            database = MathAtlas.from_mathatlas(args.json)
            # Save the dataset with refs
            with_refs = database.map(lambda ex: {'example': format_example(ex, database, use_references=True)}, batched=False)
            with_refs.save_to_disk(Path(args.save_dataset, 'with_refs'))

            # Save the dataset without refs
            no_refs = database.map(lambda ex: {'example': format_example(ex, database, use_references=False)}, batched=False)
            no_refs.save_to_disk(Path(args.save_dataset, 'no_refs'))
        case 'formalize':
            from vllm import LLM, SamplingParams
            
            # Load the dataset, and grab just the prompts
            database = MathAtlas.load_from_disk(args.dataset).select(range(args.num_examples or sys.maxsize))
            if args.shuffle:
                database = database.shuffle(seed=1234)
            prompts = database['prompt']
            completions = []

            # Enable deterministic sampling
            sampling_params = SamplingParams(temperature=0.0, top_p=1)

            llm = LLM(model=args.model)
            outputs = llm.generate(prompts, sampling_params)
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                completions.append(generated_text)

            df = pd.DataFrame.from_records(list(zip(prompts, completions)), columns=['prompt', 'completion'])
            df.to_json(args.output)
