import pandas as pd

from argparse import ArgumentParser
from openai import OpenAI
from datasets import load_dataset, load_from_disk, Dataset

from src.formalize.datasets import MathAtlas

# client = OpenAI( base_url="http://localhost:8000/v1", api_key="")
# 
# completion = client.chat.completions.create(
#   model="NousResearch/Meta-Llama-3-8B-Instruct",
#   messages=[
#     {"role": "user", "content": "Hello!"}
#   ]
# )

# print(completion.choices[0].message)

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

def format_prompt(example: dict, database: MathAtlas, use_references: bool = True):
    text = example['parent_text'][0]
    if use_references:
        refs = find_refs(example, database)
        ref_text = '\n\n'.join([f"# ID: {ref}\n# Type: {reftag}\n# Text: {reftext}" for ref, reftag, reftext in refs])
        name_text = ', '.join(find_names(example))
        prompt = PROMPT_WITH_REFS.format(tag=example['tag'], names=name_text, text=text, refs=ref_text)
    else:
        prompt = PROMPT.format(tag=example['parent_tag'], text=text)

    return prompt

if __name__ == "__main__":
    parser = ArgumentParser('formalize')

    parser.add_argument('--data', '-d', type=str, help='path to dataset')
    parser.add_argument('--output', '-o', type=str, help='path to output json')

    args = parser.parse_args()

    if args.data.endswith('.json'):
        database = MathAtlas.from_mathatlas(args.data)
    else:
        database = load_from_disk(args.data)

    with_refs = database.map(lambda ex: format_prompt(ex, database, use_references=True), batched=False)
    no_refs = database.map(lambda ex: format_prompt(ex, database, use_references=False), batched=False)
    
