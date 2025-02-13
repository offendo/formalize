import pandas as pd
import sys
from pathlib import Path
from argparse import ArgumentParser
from openai import OpenAI
from datasets import load_from_disk, Dataset

from formalize.mathatlas import MathAtlas

SYSTEM = "You are an expert mathematician who is fluent in Lean 4."

PROMPT = """Given a mathematical {tag}, formalize it into Lean 4. Be concise, and output only the formalized {tag}

-- {tag} to formalize
{text}

-- formalized {tag}
"""

PROMPT_WITH_REFS = """Given a mathematical {tag}, formalize it into Lean 4. You may use definitions found in Mathlib. You are also given the {tag}'s name(s) and a list of referenced objects. Be concise, and output only the formalized {tag}.

-- Names
{names}

-- References
{refs}

--- {tag} to formalize
{text}

--- formalized {tag}
"""

EXAMPLE_PROMPT_NO_REFS = """Given a mathematical definition, formalize it into Lean 4.

-- definition to formalize
A ring is an _integral domain_ if it is not the zero ring and if \\(ab=0\\) in the ring implies that \\(a=0\\) or \\(b=0\\).

-- formalized definition
"""

EXAMPLE_PROMPT_WITH_REFS = """Given a mathematical definition, formalize it into Lean 4. You may use definitions found in Mathlib. You are also given the definition's name(s) and a list of referenced objects. Be concise, and output only the formalized definition.

-- Names


-- References
# ID: ring
# Type: None
# Text: None

# ID: integral domain
# Type: definition
# Text: A commutative ring is said to be an _integral domain_ if \\(1_{R}\\neq 0\\) and the cancellation law holds for multiplication,

\\[ab=ac,\\,a\\neq 0,\\,\\text{implies }b=c.\\]

# ID: zero ring
# Type: definition
# Text:  If \\(1\\) and \\(0\\) do happen to coincide in \\(R\\), then it readily follows that \\(0\\) is the only element of \\(R\\), and \\(R\\) is said to be the **zero ring**.

--- definition to formalize
A ring is an _integral domain_ if it is not the zero ring and if \\(ab=0\\) in the ring implies that \\(a=0\\) or \\(b=0\\).

--- formalized definition
"""

EXAMPLE_COMPLETION = """class IntegralDomain (R : Type u) extends CommRing R, Nontrivial R where
  eq_zero_or_eq_zero_of_mul_eq_zero : ∀ a b : R, a * b = 0 → a = 0 ∨ b = 0"""


def find_refs(row: dict, database: MathAtlas):
    refs = []
    for idn, tag, links in zip(row["text"], row["tag"], row["links"]):
        if tag != "reference":
            continue
        if len(links) == 0:
            refs.append((idn, None, None))
            continue
        name = database.pandas.loc[
            links[0]["file_id"], links[0]["start"], links[0]["end"], links[0]["tag"]
        ]
        target = name.parent_text.iloc[0]
        target_tag = name.parent_tag.iloc[0]
        refs.append((idn, target_tag, target))
    return refs


def find_names(row: dict):
    return [idn for idn, tag in zip(row["text"], row["tag"]) if tag == "name"]


def format_example(example: dict, database: MathAtlas, use_references: bool = True):
    text = example["parent_text"][0]
    tag = example["tag"][0]
    parent_tag = example["parent_tag"]
    if use_references:
        refs = find_refs(example, database)
        ref_text = "\n\n".join(
            [
                f"# ID: {ref}\n# Type: {reftag}\n# Text: {reftext}"
                for ref, reftag, reftext in refs
            ]
        )
        name_text = ", ".join(find_names(example))
        prompt = PROMPT_WITH_REFS.format(
            tag=tag, names=name_text, text=text, refs=ref_text
        )
    else:
        prompt = PROMPT.format(tag=parent_tag, text=text)

    return prompt


def format_conversation(
    prompt: str, system: str, example: tuple[str, str] | None = None
):
    messages = [
        dict(role="system", content=system),
    ]
    if example is not None:
        messages += [
            dict(role="user", content=example[0]),
            dict(role="assistant", content=example[1]),
        ]
    messages.append(dict(role="user", content=prompt))
    return messages


if __name__ == "__main__":
    parser = ArgumentParser("formalize")

    subparser = parser.add_subparsers(help="command", dest="command")

    process = subparser.add_parser("process_data")
    process.add_argument("--json", "-j", type=str, help="path to MathAtlas json")
    process.add_argument(
        "--save_dataset",
        "-s",
        type=str,
        default="./mathatlas/",
        help="path where to save dataset",
    )

    formalize = subparser.add_parser("formalize")
    formalize.add_argument("--dataset", "-d", type=str, help="path to dataset")
    formalize.add_argument("--output", "-o", type=str, help="path to output directory")
    formalize.add_argument("--model", "-m", type=str, help="name of model")
    formalize.add_argument("--icl", action='store_true', help="add an example to the prompt")
    formalize.add_argument(
        "--stop", type=str, help="stop token for model, e.g., for internlm, <|im_end|>"
    )
    formalize.add_argument(
        "--num_examples",
        "-n",
        type=int,
        default=None,
        help="number of examples to process",
    )
    formalize.add_argument("--shuffle", action="store_true", help="shuffle the dataset")
    formalize.add_argument(
        "--system", "-s", type=str, default=SYSTEM, help="system instruction"
    )

    args = parser.parse_args()

    match args.command:
        case "process_data":
            database = MathAtlas.from_mathatlas(args.json)
            # Save the dataset with refs
            with_refs = database.map(
                lambda ex: {
                    "prompt": format_example(ex, database, use_references=True)
                },
                batched=False,
            )
            with_refs.save_to_disk(Path(args.save_dataset, "with_refs"))

            # Save the dataset without refs
            no_refs = database.map(
                lambda ex: {
                    "prompt": format_example(ex, database, use_references=False)
                },
                batched=False,
            )
            no_refs.save_to_disk(Path(args.save_dataset, "no_refs"))
        case "formalize":
            from vllm import LLM, SamplingParams

            # Load the dataset, and grab just the prompts
            database = MathAtlas.load_from_disk(args.dataset).select(
                range(args.num_examples or sys.maxsize)
            )
            if args.shuffle:
                database = database.shuffle(seed=1234)
            prompts = database["prompt"]

            conversations = [
                format_conversation(
                    p,
                    args.system,
                    example=(
                        None
                        if not args.icl
                        else EXAMPLE_PROMPT_WITH_REFS
                        if "with_refs" in args.dataset
                        else EXAMPLE_PROMPT_NO_REFS,
                        EXAMPLE_COMPLETION,
                    ),
                )
                for p in prompts
            ]

            # Load the model
            llm = LLM(model=args.model, trust_remote_code=True)

            # Chat with deterministic sampling
            sampling_params = SamplingParams(
                max_tokens=2048, temperature=0.0, top_p=1, stop=[':= by', args.stop]
            )
            outputs = llm.chat(
                messages=conversations, sampling_params=sampling_params, use_tqdm=True
            )
            completions = [output.outputs[0].text for output in outputs]

            # Save to json
            df = pd.DataFrame.from_records(
                list(zip(prompts, conversations, completions)),
                columns=["prompt", "conversation", "completion"],
            )
            model_name = Path(args.model).stem
            dataset_name = Path(args.dataset).stem
            icl = "icl" if args.icl else "no_icl"
            df.to_json(Path(args.output, f"{model_name}.{dataset_name}.{icl}.json"))
