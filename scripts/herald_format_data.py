#!/usr/bin/env python3

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
