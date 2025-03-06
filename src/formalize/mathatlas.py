import pandas as pd
from datasets import Dataset
from dataclasses import dataclass

# TODO
# 1. Finish writing this class, and load from miniF2F, proofnet, and MathAtlas
# 2. Write a script to run autoformalization with different huggingface models, and also with openAI models via vLLM
# 3. Models to test: llemma, DeepSeek R1 (distilled 7b, 32b), o1, 4o, 4o-mini


@dataclass
class MathAtlasExample:
    fileid: str
    text: str
    tag: str
    references: list | None = None
    names: list | None = None


class MathAtlas(Dataset):
    pandas: pd.DataFrame

    @classmethod
    def from_mathatlas(cls, json):
        # Read in MathAtlas json and fill nans
        df = pd.read_json(json).drop(["link_idx", "valid", "parent_annoid", "color"], axis=1)
        df["parent_text"] = df["parent_text"].fillna(df.text)
        df["parent_tag"] = df["parent_tag"].fillna(df.tag)
        df["parent_start"] = df["parent_start"].fillna(df.start)
        df["parent_end"] = df["parent_end"].fillna(df.end)

        # Group by parent node so we can grab all refs/names at once
        gb = df.groupby(["file_id", "parent_start", "parent_end", "parent_tag"]).agg(list).reset_index()
        ds: MathAtlas = MathAtlas.from_pandas(gb)  # type:ignore

        # Sort index
        df = df.set_index(["file_id", "start", "end", "tag"]).sort_index()
        setattr(ds, "pandas", df)
        return ds
