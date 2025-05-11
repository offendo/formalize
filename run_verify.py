from asyncio import wait_for
import re
import time
from more_itertools import chunked
import pandas as pd
import socket
from tqdm import tqdm

from argparse import ArgumentParser
from lean_server.client import Lean4Client
import logging

logging.basicConfig(level=logging.INFO)


def remove_informal_prefix(formal_statement: str) -> str:
    block_pattern = r"/-.*? -/\n"
    no_blocks = re.sub(block_pattern, "", formal_statement, flags=re.DOTALL)
    inline_pattern = r"--.*?\n"
    no_blocks_or_inline = re.sub(inline_pattern, "", no_blocks, flags=re.DOTALL)
    return no_blocks_or_inline


def anonymize(formal_statement: str):
    # Make everything a theorem
    formal_statement = formal_statement.replace("lemma", "theorem", 1)
    name_pattern = r"theorem\s*?( [\w_\d']+? )"
    name = re.sub(name_pattern, "theorem anonymous ", formal_statement, flags=re.DOTALL)
    return name


def wait_for_server(host, port, timeout=30):
    start_time = time.time()
    while True:
        try:
            # Attempt to establish a connection
            with socket.create_connection((host, port), timeout=1):
                print(f"Server {host}:{port} is active.")
                return True
        except (socket.error, socket.timeout) as e:
            # Check if timeout has been reached
            if time.time() - start_time > timeout:
                print(f"Timeout reached while waiting for server {host}:{port}.")
                return False
            # Wait before retrying
            time.sleep(1)
            print(f"Waiting for server {host}:{port}...")


if __name__ == "__main__":
    parser = ArgumentParser("verifier")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=-1)
    args = parser.parse_args()

    # Boot up client
    wait_for_server("127.0.0.1", 12332, timeout=500)
    client = Lean4Client(base_url="http://127.0.0.1:12332")
    logging.info("Connected to server.")

    # Read & format input data
    df = pd.read_json(args.data)
    if args.num_samples > 0:
        df = df[: args.num_samples]
    df["formal_statement"] = df["formal_statement"].apply(lambda xs: [remove_informal_prefix(x) for x in xs])

    records = [
        {"proof": anonymize(thm), "custom_id": f"{group_id}-{thm_id}"}
        for group_id, group in enumerate(df["formal_statement"])
        for thm_id, thm in enumerate(group)
    ]
    logging.info(f"Formatted data: {len(records)} examples to verify.")

    # Launch the query & wait for the response
    logging.info("Querying server...will take some time.")
    tik = time.time()

    # Do requests batch-wise so we can save every so often, and keep track of progress
    all_results = []
    failed = 0
    pbar = tqdm(records, desc="Querying...")
    for batch in chunked(pbar, n=256):
        try:
            response = client.verify(records, timeout=30)

            # Parse the outputs
            results = []
            for thm_output in response["results"]:
                group_id, thm_id = map(int, thm_output["custom_id"].split("-"))

                # Extract Lean errors
                error = thm_output["error"]

                # Extract compiler messages
                resp = thm_output.get("response", {})
                msg = resp.get("messages", []) if resp else []

                # Check whether it's been verified or not
                success = True
                if thm_output["error"]:
                    success = False
                elif any([m["severity"] == "error" for m in msg]):
                    success = False

                results.append(dict(group_id=group_id, theorem_id=thm_id, error=error, messages=msg, verified=success))
            all_results.extend(results)
        except Exception as e:
            empty_results = [
                dict(group_id=int(gid), theorem_id=int(tid), error=str(e), messages=[], verified=False)
                for gid, tid in [ex["custom_id"].split("-") for ex in batch]
            ]
            failed += len(batch)
            all_results.extend(empty_results)
            pbar.set_postfix({"failed": failed})

    tok = time.time()
    logging.info(f"Done in {tok-tik:0.2f}s!")

    res_df = pd.DataFrame.from_records(all_results).groupby("group_id").agg(list)
    both = pd.merge(left=df, right=res_df, left_index=True, right_index=True, how="left")

    important_cols = [*df.columns, "error", "messages", "verified"]
    both[important_cols].to_json(args.output_json)
    logging.info(f"Saved data to {args.output_json}")
