from argparse import ArgumentParser
from lean_server.client import Lean4Client
import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser("verifier")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    # Boot up client
    client = Lean4Client(base_url="http://127.0.0.1:12332")

    # Read & format input data
    df = pd.read_json(args.data)

    records = [
        {'proof': thm, 'custom_id': str(idx)} 
        for idx, thm in enumerate(df['formal_statement'])
    ]

    # Launch the query & wait for the response
    response = client.verify(records, timeout=30)

    # Parse the outputs
    results = []
    for thm_output in response['results']:
        thm_id = thm_output['custom_id']

        # Extract Lean errors
        error = thm_output['error']

        # Extract compiler messages
        msg = thm_output.get('response', {}).get('messages', [])

        # Check whether it's been verified or not
        success = True
        if thm_output['error']:
            success = False
        elif any([m['severity'] == 'error' for m in msg]):
            success = False

        results.append(dict(thm_id=int(thm_id), error=error, messages=msg, verified=success))

    res_df = pd.DataFrame.from_records(results)
    both = pd.merge(left=df, right=res_df, left_index=True, right_on='thm_id', how='left')

    important_cols = [*df.columns, 'error', 'messages', 'verified']
    both[important_cols].to_json(args.output_json)
