#!/bin/bash

git clone https://github.com/offendo/kimina-lean-server lean_server
(cd lean_server && pip install -r requirements.txt)

python run_verify.py --data $VERIFY_INPUT --output $VERIFY_OUTPUT --num_samples $VERIFY_NUM_SAMPLES
echo "saved verified outputs at $VERIFY_OUTPUT"
