#!/bin/bash

git clone https://github.com/project-numina/kimina-lean-server lean_server
(cd lean_server && pip install -r requirements.txt)

python run_verify.py --data $VERIFY_INPUT --output $VERIFY_OUTPUT
