#!/bin/bash

if [ "$SKIP_TRAIN" = "" ]; then
  bash run_em.sh
fi
if [ "$SKIP_INFERENCE" = "" ]; then
  bash run_herald.sh
fi
if [ "$SKIP_SCORE" = "" ]; then
  bash run_align.sh
fi
if [ "$SKIP_VERIFY" = "" ]; then
  bash run_verify.sh
fi
