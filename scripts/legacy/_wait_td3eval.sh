#!/bin/bash
while true; do sleep 60
  grep -q 'CLEAN_EVAL_DONE' /home/kaushik/project/deep-rl-robot-navigation/experiments/reference_checkpoints/td3_ref_eval.out 2>/dev/null && { echo DONE; exit 0; }
  pgrep -f 'clean_eval.sh.*td3_ref' >/dev/null 2>&1 || { echo GONE; exit 0; }
done
