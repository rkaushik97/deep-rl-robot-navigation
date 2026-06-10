#!/bin/bash
# Watch a still-training run and run the STANDARD test_agent eval (100 random-goal eps,
# curriculum OFF) at every Nth-episode checkpoint. Sequential: one eval at a time.
#   scripts/eval_watch.sh <algo> <model_dir> [stride=500] [poll_sec=120]
BASE=/home/kaushik/project/deep-rl-robot-navigation
ALGO=${1:?need algo}; DIR=$(readlink -f "${2:?need model_dir}"); STRIDE=${3:-500}; POLL=${4:-120}
OUT="$DIR/_eval${STRIDE}_test_agent.log"
DONE="$DIR/.eval${STRIDE}_done"; touch "$DONE"
echo "[eval_watch] $ALGO $DIR every $STRIDE eps -> $OUT"
while true; do
  for ep in $(ls "$DIR" 2>/dev/null | grep -oE 'actor_stage9_episode[0-9]+\.pt' | grep -oE '[0-9]+' | sort -n -u); do
    if [ $((ep % STRIDE)) -eq 0 ] && [ "$ep" -gt 0 ] && ! grep -qx "$ep" "$DONE"; then
      echo "[$(date '+%H:%M:%S')] === eval ep$ep (100 eps, curriculum off) ===" | tee -a "$OUT"
      res=$(bash "$BASE/scripts/eval.sh" "$ALGO" "$DIR" "$ep" 100 54 2>&1)
      line=$(echo "$res" | grep -E "Successes:" | tail -1)
      echo "ep$ep | $line" | tee -a "$OUT"
      echo "$ep" >> "$DONE"
    fi
  done
  sleep "$POLL"
done
