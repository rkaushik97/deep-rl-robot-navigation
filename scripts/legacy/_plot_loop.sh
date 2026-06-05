#!/bin/bash
# PERMANENT plot updater: refreshes both comparison plots every 90s (well within the
# ~100-ep checkpoint cadence) and only exits once ALL three replication trainers are done.
cd /home/kaushik/project/deep-rl-robot-navigation
while true; do
  python3 scripts/plot_replication.py 100 >/dev/null 2>&1
  python3 scripts/plot_val.py >/dev/null 2>&1
  sleep 90
  alive=0
  for d in 47 48 49; do
    for p in $(pgrep -f 'lib/turtlebot3_drl/train_agent'); do
      tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "ROS_DOMAIN_ID=$d" && alive=$((alive+1))
    done
  done
  if [ "$alive" -eq 0 ]; then
    python3 scripts/plot_replication.py 100 >/dev/null 2>&1; python3 scripts/plot_val.py >/dev/null 2>&1
    echo "WAKE all_replication_runs_finished"; exit 0
  fi
done
