#!/bin/bash
# Commit + push a completed experiment's artifacts plus the current code/framework
# (so the run is reproducible). Checkpoints/buffers are gitignored (*.pt/*.pkl,
# model/) — only text (config.txt, train.log, analysis.md, code) gets pushed.
#
#   scripts/push_experiment.sh <EXP_NAME> ["commit message"]      # stages + shows diff (safe, no push)
#   CONFIRM=1 scripts/push_experiment.sh <EXP_NAME> ["message"]   # actually commit + push
set -e
BASE=/home/kaushik/project/deep-rl-robot-navigation
cd "$BASE"
EXP=${1:?need EXP_NAME}
MSG=${2:-"experiment: $EXP"}

# -A so file deletions (e.g. removed algorithms) are staged too, not just adds/mods
git add -A experiments/"$EXP" scripts/ .gitignore \
        src/turtlebot3_drl/turtlebot3_drl \
        src/turtlebot3_drl/setup.py \
        src/turtlebot3_msgs runs/DDPG_RESULTS.md 2>/dev/null || true

echo "=== staged for commit (verify NO .pt/.pkl/model/ below) ==="
git status --short
echo
if [ "${CONFIRM:-0}" = "1" ]; then
  git commit -m "$MSG

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  git push origin main
  echo "pushed to origin/main ✓"
else
  echo ">> review above. To push:  CONFIRM=1 scripts/push_experiment.sh $EXP \"$MSG\""
fi
