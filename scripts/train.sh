#!/bin/bash
# Train an algorithm with its frozen reference-exact config (optionally an experiment override).
#   scripts/train.sh <algo> [experiment] [domain] [max_eps]
# e.g. scripts/train.sh ddpg                 # base reference config
#      scripts/train.sh ddpg curriculum      # base + experiments/curriculum.sh override
#      scripts/train.sh sac  '' 49 8000      # base, domain 49, stop after 8000 eps
BASE=/home/kaushik/project/deep-rl-robot-navigation
ALGO=${1:?need algo (ddpg|td3|sac)}; EXP=${2:-}; DOM=${3:-}; MAXEP=${4:-0}
ALGDIR=$BASE/src/turtlebot3_drl/turtlebot3_drl/algorithms/$ALGO
[ -f "$ALGDIR/config.sh" ] || { echo "no config.sh for algo '$ALGO'"; exit 1; }
case "$ALGO" in ddpg) DOM=${DOM:-47};; td3) DOM=${DOM:-48};; sac) DOM=${DOM:-49};; *) DOM=${DOM:-47};; esac

source /opt/ros/jazzy/setup.bash
source "$BASE/install/setup.bash"
export TURTLEBOT3_MODEL=burger DRLNAV_BASE_PATH=$BASE
echo 9 > /tmp/drlnav_current_stage.txt
export ROS_DOMAIN_ID=$DOM GZ_PARTITION=train_${ALGO}
NPROC=$(nproc); export OMP_NUM_THREADS=$NPROC MKL_NUM_THREADS=$NPROC OPENBLAS_NUM_THREADS=$NPROC

# frozen base config, then optional experiment override
source "$ALGDIR/config.sh"
if [ -n "$EXP" ]; then
  [ -f "$ALGDIR/experiments/$EXP.sh" ] || { echo "no experiment '$EXP' for $ALGO"; exit 1; }
  source "$ALGDIR/experiments/$EXP.sh"; echo "[experiment override: $EXP]"
fi
export DRL_MAX_EPISODES=$MAXEP

mkdir -p "$BASE/log"
TS=$(date +%Y%m%d-%H%M%S); LOG="$BASE/log/${ALGO}_${EXP:-base}_${TS}.log"

echo "=== train $ALGO (exp=${EXP:-base}, dom=$DOM, max_eps=$MAXEP) ==="
env | grep -E '^DRL_' | sort
# sim stack
nohup ros2 launch turtlebot3_drl_gazebo turtlebot3_drl_stage9.launch.py headless:=true > /tmp/train_${ALGO}_sim.log 2>&1 &
sleep 30
nohup ros2 run turtlebot3_drl environment  > /tmp/train_${ALGO}_env.log   2>&1 &
nohup ros2 run turtlebot3_drl gazebo_goals > /tmp/train_${ALGO}_goals.log 2>&1 &
sleep 10
# trainer (inherits the exported DRL_* config); full console tee'd to the log file
PYTHONUNBUFFERED=1 nohup ros2 run turtlebot3_drl train_agent "$ALGO" > "$LOG" 2>&1 &
echo "trainer pid $! ; console log: $LOG"
echo "metrics + plots land in model/<host>/${ALGO}_<i>_stage_9/  (_metrics.tsv, training.png)"
echo "tail -f $LOG"
