#!/bin/bash
# Tracked single-agent experiment runner.
#   scripts/run_experiment.sh <EXP_NAME> [ALGO] [REWARD] [DYNAMIC_GOALS] [MAX_EPS] [DOMAIN]
# e.g. scripts/run_experiment.sh exp001_ddpg_sparse ddpg S True 1000 42
#
# Creates experiments/<EXP_NAME>/ with config.txt + train.log (full console, incl.
# per-episode 'Epi: ... outcome:' lines) so each run is reproducible and comparable.
# Reward/curriculum/episode-budget are passed via env vars (no rebuild per experiment).
BASE=/home/kaushik/project/deep-rl-robot-navigation

EXP=${1:?need EXP_NAME}
ALGO=${2:-ddpg}        # ddpg (only supported algorithm)
REWARD=${3:-S}         # S=clean sparse, A/B=dense
DYN=${4:-True}         # dynamic-goal curriculum on/off
MAXEP=${5:-1000}       # stop after N episodes
DOM=${6:-42}           # ROS_DOMAIN_ID
WARMSTART=${7:-}       # optional: path to a prior session dir to warm-start weights from (*_best.pt)

EXPDIR=$BASE/experiments/$EXP
mkdir -p "$EXPDIR"
LOG=/tmp/drl_$EXP; mkdir -p "$LOG"

source /opt/ros/jazzy/setup.bash
source "$BASE/install/setup.bash"
export TURTLEBOT3_MODEL=burger
export DRLNAV_BASE_PATH=$BASE
echo 9 > /tmp/drlnav_current_stage.txt

export ROS_DOMAIN_ID=$DOM
export GZ_PARTITION=$EXP
# experiment knobs (read by settings.py at import)
export DRL_REWARD=$REWARD
export DRL_DYNAMIC_GOALS=$DYN
export DRL_MAX_EPISODES=$MAXEP
export DRL_WARMSTART=$WARMSTART
NPROC=$(nproc); export OMP_NUM_THREADS=$NPROC MKL_NUM_THREADS=$NPROC OPENBLAS_NUM_THREADS=$NPROC

# --- config snapshot ---
{
  echo "experiment : $EXP"
  echo "datetime   : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "git_commit : $(git -C "$BASE" rev-parse --short HEAD 2>/dev/null)  (dirty: $(git -C "$BASE" diff --quiet 2>/dev/null && echo no || echo yes))"
  echo "algorithm  : $ALGO"
  echo "reward     : $REWARD"
  echo "dynamic_goals : $DYN"
  echo "max_episodes  : $MAXEP"
  echo "warmstart  : ${WARMSTART:-none}"
  echo "stage      : 9   domain: $DOM   partition: $EXP"
  echo "--- key hyperparams (settings.py) ---"
  grep -E "^(LEARNING_RATE|TAU|BATCH_SIZE|DISCOUNT_FACTOR|HIDDEN_SIZE|STEP_TIME|OBSERVE_STEPS|CURRICULUM_MIN_RADIUS|CURRICULUM_MAX_RADIUS|VAL_EPS_PER_CHECKPOINT)" "$BASE/src/turtlebot3_drl/turtlebot3_drl/common/settings.py"
} > "$EXPDIR/config.txt"
echo "=== experiment: $EXP (algo=$ALGO reward=$REWARD dyn=$DYN max=$MAXEP) ==="
cat "$EXPDIR/config.txt"

tmux kill-session -t "$EXP" 2>/dev/null || true

# 1) gz sim
nohup ros2 launch turtlebot3_drl_gazebo turtlebot3_drl_stage9.launch.py headless:=true > "$LOG/sim.log" 2>&1 &
echo "launched gz sim; waiting 30s..."; sleep 30
# 2) env + goals
nohup ros2 run turtlebot3_drl environment > "$LOG/env.log" 2>&1 &
nohup ros2 run turtlebot3_drl gazebo_goals > "$LOG/goals.log" 2>&1 &
echo "env + goals started; waiting 10s..."; sleep 10
# 3) trainer (tmux), console teed into the experiment folder
tmux new-session -d -s "$EXP" -n train \
  "bash -c 'source /opt/ros/jazzy/setup.bash; source $BASE/install/setup.bash; \
            export DRLNAV_BASE_PATH=$BASE ROS_DOMAIN_ID=$DOM GZ_PARTITION=$EXP \
                   DRL_REWARD=$REWARD DRL_DYNAMIC_GOALS=$DYN DRL_MAX_EPISODES=$MAXEP DRL_WARMSTART=\"$WARMSTART\" \
                   OMP_NUM_THREADS=$NPROC MKL_NUM_THREADS=$NPROC OPENBLAS_NUM_THREADS=$NPROC; \
            PYTHONUNBUFFERED=1 ros2 run turtlebot3_drl train_agent $ALGO 2>&1 | tee $EXPDIR/train.log; exec bash'"

echo "experiment $EXP up. log: $EXPDIR/train.log   (tmux attach -t $EXP)"
echo "analyze when done: python3 scripts/analyze_experiment.py $EXP"
