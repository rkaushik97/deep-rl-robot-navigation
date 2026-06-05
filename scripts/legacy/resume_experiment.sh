#!/bin/bash
# Resume a prior session (weights + REPLAY BUFFER + curriculum + episode counter) and
# keep training. Unlike run_experiment.sh's warm-start (weights only, fresh buffer), this
# uses train_agent's load_session path so the proven run continues with full context.
#   scripts/resume_experiment.sh <EXP_NAME> <SESSION_NAME> <LOAD_EPISODE> <REWARD> <MAX_EPS> <DOMAIN>
# e.g. scripts/resume_experiment.sh exp012_resume_exp002 ddpg_27_stage_9 800 P 1500 46
BASE=/home/kaushik/project/deep-rl-robot-navigation

EXP=${1:?need EXP_NAME}
SESSION=${2:?need SESSION_NAME (dir under model/<host>/)}
LOADEP=${3:?need LOAD_EPISODE}
REWARD=${4:-P}
MAXEP=${5:-1500}
DOM=${6:-46}
ALGO=ddpg

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
export DRL_REWARD=$REWARD
export DRL_MAX_EPISODES=$MAXEP
NPROC=$(nproc); export OMP_NUM_THREADS=$NPROC MKL_NUM_THREADS=$NPROC OPENBLAS_NUM_THREADS=$NPROC

{
  echo "experiment : $EXP (RESUME)"
  echo "datetime   : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "resume_session : $SESSION   load_episode: $LOADEP"
  echo "reward     : $REWARD   max_episodes: $MAXEP"
  echo "stage      : 9   domain: $DOM   partition: $EXP"
} > "$EXPDIR/config.txt"
cat "$EXPDIR/config.txt"

tmux kill-session -t "$EXP" 2>/dev/null || true
nohup ros2 launch turtlebot3_drl_gazebo turtlebot3_drl_stage9.launch.py headless:=true > "$LOG/sim.log" 2>&1 &
echo "launched gz sim; waiting 30s..."; sleep 30
nohup ros2 run turtlebot3_drl environment > "$LOG/env.log" 2>&1 &
nohup ros2 run turtlebot3_drl gazebo_goals > "$LOG/goals.log" 2>&1 &
echo "env + goals started; waiting 10s..."; sleep 10
tmux new-session -d -s "$EXP" -n train \
  "bash -c 'source /opt/ros/jazzy/setup.bash; source $BASE/install/setup.bash; \
            export DRLNAV_BASE_PATH=$BASE ROS_DOMAIN_ID=$DOM GZ_PARTITION=$EXP \
                   DRL_REWARD=$REWARD DRL_MAX_EPISODES=$MAXEP \
                   DRL_OBSTACLE_K=${DRL_OBSTACLE_K:-0.5} DRL_OBSTACLE_SAFE=${DRL_OBSTACLE_SAFE:-0.40} DRL_PROGRESS_K=${DRL_PROGRESS_K:-2.0} \
                   DRL_HYSTERESIS=${DRL_HYSTERESIS:-0} DRL_CURRICULUM_MIN=${DRL_CURRICULUM_MIN:-2.0} DRL_CURRICULUM_MAX=${DRL_CURRICULUM_MAX:-4.0} \
                   DRL_NSTEP=${DRL_NSTEP:-1} DRL_PER=${DRL_PER:-0} DRL_WARMSTART_FILL=${DRL_WARMSTART_FILL:-4000} \
                   OMP_NUM_THREADS=$NPROC MKL_NUM_THREADS=$NPROC OPENBLAS_NUM_THREADS=$NPROC; \
            PYTHONUNBUFFERED=1 ros2 run turtlebot3_drl train_agent $ALGO $SESSION $LOADEP 2>&1 | tee $EXPDIR/train.log; exec bash'"
echo "resume $EXP up. log: $EXPDIR/train.log"
