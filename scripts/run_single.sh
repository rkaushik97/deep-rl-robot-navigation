#!/bin/bash
# Single-agent DDPG control run: ONE gz stack + environment + gazebo_goals + the
# synchronous reference-faithful learner (drl_agent.py / train_agent).
#
# This is the control for "distributed pipeline bug vs DDPG-can't-do-stage-9":
# same algorithm/reward/curriculum as the distributed runs, but the strictly
# synchronous act->pause->train->unpause loop the reference uses. If this learns
# where the distributed runs collapse, the distributed pipeline is the culprit.
#
# "More cores": the stage-9 world is uncapped (real_time_factor=0), so we let gz
# physics + sensors + torch use the whole box to step the sim as fast as possible.
#
# Usage: scripts/run_single.sh
BASE=/home/kaushik/project/deep-rl-robot-navigation

ALGO=${1:-ddpg}      # ddpg (only supported algorithm)
TAG=${2:-single}     # instance tag -> tmux session, log dir, gz partition
DOM=${3:-42}         # ROS_DOMAIN_ID (use a distinct one to run a 2nd stack in parallel)
LOG=/tmp/drl_$TAG
mkdir -p "$LOG"

source /opt/ros/jazzy/setup.bash
source "$BASE/install/setup.bash"
export TURTLEBOT3_MODEL=burger
export DRLNAV_BASE_PATH=$BASE
echo 9 > /tmp/drlnav_current_stage.txt
echo "algorithm: $ALGO | tag: $TAG | domain: $DOM"

# isolated stack: distinct DDS domain + gz partition (so parallel stacks don't clash)
export ROS_DOMAIN_ID=$DOM
export GZ_PARTITION=$TAG

# more cores: give gz + torch the whole machine (single-agent loop is otherwise
# sim-step bound, and the sim is uncapped)
NPROC=$(nproc)
export OMP_NUM_THREADS=$NPROC
export MKL_NUM_THREADS=$NPROC
export OPENBLAS_NUM_THREADS=$NPROC
echo "using $NPROC cores (OMP/MKL/OpenBLAS), domain $ROS_DOMAIN_ID, partition $GZ_PARTITION"

tmux kill-session -t "$TAG" 2>/dev/null || true

# 1) gz sim (headless, heavy startup)
nohup ros2 launch turtlebot3_drl_gazebo turtlebot3_drl_stage9.launch.py headless:=true > "$LOG/sim.log" 2>&1 &
echo "launched gz sim; waiting 30s for it to come up..."; sleep 30

# 2) environment + goals on the same stack
nohup ros2 run turtlebot3_drl environment   > "$LOG/env.log"   2>&1 &
nohup ros2 run turtlebot3_drl gazebo_goals   > "$LOG/goals.log" 2>&1 &
echo "env + goals started; waiting 10s..."; sleep 10

# 3) the synchronous single-agent trainer (in tmux so it can be watched live)
tmux new-session -d -s "$TAG" -n train \
  "bash -c 'source /opt/ros/jazzy/setup.bash; source $BASE/install/setup.bash; \
            export DRLNAV_BASE_PATH=$BASE ROS_DOMAIN_ID=$DOM GZ_PARTITION=$TAG \
                   OMP_NUM_THREADS=$NPROC MKL_NUM_THREADS=$NPROC OPENBLAS_NUM_THREADS=$NPROC; \
            PYTHONUNBUFFERED=1 ros2 run turtlebot3_drl train_agent $ALGO 2>&1 | tee $LOG/train.log; exec bash'"

echo "Single-agent $ALGO up (stage 9, tag=$TAG, domain=$DOM)."
echo "Watch:  tmux attach -t $TAG   (or: tail -f $LOG/train.log)"
