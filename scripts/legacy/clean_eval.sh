#!/bin/bash
# ============================================================================
# DEPRECATED — DO NOT USE. This is the custom 40-goal fixed-list benchmark.
# Use scripts/standard_eval.sh (reference test_agent, random goals, 100 eps) instead.
# ============================================================================
# Standard deterministic evaluation — the ONLY correct way to eval a checkpoint.
#   scripts/clean_eval.sh <actor_ckpt.pt> [N_EPS=40] [DOMAIN=52]
#
# CRITICAL: launches the eval env with DRL_DYNAMIC_GOALS=False so goals come from the
# FIXED known-valid list (same as the reference repo's test_agent). Evaluating with the
# curriculum ON (DYNAMIC_GOALS=True) is a BUG: the goal difficulty adapts to the policy's
# success, so a good policy gets pushed to harder 3-4m goals -> a self-penalizing, moving
# benchmark. That bug made every clean-eval in this project read ~20-25pp too low vs the
# reference (same checkpoint: 60% on the moving curve, ~84% on the fixed benchmark).
BASE=/home/kaushik/project/deep-rl-robot-navigation
CKPT=${1:?need actor checkpoint .pt path}
NEPS=${2:-40}
DOM=${3:-52}
TAG=cleaneval$DOM

source /opt/ros/jazzy/setup.bash
source "$BASE/install/setup.bash"
export TURTLEBOT3_MODEL=burger DRLNAV_BASE_PATH=$BASE
export ROS_DOMAIN_ID=$DOM GZ_PARTITION=$TAG
export DRL_DYNAMIC_GOALS=False     # <-- the fix: fixed goals, no curriculum-adapting difficulty
echo 9 > /tmp/drlnav_current_stage.txt
cd "$BASE"

LOG=/tmp/drl_$TAG; mkdir -p "$LOG"
nohup ros2 launch turtlebot3_drl_gazebo turtlebot3_drl_stage9.launch.py headless:=true > "$LOG/sim.log" 2>&1 &
sleep 30
nohup ros2 run turtlebot3_drl environment > "$LOG/env.log" 2>&1 &
nohup ros2 run turtlebot3_drl gazebo_goals > "$LOG/goals.log" 2>&1 &
sleep 12

echo "===== CLEAN EVAL (DYNAMIC_GOALS=False, fixed benchmark) : $(basename $CKPT) , $NEPS eps ====="
python3 scripts/replay_analyze.py "$CKPT" "$NEPS"
echo "CLEAN_EVAL_DONE"

# teardown this domain's stack only
for p in $(pgrep -f 'gz sim|turtlebot3_drl/lib|ros2 run turtlebot3_drl|parameter_bridge'); do
  tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "ROS_DOMAIN_ID=$DOM" && kill $p 2>/dev/null
done
