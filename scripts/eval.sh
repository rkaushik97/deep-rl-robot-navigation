#!/bin/bash
# Central evaluation — run the reference `test_agent` methodology natively in THIS repo.
#   scripts/eval.sh <ALGO> <MODEL_DIR> <EPISODE> [N_EPS=100] [DOMAIN=54]
# e.g. scripts/eval.sh ddpg src/turtlebot3_drl/turtlebot3_drl/model/fond-filly/ddpg_46_stage_9 8000
#      scripts/eval.sh sac  /home/kaushik/project/turtlebot3_deepRL/.../64891e20d104/sac_5_stage_9 6400
#
# Runs N RANDOM-goal deterministic episodes (ENABLE_DYNAMIC_GOALS=False), tallies
# success/wall/obstacle/timeout, prints the standard summary and writes it to
# evaluation/results/. test_agent self-exits after N episodes (DRL_TEST_EPISODES).
# Action space is the reference one (backward on, step_time 0.05) — override via env if needed.
BASE=/home/kaushik/project/deep-rl-robot-navigation
ALGO=${1:?need ALGO (ddpg|td3|sac)}; MODEL_DIR=${2:?need MODEL_DIR (abs/rel path to session dir)}; EP=${3:?need EPISODE}
NEPS=${4:-100}; DOM=${5:-54}
TAG="eval_${ALGO}_$$"
MODEL_DIR=$(readlink -f "$MODEL_DIR")
[ -f "$MODEL_DIR/actor_stage9_episode${EP}.pt" ] || { echo "ERROR: no actor_stage9_episode${EP}.pt in $MODEL_DIR"; exit 1; }

source /opt/ros/jazzy/setup.bash
source "$BASE/install/setup.bash"
export TURTLEBOT3_MODEL=burger DRLNAV_BASE_PATH=$BASE
echo 9 > /tmp/drlnav_current_stage.txt
export ROS_DOMAIN_ID=$DOM GZ_PARTITION=$TAG
# eval config: random goals (reference test methodology), reference action space, N-episode cap
export DRL_DYNAMIC_GOALS=False DRL_TEST_EPISODES=$NEPS DRL_VAL_EPS=0
export DRL_BACKWARD=${DRL_BACKWARD:-True} DRL_STEP_TIME=${DRL_STEP_TIME:-0.05}

# stage the checkpoint under the StorageManager model dir (src/turtlebot3_drl/model/, where our
# own runs live in model/<host>/) so it loads regardless of which host trained it.
STAGE_NAME="eval_${ALGO}_$$_stage9"        # MUST end in '9' (stage = last char of session name)
LINK="$BASE/src/turtlebot3_drl/model/examples/$STAGE_NAME"
mkdir -p "$(dirname "$LINK")"; ln -sfn "$MODEL_DIR" "$LINK"
trap 'rm -f "$LINK"' EXIT
SESSION="examples/$STAGE_NAME"

echo "===== EVAL : $ALGO  $MODEL_DIR  ep$EP  ($NEPS eps, dom $DOM) ====="
nohup ros2 launch turtlebot3_drl_gazebo turtlebot3_drl_stage9.launch.py headless:=true > /tmp/${TAG}_sim.log 2>&1 &
sleep 30
nohup ros2 run turtlebot3_drl environment  > /tmp/${TAG}_env.log   2>&1 &
nohup ros2 run turtlebot3_drl gazebo_goals > /tmp/${TAG}_goals.log 2>&1 &
sleep 10
# test_agent runs N episodes then prints the summary and exits on its own
ros2 run turtlebot3_drl test_agent "$ALGO" "$SESSION" "$EP"
echo "EVAL_DONE"

# tear down ONLY this domain+partition stack (never global pkill; protects other domains)
for p in $(pgrep -f 'gz sim|parameter_bridge|ros2 run turtlebot3_drl|ros2 launch turtlebot3_drl' 2>/dev/null); do
  e=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null) || continue
  echo "$e" | grep -q "ROS_DOMAIN_ID=$DOM" && echo "$e" | grep -q "GZ_PARTITION=$TAG" && kill $p 2>/dev/null
done
