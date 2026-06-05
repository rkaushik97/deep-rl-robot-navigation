#!/bin/bash
# Regenerate the 4-panel training figure (training.png) for a session dir from its _metrics.tsv.
#   scripts/plot.sh <session_dir>
BASE=/home/kaushik/project/deep-rl-robot-navigation
source /opt/ros/jazzy/setup.bash
source "$BASE/install/setup.bash"
export DRLNAV_BASE_PATH=$BASE
python3 -m turtlebot3_drl.training.plots "${1:?need <session_dir> (e.g. src/turtlebot3_drl/model/<host>/ddpg_3_stage_9)}"
