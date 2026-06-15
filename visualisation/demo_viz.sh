#!/bin/bash
BASE=/home/kaushik/project/deep-rl-robot-navigation
source /opt/ros/jazzy/setup.bash
source "$BASE/install/setup.bash"
export ROS_DOMAIN_ID=60 GZ_PARTITION=demo_sac
pkill -f "visualisation/viz_helper.py" 2>/dev/null
sleep 1
nohup python3 "$BASE/visualisation/viz_helper.py" --ros-args -p use_sim_time:=true > /tmp/demo_viz.log 2>&1 &
sleep 2
echo "viz_helper launched (pid $!)"
