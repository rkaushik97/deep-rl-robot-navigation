#!/bin/bash
BASE=/home/kaushik/project/deep-rl-robot-navigation
source /opt/ros/jazzy/setup.bash
source "$BASE/install/setup.bash"
export ROS_DOMAIN_ID=60 DISPLAY=:1 LIBGL_ALWAYS_SOFTWARE=1
pkill -x rviz2 2>/dev/null
sleep 2
nohup rviz2 -d "$BASE/visualisation/demo.rviz" --ros-args -p use_sim_time:=true > /tmp/rviz.log 2>&1 &
sleep 5
# restore RViz to the position you arranged it in
wmctrl -r "RViz" -b remove,maximized_vert,maximized_horz 2>/dev/null
wmctrl -r "RViz" -e '0,771,19,820,875' 2>/dev/null
echo "rviz2 (HUD config) launched pid $!"
