#!/bin/bash
# STANDARD reference evaluation — the ONLY eval we use.
#   scripts/standard_eval.sh <ALGO> <MODEL_DIR> <EPISODE> [N_EPS=100] [DOMAIN=54]
# e.g. scripts/standard_eval.sh sac  /home/kaushik/project/turtlebot3_deepRL/src/turtlebot3_drl/model/64891e20d104/sac_5_stage_9 6400 100
#      scripts/standard_eval.sh ddpg /home/kaushik/project/turtlebot3_deepRL/src/turtlebot3_drl/model/examples/ddpg_0_stage9    8000 100
#      scripts/standard_eval.sh ddpg /home/kaushik/project/deep-rl-robot-navigation/src/turtlebot3_drl/model/fond-filly/ddpg_45_stage_9 8000 100   # our replication
#
# Runs the reference repo's `test_agent` (ros2 run turtlebot3_drl test_agent) with the
# reference's own settings (ENABLE_DYNAMIC_GOALS=False, STEP_TIME=0.05, backward, 50s timeout,
# RANDOM goals, obstacle phase never reset). Lets it accumulate N_EPS episodes, then reports the
# cumulative success rate from the test log. This is exactly what produced the README numbers.
# NO custom 40-goal benchmark, NO replay_analyze, NO eval_mode. Stop using clean_eval.sh.
#
# MODEL_DIR is symlinked into <ref>/model/examples/ so StorageManager resolves it regardless of
# which host trained it (the 'examples/' path skips the hostname lookup). Works for reference AND
# our replication checkpoints identically — one eval method for everything.
# (no `set -u`: ROS setup.bash references unbound vars; required args use ${X:?} guards.)
REF=/home/kaushik/project/turtlebot3_deepRL
ALGO=${1:?need ALGO}; MODEL_DIR=${2:?need MODEL_DIR (abs path to session dir)}; EP=${3:?need EPISODE}
NEPS=${4:-100}; DOM=${5:-54}
TAG="stdeval_${ALGO}_$$"
MODEL_DIR=$(readlink -f "$MODEL_DIR")
[ -f "$MODEL_DIR/actor_stage9_episode${EP}.pt" ] || { echo "ERROR: no actor_stage9_episode${EP}.pt in $MODEL_DIR"; exit 1; }
# name MUST end in '9': StorageManager sets stage = load_session[-1], and the checkpoint
# files are actor_stage9_episode<EP>.pt.
STAGE_NAME="stdeval_${ALGO}_$$_stage9"
LINK="$REF/src/turtlebot3_drl/model/examples/$STAGE_NAME"
ln -sfn "$MODEL_DIR" "$LINK"
SESSION="examples/$STAGE_NAME"
trap 'rm -f "$LINK"' EXIT

source /opt/ros/jazzy/setup.bash
source "$REF/install/setup.bash"
export TURTLEBOT3_MODEL=burger
export DRLNAV_BASE_PATH=$REF
echo 9 > /tmp/drlnav_current_stage.txt
export ROS_DOMAIN_ID=$DOM GZ_PARTITION=$TAG
NPROC=$(nproc); export OMP_NUM_THREADS=$NPROC

echo "===== STANDARD EVAL : $ALGO $SESSION ep$EP , target $NEPS eps , domain $DOM ====="
START=$(date +%s)

# 1) gz sim + env + goals (reference stack, reference settings)
nohup ros2 launch turtlebot3_drl_gazebo turtlebot3_drl_stage9.launch.py headless:=true > /tmp/${TAG}_sim.log 2>&1 &
sleep 30
nohup ros2 run turtlebot3_drl environment  > /tmp/${TAG}_env.log   2>&1 &
nohup ros2 run turtlebot3_drl gazebo_goals > /tmp/${TAG}_goals.log 2>&1 &
sleep 10

# 2) test_agent (standard eval loop, while(True)); capture its PID to stop after N_EPS
nohup ros2 run turtlebot3_drl test_agent "$ALGO" "$SESSION" "$EP" > /tmp/${TAG}_test.log 2>&1 &
TEST_PID=$!
echo "test_agent pid=$TEST_PID ; accumulating $NEPS episodes..."

# 3) wait until the test log has N_EPS data rows, then stop test_agent
TESTFILE=""
for i in $(seq 1 600); do
  sleep 5
  # newest _test_stage9 file for this episode, modified after START
  TESTFILE=$(find "$REF/src/turtlebot3_drl/model" -name "_test_stage9_eps${EP}_*.txt" -newermt "@$START" 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
  [ -z "$TESTFILE" ] && continue
  rows=$(grep -cE '^[0-9]' "$TESTFILE" 2>/dev/null || echo 0)
  echo "  [$((i*5))s] episodes logged: $rows / $NEPS"
  [ "$rows" -ge "$NEPS" ] && break
  kill -0 $TEST_PID 2>/dev/null || { echo "test_agent died early"; break; }
done

kill $TEST_PID 2>/dev/null
sleep 2

# 4) report cumulative success from the final data row (s/cw/co/t = success/wall/obst/timeout)
echo "===== RESULT ====="
if [ -n "$TESTFILE" ] && [ -s "$TESTFILE" ]; then
  echo "log: $TESTFILE"
  python3 - "$TESTFILE" <<'PY'
import sys
rows=[l.strip() for l in open(sys.argv[1]) if l and l[0].isdigit()]
last=rows[-1]
counts=last.split(',')[-1].strip()          # "84/14/0/2/0"
s,cw,co,t,*_=[int(x) for x in counts.split('/')]
n=s+cw+co+t
print(f"episodes={n}  SUCCESS={s} ({100*s/max(1,n):.0f}%)  wall={cw} ({100*cw/max(1,n):.0f}%)  obst={co}  timeout={t} ({100*t/max(1,n):.0f}%)")
PY
else
  echo "NO test log produced — check /tmp/${TAG}_test.log"; tail -20 /tmp/${TAG}_test.log
fi
echo "STANDARD_EVAL_DONE"

# 5) tear down ONLY this domain's stack (by PID + /proc environ domain check — never global pkill)
for p in $(pgrep -f 'gz sim|parameter_bridge|ros2 run turtlebot3_drl|ros2 launch turtlebot3_drl' 2>/dev/null); do
  tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "ROS_DOMAIN_ID=$DOM" && \
  tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "GZ_PARTITION=$TAG" && kill $p 2>/dev/null
done
