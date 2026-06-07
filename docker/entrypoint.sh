#!/usr/bin/env bash
# Container entrypoint for the TurtleBot3 deep-RL stack.
#
# Brings up the full sim stack (gz sim + environment + gazebo_goals) and then runs the
# trainer or the evaluator IN THE FOREGROUND, so the container's lifetime tracks the job
# (k8s/compose see it exit when training/eval finishes).
#
# Everything is driven by env vars (so a k8s Job or compose service is fully declarative):
#   MODE          train | eval                                (default: train)
#   ALGO          ddpg | td3 | sac                            (default: ddpg)
#   EXP           experiment name under algorithms/<algo>/experiments/  (optional)
#   DRL_STAGE     gazebo stage                                (default: 9)
#   ROS_DOMAIN_ID / GZ_PARTITION   sim isolation              (defaults derived from ALGO/host)
#   DRL_MAX_EPISODES               stop training after N eps  (train; 0 = unbounded)
#   Any DRL_*     hyperparameter override — WINS over the algo's config.sh (for sweeps)
#
# eval-only:
#   MODEL_DIR     path to a session dir holding actor_stage<stage>_episode<EP>.pt (mounted)
#   EPISODE       checkpoint episode to load
#   N_EPS         number of eval episodes                     (default: 100)
# NOTE: no `set -u` — ROS/ament setup scripts reference unbound vars (AMENT_TRACE_SETUP_FILES).
set -eo pipefail

BASE=${DRLNAV_BASE_PATH:-/opt/drlnav}
ALGO=${ALGO:-ddpg}
MODE=${MODE:-train}
STAGE=${DRL_STAGE:-9}
ALGDIR="$BASE/src/turtlebot3_drl/turtlebot3_drl/algorithms/$ALGO"

# ---- ROS / workspace ----
source /opt/ros/jazzy/setup.bash
source "$BASE/install/setup.bash"
export TURTLEBOT3_MODEL=${TURTLEBOT3_MODEL:-burger} DRLNAV_BASE_PATH="$BASE"
echo "$STAGE" > /tmp/drlnav_current_stage.txt

# ---- sim isolation (let many runs coexist on one host / one node) ----
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-$(( ($$ % 100) + 1 ))}
export GZ_PARTITION=${GZ_PARTITION:-drl_${MODE}_${ALGO}_$$}

# ---- threading + unbuffered stdout ----
# PYTHONUNBUFFERED so the trainer's live line streams to `docker logs` / `kubectl logs`
# (Python block-buffers stdout when it isn't a TTY).
export PYTHONUNBUFFERED=1
NPROC=$(nproc); export OMP_NUM_THREADS=$NPROC MKL_NUM_THREADS=$NPROC OPENBLAS_NUM_THREADS=$NPROC

# ---- headless software rendering ----
# gz must still render the LiDAR sensor even headless; on a GPU-less node (incl. arm64 /
# Apple-silicon under Docker) force the mesa software rasteriser and a virtual display.
export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
export QT_QPA_PLATFORM=offscreen
if [ -z "${DISPLAY:-}" ]; then
  Xvfb :99 -screen 0 1280x1024x24 >/tmp/xvfb.log 2>&1 &
  export DISPLAY=:99
  sleep 2
fi

# ---- config: algo base config.sh, optional experiment, then env overrides win ----
# Snapshot DRL_* the caller passed in so they survive (and override) config.sh — this is
# what lets a k8s sweep set DRL_LR/DRL_BATCH_SIZE per pod without editing files.
declare -A OVERRIDE
while IFS='=' read -r k v; do [ -n "$k" ] && OVERRIDE["$k"]="$v"; done < <(env | grep '^DRL_' || true)

[ -f "$ALGDIR/config.sh" ] && source "$ALGDIR/config.sh"
if [ -n "${EXP:-}" ]; then
  [ -f "$ALGDIR/experiments/$EXP.sh" ] || { echo "no experiment '$EXP' for $ALGO"; exit 1; }
  source "$ALGDIR/experiments/$EXP.sh"; echo "[experiment override: $EXP]"
fi
for k in "${!OVERRIDE[@]}"; do export "$k=${OVERRIDE[$k]}"; done

# ---- bring up the sim stack ----
WORLD_LAUNCH="turtlebot3_drl_stage${STAGE}.launch.py"
echo "===== $MODE  algo=$ALGO  stage=$STAGE  dom=$ROS_DOMAIN_ID  part=$GZ_PARTITION ====="
env | grep -E '^DRL_' | sort || true

nohup ros2 launch turtlebot3_drl_gazebo "$WORLD_LAUNCH" headless:=true >/tmp/sim.log 2>&1 &
sleep 30
nohup ros2 run turtlebot3_drl environment  >/tmp/env.log   2>&1 &
nohup ros2 run turtlebot3_drl gazebo_goals >/tmp/goals.log 2>&1 &
sleep 10

# ---- run the job in the foreground ----
if [ "$MODE" = "eval" ]; then
  : "${MODEL_DIR:?eval needs MODEL_DIR (mounted session dir)}"
  : "${EPISODE:?eval needs EPISODE}"
  NEPS=${N_EPS:-100}
  MODEL_DIR=$(readlink -f "$MODEL_DIR")
  [ -f "$MODEL_DIR/actor_stage${STAGE}_episode${EPISODE}.pt" ] || \
    { echo "ERROR: no actor_stage${STAGE}_episode${EPISODE}.pt in $MODEL_DIR"; exit 1; }
  export DRL_TEST_EPISODES=$NEPS DRL_VAL_EPS=0 DRL_DYNAMIC_GOALS=False
  STAGE_NAME="eval_${ALGO}_$$_stage${STAGE}"
  LINK="$BASE/src/turtlebot3_drl/model/examples/$STAGE_NAME"
  mkdir -p "$(dirname "$LINK")"; ln -sfn "$MODEL_DIR" "$LINK"
  exec ros2 run turtlebot3_drl test_agent "$ALGO" "examples/$STAGE_NAME" "$EPISODE"
else
  export DRL_MAX_EPISODES=${DRL_MAX_EPISODES:-0}
  exec ros2 run turtlebot3_drl train_agent "$ALGO"
fi
