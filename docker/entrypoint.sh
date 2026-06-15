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

# ---- rendering ----
# gz always renders (the LiDAR sensor, and in VIZ mode the GUI too); on a GPU-less host
# (incl. arm64 / Apple-silicon under Docker) force the mesa software rasteriser.
export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
start_xvfb() {  # offscreen sensor rendering when there is no real display
  export QT_QPA_PLATFORM=offscreen
  if [ -z "${DISPLAY:-}" ]; then
    Xvfb :99 -screen 0 1280x1024x24 >/tmp/xvfb.log 2>&1 &
    export DISPLAY=:99
    sleep 2
  fi
}

if [ "${VIZ:-0}" = "1" ]; then
  # Everything (server, sensors, AND the GUI) renders on an internal Xvfb with llvmpipe software
  # GL — the proven path (llvmpipe gives OpenGL 4.5, so OGRE2 works headless). The finished pixels
  # are then streamed off the box, NOT pushed as GL over the network:
  #   VIZ_VNC=1 (default): x11vnc + noVNC serve the Xvfb desktop -> view in a browser / VNC client.
  #   VIZ_VNC=0          : legacy path that points the GUI at a host X server (XQuartz) over X11
  #                        forwarding. Kept for reference, but Gazebo Harmonic's Qt-Quick GUI can't
  #                        get a usable GL context over network GLX, so its viewport stays blank.
  VNC=${VIZ_VNC:-1}
  WANT_GUI=$([ "${VIZ_HEADLESS:-0}" = "1" ] && echo 0 || echo 1)
  unset DISPLAY            # ignore any inherited host DISPLAY; force a fresh internal Xvfb
  start_xvfb              # -> DISPLAY=:99 + QT_QPA_PLATFORM=offscreen for the server-side stack
  HEADLESS=true           # the launch itself never starts a gz client
  if [ "$VNC" = "1" ]; then
    GUI_DISPLAY=:99                      # GUI renders on the internal Xvfb; pixels streamed via VNC
    echo "===== VIZ(VNC): rendering on :99 (software GL); serving noVNC :6080 / VNC :5900 ====="
  else
    GUI_DISPLAY=${GUI_DISPLAY:-host.docker.internal:0}   # legacy XQuartz path (GUI viewport blank)
    echo "===== VIZ(X11): GUI -> $GUI_DISPLAY (indirect GLX — Gazebo viewport typically blank) ====="
  fi
else
  # Default headless train/eval path (compose / k8s): gz renders the LiDAR sensor offscreen.
  HEADLESS=true
  start_xvfb
fi

# ---- config: algo base config.sh, optional experiment, then env overrides win ----
# Snapshot DRL_* the caller passed in so they survive (and override) config.sh — this is
# what lets a k8s sweep set DRL_LR/DRL_BATCH_SIZE per pod without editing files.
declare -A OVERRIDE
# Drop empty DRL_* knobs first: compose/k8s emit unset overrides as "" (e.g. DRL_BATCH_SIZE=""),
# but settings.py's os.environ.get(key, default) only falls back to default when the var is UNSET
# — a set-but-empty value reaches int()/float() and crashes. Unset them so defaults apply.
while IFS='=' read -r k v; do [ -z "$v" ] && unset "$k"; done < <(env | grep '^DRL_' || true)
while IFS='=' read -r k v; do [ -n "$k" ] && OVERRIDE["$k"]="$v"; done < <(env | grep '^DRL_' || true)

[ -f "$ALGDIR/config.sh" ] && source "$ALGDIR/config.sh"
if [ -n "${EXP:-}" ]; then
  [ -f "$ALGDIR/experiments/$EXP.sh" ] || { echo "no experiment '$EXP' for $ALGO"; exit 1; }
  source "$ALGDIR/experiments/$EXP.sh"; echo "[experiment override: $EXP]"
fi
for k in "${!OVERRIDE[@]}"; do export "$k=${OVERRIDE[$k]}"; done

# ---- bring up the sim stack ----
echo "===== $MODE  algo=$ALGO  stage=$STAGE  dom=$ROS_DOMAIN_ID  part=$GZ_PARTITION ====="
env | grep -E '^DRL_' | sort || true

if [ "${VIZ:-0}" = "1" ]; then
  # Server stack offscreen on the internal Xvfb. robot_state_publisher (TF for RViz) and the
  # overhead capture cam render here too; the gz client / RViz are launched separately below.
  WORLD_LAUNCH="turtlebot3_drl_viz_stage${STAGE}.launch.py"
  nohup ros2 launch turtlebot3_drl_gazebo "$WORLD_LAUNCH" \
        headless:=true \
        rviz:=false \
        robot_state_pub:="$([ "${VIZ_RVIZ:-0}" = "1" ] && echo true || echo false)" \
        capture_cam:="$([ "${VIZ_CAPTURE:-0}" = "1" ] && echo true || echo false)" \
        >/tmp/sim.log 2>&1 &
else
  WORLD_LAUNCH="turtlebot3_drl_stage${STAGE}.launch.py"
  nohup ros2 launch turtlebot3_drl_gazebo "$WORLD_LAUNCH" headless:=true >/tmp/sim.log 2>&1 &
fi
sleep 30

# ---- GUI surfaces (gz client / RViz) + pixel streaming ----
# The GUI connects to the already-running gz server over gz-transport (same GZ_PARTITION). In VNC
# mode it renders on the internal Xvfb with software GL (works) and x11vnc + noVNC stream the
# framebuffer; in legacy X11 mode it's pushed at a host X server over indirect GLX (viewport blank).
if [ "${VIZ:-0}" = "1" ] && [ "$WANT_GUI" = "1" ]; then
  RVIZ_CFG="$BASE/install/turtlebot3_drl_gazebo/share/turtlebot3_drl_gazebo/rviz/drl.rviz"
  if [ "$VNC" = "1" ]; then
    # window manager so the gz/RViz windows are movable/resizable inside the streamed desktop
    nohup fluxbox >/tmp/fluxbox.log 2>&1 &
    # share the Xvfb framebuffer over RFB, then bridge it to a browser over WebSocket (noVNC)
    nohup x11vnc -display :99 -nopw -forever -shared -rfbport 5900 >/tmp/x11vnc.log 2>&1 &
    nohup websockify --web=/usr/share/novnc 6080 localhost:5900 >/tmp/novnc.log 2>&1 &
    sleep 2
    echo "===== open http://localhost:6080/vnc.html  (or VNC client -> localhost:5900) ====="
    GUI_ENV=(env "DISPLAY=:99" "QT_QPA_PLATFORM=xcb" "LIBGL_ALWAYS_SOFTWARE=1")
    GUI_ENGINE=${GZ_GUI_ENGINE:-ogre2}   # llvmpipe gives GL 4.5, so OGRE2 renders fine on Xvfb
  else
    GUI_ENV=(env "DISPLAY=$GUI_DISPLAY" "LIBGL_ALWAYS_INDIRECT=1" "QT_QPA_PLATFORM=xcb" "LIBGL_ALWAYS_SOFTWARE=0")
    GUI_ENGINE=${GZ_GUI_ENGINE:-ogre}    # iGLX only does old GL; try OGRE v1 (still usually blank)
  fi
  echo "===== launching gz GUI client on $GUI_DISPLAY (--render-engine-gui $GUI_ENGINE) ====="
  nohup "${GUI_ENV[@]}" gz sim -g -v2 --force-version 8 --render-engine-gui "$GUI_ENGINE" >/tmp/gui.log 2>&1 &
  if [ "${VIZ_DEMO:-0}" = "1" ]; then
    # Live SAC inference demo (the demo.rviz HUD): viz_helper publishes the world TF, goal sphere,
    # robot arrow, path, and the Algorithm/Episode/Success/Rate stats marker; RViz shows it with
    # the bundled top-down demo config. Both connect over the shared ROS_DOMAIN_ID/GZ_PARTITION.
    echo "===== DEMO: launching viz_helper + RViz (demo.rviz) ====="
    nohup "${GUI_ENV[@]}" python3 "$BASE/visualisation/viz_helper.py" --ros-args -p use_sim_time:=true \
          >/tmp/viz_helper.log 2>&1 &
    nohup "${GUI_ENV[@]}" ros2 run rviz2 rviz2 -d "$BASE/visualisation/demo.rviz" --ros-args -p use_sim_time:=true \
          >/tmp/rviz.log 2>&1 &
    # best-effort: tile gz left / RViz right in the streamed 1280x1024 desktop once they appear
    ( for _ in $(seq 1 20); do sleep 3
        DISPLAY=:99 wmctrl -r "Gazebo" -e '0,0,0,632,1000'   2>/dev/null && \
        DISPLAY=:99 wmctrl -r "RViz"   -e '0,640,0,632,1000' 2>/dev/null && break
      done ) &
  elif [ "${VIZ_RVIZ:-0}" = "1" ]; then
    echo "===== launching RViz on $GUI_DISPLAY ====="
    nohup "${GUI_ENV[@]}" ros2 run rviz2 rviz2 -d "$RVIZ_CFG" --ros-args -p use_sim_time:=true \
          >/tmp/rviz.log 2>&1 &
  fi
fi
nohup ros2 run turtlebot3_drl environment  >/tmp/env.log   2>&1 &
nohup ros2 run turtlebot3_drl gazebo_goals >/tmp/goals.log 2>&1 &
sleep 10

# ---- run the job in the foreground ----
if [ "$MODE" = "sim" ]; then
  # Sim-only: bring up the stack and hold, so an external controller (e.g.
  # scripts/capture_visual.py for the activations demo) can drive the robot. No agent runs here.
  echo "===== sim up (MODE=sim); exec your controller into this container, Ctrl-C to stop ====="
  exec sleep infinity
elif [ "$MODE" = "eval" ]; then
  : "${MODEL_DIR:?eval needs MODEL_DIR (mounted session dir)}"
  : "${EPISODE:?eval needs EPISODE}"
  NEPS=${N_EPS:-100}
  MODEL_DIR=$(readlink -f "$MODEL_DIR")
  [ -f "$MODEL_DIR/actor_stage${STAGE}_episode${EPISODE}.pt" ] || \
    { echo "ERROR: no actor_stage${STAGE}_episode${EPISODE}.pt in $MODEL_DIR"; exit 1; }
  export DRL_TEST_EPISODES=$NEPS DRL_VAL_EPS=0 DRL_DYNAMIC_GOALS=False
  STAGE_NAME="eval_${ALGO}_$$_stage${STAGE}"
  SESSION_DIR="$BASE/src/turtlebot3_drl/model/examples/$STAGE_NAME"
  # MODEL_DIR is mounted read-only, but test_agent writes its per-test log INTO the session dir.
  # So make the session dir a WRITABLE real dir and symlink just the actor weights in from the
  # ro mount (eval is weights-only — that's the only checkpoint file test_agent reads).
  mkdir -p "$SESSION_DIR"
  ln -sfn "$MODEL_DIR/actor_stage${STAGE}_episode${EPISODE}.pt" "$SESSION_DIR/"
  exec ros2 run turtlebot3_drl test_agent "$ALGO" "examples/$STAGE_NAME" "$EPISODE"
else
  export DRL_MAX_EPISODES=${DRL_MAX_EPISODES:-0}
  exec ros2 run turtlebot3_drl train_agent "$ALGO"
fi
