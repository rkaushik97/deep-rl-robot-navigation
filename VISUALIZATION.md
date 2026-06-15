# Visualized inference on macOS (Gazebo GUI, RViz, activations demo)

ROS 2 Jazzy and Gazebo Harmonic are Linux-only — there is no usable native macOS build
(Gazebo Harmonic on Apple Silicon is unsupported). So the GUI runs **inside the Linux container**.

**How the pixels reach your Mac: VNC, not X11 forwarding.** Gazebo Harmonic's GUI is built on Qt
Quick and needs a modern OpenGL context. You cannot push that OpenGL over the network to XQuartz —
direct GLX crashes the renderer (`GLXBadFBConfig`) and indirect GLX (iGLX) only exposes old OpenGL,
so Qt logs `Unrecognized OpenGL version` and the viewport stays blank white. (Both were tried; both
fail. XQuartz is a dead end for this GUI.)

Instead we **render inside the container** on a virtual X display with the mesa `llvmpipe` software
rasteriser — which provides OpenGL 4.5, so OGRE2 renders fine — and **stream the finished desktop**
out over VNC. You view it in a **browser tab** (noVNC) or any VNC client. No GL crosses the network,
so it just works. It is **slow** (software GL, no GPU in the Docker VM) — that is expected.

Inference itself is the normal `eval` / `test_agent` run; the viz path brings up the GUI alongside
it. The GUI tools (`rviz2`, camera bridge, VNC server) aren't in the lean training image, so this
uses a separate **`:viz` image** — a thin overlay on the published headless image.

---

## 1. One-time: build the `:viz` image

```bash
docker build -f docker/Dockerfile.viz -t kaushik48/turtlebot3-drl:viz .   # run from the repo root
```
(From-scratch alternative: `docker build -f docker/Dockerfile --build-arg WITH_VIZ=1 -t kaushik48/turtlebot3-drl:viz .`)

## 2. Stage a checkpoint

Point `MODEL_DIR` at a session dir holding `actor_stage9_episode<EPISODE>.pt`. The repo ships the
best DDPG checkpoint as `..._best.pt`, but the eval path wants the name without `_best`, so copy it:

```bash
mkdir -p /tmp/ddpg_demo
cp experiments/replications/ddpg/actor_stage9_episode4000_best.pt \
   /tmp/ddpg_demo/actor_stage9_episode4000.pt
export MODEL_DIR=/tmp/ddpg_demo
export EPISODE=4000
```

## 3. Run + watch in the browser

```bash
N_EPS=50 docker compose -f docker/docker-compose.yml --profile viz up viz
```
Then open **http://localhost:6080/vnc.html** and click **Connect**. After the sim warm-up (~45 s) the
Gazebo window appears with the robot navigating. `N_EPS` caps eval episodes (raise it to watch longer;
or use `MODE=sim` to hold the sim open indefinitely without running the agent). A VNC client works too
— connect to `localhost:5900` (no password).

### + RViz (laser scan, odometry, TF, robot model)

```bash
VIZ_RVIZ=1 N_EPS=50 docker compose -f docker/docker-compose.yml --profile viz up viz
```
RViz opens in the same streamed desktop alongside Gazebo (config `rviz/drl.rviz`, Fixed Frame `odom`;
`/scan`, `/odom`, `/tf` flow over the existing ros_gz bridge). If a panel looks empty, switch the
Fixed Frame (`odom` ↔ `base_footprint`).

### Activations demo (offline GIF — no display at all)

The README `visual.gif`: Gazebo top-down camera + the agent's network (state, hidden layers, action,
reward), side by side. Rendered **headless**, so it needs neither VNC nor a browser.

```bash
# 1. hold the sim open with the overhead camera spawned + bridged, no GUI:
MODE=sim VIZ_CAPTURE=1 VIZ_HEADLESS=1 \
  docker compose -f docker/docker-compose.yml --profile viz up viz
# 2. in another terminal, capture the first successful episode, then build the gif:
docker compose -f docker/docker-compose.yml --profile viz exec viz bash -lc '\
  source /opt/ros/jazzy/setup.bash && source $WS/install/setup.bash && \
  python3 /opt/drlnav/scripts/capture_visual.py /checkpoint/actor_stage9_episode'"$EPISODE"'.pt /tmp/capture.npz && \
  python3 /opt/drlnav/scripts/make_visual_gif.py /tmp/capture.npz /tmp/visual.gif'
docker compose -f docker/docker-compose.yml --profile viz cp viz:/tmp/visual.gif ./visual.gif
```
(`capture_visual.py` is DDPG-specific.)

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Browser can't reach `localhost:6080` | Container not up yet, or ports not mapped. `docker ps` should show `0.0.0.0:6080->6080`. Give the sim ~45 s to start. |
| noVNC connects but screen is grey/empty | The gz GUI is still building the scene (slow on software GL). Wait, or check `/tmp/gui.log` in the container. |
| Everything is slow / low frame rate | Expected — software GL, no GPU. Shrink the browser window; RViz is lighter than the Gazebo GUI. |
| RViz panels empty | Toggle Fixed Frame (`odom` ↔ `base_footprint`); confirm `/scan` with `ros2 topic hz /scan` inside the container. |
| `EPISODE`/`MODEL_DIR` errors from compose | Both are required for any `up` on this file (the `eval` service guards them file-wide). Export them per §2. |
| Want to force the old XQuartz path | `VIZ_VNC=0 DISPLAY_NUM=0 ... up viz` — but Gazebo's viewport will be blank (documented above); not recommended. |

## How it's wired (for reference)

- `docker/entrypoint.sh` — `VIZ=1` renders the server + GUI on an internal Xvfb with software GL.
  `VIZ_VNC=1` (default) starts `fluxbox` + `x11vnc` + `websockify`/noVNC to stream it; `VIZ_RVIZ=1`
  / `VIZ_CAPTURE=1` add RViz / the overhead camera; `VIZ_HEADLESS=1` skips the GUI (capture demo);
  `MODE=sim` holds the sim open without the agent. Default headless train/eval (k8s, compose
  `train`/`eval`) is unchanged.
- `.../launch/turtlebot3_drl_viz_stage9.launch.py` — the lean stage-9 launch + opt-in RViz,
  `robot_state_publisher`, runtime-spawned `/capture_cam`, and image bridge.
- `docker/Dockerfile.viz` — adds rviz2 / ros_gz_image / x11vnc / novnc / fluxbox over the published image.
- `docker/docker-compose.yml` — the `viz` service (profile `viz`), publishing ports 6080 + 5900.
