# Running on Apple Silicon (M1 Pro MacBook) — verified

This stack (ROS 2 Jazzy + Gazebo Harmonic + PyTorch) is Linux-only, but it runs on an M1 Pro
MacBook **inside Docker** — training/eval headless, and the full GUI (Gazebo + RViz + the SAC
inference demo) streamed to a browser over VNC. Everything below was verified on an M1 Pro
(arm64, macOS, Docker Desktop, no GPU passthrough → mesa `llvmpipe` software GL).

## What works

| Task | How | Notes |
|------|-----|-------|
| **Training** | `kaushik48/turtlebot3-drl:cpu` (multi-arch) | CPU only, slow but correct |
| **Inference / eval** | `:cpu` image, `MODE=eval` | headless, prints success/metrics |
| **Gazebo GUI (live)** | `:viz` image, noVNC | software GL, ~10–15 fps |
| **RViz (live)** | `VIZ_RVIZ=1` | `/scan` `/odom` `/tf` over the ros_gz bridge |
| **SAC inference demo** | `VIZ_DEMO=1`, SAC ep4700 | the side-by-side HUD demo (Gazebo + RViz overlay) |
| **Activations GIF** | `VIZ_CAPTURE=1 VIZ_HEADLESS=1` | offline render, no display needed |

## What does NOT work (and why)

**XQuartz / X11 forwarding for the Gazebo GUI.** You cannot push Gazebo Harmonic's OpenGL over the
network to XQuartz:
- direct GLX → `GLXBadFBConfig`, the renderer crashes;
- indirect GLX (`enable_iglx`) → window opens but Qt logs `Unrecognized OpenGL version` and the
  viewport stays blank (Qt Quick needs a modern GL context that network GLX can't provide).

The fix is **VNC**: render in-container on `llvmpipe` software GL (which gives OpenGL 4.5, so OGRE2
works) and stream the finished framebuffer. No GL crosses the network. View in a browser tab.

## Quick start

```bash
# 1. build the viz image (thin overlay on the published headless image)
docker build -f docker/Dockerfile.viz -t kaushik48/turtlebot3-drl:viz .

# 2. open the stream
#    http://localhost:6080/vnc.html   (Connect)   — appears once the sim warms up (~45 s)
```

### Live SAC inference demo (the HUD: Algorithm / Episode / Success / Rate)

```bash
mkdir -p /tmp/sac_demo
cp SAC/actor_stage9_episode4700_rewardV_best.pt /tmp/sac_demo/actor_stage9_episode4700.pt
ALGO=sac MODEL_DIR=/tmp/sac_demo EPISODE=4700 N_EPS=100 VIZ_DEMO=1 \
  docker compose -f docker/docker-compose.yml --profile viz up viz
```
In the browser: in RViz close the **Displays** dock (Panels → untick Displays) so the top-down view
fills and the stats text centers, then arrange Gazebo (left) / RViz (right) side by side.
`viz_helper.py` computes Success/Failure/Rate from goal changes + min-distance < 0.30 m, so over the
full 100 episodes it lands near the recorded **73%**.

### Plain Gazebo, or Gazebo + RViz

```bash
mkdir -p /tmp/ddpg_demo
cp experiments/replications/ddpg/actor_stage9_episode4000_best.pt /tmp/ddpg_demo/actor_stage9_episode4000.pt
MODEL_DIR=/tmp/ddpg_demo EPISODE=4000 N_EPS=50 \
  docker compose -f docker/docker-compose.yml --profile viz up viz     # Gazebo only
# add VIZ_RVIZ=1 for Gazebo + RViz
```

## Performance

No GPU is available inside the Docker VM on macOS, so all 3D rendering uses the mesa `llvmpipe`
software rasteriser. Expect ~10–15 fps in the streamed desktop and slower-than-realtime sim — fine
for watching/recording, not for fast iteration. Shrink the browser window or use RViz (lighter than
the Gazebo GUI) if it drags.

See [VISUALIZATION.md](VISUALIZATION.md) for the full reference (all flags, the X11 fallback, the
offline activations GIF, and how the pieces are wired).
