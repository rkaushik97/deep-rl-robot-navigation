# Containerization & distributed training

The whole stack (ROS 2 Jazzy + Gazebo Harmonic + PyTorch) runs in **one image**, so a training
or eval run is a single command with no host ROS install — and the same image scales to parallel
runs on Kubernetes. Everything is configured with env vars; outputs persist to mounted volumes.

## Images
Two tags, built from one [`docker/Dockerfile`](docker/Dockerfile):

| Tag | For |
|-----|-----|
| `kaushik48/turtlebot3-drl:cpu`  | CPU (amd64 + arm64); also the only path on Apple-silicon Macs (headless, slow) |
| `kaushik48/turtlebot3-drl:cuda` | NVIDIA GPU (needs `nvidia-container-toolkit` on the host) |

```bash
docker login -u kaushik48
docker/build-and-push.sh build     # build both tags locally
docker/build-and-push.sh push      # build + push to Docker Hub
```

## Run one container
```bash
# CPU training (DDPG), stop after 2000 episodes
docker run --rm --shm-size=1g -e ALGO=ddpg -e DRL_MAX_EPISODES=2000 kaushik48/turtlebot3-drl:cpu
# GPU training (TD3)
docker run --rm --shm-size=1g --gpus all -e ALGO=td3 kaushik48/turtlebot3-drl:cuda
```
Or with compose (volumes for results baked in):
```bash
ALGO=ddpg DRL_MAX_EPISODES=2000 docker compose -f docker/docker-compose.yml up train
ALGO=td3 docker compose -f docker/docker-compose.yml --profile gpu up train-gpu
```

**Env vars:** `MODE` (train|eval), `ALGO` (ddpg|td3|sac), `EXP` (experiment name),
`DRL_MAX_EPISODES` (0 = unbounded), any `DRL_*` (overrides the algo's `config.sh` — the sweep knob).
Eval also needs `MODEL_DIR`, `EPISODE`, `N_EPS`.

## Parallel sweep on Kubernetes (local kind/minikube)
One pod per config, all writing to one shared volume.
```bash
kind create cluster && kind load docker-image kaushik48/turtlebot3-drl:cpu   # one-time setup
k8s/sweep.sh                 # launch the sweep (edit the CONFIGS list inside first)
kubectl -n drlnav get jobs -l app=drl-sweep -w
k8s/collect-results.sh ./sweep-results    # pull all results off the cluster
k8s/sweep.sh delete          # tear down
```

## Gotchas
- **`docker run` needs `--shm-size=1g`** (gz/ROS use shared memory; compose & k8s already set it).
- **Building locally outside Docker?** Run `colcon build` first — the eval metrics added a new
  `DrlStep` message field (the image rebuilds it automatically).
- Logs stream live to `docker logs` / `kubectl logs`.
