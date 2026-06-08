# Containerization & distributed training

The whole stack (ROS 2 Jazzy + Gazebo Harmonic + PyTorch) runs in **one image**. Train, eval, or
run a Kubernetes pod with a single command — no host ROS install. Everything is env-var driven.

## Images
Docker Hub: **https://hub.docker.com/r/kaushik48/turtlebot3-drl**

| Tag | For | Pull |
|-----|-----|------|
| `kaushik48/turtlebot3-drl:cpu`  | CPU — amd64 **+ arm64** (Apple silicon, headless/slow) | `docker pull kaushik48/turtlebot3-drl:cpu` |
| `kaushik48/turtlebot3-drl:cuda` | NVIDIA GPU (needs `nvidia-container-toolkit` on host) | `docker pull kaushik48/turtlebot3-drl:cuda` |

Build locally instead of pulling:
```bash
docker login -u kaushik48
docker/build-cpu.sh build      # or: push  (multi-arch amd64+arm64)
docker/build-gpu.sh build      # or: push  (:cuda, amd64)
```

## Env vars
`MODE` train|eval · `ALGO` ddpg|td3|sac · `EXP` experiment name · `DRL_MAX_EPISODES` (0=unbounded) ·
any `DRL_*` overrides the algo's `config.sh`. **Eval also:** `MODEL_DIR` `EPISODE` `N_EPS`.
> `docker run` needs `--shm-size=1g` (gz/ROS shared memory). GPU adds `--gpus all` + the `:cuda` tag.

## Train
```bash
# CPU — DDPG, stop after 2000 episodes
docker run --rm --shm-size=1g -e ALGO=ddpg -e DRL_MAX_EPISODES=2000 kaushik48/turtlebot3-drl:cpu

# GPU — TD3
docker run --rm --shm-size=1g --gpus all -e ALGO=td3 kaushik48/turtlebot3-drl:cuda
```

## Eval
Point `MODEL_DIR` at a session dir holding `actor_stage<S>_episode<EP>.pt`. **`EPISODE` must be a
plain integer** (the archived `experiments/replications/*_best.pt` files are NOT loadable directly —
use the byte-identical plain-int source below).
```bash
# CPU — best DDPG checkpoint (fond-filly ep4000), 100 random-goal episodes
docker run --rm --shm-size=1g \
  -e MODE=eval -e ALGO=ddpg -e EPISODE=4000 -e N_EPS=100 -e MODEL_DIR=/checkpoint \
  -v "$PWD/src/turtlebot3_drl/model/fond-filly/ddpg_47_stage_9:/checkpoint:ro" \
  kaushik48/turtlebot3-drl:cpu

# GPU — same, add --gpus all and the :cuda tag
docker run --rm --shm-size=1g --gpus all \
  -e MODE=eval -e ALGO=ddpg -e EPISODE=4000 -e N_EPS=100 -e MODEL_DIR=/checkpoint \
  -v "$PWD/src/turtlebot3_drl/model/fond-filly/ddpg_47_stage_9:/checkpoint:ro" \
  kaushik48/turtlebot3-drl:cuda
```
Summary prints to the logs (`success/collision/timeout` + metrics).

## Compose (volumes baked in; outputs persist to named volumes)
```bash
ALGO=ddpg DRL_MAX_EPISODES=2000 docker compose -f docker/docker-compose.yml up train        # CPU train
ALGO=td3 docker compose -f docker/docker-compose.yml --profile gpu up train-gpu             # GPU train
MODEL_DIR="$PWD/src/turtlebot3_drl/model/fond-filly/ddpg_47_stage_9" EPISODE=4000 \
  docker compose -f docker/docker-compose.yml --profile eval up eval                        # CPU eval
```

## Kubernetes pod (local kind/minikube)
One Job = one pod = one full train/eval run, writing to a shared `drl-results` PVC (pod-unique subdir).

### CPU
```bash
kind create cluster --name drlnav
kubectl wait --for=condition=Ready node/drlnav-control-plane --timeout=120s
docker pull kaushik48/turtlebot3-drl:cpu
kind load docker-image kaushik48/turtlebot3-drl:cpu --name drlnav      # load into the cluster
kubectl apply -f k8s/00-namespace.yaml -f k8s/01-pvc.yaml -f k8s/02-train-job.yaml

# watch it run (success = "making new model dir" in the logs, ~40-60s after Running)
kubectl -n drlnav get pod -l app=drl-train -w
kubectl -n drlnav logs -f job/drl-train-ddpg

# teardown
kubectl delete job drl-train-ddpg -n drlnav
kind delete cluster --name drlnav
```

### GPU
Needs a **GPU-enabled cluster** (node with `nvidia-container-toolkit` + the NVIDIA device plugin —
plain kind does not expose GPUs). Then each pod requests one GPU automatically:
```bash
kind load docker-image kaushik48/turtlebot3-drl:cuda --name <cluster>
IMAGE_TAG=cuda k8s/sweep.sh        # adds  resources.limits.nvidia.com/gpu: "1"  per pod
```

### Parallel sweep (one pod per config)
```bash
k8s/sweep.sh                                  # edit the CONFIGS list inside first
kubectl -n drlnav get jobs -l app=drl-sweep -w
k8s/collect-results.sh ./sweep-results        # pull all results off the cluster
k8s/sweep.sh delete                           # tear the sweep down
```

## Gotchas
- **`docker run` needs `--shm-size=1g`** (compose & k8s already set it).
- **Eval `EPISODE` is a plain integer** — `4000`, not `4000_best`. The `*_best.pt` files under
  `experiments/replications/` are display copies; eval the identical plain-int source.
- **Apple silicon** runs only `:cpu` (MPS is unavailable inside a container) — headless & slow, fine
  for smoke tests.
- **Building outside Docker?** Run `colcon build` first (the eval metrics added a `DrlStep` field;
  the image rebuilds it automatically).
- Logs stream live to `docker logs` / `kubectl logs`.
</content>
