# Containerized training & evaluation

Packages the full stack (ROS 2 Jazzy + Gazebo Harmonic + PyTorch) into one image so a
training/eval run is a single reproducible command — no host ROS install needed.

## Images
One Dockerfile, two tags (`docker/build-and-push.sh`):

| Tag | Platforms | Use |
|-----|-----------|-----|
| `rkaushik97/turtlebot3-drl:cpu`  | linux/amd64, linux/arm64 | CPU training/eval; the **only** path on Apple-silicon (headless, software-rendered LiDAR) |
| `rkaushik97/turtlebot3-drl:cuda` | linux/amd64 | NVIDIA-GPU training (torch CUDA wheel; needs `nvidia-container-toolkit` on the host) |

The CUDA torch wheel bundles its own CUDA runtime, so both tags share the `ros:jazzy` base —
GPU is engaged at runtime via the NVIDIA container runtime, not a `nvidia/cuda` base image.

> **Apple-silicon / MPS:** Metal is not visible inside a Linux container, so MPS never
> engages here. The `:cpu` arm64 image still *runs* headless on an M1/M2 (slowly, on CPU) —
> good for smoke-testing the pipeline. Real GPU training uses the `:cuda` image on Linux+NVIDIA.

## Build & push
```bash
docker login -u rkaushik97
docker/build-and-push.sh build      # build both tags locally
docker/build-and-push.sh push       # build multi-arch + push to Docker Hub
```

## Run with docker-compose
All knobs are env vars (recorded in compose for reproducibility); outputs persist in the
`drl-model` / `drl-results` named volumes.
```bash
# CPU training (DDPG), stop after 2000 episodes
ALGO=ddpg DRL_MAX_EPISODES=2000 docker compose -f docker/docker-compose.yml up train

# GPU training (TD3) — uses the :cuda image and reserves one GPU
ALGO=td3 docker compose -f docker/docker-compose.yml --profile gpu up train-gpu

# per-run hyperparameter override (wins over the algo's config.sh)
ALGO=ddpg DRL_LR=0.001 docker compose -f docker/docker-compose.yml up train

# evaluate a checkpoint: 100 random-goal episodes (reference test_agent methodology)
MODEL_DIR=/abs/path/to/session EPISODE=8000 \
  docker compose -f docker/docker-compose.yml --profile eval up eval
```

## Run the image directly
```bash
docker run --rm -e ALGO=ddpg -e DRL_MAX_EPISODES=2000 rkaushik97/turtlebot3-drl:cpu
docker run --rm --gpus all -e ALGO=td3 rkaushik97/turtlebot3-drl:cuda
```

## Entrypoint env vars
| Var | Default | Meaning |
|-----|---------|---------|
| `MODE` | `train` | `train` or `eval` |
| `ALGO` | `ddpg` | `ddpg` \| `td3` \| `sac` |
| `EXP` | — | experiment under `algorithms/<algo>/experiments/` |
| `DRL_MAX_EPISODES` | `0` | stop training after N eps (0 = unbounded) |
| `DRL_*` | — | any hyperparameter; **overrides** the algo's `config.sh` (for sweeps) |
| `MODEL_DIR`, `EPISODE`, `N_EPS` | — | eval-only: checkpoint dir, episode, #episodes (100) |

For multi-node / parallel sweeps, see [`../k8s/`](../k8s/).
