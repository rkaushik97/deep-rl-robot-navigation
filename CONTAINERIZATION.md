# Containerization & distributed training

Packages the full stack (ROS 2 Jazzy + Gazebo Harmonic + PyTorch) into one image so a
training/eval run is a single reproducible command — no host ROS install — and scales to
parallel runs on Kubernetes. Files live in [`docker/`](docker/) and [`k8s/`](k8s/).

---

## 1. Images

One [`docker/Dockerfile`](docker/Dockerfile), two tags (the torch wheel is chosen at build
time via the `TORCH_INDEX_URL` build-arg):

| Tag | Platforms | Use |
|-----|-----------|-----|
| `rkaushik97/turtlebot3-drl:cpu`  | linux/amd64, linux/arm64 | CPU training/eval; the **only** path on Apple-silicon (headless, software-rendered LiDAR) |
| `rkaushik97/turtlebot3-drl:cuda` | linux/amd64 | NVIDIA-GPU training (CUDA torch wheel; needs `nvidia-container-toolkit` on the host) |

The CUDA torch wheel bundles its own CUDA runtime, so both tags share the `ros:jazzy` base —
the GPU is engaged at runtime via the NVIDIA container runtime, not a `nvidia/cuda` base image.

> **Apple-silicon / MPS:** Metal is not visible inside a Linux container, so MPS never engages
> here. The `:cpu` arm64 image still *runs* headless on an M1/M2 (slowly, on CPU) — good for
> smoke-testing the pipeline. Real GPU training uses the `:cuda` image on Linux + NVIDIA.

### Build & push to Docker Hub
```bash
docker login -u rkaushik97
docker/build-and-push.sh build      # build both tags locally
docker/build-and-push.sh push       # build multi-arch + push to Docker Hub
```

---

## 2. Run with docker-compose

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

### Run the image directly
```bash
docker run --rm --shm-size=1g -e ALGO=ddpg -e DRL_MAX_EPISODES=2000 rkaushik97/turtlebot3-drl:cpu
docker run --rm --shm-size=1g --gpus all -e ALGO=td3 rkaushik97/turtlebot3-drl:cuda
```

### Entrypoint env vars
| Var | Default | Meaning |
|-----|---------|---------|
| `MODE` | `train` | `train` or `eval` |
| `ALGO` | `ddpg` | `ddpg` \| `td3` \| `sac` |
| `EXP` | — | experiment under `algorithms/<algo>/experiments/` |
| `DRL_MAX_EPISODES` | `0` | stop training after N eps (0 = unbounded) |
| `DRL_*` | — | any hyperparameter; **overrides** the algo's `config.sh` (for sweeps) |
| `MODEL_DIR`, `EPISODE`, `N_EPS` | — | eval-only: checkpoint dir, episode, #episodes (100) |

---

## 3. Kubernetes scaling (local: kind / minikube)

Run **parallel training jobs** — one pod per hyperparameter/algorithm config — all writing
to one shared volume, with automated result collection. Targets a single-node local cluster;
each job runs the whole sim stack inside its container via the image entrypoint.

### Prereqs
- A local cluster: `kind create cluster` **or** `minikube start`
- Image reachable by the cluster. kind: `kind load docker-image rkaushik97/turtlebot3-drl:cpu`.
  minikube: `minikube image load rkaushik97/turtlebot3-drl:cpu` (or let it pull from Docker Hub).

### Single job
```bash
kubectl apply -f k8s/00-namespace.yaml -f k8s/01-pvc.yaml -f k8s/02-train-job.yaml
kubectl -n drlnav logs -f job/drl-train-ddpg
```

### Parallel sweep
Edit the `CONFIGS` list in [`k8s/sweep.sh`](k8s/sweep.sh) (one line per run:
`name|algo|exp|DRL_* overrides`), then:
```bash
k8s/sweep.sh                 # launch all jobs (CPU image)
IMAGE_TAG=cuda k8s/sweep.sh  # GPU image; each pod requests one nvidia.com/gpu
kubectl -n drlnav get jobs -l app=drl-sweep -w
k8s/sweep.sh delete          # tear the sweep down
```
Each pod writes models under `model/<pod-host>/` and eval summaries under `results/` on the
shared `drl-results` PVC, so runs never collide.

### Collect results
```bash
k8s/collect-results.sh ./sweep-results   # copies the whole PVC to ./sweep-results
```
Spins a short-lived collector pod that mounts the PVC, `kubectl cp`s the tree out, deletes itself.

### Notes
- **PVC access mode** is `ReadWriteOnce` — fine on a single-node local cluster (many pods share
  it on that node). A multi-node cloud cluster would need `ReadWriteMany` (NFS/EFS) or per-job volumes.
- **Resources:** Gazebo is heavy; each pod requests 2 CPU / 2Gi. Don't over-subscribe a laptop.
- **GPU sweep** needs the NVIDIA device plugin installed on the node(s).

---

## Notes / gotchas
- **First-time local checkout:** `colcon build` before training/eval — the eval metrics added a
  `DrlStep.initial_distance` field that must be regenerated. (The image does this automatically.)
- **`--shm-size=1g`** when using `docker run` directly — gz/ROS use shared-memory transport
  (compose and the k8s manifests already provide this).
- Logs stream live to `docker logs` / `kubectl logs` (trainer runs unbuffered).
