# Containerization & distributed training

The whole stack (ROS 2 Jazzy, Gazebo Harmonic, PyTorch) runs in one image. You can train, eval, or
run a Kubernetes pod with a single command. Everything is driven by env vars.

## Images
Docker Hub: **https://hub.docker.com/r/kaushik48/turtlebot3-drl**

| Tag | For | Pull |
|-----|-----|------|
| `kaushik48/turtlebot3-drl:cpu`  | CPU. Works on Apple silicon (slow, headless). | `docker pull kaushik48/turtlebot3-drl:cpu` |
| `kaushik48/turtlebot3-drl:cuda` | NVIDIA GPU. Needs `nvidia-container-toolkit`. | `docker pull kaushik48/turtlebot3-drl:cuda` |

Build them yourself instead of pulling:
```bash
docker login -u kaushik48
docker/build-cpu.sh build      # build the cpu image (use 'push' to publish)
docker/build-gpu.sh build      # build the cuda image (use 'push' to publish)
```

## Env vars
`MODE` is train or eval. `ALGO` is ddpg, td3, or sac. `EXP` is an experiment name. `DRL_MAX_EPISODES`
caps training (0 means unbounded). Any `DRL_*` overrides the algo config. Eval also needs `MODEL_DIR`,
`EPISODE`, and `N_EPS`. With `docker run` always pass `--shm-size=1g`. For GPU add `--gpus all`.

## Train
```bash
# CPU. Train DDPG and stop after 2000 episodes.
docker run --rm --shm-size=1g -e ALGO=ddpg -e DRL_MAX_EPISODES=2000 kaushik48/turtlebot3-drl:cpu

# GPU. Train TD3 on an NVIDIA GPU.
docker run --rm --shm-size=1g --gpus all -e ALGO=td3 kaushik48/turtlebot3-drl:cuda
```

## Eval
`MODEL_DIR` is a folder with `actor_stage<S>_episode<EP>.pt`. `EPISODE` must be a plain number.
```bash
# CPU. Eval the best DDPG checkpoint over 100 random goal episodes.
docker run --rm --shm-size=1g \
  -e MODE=eval -e ALGO=ddpg -e EPISODE=4000 -e N_EPS=100 -e MODEL_DIR=/checkpoint \
  -v "$PWD/src/turtlebot3_drl/model/fond-filly/ddpg_47_stage_9:/checkpoint:ro" \
  kaushik48/turtlebot3-drl:cpu

# GPU. Same eval on an NVIDIA GPU.
docker run --rm --shm-size=1g --gpus all \
  -e MODE=eval -e ALGO=ddpg -e EPISODE=4000 -e N_EPS=100 -e MODEL_DIR=/checkpoint \
  -v "$PWD/src/turtlebot3_drl/model/fond-filly/ddpg_47_stage_9:/checkpoint:ro" \
  kaushik48/turtlebot3-drl:cuda
```
The success rate and metrics print to the logs.

## Compose
```bash
# CPU train. Outputs persist to named volumes.
ALGO=ddpg DRL_MAX_EPISODES=2000 docker compose -f docker/docker-compose.yml up train

# GPU train.
ALGO=td3 docker compose -f docker/docker-compose.yml --profile gpu up train-gpu

# CPU eval.
MODEL_DIR="$PWD/src/turtlebot3_drl/model/fond-filly/ddpg_47_stage_9" EPISODE=4000 \
  docker compose -f docker/docker-compose.yml --profile eval up eval
```

## Kubernetes pod (local kind or minikube)

### CPU
```bash
kind create cluster --name drlnav                                       # start a local cluster
kubectl wait --for=condition=Ready node/drlnav-control-plane --timeout=120s   # wait for the node
docker pull kaushik48/turtlebot3-drl:cpu                                # get the image
kind load docker-image kaushik48/turtlebot3-drl:cpu --name drlnav       # load it into the cluster
kubectl apply -f k8s/00-namespace.yaml -f k8s/01-pvc.yaml -f k8s/02-train-job.yaml   # deploy the pod
kubectl -n drlnav get pod -l app=drl-train -w                           # watch it reach Running
kubectl -n drlnav logs -f job/drl-train-ddpg                            # follow the logs
kubectl delete job drl-train-ddpg -n drlnav                             # stop the run
kind delete cluster --name drlnav                                       # remove the cluster
```
Success looks like a `making new model dir` line in the logs.

### GPU
GPU pods need a cluster whose nodes expose GPUs (the NVIDIA device plugin). Plain kind cannot do this.
```bash
kind load docker-image kaushik48/turtlebot3-drl:cuda --name <cluster>   # load the cuda image
IMAGE_TAG=cuda k8s/sweep.sh                                             # each pod requests one GPU
```

### Parallel sweep
```bash
k8s/sweep.sh                                  # launch one pod per config (edit CONFIGS first)
kubectl -n drlnav get jobs -l app=drl-sweep -w   # watch all jobs
k8s/collect-results.sh ./sweep-results        # pull every result off the cluster
k8s/sweep.sh delete                           # tear the sweep down
```

## Using pods for experiments
- Each pod runs one full training run by itself.
- Set `ALGO` to ddpg, td3, or sac to choose the algorithm.
- Add any `DRL_*` var to change a hyperparameter for that run.
- The sweep runs many pods at the same time, one per line in the `CONFIGS` list.
- Edit `CONFIGS` in `k8s/sweep.sh` to pick the runs.
- Every pod writes to the shared `drl-results` volume.
- Each pod saves into its own folder named after the pod.
- Run `k8s/collect-results.sh` to copy all results to your machine.
</content>
