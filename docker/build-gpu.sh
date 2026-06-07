#!/usr/bin/env bash
# Build and (optionally) push the CUDA TurtleBot3 deep-RL image to Docker Hub.
#
#   docker/build-gpu.sh build        # build kaushik48/turtlebot3-drl:cuda locally
#   docker/build-gpu.sh push         # build + push to Docker Hub
#
# Tag produced:
#   kaushik48/turtlebot3-drl:cuda   linux/amd64, CUDA torch wheel for NVIDIA GPU
#                                    training (needs nvidia-container-toolkit on host).
set -euo pipefail

REPO=${REPO:-kaushik48/turtlebot3-drl}
CUDA_INDEX=https://download.pytorch.org/whl/cu124
CTX="$(cd "$(dirname "$0")/.." && pwd)"   # repo root = build context
DOCKERFILE="$CTX/docker/Dockerfile"
ACTION=${1:-build}

case "$ACTION" in
  push) PUSH="--push"; LOAD="" ;;
  build) PUSH=""; LOAD="--load" ;;
  *) echo "usage: $0 [build|push]"; exit 1 ;;
esac

# buildx builder. Reuse if it already exists.
docker buildx inspect drlbuilder >/dev/null 2>&1 || docker buildx create --name drlbuilder --use
docker buildx use drlbuilder

echo "== building $REPO:cuda (linux/amd64) =="
docker buildx build ${PUSH:-$LOAD} --platform linux/amd64 \
  -f "$DOCKERFILE" --build-arg TORCH_INDEX_URL="$CUDA_INDEX" \
  -t "$REPO:cuda" "$CTX"

echo "done: $REPO:cuda  (action=$ACTION)"
[ "$ACTION" = "push" ] && echo "NOTE: 'docker login' as kaushik48 must have been run first."
