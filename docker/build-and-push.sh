#!/usr/bin/env bash
# Build and (optionally) push the TurtleBot3 deep-RL images to Docker Hub.
#
#   docker/build-and-push.sh build        # build both tags locally
#   docker/build-and-push.sh push         # build + push to kaushik48/turtlebot3-drl
#
# Tags produced:
#   kaushik48/turtlebot3-drl:cpu    multi-arch (linux/amd64, linux/arm64) — runs headless
#                                    on x86 AND on Apple-silicon (M1/M2) under Docker, CPU-only
#   kaushik48/turtlebot3-drl:cuda   linux/amd64, CUDA torch wheel for NVIDIA GPU training
set -euo pipefail

REPO=${REPO:-kaushik48/turtlebot3-drl}
CPU_INDEX=https://download.pytorch.org/whl/cpu
CUDA_INDEX=https://download.pytorch.org/whl/cu124
CTX="$(cd "$(dirname "$0")/.." && pwd)"   # repo root = build context
DOCKERFILE="$CTX/docker/Dockerfile"
ACTION=${1:-build}

case "$ACTION" in
  push) PUSH="--push"; LOAD="" ;;
  build) PUSH=""; LOAD="--load" ;;
  *) echo "usage: $0 [build|push]"; exit 1 ;;
esac

# buildx builder (needed for multi-arch). Reuse if it already exists.
docker buildx inspect drlbuilder >/dev/null 2>&1 || docker buildx create --name drlbuilder --use
docker buildx use drlbuilder

echo "== building $REPO:cpu (multi-arch amd64+arm64) =="
# --load only supports a single platform; build arm64+amd64 only when pushing.
if [ "$ACTION" = "push" ]; then
  docker buildx build $PUSH --platform linux/amd64,linux/arm64 \
    -f "$DOCKERFILE" --build-arg TORCH_INDEX_URL="$CPU_INDEX" \
    -t "$REPO:cpu" "$CTX"
else
  docker buildx build $LOAD \
    -f "$DOCKERFILE" --build-arg TORCH_INDEX_URL="$CPU_INDEX" \
    -t "$REPO:cpu" "$CTX"
fi

echo "== building $REPO:cuda (linux/amd64) =="
docker buildx build ${PUSH:-$LOAD} --platform linux/amd64 \
  -f "$DOCKERFILE" --build-arg TORCH_INDEX_URL="$CUDA_INDEX" \
  -t "$REPO:cuda" "$CTX"

echo "done: $REPO:cpu  $REPO:cuda  (action=$ACTION)"
[ "$ACTION" = "push" ] && echo "NOTE: 'docker login' as kaushik48 must have been run first."
