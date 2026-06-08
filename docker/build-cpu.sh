#!/usr/bin/env bash
# Build and (optionally) push the CPU TurtleBot3 deep-RL image to Docker Hub.
#
#   docker/build-cpu.sh build        # build kaushik48/turtlebot3-drl:cpu locally
#   docker/build-cpu.sh push         # build + push (multi-arch amd64+arm64)
#
# Tag produced:
#   kaushik48/turtlebot3-drl:cpu    runs headless on x86 AND on Apple-silicon
#                                    (M1/M2) under Docker, CPU-only. Multi-arch
#                                    (linux/amd64, linux/arm64) only when pushing.
set -euo pipefail

REPO=${REPO:-kaushik48/turtlebot3-drl}
CPU_INDEX=https://download.pytorch.org/whl/cpu
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

echo "== building $REPO:cpu =="
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

echo "done: $REPO:cpu  (action=$ACTION)"
[ "$ACTION" = "push" ] && echo "NOTE: 'docker login' as kaushik48 must have been run first."
