# TurtleBot3 deep-RL navigation — training/eval image.
#
# One Dockerfile, two flavours selected at build time via TORCH_INDEX_URL:
#   CPU  : --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu   -> tag :cpu
#   CUDA : --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 -> tag :cuda
# The CUDA torch wheel bundles its own CUDA runtime, so a plain ros:jazzy base is
# enough — GPU works at runtime through the NVIDIA container runtime, no nvidia/cuda
# base image needed. (Apple-silicon MPS is NOT available inside a Linux container;
# the CPU image is multi-arch so it still RUNS on arm64, just on CPU.)
#
# Build context is the REPO ROOT:  docker build -f docker/Dockerfile .
FROM ros:jazzy-ros-base

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG DEBIAN_FRONTEND=noninteractive

# --- system + ROS build/runtime deps -------------------------------------------------
# colcon/rosdep for the workspace; mesa for headless software GL (gz sensor rendering on
# GPU-less nodes); xvfb as a virtual display for the gz client-side rendering path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
        python3-pip python3-colcon-common-extensions python3-rosdep \
        libgl1-mesa-dri libegl1 libglx-mesa0 xvfb \
    && rm -rf /var/lib/apt/lists/*

# --- python deps (torch flavour chosen by build arg) ---------------------------------
# Ubuntu 24.04 marks the system env "externally managed" (PEP 668); we install into the
# ROS system interpreter on purpose so rclpy and torch share one interpreter.
RUN pip install --no-cache-dir --break-system-packages \
        --extra-index-url ${TORCH_INDEX_URL} \
        torch numpy matplotlib pyyaml

# --- workspace ------------------------------------------------------------------------
ENV WS=/opt/drlnav
WORKDIR ${WS}
COPY src ./src

# resolve ROS deps (ros_gz / gz-harmonic vendor pkgs, turtlebot3_description, ...)
RUN . /opt/ros/jazzy/setup.sh \
    && (rosdep init 2>/dev/null || true) && rosdep update \
    && apt-get update \
    && rosdep install --from-paths src --ignore-src -r -y \
    && rm -rf /var/lib/apt/lists/*

# build the workspace in a FRESH shell so it sources the env hooks that the gz-*-vendor
# packages installed above (otherwise CMake can't find gz-sim8 — the deps weren't on the
# prefix path when the earlier RUN sourced setup.sh).
RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install

# --- entrypoint + helpers -------------------------------------------------------------
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# DRLNAV_BASE_PATH is where runs read config.sh and write model/results; defaults to the
# baked-in workspace but is typically remapped to a mounted volume in compose/k8s.
ENV DRLNAV_BASE_PATH=${WS} \
    TURTLEBOT3_MODEL=burger \
    DRL_STAGE=9 \
    MODE=train \
    ALGO=ddpg

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
