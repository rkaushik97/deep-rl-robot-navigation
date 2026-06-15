# Apple-silicon GPU (MPS) on the M1 Pro

Can the M1 Pro GPU (PyTorch **MPS** / Metal backend) accelerate this project? Short answer:
**yes for training, no for inference — and you can't get it through Docker.** Measured, not assumed.

## The hard constraint

MPS talks to macOS Metal directly, so it is **only available to a NATIVE macOS process** — it is
unreachable from inside the Linux training container (noted in `docker/Dockerfile`). The device code
already handles this: `common/utilities.py:check_gpu()` returns `mps` when running natively, `cuda`
on a GPU box, else `cpu`. Nothing to change there.

The blocker for *using* MPS in the live RL loop: the agent depends on **rclpy / ROS 2 Jazzy**, which
has no native macOS build, and **Gazebo Harmonic** can't run natively on macOS either. So today the
whole pipeline runs in the Linux container → CPU only on a Mac.

## Measured: MPS vs CPU on this M1 Pro

`scripts/mps_benchmark.py` (torch-only, no ROS — runs natively) on the actual net dims
(actor 44→512→512→2, critic hidden 512, batch 128):

```
torch 2.12.0   MPS available: True
  [cpu ] inference     48.0 us/step    train-step  20.664 ms/update
  [mps ] inference    129.4 us/step    train-step   4.288 ms/update
MPS vs CPU  —  inference: 0.37x   train-step: 4.82x
```

- **Inference (batch 1)** — CPU **~2.7× faster**. The policy is a tiny MLP; per-call GPU
  kernel-launch overhead swamps the work. The live demo / `test_agent` does one forward per step,
  so **keep inference on CPU**.
- **Training (batch 128 update)** — MPS **~4.8× faster**. Batched forward+backward is enough work to
  amortise the overhead. **MPS is worth it for training.**

Reproduce:
```bash
pip install torch                                  # native arm64 wheel (bundles MPS)
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/mps_benchmark.py
```
(`KMP_DUPLICATE_LIB_OK=TRUE` works around a macOS duplicate-`libomp` abort from mixing OpenMP runtimes.)

## To actually train on MPS (hybrid; not yet implemented)

Because the sim must stay in Docker but MPS needs a native process, MPS-accelerated training requires
a **split**: native agent ↔ Dockerized sim, talking over ROS 2 DDS.

1. Native ROS 2 Jazzy on macOS arm64 via **RoboStack** (conda-forge): `rclpy` + `turtlebot3_msgs`.
2. Native PyTorch (MPS) running `train_agent`, device `mps` (already auto-selected).
3. Gazebo + the ros_gz bridge + environment/goals in the Linux container (as today).
4. Same `ROS_DOMAIN_ID`; bridge DDS discovery across Docker Desktop's NAT (host networking or
   explicit discovery — this is the fiddly part on macOS).

This is a real project (RoboStack env + cross-boundary DDS discovery is the risky bit), not a flip of
a flag. For a single training run it is usually faster to just use a CUDA box / cloud GPU (the
`:cuda` image) than to stand this up — but the ~4.8× training speedup above is the payoff if you do.

## Bottom line

- MPS works natively on the M1 Pro and is **~4.8× faster for training**, **~2.7× slower for inference**.
- Inference/eval/the visual demo: stay on CPU (in Docker, as in [MACOS_M1.md](MACOS_M1.md)).
- Training on MPS needs the native-agent + Docker-sim hybrid above; the device selection is ready,
  the plumbing is not.
