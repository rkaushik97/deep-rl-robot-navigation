#!/usr/bin/env python3
"""Benchmark Apple-silicon MPS vs CPU for THIS project's actor/critic.

Why this exists: PyTorch's MPS (Metal) backend only works for a NATIVE macOS process — it is
unreachable from inside the Linux training container (see docker/Dockerfile). And the policy here
is a tiny MLP (state 44 -> 512 -> 512 -> action 2), so GPU kernel-launch overhead can make MPS
SLOWER than CPU at small batch sizes. This script measures it so the choice is data-driven, not
assumed. It depends only on torch (no ROS), so it runs natively:

    pip install torch            # native macOS arm64 wheel (bundles the MPS backend)
    python3 scripts/mps_benchmark.py

Dims mirror common/settings.py: STATE=44, HIDDEN=512, ACTION=2, BATCH=128.
"""
import time
import torch
import torch.nn as nn

STATE, HIDDEN, ACTION, BATCH = 44, 512, 2, 128
INFER_ITERS, TRAIN_ITERS, WARMUP = 2000, 1000, 50


class Actor(nn.Module):           # mirrors algorithms/ddpg/ddpg.py Actor
    def __init__(self):
        super().__init__()
        self.fa1 = nn.Linear(STATE, HIDDEN)
        self.fa2 = nn.Linear(HIDDEN, HIDDEN)
        self.fa3 = nn.Linear(HIDDEN, ACTION)

    def forward(self, s):
        x = torch.relu(self.fa1(s))
        x = torch.relu(self.fa2(x))
        return torch.tanh(self.fa3(x))


class Critic(nn.Module):          # mirrors algorithms/ddpg/ddpg.py Critic (concat variant)
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(STATE, HIDDEN // 2)
        self.l2 = nn.Linear(ACTION, HIDDEN // 2)
        self.l3 = nn.Linear(HIDDEN, HIDDEN)
        self.l4 = nn.Linear(HIDDEN, 1)

    def forward(self, s, a):
        x = torch.cat([torch.relu(self.l1(s)), torch.relu(self.l2(a))], dim=1)
        return self.l4(torch.relu(self.l3(x)))


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def bench_inference(dev):
    """Single-state actor forward — the actual eval/inference hot path (batch 1)."""
    actor = Actor().to(dev).eval()
    s = torch.randn(1, STATE, device=dev)
    with torch.no_grad():
        for _ in range(WARMUP):
            actor(s)
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(INFER_ITERS):
            actor(s)
        sync(dev)
    return (time.perf_counter() - t0) / INFER_ITERS * 1e6   # microseconds / step


def bench_train_step(dev):
    """One DDPG-style update: critic forward+backward + actor forward+backward (batch 128)."""
    actor, critic = Actor().to(dev), Critic().to(dev)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    s = torch.randn(BATCH, STATE, device=dev)
    a = torch.randn(BATCH, ACTION, device=dev)
    y = torch.randn(BATCH, 1, device=dev)

    def step():
        opt.zero_grad()
        q = critic(s, a)
        loss_c = ((q - y) ** 2).mean()
        loss_a = -critic(s, actor(s)).mean()
        (loss_c + loss_a).backward()
        opt.step()

    for _ in range(WARMUP):
        step()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(TRAIN_ITERS):
        step()
    sync(dev)
    return (time.perf_counter() - t0) / TRAIN_ITERS * 1e3   # milliseconds / update


def main():
    print(f"torch {torch.__version__}")
    devs = [torch.device("cpu")]
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    print(f"MPS available: {mps_ok}")
    if mps_ok:
        devs.append(torch.device("mps"))

    print(f"\nnet: actor {STATE}->{HIDDEN}->{HIDDEN}->{ACTION}, critic w/ hidden {HIDDEN}; batch {BATCH}")
    inf, tr = {}, {}
    for d in devs:
        inf[d.type] = bench_inference(d)
        tr[d.type] = bench_train_step(d)
        print(f"  [{d.type:4}] inference {inf[d.type]:8.1f} us/step    train-step {tr[d.type]:7.3f} ms/update")

    if "mps" in inf:
        print(f"\nMPS vs CPU  —  inference: {inf['cpu']/inf['mps']:.2f}x   "
              f"train-step: {tr['cpu']/tr['mps']:.2f}x   (>1 = MPS faster)")
        if inf['cpu'] < inf['mps']:
            print("note: CPU wins at batch 1 — kernel-launch overhead dominates for a net this small.")


if __name__ == "__main__":
    main()
