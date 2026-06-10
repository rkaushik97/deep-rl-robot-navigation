#!/usr/bin/env python3
"""Tracking plots for the live runs. Re-runnable — regenerates PNGs from _metrics.tsv.

  python3 scripts/plot_tracking.py

Outputs (experiment_plots/):
  dqn_tracking.png              - DQN reward-P run: success, reward, Q-loss
  ddpg_curriculum_on_vs_off.png - DDPG curriculum ON vs OFF: success + reward
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(BASE, "src/turtlebot3_drl/model/fond-filly")
OUT = os.path.join(BASE, "experiment_plots")
os.makedirs(OUT, exist_ok=True)

# runs of interest
DQN      = os.path.join(MODEL, "dqn_2_stage_9")
DDPG_ON  = os.path.join(MODEL, "ddpg_0_stage_9")    # curriculum ON  (MIN=1.0, MAX=2.8)
DDPG_OFF = os.path.join(MODEL, "ddpg_47_stage_9")   # curriculum OFF (89% reference run)
SAC_VH   = os.path.join(MODEL, "sac_0_stage_9")     # reward VH (exp007); target: beat reward V's 84%

COL = {"episode": 0, "reward": 3, "ma100": 4, "loss_critic": 5, "loss_actor": 6}


def load(run_dir):
    """Return dict of column-name -> list[float]. Empty lists if no data."""
    path = os.path.join(run_dir, "_metrics.tsv")
    out = {k: [] for k in COL}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        r = csv.reader(f, delimiter="\t")
        next(r, None)  # header
        for row in r:
            if len(row) <= COL["loss_actor"]:
                continue
            try:
                for k, i in COL.items():
                    out[k].append(float(row[i]))
            except ValueError:
                continue
    return out


def rolling(xs, w=50):
    if not xs:
        return xs
    out, acc = [], []
    for x in xs:
        acc.append(x)
        if len(acc) > w:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def plot_dqn():
    d = load(DQN)
    if not d["episode"]:
        print("DQN: no episodes logged yet — skipping")
        return
    ep = d["episode"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, d["ma100"], color="tab:blue")
    ax[0].set_title("Success rate (MA100)"); ax[0].set_xlabel("episode"); ax[0].set_ylabel("%"); ax[0].set_ylim(0, 100)
    ax[1].plot(ep, d["reward"], color="0.8", lw=0.8)
    ax[1].plot(ep, rolling(d["reward"]), color="tab:green")
    ax[1].set_title("Episode reward (raw + MA50)"); ax[1].set_xlabel("episode")
    ax[2].plot(ep, d["loss_actor"], color="tab:red")
    ax[2].set_title("Q-loss"); ax[2].set_xlabel("episode")
    fig.suptitle(f"DQN (reward P) — ep{int(ep[-1])}, MA100={d['ma100'][-1]:.0f}%")
    fig.tight_layout()
    p = os.path.join(OUT, "dqn_tracking.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print("wrote", p)


def plot_ddpg_on_off():
    on, off = load(DDPG_ON), load(DDPG_OFF)
    if not on["episode"]:
        print("DDPG-ON: no episodes logged yet — skipping")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(off["episode"], off["ma100"], color="tab:gray", label=f"OFF (ep{int(off['episode'][-1])})" if off["episode"] else "OFF")
    ax[0].plot(on["episode"], on["ma100"], color="tab:orange", label=f"ON (ep{int(on['episode'][-1])})")
    ax[0].set_title("Success rate (MA100)"); ax[0].set_xlabel("episode"); ax[0].set_ylabel("%"); ax[0].set_ylim(0, 100); ax[0].legend()
    ax[1].plot(off["episode"], rolling(off["reward"]), color="tab:gray", label="OFF")
    ax[1].plot(on["episode"], rolling(on["reward"]), color="tab:orange", label="ON")
    ax[1].set_title("Episode reward (MA50)"); ax[1].set_xlabel("episode"); ax[1].legend()
    fig.suptitle("DDPG curriculum ON vs OFF  (training-goal distributions differ; "
                 "apples-to-apples = final test_agent eval)")
    fig.tight_layout()
    p = os.path.join(OUT, "ddpg_curriculum_on_vs_off.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print("wrote", p)


def plot_sac_vh():
    d = load(SAC_VH)
    if not d["episode"]:
        print("SAC-VH: no episodes logged yet — skipping")
        return
    ep = d["episode"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, d["ma100"], color="tab:purple")
    ax[0].axhline(84, ls="--", color="tab:green", lw=1, label="reward V (84%)")
    ax[0].set_title("Success rate (MA100)"); ax[0].set_xlabel("episode"); ax[0].set_ylabel("%"); ax[0].set_ylim(0, 100); ax[0].legend()
    ax[1].plot(ep, d["reward"], color="0.8", lw=0.8)
    ax[1].plot(ep, rolling(d["reward"]), color="tab:green")
    ax[1].set_title("Episode reward (raw + MA50)"); ax[1].set_xlabel("episode")
    ax[2].plot(ep, d["loss_critic"], color="tab:red")
    ax[2].set_title("Critic loss"); ax[2].set_xlabel("episode")
    fig.suptitle(f"SAC reward VH (exp007) — ep{int(ep[-1])}, MA100={d['ma100'][-1]:.0f}%  (target: beat V's 84%)")
    fig.tight_layout()
    p = os.path.join(OUT, "sac_vh_tracking.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    plot_dqn()
    plot_ddpg_on_off()
    plot_sac_vh()
