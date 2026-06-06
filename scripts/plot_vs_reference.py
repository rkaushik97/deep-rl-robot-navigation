#!/usr/bin/env python3
# Training-curve comparison: our DDPG/TD3/SAC MA100-success vs the reference repo's,
# at matched episodes. Reference curves are computed from each reference _train_stage9 log
# (outcome col -> 100-ep moving avg of outcome==SUCCESS). Ours come from _metrics.tsv.
import os, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/kaushik/project/deep-rl-robot-navigation'
REFM = '/home/kaushik/project/turtlebot3_deepRL/src/turtlebot3_drl/model'
OUT = f'{BASE}/experiments/training_vs_reference.png'
W = 100

REF = {
    'ddpg': glob.glob(f'{REFM}/examples/ddpg_0_stage9/_train_stage9_*.txt'),
    'td3':  glob.glob(f'{REFM}/examples/td3_0_stage9/_train_stage9_*.txt'),
    'sac':  glob.glob(f'{REFM}/64891e20d104/sac_5_stage_9/_train_stage9_*.txt'),
}
COLOR = {'ddpg': '#ff7f0e', 'td3': '#2ca02c', 'sac': '#d62728'}
# reference test_agent numbers (the published benchmark these runs aim at)
TARGET = {'ddpg': 84, 'td3': 74, 'sac': 82}


def ma(success):
    s = np.asarray(success, float)
    return np.array([100.0 * s[max(0, i + 1 - W):i + 1].mean() for i in range(len(s))])


def ref_curve(path):
    eps, succ = [], []
    for l in open(path):
        p = l.split(',')
        if not p[0].strip().isdigit():
            continue
        eps.append(int(p[0]))
        succ.append(1.0 if int(float(p[2])) == 1 else 0.0)   # col 2 = outcome; 1 = SUCCESS
    return np.array(eps), ma(succ)


def our_curve(metrics):
    eps, m = [], []
    for l in open(metrics):
        p = l.rstrip('\n').split('\t')
        if not p or not p[0].isdigit():
            continue
        eps.append(int(p[0])); m.append(float(p[4]))         # col 4 = ma100_success
    return np.array(eps), np.array(m)


fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
for i, a in enumerate(['ddpg', 'td3', 'sac']):
    rp = REF[a][0] if REF[a] else None
    if rp:
        re_e, re_m = ref_curve(rp)
        ax[i].plot(re_e, re_m, color='gray', lw=2.2, alpha=0.8,
                   label=f'reference ({len(re_e)} eps, final {re_m[-1]:.0f}%)')
    cands = [d for d in glob.glob(f'{BASE}/src/turtlebot3_drl/model/fond-filly/{a}_*_stage_9')
             if os.path.exists(f'{d}/_metrics.tsv')]
    sdir = max(cands, key=lambda d: os.path.getmtime(f'{d}/_metrics.tsv')) if cands else None
    if sdir:
        oe, om = our_curve(f'{sdir}/_metrics.tsv')
        ax[i].plot(oe, om, color=COLOR[a], lw=2.2,
                   label=f'ours (ep{oe[-1]}, {om[-1]:.0f}%, peak {om.max():.0f}%)')
    ax[i].axhline(TARGET[a], ls=':', lw=1.4, color='black', alpha=0.5,
                  label=f'reference test_agent = {TARGET[a]}%')
    ax[i].set_title(a.upper()); ax[i].set_xlabel('episode')
    ax[i].set_ylim(0, 100); ax[i].grid(alpha=0.3); ax[i].legend(loc='lower right', fontsize=8)
ax[0].set_ylabel('success rate % (100-ep moving avg)')
fig.suptitle('Training curves: ours vs reference repo (stage 9)', fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=100); plt.close(fig)
print(f'plotted -> {OUT}')
