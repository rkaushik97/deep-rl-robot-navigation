#!/usr/bin/env python3
# SAC-only training-MA comparison: the reference repo's SAC (sac_5) training success vs our
# CURRENT SAC run (newest sac_*_stage_9 by mtime). Reference MA100 from its _train log
# (outcome col -> 100-ep moving avg of outcome==SUCCESS); ours from _metrics.tsv.
import glob, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/kaushik/project/deep-rl-robot-navigation'
REF = glob.glob('/home/kaushik/project/turtlebot3_deepRL/src/turtlebot3_drl/model/64891e20d104/sac_5_stage_9/_train_stage9_*.txt')
OUT = f'{BASE}/experiments/sac_vs_reference.png'
W = 100


def ma(s):
    s = np.asarray(s, float)
    return np.array([100.0 * s[max(0, i + 1 - W):i + 1].mean() for i in range(len(s))])


def ref_curve(p):
    e, s = [], []
    for l in open(p):
        q = l.split(',')
        if q[0].strip().isdigit():
            e.append(int(q[0])); s.append(1.0 if int(float(q[2])) == 1 else 0)
    return np.array(e), ma(s)


def our_curve():
    cands = [d for d in glob.glob(f'{BASE}/src/turtlebot3_drl/model/fond-filly/sac_*_stage_9')
             if os.path.exists(f'{d}/_metrics.tsv')]
    if not cands:
        return None, None, None
    d = max(cands, key=lambda x: os.path.getmtime(f'{x}/_metrics.tsv'))
    e, m = [], []
    for l in open(f'{d}/_metrics.tsv'):
        q = l.rstrip('\n').split('\t')
        if q and q[0].isdigit():
            e.append(int(q[0])); m.append(float(q[4]))
    return np.array(e), np.array(m), os.path.basename(d)


fig, ax = plt.subplots(figsize=(9, 5))
if REF:
    re_e, re_m = ref_curve(REF[0])
    ax.plot(re_e, re_m, color='gray', lw=2.2, alpha=0.85, label=f'reference SAC (sac_5, {len(re_e)} eps, final {re_m[-1]:.0f}%)')
oe, om, name = our_curve()
if oe is not None and len(oe):
    ax.plot(oe, om, color='#d62728', lw=2.2, label=f'ours: reward P + entropy −2.0 ({name}, ep{oe[-1]}, peak {om.max():.0f}%)')
ax.axhline(82, ls=':', lw=1.4, color='black', alpha=0.6, label='reference SAC test_agent = 82%')
ax.set_title('SAC training success — ours (reward P + exploration) vs reference (sac_5)')
ax.set_xlabel('episode'); ax.set_ylabel('success % (100-ep moving avg)')
ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend(loc='lower right', fontsize=9)
fig.tight_layout(); fig.savefig(OUT, dpi=100); plt.close(fig)
print(f'plotted -> {OUT}')
