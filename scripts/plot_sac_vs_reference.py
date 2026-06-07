#!/usr/bin/env python3
# SAC training-MA comparison: reference SAC (sac_5 of the reference repo) vs OUR two recipes —
# reward P (sac_4 here) and reward V (sac_5 here). MA100 success; reference from its _train log,
# ours from each run's _metrics.tsv. Annotated with the test_agent numbers.
import glob, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/kaushik/project/deep-rl-robot-navigation'
REF = glob.glob('/home/kaushik/project/turtlebot3_deepRL/src/turtlebot3_drl/model/64891e20d104/sac_5_stage_9/_train_stage9_*.txt')
OUT = f'{BASE}/experiments/sac_vs_reference.png'
W = 100
# our runs: (session dir, label, color)
OURS = [
    (f'{BASE}/src/turtlebot3_drl/model/fond-filly/sac_4_stage_9', 'ours: reward P (test ~73%)', '#1f77b4'),
    (f'{BASE}/src/turtlebot3_drl/model/fond-filly/sac_5_stage_9', 'ours: reward V (test 83%)',  '#d62728'),
]


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


def our_curve(d):
    f = f'{d}/_metrics.tsv'
    if not os.path.exists(f):
        return None, None
    e, m = [], []
    for l in open(f):
        q = l.rstrip('\n').split('\t')
        if q and q[0].isdigit():
            e.append(int(q[0])); m.append(float(q[4]))
    return np.array(e), np.array(m)


fig, ax = plt.subplots(figsize=(10, 5.5))
if REF:
    re_e, re_m = ref_curve(REF[0])
    ax.plot(re_e, re_m, color='gray', lw=2.4, alpha=0.85, label=f'reference SAC (sac_5, final {re_m[-1]:.0f}%, test 82%)')
for d, lab, c in OURS:
    oe, om = our_curve(d)
    if oe is not None and len(oe):
        ax.plot(oe, om, color=c, lw=2.0, label=f'{lab} (ep{oe[-1]}, peak {om.max():.0f}%)')
ax.axhline(82, ls=':', lw=1.4, color='black', alpha=0.6, label='reference test_agent = 82%')
ax.set_title('SAC training success — reward P vs reward V vs reference (stage 9)')
ax.set_xlabel('episode'); ax.set_ylabel('success % (100-ep moving avg)')
ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend(loc='lower right', fontsize=8)
fig.tight_layout(); fig.savefig(OUT, dpi=100); plt.close(fig)
print(f'plotted -> {OUT}')
