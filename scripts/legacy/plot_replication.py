#!/usr/bin/env python3
# TRAINING-success comparison: the REFERENCE DDPG 8000-ep curve (turtlebot3_deepRL
# examples/ddpg_0_stage9) vs OUR three replication runs (DDPG/TD3/SAC). All curves are the
# 100-ep moving average of training success. Regenerated periodically by the plot agent.
import os, re, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/kaushik/project/deep-rl-robot-navigation'
REF_PKL = '/home/kaushik/project/turtlebot3_deepRL/src/turtlebot3_drl/model/examples/ddpg_0_stage9/stage9_episode8000.pkl'
OUT = f'{BASE}/experiments/replication_comparison.png'
W = 100
RUNS = [   # (experiment, color, label)
    ('replication_ddpg', '#ff7f0e', 'our DDPG'),
    ('replication_td3',  '#2ca02c', 'our TD3'),
    ('replication_sac',  '#d62728', 'our SAC'),
]

def ma(success, w=W):
    n = len(success)
    if n < 1: return np.array([])
    cs = np.cumsum(np.insert(np.asarray(success, float), 0, 0))
    return 100.0 * np.array([(cs[i+1] - cs[max(0, i+1-w)]) / min(i+1, w) for i in range(n)])

plt.figure(figsize=(11, 6))

# reference DDPG training curve (the only reference graphdata we have locally)
try:
    g = pickle.load(open(REF_PKL, 'rb'))
    rm = ma((np.array(g[1]) == 1).astype(float))
    plt.plot(np.arange(1, len(rm)+1), rm, lw=2, color='#1f77b4', alpha=0.85,
             label=f'REFERENCE DDPG (8000ep, final {rm[-1]:.0f}%)')
except Exception as e:
    print('ref load failed:', e)

# our three runs
for exp, color, label in RUNS:
    log = f'{BASE}/experiments/{exp}/train.log'
    if not os.path.exists(log):
        continue
    succ = [1 if re.search(r'outcome:\s+SUCCESS', l) else 0 for l in open(log) if l.startswith('Epi:')]
    if len(succ) >= 2:
        m = ma(succ)
        plt.plot(np.arange(1, len(m)+1), m, lw=2, color=color,
                 label=f'{label} (n={len(m)}, last {m[-1]:.0f}%, peak {m.max():.0f}%)')

# reference fixed-benchmark TEST results (final-policy success on the reference test set)
for y, c, t in [(84, '#1f77b4', 'ref DDPG std-eval 84%'), (74, '#2ca02c', 'ref TD3 std-eval 74%'), (82, '#d62728', 'ref SAC std-eval 82%')]:
    plt.axhline(y, ls=':', lw=1, color=c, alpha=0.4)

plt.xlabel('training episode'); plt.ylabel('success rate % (100-ep moving avg)')
plt.title('Replication training curves — reference DDPG vs our DDPG / TD3 / SAC (reward A | targets = reference test_agent std-eval)')
plt.ylim(0, 100); plt.legend(loc='upper left', fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT, dpi=100); plt.close()
print(f'plotted -> {OUT}')
