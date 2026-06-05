#!/usr/bin/env python3
# Episode vs VALIDATION success on the REFERENCE test set (DYNAMIC_GOALS=False, fixed 17-goal
# list), for BOTH replication runs (DDPG and TD3). Source = each run's in-loop validation log
# _eval_stage9.tsv (written every 100 training eps, DRL_VAL_EPS=40). Lets us sweep the best
# checkpoint AND compare DDPG vs TD3 on the exact same benchmark the reference uses.
import os, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/kaushik/project/deep-rl-robot-navigation'
OUT = f'{BASE}/experiments/replication_val_curve.png'
RUNS = [   # (experiment, color, label, reference-test line)
    ('replication_ddpg', '#ff7f0e', 'our DDPG', 84),
    ('replication_td3',  '#2ca02c', 'our TD3', 74),
    ('replication_sac',  '#d62728', 'our SAC', 82),
]

def session_tsv(exp):
    log = f'{BASE}/experiments/{exp}/train.log'
    sess = None
    if os.path.exists(log):
        for l in open(log):
            m = re.search(r'location:\s*(\S*(?:ddpg|td3|sac)_\d+_stage_9)', l)
            if m: sess = m.group(1)
    return f'{sess}/_eval_stage9.tsv' if sess else None

plt.figure(figsize=(11, 6))
plt.axhline(84, ls='--', lw=1.2, color='#1f77b4', alpha=0.6, label='ref DDPG std-eval = 84%')
plt.axhline(74, ls=':', lw=1.0, color='#2ca02c', alpha=0.4, label='ref TD3 std-eval = 74%')
plt.axhline(82, ls=':', lw=1.0, color='#d62728', alpha=0.35, label='ref SAC std-eval = 82%')

summary = []
for exp, color, label, refline in RUNS:
    tsv = session_tsv(exp)
    eps, val = [], []
    if tsv and os.path.exists(tsv):
        for l in open(tsv):
            l = l.strip()
            if not l or l.startswith('episode'): continue
            f = re.split(r'\s+', l)
            try: eps.append(int(f[0])); val.append(float(f[3]) * 100.0)
            except (ValueError, IndexError): continue
    if eps:
        plt.plot(eps, val, 'o-', lw=2, color=color, ms=4, label=f'{label} (best {max(val):.0f}% @ep{eps[int(np.argmax(val))]}, last {val[-1]:.0f}%)')
        summary.append(f'{label}: best {max(val):.0f}%')
    else:
        summary.append(f'{label}: (no checkpoints yet)')

plt.xlabel('training episode'); plt.ylabel('success % on reference test set (40 det. eps, fixed goals)')
plt.title('Replication val on REFERENCE test set — ' + ' | '.join(summary))
plt.ylim(0, 100); plt.legend(loc='lower right', fontsize=9); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT, dpi=100); plt.close()
print(f'plotted -> {OUT}')
