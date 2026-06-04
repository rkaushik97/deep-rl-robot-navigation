#!/usr/bin/env python3
# Overlay the moving-average success rate of every tracked experiment.
#   python3 scripts/plot_experiments.py [window]      # default window = 100 eps
# Reads experiments/*/train.log, plots each on one figure -> experiments/_comparison_success_ma.png
import os, re, glob, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/kaushik/project/deep-rl-robot-navigation'
W = int(sys.argv[1]) if len(sys.argv) > 1 else 100

plt.figure(figsize=(11, 6))
plotted = 0
for log in sorted(glob.glob(f'{BASE}/experiments/*/train.log')):
    name = os.path.basename(os.path.dirname(log))
    succ = [1 if re.search(r'outcome:\s+SUCCESS', l) else 0 for l in open(log) if l.startswith('Epi:')]
    n = len(succ)
    if n < 2:
        continue
    s = np.array(succ, float)
    cs = np.cumsum(np.insert(s, 0, 0))
    ma = 100.0 * np.array([(cs[i+1] - cs[max(0, i+1-W)]) / min(i+1, W) for i in range(n)])
    plt.plot(np.arange(1, n+1), ma, lw=2, label=f'{name}  (n={n}, last={ma[-1]:.0f}%, peak={ma.max():.0f}%)')
    plotted += 1

plt.axhline(50, ls='--', c='k', alpha=0.25)
plt.xlabel('training episode')
plt.ylabel(f'success rate % ({W}-ep moving avg)')
plt.title('Experiments — success-rate moving average')
plt.ylim(-2, 100); plt.grid(alpha=0.3); plt.legend(loc='upper left')
out = f'{BASE}/experiments/_comparison_success_ma.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
print(f"plotted {plotted} experiment(s) (window={W}) -> {out}")
