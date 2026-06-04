#!/usr/bin/env python3
# Overlay the moving-average success rate of every tracked experiment.
#   python3 scripts/plot_experiments.py [window]      # default window = 100 eps
# Reads experiments/*/train.log, plots each on one figure -> experiments/_comparison_success_ma.png
#
# PEAK is computed only over the FULL-WINDOW region (episode >= window), so it's a real
# sustained peak — not a single early success inflating a tiny partial window to 100%.
import os, re, glob, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/kaushik/project/deep-rl-robot-navigation'
W = int(sys.argv[1]) if len(sys.argv) > 1 else 100

plt.figure(figsize=(11, 6))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plotted = 0
for idx, log in enumerate(sorted(glob.glob(f'{BASE}/experiments/*/train.log'))):
    name = os.path.basename(os.path.dirname(log))
    succ = [1 if re.search(r'outcome:\s+SUCCESS', l) else 0 for l in open(log) if l.startswith('Epi:')]
    n = len(succ)
    if n < 2:
        continue
    s = np.array(succ, float)
    cs = np.cumsum(np.insert(s, 0, 0))
    ma = 100.0 * np.array([(cs[i+1] - cs[max(0, i+1-W)]) / min(i+1, W) for i in range(n)])

    # robust peak: only over the full-window region (>= W eps); if shorter than W, use the
    # 2nd half (so a 1-episode 100% partial window can never be reported as the peak).
    lo = W-1 if n >= W else n // 2
    region = ma[lo:]
    pk_i = lo + int(region.argmax()); pk_v = region.max()

    c = colors[idx % len(colors)]
    x = np.arange(1, n+1)
    ma_plot = ma.copy(); ma_plot[:min(5, n)] = np.nan   # hide the tiniest-window spike visually
    plt.plot(x, ma_plot, lw=2, color=c, label=f'{name}  (n={n}, last={ma[-1]:.0f}%, peak={pk_v:.0f}% @ep{pk_i+1})')
    plt.scatter([pk_i+1], [pk_v], color=c, s=45, zorder=5, edgecolor='k', linewidth=0.5)
    plotted += 1

plt.axhline(50, ls='--', c='k', alpha=0.25)
plt.axhline(90, ls='--', c='g', alpha=0.25)   # the 90% eval target line
plt.xlabel('training episode')
plt.ylabel(f'success rate % ({W}-ep moving avg)')
plt.title(f'Experiments — success-rate {W}-ep moving average (dots = full-window peak)')
plt.ylim(-2, 100); plt.grid(alpha=0.3); plt.legend(loc='upper left')
out = f'{BASE}/experiments/_comparison_success_ma.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
print(f"plotted {plotted} experiment(s), window={W} -> {out}")
